use scirust::attention::slha_v2::{
    LatentCodec, SciRustSlhaTile, D_C, FLAG_MIX3, FLAG_MIXED, FLAG_NF4, FLAG_TQ3, FLAG_TQ3_NOCORR,
    FLAG_WARM, RESIDUAL_WORDS,
};
use scirust::audit;
use scirust::learned::LearnedModel;
use scirust::weights;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::mem::{align_of, size_of};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Public C ABI revision.
pub const SLHA_ABI_VERSION: u32 = 1;

/// C ABI status codes. Existing values are preserved for compatibility.
pub const SLHA_OK: i32 = 0;
pub const SLHA_ERR_NULL: i32 = -1;
pub const SLHA_ERR_PANIC: i32 = -2;
pub const SLHA_ERR_DIMENSION: i32 = -3;
pub const SLHA_ERR_CODEC: i32 = -4;
pub const SLHA_ERR_NONFINITE: i32 = -5;
pub const SLHA_ERR_INVALID_TILE: i32 = -6;
pub const SLHA_ERR_INVALID_HANDLE: i32 = -7;
pub const SLHA_ERR_IO: i32 = -8;
pub const SLHA_ERR_UTF8: i32 = -9;

const MODEL_MAGIC: u64 = 0x534C_4841_4D4F_4401;
const LAST_ERROR_CAPACITY: usize = 512;
const CODEC_FLAGS: u16 = FLAG_NF4 | FLAG_MIXED | FLAG_TQ3 | FLAG_MIX3;
const KNOWN_TILE_FLAGS: u16 =
    FLAG_WARM | FLAG_NF4 | FLAG_MIXED | FLAG_TQ3 | FLAG_TQ3_NOCORR | FLAG_MIX3;
static ERROR_UNAVAILABLE: &[u8] = b"error unavailable\0";

struct LastError {
    bytes: [u8; LAST_ERROR_CAPACITY],
}

impl LastError {
    const fn new() -> Self {
        Self {
            bytes: [0; LAST_ERROR_CAPACITY],
        }
    }
}

std::thread_local! {
    static LAST_ERROR: RefCell<LastError> = const {
        RefCell::new(LastError::new())
    };
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            slot.bytes[0] = 0;
        }
    });
}

fn set_last_error(message: &str) {
    LAST_ERROR.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            let bytes = message.as_bytes();
            let len = bytes.len().min(LAST_ERROR_CAPACITY - 1);
            slot.bytes[..len].copy_from_slice(&bytes[..len]);
            slot.bytes[len] = 0;
        }
    });
}

#[derive(Debug)]
struct FfiError {
    code: i32,
    message: String,
}

fn ffi_error(code: i32, message: impl Into<String>) -> FfiError {
    FfiError {
        code,
        message: message.into(),
    }
}

fn ffi_status(f: impl FnOnce() -> Result<(), FfiError>) -> i32 {
    clear_last_error();

    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SLHA_OK,
        Ok(Err(error)) => {
            set_last_error(&error.message);
            error.code
        }
        Err(_) => {
            set_last_error("panic caught at the SLHA C ABI boundary");
            SLHA_ERR_PANIC
        }
    }
}

fn pointer_is_aligned<T>(pointer: *const T) -> bool {
    (pointer as usize).is_multiple_of(align_of::<T>())
}

/// Opaque process-wide handle for the currently stateless SLHA context.
#[repr(C)]
pub struct SlhaContext {
    _abi_version: u32,
    _reserved: u32,
}

static SLHA_CONTEXT: SlhaContext = SlhaContext {
    _abi_version: SLHA_ABI_VERSION,
    _reserved: 0,
};

/// Opaque handle to a loaded SLHA projection model (`.slhw`).
pub struct SlhaModel {
    magic: u64,
    inner: LearnedModel,
}

/// Registry of live model handles (the `Box::into_raw` pointers handed to C).
///
/// This is the trust anchor for the C ABI: a pointer is only ever
/// dereferenced if it is a **known, registered** handle. A forged/aligned
/// garbage pointer is rejected by the registry lookup *before* any read,
/// which fixes the old deref-then-check-magic ordering (an attacker-controlled
/// address could be read/written/`Box::from_raw`-freed before the magic check
/// ran). `slha_weights_release` removes the handle *before* reconstructing
/// the `Box`, so a double release fails with `SLHA_ERR_INVALID_HANDLE` instead
/// of a double-free.
use std::sync::OnceLock;
use std::sync::RwLock;
fn model_registry() -> &'static RwLock<std::collections::HashSet<usize>> {
    static REGISTRY: OnceLock<RwLock<std::collections::HashSet<usize>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(std::collections::HashSet::new()))
}

fn register_model(pointer: *mut SlhaModel) {
    if let Ok(mut registry) = model_registry().write() {
        registry.insert(pointer as usize);
    }
}

fn unregister_model(pointer: *mut SlhaModel) -> bool {
    let mut registry = match model_registry().write() {
        Ok(r) => r,
        Err(_) => return false,
    };
    registry.remove(&(pointer as usize))
}

fn is_registered(pointer: usize) -> bool {
    match model_registry().read() {
        Ok(r) => r.contains(&pointer),
        Err(_) => false,
    }
}

/// Upper bound on `n_tiles` in [`slha_score_tiles`]: 1M tiles ≈ 128 MB, far
/// above any real batch. Rejects absurd caller-supplied counts instead of
/// letting them drive an out-of-bounds read/write of arbitrary length.
pub const MAX_TILES: usize = 1 << 20;

unsafe fn model_ref<'a>(model: *const SlhaModel) -> Result<&'a SlhaModel, FfiError> {
    if model.is_null() {
        return Err(ffi_error(SLHA_ERR_NULL, "model handle is NULL"));
    }

    if !pointer_is_aligned(model) {
        return Err(ffi_error(
            SLHA_ERR_INVALID_HANDLE,
            "model handle is misaligned",
        ));
    }

    // Registry lookup BEFORE any dereference: only a handle returned by
    // `slha_weights_load` (and not yet released) is ever read.
    if !is_registered(model as usize) {
        return Err(ffi_error(
            SLHA_ERR_INVALID_HANDLE,
            "model handle is not a live SLHA handle",
        ));
    }

    // SAFETY: the registry guarantees this pointer was produced by
    // `Box::into_raw` in `slha_weights_load` and has not been released.
    let model = unsafe { &*model };

    // Belt-and-suspenders: the magic still guards against a corrupt handle
    // (e.g. a use-after-realloc of a stale but re-registered pointer).
    if model.magic != MODEL_MAGIC {
        return Err(ffi_error(
            SLHA_ERR_INVALID_HANDLE,
            "model handle has an invalid magic value",
        ));
    }

    Ok(model)
}

unsafe fn read_array_unaligned<T: Copy, const N: usize>(pointer: *const T) -> [T; N] {
    core::array::from_fn(|index| {
        // SAFETY: the C caller guarantees that N readable elements exist.
        unsafe { pointer.add(index).read_unaligned() }
    })
}

unsafe fn read_vec_unaligned<T: Copy>(pointer: *const T, len: usize) -> Vec<T> {
    (0..len)
        .map(|index| {
            // SAFETY: the C caller guarantees that len readable elements exist.
            unsafe { pointer.add(index).read_unaligned() }
        })
        .collect()
}

unsafe fn write_slice_unaligned<T: Copy>(pointer: *mut T, values: &[T]) {
    for (index, value) in values.iter().copied().enumerate() {
        // SAFETY: the C caller guarantees that values.len() writable elements exist.
        unsafe { pointer.add(index).write_unaligned(value) };
    }
}

fn validate_finite(values: &[f32], name: &str) -> Result<(), FfiError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ffi_error(
            SLHA_ERR_NONFINITE,
            format!("{name}[{index}] is not finite"),
        ));
    }

    Ok(())
}

fn validate_tile(tile: &SciRustSlhaTile) -> Result<(), FfiError> {
    if tile.flags & !KNOWN_TILE_FLAGS != 0 {
        return Err(ffi_error(
            SLHA_ERR_INVALID_TILE,
            format!("tile contains unknown flag bits: 0x{:04x}", tile.flags),
        ));
    }

    let codec_flags = tile.flags & CODEC_FLAGS;

    if codec_flags.count_ones() > 1 {
        return Err(ffi_error(
            SLHA_ERR_INVALID_TILE,
            "tile selects more than one latent codec",
        ));
    }

    if tile.flags & FLAG_TQ3_NOCORR != 0 && tile.flags & (FLAG_TQ3 | FLAG_MIX3) == 0 {
        return Err(ffi_error(
            SLHA_ERR_INVALID_TILE,
            "correction-drop flag requires the TQ3 or MIX3 codec",
        ));
    }

    if !tile.scale.is_finite()
        || !tile.dynamic_lambda.is_finite()
        || !tile.residual_sigma.is_finite()
    {
        return Err(ffi_error(
            SLHA_ERR_NONFINITE,
            "tile metadata contains a non-finite float",
        ));
    }

    if tile.scale < 0.0 || tile.dynamic_lambda < 0.0 || tile.residual_sigma < 0.0 {
        return Err(ffi_error(
            SLHA_ERR_INVALID_TILE,
            "tile scale, dynamic lambda, and residual sigma must be non-negative",
        ));
    }

    Ok(())
}

/// Return the C ABI revision.
#[no_mangle]
pub extern "C" fn slha_abi_version() -> u32 {
    SLHA_ABI_VERSION
}

/// Return the size of `SciRustSlhaTile` in the loaded Rust library.
#[no_mangle]
pub extern "C" fn slha_tile_size() -> usize {
    size_of::<SciRustSlhaTile>()
}

/// Return the alignment of `SciRustSlhaTile` in the loaded Rust library.
#[no_mangle]
pub extern "C" fn slha_tile_align() -> usize {
    align_of::<SciRustSlhaTile>()
}

/// Return the current thread's last C ABI error message.
///
/// The pointer remains valid until the next SLHA call that updates the error
/// state on the same thread, or until the thread exits. The caller must not
/// free it.
#[no_mangle]
pub extern "C" fn slha_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.try_borrow()
            .map(|slot| slot.bytes.as_ptr() as *const c_char)
            .unwrap_or(ERROR_UNAVAILABLE.as_ptr() as *const c_char)
    })
}

/// Clear the current thread's C ABI error message.
#[no_mangle]
pub extern "C" fn slha_clear_error() {
    clear_last_error();
}

/// Initialize the SLHA environment.
///
/// The kernel is currently stateless, so this returns a stable process-wide
/// handle that points to a real object and requires no allocation.
#[no_mangle]
pub extern "C" fn slha_init() -> *mut SlhaContext {
    clear_last_error();
    std::ptr::from_ref(&SLHA_CONTEXT).cast_mut()
}

/// Validate and release a context returned by `slha_init`.
///
/// The current process-wide context owns no resources, so release is a no-op.
/// NULL is accepted. A foreign non-null pointer is rejected without being
/// dereferenced.
///
/// # Safety
/// A non-null pointer must be the handle returned by `slha_init`.
#[no_mangle]
pub unsafe extern "C" fn slha_shutdown(context: *mut SlhaContext) -> i32 {
    clear_last_error();

    if context.is_null() {
        return SLHA_OK;
    }

    let expected = std::ptr::from_ref(&SLHA_CONTEXT).cast_mut();

    if context != expected {
        set_last_error("context handle was not returned by slha_init");
        return SLHA_ERR_INVALID_HANDLE;
    }

    SLHA_OK
}

/// Process a single tile and compute the score.
///
/// Returns 0 on success. Output is not modified on failure.
///
/// # Safety
/// `tile`, `q_coarse`, `q_sign`, and `score_out` must point to readable or
/// writable storage of the documented size. Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_process_tile(
    tile: *const SciRustSlhaTile,
    q_coarse: *const f32,
    q_sign: *const u64,
    score_out: *mut f32,
) -> i32 {
    ffi_status(|| {
        if tile.is_null() || q_coarse.is_null() || q_sign.is_null() || score_out.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_process_tile received a NULL pointer",
            ));
        }

        // SAFETY: the caller guarantees readable storage. read_unaligned removes
        // any Rust-side alignment requirement.
        let tile = unsafe { tile.read_unaligned() };
        let q_coarse = unsafe { read_array_unaligned::<f32, D_C>(q_coarse) };
        let q_sign = unsafe { read_array_unaligned::<u64, RESIDUAL_WORDS>(q_sign) };

        validate_tile(&tile)?;
        validate_finite(&q_coarse, "q_coarse")?;

        let score = tile.compute_score(&q_coarse, &q_sign);

        if !score.is_finite() {
            return Err(ffi_error(
                SLHA_ERR_NONFINITE,
                "tile scoring produced a non-finite result",
            ));
        }

        // SAFETY: the caller guarantees writable storage for one f32.
        unsafe { score_out.write_unaligned(score) };
        Ok(())
    })
}

/// Run the self-audit and return a JSON string.
///
/// The returned pointer must be freed exactly once with `slha_free_string`.
#[no_mangle]
pub extern "C" fn slha_audit() -> *mut c_char {
    clear_last_error();

    match catch_unwind(|| {
        let report = audit::run();
        CString::new(report.to_compact())
    }) {
        Ok(Ok(value)) => value.into_raw(),
        Ok(Err(_)) => {
            set_last_error("self-audit JSON unexpectedly contains an interior NUL byte");
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic caught while running the SLHA self-audit");
            std::ptr::null_mut()
        }
    }
}

/// Free a string allocated by the library. NULL is a no-op.
///
/// # Safety
/// `string` must be NULL or a live pointer returned by `slha_audit`, freed
/// exactly once.
#[no_mangle]
pub unsafe extern "C" fn slha_free_string(string: *mut c_char) {
    if !string.is_null() {
        // SAFETY: the caller guarantees provenance and single release.
        unsafe { drop(CString::from_raw(string)) };
    }
}

/// Map the C codec id to a `LatentCodec`:
/// 0=int4-single, 1=int4-grouped, 2=nf4, 3=mixed, 4=tq3, 5=mix3.
fn codec_from_int(codec: i32) -> Option<LatentCodec> {
    match codec {
        0 => Some(LatentCodec::Int4Single),
        1 => Some(LatentCodec::Int4Grouped),
        2 => Some(LatentCodec::Nf4),
        3 => Some(LatentCodec::Mixed),
        4 => Some(LatentCodec::Tq3),
        5 => Some(LatentCodec::Mix3),
        _ => None,
    }
}

/// Load a projection model from a `.slhw` file.
///
/// Returns NULL on failure. Call `slha_last_error_message` for details.
///
/// # Safety
/// `path` must be a valid NUL-terminated C string when non-null. The returned
/// handle must be released exactly once with `slha_weights_free` or
/// `slha_weights_release`.
#[no_mangle]
pub unsafe extern "C" fn slha_weights_load(path: *const c_char) -> *mut SlhaModel {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut SlhaModel, FfiError> {
        if path.is_null() {
            return Err(ffi_error(SLHA_ERR_NULL, "weights path is NULL"));
        }

        // SAFETY: the caller guarantees a valid NUL-terminated C string.
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|_| ffi_error(SLHA_ERR_UTF8, "weights path is not valid UTF-8"))?;

        let inner = weights::load(path).map_err(|error| ffi_error(SLHA_ERR_IO, error))?;

        let handle = Box::into_raw(Box::new(SlhaModel {
            magic: MODEL_MAGIC,
            inner,
        }));
        register_model(handle);
        Ok(handle)
    }));

    match result {
        Ok(Ok(model)) => model,
        Ok(Err(error)) => {
            set_last_error(&error.message);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic caught while loading SLHA weights");
            std::ptr::null_mut()
        }
    }
}

/// Return the projection input dimension. Returns 0 on failure.
///
/// # Safety
/// `model` must be a live handle returned by `slha_weights_load`.
#[no_mangle]
pub unsafe extern "C" fn slha_model_dim(model: *const SlhaModel) -> usize {
    clear_last_error();

    match catch_unwind(AssertUnwindSafe(|| unsafe {
        model_ref(model).map(|model| model.inner.d)
    })) {
        Ok(Ok(dimension)) => dimension,
        Ok(Err(error)) => {
            set_last_error(&error.message);
            0
        }
        Err(_) => {
            set_last_error("panic caught while reading the SLHA model dimension");
            0
        }
    }
}

/// Project a query vector through the learned model.
///
/// On success, `q_coarse_out` receives `D_C` floats and `q_sign_out` receives
/// `RESIDUAL_WORDS` u64s representing the sign-LSH hash.  Output is not
/// modified on failure.  Unaligned output storage is accepted.
///
/// # Safety
/// `model` must be a live model handle. `query` points to `d` readable f32s.
/// `q_coarse_out` points to writable storage for `D_C` f32s.
/// `q_sign_out` points to writable storage for `RESIDUAL_WORDS` u64s.
#[no_mangle]
pub unsafe extern "C" fn slha_prepare_query(
    model: *const SlhaModel,
    query: *const f32,
    d: usize,
    q_coarse_out: *mut f32,
    q_sign_out: *mut u64,
) -> i32 {
    ffi_status(|| {
        if query.is_null() || q_coarse_out.is_null() || q_sign_out.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_prepare_query received a NULL pointer",
            ));
        }

        // SAFETY: the caller promises a live handle.
        let model = unsafe { model_ref(model)? };

        if d != model.inner.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "query dimension mismatch: received {d}, expected {}",
                    model.inner.d
                ),
            ));
        }

        // SAFETY: dimension equality is checked before reading.
        let query = unsafe { read_vec_unaligned(query, d) };
        validate_finite(&query, "query")?;

        let q_coarse = model.inner.query_coarse(&query);
        let q_sign = model.inner.sign_bits(&query);

        // SAFETY: the caller guarantees writable storage for D_C f32s and
        // RESIDUAL_WORDS u64s.
        unsafe {
            write_slice_unaligned(q_coarse_out, &q_coarse);
            write_slice_unaligned(q_sign_out, &q_sign);
        }

        Ok(())
    })
}

/// Score `n_tiles` tiles in batch against a single prepared query.
///
/// Each tile is scored independently.  The tile's `flags` field determines
/// HOT vs WARM mode automatically.  If any tile is invalid or produces a
/// non-finite score the batch stops immediately.
///
/// # Safety
/// `tiles` points to `n_tiles` readable SciRustSlhaTile values.
/// `q_coarse` points to `D_C` readable f32s.
/// `q_sign` points to `RESIDUAL_WORDS` readable u64s.
/// `scores_out` points to `n_tiles` writable f32s.
#[no_mangle]
pub unsafe extern "C" fn slha_score_tiles(
    tiles: *const SciRustSlhaTile,
    n_tiles: usize,
    q_coarse: *const f32,
    q_sign: *const u64,
    scores_out: *mut f32,
) -> i32 {
    ffi_status(|| {
        if tiles.is_null() || q_coarse.is_null() || q_sign.is_null() || scores_out.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_score_tiles received a NULL pointer",
            ));
        }

        // Bound the caller-supplied tile count: an absurd `n_tiles` would
        // otherwise drive an out-of-bounds read of `tiles` and write of
        // `scores_out` of arbitrary length.
        if n_tiles > MAX_TILES {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "slha_score_tiles: n_tiles={n_tiles} exceeds the {MAX_TILES}-tile safety bound"
                ),
            ));
        }

        // SAFETY: the caller guarantees readable storage.
        let q_coarse = unsafe { read_array_unaligned::<f32, D_C>(q_coarse) };
        let q_sign = unsafe { read_array_unaligned::<u64, RESIDUAL_WORDS>(q_sign) };

        validate_finite(&q_coarse, "q_coarse")?;

        for i in 0..n_tiles {
            // SAFETY: the caller guarantees readable storage for n_tiles tiles.
            let tile = unsafe { tiles.add(i).read_unaligned() };
            validate_tile(&tile)?;

            let score = tile.compute_score(&q_coarse, &q_sign);

            if !score.is_finite() {
                return Err(ffi_error(
                    SLHA_ERR_NONFINITE,
                    format!("tile[{i}] scoring produced a non-finite result"),
                ));
            }

            // SAFETY: the caller guarantees writable storage for n_tiles f32s.
            unsafe { scores_out.add(i).write_unaligned(score) };
        }

        Ok(())
    })
}

/// Encode a `d`-dimensional key vector into a 128-byte tile.
///
/// `codec`: 0=int4-single, 1=int4-grouped, 2=nf4, 3=mixed, 4=tq3,
/// 5=mix3. Output is not modified on failure.
///
/// # Safety
/// `model` must be a live model handle. `key` points to `d` readable f32s.
/// `out_tile` points to writable storage for one tile. Unaligned key and output
/// storage are accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_encode_key(
    model: *const SlhaModel,
    key: *const f32,
    d: usize,
    pos: u32,
    codec: i32,
    out_tile: *mut SciRustSlhaTile,
) -> i32 {
    ffi_status(|| {
        if key.is_null() || out_tile.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_encode_key received a NULL data pointer",
            ));
        }

        // SAFETY: the caller promises a live handle.
        let model = unsafe { model_ref(model)? };

        if d != model.inner.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "key dimension mismatch: received {d}, expected {}",
                    model.inner.d
                ),
            ));
        }

        let codec = codec_from_int(codec).ok_or_else(|| {
            ffi_error(
                SLHA_ERR_CODEC,
                format!("unknown codec id {codec}; expected an integer from 0 to 5"),
            )
        })?;

        // SAFETY: dimension equality is checked before reading.
        let key = unsafe { read_vec_unaligned(key, d) };
        validate_finite(&key, "key")?;

        let tile = model.inner.encode_with(&key, pos, false, codec);
        validate_tile(&tile)?;

        // SAFETY: the caller guarantees writable storage for one tile.
        unsafe { out_tile.write_unaligned(tile) };
        Ok(())
    })
}

/// Decode a tile's latent back into the original `d`-dimensional space.
///
/// Output is not modified on failure.
///
/// # Safety
/// `model` must be a live model handle. `tile` points to one readable tile.
/// `out` points to `d` writable f32s. Unaligned tile and output storage are
/// accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_decode_latent(
    model: *const SlhaModel,
    tile: *const SciRustSlhaTile,
    out: *mut f32,
    d: usize,
) -> i32 {
    ffi_status(|| {
        if tile.is_null() || out.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_decode_latent received a NULL data pointer",
            ));
        }

        // SAFETY: the caller promises a live handle.
        let model = unsafe { model_ref(model)? };

        if d != model.inner.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "output dimension mismatch: received {d}, expected {}",
                    model.inner.d
                ),
            ));
        }

        // SAFETY: the caller guarantees readable storage for one tile.
        let tile = unsafe { tile.read_unaligned() };
        validate_tile(&tile)?;

        let latent = tile.dequant_latent();
        validate_finite(&latent, "dequantized_latent")?;

        let reconstruction = model.inner.reconstruct(&latent);

        if reconstruction.len() != d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                "model reconstruction returned an unexpected length",
            ));
        }

        validate_finite(&reconstruction, "reconstruction")?;

        // SAFETY: the caller guarantees writable storage for d f32s.
        unsafe { write_slice_unaligned(out, &reconstruction) };
        Ok(())
    })
}

/// Decode a full tile (latent + residual sign sketch) back into the original
/// `d`-dimensional key vector.
///
/// Output is not modified on failure.
///
/// # Safety
/// `model` must be a live model handle. `tile` points to one readable tile.
/// `out` points to `d` writable f32s. Unaligned tile and output storage are
/// accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_decode_key(
    model: *const SlhaModel,
    tile: *const SciRustSlhaTile,
    out: *mut f32,
    d: usize,
) -> i32 {
    ffi_status(|| {
        if tile.is_null() || out.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_decode_key received a NULL data pointer",
            ));
        }

        // SAFETY: the caller promises a live handle.
        let model = unsafe { model_ref(model)? };

        if d != model.inner.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "output dimension mismatch: received {d}, expected {}",
                    model.inner.d
                ),
            ));
        }

        // SAFETY: the caller guarantees readable storage for one tile.
        let tile = unsafe { tile.read_unaligned() };
        validate_tile(&tile)?;

        let reconstruction = model.inner.reconstruct_from_tile(&tile);

        if reconstruction.len() != d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                "model reconstruction returned an unexpected length",
            ));
        }

        validate_finite(&reconstruction, "reconstruction")?;

        // SAFETY: the caller guarantees writable storage for d f32s.
        unsafe { write_slice_unaligned(out, &reconstruction) };
        Ok(())
    })
}

/// Release a model and return an explicit status. NULL is a no-op.
///
/// # Safety
/// A non-null pointer must be a live handle returned by `slha_weights_load`,
/// released exactly once.
#[no_mangle]
pub unsafe extern "C" fn slha_weights_release(model: *mut SlhaModel) -> i32 {
    if model.is_null() {
        clear_last_error();
        return SLHA_OK;
    }

    ffi_status(|| {
        // Validate the handle via the registry (deref happens only after the
        // registry confirms the pointer is a live, owned handle).
        unsafe {
            model_ref(model)?;
        }

        // Remove from the registry BEFORE reconstructing the Box: a second
        // release of the same pointer now fails the registry lookup instead of
        // double-freeing.
        if !unregister_model(model) {
            return Err(ffi_error(
                SLHA_ERR_INVALID_HANDLE,
                "model handle is not registered (already released?)",
            ));
        }

        // The registry removal guarantees the pointer is an unreleased
        // `Box::into_raw` handle, so `Box::from_raw` is sound. Wrapped in
        // catch_unwind (via ffi_status) for defense-in-depth.
        unsafe { drop(Box::from_raw(model)) };
        Ok(())
    })
}

/// Compatibility wrapper for the original void-returning release function.
///
/// Prefer `slha_weights_release` when the caller needs an explicit status.
///
/// # Safety
/// Same contract as `slha_weights_release`.
#[no_mangle]
pub unsafe extern "C" fn slha_weights_free(model: *mut SlhaModel) {
    // SAFETY: forwarded under the caller's contract.
    let _ = unsafe { slha_weights_release(model) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust::learned::gen_keys;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Process-wide counter combined with a `tempfile::TempDir` guarantees a
    /// unique, non-colliding path for every `TempModel`, even when tests run in
    /// parallel under the default multi-threaded `cargo test` harness.
    static TEMP_MODEL_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn last_error() -> String {
        // SAFETY: slha_last_error_message returns a valid NUL-terminated pointer
        // owned by thread-local storage.
        unsafe {
            CStr::from_ptr(slha_last_error_message())
                .to_string_lossy()
                .into_owned()
        }
    }

    fn zero_tile() -> SciRustSlhaTile {
        SciRustSlhaTile {
            latent_kv: [0; 64],
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale: 0.0,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: 0,
            group_scales: [0; 8],
        }
    }

    #[test]
    fn abi_metadata_and_context_lifecycle_are_consistent() {
        assert_eq!(slha_abi_version(), SLHA_ABI_VERSION);
        assert_eq!(slha_tile_size(), 128);
        assert_eq!(slha_tile_size(), size_of::<SciRustSlhaTile>());
        assert_eq!(slha_tile_align(), align_of::<SciRustSlhaTile>());

        let context = slha_init();
        assert!(!context.is_null(), "slha_init failed: {}", last_error());
        assert_eq!(context, slha_init(), "context handle must be stable");
        assert_eq!(unsafe { slha_shutdown(context) }, SLHA_OK);
        assert_eq!(unsafe { slha_shutdown(std::ptr::null_mut()) }, SLHA_OK);

        let mut foreign = SlhaContext {
            _abi_version: SLHA_ABI_VERSION,
            _reserved: 0,
        };
        assert_eq!(
            unsafe { slha_shutdown(&mut foreign) },
            SLHA_ERR_INVALID_HANDLE
        );
    }

    #[test]
    fn process_tile_rejects_invalid_inputs_without_overwriting_output() {
        let tile = zero_tile();
        let mut query = [0.0f32; D_C];
        let signs = [0u64; RESIDUAL_WORDS];
        let mut output = 123.0f32;

        assert_eq!(
            unsafe {
                slha_process_tile(
                    std::ptr::null(),
                    query.as_ptr(),
                    signs.as_ptr(),
                    &mut output,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(output, 123.0);

        query[7] = f32::NAN;
        assert_eq!(
            unsafe { slha_process_tile(&tile, query.as_ptr(), signs.as_ptr(), &mut output,) },
            SLHA_ERR_NONFINITE
        );
        assert_eq!(output, 123.0);

        query[7] = 0.0;
        let mut invalid_tile = tile;
        invalid_tile.flags = FLAG_NF4 | FLAG_TQ3;

        assert_eq!(
            unsafe {
                slha_process_tile(&invalid_tile, query.as_ptr(), signs.as_ptr(), &mut output)
            },
            SLHA_ERR_INVALID_TILE
        );
        assert_eq!(output, 123.0);
    }

    #[test]
    fn audit_string_and_last_error_channel_work() {
        let audit = slha_audit();
        assert!(!audit.is_null(), "slha_audit failed: {}", last_error());

        // SAFETY: audit is non-null and owned by the library.
        let audit_text = unsafe { CStr::from_ptr(audit) }
            .to_string_lossy()
            .into_owned();
        assert!(audit_text.contains("\"verdict\""));

        // SAFETY: audit came from slha_audit and is freed once.
        unsafe { slha_free_string(audit) };

        let handle = unsafe { slha_weights_load(std::ptr::null()) };
        assert!(handle.is_null());
        assert!(last_error().contains("NULL"));
    }

    #[test]
    fn encode_decode_round_trip_through_c_abi() {
        let d = 256usize;
        let temp = TempModel::new(d, 0xC0FFEE);
        let handle = temp.handle();
        assert_eq!(unsafe { slha_model_dim(handle) }, d);

        let key = &gen_keys(2, 1, d, 64, 0.9, 0.02)[0];
        let expected_flags = [0, 0, FLAG_NF4, FLAG_MIXED, FLAG_TQ3, FLAG_MIX3];

        let mut key_storage = vec![0u8; d * size_of::<f32>() + 1];
        let unaligned_key = unsafe { key_storage.as_mut_ptr().add(1) as *mut f32 };

        for (index, value) in key.iter().copied().enumerate() {
            // SAFETY: key_storage has space for d f32s after the one-byte offset.
            unsafe { unaligned_key.add(index).write_unaligned(value) };
        }

        for (codec, expected_flag) in expected_flags.into_iter().enumerate() {
            let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>() + 1];
            let unaligned_tile =
                unsafe { tile_storage.as_mut_ptr().add(1) as *mut SciRustSlhaTile };

            let rc = unsafe {
                slha_encode_key(
                    handle,
                    unaligned_key,
                    d,
                    codec as u32,
                    codec as i32,
                    unaligned_tile,
                )
            };
            assert_eq!(rc, SLHA_OK, "codec {codec}: {}", last_error());

            // SAFETY: encode succeeded and initialized one complete tile.
            let tile = unsafe { unaligned_tile.read_unaligned() };
            assert_eq!(
                tile.flags & CODEC_FLAGS,
                expected_flag,
                "codec {codec} produced the wrong flag"
            );

            let mut out_storage = vec![0u8; d * size_of::<f32>() + 1];
            let unaligned_out = unsafe { out_storage.as_mut_ptr().add(1) as *mut f32 };

            let rc = unsafe { slha_decode_latent(handle, unaligned_tile, unaligned_out, d) };
            assert_eq!(rc, SLHA_OK, "codec {codec}: {}", last_error());

            let out: Vec<f32> = (0..d)
                .map(|index| {
                    // SAFETY: decode initialized d unaligned f32 outputs.
                    unsafe { unaligned_out.add(index).read_unaligned() }
                })
                .collect();

            assert!(out.iter().all(|value| value.is_finite()));

            if codec == 5 {
                let signal: f32 = key.iter().map(|value| value * value).sum();
                let error: f32 = key
                    .iter()
                    .zip(&out)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum();
                let snr = 10.0 * (signal / error.max(1e-12)).log10();
                assert!(snr > 2.0, "MIX3 reconstruction SNR too low: {snr} dB");
            }
        }

        let mut tile = zero_tile();
        assert_eq!(
            unsafe { slha_encode_key(handle, key.as_ptr(), d + 1, 0, 5, &mut tile,) },
            SLHA_ERR_DIMENSION
        );
        assert_eq!(
            unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 99, &mut tile,) },
            SLHA_ERR_CODEC
        );
        assert_eq!(
            unsafe { slha_encode_key(std::ptr::null(), key.as_ptr(), d, 0, 5, &mut tile,) },
            SLHA_ERR_NULL
        );

        let mut non_finite_key = key.to_vec();
        non_finite_key[0] = f32::INFINITY;
        assert_eq!(
            unsafe { slha_encode_key(handle, non_finite_key.as_ptr(), d, 0, 5, &mut tile,) },
            SLHA_ERR_NONFINITE
        );

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn decode_key_reconstructs_residual_beyond_latent_only() {
        let d = 256usize;
        let temp = TempModel::new(d, 0xB0BA);
        let handle = temp.handle();
        assert!(!handle.is_null());

        let key = &gen_keys(2, 1, d, 64, 0.9, 0.02)[0];
        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;

        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 3, tile) };
        assert_eq!(rc, SLHA_OK);

        let mut latent = vec![0.0f32; d];
        let mut full = vec![0.0f32; d];
        assert_eq!(
            unsafe { slha_decode_latent(handle, tile, latent.as_mut_ptr(), d) },
            SLHA_OK
        );
        assert_eq!(
            unsafe { slha_decode_key(handle, tile, full.as_mut_ptr(), d) },
            SLHA_OK
        );

        let snr_latent = snr(key, &latent);
        let snr_full = snr(key, &full);
        assert!(
            snr_full > snr_latent,
            "full reconstruction ({snr_full} dB) should beat latent-only ({snr_latent} dB)"
        );

        // Model and temp directory are released by the `TempModel` RAII guard.

        fn snr(a: &[f32], b: &[f32]) -> f32 {
            let signal: f32 = a.iter().map(|x| x * x).sum();
            let error: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
            10.0 * (signal / error.max(1e-12)).log10()
        }
    }

    // ------------------------------------------------------------------
    //  Milestone 1: offline C-ABI scoring parity
    // ------------------------------------------------------------------

    /// RAII guard for one unique `.slhw` model file.
    ///
    /// Every instance creates its own `tempfile::TempDir` and a unique file
    /// inside it, so parallel tests cannot collide on the same path.  Drop
    /// releases the C handle *before* the temp directory is removed, so the
    /// file is never deleted while the loaded model is still live.
    struct TempModel {
        handle: *mut SlhaModel,
        model: LearnedModel,
        _dir: tempfile::TempDir,
        path: std::path::PathBuf,
    }

    impl TempModel {
        fn new(d: usize, seed: u64) -> Self {
            let counter = TEMP_MODEL_COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = tempfile::Builder::new()
                .prefix(&format!("slha_c_{counter}_"))
                .tempdir()
                .expect("create temp dir for model");
            let path = dir.path().join("model.slhw");

            let train = gen_keys(seed, 512, d, d / 2, 0.9, 0.02);
            let model = LearnedModel::fit(&train, d, seed ^ 0xA5A5, false);
            weights::save(path.to_str().unwrap(), &model, seed ^ 0xA5A5, false)
                .expect("save weights");

            let cpath = CString::new(path.to_str().unwrap()).unwrap();
            let handle = unsafe { slha_weights_load(cpath.as_ptr()) };
            assert!(!handle.is_null(), "load failed: {}", last_error());

            Self {
                handle,
                model,
                _dir: dir,
                path,
            }
        }

        fn handle(&self) -> *mut SlhaModel {
            self.handle
        }

        fn model(&self) -> &LearnedModel {
            &self.model
        }

        #[allow(dead_code)]
        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempModel {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                // SAFETY: handle was returned by slha_weights_load and is
                // released exactly once here.
                unsafe {
                    let _ = slha_weights_release(self.handle);
                }
                self.handle = std::ptr::null_mut();
            }
            // `_dir` is dropped after this, removing the unique directory.
        }
    }

    /// Shared helper: fit a model, save to a unique temp file, load via C ABI,
    /// and return an RAII guard for use in multiple tests.
    fn setup_model_and_handle(d: usize, seed: u64) -> TempModel {
        TempModel::new(d, seed)
    }

    /// Score a single tile through the canonical Rust path for comparison.
    fn canonical_score(model: &LearnedModel, tile: &SciRustSlhaTile, query: &[f32]) -> f32 {
        let qc = model.query_coarse(query);
        let qs = model.sign_bits(query);
        tile.compute_score(&qc, &qs)
    }

    /// Allocate a buffer for `n` tiles at the correct alignment.
    fn alloc_tiles(n: usize) -> (Vec<u8>, *mut SciRustSlhaTile) {
        // Overallocate and align: 128-byte tile, up to 128-byte alignment.
        const TILE_SIZE: usize = size_of::<SciRustSlhaTile>();
        let mut buf = vec![0u8; n * TILE_SIZE + 128];
        let base = buf.as_mut_ptr() as usize;
        let aligned = (base + 127) & !127;
        let offset = aligned - base;
        let ptr = unsafe { buf.as_mut_ptr().add(offset) as *mut SciRustSlhaTile };
        (buf, ptr)
    }

    #[test]
    fn prepare_query_matches_canonical_rust_coarse_and_sign() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xCAFE);
        let handle = temp.handle();
        let model = temp.model();

        let query = &gen_keys(0xB0BA, 1, d, d / 2, 0.9, 0.02)[0];

        let mut q_coarse_out = [0.0f32; D_C];
        let mut q_sign_out = [0u64; RESIDUAL_WORDS];

        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse_out.as_mut_ptr(),
                q_sign_out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK, "prepare_query failed: {}", last_error());

        let expected_coarse = model.query_coarse(query);
        let expected_sign = model.sign_bits(query);

        assert_eq!(
            q_coarse_out.as_slice(),
            expected_coarse.as_slice(),
            "q_coarse mismatch"
        );
        assert_eq!(
            q_sign_out.as_slice(),
            expected_sign.as_slice(),
            "q_sign mismatch"
        );

        // Also verify through process_tile: score using C-prepared Q must
        // match score using Rust-prepared Q.
        let key = &gen_keys(0xBEAD, 1, d, d / 2, 0.9, 0.02)[0];
        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile_ptr = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 3, tile_ptr) };
        assert_eq!(rc, SLHA_OK);

        let mut c_score = 123.0f32;
        let rc = unsafe {
            slha_process_tile(
                tile_ptr,
                q_coarse_out.as_ptr(),
                q_sign_out.as_ptr(),
                &mut c_score,
            )
        };
        assert_eq!(rc, SLHA_OK);

        let tile = unsafe { tile_ptr.read_unaligned() };
        let canonical = canonical_score(model, &tile, query);
        assert!(
            (c_score - canonical).abs() <= f32::EPSILON * 16.0,
            "C ABI score {c_score} != canonical Rust {canonical}"
        );

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn batch_score_matches_repeated_scalar() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xD0CE);
        let handle = temp.handle();
        let model = temp.model();

        let keys = gen_keys(0xDEAD, 8, d, d / 2, 0.9, 0.02);
        let query = &gen_keys(0xBEEF, 1, d, d / 2, 0.9, 0.02)[0];

        // Encode all keys to tiles.
        let (_tiles_buf, tiles_ptr) = alloc_tiles(8);
        for (i, key) in keys.iter().enumerate() {
            let tile_ptr = unsafe { tiles_ptr.add(i) };
            let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, i as u32, 3, tile_ptr) };
            assert_eq!(rc, SLHA_OK, "encode key {i} failed: {}", last_error());
        }

        // Prepare query.
        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        // Batch score.
        let mut batch_scores = vec![-1.0f32; 8];
        let rc = unsafe {
            slha_score_tiles(
                tiles_ptr,
                8,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                batch_scores.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        // Repeated scalar score.
        for (i, _key) in keys.iter().enumerate() {
            let tile = unsafe { &*tiles_ptr.add(i) };
            let expected = canonical_score(model, tile, query);
            let actual = batch_scores[i];
            assert!(
                (actual - expected).abs() <= f32::EPSILON * 16.0,
                "tile[{i}]: batch {actual} != canonical {expected}"
            );
        }

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn mixed_codec_parity() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xCAFE);
        let handle = temp.handle();
        let model = temp.model();
        let key = &gen_keys(0xBABE, 1, d, d / 2, 0.9, 0.02)[0];
        let query = &gen_keys(0xFACE, 1, d, d / 2, 0.9, 0.02)[0];

        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile_ptr = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;
        // Codec 3 = mixed
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 3, tile_ptr) };
        assert_eq!(rc, SLHA_OK, "encode mixed failed: {}", last_error());

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        let mut c_score = -2.0f32;
        let rc = unsafe {
            slha_process_tile(tile_ptr, q_coarse.as_ptr(), q_sign.as_ptr(), &mut c_score)
        };
        assert_eq!(rc, SLHA_OK);

        let tile = unsafe { tile_ptr.read_unaligned() };
        let canonical = canonical_score(model, &tile, query);
        assert!(
            (c_score - canonical).abs() <= f32::EPSILON * 16.0,
            "mixed: C {c_score} != Rust {canonical}"
        );

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn mix3_codec_parity() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xDEAD);
        let handle = temp.handle();
        let model = temp.model();
        let key = &gen_keys(0xBEEF, 1, d, d / 2, 0.9, 0.02)[0];
        let query = &gen_keys(0xCAFE, 1, d, d / 2, 0.9, 0.02)[0];

        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile_ptr = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;
        // Codec 5 = MIX3
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 5, tile_ptr) };
        assert_eq!(rc, SLHA_OK, "encode mix3 failed: {}", last_error());

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        let mut c_score = -2.0f32;
        let rc = unsafe {
            slha_process_tile(tile_ptr, q_coarse.as_ptr(), q_sign.as_ptr(), &mut c_score)
        };
        assert_eq!(rc, SLHA_OK);

        let tile = unsafe { tile_ptr.read_unaligned() };
        let canonical = canonical_score(model, &tile, query);
        assert!(
            (c_score - canonical).abs() <= f32::EPSILON * 16.0,
            "mix3: C {c_score} != Rust {canonical}"
        );

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn hot_mode_parity() {
        // HOT = residual enabled (FLAG_WARM not set). Encode with warm=false.
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xBEAD);
        let handle = temp.handle();
        let model = temp.model();
        let key = &gen_keys(0xCAFE, 1, d, d / 2, 0.9, 0.02)[0];
        let query = &gen_keys(0xD0CE, 1, d, d / 2, 0.9, 0.02)[0];

        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile_ptr = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 3, tile_ptr) };
        assert_eq!(rc, SLHA_OK);

        // Verify HOT: FLAG_WARM is not set.
        let tile = unsafe { tile_ptr.read_unaligned() };
        assert_eq!(tile.flags & FLAG_WARM, 0, "expected HOT tile");

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        let mut c_score = -2.0f32;
        let rc = unsafe {
            slha_process_tile(tile_ptr, q_coarse.as_ptr(), q_sign.as_ptr(), &mut c_score)
        };
        assert_eq!(rc, SLHA_OK);

        let canonical = canonical_score(model, &tile, query);
        assert!(
            (c_score - canonical).abs() <= f32::EPSILON * 16.0,
            "HOT: C {c_score} != Rust {canonical}"
        );

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn warm_mode_parity() {
        // WARM = FLAG_WARM set, residual disabled.
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xDEAD);
        let handle = temp.handle();
        let model = temp.model();

        // Use encode_with to produce a HOT tile, then manually set FLAG_WARM.
        let key = &gen_keys(0xCAFE, 1, d, d / 2, 0.9, 0.02)[0];
        let query = &gen_keys(0xBEEF, 1, d, d / 2, 0.9, 0.02)[0];

        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile_ptr = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 3, tile_ptr) };
        assert_eq!(rc, SLHA_OK);

        // Set WARM flag.
        {
            let mut t = unsafe { tile_ptr.read_unaligned() };
            t.flags |= FLAG_WARM;
            unsafe { tile_ptr.write_unaligned(t) };
        }

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        let tile = unsafe { tile_ptr.read_unaligned() };
        let mut c_score = -2.0f32;
        let rc = unsafe {
            slha_process_tile(tile_ptr, q_coarse.as_ptr(), q_sign.as_ptr(), &mut c_score)
        };
        assert_eq!(rc, SLHA_OK);

        let canonical = canonical_score(model, &tile, query);
        assert!(
            (c_score - canonical).abs() <= f32::EPSILON * 16.0,
            "WARM: C {c_score} != Rust {canonical}"
        );

        // WARM score should equal the coarse term alone.
        let qc = model.query_coarse(query);
        let coarse_only: f32 = tile
            .dequant_latent()
            .iter()
            .zip(qc.iter())
            .map(|(k, q)| k * q)
            .sum();
        assert!(
            (c_score - coarse_only).abs() <= f32::EPSILON * 16.0,
            "WARM {c_score} != coarse-only {coarse_only}"
        );

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn prepare_query_rejects_invalid_dimensions() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xABCD);
        let handle = temp.handle();

        let query = &gen_keys(0xDCBA, 1, d + 8, d / 2, 0.9, 0.02)[0];
        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];

        // Wrong dimension should fail.
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d + 8,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_ERR_DIMENSION, "expected DIMENSION error");

        // Output must be unchanged (coarse buffer still zero).
        assert_eq!(q_coarse, [0.0f32; D_C]);
        assert_eq!(q_sign, [0u64; RESIDUAL_WORDS]);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn prepare_query_rejects_null_pointers() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xBEEF);
        let handle = temp.handle();

        let query = &gen_keys(0xCAFE, 1, d, d / 2, 0.9, 0.02)[0];
        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];

        // NULL query
        let rc = unsafe {
            slha_prepare_query(
                handle,
                std::ptr::null(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NULL);

        // NULL q_coarse_out
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                std::ptr::null_mut(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NULL);

        // NULL q_sign_out
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NULL);

        // NULL model handle
        let rc = unsafe {
            slha_prepare_query(
                std::ptr::null(),
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NULL);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn score_tiles_rejects_null_pointers() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xDEAD);
        let handle = temp.handle();

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let query = &gen_keys(0xBEEF, 1, d, d / 2, 0.9, 0.02)[0];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        let mut output = -1.0f32;

        // NULL tiles
        let rc = unsafe {
            slha_score_tiles(
                std::ptr::null(),
                1,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut output,
            )
        };
        assert_eq!(rc, SLHA_ERR_NULL);

        // NULL q_coarse
        let tile = zero_tile();
        let rc =
            unsafe { slha_score_tiles(&tile, 1, std::ptr::null(), q_sign.as_ptr(), &mut output) };
        assert_eq!(rc, SLHA_ERR_NULL);

        // NULL q_sign
        let rc =
            unsafe { slha_score_tiles(&tile, 1, q_coarse.as_ptr(), std::ptr::null(), &mut output) };
        assert_eq!(rc, SLHA_ERR_NULL);

        // NULL scores_out
        let rc = unsafe {
            slha_score_tiles(
                &tile,
                1,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NULL);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn prepare_query_rejects_non_finite_q() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xFACE);
        let handle = temp.handle();

        let mut non_finite = gen_keys(0xCAFE, 1, d, d / 2, 0.9, 0.02)
            .into_iter()
            .next()
            .unwrap();
        non_finite[7] = f32::NAN;

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];

        let rc = unsafe {
            slha_prepare_query(
                handle,
                non_finite.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NONFINITE, "expected NONFINITE error");

        // Output must be unchanged.
        assert_eq!(q_coarse, [0.0f32; D_C]);

        // Also test with Infinity.
        let mut inf_query = gen_keys(0xDEAD, 1, d, d / 2, 0.9, 0.02)
            .into_iter()
            .next()
            .unwrap();
        inf_query[3] = f32::INFINITY;
        let rc = unsafe {
            slha_prepare_query(
                handle,
                inf_query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_ERR_NONFINITE);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn score_tiles_rejects_corrupted_tile() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xABCD);
        let handle = temp.handle();

        let key = &gen_keys(0xDCBA, 1, d, d / 2, 0.9, 0.02)[0];
        let query = &gen_keys(0xBEEF, 1, d, d / 2, 0.9, 0.02)[0];

        // Encode a valid tile.
        let mut tile_storage = vec![0u8; size_of::<SciRustSlhaTile>()];
        let tile_ptr = tile_storage.as_mut_ptr() as *mut SciRustSlhaTile;
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 3, tile_ptr) };
        assert_eq!(rc, SLHA_OK);

        // Prepare query.
        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        // Corrupt the tile: set two conflicting codec flags.
        {
            let mut t = unsafe { tile_ptr.read_unaligned() };
            t.flags = FLAG_NF4 | FLAG_TQ3;
            unsafe { tile_ptr.write_unaligned(t) };
        }

        let mut bad_score = 123.0f32;
        let rc = unsafe {
            slha_process_tile(tile_ptr, q_coarse.as_ptr(), q_sign.as_ptr(), &mut bad_score)
        };
        assert_eq!(rc, SLHA_ERR_INVALID_TILE);

        // score_out must be unchanged.
        assert_eq!(bad_score, 123.0);

        // Same for batch scoring with a zeroed tile (scale=0 is valid but
        // produces a score of 0; no error expected).
        let batch_tile = zero_tile(); // valid (flags=0, scale=0)
        let mut batch_scores = [-1.0f32];
        let rc = unsafe {
            slha_score_tiles(
                &batch_tile,
                1,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                batch_scores.as_mut_ptr(),
            )
        };
        // Zeroed tile is valid (flags=0, scale >= 0) and produces score=0.
        assert_eq!(rc, SLHA_OK);
        assert_eq!(batch_scores[0], 0.0);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn score_tiles_rejects_non_finite_score() {
        // Create a tile that produces a non-finite score by using NaN in the
        // scale field while keeping the flags clean.
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xBEAD);
        let handle = temp.handle();

        let query = &gen_keys(0xCAFE, 1, d, d / 2, 0.9, 0.02)[0];
        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        // A tile whose scale is NaN is rejected by validate_tile.
        let mut bad_tile = zero_tile();
        bad_tile.scale = f32::NAN;
        bad_tile.flags = 0; // HOT, uniform INT4

        let mut bad_score = 456.0f32;
        let rc = unsafe {
            slha_process_tile(
                &bad_tile,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut bad_score,
            )
        };
        assert_eq!(rc, SLHA_ERR_NONFINITE);
        assert_eq!(bad_score, 456.0);

        // Non-finite lambda also rejected.
        let mut bad_tile2 = zero_tile();
        bad_tile2.scale = 1.0;
        bad_tile2.dynamic_lambda = f32::NAN;
        bad_tile2.flags = 0;
        let rc = unsafe {
            slha_process_tile(
                &bad_tile2,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut bad_score,
            )
        };
        assert_eq!(rc, SLHA_ERR_NONFINITE);
        assert_eq!(bad_score, 456.0);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn deterministic_repeated_execution() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xABCD);
        let handle = temp.handle();
        let model = temp.model();

        let keys = gen_keys(0xDCBA, 4, d, d / 2, 0.9, 0.02);
        let query = &gen_keys(0xFEED, 1, d, d / 2, 0.9, 0.02)[0];

        // Encode all keys.
        let (_tiles_buf, tiles_ptr) = alloc_tiles(4);
        for (i, key) in keys.iter().enumerate() {
            let tile_ptr = unsafe { tiles_ptr.add(i) };
            let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, i as u32, 3, tile_ptr) };
            assert_eq!(rc, SLHA_OK);
        }

        // Prepare query.
        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        // Two batch calls must produce identical results.
        let mut first = vec![-1.0f32; 4];
        let mut second = vec![-2.0f32; 4];
        let rc1 = unsafe {
            slha_score_tiles(
                tiles_ptr,
                4,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                first.as_mut_ptr(),
            )
        };
        let rc2 = unsafe {
            slha_score_tiles(
                tiles_ptr,
                4,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                second.as_mut_ptr(),
            )
        };
        assert_eq!(rc1, SLHA_OK);
        assert_eq!(rc2, SLHA_OK);
        assert_eq!(first, second, "non-deterministic batch scores");

        // Per-tile scores must also match.
        let tiles_slice = unsafe { std::slice::from_raw_parts(tiles_ptr, 4) };
        for (i, tile) in tiles_slice.iter().enumerate() {
            let expected = canonical_score(model, tile, query);
            let actual = first[i];
            assert!(
                (actual - expected).abs() <= f32::EPSILON * 16.0,
                "tile[{i}]: deterministic {actual} != canonical {expected}"
            );
        }

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn batch_empty_tiles_array_succeeds() {
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xDEAD);
        let _handle = temp.handle();

        let q_coarse = [0.0f32; D_C];
        let q_sign = [0u64; RESIDUAL_WORDS];
        // Empty batch (n_tiles = 0) should succeed trivially.
        let dummy_tile = zero_tile();
        let mut dummy_out = 999.0f32;
        let rc = unsafe {
            slha_score_tiles(
                &dummy_tile,
                0,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut dummy_out,
            )
        };
        assert_eq!(rc, SLHA_OK);
        // Output is not written.
        assert_eq!(dummy_out, 999.0);

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn prepare_query_and_score_tiles_full_roundtrip_all_codecs() {
        let codecs: [(i32, &str); 6] = [
            (0, "int4-single"),
            (1, "int4-grouped"),
            (2, "nf4"),
            (3, "mixed"),
            (4, "tq3"),
            (5, "mix3"),
        ];
        let d = 256usize;
        let temp = setup_model_and_handle(d, 0xABCD);
        let handle = temp.handle();
        let model = temp.model();

        let keys = gen_keys(0xDCBA, 4, d, d / 2, 0.9, 0.02);
        let query = &gen_keys(0xFEED, 1, d, d / 2, 0.9, 0.02)[0];

        let mut q_coarse = [0.0f32; D_C];
        let mut q_sign = [0u64; RESIDUAL_WORDS];
        let rc = unsafe {
            slha_prepare_query(
                handle,
                query.as_ptr(),
                d,
                q_coarse.as_mut_ptr(),
                q_sign.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);

        for (codec_id, codec_name) in &codecs {
            // Encode 4 keys with this codec.
            let (_tiles_buf, tiles_ptr) = alloc_tiles(4);
            for (i, _key) in keys.iter().enumerate() {
                let tile_ptr = unsafe { tiles_ptr.add(i) };
                let rc = unsafe {
                    slha_encode_key(handle, keys[i].as_ptr(), d, i as u32, *codec_id, tile_ptr)
                };
                assert_eq!(rc, SLHA_OK, "encode {codec_name} key {i}: {}", last_error());
            }

            // Score via batch.
            let mut batch_scores = [-1.0f32; 4];
            let rc = unsafe {
                slha_score_tiles(
                    tiles_ptr,
                    4,
                    q_coarse.as_ptr(),
                    q_sign.as_ptr(),
                    batch_scores.as_mut_ptr(),
                )
            };
            assert_eq!(rc, SLHA_OK, "batch score {codec_name}: {}", last_error());

            // Verify each tile against canonical Rust.
            let tiles_slice = unsafe { std::slice::from_raw_parts(tiles_ptr, 4) };
            for (i, tile) in tiles_slice.iter().enumerate() {
                let expected = canonical_score(model, tile, query);
                assert!(
                    (batch_scores[i] - expected).abs() <= f32::EPSILON * 16.0,
                    "{codec_name} tile[{i}]: batch {batch} != canonical {expected}",
                    batch = batch_scores[i]
                );
            }
        }

        // Model and temp directory are released by the `TempModel` RAII guard.
    }

    #[test]
    fn parallel_temp_models_do_not_collide() {
        // Regression test: multiple threads must each create, load, and use a
        // temporary `.slhw` model without path collisions or use-after-delete.
        let d = 256usize;
        let threads: Vec<_> = (0..8)
            .map(|index| {
                std::thread::spawn(move || {
                    let temp = TempModel::new(d, 0x1000 + index as u64);
                    assert_eq!(unsafe { slha_model_dim(temp.handle()) }, d);

                    let key = &gen_keys(0x2000 + index as u64, 1, d, d / 2, 0.9, 0.02)[0];
                    let mut tile = zero_tile();
                    let rc = unsafe {
                        slha_encode_key(temp.handle(), key.as_ptr(), d, index as u32, 3, &mut tile)
                    };
                    assert_eq!(rc, SLHA_OK, "thread {index} encode: {}", last_error());

                    let qc = temp.model().query_coarse(key);
                    let qs = temp.model().sign_bits(key);
                    let mut score = 0.0f32;
                    let rc =
                        unsafe { slha_score_tiles(&tile, 1, qc.as_ptr(), qs.as_ptr(), &mut score) };
                    assert_eq!(rc, SLHA_OK, "thread {index} score: {}", last_error());
                    assert!(score.is_finite(), "thread {index} score is non-finite");
                })
            })
            .collect();

        for thread in threads {
            thread.join().expect("parallel temp model thread panicked");
        }
    }

    #[test]
    fn double_release_fails_with_invalid_handle() {
        let temp = TempModel::new(256, 0x3000);
        let handle = temp.handle();
        // Forget the RAII guard: the manual releases below are the only ones
        // that should touch this handle.
        std::mem::forget(temp);
        // First release succeeds.
        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        // The registry has removed it; a second release must fail cleanly
        // instead of double-freeing.
        assert_eq!(
            unsafe { slha_weights_release(handle) },
            SLHA_ERR_INVALID_HANDLE
        );
        // The compatibility wrapper must not crash on the stale handle either.
        unsafe { slha_weights_free(handle) };
    }

    #[test]
    fn forged_handle_is_rejected_before_deref() {
        // A pointer that is aligned but NOT a registered handle must be
        // rejected by the registry lookup before any read — no crash, no
        // arbitrary-address read.
        let forged = SlhaModel {
            magic: MODEL_MAGIC,
            inner: temp_model(256, 0x3001),
        };
        let rc = unsafe { slha_model_dim(&forged) };
        assert_eq!(rc, 0, "forged handle must be rejected");
        assert!(!last_error().is_empty(), "rejection must set the error");
    }

    #[test]
    fn absurd_tile_count_is_rejected() {
        let temp = TempModel::new(256, 0x3002);
        let _handle = temp.handle();
        let tile = zero_tile();
        let qc = [0.0f32; D_C];
        let qs = [0u64; RESIDUAL_WORDS];
        let mut score = 0.0f32;
        let rc =
            unsafe { slha_score_tiles(&tile, MAX_TILES + 1, qc.as_ptr(), qs.as_ptr(), &mut score) };
        assert_eq!(rc, SLHA_ERR_DIMENSION, "oversized n_tiles must be rejected");
    }

    fn temp_model(d: usize, seed: u64) -> LearnedModel {
        let keys = gen_keys(seed, 1, d, d, 0.9, 0.02);
        LearnedModel::fit_with(&keys, d, seed, false, false)
    }
}
