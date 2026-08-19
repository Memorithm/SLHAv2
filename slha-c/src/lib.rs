// Every unsafe block must state its precondition. Permanent gate: clippy
// runs with -D warnings in CI, so an undocumented block fails the build.
#![deny(clippy::undocumented_unsafe_blocks)]

use scirust::attention::slha_v2::{
    LatentCodec, SciRustSlhaTile, D_C, FLAG_MIX3, FLAG_MIXED, FLAG_NF4, FLAG_TQ3, FLAG_TQ3_NOCORR,
    FLAG_WARM, RESIDUAL_WORDS,
};
use scirust::audit;
use scirust::learned::LearnedModel;
use scirust::weights;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::mem::{align_of, size_of};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock, RwLock};

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

#[allow(unknown_lints, clippy::manual_is_multiple_of)]
// `manual_is_multiple_of` was added in a clippy newer than this repo's MSRV;
// the allow is harmless on older toolchains once unknown_lints is permitted.
fn pointer_is_aligned<T>(pointer: *const T) -> bool {
    (pointer as usize) % align_of::<T>() == 0
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

/// Opaque token returned to C. The projection itself is owned by the
/// process-wide registry and is never reached by dereferencing this pointer.
#[repr(C, align(8))]
pub struct SlhaModel {
    _opaque: u64,
}

struct ModelEntry {
    token: Box<SlhaModel>,
    inner: Arc<LearnedModel>,
}

// The boxes are intentional: their stable allocations quarantine every
// released token address and prevent stale handles becoming live again.
#[allow(clippy::vec_box)]
struct ModelRegistry {
    live: HashMap<usize, ModelEntry>,
    retired_tokens: Vec<Box<SlhaModel>>,
}

fn model_registry() -> &'static RwLock<ModelRegistry> {
    static REGISTRY: OnceLock<RwLock<ModelRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        RwLock::new(ModelRegistry {
            live: HashMap::new(),
            retired_tokens: Vec::new(),
        })
    })
}

fn register_model(inner: LearnedModel) -> *mut SlhaModel {
    let mut token = Box::new(SlhaModel { _opaque: 0 });
    let pointer = (&mut *token) as *mut SlhaModel;
    let entry = ModelEntry {
        token,
        inner: Arc::new(inner),
    };
    let previous = model_registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .live
        .insert(pointer as usize, entry);
    debug_assert!(previous.is_none());
    pointer
}

fn model_arc(model: *const SlhaModel) -> Result<Arc<LearnedModel>, FfiError> {
    if model.is_null() {
        return Err(ffi_error(SLHA_ERR_NULL, "model handle is NULL"));
    }

    if !pointer_is_aligned(model) {
        return Err(ffi_error(
            SLHA_ERR_INVALID_HANDLE,
            "model handle is misaligned",
        ));
    }

    model_registry()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .live
        .get(&(model as usize))
        .map(|entry| Arc::clone(&entry.inner))
        .ok_or_else(|| {
            ffi_error(
                SLHA_ERR_INVALID_HANDLE,
                "model handle is not a live SLHA handle",
            )
        })
}

fn unregister_model(model: *mut SlhaModel) -> bool {
    let mut registry = model_registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(entry) = registry.live.remove(&(model as usize)) else {
        return false;
    };
    registry.retired_tokens.push(entry.token);
    true
}

/// Defensive upper bound for caller-controlled batch lengths.
pub const MAX_TILES: usize = 1 << 20;

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
        // SAFETY: same caller contract — q_coarse holds D_C readable f32s.
        let q_coarse = unsafe { read_array_unaligned::<f32, D_C>(q_coarse) };
        // SAFETY: same caller contract — q_sign holds RESIDUAL_WORDS readable u64s.
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

/// Prepare a query for direct SLHA scoring.
///
/// Returns 0 on success. Output buffers are not modified on failure.
///
/// # Safety
/// `model` must be a live model handle. `q` points to `d` readable f32s.
/// `q_coarse` points to writable storage for `SLHA_D_C` f32s.
/// `q_sign` points to writable storage for `SLHA_RESIDUAL_WORDS` uint64s.
/// Unaligned input/output storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_prepare_query(
    model: *const SlhaModel,
    q: *const f32,
    d: usize,
    q_coarse: *mut f32,
    q_sign: *mut u64,
) -> i32 {
    ffi_status(|| {
        if q.is_null() || q_coarse.is_null() || q_sign.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_prepare_query received a NULL data pointer",
            ));
        }

        // SAFETY: the caller promises a live handle.
        let model = model_arc(model)?;

        if d != model.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "query dimension mismatch: received {d}, expected {}",
                    model.d
                ),
            ));
        }

        // SAFETY: dimension equality is checked before reading.
        let q = unsafe { read_vec_unaligned(q, d) };
        validate_finite(&q, "q")?;

        let coarse = model.query_coarse(&q);
        let signs = model.sign_bits(&q);

        validate_finite(&coarse, "q_coarse")?;

        // SAFETY: the caller guarantees writable storage of the documented size.
        unsafe { write_slice_unaligned(q_coarse, &coarse) };
        // SAFETY: same caller contract — q_sign has RESIDUAL_WORDS writable u64s.
        unsafe { write_slice_unaligned(q_sign, &signs) };
        Ok(())
    })
}

/// Score one tile with a model-validated prepared query.
///
/// Returns 0 on success. Output is not modified on failure.
///
/// # Safety
/// `model` must be a live model handle. `tile`, `q_coarse`, `q_sign`, and
/// `score_out` must point to readable or writable storage of the documented
/// size. Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_score_tile(
    model: *const SlhaModel,
    tile: *const SciRustSlhaTile,
    q_coarse: *const f32,
    q_sign: *const u64,
    score_out: *mut f32,
) -> i32 {
    ffi_status(|| {
        if tile.is_null() || q_coarse.is_null() || q_sign.is_null() || score_out.is_null() {
            return Err(ffi_error(
                SLHA_ERR_NULL,
                "slha_score_tile received a NULL pointer",
            ));
        }

        // SAFETY: the caller promises a live handle.
        let _model = model_arc(model)?;

        // SAFETY: the caller guarantees readable storage.
        let tile = unsafe { tile.read_unaligned() };
        // SAFETY: same caller contract — q_coarse holds D_C readable f32s.
        let q_coarse = unsafe { read_array_unaligned::<f32, D_C>(q_coarse) };
        // SAFETY: same caller contract — q_sign holds RESIDUAL_WORDS readable u64s.
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

/// Score N tiles with a single prepared query.
///
/// Returns 0 on success. Output is not modified on failure.
///
/// # Safety
/// `model` must be a live model handle. `tiles` points to `n_tiles` readable
/// tiles. `q_coarse` points to `SLHA_D_C` readable f32s. `q_sign` points to
/// `SLHA_RESIDUAL_WORDS` readable uint64s. `scores_out` points to `n_tiles`
/// writable f32s. Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_score_tiles(
    model: *const SlhaModel,
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

        let _model = model_arc(model)?;

        if n_tiles > MAX_TILES {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!("n_tiles={n_tiles} exceeds the safety bound {MAX_TILES}"),
            ));
        }

        if n_tiles == 0 {
            return Ok(());
        }

        // SAFETY: caller contract (# Safety) — q_coarse holds D_C readable
        // f32s and q_sign holds RESIDUAL_WORDS readable u64s.
        let q_coarse = unsafe { read_array_unaligned::<f32, D_C>(q_coarse) };
        // SAFETY: see the caller contract stated on the previous block.
        let q_sign = unsafe { read_array_unaligned::<u64, RESIDUAL_WORDS>(q_sign) };
        validate_finite(&q_coarse, "q_coarse")?;

        for i in 0..n_tiles {
            // SAFETY: the caller guarantees n_tiles readable tiles starting at
            // tiles; each tile is read unaligned to avoid Rust alignment checks.
            let tile = unsafe { tiles.add(i).read_unaligned() };

            validate_tile(&tile)?;

            let score = tile.compute_score(&q_coarse, &q_sign);

            if !score.is_finite() {
                return Err(ffi_error(
                    SLHA_ERR_NONFINITE,
                    format!("tile {i} scoring produced a non-finite result"),
                ));
            }

            // SAFETY: the caller guarantees n_tiles writable f32s.
            unsafe { scores_out.add(i).write_unaligned(score) };
        }

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

        Ok(register_model(inner))
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

    match catch_unwind(AssertUnwindSafe(|| model_arc(model).map(|model| model.d))) {
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
        let model = model_arc(model)?;

        if d != model.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!("key dimension mismatch: received {d}, expected {}", model.d),
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

        let tile = model.encode_with(&key, pos, false, codec);
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
        let model = model_arc(model)?;

        if d != model.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "output dimension mismatch: received {d}, expected {}",
                    model.d
                ),
            ));
        }

        // SAFETY: the caller guarantees readable storage for one tile.
        let tile = unsafe { tile.read_unaligned() };
        validate_tile(&tile)?;

        let latent = tile.dequant_latent();
        validate_finite(&latent, "dequantized_latent")?;

        let reconstruction = model.reconstruct(&latent);

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
        let model = model_arc(model)?;

        if d != model.d {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!(
                    "output dimension mismatch: received {d}, expected {}",
                    model.d
                ),
            ));
        }

        // SAFETY: the caller guarantees readable storage for one tile.
        let tile = unsafe { tile.read_unaligned() };
        validate_tile(&tile)?;

        let reconstruction = model.reconstruct_from_tile(&tile);

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
        if !pointer_is_aligned(model) {
            return Err(ffi_error(
                SLHA_ERR_INVALID_HANDLE,
                "model handle is misaligned",
            ));
        }

        if !unregister_model(model) {
            return Err(ffi_error(
                SLHA_ERR_INVALID_HANDLE,
                "model handle is not a live SLHA handle",
            ));
        }

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
// SAFETY (module-wide): every unsafe block below calls one of the crate's own
// `unsafe extern "C"` entry points with pointers derived from live local
// values the test owns, and lengths that are the true buffer lengths. The
// per-call preconditions are the documented `# Safety` contracts of those
// entry points; repeating that sentence on ~70 call sites would bury the
// production annotations this lint exists to protect.
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use super::*;
    use scirust::learned::gen_keys;

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
        let train = gen_keys(1, 512, d, 64, 0.9, 0.02);
        let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);
        let mut path = std::env::temp_dir();
        path.push(format!("slha_c_roundtrip_{}.slhw", std::process::id()));
        weights::save(
            path.to_str().expect("temporary path must be UTF-8"),
            &model,
            0xC0FFEE,
            false,
        )
        .expect("save weights");

        let cpath = CString::new(path.to_str().expect("temporary path must be valid UTF-8"))
            .expect("temporary path cannot contain NUL");

        // SAFETY: valid path; handle released at the end.
        let handle = unsafe { slha_weights_load(cpath.as_ptr()) };
        assert!(!handle.is_null(), "load failed: {}", last_error());
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

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn decode_key_reconstructs_residual_beyond_latent_only() {
        let d = 256usize;
        let train = gen_keys(1, 512, d, 64, 0.9, 0.02);
        let model = LearnedModel::fit(&train, d, 0xB0BA, false);
        let mut path = std::env::temp_dir();
        path.push(format!("slha_c_decode_key_{}.slhw", std::process::id()));
        weights::save(path.to_str().unwrap(), &model, 0xB0BA, false).expect("save weights");

        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let handle = unsafe { slha_weights_load(cpath.as_ptr()) };
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

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);

        fn snr(a: &[f32], b: &[f32]) -> f32 {
            let signal: f32 = a.iter().map(|x| x * x).sum();
            let error: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
            10.0 * (signal / error.max(1e-12)).log10()
        }
    }

    // ----------------------------------------------------------------------
    // Direct compressed-score parity tests (Milestone 1)
    // ----------------------------------------------------------------------

    use scirust::attention::slha_v2::LatentCodec;

    fn fitted_and_loaded_model(
        d: usize,
        seed: u64,
    ) -> (LearnedModel, std::path::PathBuf, *mut SlhaModel) {
        let train = gen_keys(seed.wrapping_add(1), 128, d, 64, 0.9, 0.02);
        let model = LearnedModel::fit(&train, d, seed, false);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "slha_c_direct_score_{}_{}.slhw",
            seed,
            std::process::id()
        ));
        weights::save(path.to_str().unwrap(), &model, seed, false).expect("save weights");

        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let handle = unsafe { slha_weights_load(cpath.as_ptr()) };
        assert!(!handle.is_null(), "load failed: {}", last_error());
        assert_eq!(unsafe { slha_model_dim(handle) }, d);

        (model, path, handle)
    }

    #[test]
    fn model_handles_reject_forgery_double_release_and_oversized_batches() {
        let (_model, path, handle) = fitted_and_loaded_model(160, 0xA11CE);
        let tile = zero_tile();
        let coarse = [0.0f32; D_C];
        let signs = [0u64; RESIDUAL_WORDS];
        let mut score = 123.0f32;

        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    &tile,
                    MAX_TILES + 1,
                    coarse.as_ptr(),
                    signs.as_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_DIMENSION
        );
        assert_eq!(score, 123.0);

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        assert_eq!(
            unsafe { slha_weights_release(handle) },
            SLHA_ERR_INVALID_HANDLE
        );
        assert!(last_error().contains("not a live"));

        let forged = std::ptr::dangling::<SlhaModel>();
        assert_eq!(unsafe { slha_model_dim(forged) }, 0);
        assert!(last_error().contains("not a live"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_rust_canonical_matches_c_abi() {
        let d = 160usize;
        let (model, path, handle) = fitted_and_loaded_model(d, 0xD1EC7);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let key = &gen_keys(0x5678, 1, d, 64, 0.9, 0.02)[0];

        let rust_coarse = model.query_coarse(q);
        let rust_sign = model.sign_bits(q);
        let tile = model.encode(key, 0, false);
        let rust_score = tile.compute_score(&rust_coarse, &rust_sign);

        let mut c_coarse = [0.0f32; D_C];
        let mut c_sign = [0u64; RESIDUAL_WORDS];
        let mut c_score = f32::NAN;

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    c_coarse.as_mut_ptr(),
                    c_sign.as_mut_ptr(),
                )
            },
            SLHA_OK,
            "prepare_query failed: {}",
            last_error()
        );
        assert_eq!(
            unsafe {
                slha_score_tile(
                    handle,
                    &tile,
                    c_coarse.as_ptr(),
                    c_sign.as_ptr(),
                    &mut c_score,
                )
            },
            SLHA_OK,
            "score_tile failed: {}",
            last_error()
        );

        assert!(
            (rust_score - c_score).abs() < 1e-4,
            "Rust canonical score {rust_score} != C ABI score {c_score}"
        );

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_batch_matches_repeated_scalar() {
        let d = 160usize;
        let (model, path, handle) = fitted_and_loaded_model(d, 0xBA7C4);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let keys: Vec<Vec<f32>> = (0..16)
            .map(|i| gen_keys(0x5000 + i as u64, 1, d, 64, 0.9, 0.02)[0].clone())
            .collect();
        let tiles: Vec<SciRustSlhaTile> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| model.encode(k, i as u32, false))
            .collect();

        let mut c_coarse = [0.0f32; D_C];
        let mut c_sign = [0u64; RESIDUAL_WORDS];
        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    c_coarse.as_mut_ptr(),
                    c_sign.as_mut_ptr(),
                )
            },
            SLHA_OK
        );

        let mut batch_scores = vec![0.0f32; tiles.len()];
        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    tiles.as_ptr(),
                    tiles.len(),
                    c_coarse.as_ptr(),
                    c_sign.as_ptr(),
                    batch_scores.as_mut_ptr(),
                )
            },
            SLHA_OK,
            "score_tiles failed: {}",
            last_error()
        );

        for (i, tile) in tiles.iter().enumerate() {
            let mut scalar = f32::NAN;
            assert_eq!(
                unsafe { slha_process_tile(tile, c_coarse.as_ptr(), c_sign.as_ptr(), &mut scalar) },
                SLHA_OK
            );
            assert!(
                (batch_scores[i] - scalar).abs() < 1e-6,
                "batch[{}] {} != scalar {}",
                i,
                batch_scores[i],
                scalar
            );
        }

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    fn parity_for_codec(codec: LatentCodec, seed: u64) {
        let d = 160usize;
        let (model, path, handle) = fitted_and_loaded_model(d, seed);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let key = &gen_keys(0x5678, 1, d, 64, 0.9, 0.02)[0];

        let rust_coarse = model.query_coarse(q);
        let rust_sign = model.sign_bits(q);
        let tile = model.encode_with(key, 0, false, codec);
        let rust_score = tile.compute_score(&rust_coarse, &rust_sign);

        let mut c_coarse = [0.0f32; D_C];
        let mut c_sign = [0u64; RESIDUAL_WORDS];
        let mut c_score = f32::NAN;

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    c_coarse.as_mut_ptr(),
                    c_sign.as_mut_ptr(),
                )
            },
            SLHA_OK
        );
        assert_eq!(
            unsafe {
                slha_score_tile(
                    handle,
                    &tile,
                    c_coarse.as_ptr(),
                    c_sign.as_ptr(),
                    &mut c_score,
                )
            },
            SLHA_OK
        );

        assert!(
            (rust_score - c_score).abs() < 1e-4,
            "codec {:?} Rust score {rust_score} != C score {c_score}",
            codec
        );

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_mixed_parity() {
        parity_for_codec(LatentCodec::Mixed, 0xA1A3D);
    }

    #[test]
    fn direct_score_mix3_parity() {
        parity_for_codec(LatentCodec::Mix3, 0xA1A3E);
    }

    fn parity_for_mode(warm: bool, seed: u64) {
        let d = 160usize;
        let (model, path, handle) = fitted_and_loaded_model(d, seed);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let key = &gen_keys(0x5678, 1, d, 64, 0.9, 0.02)[0];

        let rust_coarse = model.query_coarse(q);
        let rust_sign = model.sign_bits(q);
        let mut tile = model.encode(key, 0, false);
        if warm {
            tile.flags |= FLAG_WARM;
        }
        let rust_score = tile.compute_score(&rust_coarse, &rust_sign);

        let mut c_coarse = [0.0f32; D_C];
        let mut c_sign = [0u64; RESIDUAL_WORDS];
        let mut c_score = f32::NAN;

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    c_coarse.as_mut_ptr(),
                    c_sign.as_mut_ptr(),
                )
            },
            SLHA_OK
        );
        assert_eq!(
            unsafe {
                slha_score_tile(
                    handle,
                    &tile,
                    c_coarse.as_ptr(),
                    c_sign.as_ptr(),
                    &mut c_score,
                )
            },
            SLHA_OK
        );

        assert!(
            (rust_score - c_score).abs() < 1e-4,
            "warm={warm} Rust score {rust_score} != C score {c_score}"
        );

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_hot_parity() {
        parity_for_mode(false, 0xB0777);
    }

    #[test]
    fn direct_score_warm_parity() {
        parity_for_mode(true, 0xC4A10);
    }

    #[test]
    fn direct_score_invalid_dimensions_rejected() {
        let d = 160usize;
        let (_model, path, handle) = fitted_and_loaded_model(d, 0xD1AEA);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let mut c_coarse = [0.0f32; D_C];
        let mut c_sign = [0u64; RESIDUAL_WORDS];

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d + 1,
                    c_coarse.as_mut_ptr(),
                    c_sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_DIMENSION
        );
        assert!(last_error().contains("dimension mismatch"));

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d - 1,
                    c_coarse.as_mut_ptr(),
                    c_sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_DIMENSION
        );

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_null_pointers_rejected() {
        let d = 160usize;
        let (_model, path, handle) = fitted_and_loaded_model(d, 0xD0115);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let mut coarse = [0.0f32; D_C];
        let mut sign = [0u64; RESIDUAL_WORDS];
        let tile = zero_tile();
        let mut score = 0.0f32;

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    std::ptr::null(),
                    q.as_ptr(),
                    d,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    std::ptr::null(),
                    d,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    std::ptr::null_mut(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    coarse.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            SLHA_ERR_NULL
        );

        assert_eq!(
            unsafe {
                slha_score_tile(
                    std::ptr::null(),
                    &tile,
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tile(
                    handle,
                    std::ptr::null(),
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe { slha_score_tile(handle, &tile, std::ptr::null(), sign.as_ptr(), &mut score) },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tile(handle, &tile, coarse.as_ptr(), std::ptr::null(), &mut score)
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tile(
                    handle,
                    &tile,
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    std::ptr::null_mut(),
                )
            },
            SLHA_ERR_NULL
        );

        assert_eq!(
            unsafe {
                slha_score_tiles(
                    std::ptr::null(),
                    &tile,
                    1,
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    std::ptr::null(),
                    1,
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    &tile,
                    1,
                    std::ptr::null(),
                    sign.as_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    &tile,
                    1,
                    coarse.as_ptr(),
                    std::ptr::null(),
                    &mut score,
                )
            },
            SLHA_ERR_NULL
        );
        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    &tile,
                    1,
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    std::ptr::null_mut(),
                )
            },
            SLHA_ERR_NULL
        );

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_non_finite_q_rejected() {
        let d = 160usize;
        let (_model, path, handle) = fitted_and_loaded_model(d, 0xE1A11);

        let mut q = gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0].clone();
        q[0] = f32::NAN;

        let mut coarse = [0.0f32; D_C];
        let mut sign = [0u64; RESIDUAL_WORDS];

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_NONFINITE
        );
        assert!(last_error().contains("not finite"));

        q[0] = f32::INFINITY;
        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_ERR_NONFINITE
        );

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_corrupted_tile_rejected() {
        let d = 160usize;
        let (model, path, handle) = fitted_and_loaded_model(d, 0xC0BB0);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let key = &gen_keys(0x5678, 1, d, 64, 0.9, 0.02)[0];

        let tile = model.encode(key, 0, false);
        let mut corrupted = tile;
        corrupted.flags = FLAG_MIXED | FLAG_MIX3; // two codecs at once

        let mut coarse = model.query_coarse(q);
        let mut sign = model.sign_bits(q);
        let mut score = 0.0f32;

        assert_eq!(
            unsafe {
                slha_score_tile(
                    handle,
                    &corrupted,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                    &mut score,
                )
            },
            SLHA_ERR_INVALID_TILE
        );
        assert!(last_error().contains("more than one latent codec"));

        // Output must not have been modified on failure.
        assert_eq!(score, 0.0f32);

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_zero_tiles_is_noop() {
        let d = 160usize;
        let (_model, path, handle) = fitted_and_loaded_model(d, 0x2EA00);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let mut coarse = [0.0f32; D_C];
        let mut sign = [0u64; RESIDUAL_WORDS];
        let tile = zero_tile();
        let mut sentinel = 123.0f32;

        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_OK
        );
        assert_eq!(
            unsafe {
                slha_score_tiles(
                    handle,
                    &tile,
                    0,
                    coarse.as_ptr(),
                    sign.as_ptr(),
                    &mut sentinel,
                )
            },
            SLHA_OK
        );
        assert_eq!(sentinel, 123.0f32, "n_tiles=0 must not write scores_out");

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_score_deterministic_repeated_execution() {
        let d = 160usize;
        let (model, path, handle) = fitted_and_loaded_model(d, 0xD37E8);

        let q = &gen_keys(0x1234, 1, d, 64, 0.9, 0.02)[0];
        let key = &gen_keys(0x5678, 1, d, 64, 0.9, 0.02)[0];
        let tile = model.encode(key, 0, false);

        let mut coarse = [0.0f32; D_C];
        let mut sign = [0u64; RESIDUAL_WORDS];
        assert_eq!(
            unsafe {
                slha_prepare_query(
                    handle,
                    q.as_ptr(),
                    d,
                    coarse.as_mut_ptr(),
                    sign.as_mut_ptr(),
                )
            },
            SLHA_OK
        );

        let mut first = f32::NAN;
        assert_eq!(
            unsafe { slha_score_tile(handle, &tile, coarse.as_ptr(), sign.as_ptr(), &mut first) },
            SLHA_OK
        );

        for _ in 0..10 {
            let mut score = f32::NAN;
            assert_eq!(
                unsafe {
                    slha_score_tile(handle, &tile, coarse.as_ptr(), sign.as_ptr(), &mut score)
                },
                SLHA_OK
            );
            assert_eq!(score, first, "repeated execution must be bit-identical");
        }

        assert_eq!(unsafe { slha_weights_release(handle) }, SLHA_OK);
        let _ = std::fs::remove_file(&path);
    }
}
