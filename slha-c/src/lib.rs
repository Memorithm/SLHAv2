use scirust::attention::slha_v2::{LatentCodec, SciRustSlhaTile, D_C, RESIDUAL_WORDS};
use scirust::audit;
use scirust::learned::LearnedModel;
use scirust::weights;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;

/// Opaque handle for the SLHA context (if needed in future, currently stateless).
#[repr(C)]
pub struct SlhaContext {
    _unused: [u8; 0],
}

/// Initialize the SLHA environment.
///
/// The kernel is currently stateless, so no context is allocated. We still
/// return a non-null, well-aligned sentinel so C callers can keep a uniform
/// `if (ctx == NULL) { /* error */ }` check without special-casing. The handle
/// is a dangling (zero-sized) pointer that must never be dereferenced; when a
/// real context is added later this will allocate it and gain a paired
/// `slha_shutdown`.
#[no_mangle]
pub extern "C" fn slha_init() -> *mut SlhaContext {
    NonNull::<SlhaContext>::dangling().as_ptr()
}

/// Process a single tile and compute the score.
///
/// # Safety
/// `tile`, `q_coarse`, and `q_sign` must be valid non-null pointers.
/// `q_coarse` must point to an array of `D_C` (128) f32s.
/// `q_sign` must point to an array of `RESIDUAL_WORDS` (4) u64s.
#[no_mangle]
pub unsafe extern "C" fn slha_process_tile(
    tile: *const SciRustSlhaTile,
    q_coarse: *const f32,
    q_sign: *const u64,
    score_out: *mut f32,
) -> i32 {
    let result = catch_unwind(|| {
        if tile.is_null() || q_coarse.is_null() || q_sign.is_null() || score_out.is_null() {
            return -1;
        }

        let tile_ref = &*tile;
        let q_coarse_ref = &*(q_coarse as *const [f32; D_C]);
        let q_sign_ref = &*(q_sign as *const [u64; RESIDUAL_WORDS]);

        *score_out = tile_ref.compute_score(q_coarse_ref, q_sign_ref);
        0
    });

    result.unwrap_or(-2) // -2 if panic occurred
}

/// Run the self-audit and return a JSON string.
///
/// # Safety
/// The returned pointer must be freed using `slha_free_string`.
#[no_mangle]
pub unsafe extern "C" fn slha_audit() -> *mut c_char {
    let result = catch_unwind(|| {
        let report = audit::run();
        let s = report.to_compact();
        // CString::into_raw returns *mut c_char on every target; using c_char
        // (not i8) keeps this compiling on aarch64 where c_char == u8.
        std::ffi::CString::new(s).unwrap().into_raw()
    });

    result.unwrap_or(std::ptr::null_mut())
}

/// Free a string allocated by the library.
///
/// # Safety
/// `s` must be a pointer returned by `slha_audit`.
#[no_mangle]
pub unsafe extern "C" fn slha_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: `s` was produced by CString::into_raw in slha_audit and is
        // freed exactly once by the caller.
        let _ = std::ffi::CString::from_raw(s);
    }
}

// ---------------------------------------------------------------------------
// Codec path — the bridge an inference engine needs: load a learned
// projection, encode a d-dim key vector into a 128-byte tile, and decode the
// tile's latent back into the original d-dim space. See docs/INTEGRATION.md.
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded SLHA projection model (`.slhw`). Create with
/// [`slha_weights_load`], release with [`slha_weights_free`].
pub struct SlhaModel {
    inner: LearnedModel,
}

/// Map the C codec id to a [`LatentCodec`]:
/// 0=int4-single, 1=int4-grouped, 2=nf4, 3=mixed, 4=tq3, 5=mix3.
fn codec_from_int(c: i32) -> Option<LatentCodec> {
    match c {
        0 => Some(LatentCodec::Int4Single),
        1 => Some(LatentCodec::Int4Grouped),
        2 => Some(LatentCodec::Nf4),
        3 => Some(LatentCodec::Mixed),
        4 => Some(LatentCodec::Tq3),
        5 => Some(LatentCodec::Mix3),
        _ => None,
    }
}

/// Load a projection model from a `.slhw` file. Returns NULL on any error
/// (null/invalid path, read/parse failure).
///
/// # Safety
/// `path` must be NULL or a valid NUL-terminated UTF-8 C string. The returned
/// handle must be released with [`slha_weights_free`].
#[no_mangle]
pub unsafe extern "C" fn slha_weights_load(path: *const c_char) -> *mut SlhaModel {
    catch_unwind(|| {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: caller guarantees a valid NUL-terminated string.
        let Ok(s) = std::ffi::CStr::from_ptr(path).to_str() else {
            return std::ptr::null_mut();
        };
        match weights::load(s) {
            Ok(inner) => Box::into_raw(Box::new(SlhaModel { inner })),
            Err(_) => std::ptr::null_mut(),
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

/// The projection's input dimension `d` (the length of a key/query vector).
/// Returns 0 if `model` is NULL.
///
/// # Safety
/// `model` must be NULL or a valid handle from [`slha_weights_load`].
#[no_mangle]
pub unsafe extern "C" fn slha_model_dim(model: *const SlhaModel) -> usize {
    if model.is_null() {
        return 0;
    }
    // SAFETY: non-null handle from slha_weights_load.
    (*model).inner.d
}

/// Encode a `d`-dimensional key vector into a 128-byte tile.
///
/// `codec` selects the latent quantiser (see [`codec_from_int`]). The tile is
/// written to `*out_tile`. Returns 0 on success; -1 null argument, -2 panic,
/// -3 dimension mismatch (`d` != model dim), -4 unknown codec.
///
/// # Safety
/// `model` is a valid handle; `key` points to `d` f32s; `out_tile` points to
/// writable storage for one tile.
#[no_mangle]
pub unsafe extern "C" fn slha_encode_key(
    model: *const SlhaModel,
    key: *const f32,
    d: usize,
    pos: u32,
    codec: i32,
    out_tile: *mut SciRustSlhaTile,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() || key.is_null() || out_tile.is_null() {
            return -1;
        }
        // SAFETY: guarded above; caller guarantees the pointees.
        let m = &(*model).inner;
        if d != m.d {
            return -3;
        }
        let Some(codec) = codec_from_int(codec) else {
            return -4;
        };
        let key_slice = std::slice::from_raw_parts(key, d);
        let tile = m.encode_with(key_slice, pos, false, codec);
        std::ptr::write(out_tile, tile);
        0
    }))
    .unwrap_or(-2)
}

/// Decode a tile's latent back into the original `d`-dimensional space
/// (dequantise + inverse projection / reconstruct). `out` receives `d` f32s.
/// Returns 0 on success; -1 null, -2 panic, -3 dimension mismatch.
///
/// # Safety
/// `model` is a valid handle; `tile` points to one tile; `out` points to
/// writable storage for `d` f32s.
#[no_mangle]
pub unsafe extern "C" fn slha_decode_latent(
    model: *const SlhaModel,
    tile: *const SciRustSlhaTile,
    out: *mut f32,
    d: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() || tile.is_null() || out.is_null() {
            return -1;
        }
        // SAFETY: guarded above.
        let m = &(*model).inner;
        if d != m.d {
            return -3;
        }
        let h = (*tile).dequant_latent();
        let recon = m.reconstruct(&h); // length d
        let out_slice = std::slice::from_raw_parts_mut(out, d);
        out_slice.copy_from_slice(&recon);
        0
    }))
    .unwrap_or(-2)
}

/// Release a model loaded by [`slha_weights_load`]. NULL is a no-op.
///
/// # Safety
/// `model` must be NULL or a handle from [`slha_weights_load`], freed once.
#[no_mangle]
pub unsafe extern "C" fn slha_weights_free(model: *mut SlhaModel) {
    if !model.is_null() {
        // SAFETY: produced by Box::into_raw in slha_weights_load, freed once.
        drop(Box::from_raw(model));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust::learned::gen_keys;

    #[test]
    fn encode_decode_round_trip_through_c_abi() {
        // Fit a projection on synthetic keys, persist it, then drive the C ABI
        // exactly as an engine would: load, encode a key, decode it back, and
        // check the reconstruction is sane (positive SNR vs the original key).
        let d = 256usize;
        let train = gen_keys(1, 512, d, 64, 0.9, 0.02);
        let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);
        let mut path = std::env::temp_dir();
        path.push(format!("slha_c_roundtrip_{}.slhw", std::process::id()));
        weights::save(path.to_str().unwrap(), &model, 0xC0FFEE, false).unwrap();

        let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        // SAFETY: valid path; handle freed at the end.
        let handle = unsafe { slha_weights_load(cpath.as_ptr()) };
        assert!(!handle.is_null(), "load failed");
        assert_eq!(unsafe { slha_model_dim(handle) }, d);

        let key = &gen_keys(2, 1, d, 64, 0.9, 0.02)[0];
        let mut tile = unsafe { std::mem::zeroed::<SciRustSlhaTile>() };
        // codec 5 = mix3
        let rc = unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 5, &mut tile) };
        assert_eq!(rc, 0, "encode failed");

        let mut out = vec![0.0f32; d];
        let rc = unsafe { slha_decode_latent(handle, &tile, out.as_mut_ptr(), d) };
        assert_eq!(rc, 0, "decode failed");

        // The decode must reconstruct real signal (clearly positive SNR vs the
        // original key), proving the load→encode→decode C path is wired end to
        // end. This is a plumbing check, not a projection-quality benchmark —
        // rank-128 truncation of a 256-dim key at this spectrum inherently caps
        // the SNR (offline_validation measures quality proper).
        let sig: f32 = key.iter().map(|x| x * x).sum();
        let err: f32 = key.iter().zip(&out).map(|(a, b)| (a - b).powi(2)).sum();
        let snr = 10.0 * (sig / err.max(1e-12)).log10();
        assert!(snr > 2.0, "reconstruction SNR too low: {snr} dB");

        // Error codes.
        assert_eq!(
            unsafe { slha_encode_key(handle, key.as_ptr(), d + 1, 0, 5, &mut tile) },
            -3,
            "dim mismatch must be -3"
        );
        assert_eq!(
            unsafe { slha_encode_key(handle, key.as_ptr(), d, 0, 99, &mut tile) },
            -4,
            "unknown codec must be -4"
        );
        assert_eq!(
            unsafe { slha_encode_key(std::ptr::null(), key.as_ptr(), d, 0, 5, &mut tile) },
            -1,
            "null model must be -1"
        );

        unsafe { slha_weights_free(handle) };
        let _ = std::fs::remove_file(&path);
    }
}
