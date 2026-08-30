use scirust::attention::slha_v2::{SciRustSlhaTile, D_C, RESIDUAL_WORDS};
use slhav2_vram::{
    codec,
    elastic_cache::{ElasticKvCache, PhysicalTier},
};
use std::collections::HashMap;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::{
    ffi_error, ffi_status, pointer_is_aligned, validate_tile, FfiError, MAX_TILES,
    SLHA_ERR_DIMENSION, SLHA_ERR_INVALID_HANDLE, SLHA_ERR_NULL, SLHA_ERR_PANIC, SLHA_OK,
};

/// A requested slot exists but is physically COLD and therefore cannot
/// participate in dense attention until the caller explicitly restores it.
pub const SLHA_ERR_NOT_RESIDENT: i32 = -10;

pub const SLHA_ELASTIC_TIER_HOT: i32 = 0;
pub const SLHA_ELASTIC_TIER_WARM: i32 = 1;
pub const SLHA_ELASTIC_TIER_COLD: i32 = 2;
pub const SLHA_ELASTIC_TIER_PINNED: i32 = 3;
pub const SLHA_ELASTIC_TIER_ABSENT: i32 = -1;

/// Opaque token returned to C. The cache itself is registry-owned and this
/// pointer is never dereferenced after crossing the ABI boundary.
#[repr(C, align(8))]
pub struct SlhaElasticKvCache {
    _opaque: u64,
}

struct CacheEntry {
    token: Box<SlhaElasticKvCache>,
    inner: Arc<Mutex<ElasticKvCache>>,
}

// Stable token allocations are intentionally retained after release. This
// quarantines stale addresses so a freed handle can never become a valid handle
// for a later cache merely because the allocator reused the same address.
#[allow(clippy::vec_box)]
struct CacheRegistry {
    live: HashMap<usize, CacheEntry>,
    retired_tokens: Vec<Box<SlhaElasticKvCache>>,
}

fn cache_registry() -> &'static RwLock<CacheRegistry> {
    static REGISTRY: OnceLock<RwLock<CacheRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        RwLock::new(CacheRegistry {
            live: HashMap::new(),
            retired_tokens: Vec::new(),
        })
    })
}

fn register_cache(inner: ElasticKvCache) -> *mut SlhaElasticKvCache {
    let mut token = Box::new(SlhaElasticKvCache { _opaque: 0 });
    let pointer = (&mut *token) as *mut SlhaElasticKvCache;
    let entry = CacheEntry {
        token,
        inner: Arc::new(Mutex::new(inner)),
    };
    let previous = cache_registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .live
        .insert(pointer as usize, entry);
    debug_assert!(previous.is_none());
    pointer
}

fn cache_arc(handle: *const SlhaElasticKvCache) -> Result<Arc<Mutex<ElasticKvCache>>, FfiError> {
    if handle.is_null() {
        return Err(ffi_error(SLHA_ERR_NULL, "elastic cache handle is NULL"));
    }
    if !pointer_is_aligned(handle) {
        return Err(ffi_error(
            SLHA_ERR_INVALID_HANDLE,
            "elastic cache handle is misaligned",
        ));
    }

    cache_registry()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .live
        .get(&(handle as usize))
        .map(|entry| Arc::clone(&entry.inner))
        .ok_or_else(|| {
            ffi_error(
                SLHA_ERR_INVALID_HANDLE,
                "elastic cache handle is not a live SLHA handle",
            )
        })
}

fn unregister_cache(handle: *mut SlhaElasticKvCache) -> bool {
    let mut registry = cache_registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(entry) = registry.live.remove(&(handle as usize)) else {
        return false;
    };
    registry.retired_tokens.push(entry.token);
    true
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlhaElasticKvCacheStats {
    pub resident_bytes: usize,
    pub offloaded_bytes: usize,
    pub hard_budget_bytes: usize,
    pub hot_slots: usize,
    pub warm_slots: usize,
    pub cold_slots: usize,
    pub pinned_slots: usize,
    pub evictions: u64,
}

fn lock_cache(cache: &Arc<Mutex<ElasticKvCache>>) -> std::sync::MutexGuard<'_, ElasticKvCache> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn cache_error(context: &str, error: &'static str) -> FfiError {
    ffi_error(SLHA_ERR_DIMENSION, format!("{context}: {error}"))
}

unsafe fn read_tile_unaligned(tile: *const SciRustSlhaTile) -> Result<[u8; 128], FfiError> {
    if tile.is_null() {
        return Err(ffi_error(SLHA_ERR_NULL, "tile pointer is NULL"));
    }
    if size_of::<SciRustSlhaTile>() != 128 {
        return Err(ffi_error(
            SLHA_ERR_DIMENSION,
            "SciRustSlhaTile ABI size is not 128 bytes",
        ));
    }

    // SAFETY: caller guarantees one readable tile. read_unaligned removes any
    // Rust-side alignment requirement.
    let value = unsafe { tile.read_unaligned() };
    validate_tile(&value)?;

    let mut bytes = [0u8; 128];
    // SAFETY: `value` is a fully initialized local tile and `bytes` has exactly
    // the same ABI size. Reading its object representation as bytes is valid.
    unsafe {
        ptr::copy_nonoverlapping(
            ptr::from_ref(&value).cast::<u8>(),
            bytes.as_mut_ptr(),
            bytes.len(),
        );
    }
    Ok(bytes)
}

unsafe fn read_query(
    q_coarse: *const f32,
    q_sign: *const u64,
) -> Result<([f32; D_C], [u64; RESIDUAL_WORDS]), FfiError> {
    if q_coarse.is_null() || q_sign.is_null() {
        return Err(ffi_error(
            SLHA_ERR_NULL,
            "query coarse/sign pointer is NULL",
        ));
    }
    let mut coarse = [0.0f32; D_C];
    let mut sign = [0u64; RESIDUAL_WORDS];
    for (index, value) in coarse.iter_mut().enumerate() {
        // SAFETY: caller guarantees D_C readable f32 elements.
        *value = unsafe { q_coarse.add(index).read_unaligned() };
    }
    for (index, value) in sign.iter_mut().enumerate() {
        // SAFETY: caller guarantees RESIDUAL_WORDS readable u64 elements.
        *value = unsafe { q_sign.add(index).read_unaligned() };
    }
    if let Some(index) = coarse.iter().position(|value| !value.is_finite()) {
        return Err(ffi_error(
            crate::SLHA_ERR_NONFINITE,
            format!("q_coarse[{index}] is not finite"),
        ));
    }
    Ok((coarse, sign))
}

unsafe fn read_scores(scores: *const f32, count: usize) -> Result<Vec<f32>, FfiError> {
    if scores.is_null() {
        return Err(ffi_error(SLHA_ERR_NULL, "score pointer is NULL"));
    }
    if count > MAX_TILES {
        return Err(ffi_error(
            SLHA_ERR_DIMENSION,
            format!("score count {count} exceeds safety bound {MAX_TILES}"),
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| ffi_error(SLHA_ERR_PANIC, "score buffer allocation failed"))?;
    for index in 0..count {
        // SAFETY: caller guarantees `count` readable f32 elements.
        let value = unsafe { scores.add(index).read_unaligned() };
        if !value.is_finite() {
            return Err(ffi_error(
                crate::SLHA_ERR_NONFINITE,
                format!("score[{index}] is not finite"),
            ));
        }
        values.push(value);
    }
    Ok(values)
}

unsafe fn write_scores(scores_out: *mut f32, values: &[f32]) {
    for (index, value) in values.iter().copied().enumerate() {
        // SAFETY: caller guarantees values.len() writable f32 elements.
        unsafe { scores_out.add(index).write_unaligned(value) };
    }
}

unsafe fn write_tile(out_tile: *mut SciRustSlhaTile, bytes: &[u8; 128]) {
    // SAFETY: caller guarantees one writable tile. Byte-wise copy intentionally
    // accepts unaligned C storage.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out_tile.cast::<u8>(), bytes.len());
    }
}

/// Resident bytes charged by one full HOT tile.
#[no_mangle]
pub extern "C" fn slha_elastic_hot_resident_bytes() -> usize {
    codec::HOT_BYTES
}

/// Resident bytes charged by one WARM tile.
#[no_mangle]
pub extern "C" fn slha_elastic_warm_resident_bytes() -> usize {
    codec::WARM_BYTES
}

fn strided_slot(start_slot: usize, stride: usize, offset: usize) -> Result<usize, FfiError> {
    let delta = stride.checked_mul(offset).ok_or_else(|| {
        ffi_error(
            SLHA_ERR_DIMENSION,
            "elastic strided slot multiplication overflows usize",
        )
    })?;
    start_slot.checked_add(delta).ok_or_else(|| {
        ffi_error(
            SLHA_ERR_DIMENSION,
            "elastic strided slot addition overflows usize",
        )
    })
}

/// Create a registry-backed elastic fixed-slot KV cache.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_new(hard_budget_bytes: usize) -> *mut SlhaElasticKvCache {
    crate::clear_last_error();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        register_cache(ElasticKvCache::new(hard_budget_bytes, "slha-c-ffi"))
    })) {
        Ok(handle) => handle,
        Err(_) => {
            crate::set_last_error("panic caught while creating elastic KV cache");
            ptr::null_mut()
        }
    }
}

/// Release a live cache handle. NULL is a no-op. Foreign and already released
/// handles are rejected without dereferencing them.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_free(handle: *mut SlhaElasticKvCache) -> i32 {
    crate::clear_last_error();
    if handle.is_null() {
        return SLHA_OK;
    }
    if !pointer_is_aligned(handle) || !unregister_cache(handle) {
        crate::set_last_error("elastic cache handle is not a live SLHA handle");
        return SLHA_ERR_INVALID_HANDLE;
    }
    SLHA_OK
}

/// Write a full tile at one exact stable slot.
///
/// # Safety
/// `tile` must point to one readable `SciRustSlhaTile`. Unaligned storage is
/// accepted. `handle` may be any pointer value; only registered handles are
/// accepted and the handle itself is never dereferenced.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_write(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
    tile: *const SciRustSlhaTile,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        // SAFETY: guaranteed by this function's caller contract.
        let bytes = unsafe { read_tile_unaligned(tile) }?;
        let result = lock_cache(&cache)
            .write_at(slot, bytes)
            .map_err(|error| cache_error("elastic fixed-slot write failed", error));
        result
    })
}

/// Transactionally write a dense-attention tile under an exact resident target.
///
/// Existing HOT slots may become WARM, but the operation never creates COLD
/// slots and never commits resident accounting above `target_resident_bytes`.
///
/// # Safety
/// `tile` must point to one readable `SciRustSlhaTile`. Unaligned storage is
/// accepted. `handle` may be any pointer value; only registered handles are
/// accepted and the handle itself is never dereferenced.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_write_dense_budget(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
    tile: *const SciRustSlhaTile,
    target_resident_bytes: usize,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        // SAFETY: guaranteed by this function's caller contract.
        let bytes = unsafe { read_tile_unaligned(tile) }?;
        let result = lock_cache(&cache)
            .write_at_dense_budget(slot, bytes, target_resident_bytes)
            .map(|_| ())
            .map_err(|error| cache_error("elastic dense-budget write failed", error));
        result
    })
}

/// Clear one stable slot and all backing owned by it.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_clear_slot(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if lock_cache(&cache).clear_slot(slot) {
            Ok(())
        } else {
            Err(ffi_error(
                SLHA_ERR_NOT_RESIDENT,
                format!("elastic cache slot {slot} is absent"),
            ))
        }
    })
}

/// Clear every stable slot while retaining controller configuration.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_clear(handle: *mut SlhaElasticKvCache) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        lock_cache(&cache).clear();
        Ok(())
    })
}

/// Score a contiguous fixed-slot range without partially modifying output on
/// failure.
///
/// # Safety
/// `q_coarse` points to `D_C` readable f32 values, `q_sign` points to
/// `RESIDUAL_WORDS` readable u64 values and `scores_out` points to `count`
/// writable f32 values. Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_score_range(
    handle: *mut SlhaElasticKvCache,
    start_slot: usize,
    count: usize,
    q_coarse: *const f32,
    q_sign: *const u64,
    scores_out: *mut f32,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if count > MAX_TILES {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!("score count {count} exceeds safety bound {MAX_TILES}"),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        if scores_out.is_null() {
            return Err(ffi_error(SLHA_ERR_NULL, "score output pointer is NULL"));
        }
        // SAFETY: guaranteed by this function's caller contract.
        let (coarse, sign) = unsafe { read_query(q_coarse, q_sign) }?;
        let guard = lock_cache(&cache);
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| ffi_error(SLHA_ERR_PANIC, "score buffer allocation failed"))?;
        for offset in 0..count {
            let slot = start_slot.checked_add(offset).ok_or_else(|| {
                ffi_error(
                    SLHA_ERR_DIMENSION,
                    "elastic score slot range overflows usize",
                )
            })?;
            values.push(guard.score(slot, &coarse, &sign).ok_or_else(|| {
                ffi_error(
                    SLHA_ERR_NOT_RESIDENT,
                    format!("elastic cache slot {slot} is absent or COLD"),
                )
            })?);
        }
        drop(guard);
        // SAFETY: output is validated above and caller guarantees `count` slots.
        unsafe { write_scores(scores_out, &values) };
        Ok(())
    })
}

/// Score a fixed-slot arithmetic progression without partially modifying
/// output on failure. This lets an engine interleave layers by physical KV
/// position without materializing a dense `layers * context` backing array.
///
/// # Safety
/// `q_coarse` points to `D_C` readable f32 values, `q_sign` points to
/// `RESIDUAL_WORDS` readable u64 values and `scores_out` points to `count`
/// writable f32 values. Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_score_strided(
    handle: *mut SlhaElasticKvCache,
    start_slot: usize,
    stride: usize,
    count: usize,
    q_coarse: *const f32,
    q_sign: *const u64,
    scores_out: *mut f32,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if count > MAX_TILES {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!("score count {count} exceeds safety bound {MAX_TILES}"),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        if count > 1 && stride == 0 {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                "elastic strided score requires non-zero stride when count > 1",
            ));
        }
        if scores_out.is_null() {
            return Err(ffi_error(SLHA_ERR_NULL, "score output pointer is NULL"));
        }
        // SAFETY: guaranteed by this function's caller contract.
        let (coarse, sign) = unsafe { read_query(q_coarse, q_sign) }?;
        let guard = lock_cache(&cache);
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| ffi_error(SLHA_ERR_PANIC, "score buffer allocation failed"))?;
        for offset in 0..count {
            let slot = strided_slot(start_slot, stride, offset)?;
            values.push(guard.score(slot, &coarse, &sign).ok_or_else(|| {
                ffi_error(
                    SLHA_ERR_NOT_RESIDENT,
                    format!("elastic cache slot {slot} is absent or COLD"),
                )
            })?);
        }
        drop(guard);
        // SAFETY: output is validated above and caller guarantees `count` slots.
        unsafe { write_scores(scores_out, &values) };
        Ok(())
    })
}

/// Update slot importance from a contiguous range of attention scores.
///
/// # Safety
/// `scores` points to `count` readable f32 values. Unaligned storage is
/// accepted. `temperature` must be finite and strictly positive.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_observe_scores(
    handle: *mut SlhaElasticKvCache,
    start_slot: usize,
    scores: *const f32,
    count: usize,
    temperature: f32,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                "elastic score temperature must be finite and positive",
            ));
        }
        if count == 0 {
            return Ok(());
        }
        // SAFETY: guaranteed by this function's caller contract.
        let values = unsafe { read_scores(scores, count) }?;
        let mut observations = Vec::new();
        observations
            .try_reserve_exact(count)
            .map_err(|_| ffi_error(SLHA_ERR_PANIC, "importance buffer allocation failed"))?;
        for (offset, score) in values.into_iter().enumerate() {
            let slot = start_slot.checked_add(offset).ok_or_else(|| {
                ffi_error(
                    SLHA_ERR_DIMENSION,
                    "elastic importance slot range overflows usize",
                )
            })?;
            observations.push((slot, score));
        }
        lock_cache(&cache).observe_scores(&observations, temperature);
        Ok(())
    })
}

/// Update slot importance over a fixed-slot arithmetic progression.
///
/// # Safety
/// `scores` points to `count` readable f32 values. Unaligned storage is
/// accepted. `temperature` must be finite and strictly positive.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_observe_scores_strided(
    handle: *mut SlhaElasticKvCache,
    start_slot: usize,
    stride: usize,
    scores: *const f32,
    count: usize,
    temperature: f32,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                "elastic score temperature must be finite and positive",
            ));
        }
        if count > MAX_TILES {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                format!("score count {count} exceeds safety bound {MAX_TILES}"),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        if count > 1 && stride == 0 {
            return Err(ffi_error(
                SLHA_ERR_DIMENSION,
                "elastic strided observation requires non-zero stride when count > 1",
            ));
        }
        // SAFETY: guaranteed by this function's caller contract.
        let values = unsafe { read_scores(scores, count) }?;
        let mut observations = Vec::new();
        observations
            .try_reserve_exact(count)
            .map_err(|_| ffi_error(SLHA_ERR_PANIC, "importance buffer allocation failed"))?;
        for (offset, score) in values.into_iter().enumerate() {
            observations.push((strided_slot(start_slot, stride, offset)?, score));
        }
        lock_cache(&cache).observe_scores(&observations, temperature);
        Ok(())
    })
}

/// Transactionally demote resident slots toward a target residency.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_demote_to(
    handle: *mut SlhaElasticKvCache,
    target_resident_bytes: usize,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        let result = lock_cache(&cache)
            .demote_to(target_resident_bytes)
            .map(|_| ())
            .map_err(|error| cache_error("elastic demotion failed", error));
        result
    })
}

/// Transactionally offload resident slots toward a COLD target.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_offload_to(
    handle: *mut SlhaElasticKvCache,
    target_resident_bytes: usize,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        let result = lock_cache(&cache)
            .offload_to(target_resident_bytes)
            .map(|_| ())
            .map_err(|error| cache_error("elastic offload failed", error));
        result
    })
}

/// Restore a COLD slot to HOT.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_restore_slot(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        let result = lock_cache(&cache).restore_slot(slot).map_err(|error| {
            ffi_error(
                SLHA_ERR_NOT_RESIDENT,
                format!("elastic restore failed for slot {slot}: {error}"),
            )
        });
        result
    })
}

/// Promote a WARM slot losslessly back to HOT.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_promote_slot(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        let result = lock_cache(&cache).promote_slot(slot).map_err(|error| {
            ffi_error(
                SLHA_ERR_NOT_RESIDENT,
                format!("elastic promotion failed for slot {slot}: {error}"),
            )
        });
        result
    })
}

/// Return the physical tier of a live slot, or ABSENT for a missing/invalid
/// handle or slot. Use `slha_elastic_cache_stats` when an explicit handle error
/// code is required.
#[no_mangle]
pub extern "C" fn slha_elastic_cache_tier(handle: *mut SlhaElasticKvCache, slot: usize) -> i32 {
    let Ok(cache) = cache_arc(handle) else {
        return SLHA_ELASTIC_TIER_ABSENT;
    };
    let tier = lock_cache(&cache).tier(slot);
    match tier {
        Some(PhysicalTier::Hot) => SLHA_ELASTIC_TIER_HOT,
        Some(PhysicalTier::Warm) => SLHA_ELASTIC_TIER_WARM,
        Some(PhysicalTier::Cold) => SLHA_ELASTIC_TIER_COLD,
        Some(PhysicalTier::Pinned) => SLHA_ELASTIC_TIER_PINNED,
        None => SLHA_ELASTIC_TIER_ABSENT,
    }
}

/// Copy a currently scoreable HOT/WARM/PINNED representation to C storage.
///
/// # Safety
/// `out_tile` must point to writable storage for one `SciRustSlhaTile`.
/// Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_resident_tile(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
    out_tile: *mut SciRustSlhaTile,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if out_tile.is_null() {
            return Err(ffi_error(SLHA_ERR_NULL, "tile output pointer is NULL"));
        }
        let bytes = lock_cache(&cache).resident_tile(slot).ok_or_else(|| {
            ffi_error(
                SLHA_ERR_NOT_RESIDENT,
                format!("elastic cache slot {slot} is absent or COLD"),
            )
        })?;
        // SAFETY: guaranteed by this function's caller contract.
        unsafe { write_tile(out_tile, &bytes) };
        Ok(())
    })
}

/// Read cache residency/accounting statistics.
///
/// # Safety
/// `out` must point to writable storage for one `SlhaElasticKvCacheStats`.
/// Unaligned storage is accepted.
#[no_mangle]
pub unsafe extern "C" fn slha_elastic_cache_stats(
    handle: *mut SlhaElasticKvCache,
    out: *mut SlhaElasticKvCacheStats,
) -> i32 {
    ffi_status(|| {
        let cache = cache_arc(handle)?;
        if out.is_null() {
            return Err(ffi_error(SLHA_ERR_NULL, "stats output pointer is NULL"));
        }
        let guard = lock_cache(&cache);
        let (hot, warm, cold, pinned) = guard.counts();
        let stats = SlhaElasticKvCacheStats {
            resident_bytes: guard.resident_bytes(),
            offloaded_bytes: guard.offloaded_bytes(),
            hard_budget_bytes: guard.hard_budget_bytes(),
            hot_slots: hot,
            warm_slots: warm,
            cold_slots: cold,
            pinned_slots: pinned,
            evictions: guard.evictions(),
        };
        drop(guard);
        // SAFETY: guaranteed by this function's caller contract.
        unsafe { out.write_unaligned(stats) };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile() -> SciRustSlhaTile {
        SciRustSlhaTile {
            latent_kv: [0; 64],
            residual_bitmap: [0; 4],
            scale: 1.0,
            dynamic_lambda: 0.25,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: 0,
            group_scales: [255; 8],
        }
    }

    fn write(handle: *mut SlhaElasticKvCache, slot: usize, tile: &SciRustSlhaTile) -> i32 {
        // SAFETY: test passes one live local tile and a handle obtained from new.
        unsafe { slha_elastic_cache_write(handle, slot, tile) }
    }

    fn score(
        handle: *mut SlhaElasticKvCache,
        start_slot: usize,
        count: usize,
        coarse: &[f32; D_C],
        sign: &[u64; RESIDUAL_WORDS],
        out: &mut [f32],
    ) -> i32 {
        // SAFETY: all buffers are local and sized according to the ABI contract.
        unsafe {
            slha_elastic_cache_score_range(
                handle,
                start_slot,
                count,
                coarse.as_ptr(),
                sign.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    fn stats(handle: *mut SlhaElasticKvCache, out: &mut SlhaElasticKvCacheStats) -> i32 {
        // SAFETY: `out` is one live local stats record.
        unsafe { slha_elastic_cache_stats(handle, out) }
    }

    #[test]
    fn ffi_exposes_hot_warm_cold_without_partial_score_writes() {
        let handle = slha_elastic_cache_new(128);
        assert!(!handle.is_null());
        let input = tile();
        assert_eq!(write(handle, 0, &input), SLHA_OK);

        let q_coarse = [0.0f32; D_C];
        let q_sign = [0u64; RESIDUAL_WORDS];
        let mut output = [-1.0f32];
        assert_eq!(
            score(handle, 0, 1, &q_coarse, &q_sign, &mut output),
            SLHA_OK
        );
        assert_eq!(output[0], 64.0);

        assert_eq!(slha_elastic_cache_demote_to(handle, 96), SLHA_OK);
        assert_eq!(slha_elastic_cache_tier(handle, 0), SLHA_ELASTIC_TIER_WARM);
        assert_eq!(
            score(handle, 0, 1, &q_coarse, &q_sign, &mut output),
            SLHA_OK
        );
        assert_eq!(output[0], 0.0);

        assert_eq!(slha_elastic_cache_offload_to(handle, 0), SLHA_OK);
        assert_eq!(slha_elastic_cache_tier(handle, 0), SLHA_ELASTIC_TIER_COLD);
        output[0] = 123.0;
        assert_eq!(
            score(handle, 0, 1, &q_coarse, &q_sign, &mut output),
            SLHA_ERR_NOT_RESIDENT
        );
        assert_eq!(output[0], 123.0);
        assert_eq!(slha_elastic_cache_restore_slot(handle, 0), SLHA_OK);
        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);
    }

    #[test]
    fn resident_byte_constants_match_physical_codec() {
        assert_eq!(slha_elastic_hot_resident_bytes(), codec::HOT_BYTES);
        assert_eq!(slha_elastic_warm_resident_bytes(), codec::WARM_BYTES);
        assert_eq!(slha_elastic_hot_resident_bytes(), 128);
        assert_eq!(slha_elastic_warm_resident_bytes(), 96);
    }

    #[test]
    fn ffi_dense_budget_write_creates_warm_without_cold_and_rolls_back() {
        let handle = slha_elastic_cache_new(192);
        assert!(!handle.is_null());
        let input = tile();

        assert_eq!(
            // SAFETY: test passes one live local tile and a registered handle.
            unsafe { slha_elastic_cache_write_dense_budget(handle, 0, &input, 96) },
            SLHA_OK
        );
        let mut snapshot = SlhaElasticKvCacheStats::default();
        assert_eq!(stats(handle, &mut snapshot), SLHA_OK);
        assert_eq!(snapshot.resident_bytes, 96);
        assert_eq!(snapshot.offloaded_bytes, 32);
        assert_eq!(snapshot.hot_slots, 0);
        assert_eq!(snapshot.warm_slots, 1);
        assert_eq!(snapshot.cold_slots, 0);

        assert_eq!(
            // SAFETY: the same tile remains readable for the second call.
            unsafe { slha_elastic_cache_write_dense_budget(handle, 1, &input, 96) },
            SLHA_ERR_DIMENSION
        );
        let mut after = SlhaElasticKvCacheStats::default();
        assert_eq!(stats(handle, &mut after), SLHA_OK);
        assert_eq!(after, snapshot);
        assert_eq!(slha_elastic_cache_tier(handle, 1), SLHA_ELASTIC_TIER_ABSENT);
        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);
    }

    #[test]
    fn strided_scoring_addresses_one_interleaved_layer_transactionally() {
        let handle = slha_elastic_cache_new(4096);
        assert!(!handle.is_null());
        let mut first = tile();
        first.latent_kv.fill(0x88);
        let mut gap = tile();
        gap.latent_kv.fill(0x99);
        let mut second = tile();
        second.latent_kv.fill(0x88);
        assert_eq!(write(handle, 1, &first), SLHA_OK);
        assert_eq!(write(handle, 2, &gap), SLHA_OK);
        assert_eq!(write(handle, 4, &second), SLHA_OK);

        let q_coarse = [1.0f32; D_C];
        let q_sign = [0u64; RESIDUAL_WORDS];
        let mut output = [-7.0f32; 2];
        // SAFETY: all buffers are local and sized according to the ABI contract.
        let rc = unsafe {
            slha_elastic_cache_score_strided(
                handle,
                1,
                3,
                2,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        assert_eq!(rc, SLHA_OK);
        assert_eq!(output[0], output[1]);

        let observations = [4.0f32, 1.0f32];
        assert_eq!(
            // SAFETY: the score buffer is local and contains exactly two elements.
            unsafe {
                slha_elastic_cache_observe_scores_strided(
                    handle,
                    1,
                    3,
                    observations.as_ptr(),
                    observations.len(),
                    1.0,
                )
            },
            SLHA_OK
        );

        output.fill(123.0);
        assert_eq!(slha_elastic_cache_clear_slot(handle, 4), SLHA_OK);
        assert_eq!(
            // SAFETY: all buffers remain valid; the missing second slot must
            // leave the complete output unchanged.
            unsafe {
                slha_elastic_cache_score_strided(
                    handle,
                    1,
                    3,
                    2,
                    q_coarse.as_ptr(),
                    q_sign.as_ptr(),
                    output.as_mut_ptr(),
                )
            },
            SLHA_ERR_NOT_RESIDENT
        );
        assert_eq!(output, [123.0, 123.0]);
        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);
    }

    #[test]
    fn forged_and_released_handles_are_rejected_without_dereference() {
        let forged = 0x1000usize as *mut SlhaElasticKvCache;
        let mut snapshot = SlhaElasticKvCacheStats::default();
        assert_eq!(stats(forged, &mut snapshot), SLHA_ERR_INVALID_HANDLE);
        assert_eq!(slha_elastic_cache_free(forged), SLHA_ERR_INVALID_HANDLE);

        let handle = slha_elastic_cache_new(128);
        assert!(!handle.is_null());
        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);
        assert_eq!(slha_elastic_cache_free(handle), SLHA_ERR_INVALID_HANDLE);
        assert_eq!(stats(handle, &mut snapshot), SLHA_ERR_INVALID_HANDLE);
    }

    #[test]
    fn fixed_slot_rewrite_does_not_accumulate_backing_through_ffi() {
        let handle = slha_elastic_cache_new(1024);
        assert!(!handle.is_null());
        let input = tile();
        assert_eq!(write(handle, 7, &input), SLHA_OK);
        assert_eq!(slha_elastic_cache_offload_to(handle, 0), SLHA_OK);
        assert_eq!(write(handle, 7, &input), SLHA_OK);

        let mut snapshot = SlhaElasticKvCacheStats::default();
        assert_eq!(stats(handle, &mut snapshot), SLHA_OK);
        assert_eq!(snapshot.resident_bytes, 128);
        assert_eq!(snapshot.offloaded_bytes, 0);
        assert_eq!(snapshot.hot_slots, 1);
        assert_eq!(snapshot.cold_slots, 0);
        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);
    }
}
