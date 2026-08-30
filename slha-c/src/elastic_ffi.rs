use scirust::attention::slha_v2::{SciRustSlhaTile, D_C, RESIDUAL_WORDS};
use slhav2_vram::elastic_cache::{ElasticKvCache, PhysicalTier};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use crate::{SLHA_ERR_DIMENSION, SLHA_ERR_INVALID_HANDLE, SLHA_ERR_NULL, SLHA_ERR_PANIC, SLHA_OK};

/// A requested slot exists but is physically COLD and therefore cannot
/// participate in dense attention until the caller explicitly restores it.
pub const SLHA_ERR_NOT_RESIDENT: i32 = -10;

pub const SLHA_ELASTIC_TIER_HOT: i32 = 0;
pub const SLHA_ELASTIC_TIER_WARM: i32 = 1;
pub const SLHA_ELASTIC_TIER_COLD: i32 = 2;
pub const SLHA_ELASTIC_TIER_PINNED: i32 = 3;
pub const SLHA_ELASTIC_TIER_ABSENT: i32 = -1;

/// Opaque, internally synchronized elastic KV cache handle.
pub struct SlhaElasticKvCache {
    inner: Mutex<ElasticKvCache>,
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

fn status<F>(f: F) -> i32
where
    F: FnOnce() -> Result<(), i32>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SLHA_OK,
        Ok(Err(code)) => code,
        Err(_) => SLHA_ERR_PANIC,
    }
}

fn cache<'a>(handle: *mut SlhaElasticKvCache) -> Result<&'a SlhaElasticKvCache, i32> {
    if handle.is_null() {
        return Err(SLHA_ERR_INVALID_HANDLE);
    }
    // SAFETY: handles are created by `slha_elastic_cache_new` and remain valid
    // until the caller passes the same pointer exactly once to
    // `slha_elastic_cache_free`. The internal Mutex synchronizes worker threads.
    Ok(unsafe { &*handle })
}

fn drop_cache_handle(handle: *mut SlhaElasticKvCache) {
    // SAFETY: ownership of a pointer returned by `slha_elastic_cache_new` is
    // transferred back exactly once by the public free function.
    drop(unsafe { Box::from_raw(handle) });
}

fn read_f32_at(base: *const f32, index: usize) -> f32 {
    // SAFETY: the caller-facing function validates the contract that `base`
    // contains the requested readable range; unaligned C storage is accepted.
    unsafe { ptr::read_unaligned(base.add(index)) }
}

fn write_f32_at(base: *mut f32, index: usize, value: f32) {
    // SAFETY: the caller-facing function validates the contract that `base`
    // contains the requested writable range; unaligned C storage is accepted.
    unsafe { ptr::write_unaligned(base.add(index), value) };
}

fn copy_tile_out(bytes: &[u8; 128], out: *mut SciRustSlhaTile) {
    // SAFETY: the caller-facing function requires one writable ABI tile.
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), bytes.len()) };
}

fn write_stats_out(out: *mut SlhaElasticKvCacheStats, stats: SlhaElasticKvCacheStats) {
    // SAFETY: the caller-facing function requires writable storage for one
    // stats record; write_unaligned preserves the C ABI alignment contract.
    unsafe { ptr::write_unaligned(out, stats) };
}

fn tile_bytes(tile: *const SciRustSlhaTile) -> Result<[u8; 128], i32> {
    if tile.is_null() {
        return Err(SLHA_ERR_NULL);
    }
    if std::mem::size_of::<SciRustSlhaTile>() != 128 {
        return Err(SLHA_ERR_DIMENSION);
    }
    let mut out = [0u8; 128];
    // SAFETY: `tile` points to one readable C ABI tile. Byte-wise copying
    // intentionally accepts unaligned external storage.
    unsafe {
        ptr::copy_nonoverlapping(tile.cast::<u8>(), out.as_mut_ptr(), out.len());
    }
    Ok(out)
}

fn read_query(
    q_coarse: *const f32,
    q_sign: *const u64,
) -> Result<([f32; D_C], [u64; RESIDUAL_WORDS]), i32> {
    if q_coarse.is_null() || q_sign.is_null() {
        return Err(SLHA_ERR_NULL);
    }
    let mut coarse = [0.0f32; D_C];
    let mut sign = [0u64; RESIDUAL_WORDS];
    for (i, value) in coarse.iter_mut().enumerate() {
        // SAFETY: caller provides D_C readable floats; read_unaligned preserves
        // the established C ABI unaligned-input contract.
        *value = unsafe { ptr::read_unaligned(q_coarse.add(i)) };
    }
    for (i, value) in sign.iter_mut().enumerate() {
        // SAFETY: caller provides RESIDUAL_WORDS readable u64 values.
        *value = unsafe { ptr::read_unaligned(q_sign.add(i)) };
    }
    Ok((coarse, sign))
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_new(hard_budget_bytes: usize) -> *mut SlhaElasticKvCache {
    match catch_unwind(AssertUnwindSafe(|| SlhaElasticKvCache {
        inner: Mutex::new(ElasticKvCache::new(hard_budget_bytes, "slha-c-ffi")),
    })) {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_free(handle: *mut SlhaElasticKvCache) -> i32 {
    if handle.is_null() {
        return SLHA_OK;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        drop_cache_handle(handle);
    })) {
        Ok(()) => SLHA_OK,
        Err(_) => SLHA_ERR_PANIC,
    }
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_write(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
    tile: *const SciRustSlhaTile,
) -> i32 {
    status(|| {
        let bytes = tile_bytes(tile)?;
        let handle = cache(handle)?;
        handle
            .inner
            .lock()
            .map_err(|_| SLHA_ERR_PANIC)?
            .write_at(slot, bytes)
            .map_err(|_| SLHA_ERR_DIMENSION)
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_clear_slot(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
    status(|| {
        let handle = cache(handle)?;
        let mut guard = handle.inner.lock().map_err(|_| SLHA_ERR_PANIC)?;
        if guard.clear_slot(slot) {
            Ok(())
        } else {
            Err(SLHA_ERR_NOT_RESIDENT)
        }
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_clear(handle: *mut SlhaElasticKvCache) -> i32 {
    status(|| {
        let handle = cache(handle)?;
        handle.inner.lock().map_err(|_| SLHA_ERR_PANIC)?.clear();
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_score_range(
    handle: *mut SlhaElasticKvCache,
    start_slot: usize,
    count: usize,
    q_coarse: *const f32,
    q_sign: *const u64,
    scores_out: *mut f32,
) -> i32 {
    if count == 0 {
        return SLHA_OK;
    }
    status(|| {
        if scores_out.is_null() {
            return Err(SLHA_ERR_NULL);
        }
        let (coarse, sign) = read_query(q_coarse, q_sign)?;
        let handle = cache(handle)?;
        let guard = handle.inner.lock().map_err(|_| SLHA_ERR_PANIC)?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(count)
            .map_err(|_| SLHA_ERR_PANIC)?;
        for offset in 0..count {
            let slot = start_slot.checked_add(offset).ok_or(SLHA_ERR_DIMENSION)?;
            scores.push(
                guard
                    .score(slot, &coarse, &sign)
                    .ok_or(SLHA_ERR_NOT_RESIDENT)?,
            );
        }
        drop(guard);
        for (index, value) in scores.into_iter().enumerate() {
            write_f32_at(scores_out, index, value);
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_observe_scores(
    handle: *mut SlhaElasticKvCache,
    start_slot: usize,
    scores: *const f32,
    count: usize,
    temperature: f32,
) -> i32 {
    if count == 0 {
        return SLHA_OK;
    }
    status(|| {
        if scores.is_null() || !temperature.is_finite() || temperature <= 0.0 {
            return Err(SLHA_ERR_DIMENSION);
        }
        let mut observations = Vec::new();
        observations
            .try_reserve_exact(count)
            .map_err(|_| SLHA_ERR_PANIC)?;
        for offset in 0..count {
            let slot = start_slot.checked_add(offset).ok_or(SLHA_ERR_DIMENSION)?;
            let score = read_f32_at(scores, offset);
            if !score.is_finite() {
                return Err(SLHA_ERR_DIMENSION);
            }
            observations.push((slot, score));
        }
        let handle = cache(handle)?;
        handle
            .inner
            .lock()
            .map_err(|_| SLHA_ERR_PANIC)?
            .observe_scores(&observations, temperature);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_demote_to(
    handle: *mut SlhaElasticKvCache,
    target_resident_bytes: usize,
) -> i32 {
    status(|| {
        let handle = cache(handle)?;
        handle
            .inner
            .lock()
            .map_err(|_| SLHA_ERR_PANIC)?
            .demote_to(target_resident_bytes)
            .map(|_| ())
            .map_err(|_| SLHA_ERR_DIMENSION)
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_offload_to(
    handle: *mut SlhaElasticKvCache,
    target_resident_bytes: usize,
) -> i32 {
    status(|| {
        let handle = cache(handle)?;
        handle
            .inner
            .lock()
            .map_err(|_| SLHA_ERR_PANIC)?
            .offload_to(target_resident_bytes)
            .map(|_| ())
            .map_err(|_| SLHA_ERR_DIMENSION)
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_restore_slot(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
    status(|| {
        let handle = cache(handle)?;
        handle
            .inner
            .lock()
            .map_err(|_| SLHA_ERR_PANIC)?
            .restore_slot(slot)
            .map_err(|_| SLHA_ERR_NOT_RESIDENT)
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_promote_slot(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
    status(|| {
        let handle = cache(handle)?;
        handle
            .inner
            .lock()
            .map_err(|_| SLHA_ERR_PANIC)?
            .promote_slot(slot)
            .map_err(|_| SLHA_ERR_NOT_RESIDENT)
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_tier(handle: *mut SlhaElasticKvCache, slot: usize) -> i32 {
    let Ok(handle) = cache(handle) else {
        return SLHA_ELASTIC_TIER_ABSENT;
    };
    let Ok(guard) = handle.inner.lock() else {
        return SLHA_ELASTIC_TIER_ABSENT;
    };
    match guard.tier(slot) {
        Some(PhysicalTier::Hot) => SLHA_ELASTIC_TIER_HOT,
        Some(PhysicalTier::Warm) => SLHA_ELASTIC_TIER_WARM,
        Some(PhysicalTier::Cold) => SLHA_ELASTIC_TIER_COLD,
        Some(PhysicalTier::Pinned) => SLHA_ELASTIC_TIER_PINNED,
        None => SLHA_ELASTIC_TIER_ABSENT,
    }
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_resident_tile(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
    out_tile: *mut SciRustSlhaTile,
) -> i32 {
    status(|| {
        if out_tile.is_null() {
            return Err(SLHA_ERR_NULL);
        }
        let handle = cache(handle)?;
        let guard = handle.inner.lock().map_err(|_| SLHA_ERR_PANIC)?;
        let tile = guard.resident_tile(slot).ok_or(SLHA_ERR_NOT_RESIDENT)?;
        drop(guard);
        copy_tile_out(&tile, out_tile);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn slha_elastic_cache_stats(
    handle: *mut SlhaElasticKvCache,
    out: *mut SlhaElasticKvCacheStats,
) -> i32 {
    status(|| {
        if out.is_null() {
            return Err(SLHA_ERR_NULL);
        }
        let handle = cache(handle)?;
        let guard = handle.inner.lock().map_err(|_| SLHA_ERR_PANIC)?;
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
        write_stats_out(out, stats);
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

    #[test]
    fn ffi_exposes_hot_warm_cold_without_partial_score_writes() {
        let handle = slha_elastic_cache_new(128);
        assert!(!handle.is_null());
        let input = tile();
        assert_eq!(slha_elastic_cache_write(handle, 0, &input), SLHA_OK);

        let q_coarse = [0.0f32; D_C];
        let q_sign = [0u64; RESIDUAL_WORDS];
        let mut score = -1.0f32;
        assert_eq!(
            slha_elastic_cache_score_range(
                handle,
                0,
                1,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut score,
            ),
            SLHA_OK
        );
        assert_eq!(score, 64.0);

        assert_eq!(slha_elastic_cache_demote_to(handle, 96), SLHA_OK);
        assert_eq!(slha_elastic_cache_tier(handle, 0), SLHA_ELASTIC_TIER_WARM);
        assert_eq!(
            slha_elastic_cache_score_range(
                handle,
                0,
                1,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut score,
            ),
            SLHA_OK
        );
        assert_eq!(score, 0.0);

        assert_eq!(slha_elastic_cache_offload_to(handle, 0), SLHA_OK);
        assert_eq!(slha_elastic_cache_tier(handle, 0), SLHA_ELASTIC_TIER_COLD);
        score = 123.0;
        assert_eq!(
            slha_elastic_cache_score_range(
                handle,
                0,
                1,
                q_coarse.as_ptr(),
                q_sign.as_ptr(),
                &mut score,
            ),
            SLHA_ERR_NOT_RESIDENT
        );
        assert_eq!(score, 123.0);
        assert_eq!(slha_elastic_cache_restore_slot(handle, 0), SLHA_OK);
        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);
    }
}
