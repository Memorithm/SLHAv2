from pathlib import Path
import textwrap


def insert_once(text: str, anchor: str, addition: str, label: str) -> str:
    if anchor not in text:
        raise RuntimeError(f"missing anchor: {label}")
    return text.replace(anchor, addition + anchor, 1)


cache_path = Path("slhav2-vram/src/elastic_cache.rs")
cache = cache_path.read_text()
method_anchor = "    /// Protect a resident slot from adaptation. WARM is promoted losslessly\n"
method_body = textwrap.dedent(
    r'''
    /// Write a full tile at an exact stable slot id.
    ///
    /// This is the runtime-facing counterpart of [`Self::insert`]: KV engines
    /// recycle physical positions, so replacing a position must not append a
    /// second backing entry. Existing HOT/WARM/COLD backing for the slot is
    /// dropped atomically and the new value becomes HOT (or remains PINNED if
    /// the position itself was pinned). Importance is reset because the slot
    /// now represents a different logical key.
    pub fn write_at(
        &mut self,
        slot: usize,
        mut tile: [u8; codec::TILE_BYTES],
    ) -> Result<(), &'static str> {
        Self::set_warm_flag(&mut tile, false);
        let required_len = slot.checked_add(1).ok_or("slot index overflow")?;
        if self.slots.len() < required_len {
            self.slots
                .try_reserve(required_len - self.slots.len())
                .map_err(|_| "slot allocation failed")?;
            self.slots.resize_with(required_len, || None);
        }

        let old_resident = self.slots[slot]
            .as_ref()
            .map_or(0, PhysicalSlot::allocated_bytes);
        let pinned = self.slots[slot]
            .as_ref()
            .is_some_and(|state| state.tier == PhysicalTier::Pinned);
        let next_resident = self
            .resident_bytes
            .checked_sub(old_resident)
            .and_then(|bytes| bytes.checked_add(codec::TILE_BYTES))
            .ok_or("resident byte accounting overflow")?;

        self.slots[slot] = Some(PhysicalSlot {
            bytes: tile.to_vec(),
            offloaded: None,
            warm_residual: None,
            tier: if pinned {
                PhysicalTier::Pinned
            } else {
                PhysicalTier::Hot
            },
            seq: self.next_seq,
            importance: 0.0,
        });
        self.next_seq = self.next_seq.saturating_add(1);
        self.resident_bytes = next_resident;
        self.verify_accounting()
    }

    /// Remove one stable slot and all resident/backing bytes it owns.
    /// Returns false when the slot was already absent.
    pub fn clear_slot(&mut self, slot: usize) -> bool {
        let Some(entry) = self.slots.get_mut(slot) else {
            return false;
        };
        let Some(state) = entry.take() else {
            return false;
        };
        self.resident_bytes = self.resident_bytes.saturating_sub(state.allocated_bytes());
        debug_assert!(self.verify_accounting().is_ok());
        true
    }

    /// Remove every logical slot while retaining controller configuration.
    /// Sequence numbering remains monotonic so controller observations never
    /// move backwards across a runtime cache clear.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.resident_bytes = 0;
        self.last_reason = None;
        debug_assert!(self.verify_accounting().is_ok());
    }

    /// Current physical tier of a stable slot.
    pub fn tier(&self, slot: usize) -> Option<PhysicalTier> {
        self.slots.get(slot)?.as_ref().map(|state| state.tier)
    }

    /// Copy the representation that is currently resident and scoreable.
    ///
    /// HOT/PINNED return the full tile. WARM expands its 96 resident bytes
    /// into a canonical 128-byte tile with a zero residual plane and the WARM
    /// flag preserved, exactly matching [`Self::score`] semantics. COLD/absent
    /// slots return `None`; this function never restores them.
    pub fn resident_tile(&self, slot: usize) -> Option<[u8; codec::TILE_BYTES]> {
        let state = self.slots.get(slot)?.as_ref()?;
        match state.tier {
            PhysicalTier::Hot | PhysicalTier::Pinned => state.bytes.as_slice().try_into().ok(),
            PhysicalTier::Warm => {
                let packed: &[u8; WARM_PACKED_BYTES] =
                    state.bytes.as_slice().try_into().ok()?;
                Some(codec::unpack_warm(packed))
            }
            PhysicalTier::Cold => None,
        }
    }

    /// Transactionally reduce residency using the HOT→WARM→COLD demotion
    /// policy. Callers that must keep all active keys scoreable should set a
    /// target no lower than 96 bytes per active slot.
    pub fn demote_to(&mut self, target_resident_bytes: usize) -> Result<bool, &'static str> {
        self.apply_action(ActionRequest::Demote, target_resident_bytes)
    }

    /// Transactionally offload resident slots toward a COLD target.
    pub fn offload_to(&mut self, target_resident_bytes: usize) -> Result<bool, &'static str> {
        self.apply_action(ActionRequest::Offload, target_resident_bytes)
    }

    '''
).lstrip("\n")
cache = insert_once(cache, method_anchor, textwrap.indent(method_body, "    "), "elastic methods")

test_anchor = "    #[test]\n    fn pinned_slots_are_never_demoted_or_offloaded() {\n"
test_body = textwrap.dedent(
    r'''
    #[test]
    fn fixed_slot_rewrite_is_bounded_and_drops_old_backing() {
        let mut cache = ElasticKvCache::new(1024, "test");
        cache.write_at(7, make_tile(1)).unwrap();
        assert_eq!(cache.counts(), (1, 0, 0, 0));
        assert_eq!(cache.resident_bytes(), codec::TILE_BYTES);

        cache.evict_slot(7).unwrap();
        assert_eq!(cache.counts(), (0, 0, 1, 0));
        assert_eq!(cache.resident_bytes(), 0);
        assert_eq!(cache.offloaded_bytes(), codec::TILE_BYTES);

        cache.write_at(7, make_tile(2)).unwrap();
        assert_eq!(cache.counts(), (1, 0, 0, 0));
        assert_eq!(cache.resident_bytes(), codec::TILE_BYTES);
        assert_eq!(cache.offloaded_bytes(), 0);
        assert_eq!(cache.slots.iter().flatten().count(), 1);
    }

    #[test]
    fn resident_tile_matches_warm_scoring_representation() {
        let mut cache = ElasticKvCache::new(1024, "test");
        cache.write_at(2, make_tile(5)).unwrap();
        cache.demote_slot(2).unwrap();
        let tile = cache.resident_tile(2).unwrap();
        assert!(
            u16::from_le_bytes([tile[codec::FLAGS_OFFSET], tile[codec::FLAGS_OFFSET + 1]])
                & codec::FLAG_WARM
                != 0
        );
        assert!(tile[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES]
            .iter()
            .all(|&byte| byte == 0));
        assert_eq!(cache.tier(2), Some(PhysicalTier::Warm));
    }

    #[test]
    fn clear_slot_releases_resident_and_backing_bytes() {
        let mut cache = ElasticKvCache::new(1024, "test");
        cache.write_at(3, make_tile(9)).unwrap();
        cache.demote_slot(3).unwrap();
        assert!(cache.clear_slot(3));
        assert!(!cache.clear_slot(3));
        assert_eq!(cache.resident_bytes(), 0);
        assert_eq!(cache.offloaded_bytes(), 0);
        assert_eq!(cache.tier(3), None);
    }

    #[test]
    fn explicit_warm_target_never_requires_cold() {
        let mut cache = ElasticKvCache::new(1024, "test");
        for slot in 0..4 {
            cache.write_at(slot, make_tile(slot as u8)).unwrap();
        }
        let target = 4 * WARM_PACKED_BYTES;
        assert!(cache.demote_to(target).unwrap());
        assert_eq!(cache.resident_bytes(), target);
        assert_eq!(cache.counts(), (0, 4, 0, 0));
    }

    '''
).lstrip("\n")
cache = insert_once(cache, test_anchor, textwrap.indent(test_body, "    "), "elastic tests")
cache_path.write_text(cache)

cargo_path = Path("slha-c/Cargo.toml")
cargo = cargo_path.read_text()
dep_anchor = 'scirust = { version = "0.2", path = "../scirust" }\n'
if dep_anchor not in cargo:
    raise RuntimeError("missing slha-c dependency anchor")
cargo = cargo.replace(
    dep_anchor,
    dep_anchor + 'slhav2-vram = { version = "0.1.0", path = "../slhav2-vram" }\n',
    1,
)
cargo_path.write_text(cargo)

lib_path = Path("slha-c/src/lib.rs")
lib = lib_path.read_text()
lib_anchor = "#![deny(clippy::undocumented_unsafe_blocks)]\n"
if lib_anchor not in lib:
    raise RuntimeError("missing slha-c crate attribute anchor")
lib = lib.replace(lib_anchor, lib_anchor + "\nmod elastic_ffi;\npub use elastic_ffi::*;\n", 1)
lib_path.write_text(lib)

ffi = r'''use scirust::attention::slha_v2::{SciRustSlhaTile, D_C, RESIDUAL_WORDS};
use slhav2_vram::elastic_cache::{ElasticKvCache, PhysicalTier};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use crate::{
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
        // SAFETY: ownership of a pointer returned by `slha_elastic_cache_new`
        // is transferred back exactly once.
        drop(unsafe { Box::from_raw(handle) });
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
            // SAFETY: caller provides `count` writable float slots.
            unsafe { ptr::write_unaligned(scores_out.add(index), value) };
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
            // SAFETY: caller provides `count` readable float scores.
            let score = unsafe { ptr::read_unaligned(scores.add(offset)) };
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
pub extern "C" fn slha_elastic_cache_tier(
    handle: *mut SlhaElasticKvCache,
    slot: usize,
) -> i32 {
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
        // SAFETY: caller provides one writable tile. Byte-wise copy accepts
        // unaligned output storage.
        unsafe {
            ptr::copy_nonoverlapping(tile.as_ptr(), out_tile.cast::<u8>(), tile.len());
        }
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
        // SAFETY: caller provides writable storage for one stats value; unaligned
        // output is explicitly accepted.
        unsafe { ptr::write_unaligned(out, stats) };
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
'''
Path("slha-c/src/elastic_ffi.rs").write_text(ffi)

header_path = Path("slha-c/include/slha.h")
header = header_path.read_text()
status_anchor = "#define SLHA_ERR_UTF8 (-9)\n"
if status_anchor not in header:
    raise RuntimeError("missing header status anchor")
header = header.replace(status_anchor, status_anchor + "#define SLHA_ERR_NOT_RESIDENT (-10)\n", 1)

type_anchor = "typedef struct SlhaContext SlhaContext;\ntypedef struct SlhaModel SlhaModel;\n"
if type_anchor not in header:
    raise RuntimeError("missing header opaque-type anchor")
type_block = '''typedef struct SlhaContext SlhaContext;
typedef struct SlhaModel SlhaModel;
typedef struct SlhaElasticKvCache SlhaElasticKvCache;

#define SLHA_ELASTIC_TIER_ABSENT (-1)
#define SLHA_ELASTIC_TIER_HOT 0
#define SLHA_ELASTIC_TIER_WARM 1
#define SLHA_ELASTIC_TIER_COLD 2
#define SLHA_ELASTIC_TIER_PINNED 3

typedef struct {
    size_t resident_bytes;
    size_t offloaded_bytes;
    size_t hard_budget_bytes;
    size_t hot_slots;
    size_t warm_slots;
    size_t cold_slots;
    size_t pinned_slots;
    uint64_t evictions;
} SlhaElasticKvCacheStats;
'''
header = header.replace(type_anchor, type_block, 1)

api_anchor = "/* ABI and layout introspection. */\n"
if api_anchor not in header:
    raise RuntimeError("missing header API anchor")
api = '''/* Elastic fixed-slot KV cache. Mutating/scoring calls are internally synchronized. */
SlhaElasticKvCache* slha_elastic_cache_new(size_t hard_budget_bytes);
int32_t slha_elastic_cache_free(SlhaElasticKvCache* cache);
int32_t slha_elastic_cache_write(SlhaElasticKvCache* cache, size_t slot, const SciRustSlhaTile* tile);
int32_t slha_elastic_cache_clear_slot(SlhaElasticKvCache* cache, size_t slot);
int32_t slha_elastic_cache_clear(SlhaElasticKvCache* cache);
int32_t slha_elastic_cache_score_range(
    SlhaElasticKvCache* cache,
    size_t start_slot,
    size_t count,
    const float* q_coarse,
    const uint64_t* q_sign,
    float* scores_out
);
int32_t slha_elastic_cache_observe_scores(
    SlhaElasticKvCache* cache,
    size_t start_slot,
    const float* scores,
    size_t count,
    float temperature
);
int32_t slha_elastic_cache_demote_to(SlhaElasticKvCache* cache, size_t target_resident_bytes);
int32_t slha_elastic_cache_offload_to(SlhaElasticKvCache* cache, size_t target_resident_bytes);
int32_t slha_elastic_cache_restore_slot(SlhaElasticKvCache* cache, size_t slot);
int32_t slha_elastic_cache_promote_slot(SlhaElasticKvCache* cache, size_t slot);
int32_t slha_elastic_cache_tier(SlhaElasticKvCache* cache, size_t slot);
int32_t slha_elastic_cache_resident_tile(
    SlhaElasticKvCache* cache,
    size_t slot,
    SciRustSlhaTile* out_tile
);
int32_t slha_elastic_cache_stats(SlhaElasticKvCache* cache, SlhaElasticKvCacheStats* out);

'''
header = header.replace(api_anchor, api + api_anchor, 1)
header_path.write_text(header)
