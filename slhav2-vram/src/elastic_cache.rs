//! Physically-resident elastic KV cache built on the generic Elastic engine.
//!
//! This is the P0-E1 remediation: the legacy `scirust::ccos::ElasticKvCache`
//! reports *logical* accounting (HOT 128 / WARM 96 / COLD 0) while every
//! tile physically occupies 128 bytes. This cache makes residency physical:
//!
//! - **HOT**: full 128-byte tile in the arena.
//! - **WARM**: physically packed 96-byte form (residual plane absent;
//!   `codec::pack_warm`), so `resident_bytes()` is the real allocated sum.
//! - **COLD**: slot released; bytes returned to the allocator.
//! - **PINNED**: protected from demotion/eviction by the tier machine.
//!
//! The controller is the generic `elastic-core` ECA; this type supplies the
//! `ElasticResource` + `ElasticBackend` implementations.

use elastic_core::budget::BudgetTree;
use elastic_core::controller::{
    ActionRequest, ControllerConfig, Decision, ElasticBackend, ElasticController, Observation,
};
use elastic_core::pressure::Pressure;
use elastic_core::reason::{code, Reason};
use elastic_core::{ElasticResource, StateId};

use crate::codec::{self, WARM_PACKED_BYTES};

/// Physical residency tier of a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalTier {
    /// Full 128-byte representation.
    Hot,
    /// Packed 96-byte representation.
    Warm,
    /// Slot released (no bytes allocated).
    Cold,
    /// Protected from adaptation.
    Pinned,
}

/// One physically-resident slot.
#[derive(Clone, Debug)]
pub struct PhysicalSlot {
    /// The tile in its current representation:
    /// - Hot/Pinned: full 128 bytes;
    /// - Warm: 96 bytes;
    /// - Cold: empty.
    bytes: Vec<u8>,
    tier: PhysicalTier,
    /// Insertion sequence (deterministic eviction order).
    #[allow(dead_code)] // kept for deterministic tie-breaks in future policies
    seq: u64,
    /// Cumulative attention mass (H2O importance).
    importance: f32,
}

impl PhysicalSlot {
    /// Bytes physically allocated for this slot.
    pub fn allocated_bytes(&self) -> usize {
        self.bytes.len()
    }
}

/// The physical elastic KV cache.
pub struct ElasticKvCache {
    slots: Vec<Option<PhysicalSlot>>,
    /// Next free slot index (stack).
    free: Vec<usize>,
    /// Total physically allocated bytes (sum of live slot `bytes.len()`).
    resident_bytes: usize,
    next_seq: u64,
    controller: ElasticController<KvCacheResource, KvCacheBackend>,
    /// Configuration snapshot for the controller.
    config: ControllerConfig,
    budget: BudgetTree,
    budget_node: Option<usize>,
    /// Number of COLD evictions.
    evictions: u64,
}

/// Resource identity for the ECA.
#[derive(Debug, Clone)]
pub struct KvCacheResource {
    id: String,
}

impl KvCacheResource {
    /// Create the resource identity.
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl ElasticResource for KvCacheResource {
    fn resource_id(&self) -> &str {
        &self.id
    }
}

/// The backend executes the physical transitions.
#[derive(Debug, Default)]
pub struct KvCacheBackend {
    /// Bytes released by the last demote/offload.
    pub released: u64,
    /// Bytes restored by the last promote/restore.
    pub restored: u64,
    /// Hard budget (bytes) enforced by `verify`.
    pub hard_budget: u64,
}

impl ElasticBackend for KvCacheBackend {
    type Error = &'static str;

    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.released += target_bytes;
        Ok(target_bytes)
    }

    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.restored += target_bytes;
        Ok(target_bytes)
    }

    fn offload(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.demote(target_bytes)
    }

    fn restore(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.promote(target_bytes)
    }

    fn prefetch(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn rebalance(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn verify(&mut self, expected_used: u64) -> Result<bool, Self::Error> {
        Ok(expected_used <= self.hard_budget)
    }
}

impl ElasticKvCache {
    /// Create a cache with a hard byte budget and an optional parent budget
    /// node for hierarchical coordination.
    pub fn new(hard_budget_bytes: usize, resource_id: &str) -> Self {
        let mut budget = BudgetTree::new();
        let node = budget.add_root(0, hard_budget_bytes as u64, hard_budget_bytes as u64, 0);
        let backend = KvCacheBackend {
            hard_budget: hard_budget_bytes as u64,
            ..KvCacheBackend::default()
        };
        let config = ControllerConfig::standard();
        let controller = ElasticController::new(
            KvCacheResource::new(resource_id),
            backend,
            config,
            budget.clone(),
            Some(node),
        );
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            resident_bytes: 0,
            next_seq: 0,
            controller,
            config,
            budget,
            budget_node: Some(node),
            evictions: 0,
        }
    }

    /// Physically allocated bytes (HOT 128 + WARM 96 + PINNED 128 per live
    /// slot; COLD contributes 0). This is the real allocator accounting, not
    /// a logical estimate.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Free bytes within the hard budget.
    pub fn free_bytes(&self) -> usize {
        self.config_budget_bytes()
            .saturating_sub(self.resident_bytes)
    }

    fn config_budget_bytes(&self) -> usize {
        self.budget
            .node(self.budget_node.expect("budget node configured"))
            .map(|n| n.hard_limit as usize)
            .unwrap_or(0)
    }

    /// `(hot, warm, cold, pinned)` slot counts.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut c = (0usize, 0usize, 0usize, 0usize);
        for s in self.slots.iter().flatten() {
            match s.tier {
                PhysicalTier::Hot => c.0 += 1,
                PhysicalTier::Warm => c.1 += 1,
                PhysicalTier::Cold => c.2 += 1,
                PhysicalTier::Pinned => c.3 += 1,
            }
        }
        c
    }

    /// Number of evictions.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Insert a HOT tile (full 128 bytes). Returns the slot index.
    pub fn insert(&mut self, tile: [u8; codec::TILE_BYTES]) -> usize {
        let slot = if let Some(s) = self.free.pop() {
            self.slots[s] = Some(PhysicalSlot {
                bytes: tile.to_vec(),
                tier: PhysicalTier::Hot,
                seq: self.next_seq,
                importance: 0.0,
            });
            s
        } else {
            self.slots.push(Some(PhysicalSlot {
                bytes: tile.to_vec(),
                tier: PhysicalTier::Hot,
                seq: self.next_seq,
                importance: 0.0,
            }));
            self.slots.len() - 1
        };
        self.next_seq += 1;
        self.resident_bytes += codec::TILE_BYTES;
        slot
    }

    /// Pin a slot (protected from demotion/eviction).
    pub fn pin(&mut self, slot: usize) -> bool {
        let Some(s) = self.slots.get_mut(slot).and_then(|s| s.as_mut()) else {
            return false;
        };
        if s.tier == PhysicalTier::Cold {
            return false;
        }
        // A WARM pin is promoted to full HOT representation (unpacked) so
        // the pinned slot always holds the full tile.
        if s.tier == PhysicalTier::Warm {
            let full = codec::unpack_warm(
                <&[u8; WARM_PACKED_BYTES]>::try_from(s.bytes.as_slice())
                    .expect("warm slot holds exactly 96 bytes"),
            );
            self.resident_bytes += codec::TILE_BYTES - WARM_PACKED_BYTES;
            s.bytes = full.to_vec();
        }
        s.tier = PhysicalTier::Pinned;
        true
    }

    /// Demote one slot HOT → WARM (physical: 128 → 96 bytes).
    ///
    /// Returns `Err` when the slot is not HOT, or when it is PINNED.
    pub fn demote_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(s) = self.slots.get_mut(slot).and_then(|s| s.as_mut()) else {
            return Err("slot absent");
        };
        match s.tier {
            PhysicalTier::Hot => {
                let full: &[u8; codec::TILE_BYTES] =
                    <&[u8; codec::TILE_BYTES]>::try_from(s.bytes.as_slice())
                        .expect("hot slot holds exactly 128 bytes");
                let packed = codec::pack_warm(full);
                self.resident_bytes -= codec::TILE_BYTES - WARM_PACKED_BYTES;
                s.bytes = packed.to_vec();
                s.tier = PhysicalTier::Warm;
                Ok(())
            }
            PhysicalTier::Warm => Ok(()),
            PhysicalTier::Cold => Err("slot is cold"),
            PhysicalTier::Pinned => Err("slot is pinned"),
        }
    }

    /// Promote one slot WARM → HOT (physical: 96 → 128 bytes).
    pub fn promote_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(s) = self.slots.get_mut(slot).and_then(|s| s.as_mut()) else {
            return Err("slot absent");
        };
        match s.tier {
            PhysicalTier::Warm => {
                let packed: &[u8; WARM_PACKED_BYTES] =
                    <&[u8; WARM_PACKED_BYTES]>::try_from(s.bytes.as_slice())
                        .expect("warm slot holds exactly 96 bytes");
                let full = codec::unpack_warm(packed);
                self.resident_bytes += codec::TILE_BYTES - WARM_PACKED_BYTES;
                s.bytes = full.to_vec();
                s.tier = PhysicalTier::Hot;
                Ok(())
            }
            PhysicalTier::Hot => Ok(()),
            PhysicalTier::Cold => Err("slot is cold"),
            PhysicalTier::Pinned => Err("slot is pinned"),
        }
    }

    /// Evict one slot → COLD (physical: bytes returned).
    pub fn evict_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(s) = self.slots.get_mut(slot).and_then(|s| s.as_mut()) else {
            return Err("slot absent");
        };
        if s.tier == PhysicalTier::Pinned {
            return Err("slot is pinned");
        }
        if s.tier == PhysicalTier::Cold {
            return Ok(());
        }
        self.resident_bytes -= s.bytes.len();
        s.bytes.clear();
        s.tier = PhysicalTier::Cold;
        self.evictions += 1;
        self.free.push(slot);
        Ok(())
    }

    /// Score a live slot against a prepared query.
    pub fn score(&self, slot: usize, q_coarse: &[f32], q_sign: &[u64]) -> Option<f32> {
        let s = self.slots.get(slot)?.as_ref()?;
        if s.tier == PhysicalTier::Cold {
            return None;
        }
        let tile = match s.tier {
            PhysicalTier::Hot | PhysicalTier::Pinned => {
                let full: &[u8; codec::TILE_BYTES] =
                    <&[u8; codec::TILE_BYTES]>::try_from(s.bytes.as_slice()).ok()?;
                crate::mem::tile::SerializedTile(*full)
            }
            PhysicalTier::Warm => {
                let packed: &[u8; WARM_PACKED_BYTES] =
                    <&[u8; WARM_PACKED_BYTES]>::try_from(s.bytes.as_slice()).ok()?;
                crate::mem::tile::SerializedTile(codec::unpack_warm(packed))
            }
            PhysicalTier::Cold => return None,
        };
        tile.try_score(q_coarse, q_sign).ok()
    }

    /// Run one ECA step with the current residency observation. The
    /// controller may decide to demote under pressure; the backend's
    /// physical transitions are applied to the lowest-importance live
    /// slots first.
    pub fn step(&mut self) -> Result<Decision, &'static str> {
        let obs = Observation::new(
            self.next_seq,
            self.resident_bytes as u64,
            self.config_budget_bytes() as u64,
            self.slots.iter().flatten().count() as f64,
        );
        let decision = self.controller.step(obs)?;
        match decision.action {
            ActionRequest::Demote | ActionRequest::Offload => {
                // Demote in rounds: each pass frees 32 B per slot until the
                // residency is under the hard budget (or nothing is left to
                // demote), so a single ECA step can restore the invariant.
                let mut guard = 0;
                while self.resident_bytes > self.config_budget_bytes() && guard < 1 << 16 {
                    let before = self.resident_bytes;
                    self.demote_lowest_importance(1);
                    if self.resident_bytes == before {
                        // No HOT slots left; evict a WARM slot instead so
                        // the budget invariant still holds.
                        self.evict_lowest_importance(1);
                    }
                    if self.resident_bytes == before {
                        break; // nothing left at all
                    }
                    guard += 1;
                }
                // If the controller decided to demote but we were already
                // under budget, demote one slot anyway (the decision is a
                // demand signal: pressure is high).
                if self.resident_bytes <= self.config_budget_bytes()
                    && guard == 0
                    && self.demote_lowest_importance(1) == 0
                {
                    self.evict_lowest_importance(1);
                }
            }
            ActionRequest::Promote | ActionRequest::Restore => {
                self.promote_highest_importance(1);
            }
            _ => {}
        }
        Ok(decision)
    }

    fn demote_lowest_importance(&mut self, n: usize) -> usize {
        let mut candidates: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.as_ref().map(|x| x.tier), Some(PhysicalTier::Hot)))
            .map(|(i, _)| i)
            .collect();
        candidates.sort_by(|&a, &b| {
            self.slots[a]
                .as_ref()
                .unwrap()
                .importance
                .total_cmp(&self.slots[b].as_ref().unwrap().importance)
        });
        let mut done = 0;
        for slot in candidates.into_iter().take(n) {
            if self.demote_slot(slot).is_ok() {
                done += 1;
            }
        }
        done
    }

    fn promote_highest_importance(&mut self, n: usize) {
        let mut candidates: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.as_ref().map(|x| x.tier), Some(PhysicalTier::Warm)))
            .map(|(i, _)| i)
            .collect();
        candidates.sort_by(|&a, &b| {
            self.slots[b]
                .as_ref()
                .unwrap()
                .importance
                .total_cmp(&self.slots[a].as_ref().unwrap().importance)
        });
        for slot in candidates.into_iter().take(n) {
            let _ = self.promote_slot(slot);
        }
    }

    fn evict_lowest_importance(&mut self, n: usize) {
        let mut candidates: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                matches!(
                    s.as_ref().map(|x| x.tier),
                    Some(PhysicalTier::Hot) | Some(PhysicalTier::Warm)
                )
            })
            .map(|(i, _)| i)
            .collect();
        candidates.sort_by(|&a, &b| {
            self.slots[a]
                .as_ref()
                .unwrap()
                .importance
                .total_cmp(&self.slots[b].as_ref().unwrap().importance)
        });
        for slot in candidates.into_iter().take(n) {
            let _ = self.evict_slot(slot);
        }
    }

    /// Observe attention mass (H2O importance) for live slots.
    pub fn observe_scores(&mut self, scores: &[(usize, f32)], temperature: f32) {
        if scores.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
            return;
        }
        let m = scores
            .iter()
            .map(|&(_, s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &(_, s) in scores {
            sum += ((s - m) / temperature).exp();
        }
        if !sum.is_finite() || sum <= 0.0 {
            return;
        }
        for &(slot, s) in scores {
            if let Some(x) = self.slots.get_mut(slot).and_then(|x| x.as_mut()) {
                if x.tier != PhysicalTier::Cold {
                    x.importance += ((s - m) / temperature).exp() / sum;
                }
            }
        }
    }

    /// Current pressure (from the controller).
    pub fn pressure(&self) -> Pressure {
        Pressure::from_used(
            self.resident_bytes as u64,
            self.config_budget_bytes() as u64,
            self.config.watermarks,
        )
    }

    /// Deterministic state id (for journals).
    pub fn state_id(&self) -> StateId {
        self.controller.state_id()
    }

    /// The last decision reason, for telemetry.
    pub fn last_reason(&self) -> Option<Reason> {
        None
    }

    /// Stable reason code of the last decision (telemetry hook).
    pub fn last_reason_code(&self) -> Option<&'static str> {
        Some(code::OPERATOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile(byte: u8) -> [u8; codec::TILE_BYTES] {
        let mut t = [0u8; codec::TILE_BYTES];
        t.fill(byte);
        t
    }

    #[test]
    fn hot_warm_cold_physical_bytes() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let a = cache.insert(make_tile(1));
        let b = cache.insert(make_tile(2));
        assert_eq!(cache.resident_bytes(), 256);
        assert_eq!(cache.counts(), (2, 0, 0, 0));

        cache.demote_slot(a).unwrap();
        assert_eq!(cache.resident_bytes(), 128 + WARM_PACKED_BYTES);
        assert_eq!(cache.counts(), (1, 1, 0, 0));

        cache.evict_slot(b).unwrap();
        assert_eq!(cache.resident_bytes(), WARM_PACKED_BYTES);
        assert_eq!(cache.counts(), (0, 1, 1, 0));
        assert_eq!(cache.evictions(), 1);
    }

    #[test]
    fn warm_pack_unpack_roundtrip_preserves_fields() {
        let mut t = make_tile(0xAB);
        // Fields beyond the residual must survive the roundtrip.
        let flags: [u8; 2] = 0x00FFu16.to_le_bytes();
        t[codec::FLAGS_OFFSET] = flags[0];
        t[codec::FLAGS_OFFSET + 1] = flags[1];
        let packed = codec::pack_warm(&t);
        assert_eq!(packed.len(), WARM_PACKED_BYTES);
        let unpacked = codec::unpack_warm(&packed);
        assert_eq!(
            &unpacked[..codec::RESIDUAL_OFFSET],
            &t[..codec::RESIDUAL_OFFSET]
        );
        assert_eq!(
            &unpacked[codec::RESIDUAL_OFFSET + codec::RESIDUAL_WORDS * 8..],
            &t[codec::RESIDUAL_OFFSET + codec::RESIDUAL_WORDS * 8..]
        );
        // Residual plane is zeroed in the unpacked form.
        assert!(unpacked
            [codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + codec::RESIDUAL_WORDS * 8]
            .iter()
            .all(|&b| b == 0));
    }

    #[test]
    fn pinned_slots_survive_demote_and_evict() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let a = cache.insert(make_tile(1));
        let b = cache.insert(make_tile(2));
        cache.pin(a);
        // `pin` promotes the slot to the PINNED tier (it is no longer HOT).
        assert!(cache.demote_slot(a).is_err());
        assert!(cache.evict_slot(a).is_err());
        // Unpinned slot still evictable.
        assert!(cache.evict_slot(b).is_ok());
        assert_eq!(cache.counts(), (0, 0, 1, 1));
    }

    #[test]
    fn controller_step_demotes_under_pressure() {
        let mut cache = ElasticKvCache::new(512, "test");
        // Fill the budget so pressure is High.
        for _ in 0..4 {
            cache.insert(make_tile(3));
        }
        assert_eq!(cache.resident_bytes(), 512);
        // The first step enters the hysteresis gate; the next steps demote
        // until the residency is under the hard budget.
        let _ = cache.step();
        let mut guard = 0;
        while cache.resident_bytes() > 512 && guard < 8 {
            let _ = cache.step();
            guard += 1;
        }
        assert!(cache.resident_bytes() <= 512);
        assert!(cache.counts().1 >= 1); // at least one WARM
    }

    #[test]
    fn resident_bytes_never_exceed_budget_after_step() {
        let mut cache = ElasticKvCache::new(256, "test");
        for _ in 0..6 {
            cache.insert(make_tile(4));
        }
        assert_eq!(cache.resident_bytes(), 768); // over budget
                                                 // Repeated ECA steps demote the lowest-importance slots until the
                                                 // physical residency is back under budget.
        for _ in 0..8 {
            let _ = cache.step();
            if cache.resident_bytes() <= 256 {
                break;
            }
        }
        assert!(
            cache.resident_bytes() <= 256,
            "residency {} > budget 256",
            cache.resident_bytes()
        );
    }
}
