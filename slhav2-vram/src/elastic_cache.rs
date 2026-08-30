//! Physically-resident elastic KV cache backed by the generic Elastic engine.
//!
//! Residency and backing are separate resources:
//! - HOT: 128 resident bytes;
//! - WARM: 96 resident bytes + 32 residual backing bytes;
//! - COLD: 0 resident bytes + a restorable backing representation;
//! - PINNED: 128 resident bytes protected from adaptation.
//!
//! Multi-slot controller actions are transactional: either the requested
//! residency target is reached and verified, or the complete pre-action state
//! is restored.

use elastic_core::budget::BudgetTree;
use elastic_core::controller::{
    ActionRequest, ControllerConfig, Decision, ElasticBackend, ElasticController, Observation,
};
use elastic_core::pressure::Pressure;
use elastic_core::reason::Reason;
use elastic_core::{ElasticResource, StateId};

use crate::codec::{self, WARM_PACKED_BYTES};

const RESIDUAL_BYTES: usize = codec::RESIDUAL_WORDS * core::mem::size_of::<u64>();

/// Physical residency tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalTier {
    /// Full resident representation.
    Hot,
    /// Packed resident representation with residual backing.
    Warm,
    /// Offloaded representation with no resident bytes.
    Cold,
    /// Full resident representation protected from adaptation.
    Pinned,
}

/// One cache slot and its reversible backing state.
#[derive(Clone, Debug)]
pub struct PhysicalSlot {
    bytes: Vec<u8>,
    offloaded: Option<Box<[u8]>>,
    warm_residual: Option<[u8; RESIDUAL_BYTES]>,
    tier: PhysicalTier,
    seq: u64,
    importance: f32,
}

impl PhysicalSlot {
    /// Bytes charged to the resident budget.
    pub fn allocated_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Bytes retained outside the resident budget for reversible restoration.
    pub fn offloaded_bytes(&self) -> usize {
        self.offloaded.as_deref().map_or(0, |b| b.len())
            + self.warm_residual.as_ref().map_or(0, |_| RESIDUAL_BYTES)
    }
}

/// Physically managed SLHAv2 tile cache.
pub struct ElasticKvCache {
    slots: Vec<Option<PhysicalSlot>>,
    resident_bytes: usize,
    next_seq: u64,
    controller: ElasticController<KvCacheResource, KvCacheBackend>,
    config: ControllerConfig,
    budget: BudgetTree,
    budget_node: Option<usize>,
    evictions: u64,
    last_reason: Option<Reason>,
}

/// Identity exposed to the generic ECA.
#[derive(Debug, Clone)]
pub struct KvCacheResource {
    id: String,
}

impl KvCacheResource {
    /// Construct a resource identity.
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl ElasticResource for KvCacheResource {
    fn resource_id(&self) -> &str {
        &self.id
    }
}

/// Deterministic accounting backend used by the generic decision controller.
/// Real mutation and verification are performed by [`ElasticKvCache`].
#[derive(Debug, Default)]
pub struct KvCacheBackend {
    /// Bytes the decision model asked to release.
    pub released: u64,
    /// Bytes the decision model asked to restore.
    pub restored: u64,
    /// Hard resident budget.
    pub hard_budget: u64,
}

impl ElasticBackend for KvCacheBackend {
    type Error = &'static str;

    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.released = self.released.saturating_add(target_bytes);
        Ok(target_bytes)
    }

    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.restored = self.restored.saturating_add(target_bytes);
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
    /// Create an empty cache with a resident-byte hard budget.
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
            resident_bytes: 0,
            next_seq: 0,
            controller,
            config,
            budget,
            budget_node: Some(node),
            evictions: 0,
            last_reason: None,
        }
    }

    /// Current bytes charged to the resident budget.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Current reversible backing bytes outside the resident budget.
    pub fn offloaded_bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(PhysicalSlot::offloaded_bytes)
            .sum()
    }

    /// Configured hard resident budget.
    pub fn hard_budget_bytes(&self) -> usize {
        self.config_budget_bytes()
    }

    /// Remaining resident-budget headroom.
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

    /// Return `(hot, warm, cold, pinned)` counts.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut counts = (0, 0, 0, 0);
        for slot in self.slots.iter().flatten() {
            match slot.tier {
                PhysicalTier::Hot => counts.0 += 1,
                PhysicalTier::Warm => counts.1 += 1,
                PhysicalTier::Cold => counts.2 += 1,
                PhysicalTier::Pinned => counts.3 += 1,
            }
        }
        counts
    }

    /// Number of successful resident→COLD transitions.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Insert a full tile as HOT and return its stable slot id.
    pub fn insert(&mut self, mut tile: [u8; codec::TILE_BYTES]) -> usize {
        Self::set_warm_flag(&mut tile, false);
        let slot = self.slots.len();
        self.slots.push(Some(PhysicalSlot {
            bytes: tile.to_vec(),
            offloaded: None,
            warm_residual: None,
            tier: PhysicalTier::Hot,
            seq: self.next_seq,
            importance: 0.0,
        }));
        self.next_seq = self.next_seq.saturating_add(1);
        self.resident_bytes = self.resident_bytes.saturating_add(codec::TILE_BYTES);
        slot
    }

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
                let packed: &[u8; WARM_PACKED_BYTES] = state.bytes.as_slice().try_into().ok()?;
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

    /// Protect a resident slot from adaptation. WARM is promoted losslessly
    /// before pinning; COLD cannot be pinned without an explicit restore.
    pub fn pin(&mut self, slot: usize) -> bool {
        let tier = match self.slots.get(slot).and_then(Option::as_ref) {
            Some(slot) => slot.tier,
            None => return false,
        };
        if tier == PhysicalTier::Cold {
            return false;
        }
        if tier == PhysicalTier::Warm && self.promote_slot(slot).is_err() {
            return false;
        }
        let Some(slot) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return false;
        };
        slot.tier = PhysicalTier::Pinned;
        true
    }

    /// Demote HOT→WARM and retain the removed residual in backing storage.
    pub fn demote_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(state) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return Err("slot absent");
        };
        match state.tier {
            PhysicalTier::Hot => {
                let mut full: [u8; codec::TILE_BYTES] = state
                    .bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| "hot slot does not hold exactly 128 bytes")?;
                let mut residual = [0u8; RESIDUAL_BYTES];
                residual.copy_from_slice(
                    &full[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES],
                );
                Self::set_warm_flag(&mut full, true);
                state.bytes = codec::pack_warm(&full).to_vec();
                state.warm_residual = Some(residual);
                state.offloaded = None;
                state.tier = PhysicalTier::Warm;
                self.resident_bytes = self
                    .resident_bytes
                    .saturating_sub(codec::TILE_BYTES - WARM_PACKED_BYTES);
                Ok(())
            }
            PhysicalTier::Warm => Ok(()),
            PhysicalTier::Cold => Err("slot is cold"),
            PhysicalTier::Pinned => Err("slot is pinned"),
        }
    }

    /// Promote WARM→HOT and restore the exact residual plane.
    pub fn promote_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let tier = self
            .slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or("slot absent")?
            .tier;
        match tier {
            PhysicalTier::Hot => return Ok(()),
            PhysicalTier::Pinned => return Err("slot is pinned"),
            PhysicalTier::Cold => return Err("slot is cold; call restore_slot first"),
            PhysicalTier::Warm => {}
        }
        let growth = codec::TILE_BYTES - WARM_PACKED_BYTES;
        if self.resident_bytes.saturating_add(growth) > self.config_budget_bytes() {
            return Err("promotion would exceed resident hard budget");
        }
        let state = self.slots[slot].as_mut().expect("slot checked above");
        let packed: &[u8; WARM_PACKED_BYTES] = state
            .bytes
            .as_slice()
            .try_into()
            .map_err(|_| "warm slot does not hold exactly 96 bytes")?;
        let residual = state
            .warm_residual
            .take()
            .ok_or("warm slot has no residual backing")?;
        let mut full = codec::unpack_warm(packed);
        full[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES]
            .copy_from_slice(&residual);
        Self::set_warm_flag(&mut full, false);
        state.bytes = full.to_vec();
        state.tier = PhysicalTier::Hot;
        self.resident_bytes = self.resident_bytes.saturating_add(growth);
        Ok(())
    }

    /// Move HOT/WARM→COLD while retaining a restorable backing representation.
    pub fn evict_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(state) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return Err("slot absent");
        };
        if state.tier == PhysicalTier::Pinned {
            return Err("slot is pinned");
        }
        if state.tier == PhysicalTier::Cold {
            return Ok(());
        }
        let released = state.bytes.len();
        if released != codec::TILE_BYTES && released != WARM_PACKED_BYTES {
            return Err("resident slot has invalid representation length");
        }
        state.offloaded = Some(core::mem::take(&mut state.bytes).into_boxed_slice());
        state.tier = PhysicalTier::Cold;
        self.resident_bytes = self.resident_bytes.saturating_sub(released);
        self.evictions = self.evictions.saturating_add(1);
        Ok(())
    }

    /// Restore a COLD slot to its prior HOT or WARM representation.
    pub fn restore_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let state = self
            .slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or("slot absent")?;
        if state.tier != PhysicalTier::Cold {
            return Ok(());
        }
        let len = state
            .offloaded
            .as_deref()
            .ok_or("cold slot has no backing")?
            .len();
        if self.resident_bytes.saturating_add(len) > self.config_budget_bytes() {
            return Err("restore would exceed resident hard budget");
        }
        let state = self.slots[slot].as_mut().expect("slot checked above");
        state.bytes = state
            .offloaded
            .take()
            .ok_or("cold slot has no backing")?
            .into_vec();
        state.tier = match state.bytes.len() {
            codec::TILE_BYTES => PhysicalTier::Hot,
            WARM_PACKED_BYTES => PhysicalTier::Warm,
            _ => return Err("cold backing has invalid representation length"),
        };
        self.resident_bytes = self.resident_bytes.saturating_add(state.bytes.len());
        Ok(())
    }

    /// Score a resident slot. COLD returns `None` until restored.
    pub fn score(&self, slot: usize, q_coarse: &[f32], q_sign: &[u64]) -> Option<f32> {
        let state = self.slots.get(slot)?.as_ref()?;
        let tile = match state.tier {
            PhysicalTier::Hot | PhysicalTier::Pinned => {
                let full: &[u8; codec::TILE_BYTES] = state.bytes.as_slice().try_into().ok()?;
                crate::mem::tile::SerializedTile(*full)
            }
            PhysicalTier::Warm => {
                let packed: &[u8; WARM_PACKED_BYTES] = state.bytes.as_slice().try_into().ok()?;
                crate::mem::tile::SerializedTile(codec::unpack_warm(packed))
            }
            PhysicalTier::Cold => return None,
        };
        tile.try_score(q_coarse, q_sign).ok()
    }

    /// Transactionally apply one ECA action to the real cache.
    ///
    /// For demotion/offload, `target_resident_bytes` is a strict target. If it
    /// cannot be reached without touching a PINNED slot or violating another
    /// invariant, every mutation made by this call is rolled back.
    pub fn apply_action(
        &mut self,
        action: ActionRequest,
        target_resident_bytes: usize,
    ) -> Result<bool, &'static str> {
        let snapshot_slots = self.slots.clone();
        let snapshot_resident = self.resident_bytes;
        let snapshot_evictions = self.evictions;
        let target = target_resident_bytes.min(self.config_budget_bytes());

        let result = (|| {
            match action {
                ActionRequest::Demote => {
                    while self.resident_bytes > target {
                        let prior = self.resident_bytes;
                        if self.demote_lowest_importance(1) == 0 {
                            self.evict_lowest_importance(1);
                        }
                        if self.resident_bytes == prior {
                            break;
                        }
                    }
                }
                ActionRequest::Offload => {
                    while self.resident_bytes > target {
                        let prior = self.resident_bytes;
                        self.evict_lowest_importance(1);
                        if self.resident_bytes == prior {
                            break;
                        }
                    }
                }
                ActionRequest::Promote => {
                    self.promote_highest_importance(1);
                }
                ActionRequest::Restore => {
                    self.restore_highest_importance(1);
                }
                ActionRequest::None
                | ActionRequest::Prefetch
                | ActionRequest::Rebalance
                | ActionRequest::Operator(_) => {}
            }

            self.verify_accounting()?;
            if matches!(action, ActionRequest::Demote | ActionRequest::Offload)
                && self.resident_bytes > target
            {
                return Err("requested residency target cannot be satisfied");
            }
            if self.resident_bytes > self.config_budget_bytes() {
                return Err("physical residency exceeds hard budget");
            }
            Ok(self.resident_bytes != snapshot_resident)
        })();

        if result.is_err() {
            self.slots = snapshot_slots;
            self.resident_bytes = snapshot_resident;
            self.evictions = snapshot_evictions;
            debug_assert!(self.verify_accounting().is_ok());
        }
        result
    }

    /// Ask the cache-local ECA for a decision and transactionally execute it.
    pub fn step(&mut self) -> Result<Decision, &'static str> {
        let budget = self.config_budget_bytes();
        let observation = Observation::new(
            self.next_seq,
            self.resident_bytes as u64,
            budget as u64,
            self.slots.iter().flatten().count() as f64,
        );
        let mut decision = self.controller.step(observation)?;
        self.last_reason = decision.trace.reason;

        let target = match decision.action {
            ActionRequest::Demote | ActionRequest::Offload => {
                if self.resident_bytes > budget {
                    budget
                } else {
                    self.resident_bytes.saturating_sub(RESIDUAL_BYTES)
                }
            }
            _ => budget,
        };
        let changed = self.apply_action(decision.action, target)?;
        let verified = self.verify_accounting().is_ok() && self.resident_bytes <= budget;
        let outcome = match (decision.action, changed) {
            (ActionRequest::Demote, true) => "demoted",
            (ActionRequest::Offload, true) => "offloaded",
            (ActionRequest::Promote, true) => "promoted",
            (ActionRequest::Restore, true) => "restored",
            (_, false) => "no_change",
            _ => "applied",
        };
        decision.trace.complete(outcome, verified);
        if !verified {
            return Err("physical residency verification failed");
        }
        Ok(decision)
    }

    fn demote_lowest_importance(&mut self, n: usize) -> usize {
        let mut candidates = self.candidates(PhysicalTier::Hot);
        candidates.sort_by(|&a, &b| self.cmp_low_first(a, b));
        let mut done = 0;
        for slot in candidates.into_iter().take(n) {
            if self.demote_slot(slot).is_ok() {
                done += 1;
            }
        }
        done
    }

    fn promote_highest_importance(&mut self, n: usize) -> usize {
        let mut candidates = self.candidates(PhysicalTier::Warm);
        candidates.sort_by(|&a, &b| self.cmp_high_first(a, b));
        let mut done = 0;
        for slot in candidates {
            if done == n {
                break;
            }
            if self.promote_slot(slot).is_ok() {
                done += 1;
            }
        }
        done
    }

    fn restore_highest_importance(&mut self, n: usize) -> usize {
        let mut candidates = self.candidates(PhysicalTier::Cold);
        candidates.sort_by(|&a, &b| self.cmp_high_first(a, b));
        let mut done = 0;
        for slot in candidates {
            if done == n {
                break;
            }
            if self.restore_slot(slot).is_ok() {
                done += 1;
            }
        }
        done
    }

    fn evict_lowest_importance(&mut self, n: usize) -> usize {
        let mut candidates: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                matches!(
                    slot.as_ref().map(|state| state.tier),
                    Some(PhysicalTier::Hot) | Some(PhysicalTier::Warm)
                )
            })
            .map(|(index, _)| index)
            .collect();
        candidates.sort_by(|&a, &b| self.cmp_low_first(a, b));
        let mut done = 0;
        for slot in candidates.into_iter().take(n) {
            if self.evict_slot(slot).is_ok() {
                done += 1;
            }
        }
        done
    }

    fn candidates(&self, tier: PhysicalTier) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.as_ref().is_some_and(|state| state.tier == tier))
            .map(|(index, _)| index)
            .collect()
    }

    fn cmp_low_first(&self, a: usize, b: usize) -> core::cmp::Ordering {
        let a = self.slots[a].as_ref().expect("candidate exists");
        let b = self.slots[b].as_ref().expect("candidate exists");
        a.importance
            .total_cmp(&b.importance)
            .then_with(|| a.seq.cmp(&b.seq))
    }

    fn cmp_high_first(&self, a: usize, b: usize) -> core::cmp::Ordering {
        self.cmp_low_first(b, a)
    }

    fn verify_accounting(&self) -> Result<(), &'static str> {
        let actual: usize = self
            .slots
            .iter()
            .flatten()
            .map(PhysicalSlot::allocated_bytes)
            .sum();
        if actual != self.resident_bytes {
            return Err("resident byte accounting mismatch");
        }
        for state in self.slots.iter().flatten() {
            let valid = match state.tier {
                PhysicalTier::Hot | PhysicalTier::Pinned => {
                    state.bytes.len() == codec::TILE_BYTES && state.offloaded.is_none()
                }
                PhysicalTier::Warm => {
                    state.bytes.len() == WARM_PACKED_BYTES
                        && state.warm_residual.is_some()
                        && state.offloaded.is_none()
                }
                PhysicalTier::Cold => state.bytes.is_empty() && state.offloaded.is_some(),
            };
            if !valid {
                return Err("slot residency invariant violated");
            }
        }
        Ok(())
    }

    fn set_warm_flag(tile: &mut [u8; codec::TILE_BYTES], warm: bool) {
        let mut flags =
            u16::from_le_bytes([tile[codec::FLAGS_OFFSET], tile[codec::FLAGS_OFFSET + 1]]);
        if warm {
            flags |= codec::FLAG_WARM;
        } else {
            flags &= !codec::FLAG_WARM;
        }
        tile[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2].copy_from_slice(&flags.to_le_bytes());
    }

    /// Accumulate deterministic softmax-normalized attention importance.
    pub fn observe_scores(&mut self, scores: &[(usize, f32)], temperature: f32) {
        if scores.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
            return;
        }
        let max_score = scores
            .iter()
            .map(|&(_, score)| score)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores
            .iter()
            .map(|&(_, score)| ((score - max_score) / temperature).exp())
            .sum();
        if !sum.is_finite() || sum <= 0.0 {
            return;
        }
        for &(slot, score) in scores {
            if let Some(state) = self.slots.get_mut(slot).and_then(Option::as_mut) {
                state.importance += ((score - max_score) / temperature).exp() / sum;
            }
        }
    }

    /// Current resident-budget pressure.
    pub fn pressure(&self) -> Pressure {
        Pressure::from_used(
            self.resident_bytes as u64,
            self.config_budget_bytes() as u64,
            self.config.watermarks,
        )
    }

    /// Deterministic ECA state identifier.
    pub fn state_id(&self) -> StateId {
        self.controller.state_id()
    }

    /// Last ECA reason.
    pub fn last_reason(&self) -> Option<Reason> {
        self.last_reason
    }

    /// Stable code of the last ECA reason.
    pub fn last_reason_code(&self) -> Option<&'static str> {
        self.last_reason.map(|reason| reason.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile(seed: u8) -> [u8; codec::TILE_BYTES] {
        let mut tile = [0u8; codec::TILE_BYTES];
        tile[..codec::LATENT_BYTES].fill(seed);
        tile[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES].fill(seed);
        tile[codec::SCALE_OFFSET..codec::SCALE_OFFSET + 4].copy_from_slice(&1.0f32.to_le_bytes());
        tile[codec::DYNAMIC_LAMBDA_OFFSET..codec::DYNAMIC_LAMBDA_OFFSET + 4]
            .copy_from_slice(&0.25f32.to_le_bytes());
        tile[codec::GROUP_SCALES_OFFSET..codec::GROUP_SCALES_OFFSET + codec::N_GROUP_SCALES]
            .fill(255);
        tile
    }

    #[test]
    fn warm_is_smaller_and_losslessly_reversible() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let slot = cache.insert(make_tile(0));
        let query = [0.0f32; codec::D_C];
        let signs = [0u64; codec::RESIDUAL_WORDS];
        let hot = cache.score(slot, &query, &signs).unwrap();
        assert_eq!(hot, 64.0);

        cache.demote_slot(slot).unwrap();
        assert_eq!(cache.resident_bytes(), WARM_PACKED_BYTES);
        assert_eq!(cache.offloaded_bytes(), RESIDUAL_BYTES);
        assert_eq!(cache.score(slot, &query, &signs).unwrap(), 0.0);

        cache.promote_slot(slot).unwrap();
        assert_eq!(cache.resident_bytes(), codec::TILE_BYTES);
        assert_eq!(cache.offloaded_bytes(), 0);
        assert_eq!(cache.score(slot, &query, &signs).unwrap(), hot);
    }

    #[test]
    fn cold_is_reversible_and_slot_identity_is_stable() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let slot = cache.insert(make_tile(7));
        cache.evict_slot(slot).unwrap();
        assert_eq!(cache.resident_bytes(), 0);
        assert_eq!(cache.offloaded_bytes(), codec::TILE_BYTES);
        let other = cache.insert(make_tile(8));
        assert_ne!(slot, other);
        cache.restore_slot(slot).unwrap();
        assert_eq!(cache.counts(), (2, 0, 0, 0));
    }

    #[test]
    fn exact_residency_target_is_enforced() {
        let mut cache = ElasticKvCache::new(1024, "test");
        for seed in 0..4 {
            cache.insert(make_tile(seed));
        }
        assert!(cache.apply_action(ActionRequest::Demote, 384).unwrap());
        assert!(cache.resident_bytes() <= 384);
        assert!(cache.apply_action(ActionRequest::Offload, 192).unwrap());
        assert!(cache.resident_bytes() <= 192);
    }

    #[test]
    fn impossible_target_rolls_back_every_partial_mutation() {
        let mut cache = ElasticKvCache::new(512, "test");
        let pinned = cache.insert(make_tile(1));
        cache.insert(make_tile(2));
        cache.insert(make_tile(3));
        assert!(cache.pin(pinned));
        let before_counts = cache.counts();
        let before_resident = cache.resident_bytes();
        let before_offloaded = cache.offloaded_bytes();
        let before_evictions = cache.evictions();

        assert!(cache.apply_action(ActionRequest::Offload, 0).is_err());
        assert_eq!(cache.counts(), before_counts);
        assert_eq!(cache.resident_bytes(), before_resident);
        assert_eq!(cache.offloaded_bytes(), before_offloaded);
        assert_eq!(cache.evictions(), before_evictions);
    }

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
        assert!(
            tile[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES]
                .iter()
                .all(|&byte| byte == 0)
        );
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

    #[test]
    fn pinned_slots_are_never_demoted_or_offloaded() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let pinned = cache.insert(make_tile(1));
        let other = cache.insert(make_tile(2));
        assert!(cache.pin(pinned));
        assert!(cache.demote_slot(pinned).is_err());
        assert!(cache.evict_slot(pinned).is_err());
        assert!(cache.apply_action(ActionRequest::Offload, 128).unwrap());
        assert_eq!(cache.counts(), (0, 0, 1, 1));
        assert!(cache
            .score(other, &[0.0; codec::D_C], &[0; codec::RESIDUAL_WORDS])
            .is_none());
    }

    #[test]
    fn controller_step_restores_hard_budget() {
        let mut cache = ElasticKvCache::new(256, "test");
        for _ in 0..6 {
            cache.insert(make_tile(4));
        }
        for _ in 0..8 {
            let _ = cache.step();
            if cache.resident_bytes() <= 256 {
                break;
            }
        }
        assert!(cache.resident_bytes() <= 256);
        cache.verify_accounting().unwrap();
    }
}
