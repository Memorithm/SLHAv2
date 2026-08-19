//! Physically-resident elastic KV cache built on the generic Elastic engine.
//!
//! Logical residency and backing storage are separate:
//! - HOT: 128 resident bytes;
//! - WARM: 96 resident bytes + the removed 32-byte residual in backing;
//! - COLD: 0 resident bytes + a restorable backing representation;
//! - PINNED: 128 resident bytes protected from adaptation.
//!
//! This makes every transition reversible while keeping `resident_bytes()` an
//! exact accounting of the bytes charged to the active residency budget.

use elastic_core::budget::BudgetTree;
use elastic_core::controller::{
    ActionRequest, ControllerConfig, Decision, ElasticBackend, ElasticController, Observation,
};
use elastic_core::pressure::Pressure;
use elastic_core::reason::Reason;
use elastic_core::{ElasticResource, StateId};

use crate::codec::{self, WARM_PACKED_BYTES};

const RESIDUAL_BYTES: usize = codec::RESIDUAL_WORDS * core::mem::size_of::<u64>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalTier {
    Hot,
    Warm,
    Cold,
    Pinned,
}

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
    pub fn allocated_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn offloaded_bytes(&self) -> usize {
        self.offloaded.as_deref().map_or(0, |b| b.len())
            + self.warm_residual.as_ref().map_or(0, |_| RESIDUAL_BYTES)
    }
}

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

#[derive(Debug, Clone)]
pub struct KvCacheResource {
    id: String,
}

impl KvCacheResource {
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
/// Real slot mutation and verification happen in [`ElasticKvCache::apply_action`].
#[derive(Debug, Default)]
pub struct KvCacheBackend {
    pub released: u64,
    pub restored: u64,
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

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn offloaded_bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(PhysicalSlot::offloaded_bytes)
            .sum()
    }

    pub fn hard_budget_bytes(&self) -> usize {
        self.config_budget_bytes()
    }

    pub fn free_bytes(&self) -> usize {
        self.config_budget_bytes().saturating_sub(self.resident_bytes)
    }

    fn config_budget_bytes(&self) -> usize {
        self.budget
            .node(self.budget_node.expect("budget node configured"))
            .map(|n| n.hard_limit as usize)
            .unwrap_or(0)
    }

    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut c = (0, 0, 0, 0);
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

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

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

    pub fn pin(&mut self, slot: usize) -> bool {
        let tier = match self.slots.get(slot).and_then(Option::as_ref) {
            Some(s) => s.tier,
            None => return false,
        };
        if tier == PhysicalTier::Cold {
            return false;
        }
        if tier == PhysicalTier::Warm && self.promote_slot(slot).is_err() {
            return false;
        }
        let Some(s) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return false;
        };
        s.tier = PhysicalTier::Pinned;
        true
    }

    pub fn demote_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(s) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return Err("slot absent");
        };
        match s.tier {
            PhysicalTier::Hot => {
                let mut full: [u8; codec::TILE_BYTES] = s
                    .bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| "hot slot does not hold exactly 128 bytes")?;
                let mut residual = [0u8; RESIDUAL_BYTES];
                residual.copy_from_slice(
                    &full[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES],
                );
                Self::set_warm_flag(&mut full, true);
                s.bytes = codec::pack_warm(&full).to_vec();
                s.warm_residual = Some(residual);
                s.offloaded = None;
                s.tier = PhysicalTier::Warm;
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

    pub fn promote_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let tier = self
            .slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or("slot absent")?
            .tier;
        if tier == PhysicalTier::Hot {
            return Ok(());
        }
        if tier == PhysicalTier::Pinned {
            return Err("slot is pinned");
        }
        if tier == PhysicalTier::Cold {
            return Err("slot is cold; call restore_slot first");
        }
        let growth = codec::TILE_BYTES - WARM_PACKED_BYTES;
        if self.resident_bytes.saturating_add(growth) > self.config_budget_bytes() {
            return Err("promotion would exceed resident hard budget");
        }
        let s = self.slots[slot].as_mut().expect("slot checked above");
        let packed: &[u8; WARM_PACKED_BYTES] = s
            .bytes
            .as_slice()
            .try_into()
            .map_err(|_| "warm slot does not hold exactly 96 bytes")?;
        let residual = s
            .warm_residual
            .take()
            .ok_or("warm slot has no residual backing")?;
        let mut full = codec::unpack_warm(packed);
        full[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES]
            .copy_from_slice(&residual);
        Self::set_warm_flag(&mut full, false);
        s.bytes = full.to_vec();
        s.tier = PhysicalTier::Hot;
        self.resident_bytes = self.resident_bytes.saturating_add(growth);
        Ok(())
    }

    pub fn evict_slot(&mut self, slot: usize) -> Result<(), &'static str> {
        let Some(s) = self.slots.get_mut(slot).and_then(Option::as_mut) else {
            return Err("slot absent");
        };
        if s.tier == PhysicalTier::Pinned {
            return Err("slot is pinned");
        }
        if s.tier == PhysicalTier::Cold {
            return Ok(());
        }
        let released = s.bytes.len();
        if released != codec::TILE_BYTES && released != WARM_PACKED_BYTES {
            return Err("resident slot has invalid representation length");
        }
        s.offloaded = Some(core::mem::take(&mut s.bytes).into_boxed_slice());
        s.tier = PhysicalTier::Cold;
        self.resident_bytes = self.resident_bytes.saturating_sub(released);
        self.evictions = self.evictions.saturating_add(1);
        Ok(())
    }

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
        let s = self.slots[slot].as_mut().expect("slot checked above");
        s.bytes = s
            .offloaded
            .take()
            .ok_or("cold slot has no backing")?
            .into_vec();
        s.tier = match s.bytes.len() {
            codec::TILE_BYTES => PhysicalTier::Hot,
            WARM_PACKED_BYTES => PhysicalTier::Warm,
            _ => return Err("cold backing has invalid representation length"),
        };
        self.resident_bytes = self.resident_bytes.saturating_add(s.bytes.len());
        Ok(())
    }

    pub fn score(&self, slot: usize, q_coarse: &[f32], q_sign: &[u64]) -> Option<f32> {
        let s = self.slots.get(slot)?.as_ref()?;
        let tile = match s.tier {
            PhysicalTier::Hot | PhysicalTier::Pinned => {
                let full: &[u8; codec::TILE_BYTES] = s.bytes.as_slice().try_into().ok()?;
                crate::mem::tile::SerializedTile(*full)
            }
            PhysicalTier::Warm => {
                let packed: &[u8; WARM_PACKED_BYTES] = s.bytes.as_slice().try_into().ok()?;
                crate::mem::tile::SerializedTile(codec::unpack_warm(packed))
            }
            PhysicalTier::Cold => return None,
        };
        tile.try_score(q_coarse, q_sign).ok()
    }

    /// Apply exactly the requested ECA action to real slots.
    ///
    /// `target_resident_bytes` is a byte target, not a token count. Demote and
    /// offload keep adapting the lowest-importance unpinned slots until the
    /// target is reached or no legal transition remains. Promote/restore pick
    /// the highest-importance eligible slot and still respect the hard budget.
    pub fn apply_action(
        &mut self,
        action: ActionRequest,
        target_resident_bytes: usize,
    ) -> Result<bool, &'static str> {
        let target = target_resident_bytes.min(self.config_budget_bytes());
        let before = self.resident_bytes;
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
            && self.resident_bytes > self.config_budget_bytes()
        {
            return Err("physical residency remains above hard budget");
        }
        Ok(self.resident_bytes != before)
    }

    pub fn step(&mut self) -> Result<Decision, &'static str> {
        let budget = self.config_budget_bytes();
        let obs = Observation::new(
            self.next_seq,
            self.resident_bytes as u64,
            budget as u64,
            self.slots.iter().flatten().count() as f64,
        );
        let mut decision = self.controller.step(obs)?;
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
        let verified = self.verify_accounting().is_ok()
            && (self.resident_bytes <= budget
                || !matches!(decision.action, ActionRequest::Demote | ActionRequest::Offload));
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
            .filter(|(_, s)| {
                matches!(
                    s.as_ref().map(|x| x.tier),
                    Some(PhysicalTier::Hot) | Some(PhysicalTier::Warm)
                )
            })
            .map(|(i, _)| i)
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
            .filter(|(_, s)| s.as_ref().is_some_and(|x| x.tier == tier))
            .map(|(i, _)| i)
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
        for s in self.slots.iter().flatten() {
            let valid = match s.tier {
                PhysicalTier::Hot | PhysicalTier::Pinned => {
                    s.bytes.len() == codec::TILE_BYTES && s.offloaded.is_none()
                }
                PhysicalTier::Warm => {
                    s.bytes.len() == WARM_PACKED_BYTES
                        && s.warm_residual.is_some()
                        && s.offloaded.is_none()
                }
                PhysicalTier::Cold => s.bytes.is_empty() && s.offloaded.is_some(),
            };
            if !valid {
                return Err("slot residency invariant violated");
            }
        }
        Ok(())
    }

    fn set_warm_flag(tile: &mut [u8; codec::TILE_BYTES], warm: bool) {
        let mut flags = u16::from_le_bytes([
            tile[codec::FLAGS_OFFSET],
            tile[codec::FLAGS_OFFSET + 1],
        ]);
        if warm {
            flags |= codec::FLAG_WARM;
        } else {
            flags &= !codec::FLAG_WARM;
        }
        tile[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2]
            .copy_from_slice(&flags.to_le_bytes());
    }

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
            if let Some(x) = self.slots.get_mut(slot).and_then(Option::as_mut) {
                x.importance += ((score - max_score) / temperature).exp() / sum;
            }
        }
    }

    pub fn pressure(&self) -> Pressure {
        Pressure::from_used(
            self.resident_bytes as u64,
            self.config_budget_bytes() as u64,
            self.config.watermarks,
        )
    }

    pub fn state_id(&self) -> StateId {
        self.controller.state_id()
    }

    pub fn last_reason(&self) -> Option<Reason> {
        self.last_reason
    }

    pub fn last_reason_code(&self) -> Option<&'static str> {
        self.last_reason.map(|r| r.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile(seed: u8) -> [u8; codec::TILE_BYTES] {
        let mut t = [0u8; codec::TILE_BYTES];
        t[..codec::LATENT_BYTES].fill(seed);
        t[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + RESIDUAL_BYTES].fill(seed);
        t[codec::SCALE_OFFSET..codec::SCALE_OFFSET + 4]
            .copy_from_slice(&1.0f32.to_le_bytes());
        t[codec::DYNAMIC_LAMBDA_OFFSET..codec::DYNAMIC_LAMBDA_OFFSET + 4]
            .copy_from_slice(&0.25f32.to_le_bytes());
        t[codec::GROUP_SCALES_OFFSET..codec::GROUP_SCALES_OFFSET + codec::N_GROUP_SCALES]
            .fill(255);
        t
    }

    #[test]
    fn warm_is_physically_smaller_and_losslessly_reversible() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let slot = cache.insert(make_tile(0));
        let q = [0.0f32; codec::D_C];
        let qs = [0u64; codec::RESIDUAL_WORDS];
        let hot = cache.score(slot, &q, &qs).unwrap();
        assert_eq!(hot, 64.0);

        cache.demote_slot(slot).unwrap();
        assert_eq!(cache.resident_bytes(), WARM_PACKED_BYTES);
        assert_eq!(cache.offloaded_bytes(), RESIDUAL_BYTES);
        assert_eq!(cache.score(slot, &q, &qs).unwrap(), 0.0);

        cache.promote_slot(slot).unwrap();
        assert_eq!(cache.resident_bytes(), codec::TILE_BYTES);
        assert_eq!(cache.offloaded_bytes(), 0);
        assert_eq!(cache.score(slot, &q, &qs).unwrap(), hot);
    }

    #[test]
    fn cold_is_reversible_and_identity_is_not_reused() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let slot = cache.insert(make_tile(7));
        cache.evict_slot(slot).unwrap();
        assert_eq!(cache.resident_bytes(), 0);
        assert_eq!(cache.offloaded_bytes(), codec::TILE_BYTES);
        assert_eq!(cache.counts(), (0, 0, 1, 0));
        let other = cache.insert(make_tile(8));
        assert_ne!(slot, other);
        cache.restore_slot(slot).unwrap();
        assert_eq!(cache.counts(), (2, 0, 0, 0));
    }

    #[test]
    fn apply_action_hits_requested_resident_target() {
        let mut cache = ElasticKvCache::new(1024, "test");
        for i in 0..4 {
            cache.insert(make_tile(i));
        }
        assert_eq!(cache.resident_bytes(), 512);
        assert!(cache.apply_action(ActionRequest::Demote, 384).unwrap());
        assert!(cache.resident_bytes() <= 384);
        assert!(cache.apply_action(ActionRequest::Offload, 192).unwrap());
        assert!(cache.resident_bytes() <= 192);
    }

    #[test]
    fn pinned_slots_are_never_demoted_or_offloaded() {
        let mut cache = ElasticKvCache::new(1024, "test");
        let pinned = cache.insert(make_tile(1));
        let other = cache.insert(make_tile(2));
        assert!(cache.pin(pinned));
        assert!(cache.demote_slot(pinned).is_err());
        assert!(cache.evict_slot(pinned).is_err());
        cache.apply_action(ActionRequest::Offload, 128).unwrap();
        assert_eq!(cache.counts(), (0, 0, 1, 1));
        assert!(cache.score(other, &[0.0; codec::D_C], &[0; codec::RESIDUAL_WORDS]).is_none());
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
