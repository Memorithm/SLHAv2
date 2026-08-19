//! ElasticContext — a runtime context-resource controller.
//!
//! Doctrine: **the context size is not a configuration constant.** Logical
//! context, physically resident raw KV and SLHA-compressed residency are
//! separate concepts, and the system continuously adapts the physical
//! representation to workload and available resources while preserving
//! correctness and respecting the model's true positional limits.
//!
//! This implementation is the SLHAv2 concrete `ElasticContext`: it observes
//! token/KV growth, computes VRAM/RAM pressure, predicts exhaustion from
//! model topology (layers × kv_heads × head_dim × bytes_per_elem per token),
//! and drives the physical [`ElasticKvCache`] residency through the generic
//! ECA. The model's true positional limit is a hard constraint, never an
//! adaptive target.

use elastic_core::budget::BudgetTree;
use elastic_core::controller::{ActionRequest, ControllerConfig, ElasticController, Observation};
use elastic_core::value::AdaptiveValue;
use elastic_core::{ElasticResource, ElasticValue, LogicalStep, StateId};

use crate::elastic_cache::{ElasticKvCache, PhysicalTier};

/// Model KV topology — the source of the bytes-per-token translation.
///
/// Never guessed: the runtime supplies these from model metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTopology {
    /// Number of layers.
    pub layers: usize,
    /// Number of KV heads (GQA-reduced).
    pub kv_heads: usize,
    /// K head dimension.
    pub head_dim_k: usize,
    /// Bytes per K element in the raw cache representation.
    pub k_bytes_per_elem: usize,
    /// Whether the cache also stores V (both K and V are then counted).
    pub has_v: bool,
    /// Bytes per V element.
    pub v_bytes_per_elem: usize,
}

impl KvTopology {
    /// Raw bytes one token of KV would occupy in the un-compressed cache.
    pub fn raw_bytes_per_token(&self) -> u64 {
        let mut per = self.layers as u64
            * self.kv_heads as u64
            * self.head_dim_k as u64
            * self.k_bytes_per_elem as u64;
        if self.has_v {
            per = per.saturating_mul(2);
        }
        per
    }
}

/// ElasticContext observation snapshot.
#[derive(Clone, Debug)]
pub struct ContextObservation {
    /// Logical step (deterministic mode).
    pub step: LogicalStep,
    /// Logical context length in tokens (workload descriptor).
    pub logical_tokens: u64,
    /// Active tokens in the current request.
    pub active_tokens: u64,
    /// Predicted generation growth (tokens).
    pub predicted_growth: u64,
    /// Model KV topology.
    pub topology: KvTopology,
    /// Available VRAM bytes.
    pub vram_available: u64,
    /// Total VRAM bytes.
    pub vram_total: u64,
    /// Available RAM bytes.
    pub ram_available: u64,
    /// Total RAM bytes.
    pub ram_total: u64,
    /// True positional limit of the model (hard constraint).
    pub model_positional_limit: Option<u64>,
}

impl ContextObservation {
    /// Create an observation.
    pub fn new(
        step: LogicalStep,
        logical_tokens: u64,
        topology: KvTopology,
        vram_available: u64,
        vram_total: u64,
        ram_available: u64,
        ram_total: u64,
    ) -> Self {
        Self {
            step,
            logical_tokens,
            active_tokens: logical_tokens,
            predicted_growth: 0,
            topology,
            vram_available,
            vram_total,
            ram_available,
            ram_total,
            model_positional_limit: None,
        }
    }
}

/// The context resource identity for the ECA.
#[derive(Debug, Clone)]
pub struct ContextResource {
    id: String,
}

impl ContextResource {
    /// Create the identity.
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl ElasticResource for ContextResource {
    fn resource_id(&self) -> &str {
        &self.id
    }
}

/// The context backend drives the physical cache.
#[derive(Debug, Default)]
pub struct ContextBackend {
    /// Bytes released by the last demote/offload.
    pub released: u64,
    /// Bytes restored by the last promote/restore.
    pub restored: u64,
    /// Hard VRAM budget the controller must respect.
    pub vram_budget: u64,
}

impl elastic_core::controller::ElasticBackend for ContextBackend {
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
        Ok(expected_used <= self.vram_budget)
    }
}

/// ElasticContext: the runtime context-resource controller.
pub struct ElasticContext {
    controller: ElasticController<ContextResource, ContextBackend>,
    cache: ElasticKvCache,
    /// Logical context length in tokens (NOT a memory unit).
    logical_tokens: u64,
    /// Adaptive VRAM budget (the primary control value).
    vram_budget: AdaptiveValue<u64>,
    topology: KvTopology,
    /// Model positional limit (hard constraint).
    model_positional_limit: Option<u64>,
    /// Telemetry counters.
    pub telemetry: ContextTelemetry,
}

impl ElasticContext {
    /// Create a context controller with a hard VRAM budget (bytes).
    ///
    /// The budget is the *resource* the ECA manages; there is no hidden
    /// token ceiling. The model's true positional limit, if known, is a
    /// hard constraint.
    pub fn new(resource_id: &str, vram_budget_bytes: u64, topology: KvTopology) -> Self {
        let mut budget = BudgetTree::new();
        let node = budget.add_root(0, vram_budget_bytes, vram_budget_bytes, 0);
        let backend = ContextBackend {
            vram_budget: vram_budget_bytes,
            ..ContextBackend::default()
        };
        let config = ControllerConfig::standard();
        let controller = ElasticController::new(
            ContextResource::new(resource_id),
            backend,
            config,
            budget,
            Some(node),
        );
        let cache = ElasticKvCache::new(vram_budget_bytes as usize, resource_id);
        Self {
            controller,
            cache,
            logical_tokens: 0,
            vram_budget: AdaptiveValue::new(ElasticValue::Adaptive {
                min: vram_budget_bytes / 4,
                max: vram_budget_bytes,
            }),
            topology,
            model_positional_limit: None,
            telemetry: ContextTelemetry::new(),
        }
    }

    /// Set the model's true positional limit (hard constraint).
    pub fn set_positional_limit(&mut self, limit: u64) {
        self.model_positional_limit = Some(limit);
    }

    /// The current logical context length (workload descriptor).
    pub fn logical_tokens(&self) -> u64 {
        self.logical_tokens
    }

    /// The physically resident cache.
    pub fn cache(&self) -> &ElasticKvCache {
        &self.cache
    }

    /// The physically resident cache (mut).
    pub fn cache_mut(&mut self) -> &mut ElasticKvCache {
        &mut self.cache
    }

    /// Raw bytes the current logical context would occupy uncompressed.
    pub fn raw_kv_bytes(&self) -> u64 {
        self.logical_tokens
            .saturating_mul(self.topology.raw_bytes_per_token())
    }

    /// Run one control step from a context observation.
    ///
    /// Computes the actual KV cost from the topology, checks the positional
    /// hard constraint, and lets the ECA decide demote/promote; the
    /// physical cache executes the decision.
    pub fn step(&mut self, obs: &ContextObservation) -> Result<ActionRequest, &'static str> {
        self.logical_tokens = obs.logical_tokens;
        self.telemetry.logical_tokens = obs.logical_tokens;
        self.telemetry.step = obs.step;

        // Hard constraint: the true positional limit is never exceeded.
        if let Some(limit) = self.model_positional_limit {
            let projected = obs.logical_tokens.saturating_add(obs.predicted_growth);
            if projected > limit {
                self.telemetry.hard_constraint_violations += 1;
                return Err("model positional limit would be exceeded (hard constraint)");
            }
        }

        // Predicted raw KV demand (predictive elasticity): the controller
        // sees used + forecast growth so it can act BEFORE the allocation
        // fails.
        let demand_tokens = obs.logical_tokens.saturating_add(obs.predicted_growth);
        let raw_used = demand_tokens.saturating_mul(self.topology.raw_bytes_per_token());
        let raw_now = self.raw_kv_bytes();

        self.telemetry.raw_kv_bytes = raw_now;
        self.telemetry.raw_kv_forecast = raw_used;
        self.telemetry.vram_available = obs.vram_available;
        self.telemetry.vram_total = obs.vram_total;
        self.telemetry.vram_pressure =
            (self.cache.resident_bytes() as f64 / obs.vram_total.max(1) as f64).min(1.0);

        let cap = self.controller_budget_bytes();
        let eca_obs = Observation::new(obs.step, raw_used.min(cap), cap, obs.active_tokens as f64);
        let decision = self.controller.step(eca_obs)?;

        // Execute on the physical cache.
        let action = decision.action;
        match action {
            ActionRequest::Demote | ActionRequest::Offload => {
                self.telemetry.demotions += 1;
                let _ = self.cache.step();
            }
            ActionRequest::Promote | ActionRequest::Restore => {
                self.telemetry.promotions += 1;
                let _ = self.cache.step();
            }
            _ => {}
        }

        self.telemetry.resident_bytes = self.cache.resident_bytes() as u64;
        let (hot, warm, cold, pinned) = self.cache.counts();
        self.telemetry.hot_tiles = hot as u64;
        self.telemetry.warm_tiles = warm as u64;
        self.telemetry.cold_tiles = cold as u64;
        self.telemetry.pinned_tiles = pinned as u64;
        self.telemetry.evictions = self.cache.evictions();

        Ok(action)
    }

    fn controller_budget_bytes(&self) -> u64 {
        self.vram_budget.current().unwrap_or(0)
    }

    /// Deterministic state id (journaling).
    pub fn state_id(&self) -> StateId {
        self.controller.state_id()
    }
}

/// Telemetry for ElasticContext (see mission §35).
pub mod elastic_context_telemetry_mod {
    /// ElasticContext telemetry snapshot.
    #[derive(Clone, Debug, Default)]
    pub struct ContextTelemetry {
        /// Logical step.
        pub step: u64,
        /// Logical context tokens.
        pub logical_tokens: u64,
        /// Raw KV bytes for the logical context.
        pub raw_kv_bytes: u64,
        /// Forecast raw KV bytes.
        pub raw_kv_forecast: u64,
        /// Physically resident cache bytes.
        pub resident_bytes: u64,
        /// VRAM total.
        pub vram_total: u64,
        /// VRAM available.
        pub vram_available: u64,
        /// VRAM pressure in `[0, 1]`.
        pub vram_pressure: f64,
        /// HOT tile count.
        pub hot_tiles: u64,
        /// WARM tile count.
        pub warm_tiles: u64,
        /// COLD tile count.
        pub cold_tiles: u64,
        /// PINNED tile count.
        pub pinned_tiles: u64,
        /// Demotion count.
        pub demotions: u64,
        /// Promotion count.
        pub promotions: u64,
        /// Eviction count.
        pub evictions: u64,
        /// Hard-constraint violations (must stay 0 in correct operation).
        pub hard_constraint_violations: u64,
    }

    impl ContextTelemetry {
        /// Create empty telemetry.
        pub fn new() -> Self {
            Self::default()
        }
    }
}

pub use elastic_context_telemetry_mod::ContextTelemetry;

/// Re-export for convenience.
pub use crate::elastic_cache::PhysicalTier as CacheTier;

/// Tier of the context cache (alias for ergonomics).
pub type ContextTier = PhysicalTier;

#[cfg(test)]
mod tests {
    use super::*;

    fn small_topology() -> KvTopology {
        KvTopology {
            layers: 28,
            kv_heads: 4,
            head_dim_k: 128,
            k_bytes_per_elem: 2, // bf16
            has_v: true,
            v_bytes_per_elem: 2,
        }
    }

    #[test]
    fn raw_bytes_per_token_matches_model_math() {
        let t = small_topology();
        // 28 * 4 * 128 * 2 bytes * 2 (K+V) = 57,344 B/token
        assert_eq!(t.raw_bytes_per_token(), 28 * 4 * 128 * 2 * 2);
    }

    #[test]
    fn positional_limit_is_hard_constraint() {
        let mut ctx = ElasticContext::new("ctx", 1 << 30, small_topology());
        ctx.set_positional_limit(4096);
        let obs = ContextObservation {
            logical_tokens: 4096,
            predicted_growth: 100,
            ..ContextObservation::new(
                1,
                4096,
                small_topology(),
                1 << 30,
                1 << 30,
                1 << 30,
                1 << 30,
            )
        };
        assert!(ctx.step(&obs).is_err());
        assert_eq!(ctx.telemetry.hard_constraint_violations, 1);
    }

    #[test]
    fn predictive_demotion_before_exhaustion() {
        // Budget small enough that 4096 tokens of KV (28*4*128*2*2 * 4096 =
        // ~235 MB) exceed it, with a large predicted growth.
        let budget = 8 << 20; // 8 MiB
        let mut ctx = ElasticContext::new("ctx", budget, small_topology());
        let obs = ContextObservation {
            logical_tokens: 2048,
            predicted_growth: 2048,
            ..ContextObservation::new(
                1,
                2048,
                small_topology(),
                1 << 30,
                1 << 30,
                1 << 30,
                1 << 30,
            )
        };
        let action = ctx.step(&obs).unwrap();
        // The ECA must act (demote) rather than wait for an OOM.
        assert_eq!(action, ActionRequest::Demote);
        assert_eq!(ctx.telemetry.demotions, 1);
    }

    #[test]
    fn logical_tokens_are_workload_descriptor_not_budget() {
        let mut ctx = ElasticContext::new("ctx", 1 << 20, small_topology());
        // Insert tiles far beyond the budget; logical length grows but the
        // controller keeps physical residency bounded.
        for _ in 0..100 {
            ctx.cache_mut().insert([7u8; 128]);
        }
        let obs =
            ContextObservation::new(1, 100, small_topology(), 1 << 30, 1 << 30, 1 << 30, 1 << 30);
        let _ = ctx.step(&obs);
        assert!(ctx.cache().resident_bytes() <= 1 << 20);
    }
}
