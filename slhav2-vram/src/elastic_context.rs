//! ElasticContext — runtime management of logical context and physical KV residency.
//!
//! The context size is not a configuration constant. Logical context,
//! physical KV residency and SLHA-compressed/offloaded residency are separate
//! concepts. The controller adapts the physical representation to workload
//! and available resources while respecting the model's true positional
//! limits.

use elastic_core::budget::BudgetTree;
use elastic_core::controller::{ActionRequest, ControllerConfig, ElasticController, Observation};
use elastic_core::value::AdaptiveValue;
use elastic_core::{ElasticResource, ElasticValue, LogicalStep, StateId};

use crate::elastic_cache::{ElasticKvCache, PhysicalTier};

/// Model KV topology supplied by the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTopology {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim_k: usize,
    pub k_bytes_per_elem: usize,
    pub has_v: bool,
    pub v_bytes_per_elem: usize,
}

impl KvTopology {
    /// Validate dimensions and element widths before they participate in
    /// memory-budget decisions.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.layers == 0 {
            return Err("KV topology must contain at least one layer");
        }
        if self.kv_heads == 0 {
            return Err("KV topology must contain at least one KV head");
        }
        if self.head_dim_k == 0 {
            return Err("KV topology head dimension must be non-zero");
        }
        if self.k_bytes_per_elem == 0 {
            return Err("KV topology K element width must be non-zero");
        }
        if self.has_v && self.v_bytes_per_elem == 0 {
            return Err("KV topology V element width must be non-zero when V is present");
        }
        Ok(())
    }

    /// Checked raw K/V bytes occupied by one logical token.
    ///
    /// K and V element widths are accounted independently. The current public
    /// topology uses the same head dimension for K and V; runtimes with a
    /// distinct V dimension must normalize that dimension before constructing
    /// this compatibility type.
    pub fn try_raw_bytes_per_token(&self) -> Result<u64, &'static str> {
        self.validate()?;
        let layers = u64::try_from(self.layers).map_err(|_| "KV layer count exceeds u64")?;
        let heads = u64::try_from(self.kv_heads).map_err(|_| "KV head count exceeds u64")?;
        let head_dim =
            u64::try_from(self.head_dim_k).map_err(|_| "KV head dimension exceeds u64")?;
        let k_width = u64::try_from(self.k_bytes_per_elem)
            .map_err(|_| "KV K element width exceeds u64")?;
        let v_width = u64::try_from(self.v_bytes_per_elem)
            .map_err(|_| "KV V element width exceeds u64")?;

        let vectors = layers
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or("KV vector geometry overflows u64")?;
        let k = vectors
            .checked_mul(k_width)
            .ok_or("KV K bytes per token overflow u64")?;
        let v = if self.has_v {
            vectors
                .checked_mul(v_width)
                .ok_or("KV V bytes per token overflow u64")?
        } else {
            0
        };
        k.checked_add(v)
            .ok_or("KV total bytes per token overflow u64")
    }

    /// Raw K/V bytes occupied by one logical token.
    ///
    /// # Panics
    /// Panics for an invalid or overflowing topology. Runtime/untrusted model
    /// metadata should use [`Self::try_raw_bytes_per_token`] instead.
    pub fn raw_bytes_per_token(&self) -> u64 {
        self.try_raw_bytes_per_token()
            .expect("valid non-overflowing KV topology")
    }
}

/// One runtime observation.
#[derive(Clone, Debug)]
pub struct ContextObservation {
    pub step: LogicalStep,
    pub logical_tokens: u64,
    pub active_tokens: u64,
    pub predicted_growth: u64,
    pub topology: KvTopology,
    pub vram_available: u64,
    pub vram_total: u64,
    pub ram_available: u64,
    pub ram_total: u64,
    pub model_positional_limit: Option<u64>,
}

impl ContextObservation {
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

    /// Validate externally supplied resource/model metadata before control.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.topology.validate()?;
        if self.vram_total != 0 && self.vram_available > self.vram_total {
            return Err("VRAM available bytes exceed VRAM total bytes");
        }
        if self.ram_total != 0 && self.ram_available > self.ram_total {
            return Err("RAM available bytes exceed RAM total bytes");
        }
        self.logical_tokens
            .checked_add(self.predicted_growth)
            .ok_or("logical token forecast overflows u64")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ContextResource {
    id: String,
}

impl ContextResource {
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl ElasticResource for ContextResource {
    fn resource_id(&self) -> &str {
        &self.id
    }
}

/// Decision-only backend for the context ECA.
///
/// `ElasticContext` runs this controller in dry-run mode and applies the
/// selected action exactly once through `ElasticKvCache::apply_action`, so a
/// second controller can never choose an opposite physical transition.
#[derive(Debug, Default)]
pub struct ContextBackend {
    pub released: u64,
    pub restored: u64,
    pub vram_budget: u64,
}

impl elastic_core::controller::ElasticBackend for ContextBackend {
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
        Ok(expected_used <= self.vram_budget)
    }
}

pub struct ElasticContext {
    controller: ElasticController<ContextResource, ContextBackend>,
    cache: ElasticKvCache,
    logical_tokens: u64,
    vram_budget: AdaptiveValue<u64>,
    topology: KvTopology,
    model_positional_limit: Option<u64>,
    pub telemetry: ContextTelemetry,
}

impl ElasticContext {
    pub fn new(resource_id: &str, vram_budget_bytes: u64, topology: KvTopology) -> Self {
        topology
            .validate()
            .expect("ElasticContext requires a valid KV topology");
        let mut budget = BudgetTree::new();
        let node = budget.add_root(0, vram_budget_bytes, vram_budget_bytes, 0);
        let backend = ContextBackend {
            vram_budget: vram_budget_bytes,
            ..ContextBackend::default()
        };
        let config = ControllerConfig {
            dry_run: true,
            ..ControllerConfig::standard()
        };
        let controller = ElasticController::new(
            ContextResource::new(resource_id),
            backend,
            config,
            budget,
            Some(node),
        );
        Self {
            controller,
            cache: ElasticKvCache::new(vram_budget_bytes as usize, resource_id),
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

    pub fn set_positional_limit(&mut self, limit: u64) {
        self.model_positional_limit = Some(limit);
    }

    pub fn logical_tokens(&self) -> u64 {
        self.logical_tokens
    }

    pub fn cache(&self) -> &ElasticKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut ElasticKvCache {
        &mut self.cache
    }

    /// Checked total raw KV bytes for the current logical context.
    pub fn try_raw_kv_bytes(&self) -> Result<u64, &'static str> {
        self.logical_tokens
            .checked_mul(self.topology.try_raw_bytes_per_token()?)
            .ok_or("logical raw KV bytes overflow u64")
    }

    /// Total raw KV bytes for the current logical context.
    ///
    /// # Panics
    /// Panics only when previously accepted topology/token state cannot be
    /// represented in `u64`; runtime control paths use checked arithmetic.
    pub fn raw_kv_bytes(&self) -> u64 {
        self.try_raw_kv_bytes()
            .expect("valid non-overflowing logical KV byte count")
    }

    /// Observe workload/resources, choose one ECA action and apply that exact
    /// action to the physical cache.
    pub fn step(&mut self, obs: &ContextObservation) -> Result<ActionRequest, &'static str> {
        obs.validate().map_err(|error| {
            self.telemetry.hard_constraint_violations =
                self.telemetry.hard_constraint_violations.saturating_add(1);
            error
        })?;

        let logical_forecast = obs
            .logical_tokens
            .checked_add(obs.predicted_growth)
            .ok_or("logical token forecast overflows u64")?;
        let effective_limit = match (self.model_positional_limit, obs.model_positional_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.telemetry.model_positional_limit = effective_limit;
        if effective_limit.is_some_and(|limit| logical_forecast > limit) {
            self.telemetry.hard_constraint_violations =
                self.telemetry.hard_constraint_violations.saturating_add(1);
            return Err("model positional limit would be exceeded (hard constraint)");
        }

        let bytes_per_token = obs.topology.try_raw_bytes_per_token()?;
        let raw_now = obs
            .logical_tokens
            .checked_mul(bytes_per_token)
            .ok_or("current raw KV byte count overflows u64")?;
        let raw_forecast = logical_forecast
            .checked_mul(bytes_per_token)
            .ok_or("forecast raw KV byte count overflows u64")?;
        let growth_bytes = obs
            .predicted_growth
            .checked_mul(bytes_per_token)
            .ok_or("predicted KV growth byte count overflows u64")?;
        let growth_shortfall = growth_bytes.saturating_sub(obs.vram_available);

        self.logical_tokens = obs.logical_tokens;
        self.topology = obs.topology;
        self.telemetry.step = obs.step;
        self.telemetry.logical_tokens = obs.logical_tokens;
        self.telemetry.raw_kv_bytes = raw_now;
        self.telemetry.raw_kv_forecast = raw_forecast;
        self.telemetry.vram_available = obs.vram_available;
        self.telemetry.vram_total = obs.vram_total;
        self.telemetry.ram_available = obs.ram_available;
        self.telemetry.ram_total = obs.ram_total;
        self.telemetry.vram_pressure = if obs.vram_total == 0 {
            0.0
        } else {
            (obs.vram_total - obs.vram_available) as f64 / obs.vram_total as f64
        };

        let budget = self.controller_budget_bytes();
        let resident = self.cache.resident_bytes() as u64;
        let resident_and_growth = resident.saturating_add(growth_shortfall);
        let system_pressure_used = scaled_system_pressure(
            obs.vram_total,
            obs.vram_available,
            budget,
        );
        // Both local cache demand and global VRAM pressure are first-class
        // controller inputs. The stronger one wins; logical raw history is
        // deliberately not confused with physical residency.
        let control_used = resident_and_growth.max(system_pressure_used);
        let eca_observation =
            Observation::new(obs.step, control_used, budget, obs.active_tokens as f64);
        let decision = self.controller.step(eca_observation)?;
        let action = decision.action;

        let recovery_target = budget.saturating_mul(3) / 4;
        let mut release_target = resident
            .saturating_sub(growth_shortfall)
            .min(recovery_target);
        if matches!(action, ActionRequest::Demote | ActionRequest::Offload)
            && resident > 0
            && release_target >= resident
        {
            // High global VRAM pressure may demand action even with no forecast
            // growth. Force at least one 32-byte WARM rung (or one eviction if
            // no HOT tile exists) instead of turning the decision into a no-op.
            release_target = resident.saturating_sub(32);
        }
        let target = match action {
            ActionRequest::Demote | ActionRequest::Offload => release_target as usize,
            _ => budget as usize,
        };
        let changed = self.cache.apply_action(action, target)?;

        if growth_shortfall > 0
            && matches!(action, ActionRequest::Demote | ActionRequest::Offload)
            && !changed
        {
            self.telemetry.hard_constraint_violations =
                self.telemetry.hard_constraint_violations.saturating_add(1);
            return Err("insufficient demotable residency for predicted VRAM growth");
        }

        if changed {
            match action {
                ActionRequest::Demote | ActionRequest::Offload => {
                    self.telemetry.demotions = self.telemetry.demotions.saturating_add(1)
                }
                ActionRequest::Promote | ActionRequest::Restore => {
                    self.telemetry.promotions = self.telemetry.promotions.saturating_add(1)
                }
                _ => {}
            }
        }
        self.refresh_cache_telemetry();
        Ok(action)
    }

    fn refresh_cache_telemetry(&mut self) {
        self.telemetry.resident_bytes = self.cache.resident_bytes() as u64;
        self.telemetry.offloaded_bytes = self.cache.offloaded_bytes() as u64;
        let (hot, warm, cold, pinned) = self.cache.counts();
        self.telemetry.hot_tiles = hot as u64;
        self.telemetry.warm_tiles = warm as u64;
        self.telemetry.cold_tiles = cold as u64;
        self.telemetry.pinned_tiles = pinned as u64;
        self.telemetry.evictions = self.cache.evictions();
        self.telemetry.compression_ratio = if self.telemetry.raw_kv_bytes == 0 {
            1.0
        } else {
            self.telemetry.resident_bytes as f64 / self.telemetry.raw_kv_bytes as f64
        };
    }

    fn controller_budget_bytes(&self) -> u64 {
        self.vram_budget.current().unwrap_or(0)
    }

    pub fn state_id(&self) -> StateId {
        self.controller.state_id()
    }
}

fn scaled_system_pressure(total: u64, available: u64, budget: u64) -> u64 {
    if total == 0 || budget == 0 {
        return 0;
    }
    let used = total - available;
    let numerator = (used as u128).saturating_mul(budget as u128);
    let scaled = numerator.div_ceil(total as u128);
    u64::try_from(scaled.min(budget as u128)).expect("clamped to u64 budget")
}

pub mod elastic_context_telemetry_mod {
    #[derive(Clone, Debug, Default)]
    pub struct ContextTelemetry {
        pub step: u64,
        pub logical_tokens: u64,
        pub raw_kv_bytes: u64,
        pub raw_kv_forecast: u64,
        pub resident_bytes: u64,
        pub offloaded_bytes: u64,
        pub vram_total: u64,
        pub vram_available: u64,
        pub ram_total: u64,
        pub ram_available: u64,
        pub vram_pressure: f64,
        pub compression_ratio: f64,
        pub model_positional_limit: Option<u64>,
        pub hot_tiles: u64,
        pub warm_tiles: u64,
        pub cold_tiles: u64,
        pub pinned_tiles: u64,
        pub demotions: u64,
        pub promotions: u64,
        pub evictions: u64,
        pub hard_constraint_violations: u64,
    }

    impl ContextTelemetry {
        pub fn new() -> Self {
            Self::default()
        }
    }
}

pub use crate::elastic_cache::PhysicalTier as CacheTier;
pub use elastic_context_telemetry_mod::ContextTelemetry;
pub type ContextTier = PhysicalTier;

#[cfg(test)]
mod tests {
    use super::*;

    fn small_topology() -> KvTopology {
        KvTopology {
            layers: 28,
            kv_heads: 4,
            head_dim_k: 128,
            k_bytes_per_elem: 2,
            has_v: true,
            v_bytes_per_elem: 2,
        }
    }

    fn tile() -> [u8; 128] {
        let mut tile = [0u8; 128];
        tile[96..100].copy_from_slice(&1.0f32.to_le_bytes());
        tile[120..128].fill(255);
        tile
    }

    #[test]
    fn raw_bytes_per_token_accounts_k_and_v_widths_independently() {
        let mut topology = small_topology();
        assert_eq!(topology.raw_bytes_per_token(), 28 * 4 * 128 * (2 + 2));
        topology.v_bytes_per_elem = 1;
        assert_eq!(topology.raw_bytes_per_token(), 28 * 4 * 128 * (2 + 1));
        topology.has_v = false;
        assert_eq!(topology.raw_bytes_per_token(), 28 * 4 * 128 * 2);
    }

    #[test]
    fn invalid_topology_and_resource_observations_fail_closed() {
        let mut bad = small_topology();
        bad.layers = 0;
        assert!(bad.try_raw_bytes_per_token().is_err());

        let mut ctx = ElasticContext::new("ctx", 1 << 20, small_topology());
        let obs = ContextObservation::new(1, 1, small_topology(), 1025, 1024, 1, 1);
        assert!(ctx.step(&obs).is_err());
        assert_eq!(ctx.telemetry.hard_constraint_violations, 1);
    }

    #[test]
    fn observation_limit_can_only_tighten_configured_limit() {
        let mut ctx = ElasticContext::new("ctx", 1 << 30, small_topology());
        ctx.set_positional_limit(8192);
        let obs = ContextObservation {
            model_positional_limit: Some(4096),
            logical_tokens: 4096,
            predicted_growth: 1,
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
        assert_eq!(ctx.telemetry.model_positional_limit, Some(4096));
    }

    #[test]
    fn configured_limit_is_hard_constraint() {
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
    fn predictive_action_changes_the_physical_cache_directly() {
        let budget = 8 << 20;
        let mut ctx = ElasticContext::new("ctx", budget, small_topology());
        for _ in 0..8 {
            ctx.cache_mut().insert(tile());
        }
        let before = ctx.cache().resident_bytes();
        let obs = ContextObservation {
            logical_tokens: 2048,
            predicted_growth: 2048,
            vram_available: 0,
            ..ContextObservation::new(
                1,
                2048,
                small_topology(),
                0,
                1 << 30,
                1 << 30,
                1 << 30,
            )
        };
        let action = ctx.step(&obs).unwrap();
        assert_eq!(action, ActionRequest::Demote);
        assert!(ctx.cache().resident_bytes() < before);
    }

    #[test]
    fn high_system_vram_pressure_drives_real_demotion_without_growth() {
        let budget = 1024;
        let mut ctx = ElasticContext::new("ctx", budget, small_topology());
        for _ in 0..4 {
            ctx.cache_mut().insert(tile());
        }
        let before = ctx.cache().resident_bytes();
        let obs = ContextObservation::new(
            1,
            4,
            small_topology(),
            100,
            1000,
            1000,
            2000,
        );
        let action = ctx.step(&obs).unwrap();
        assert_eq!(action, ActionRequest::Demote);
        assert!(ctx.cache().resident_bytes() < before);
        assert!((ctx.telemetry.vram_pressure - 0.9).abs() < 1e-12);
    }

    #[test]
    fn vram_pressure_uses_system_availability_not_cache_fraction() {
        let mut ctx = ElasticContext::new("ctx", 1 << 20, small_topology());
        let obs = ContextObservation::new(
            1,
            0,
            small_topology(),
            256,
            1024,
            512,
            1024,
        );
        let _ = ctx.step(&obs);
        assert!((ctx.telemetry.vram_pressure - 0.75).abs() < 1e-12);
    }
}
