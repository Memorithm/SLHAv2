//! Generic Elastic Control Algorithm (ECA).
//!
//! The controller implements the deterministic cycle
//! `OBSERVE → MODEL → PREDICT → OPTIMIZE → ACT → VERIFY → LEARN`.
//! Generic backend actions are verified but are not implicitly claimed to be
//! transactional: resources needing rollback must either expose an atomic
//! backend operation or use the explicit [`crate::transaction`] seam. The
//! SLHAv2 physical cache uses snapshot/rollback around multi-slot actions.

use crate::budget::BudgetTree;
use crate::decision::{Candidate, DecisionTrace, Rejection};
use crate::forecast::{steps_to_exhaustion, Forecast};
use crate::hysteresis::HysteresisGate;
use crate::pressure::{Pressure, PressureLevel, Watermarks};
use crate::reason::{code, Reason};
use crate::transaction::{run_transaction, Transaction, TransitionOutcome};
use crate::{ElasticResource, LogicalStep, StateId};

/// Resource state observed at one deterministic logical step.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Injected logical clock.
    pub step: LogicalStep,
    /// Bytes currently committed by the resource.
    pub used_bytes: u64,
    /// Runtime physical/format capacity.
    pub capacity_bytes: u64,
    /// Application demand signal (tokens, queue length, etc.).
    pub demand: f64,
    /// Optional domain-specific observation.
    pub extra: f64,
}

impl Observation {
    /// Construct an observation with `extra = 0`.
    pub const fn new(step: LogicalStep, used_bytes: u64, capacity_bytes: u64, demand: f64) -> Self {
        Self {
            step,
            used_bytes,
            capacity_bytes,
            demand,
            extra: 0.0,
        }
    }
}

/// Backend action selected by the ECA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionRequest {
    /// No adaptation.
    None,
    /// Release bytes by moving to a cheaper representation.
    Demote,
    /// Restore bytes to a richer representation.
    Promote,
    /// Move bytes to a slower/non-resident tier.
    Offload,
    /// Restore bytes from a slower/non-resident tier.
    Restore,
    /// Prefetch predicted-needed data.
    Prefetch,
    /// Rebalance resources between tiers.
    Rebalance,
    /// Explicit externally initiated action.
    Operator(&'static str),
}

impl ActionRequest {
    /// Stable action name for traces and telemetry.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Demote => "demote",
            Self::Promote => "promote",
            Self::Offload => "offload",
            Self::Restore => "restore",
            Self::Prefetch => "prefetch",
            Self::Rebalance => "rebalance",
            Self::Operator(name) => name,
        }
    }
}

/// Decision and complete explainability trace for one step.
#[derive(Clone, Debug)]
pub struct Decision {
    /// Selected action.
    pub action: ActionRequest,
    /// Decision trace.
    pub trace: DecisionTrace,
}

/// Backend contract for one ECA resource.
///
/// Each mutating method must either be atomic for that resource or be wrapped
/// in an explicit transaction by the caller. [`ElasticController::step`] never
/// fabricates rollback semantics that the backend cannot provide.
pub trait ElasticBackend {
    /// Backend error type.
    type Error: core::fmt::Debug;

    /// Release approximately `target_bytes` through demotion.
    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;
    /// Restore approximately `target_bytes` through promotion.
    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;
    /// Release approximately `target_bytes` through offload.
    fn offload(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;
    /// Restore approximately `target_bytes` from offload.
    fn restore(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;
    /// Prefetch predicted-needed data.
    fn prefetch(&mut self) -> Result<(), Self::Error>;
    /// Rebalance tiers/resources.
    fn rebalance(&mut self) -> Result<(), Self::Error>;
    /// Verify the backend after an action. `expected_used` is the controller's
    /// saturating estimate based on the backend-reported byte delta.
    fn verify(&mut self, expected_used: u64) -> Result<bool, Self::Error>;
}

/// Deterministic ECA configuration.
#[derive(Clone, Copy, Debug)]
pub struct ControllerConfig {
    /// Hysteresis pressure watermarks.
    pub watermarks: Watermarks,
    /// Cooldown after an action.
    pub cooldown_steps: u64,
    /// Minimum interval between adaptations.
    pub min_interval_steps: u64,
    /// EWMA alpha for pressure smoothing.
    pub smooth_alpha: f64,
    /// Trend beta for forecasting.
    pub trend_beta: f64,
    /// Maximum absolute forecast trend per step.
    pub max_trend: f64,
    /// Produce decisions without executing backend actions.
    pub dry_run: bool,
    /// Forecast horizon used for predictive exhaustion.
    pub forecast_horizon: u64,
}

impl ControllerConfig {
    /// Standard deterministic policy.
    pub const fn standard() -> Self {
        Self {
            watermarks: Watermarks::standard(),
            cooldown_steps: 3,
            min_interval_steps: 2,
            smooth_alpha: 0.5,
            trend_beta: 0.2,
            max_trend: 0.1,
            dry_run: false,
            forecast_horizon: 4,
        }
    }

    /// Standard policy in decision-only/explain mode.
    pub const fn explain() -> Self {
        Self {
            dry_run: true,
            ..Self::standard()
        }
    }
}

/// Deterministic controller for resource `R` and backend `B`.
pub struct ElasticController<R, B> {
    resource: R,
    backend: B,
    config: ControllerConfig,
    budget: BudgetTree,
    budget_node: Option<usize>,
    smoother: crate::pressure::PressureSmoother,
    forecast: Forecast,
    gate: HysteresisGate,
    last_action_step: LogicalStep,
    last_promoted_step: LogicalStep,
    last_pressure: f64,
    history: alloc::vec::Vec<f64>,
}

impl<R: ElasticResource, B: ElasticBackend> ElasticController<R, B> {
    /// Construct a controller. When `budget_node` exists, its hard limit
    /// tightens runtime capacity and its soft target becomes the preferred
    /// recovery occupancy.
    pub fn new(
        resource: R,
        backend: B,
        config: ControllerConfig,
        budget: BudgetTree,
        budget_node: Option<usize>,
    ) -> Self {
        Self {
            resource,
            backend,
            config,
            budget,
            budget_node,
            smoother: crate::pressure::PressureSmoother::new(config.smooth_alpha),
            forecast: Forecast::new(config.smooth_alpha, config.trend_beta, config.max_trend),
            gate: HysteresisGate::new(
                config.watermarks.high,
                config.watermarks.low,
                config.cooldown_steps,
                config.min_interval_steps,
            ),
            last_action_step: 0,
            last_promoted_step: 0,
            last_pressure: 0.0,
            history: alloc::vec::Vec::new(),
        }
    }

    /// Controlled resource.
    pub fn resource(&self) -> &R {
        &self.resource
    }

    /// Backend read access for telemetry.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Budget tree read access.
    pub fn budget(&self) -> &BudgetTree {
        &self.budget
    }

    /// Configured resource budget node.
    pub fn budget_node(&self) -> Option<usize> {
        self.budget_node
    }

    /// Current smoothed pressure.
    pub fn pressure(&self) -> f64 {
        self.last_pressure
    }

    /// Bounded recent pressure history.
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Deterministic state identifier.
    pub fn state_id(&self) -> StateId {
        let mut hash = crate::StateId::default();
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.last_pressure.to_bits());
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.budget.total_committed());
        hash.wrapping_mul(31)
            .wrapping_add(self.history.len() as u64)
    }

    /// Execute one complete ECA cycle.
    pub fn step(&mut self, obs: Observation) -> Result<Decision, B::Error> {
        let capacity = self.effective_capacity(obs.capacity_bytes);
        let raw_pressure = Pressure::from_used(obs.used_bytes, capacity, self.config.watermarks);
        let smoothed = self.smoother.push(raw_pressure.value);
        self.last_pressure = smoothed;
        self.history.push(smoothed);
        if self.history.len() > 256 {
            self.history.remove(0);
        }

        self.forecast.observe(smoothed);
        let forecast_pressure = self.forecast.forecast(self.config.forecast_horizon);
        let exhaustion = steps_to_exhaustion(&self.forecast, 1.0);

        let mut trace = DecisionTrace::new(obs.step, self.state_id());
        trace.measured_pressure = smoothed;
        trace.pressure_level = raw_pressure.level;
        trace.forecast_pressure = Some(forecast_pressure.clamp(0.0, 1.0));

        let action = if raw_pressure.level == PressureLevel::Critical {
            let bytes_over = obs.used_bytes.saturating_sub(capacity);
            trace.consider(
                Candidate::new("demote", bytes_over as f64, 0.0, 0.0, 0.0),
                None,
            );
            trace.choose(
                ActionRequest::Demote.name(),
                Reason::new(code::HARD_CONSTRAINT),
                bytes_over as f64,
                0.0,
            );
            ActionRequest::Demote
        } else if let Some(steps) = exhaustion {
            if steps <= self.config.forecast_horizon && !self.gate.state().is_active() {
                trace.consider(
                    Candidate::new("demote", self.release_target(obs.used_bytes, capacity) as f64, 0.0, 0.0, 0.0),
                    None,
                );
                trace.choose(
                    ActionRequest::Demote.name(),
                    Reason::new(code::FORECAST_EXHAUSTION),
                    0.0,
                    0.0,
                );
                ActionRequest::Demote
            } else {
                self.no_action(&mut trace);
                ActionRequest::None
            }
        } else if self.gate.update(smoothed, obs.step) {
            if self.gate.state().is_active()
                && (self.last_action_step != self.gate.last_action_step()
                    || self.last_action_step == 0)
            {
                self.last_action_step = obs.step;
                trace.choose(
                    ActionRequest::Demote.name(),
                    Reason::new(code::PRESSURE_HIGH),
                    0.0,
                    0.0,
                );
                ActionRequest::Demote
            } else {
                self.no_action(&mut trace);
                ActionRequest::None
            }
        } else if smoothed <= self.config.watermarks.low
            && self.gate.state() == crate::hysteresis::HysteresisState::Idle
        {
            if self.gate.in_cooldown(obs.step) {
                self.no_action(&mut trace);
                ActionRequest::None
            } else if self.last_promoted_step == 0
                || obs.step.saturating_sub(self.last_promoted_step)
                    > self.config.cooldown_steps.saturating_mul(2)
            {
                self.last_promoted_step = obs.step;
                self.gate.note_opposite_action(obs.step);
                self.last_action_step = obs.step;
                trace.choose(
                    ActionRequest::Promote.name(),
                    Reason::new(code::PRESSURE_LOW),
                    0.0,
                    0.0,
                );
                ActionRequest::Promote
            } else {
                self.no_action(&mut trace);
                ActionRequest::None
            }
        } else {
            self.no_action(&mut trace);
            ActionRequest::None
        };

        let mut decision = Decision { action, trace };
        if !self.config.dry_run && action != ActionRequest::None {
            let (outcome, verified) = self.execute(action, &obs, capacity)?;
            decision.trace.complete(outcome, verified);
        }
        Ok(decision)
    }

    fn effective_capacity(&self, runtime_capacity: u64) -> u64 {
        let budget_limit = self
            .budget_node
            .and_then(|index| self.budget.node(index).ok())
            .map(|node| node.hard_limit)
            .unwrap_or(runtime_capacity);
        runtime_capacity.min(budget_limit)
    }

    fn recovery_target(&self, capacity: u64) -> u64 {
        let configured = self
            .budget_node
            .and_then(|index| self.budget.node(index).ok())
            .map(|node| node.soft_target)
            .filter(|&target| target > 0)
            .unwrap_or_else(|| capacity.saturating_mul(3) / 4);
        configured.min(capacity)
    }

    fn release_target(&self, used: u64, capacity: u64) -> u64 {
        used.saturating_sub(self.recovery_target(capacity))
    }

    fn restore_target(&self, used: u64, capacity: u64) -> u64 {
        self.recovery_target(capacity).saturating_sub(used)
    }

    fn no_action(&self, trace: &mut DecisionTrace) {
        trace.consider(
            Candidate::new("none", 0.0, 0.0, 0.0, 0.0),
            Some(Rejection::CostTooHigh),
        );
    }

    fn execute(
        &mut self,
        action: ActionRequest,
        obs: &Observation,
        capacity: u64,
    ) -> Result<(&'static str, bool), B::Error> {
        let (outcome, expected_used) = match action {
            ActionRequest::Demote => {
                let released = self.backend.demote(self.release_target(obs.used_bytes, capacity))?;
                ("demoted", obs.used_bytes.saturating_sub(released))
            }
            ActionRequest::Promote => {
                let restored = self.backend.promote(self.restore_target(obs.used_bytes, capacity))?;
                ("promoted", obs.used_bytes.saturating_add(restored))
            }
            ActionRequest::Offload => {
                let released = self.backend.offload(self.release_target(obs.used_bytes, capacity))?;
                ("offloaded", obs.used_bytes.saturating_sub(released))
            }
            ActionRequest::Restore => {
                let restored = self.backend.restore(self.restore_target(obs.used_bytes, capacity))?;
                ("restored", obs.used_bytes.saturating_add(restored))
            }
            ActionRequest::Prefetch => {
                self.backend.prefetch()?;
                ("prefetched", obs.used_bytes)
            }
            ActionRequest::Rebalance => {
                self.backend.rebalance()?;
                ("rebalanced", obs.used_bytes)
            }
            ActionRequest::None => ("none", obs.used_bytes),
            ActionRequest::Operator(name) => (name, obs.used_bytes),
        };
        let verified = self.backend.verify(expected_used)?;
        Ok((outcome, verified))
    }

    /// Run an explicit prepare/validate/commit/rollback transition.
    pub fn run_transaction<T: Transaction<State = S> + crate::transaction::RollbackAware, S>(
        &mut self,
        tx: &mut T,
        state: &mut S,
    ) -> TransitionOutcome {
        run_transaction(tx, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Resource(&'static str);
    impl ElasticResource for Resource {
        fn resource_id(&self) -> &str {
            self.0
        }
    }

    #[derive(Default)]
    struct Backend {
        demoted: u64,
        promoted: u64,
        last_verified: u64,
    }

    impl ElasticBackend for Backend {
        type Error = ();

        fn demote(&mut self, bytes: u64) -> Result<u64, Self::Error> {
            self.demoted = self.demoted.saturating_add(bytes);
            Ok(bytes)
        }
        fn promote(&mut self, bytes: u64) -> Result<u64, Self::Error> {
            self.promoted = self.promoted.saturating_add(bytes);
            Ok(bytes)
        }
        fn offload(&mut self, bytes: u64) -> Result<u64, Self::Error> {
            self.demote(bytes)
        }
        fn restore(&mut self, bytes: u64) -> Result<u64, Self::Error> {
            self.promote(bytes)
        }
        fn prefetch(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn rebalance(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn verify(&mut self, expected: u64) -> Result<bool, Self::Error> {
            self.last_verified = expected;
            Ok(true)
        }
    }

    fn controller(capacity: u64, soft: u64) -> ElasticController<Resource, Backend> {
        let mut budget = BudgetTree::new();
        let node = budget.add_root(0, capacity, soft, 0);
        ElasticController::new(
            Resource("test"),
            Backend::default(),
            ControllerConfig::standard(),
            budget,
            Some(node),
        )
    }

    #[test]
    fn low_pressure_promotion_requests_nonzero_bytes() {
        let mut controller = controller(1000, 750);
        let decision = controller.step(Observation::new(10, 100, 1000, 1.0)).unwrap();
        assert_eq!(decision.action, ActionRequest::Promote);
        assert_eq!(controller.backend().promoted, 650);
        assert_eq!(controller.backend().last_verified, 750);
    }

    #[test]
    fn high_pressure_demotes_toward_soft_target() {
        let mut controller = controller(1000, 750);
        let decision = controller.step(Observation::new(1, 1100, 2000, 1.0)).unwrap();
        assert_eq!(decision.action, ActionRequest::Demote);
        assert_eq!(controller.backend().demoted, 350);
        assert_eq!(controller.backend().last_verified, 750);
    }

    #[test]
    fn budget_hard_limit_tightens_runtime_capacity() {
        let mut controller = controller(1000, 750);
        let decision = controller.step(Observation::new(1, 1100, 10_000, 1.0)).unwrap();
        assert_eq!(decision.action, ActionRequest::Demote);
    }

    #[test]
    fn explain_mode_never_mutates_backend() {
        let mut budget = BudgetTree::new();
        let node = budget.add_root(0, 1000, 750, 0);
        let mut controller = ElasticController::new(
            Resource("test"),
            Backend::default(),
            ControllerConfig::explain(),
            budget,
            Some(node),
        );
        let decision = controller.step(Observation::new(1, 1100, 1000, 1.0)).unwrap();
        assert_eq!(decision.action, ActionRequest::Demote);
        assert_eq!(controller.backend().demoted, 0);
    }
}
