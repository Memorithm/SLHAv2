//! The generic Elastic Control Algorithm (ECA) controller.
//!
//! The controller is generic over the resource's observation type and the
//! backend that executes actions. It implements the full cycle:
//!
//! ```text
//! OBSERVE → MODEL → PREDICT → OPTIMIZE → ACT → VERIFY → LEARN
//! ```
//!
//! with hard constraints (`ElasticConstraints`) evaluated before any utility
//! optimization, hysteresis/anti-thrash, deterministic forecasting, and
//! decision traces for explainability and dry-run mode.

use crate::budget::BudgetTree;
use crate::decision::{Candidate, DecisionTrace, Rejection};
use crate::forecast::{steps_to_exhaustion, Forecast};
use crate::hysteresis::HysteresisGate;
use crate::pressure::{Pressure, PressureLevel, Watermarks};
use crate::reason::{code, Reason};
use crate::transaction::{run_transaction, Transaction, TransitionOutcome};
use crate::{ElasticResource, LogicalStep, StateId};

/// An observation of the resource state at one step.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Logical step (injected clock; deterministic mode never uses
    /// wall-clock time).
    pub step: LogicalStep,
    /// Bytes currently committed/used by the resource.
    pub used_bytes: u64,
    /// Total capacity the resource can commit (hard physical/format bound).
    pub capacity_bytes: u64,
    /// Application-level demand signal (tokens, queue length, …).
    pub demand: f64,
    /// Extra domain observations (latency, quality, …); optional.
    pub extra: f64,
}

impl Observation {
    /// Create an observation.
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

/// A backend action request produced by the controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionRequest {
    /// Nothing to do.
    None,
    /// Demote units (release bytes).
    Demote,
    /// Promote units (restore bytes).
    Promote,
    /// Offload units to a slower tier.
    Offload,
    /// Restore units from a slower tier.
    Restore,
    /// Prefetch units predicted to be needed.
    Prefetch,
    /// Rebalance between tiers.
    Rebalance,
    /// Explicit operator action (already applied; just verified).
    Operator(&'static str),
}

impl ActionRequest {
    /// Stable name for traces.
    pub fn name(&self) -> &'static str {
        match self {
            ActionRequest::None => "none",
            ActionRequest::Demote => "demote",
            ActionRequest::Promote => "promote",
            ActionRequest::Offload => "offload",
            ActionRequest::Restore => "restore",
            ActionRequest::Prefetch => "prefetch",
            ActionRequest::Rebalance => "rebalance",
            ActionRequest::Operator(a) => a,
        }
    }
}

/// The decision a controller step produces.
#[derive(Clone, Debug)]
pub struct Decision {
    /// The action to execute.
    pub action: ActionRequest,
    /// Full trace for explainability / journaling.
    pub trace: DecisionTrace,
}

/// How a backend executes an action. The controller never mutates resources
/// directly; the backend does, through this trait (optionally
/// transactionally).
pub trait ElasticBackend {
    /// Error type for backend operations.
    type Error: core::fmt::Debug;

    /// Execute a demote action (release a target amount).
    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;

    /// Execute a promote action (restore a target amount).
    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;

    /// Execute an offload action.
    fn offload(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;

    /// Execute a restore action.
    fn restore(&mut self, target_bytes: u64) -> Result<u64, Self::Error>;

    /// Execute a prefetch action.
    fn prefetch(&mut self) -> Result<(), Self::Error>;

    /// Execute a rebalance action.
    fn rebalance(&mut self) -> Result<(), Self::Error>;

    /// Verify that the post-action state satisfies the hard invariants
    /// (budget, residency, pins). Returns `Ok(true)` when verified.
    fn verify(&mut self, expected_used: u64) -> Result<bool, Self::Error>;
}

/// Configuration of the generic controller.
#[derive(Clone, Copy, Debug)]
pub struct ControllerConfig {
    /// Hysteresis watermarks.
    pub watermarks: Watermarks,
    /// Cooldown steps after an action (anti-thrash).
    pub cooldown_steps: u64,
    /// Minimum steps between two adaptations.
    pub min_interval_steps: u64,
    /// EWMA alpha for pressure smoothing.
    pub smooth_alpha: f64,
    /// EWMA beta for the trend.
    pub trend_beta: f64,
    /// Maximum absolute trend per step.
    pub max_trend: f64,
    /// If `true`, run in dry-run mode: produce decisions but never execute.
    pub dry_run: bool,
    /// If `true`, use transactional transitions (requires a transactional
    /// backend).
    pub transactional: bool,
    /// Forecast horizon in steps for exhaustion prediction.
    pub forecast_horizon: u64,
}

impl ControllerConfig {
    /// A deterministic default configuration.
    pub const fn standard() -> Self {
        Self {
            watermarks: Watermarks::standard(),
            cooldown_steps: 3,
            min_interval_steps: 2,
            smooth_alpha: 0.5,
            trend_beta: 0.2,
            max_trend: 0.1,
            dry_run: false,
            transactional: true,
            forecast_horizon: 4,
        }
    }

    /// Same as [`Self::standard`] but in dry-run (explain) mode.
    pub const fn explain() -> Self {
        Self {
            dry_run: true,
            ..Self::standard()
        }
    }
}

/// The generic Elastic controller.
///
/// `R` is the resource, `B` the backend. The controller is deterministic:
/// given the same observations, budgets, history and configuration it
/// produces the same decisions.
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
    /// Step of the last promotion (0 = never).
    last_promoted_step: LogicalStep,
    last_pressure: f64,
    history: alloc::vec::Vec<f64>,
}

impl<R: ElasticResource, B: ElasticBackend> ElasticController<R, B> {
    /// Create a controller with an initial budget tree. `budget_node` is the
    /// node this resource commits against (hierarchical coordination).
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

    /// The resource under control.
    pub fn resource(&self) -> &R {
        &self.resource
    }

    /// The backend (read access for telemetry).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The budget tree (read access for telemetry/reporting).
    pub fn budget(&self) -> &BudgetTree {
        &self.budget
    }

    /// The resource's budget node, if configured.
    pub fn budget_node(&self) -> Option<usize> {
        self.budget_node
    }

    /// Current smoothed pressure.
    pub fn pressure(&self) -> f64 {
        self.last_pressure
    }

    /// The recent pressure history (bounded; for LEARN/telemetry).
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Compute a deterministic state id from the current budget/history.
    pub fn state_id(&self) -> StateId {
        let mut h = crate::StateId::default();
        h = h
            .wrapping_mul(31)
            .wrapping_add(self.last_pressure.to_bits());
        h = h
            .wrapping_mul(31)
            .wrapping_add(self.budget.total_committed());
        h = h.wrapping_mul(31).wrapping_add(self.history.len() as u64);
        h
    }

    /// Run one control step. This is the ECA cycle entry point.
    ///
    /// Returns the decision; when not in dry-run, the backend executes it and
    /// the trace is completed with the actual outcome and verification.
    pub fn step(&mut self, obs: Observation) -> Result<Decision, B::Error> {
        // ── OBSERVE ──────────────────────────────────────────────────────────
        let raw_pressure =
            Pressure::from_used(obs.used_bytes, obs.capacity_bytes, self.config.watermarks);
        let smoothed = self.smoother.push(raw_pressure.value);
        self.last_pressure = smoothed;
        self.history.push(smoothed);
        if self.history.len() > 256 {
            self.history.remove(0);
        }

        // ── MODEL / PREDICT ──────────────────────────────────────────────────
        self.forecast.observe(smoothed);
        let forecast_pressure = self.forecast.forecast(self.config.forecast_horizon);
        let exhaustion = steps_to_exhaustion(&self.forecast, 1.0);

        let mut trace = DecisionTrace::new(obs.step, self.state_id());
        trace.measured_pressure = smoothed;
        trace.pressure_level = raw_pressure.level;
        trace.forecast_pressure = Some(forecast_pressure.clamp(0.0, 1.0));

        // ── OPTIMIZE ─────────────────────────────────────────────────────────
        // Hard constraints first: critical pressure overrides everything.
        let action = if raw_pressure.level == PressureLevel::Critical {
            trace.consider(
                Candidate::new(
                    "demote",
                    obs.used_bytes.saturating_sub(obs.capacity_bytes) as f64,
                    0.0,
                    0.0,
                    0.0,
                ),
                None,
            );
            trace.choose(
                ActionRequest::Demote.name(),
                Reason::new(code::HARD_CONSTRAINT),
                obs.used_bytes.saturating_sub(obs.capacity_bytes) as f64,
                0.0,
            );
            ActionRequest::Demote
        } else if let Some(steps) = exhaustion {
            if steps <= self.config.forecast_horizon && !self.gate.state().is_active() {
                // Predictive elasticity: act before exhaustion (but never
                // re-fire while the hysteresis gate is already active).
                trace.consider(
                    Candidate::new(
                        "demote",
                        (obs.used_bytes - obs.capacity_bytes.min(obs.used_bytes)) as f64,
                        0.0,
                        0.0,
                        0.0,
                    ),
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
                self.no_action(&mut trace, &obs);
                ActionRequest::None
            }
        } else if self.gate.update(smoothed, obs.step) {
            // The gate reports the action should be held (Active) or has just
            // been entered; execute only when the gate is Active (the very
            // first activation sets it Active), and never re-execute an
            // already-executed step.
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
                self.no_action(&mut trace, &obs);
                ActionRequest::None
            }
        } else if smoothed <= self.config.watermarks.low
            && self.gate.state() == crate::hysteresis::HysteresisState::Idle
        {
            // Low pressure: promote exactly once per pressure episode. The
            // cooldown (keyed on the gate's last-action step) prevents
            // re-promoting every step; the min-interval guard prevents
            // promote/demote oscillation.
            if self.gate.in_cooldown(obs.step) {
                self.no_action(&mut trace, &obs);
                ActionRequest::None
            } else if self.last_promoted_step == 0
                || obs.step.saturating_sub(self.last_promoted_step) > self.config.cooldown_steps * 2
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
                self.no_action(&mut trace, &obs);
                ActionRequest::None
            }
        } else {
            // Idle normal band: nothing to do.
            self.no_action(&mut trace, &obs);
            ActionRequest::None
        };

        let mut decision = Decision { action, trace };

        // ── ACT / VERIFY ─────────────────────────────────────────────────────
        if !self.config.dry_run && action != ActionRequest::None {
            let (outcome, verified) = self.execute(action, obs)?;
            decision.trace.complete(outcome, verified);
        }

        Ok(decision)
    }

    fn no_action(&mut self, trace: &mut DecisionTrace, _obs: &Observation) {
        trace.consider(
            Candidate::new("none", 0.0, 0.0, 0.0, 0.0),
            Some(Rejection::CostTooHigh),
        );
    }

    fn execute(
        &mut self,
        action: ActionRequest,
        obs: Observation,
    ) -> Result<(&'static str, bool), B::Error> {
        let target = obs
            .used_bytes
            .saturating_sub(obs.capacity_bytes.saturating_mul(3) / 4);
        let (outcome, expected_used) = match action {
            ActionRequest::Demote => {
                let released = self.backend.demote(target)?;
                ("demoted", obs.used_bytes.saturating_sub(released))
            }
            ActionRequest::Promote => {
                let restored = self.backend.promote(target)?;
                ("promoted", obs.used_bytes.saturating_add(restored))
            }
            ActionRequest::Offload => {
                let released = self.backend.offload(target)?;
                ("offloaded", obs.used_bytes.saturating_sub(released))
            }
            ActionRequest::Restore => {
                let restored = self.backend.restore(target)?;
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
        // Verify the post-action invariant (budget/residency).
        let verified = self.backend.verify(expected_used)?;
        Ok((outcome, verified))
    }

    /// Run a transactional transition through the backend, if supported.
    ///
    /// `T` is a transaction whose state is `S`; this is the generic
    /// transactional seam (see [`run_transaction`]).
    pub fn run_transaction<T: Transaction<State = S> + crate::transaction::RollbackAware, S>(
        &mut self,
        tx: &mut T,
        state: &mut S,
    ) -> TransitionOutcome {
        run_transaction(tx, state)
    }
}
