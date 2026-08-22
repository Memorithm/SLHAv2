//! Decision traces: explainable elastic decisions.
//!
//! Every important Elastic decision should be explainable: state id,
//! measured/forecast pressure, hard constraints, candidates considered,
//! rejection reasons, chosen action, expected benefit/cost, actual outcome,
//! verification result. Traces support dry-run/explain mode (compute
//! decisions without mutating resources).

use crate::pressure::PressureLevel;
use crate::reason::Reason;
use crate::LogicalStep;

/// A candidate action considered by the controller.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Stable action identifier (backend-defined, e.g. "demote", "promote").
    pub action: &'static str,
    /// Expected benefit (bytes released / units gained).
    pub expected_benefit: f64,
    /// Expected transition cost.
    pub expected_cost: f64,
    /// Estimated quality risk in `[0, 1]` (0 = none).
    pub quality_risk: f64,
    /// Predicted reuse probability in `[0, 1]` (0 = never reused).
    pub reuse_probability: f64,
}

impl Candidate {
    /// Create a candidate.
    pub const fn new(
        action: &'static str,
        expected_benefit: f64,
        expected_cost: f64,
        quality_risk: f64,
        reuse_probability: f64,
    ) -> Self {
        Self {
            action,
            expected_benefit,
            expected_cost,
            quality_risk,
            reuse_probability,
        }
    }
}

/// Why a candidate was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// Violates a hard constraint (`reason` carries the violation code).
    HardConstraint(&'static str),
    /// Would oscillate (hysteresis/cooldown).
    Oscillation,
    /// Transition cost exceeds expected benefit.
    CostTooHigh,
    /// A pinned unit is protected.
    Pinned,
    /// Would exceed the model/format true limit.
    TrueLimit,
}

/// A full decision trace for one control step.
#[derive(Clone, Debug)]
pub struct DecisionTrace {
    /// Logical step.
    pub step: LogicalStep,
    /// State identifier/hash before the decision.
    pub state_id: crate::StateId,
    /// Measured pressure.
    pub measured_pressure: f64,
    /// Pressure level.
    pub pressure_level: PressureLevel,
    /// Forecast pressure (if any).
    pub forecast_pressure: Option<f64>,
    /// Candidates considered, in consideration order.
    pub candidates: alloc::vec::Vec<Candidate>,
    /// Rejection reasons per candidate (parallel to `candidates`; empty when
    /// accepted).
    pub rejections: alloc::vec::Vec<Option<Rejection>>,
    /// The chosen action, if any.
    pub chosen: Option<&'static str>,
    /// Stable reason for the decision.
    pub reason: Option<Reason>,
    /// Expected benefit of the chosen action.
    pub expected_benefit: f64,
    /// Expected transition cost.
    pub expected_cost: f64,
    /// Actual outcome after execution (filled by the caller).
    pub actual_outcome: Option<&'static str>,
    /// Verification result after execution (filled by the caller).
    pub verified: Option<bool>,
}

impl DecisionTrace {
    /// Create an empty trace for a step.
    pub fn new(step: LogicalStep, state_id: crate::StateId) -> Self {
        Self {
            step,
            state_id,
            measured_pressure: 0.0,
            pressure_level: PressureLevel::Low,
            forecast_pressure: None,
            candidates: alloc::vec::Vec::new(),
            rejections: alloc::vec::Vec::new(),
            chosen: None,
            reason: None,
            expected_benefit: 0.0,
            expected_cost: 0.0,
            actual_outcome: None,
            verified: None,
        }
    }

    /// Record a candidate and its rejection (or `None` when accepted).
    pub fn consider(&mut self, candidate: Candidate, rejection: Option<Rejection>) {
        self.candidates.push(candidate);
        self.rejections.push(rejection);
    }

    /// Set the chosen action.
    pub fn choose(
        &mut self,
        action: &'static str,
        reason: Reason,
        expected_benefit: f64,
        expected_cost: f64,
    ) {
        self.chosen = Some(action);
        self.reason = Some(reason);
        self.expected_benefit = expected_benefit;
        self.expected_cost = expected_cost;
    }

    /// Record the actual outcome and verification.
    pub fn complete(&mut self, actual_outcome: &'static str, verified: bool) {
        self.actual_outcome = Some(actual_outcome);
        self.verified = Some(verified);
    }

    /// Human-readable one-line explanation.
    pub fn explain(&self) -> alloc::string::String {
        use alloc::format;
        let mut s = alloc::string::String::new();
        s.push_str(&format!(
            "step {} state {} pressure {:.3} ({:?})",
            self.step, self.state_id, self.measured_pressure, self.pressure_level
        ));
        if let Some(fp) = self.forecast_pressure {
            s.push_str(&format!(" forecast {fp:.3}"));
        }
        match (self.chosen, self.reason) {
            (Some(a), Some(r)) => {
                s.push_str(&format!(" -> {a} [{}]", r.code()));
            }
            (Some(a), None) => s.push_str(&format!(" -> {a}")),
            (None, _) => s.push_str(" -> no action"),
        }
        if let Some(out) = self.actual_outcome {
            s.push_str(&format!(" outcome {out}"));
        }
        if let Some(v) = self.verified {
            s.push_str(&format!(" verified {v}"));
        }
        s
    }
}
