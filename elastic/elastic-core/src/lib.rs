//! elastic-core — the generic Elastic resource language kernel.
//!
//! This crate is the dependency-minimal heart of the Elastic architecture:
//! generic traits and deterministic types for constrained adaptive-resource
//! control. It has **zero external dependencies**, compiles on stable Rust
//! 1.89, contains **no unsafe code**, and is `no_std`-friendly by design
//! (it only uses `core`; the crate is unconditionally `no_std` so the
//! guarantee cannot rot).
//!
//! The dependency direction is strict:
//!
//! ```text
//! elastic-core  ←  elastic-runtime  ←  elastic  ←  SLHAv2  ←  integrations
//! ```
//!
//! Nothing in this crate knows about SLHAv2, CCOS, CUDA, llama.cpp or any
//! application. `ElasticXxx` is a naming convention only when a type actually
//! implements these traits.
//!
//! # The Elastic Control Algorithm (ECA)
//!
//! The generic cycle is:
//!
//! ```text
//! OBSERVE → MODEL → PREDICT → OPTIMIZE → ACT → VERIFY → LEARN → REPEAT
//! ```
//!
//! with hard constraints always taking priority over utility optimization.
//! The controller types live in [`controller`]; the state machine model for
//! tiers (HOT/WARM/COLD/…) in [`tiers`]; pressure in [`pressure`]; budgets in
//! [`budget`]; forecasting in [`forecast`]; decisions and their explanations
//! in [`decision`]; hysteresis/anti-thrash in [`hysteresis`]; transactional
//! transitions in [`transaction`]; and the adaptive-value concept
//! (`Fixed/Auto/Adaptive/Pinned`) in [`value`].
//!
//! Everything here is deterministic: the same observations, budgets, history,
//! policy and configuration produce the same decisions, because no wall clock
//! and no randomness exist in this crate.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod budget;
pub mod controller;
pub mod decision;
pub mod forecast;
pub mod hysteresis;
pub mod pressure;
pub mod tiers;
pub mod transaction;
pub mod value;

/// A stable, hashable state identifier used by decision traces and journals.
pub type StateId = u64;

/// A logical step counter used in deterministic mode instead of wall-clock
/// time. Controllers and journals prefer logical steps for reproducibility.
pub type LogicalStep = u64;

/// Reason codes for elastic decisions and transitions.
///
/// Codes are stable strings so journals, MCP surfaces and Enterprise audits
/// can rely on them across versions.
pub mod reason {
    /// Stable reason-code constants. New codes may be added; existing codes
    /// never change meaning.
    pub mod code {
        /// Pressure crossed the high watermark; demote low-value units.
        pub const PRESSURE_HIGH: &str = "pressure_high";
        /// Pressure fell below the low watermark; promote high-value units.
        pub const PRESSURE_LOW: &str = "pressure_low";
        /// Forecast predicts exhaustion before the next cycle; act early.
        pub const FORECAST_EXHAUSTION: &str = "forecast_exhaustion";
        /// A hard constraint (budget, pin, model limit) requires action.
        pub const HARD_CONSTRAINT: &str = "hard_constraint";
        /// A pinned unit was protected from adaptation.
        pub const PINNED_PROTECTED: &str = "pinned_protected";
        /// No feasible action exists within the hard constraints.
        pub const NO_FEASIBLE_ACTION: &str = "no_feasible_action";
        /// A transition was rejected before commit (prepare/validate failure).
        pub const TRANSITION_REJECTED: &str = "transition_rejected";
        /// A transition committed successfully.
        pub const TRANSITION_COMMITTED: &str = "transition_committed";
        /// An explicit operator action (not controller-driven).
        pub const OPERATOR: &str = "operator";
    }

    /// A structured reason: stable code plus optional human detail.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Reason {
        code: &'static str,
        detail: Option<&'static str>,
    }

    impl Reason {
        /// Create a reason from a stable code.
        pub const fn new(code: &'static str) -> Self {
            Self { code, detail: None }
        }

        /// Attach a static detail string (no allocation).
        pub const fn with_detail(code: &'static str, detail: &'static str) -> Self {
            Self {
                code,
                detail: Some(detail),
            }
        }

        /// The stable machine-readable code.
        pub const fn code(&self) -> &'static str {
            self.code
        }

        /// Optional human detail.
        pub const fn detail(&self) -> Option<&'static str> {
            self.detail
        }
    }
}

/// A named physical or logical resource (memory, context, queue, workers…).
pub trait ElasticResource: core::fmt::Debug {
    /// Stable, unique identifier for this resource instance.
    fn resource_id(&self) -> &str;
}

/// The capabilities a resource backend exposes to the controller.
pub trait ElasticCapabilities {
    /// Whether the backend can physically release bytes.
    fn can_release(&self) -> bool;
    /// Whether the backend can restore released units.
    fn can_restore(&self) -> bool;
    /// Whether the backend supports predictive (pre-emptive) action.
    fn can_predict(&self) -> bool;
    /// Whether transitions are transactional (prepare/commit/rollback).
    fn is_transactional(&self) -> bool;
}

/// Hard constraints that must hold before any candidate action is considered.
///
/// Hard constraints have priority over utility optimization. They are
/// evaluated in order; the first violated constraint rejects the action.
pub trait ElasticConstraints {
    /// Validate a candidate action against the hard constraints.
    /// `Ok(())` means the action is *permitted*; `Err(reason)` rejects it.
    fn validate_action(&self, action: &dyn core::any::Any) -> Result<(), ElasticViolation>;
}

/// A hard-constraint violation, with a stable code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElasticViolation {
    /// Stable violation code (see [`reason`] conventions).
    pub code: &'static str,
    /// Human-readable detail.
    pub detail: &'static str,
}

impl ElasticViolation {
    /// Create a violation.
    pub const fn new(code: &'static str, detail: &'static str) -> Self {
        Self { code, detail }
    }
}

/// Soft objectives the controller maximizes when all hard constraints pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElasticObjective {
    /// Maximize retained information/quality (e.g. keep high-value units).
    MaximizeRetainedInformation,
    /// Maximize throughput.
    MaximizeThroughput,
    /// Minimize latency.
    MinimizeLatency,
    /// Minimize memory footprint.
    MinimizeMemory,
    /// Minimize transition/adaptation cost.
    MinimizeTransitionCost,
}

/// The adaptive-value concept: who owns a value.
///
/// - `Fixed(x)` — the operator explicitly fixes it.
/// - `Auto` — the ECA owns the value **continuously**. This never means
///   "choose a default once".
/// - `Adaptive { min, max }` — the ECA chooses inside a legal range.
/// - `Pinned(x)` — the value is protected from adaptation that would violate
///   pinning semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElasticValue<T: Copy + PartialOrd> {
    /// Operator-fixed value.
    Fixed(T),
    /// Continuously controller-owned.
    Auto,
    /// Controller-chosen within an inclusive legal range.
    Adaptive {
        /// Lower bound of the legal range.
        min: T,
        /// Upper bound of the legal range.
        max: T,
    },
    /// Protected from adaptation.
    Pinned(T),
}

impl<T: Copy + PartialOrd> ElasticValue<T> {
    /// The currently effective value, if any (`Auto` has no value until the
    /// controller assigns one).
    pub fn current(&self) -> Option<T> {
        match *self {
            ElasticValue::Fixed(v) | ElasticValue::Pinned(v) => Some(v),
            ElasticValue::Adaptive { min, max } => Some(if min >= max { min } else { max }),
            ElasticValue::Auto => None,
        }
    }

    /// Whether a candidate value is legal under this value contract.
    pub fn allows(&self, candidate: T) -> bool {
        match *self {
            ElasticValue::Fixed(v) | ElasticValue::Pinned(v) => candidate == v,
            ElasticValue::Adaptive { min, max } => candidate >= min && candidate <= max,
            ElasticValue::Auto => true,
        }
    }

    /// Whether the controller may change this value at all.
    pub fn is_adaptive(&self) -> bool {
        matches!(*self, ElasticValue::Auto | ElasticValue::Adaptive { .. })
    }
}
