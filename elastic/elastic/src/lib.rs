//! elastic — the Elastic resource language facade.
//!
//! The intended public programming model: re-export the core types, the
//! runtime, and the macros so a consumer can write:
//!
//! ```ignore
//! use elastic::prelude::*;
//!
//! elastic_state! { ContextTier { Pinned, Hot, Warm, Cold, Evicted }
//!     transitions { Hot => Warm, Warm => Hot, Warm => Cold, Cold => Warm,
//!                   Cold => Evicted, Evicted => Cold, Pinned => !Evicted } }
//! ```
//!
//! The facade depends only on `elastic-core`, `elastic-runtime` and
//! `elastic-macros` — never on any application crate.

pub use elastic_core::*;
pub use elastic_macros::{elastic_budget, elastic_policy, elastic_state, elastic_target};
pub use elastic_runtime::*;

/// Common imports for Elastic consumers.
pub mod prelude {
    pub use crate::budget::{BudgetError, BudgetNode, BudgetTree};
    pub use crate::controller::{
        ActionRequest, ControllerConfig, Decision, ElasticBackend, ElasticController, Observation,
    };
    pub use crate::decision::{Candidate, DecisionTrace, Rejection};
    pub use crate::forecast::{steps_to_exhaustion, Forecast};
    pub use crate::hysteresis::{HysteresisGate, HysteresisState};
    pub use crate::pressure::{Pressure, PressureLevel, PressureSmoother, Watermarks};
    pub use crate::reason::{code, Reason};
    pub use crate::tiers::{Tier, TierError, TierLike, TierMachine, TierState, TierTransition};
    pub use crate::transaction::{
        run_transaction, PhaseOutcome, RollbackAware, Transaction, TransitionOutcome,
    };
    pub use crate::value::AdaptiveValue;
    pub use crate::{
        ElasticCapabilities, ElasticConstraints, ElasticObjective, ElasticResource, ElasticValue,
        ElasticViolation, LogicalStep, StateId,
    };
    pub use elastic_runtime::coordinator::ElasticCoordinator;
    pub use elastic_runtime::journal::{ElasticJournal, JournalEntry};
    pub use elastic_runtime::telemetry::ElasticTelemetry;
}

/// The adaptive-value holder.
pub type Adaptive<T> = value::AdaptiveValue<T>;
