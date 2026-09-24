//! RPL-0 versioned replayable-state contract.
//!
//! This module defines only the identity and fail-closed validation boundary for
//! bounded replay research. It does not reconstruct KV state and does not claim
//! equivalence with DeepSeek SWA Bounded Replay.

use std::collections::BTreeSet;
use std::fmt;

use crate::representation_contract::RepresentationCapability;

/// Version of the SLHAv2 replayable-state contract.
pub const REPLAYABLE_STATE_CONTRACT_VERSION: u32 = 1;

/// Maximum UTF-8 bytes accepted for one replay identity.
pub const MAX_REPLAY_ID_BYTES: usize = 128;

/// Whether reconstruction is claimed exact or requires a quality verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayReconstruction {
    /// Owning runtime/model contract asserts exact reconstruction semantics.
    Exact,
    /// Reconstruction is approximate and must name a quality verifier.
    Approximate,
}

/// One versioned dependency required to reconstruct replayable state.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReplayDependency {
    id: String,
    generation: u64,
}

impl ReplayDependency {
    /// Construct one dependency identity.
    pub fn new(id: impl Into<String>, generation: u64) -> Result<Self, ReplayContractError> {
        let id = id.into();
        validate_id("dependency_id", &id)?;
        Ok(Self { id, generation })
    }

    /// Stable dependency identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Exact generation that must still be available at replay time.
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Exact source snapshot that may satisfy a replay contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplaySourceSnapshot {
    source_id: String,
    representation: RepresentationCapability,
    materialization_epoch: u64,
    logical_end_position: u64,
    dependencies: Vec<ReplayDependency>,
}

impl ReplaySourceSnapshot {
    /// Construct a source snapshot after validating identity/dependency shape.
    pub fn new(
        source_id: impl Into<String>,
        representation: RepresentationCapability,
        materialization_epoch: u64,
        logical_end_position: u64,
        dependencies: Vec<ReplayDependency>,
    ) -> Result<Self, ReplayContractError> {
        let source_id = source_id.into();
        validate_id("source_id", &source_id)?;
        validate_representation(&representation)?;
        validate_dependencies(&dependencies)?;
        Ok(Self {
            source_id,
            representation,
            materialization_epoch,
            logical_end_position,
            dependencies,
        })
    }
}

/// Versioned replay declaration for one short-horizon derived state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayableStateContract {
    state_id: String,
    source_id: String,
    representation: RepresentationCapability,
    materialization_epoch: u64,
    logical_end_position: u64,
    window_items: u32,
    reconstruction: ReplayReconstruction,
    verifier_id: Option<String>,
    dependencies: Vec<ReplayDependency>,
}

impl ReplayableStateContract {
    /// Construct one replayable-state contract.
    ///
    /// The replay window ends at `logical_end_position` and contains exactly
    /// `window_items` logical items. Approximate reconstruction requires an
    /// explicit verifier id. Dependencies are mandatory and unique.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state_id: impl Into<String>,
        source_id: impl Into<String>,
        representation: RepresentationCapability,
        materialization_epoch: u64,
        logical_end_position: u64,
        window_items: u32,
        reconstruction: ReplayReconstruction,
        verifier_id: Option<String>,
        dependencies: Vec<ReplayDependency>,
    ) -> Result<Self, ReplayContractError> {
        let state_id = state_id.into();
        let source_id = source_id.into();
        validate_id("state_id", &state_id)?;
        validate_id("source_id", &source_id)?;
        validate_representation(&representation)?;
        validate_dependencies(&dependencies)?;

        if window_items == 0 {
            return Err(ReplayContractError::ZeroWindow);
        }
        let available = logical_end_position
            .checked_add(1)
            .ok_or(ReplayContractError::PositionOverflow)?;
        if u64::from(window_items) > available {
            return Err(ReplayContractError::WindowBeforeOrigin {
                end_position: logical_end_position,
                window_items,
            });
        }

        if let Some(verifier) = verifier_id.as_deref() {
            validate_id("verifier_id", verifier)?;
        }
        if matches!(reconstruction, ReplayReconstruction::Approximate) && verifier_id.is_none() {
            return Err(ReplayContractError::ApproximateMissingVerifier);
        }

        Ok(Self {
            state_id,
            source_id,
            representation,
            materialization_epoch,
            logical_end_position,
            window_items,
            reconstruction,
            verifier_id,
            dependencies,
        })
    }

    /// Stable derived-state identity.
    pub fn state_id(&self) -> &str {
        &self.state_id
    }

    /// Number of recent logical items to replay.
    pub const fn window_items(&self) -> u32 {
        self.window_items
    }

    /// Inclusive logical start position of the replay window.
    pub fn replay_start_position(&self) -> Result<u64, ReplayContractError> {
        let available = self
            .logical_end_position
            .checked_add(1)
            .ok_or(ReplayContractError::PositionOverflow)?;
        available
            .checked_sub(u64::from(self.window_items))
            .ok_or(ReplayContractError::WindowBeforeOrigin {
                end_position: self.logical_end_position,
                window_items: self.window_items,
            })
    }

    /// Inclusive logical end position of the replay window.
    pub const fn replay_end_position(&self) -> u64 {
        self.logical_end_position
    }

    /// Reconstruction class.
    pub const fn reconstruction(&self) -> ReplayReconstruction {
        self.reconstruction
    }

    /// Domain-owned verifier required for approximate reconstruction.
    pub fn verifier_id(&self) -> Option<&str> {
        self.verifier_id.as_deref()
    }

    /// Exact dependencies bound into the replay contract.
    pub fn dependencies(&self) -> &[ReplayDependency] {
        &self.dependencies
    }

    /// Validate that an observed source snapshot is exactly the one this
    /// contract permits. No best-effort lineage substitution is allowed.
    pub fn validate_source(&self, observed: &ReplaySourceSnapshot) -> Result<(), ReplayContractError> {
        if observed.source_id != self.source_id {
            return Err(ReplayContractError::SourceIdMismatch);
        }
        if observed.representation != self.representation {
            return Err(ReplayContractError::RepresentationMismatch);
        }
        if observed.materialization_epoch != self.materialization_epoch {
            return Err(ReplayContractError::EpochMismatch {
                expected: self.materialization_epoch,
                observed: observed.materialization_epoch,
            });
        }
        if observed.logical_end_position != self.logical_end_position {
            return Err(ReplayContractError::LogicalPositionMismatch {
                expected: self.logical_end_position,
                observed: observed.logical_end_position,
            });
        }
        if observed.dependencies != self.dependencies {
            return Err(ReplayContractError::DependencyMismatch);
        }
        Ok(())
    }
}

fn validate_representation(
    representation: &RepresentationCapability,
) -> Result<(), ReplayContractError> {
    if representation.schema_version == 0 {
        return Err(ReplayContractError::ZeroRepresentationSchemaVersion);
    }
    Ok(())
}

fn validate_dependencies(dependencies: &[ReplayDependency]) -> Result<(), ReplayContractError> {
    if dependencies.is_empty() {
        return Err(ReplayContractError::NoDependencies);
    }
    let mut seen = BTreeSet::new();
    for dependency in dependencies {
        if !seen.insert(dependency.id.as_str()) {
            return Err(ReplayContractError::DuplicateDependency {
                id: dependency.id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_id(field: &'static str, value: &str) -> Result<(), ReplayContractError> {
    if value.trim().is_empty() {
        return Err(ReplayContractError::EmptyId { field });
    }
    if value.len() > MAX_REPLAY_ID_BYTES {
        return Err(ReplayContractError::IdTooLong {
            field,
            bytes: value.len(),
        });
    }
    Ok(())
}

/// Fail-closed replay-contract errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayContractError {
    EmptyId {
        field: &'static str,
    },
    IdTooLong {
        field: &'static str,
        bytes: usize,
    },
    ZeroRepresentationSchemaVersion,
    NoDependencies,
    DuplicateDependency {
        id: String,
    },
    ZeroWindow,
    PositionOverflow,
    WindowBeforeOrigin {
        end_position: u64,
        window_items: u32,
    },
    ApproximateMissingVerifier,
    SourceIdMismatch,
    RepresentationMismatch,
    EpochMismatch {
        expected: u64,
        observed: u64,
    },
    LogicalPositionMismatch {
        expected: u64,
        observed: u64,
    },
    DependencyMismatch,
}

impl fmt::Display for ReplayContractError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId { field } => write!(output, "{field} must not be empty"),
            Self::IdTooLong { field, bytes } => {
                write!(output, "{field} uses {bytes} bytes, maximum is {MAX_REPLAY_ID_BYTES}")
            }
            Self::ZeroRepresentationSchemaVersion => {
                write!(output, "representation schema version must be non-zero")
            }
            Self::NoDependencies => write!(output, "replay contract requires dependencies"),
            Self::DuplicateDependency { id } => {
                write!(output, "duplicate replay dependency {id:?}")
            }
            Self::ZeroWindow => write!(output, "replay window must be non-zero"),
            Self::PositionOverflow => write!(output, "logical replay position overflow"),
            Self::WindowBeforeOrigin {
                end_position,
                window_items,
            } => write!(
                output,
                "replay window of {window_items} items ends at {end_position} before logical origin"
            ),
            Self::ApproximateMissingVerifier => {
                write!(output, "approximate replay requires a verifier id")
            }
            Self::SourceIdMismatch => write!(output, "replay source id mismatch"),
            Self::RepresentationMismatch => write!(output, "replay representation mismatch"),
            Self::EpochMismatch { expected, observed } => write!(
                output,
                "replay materialization epoch mismatch: expected {expected}, observed {observed}"
            ),
            Self::LogicalPositionMismatch { expected, observed } => write!(
                output,
                "replay logical position mismatch: expected {expected}, observed {observed}"
            ),
            Self::DependencyMismatch => write!(output, "replay dependency identity mismatch"),
        }
    }
}

impl std::error::Error for ReplayContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn representation() -> RepresentationCapability {
        RepresentationCapability::new("slha.short-horizon-state", 1).unwrap()
    }

    fn dependencies() -> Vec<ReplayDependency> {
        vec![
            ReplayDependency::new("prompt.tokens", 7).unwrap(),
            ReplayDependency::new("model.state", 11).unwrap(),
        ]
    }

    fn contract() -> ReplayableStateContract {
        ReplayableStateContract::new(
            "decoder-local-state",
            "session-42",
            representation(),
            3,
            255,
            128,
            ReplayReconstruction::Exact,
            None,
            dependencies(),
        )
        .unwrap()
    }

    #[test]
    fn exact_contract_computes_bounded_recent_window() {
        let contract = contract();
        assert_eq!(contract.replay_start_position().unwrap(), 128);
        assert_eq!(contract.replay_end_position(), 255);
        assert_eq!(contract.window_items(), 128);
        assert_eq!(contract.reconstruction(), ReplayReconstruction::Exact);
    }

    #[test]
    fn approximate_replay_requires_named_verifier() {
        assert_eq!(
            ReplayableStateContract::new(
                "state",
                "source",
                representation(),
                1,
                63,
                32,
                ReplayReconstruction::Approximate,
                None,
                dependencies(),
            ),
            Err(ReplayContractError::ApproximateMissingVerifier)
        );
        assert!(
            ReplayableStateContract::new(
                "state",
                "source",
                representation(),
                1,
                63,
                32,
                ReplayReconstruction::Approximate,
                Some("slha.replay-quality.v1".into()),
                dependencies(),
            )
            .is_ok()
        );
    }

    #[test]
    fn source_snapshot_must_match_epoch_position_and_dependencies_exactly() {
        let contract = contract();
        let exact = ReplaySourceSnapshot::new(
            "session-42",
            representation(),
            3,
            255,
            dependencies(),
        )
        .unwrap();
        contract.validate_source(&exact).unwrap();

        let stale = ReplaySourceSnapshot::new(
            "session-42",
            representation(),
            2,
            255,
            dependencies(),
        )
        .unwrap();
        assert_eq!(
            contract.validate_source(&stale),
            Err(ReplayContractError::EpochMismatch {
                expected: 3,
                observed: 2,
            })
        );
    }

    #[test]
    fn dependencies_are_required_and_unique() {
        assert_eq!(
            ReplayableStateContract::new(
                "state",
                "source",
                representation(),
                1,
                15,
                8,
                ReplayReconstruction::Exact,
                None,
                Vec::new(),
            ),
            Err(ReplayContractError::NoDependencies)
        );
        let duplicate = vec![
            ReplayDependency::new("same", 1).unwrap(),
            ReplayDependency::new("same", 2).unwrap(),
        ];
        assert!(matches!(
            ReplayableStateContract::new(
                "state",
                "source",
                representation(),
                1,
                15,
                8,
                ReplayReconstruction::Exact,
                None,
                duplicate,
            ),
            Err(ReplayContractError::DuplicateDependency { .. })
        ));
    }

    #[test]
    fn window_cannot_extend_before_logical_origin() {
        assert_eq!(
            ReplayableStateContract::new(
                "state",
                "source",
                representation(),
                1,
                3,
                8,
                ReplayReconstruction::Exact,
                None,
                dependencies(),
            ),
            Err(ReplayContractError::WindowBeforeOrigin {
                end_position: 3,
                window_items: 8,
            })
        );
    }
}
