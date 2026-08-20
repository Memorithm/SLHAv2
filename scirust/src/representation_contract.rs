//! Typed intermediate representation for SLHAv2 representational-resource policy.
//!
//! This module deliberately does not execute transitions. It describes what a
//! model declares as admissible so an ElasticXxx runtime can later compile or
//! validate the policy without SLHAv2 depending on that runtime implementation.

use std::collections::BTreeSet;
use std::fmt;

/// Stable representation contract identifier plus schema version.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepresentationCapability {
    /// Stable representation identifier, e.g. `epg.so2`.
    pub id: String,
    /// Contract/schema version understood by the model.
    pub schema_version: u32,
}

impl RepresentationCapability {
    /// Construct a non-empty capability identifier.
    pub fn new(id: impl Into<String>, schema_version: u32) -> Result<Self, ContractError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ContractError::EmptyRepresentationId);
        }
        Ok(Self { id, schema_version })
    }
}

/// Transition mechanism a policy may authorize.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RepresentationMechanism {
    /// Reuse materialized bytes as-is. Cross-contract use requires an explicit
    /// equivalence proof at runtime.
    Reinterpret,
    /// Transform an existing materialization into the target representation.
    Reencode,
    /// Regenerate the target from a trusted source representation.
    Recompute,
}

/// Cache-safety requirement attached to a target representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheScopeRequirement {
    /// No cache-specific restriction declared.
    Unspecified,
    /// Stored K must be independent of all future queries.
    FutureQueryIndependent,
}

/// One declared legal edge in the model's representation state graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationTransitionPolicy {
    /// Source representation contract.
    pub from: RepresentationCapability,
    /// Target representation contract.
    pub to: RepresentationCapability,
    /// Mechanisms the runtime is allowed to use for this edge.
    pub allowed_mechanisms: BTreeSet<RepresentationMechanism>,
    /// Whether committing this edge must advance the materialization epoch.
    pub require_epoch_advance: bool,
    /// Cache-safety requirement of the target materialization.
    pub cache_scope: CacheScopeRequirement,
}

impl RepresentationTransitionPolicy {
    /// Construct a transition edge with no authorized mechanism yet.
    pub fn new(from: RepresentationCapability, to: RepresentationCapability) -> Self {
        Self {
            from,
            to,
            allowed_mechanisms: BTreeSet::new(),
            require_epoch_advance: true,
            cache_scope: CacheScopeRequirement::Unspecified,
        }
    }

    /// Authorize a transition mechanism.
    pub fn allow(mut self, mechanism: RepresentationMechanism) -> Self {
        self.allowed_mechanisms.insert(mechanism);
        self
    }

    /// Require stored keys produced by the target to be future-query independent.
    pub fn require_reusable_kv(mut self) -> Self {
        self.cache_scope = CacheScopeRequirement::FutureQueryIndependent;
        self
    }
}

/// Complete representation capability/transition declaration for a model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelRepresentationContract {
    capabilities: BTreeSet<RepresentationCapability>,
    transitions: Vec<RepresentationTransitionPolicy>,
}

impl ModelRepresentationContract {
    /// Empty contract.
    pub const fn new() -> Self {
        Self {
            capabilities: BTreeSet::new(),
            transitions: Vec::new(),
        }
    }

    /// Declare a representation contract the model explicitly supports.
    pub fn declare(&mut self, capability: RepresentationCapability) {
        self.capabilities.insert(capability);
    }

    /// Add one legal transition edge.
    pub fn add_transition(
        &mut self,
        transition: RepresentationTransitionPolicy,
    ) -> Result<(), ContractError> {
        if !self.capabilities.contains(&transition.from) {
            return Err(ContractError::UndeclaredEndpoint(transition.from));
        }
        if !self.capabilities.contains(&transition.to) {
            return Err(ContractError::UndeclaredEndpoint(transition.to));
        }
        if transition.allowed_mechanisms.is_empty() {
            return Err(ContractError::NoAllowedMechanism);
        }
        if self.transitions.iter().any(|existing| {
            existing.from == transition.from && existing.to == transition.to
        }) {
            return Err(ContractError::DuplicateTransition {
                from: transition.from,
                to: transition.to,
            });
        }
        self.transitions.push(transition);
        Ok(())
    }

    /// Declared representation capabilities.
    pub fn capabilities(&self) -> impl Iterator<Item = &RepresentationCapability> {
        self.capabilities.iter()
    }

    /// Declared legal transition edges.
    pub fn transitions(&self) -> &[RepresentationTransitionPolicy] {
        &self.transitions
    }

    /// Whether a model explicitly supports a representation contract.
    pub fn supports(&self, capability: &RepresentationCapability) -> bool {
        self.capabilities.contains(capability)
    }
}

/// Invalid SLHAv2 representational-resource declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// Representation identifier is blank.
    EmptyRepresentationId,
    /// A transition references a capability not declared by the model.
    UndeclaredEndpoint(RepresentationCapability),
    /// A legal edge must authorize at least one materialization mechanism.
    NoAllowedMechanism,
    /// The same source/target edge was declared twice.
    DuplicateTransition {
        /// Source capability.
        from: RepresentationCapability,
        /// Target capability.
        to: RepresentationCapability,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRepresentationId => write!(f, "representation identifier must not be empty"),
            Self::UndeclaredEndpoint(cap) => write!(
                f,
                "transition endpoint {} v{} was not declared as a model capability",
                cap.id, cap.schema_version
            ),
            Self::NoAllowedMechanism => write!(f, "representation transition must allow at least one mechanism"),
            Self::DuplicateTransition { from, to } => write!(
                f,
                "duplicate representation transition {} v{} -> {} v{}",
                from.id, from.schema_version, to.id, to.schema_version
            ),
        }
    }
}

impl std::error::Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epg_like_contract_builds_a_typed_transition_graph() {
        let so2 = RepresentationCapability::new("epg.so2", 1).unwrap();
        let so4 = RepresentationCapability::new("epg.so4.structural", 1).unwrap();
        let mut contract = ModelRepresentationContract::new();
        contract.declare(so2.clone());
        contract.declare(so4.clone());
        contract
            .add_transition(
                RepresentationTransitionPolicy::new(so2, so4)
                    .allow(RepresentationMechanism::Reencode)
                    .allow(RepresentationMechanism::Recompute)
                    .require_reusable_kv(),
            )
            .unwrap();
        assert_eq!(contract.transitions().len(), 1);
        assert_eq!(
            contract.transitions()[0].cache_scope,
            CacheScopeRequirement::FutureQueryIndependent
        );
    }

    #[test]
    fn transition_to_undeclared_representation_is_rejected() {
        let so2 = RepresentationCapability::new("epg.so2", 1).unwrap();
        let so4 = RepresentationCapability::new("epg.so4", 1).unwrap();
        let mut contract = ModelRepresentationContract::new();
        contract.declare(so2.clone());
        let error = contract
            .add_transition(
                RepresentationTransitionPolicy::new(so2, so4)
                    .allow(RepresentationMechanism::Reencode),
            )
            .unwrap_err();
        assert!(matches!(error, ContractError::UndeclaredEndpoint(_)));
    }
}
