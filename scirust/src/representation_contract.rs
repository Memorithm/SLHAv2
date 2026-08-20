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
    id: String,
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

    /// Validated stable representation identifier.
    pub fn id(&self) -> &str {
        &self.id
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

/// Ordered key-materialization semantics declared by the model.
///
/// SLHAv2 intentionally treats transform/codec order as semantic. Declaring one
/// order does not imply equivalence with another order; a runtime must only use
/// the order associated with an admitted representation contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEncodingOrder {
    /// K remains in its canonical/raw domain; any positional/structural transform
    /// is deferred to execution.
    Raw,
    /// Apply the key transform before the numerical codec/quantizer.
    TransformThenCodec,
    /// Apply the numerical codec/domain conversion before the key transform.
    CodecThenTransform,
    /// Model declares a fused materialization with representation-specific
    /// semantics; no commutation with either ordered form is implied.
    FusedDeclared,
}

/// Source requirement for future rematerialization of an admitted target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RematerializationRequirement {
    /// The model does not impose a rematerialization-source requirement.
    Unspecified,
    /// A canonical/raw source must be retained so the runtime can produce a
    /// later representation epoch through an admitted re-encode.
    CanonicalSourceRequired,
    /// A trusted upstream source must remain available for recomputation.
    RecomputeSourceRequired,
}

/// Representation uniformity required inside one logical attention view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionViewContract {
    /// All pages in one logical attention view must share one representation
    /// contract and materialization epoch. This is the conservative default.
    HomogeneousRepresentationEpoch,
    /// The model explicitly admits per-page representation descriptors and the
    /// runtime/kernel may dispatch them independently.
    PerPageRepresentationAllowed,
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
    /// Ordered K materialization admitted for the target when this edge is used
    /// for a reusable KV representation.
    pub target_key_encoding: Option<KeyEncodingOrder>,
    /// Source state that must remain available for later rematerialization.
    pub rematerialization: RematerializationRequirement,
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
            target_key_encoding: None,
            rematerialization: RematerializationRequirement::Unspecified,
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

    /// Declare the exact ordered key-materialization semantics for the target.
    pub fn with_key_encoding(mut self, order: KeyEncodingOrder) -> Self {
        self.target_key_encoding = Some(order);
        self
    }

    /// Require a retained canonical/raw source for later epoch re-encoding.
    pub fn require_canonical_rematerialization_source(mut self) -> Self {
        self.rematerialization = RematerializationRequirement::CanonicalSourceRequired;
        self
    }

    /// Require a trusted recomputation source for later rematerialization.
    pub fn require_recompute_source(mut self) -> Self {
        self.rematerialization = RematerializationRequirement::RecomputeSourceRequired;
        self
    }
}

/// Complete representation capability/transition declaration for a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRepresentationContract {
    capabilities: BTreeSet<RepresentationCapability>,
    transitions: Vec<RepresentationTransitionPolicy>,
    attention_view: AttentionViewContract,
}

impl Default for ModelRepresentationContract {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRepresentationContract {
    /// Empty contract with conservative homogeneous representation/epoch views.
    pub const fn new() -> Self {
        Self {
            capabilities: BTreeSet::new(),
            transitions: Vec::new(),
            attention_view: AttentionViewContract::HomogeneousRepresentationEpoch,
        }
    }

    /// Declare a representation contract the model explicitly supports.
    pub fn declare(&mut self, capability: RepresentationCapability) {
        self.capabilities.insert(capability);
    }

    /// Explicitly admit per-page representation descriptors in one logical
    /// attention view. The runtime still has to prove that its kernel supports
    /// such dispatch; this declaration only states model-level admissibility.
    pub fn allow_per_page_representation_views(&mut self) {
        self.attention_view = AttentionViewContract::PerPageRepresentationAllowed;
    }

    /// Representation-uniformity rule declared by the model.
    pub const fn attention_view_contract(&self) -> AttentionViewContract {
        self.attention_view
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
        if matches!(
            transition.cache_scope,
            CacheScopeRequirement::FutureQueryIndependent
        ) && transition.target_key_encoding.is_none()
        {
            return Err(ContractError::ReusableKvMissingKeyEncoding);
        }
        if self
            .transitions
            .iter()
            .any(|existing| existing.from == transition.from && existing.to == transition.to)
        {
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
    /// A transition that promises reusable KV failed to declare the exact K
    /// materialization order.
    ReusableKvMissingKeyEncoding,
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
            Self::NoAllowedMechanism => write!(
                f,
                "representation transition must allow at least one mechanism"
            ),
            Self::ReusableKvMissingKeyEncoding => write!(
                f,
                "reusable KV transition must declare an exact key-materialization order"
            ),
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
    fn capability_identifier_is_validated_and_read_only() {
        assert!(matches!(
            RepresentationCapability::new("   ", 1),
            Err(ContractError::EmptyRepresentationId)
        ));
        let capability = RepresentationCapability::new("epg.so2", 1).unwrap();
        assert_eq!(capability.id(), "epg.so2");
    }

    #[test]
    fn conservative_attention_view_is_the_default() {
        let contract = ModelRepresentationContract::new();
        assert_eq!(
            contract.attention_view_contract(),
            AttentionViewContract::HomogeneousRepresentationEpoch
        );
    }

    #[test]
    fn per_page_representation_requires_explicit_model_declaration() {
        let mut contract = ModelRepresentationContract::new();
        contract.allow_per_page_representation_views();
        assert_eq!(
            contract.attention_view_contract(),
            AttentionViewContract::PerPageRepresentationAllowed
        );
    }

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
                    .require_reusable_kv()
                    .with_key_encoding(KeyEncodingOrder::TransformThenCodec)
                    .require_canonical_rematerialization_source(),
            )
            .unwrap();
        assert_eq!(contract.transitions().len(), 1);
        assert_eq!(
            contract.transitions()[0].cache_scope,
            CacheScopeRequirement::FutureQueryIndependent
        );
        assert_eq!(
            contract.transitions()[0].target_key_encoding,
            Some(KeyEncodingOrder::TransformThenCodec)
        );
        assert_eq!(
            contract.transitions()[0].rematerialization,
            RematerializationRequirement::CanonicalSourceRequired
        );
    }

    #[test]
    fn reusable_kv_requires_explicit_key_materialization_order() {
        let raw = RepresentationCapability::new("epg.raw", 1).unwrap();
        let transformed = RepresentationCapability::new("epg.token-stable", 1).unwrap();
        let mut contract = ModelRepresentationContract::new();
        contract.declare(raw.clone());
        contract.declare(transformed.clone());
        assert_eq!(
            contract.add_transition(
                RepresentationTransitionPolicy::new(raw, transformed)
                    .allow(RepresentationMechanism::Reencode)
                    .require_reusable_kv(),
            ),
            Err(ContractError::ReusableKvMissingKeyEncoding)
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
