//! Optional ElasticXxx BE14d bridge for the physical SLHAv2 KV cache.
//!
//! The bridge maps one concrete HOT SLHA slot to an ElasticXxx KV page and
//! implements a real HOT -> WARM representation transition. SLHAv2 retains
//! ownership of tile/codec/residency semantics; ElasticXxx owns the generic
//! validate/prepare/act/verify/commit-or-rollback transaction.
//!
//! This module is available only with the `elasticxxx` feature. The dependency
//! is pinned to the exact ElasticXxx revision reviewed for this contract.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use elasticxxx::kv::boolean_admission::KvCapacityObservationV1;
use elasticxxx::kv::{
    CapabilitySet, KeyEncodingPipeline, KeyTransformScope, KvPageDescriptor, KvPageId, KvPrecision,
    KvRecoverySource, KvResidency, KvTargetMaterialization, KvTransitionBackendV1,
    KvTransitionPlan, RepresentationEpoch, RepresentationId, RepresentationState,
    TransactionalKvPageV1, TransitionAttestations,
};
use elasticxxx::resource::{DimensionId, InvariantKind, LogicalResourceId};
use elasticxxx::{
    EirResource, InvariantCheck, Plan, RuntimeError, TransitionMechanism, VerificationResult,
};

use crate::codec;
use crate::elastic_cache::{ElasticKvCache, PhysicalTier};

/// Version of the SLHAv2 -> ElasticXxx KV consumer contract.
pub const SLHAV2_ELASTICXXX_KV_BRIDGE_V1: u16 = 1;

/// Exact ElasticXxx revision pinned by the optional Cargo dependencies.
pub const ELASTICXXX_BE14D_CONTRACT_REVISION: &str = "6a62519c2f18f0e0ad8428c390c4acb11f909ec4";

const REPRESENTATION_SCHEMA_V1: u32 = 1;
const BACKEND_NAME: &str = "slhav2-elastic-kv-cache-v1";
const SLHA_CACHE_RESIDENCY: &str = "slhav2-host-cache";

/// SLHAv2 semantics that cannot be inferred from physical tile bytes alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlhaKvSemanticContractV1 {
    /// Relationship of stored K to positional/structural transformation.
    pub key_transform_scope: KeyTransformScope,
    /// Ordered transform/codec materialization pipeline.
    pub key_encoding_pipeline: KeyEncodingPipeline,
    /// Independently retained source for future rematerialization, if any.
    pub recovery_source: KvRecoverySource,
}

impl SlhaKvSemanticContractV1 {
    /// Conservative contract for a token-stable SLHA tile with no independent
    /// canonical/raw rematerialization source outside the cache itself.
    #[must_use]
    pub const fn token_stable() -> Self {
        Self {
            key_transform_scope: KeyTransformScope::TokenStable,
            key_encoding_pipeline: KeyEncodingPipeline::TransformThenCodec,
            recovery_source: KvRecoverySource::None,
        }
    }
}

/// Shared physical cache handle. All external mutation through this handle and
/// every ElasticXxx source-check/mutation use the same mutex boundary.
#[derive(Clone)]
pub struct SlhaKvCacheHandleV1 {
    inner: Arc<Mutex<ElasticKvCache>>,
}

impl SlhaKvCacheHandleV1 {
    /// Wrap one real physical SLHAv2 cache.
    #[must_use]
    pub fn new(cache: ElasticKvCache) -> Self {
        Self {
            inner: Arc::new(Mutex::new(cache)),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, ElasticKvCache>, String> {
        self.inner
            .lock()
            .map_err(|_| "SLHAv2 KV cache mutex is poisoned".to_owned())
    }

    /// Read physical cache state under the same concurrency boundary used by
    /// the transaction backend.
    pub fn with_cache<R>(&self, f: impl FnOnce(&ElasticKvCache) -> R) -> Result<R, String> {
        let cache = self.lock()?;
        Ok(f(&cache))
    }

    /// Mutate the physical cache under the same concurrency boundary used by
    /// the transaction backend. This is intentionally exposed so the owning
    /// runtime can continue normal KV writes; slot-generation drift is checked
    /// by the ElasticXxx adapter before every transition.
    pub fn with_cache_mut<R>(&self, f: impl FnOnce(&mut ElasticKvCache) -> R) -> Result<R, String> {
        let mut cache = self.lock()?;
        Ok(f(&mut cache))
    }

    /// Publish current physical resident-budget headroom as source-bound
    /// ElasticXxx BE14d observation evidence.
    ///
    /// The observation reuses the exact resource identity owned by the physical
    /// cache. It does not predict future capacity or authorize a transition.
    pub fn capacity_observation(
        &self,
        observed_at: Instant,
    ) -> Result<KvCapacityObservationV1, String> {
        let cache = self.lock()?;
        let resource = LogicalResourceId::new(cache.resource_id())
            .map_err(|error| format!("invalid SLHAv2 KV resource id for ElasticXxx: {error}"))?;
        let free_capacity_bytes = u64::try_from(cache.free_bytes())
            .map_err(|_| "SLHAv2 free KV capacity does not fit u64".to_owned())?;
        Ok(KvCapacityObservationV1::measured(
            resource,
            free_capacity_bytes,
            observed_at,
        ))
    }
}

/// Concrete SLHAv2 implementation of the ElasticXxx BE14d KV backend contract.
pub struct SlhaKvTransitionBackendV1 {
    cache: SlhaKvCacheHandleV1,
    slot: usize,
    slot_generation: u64,
    source: KvPageDescriptor,
    target: KvPageDescriptor,
    source_hot_tile: [u8; codec::TILE_BYTES],
    codec_name: &'static str,
}

impl SlhaKvTransitionBackendV1 {
    /// Bind one current HOT physical slot to an exact ElasticXxx page identity.
    ///
    /// Page identity is the cache's monotonic slot generation, not the recyclable
    /// physical slot index. The binding therefore becomes stale immediately if
    /// the owning runtime writes a different logical KV value into that slot.
    pub fn bind_hot_slot(
        cache: SlhaKvCacheHandleV1,
        slot: usize,
        source_epoch: RepresentationEpoch,
        semantics: SlhaKvSemanticContractV1,
    ) -> Result<Self, String> {
        let (generation, tile, codec_name) = {
            let physical = cache.lock()?;
            if physical.tier(slot) != Some(PhysicalTier::Hot) {
                return Err("ElasticXxx SLHA binding requires a HOT source slot".to_owned());
            }
            let generation = physical
                .slot_generation(slot)
                .ok_or_else(|| "SLHAv2 source slot is absent".to_owned())?;
            let tile = physical
                .restorable_hot_tile(slot)
                .ok_or_else(|| "SLHAv2 HOT source tile is not reconstructable".to_owned())?;
            let codec_name = codec_name(&tile)?;
            (generation, tile, codec_name)
        };

        let target_epoch = source_epoch.next().map_err(|error| error.to_string())?;
        let source = descriptor(
            generation,
            codec_name,
            "hot",
            source_epoch,
            precision_for_codec(codec_name),
            semantics,
        )?;
        let target = descriptor(
            generation,
            codec_name,
            "warm",
            target_epoch,
            source.precision.clone(),
            semantics,
        )?;

        Ok(Self {
            cache,
            slot,
            slot_generation: generation,
            source,
            target,
            source_hot_tile: tile,
            codec_name,
        })
    }

    /// Physical cache handle used by the owning SLHAv2 runtime.
    #[must_use]
    pub fn cache_handle(&self) -> SlhaKvCacheHandleV1 {
        self.cache.clone()
    }

    /// Exact source descriptor bound to the current physical HOT slot.
    #[must_use]
    pub const fn source(&self) -> &KvPageDescriptor {
        &self.source
    }

    /// Exact target descriptor produced by the HOT -> WARM transition.
    #[must_use]
    pub const fn target(&self) -> &KvPageDescriptor {
        &self.target
    }

    /// Capability declaration implemented by this concrete backend.
    #[must_use]
    pub fn capabilities(&self) -> CapabilitySet {
        let mut capabilities = CapabilitySet::new();
        capabilities.insert(
            self.target.representation.id.clone(),
            self.target.representation.schema_version,
        );
        capabilities
    }

    /// Attestations implemented by this concrete backend.
    #[must_use]
    pub fn attestations(&self) -> TransitionAttestations {
        TransitionAttestations::none().attest_reencoder_available()
    }

    /// Build the authoritative ElasticXxx transition plan for this physical slot.
    pub fn transition_plan(&self) -> Result<KvTransitionPlan, String> {
        self.source
            .validate_reusable_representation_change(
                self.target.representation.clone(),
                TransitionMechanism::Reencode,
                &self.capabilities(),
                self.attestations(),
                KvTargetMaterialization::new(
                    self.target.key_transform_scope,
                    self.target.key_encoding_pipeline,
                    self.target.recovery_source,
                ),
            )
            .map_err(|error| error.to_string())
    }

    /// Bind this physical backend to the reviewed ElasticXxx transactional adapter.
    pub fn into_transactional(
        self,
        resource: &EirResource,
    ) -> Result<TransactionalKvPageV1<Self>, RuntimeError> {
        let source = self.source.clone();
        let transition = self
            .transition_plan()
            .map_err(RuntimeError::configuration)?;
        let capabilities = self.capabilities();
        let attestations = self.attestations();
        TransactionalKvPageV1::new(
            self,
            resource,
            source,
            transition,
            &capabilities,
            attestations,
        )
    }

    fn current_descriptor_locked(
        &self,
        cache: &ElasticKvCache,
    ) -> Result<KvPageDescriptor, String> {
        let generation = cache
            .slot_generation(self.slot)
            .ok_or_else(|| "SLHAv2 KV slot disappeared".to_owned())?;
        if generation != self.slot_generation {
            return Err("SLHAv2 KV slot was recycled for another logical page".to_owned());
        }
        let restorable = cache
            .restorable_hot_tile(self.slot)
            .ok_or_else(|| "SLHAv2 KV slot is not losslessly reconstructable".to_owned())?;
        if restorable != self.source_hot_tile {
            return Err("SLHAv2 KV slot contents drifted from the bound source page".to_owned());
        }
        if codec_name(&restorable)? != self.codec_name {
            return Err("SLHAv2 KV codec drifted from the bound representation".to_owned());
        }

        match cache.tier(self.slot) {
            Some(PhysicalTier::Hot) => Ok(self.source.clone()),
            Some(PhysicalTier::Warm) => {
                if cache.slot_backing_bytes(self.slot) != Some(codec::RESIDUAL_WORDS * 8) {
                    return Err("SLHAv2 WARM slot lacks exact residual backing".to_owned());
                }
                Ok(self.target.clone())
            }
            Some(PhysicalTier::Cold) => {
                Err("SLHAv2 KV slot is COLD, outside the bound HOT/WARM contract".to_owned())
            }
            Some(PhysicalTier::Pinned) => {
                Err("SLHAv2 KV slot became PINNED, outside the bound transition".to_owned())
            }
            None => Err("SLHAv2 KV slot is absent".to_owned()),
        }
    }

    fn physical_matches(&self, expected: &KvPageDescriptor) -> Result<bool, String> {
        let cache = self.cache.lock()?;
        Ok(self.current_descriptor_locked(&cache)? == *expected)
    }
}

impl KvTransitionBackendV1 for SlhaKvTransitionBackendV1 {
    fn name(&self) -> &str {
        BACKEND_NAME
    }

    fn read_page(&self, page: KvPageId) -> Result<KvPageDescriptor, String> {
        if page != self.source.page {
            return Err("ElasticXxx requested a foreign SLHAv2 KV page".to_owned());
        }
        let cache = self.cache.lock()?;
        self.current_descriptor_locked(&cache)
    }

    fn validate_transition(
        &self,
        runtime_plan: &Plan,
        source: &KvPageDescriptor,
        target: &KvPageDescriptor,
        transition: &KvTransitionPlan,
    ) -> Result<Vec<InvariantCheck>, String> {
        if source != &self.source
            || target != &self.target
            || transition != &self.transition_plan()?
        {
            return Err(
                "ElasticXxx transition differs from the bound SLHAv2 physical contract".to_owned(),
            );
        }
        if !self.physical_matches(source)? {
            return Err("SLHAv2 source physical state drifted before validation".to_owned());
        }

        let mut checks = Vec::new();
        for invariant in runtime_plan.resource.invariants() {
            if invariant
                .scope()
                .is_some_and(|scope| scope != &DimensionId::REPRESENTATION)
            {
                continue;
            }
            let (holds, detail) = match invariant.kind() {
                InvariantKind::PreserveContents => (
                    true,
                    "WARM retains the exact residual backing and reconstructs the bound HOT tile byte-for-byte",
                ),
                InvariantKind::PreserveIdentity => (
                    true,
                    "SLHAv2 slot generation remains the same logical KV page across HOT/WARM",
                ),
                InvariantKind::UpholdContract(_) => (
                    false,
                    "SLHAv2 BE14d v1 does not implement this external invariant contract",
                ),
            };
            checks.push(InvariantCheck::new(
                invariant.clone(),
                holds,
                Some(detail.to_owned()),
            ));
        }
        Ok(checks)
    }

    fn prepare_transition(
        &mut self,
        source: &KvPageDescriptor,
        target: &KvPageDescriptor,
        transition: &KvTransitionPlan,
    ) -> Result<(), String> {
        if source != &self.source
            || target != &self.target
            || transition != &self.transition_plan()?
        {
            return Err("SLHAv2 prepare received a foreign KV transition".to_owned());
        }
        if !self.physical_matches(source)? {
            return Err("SLHAv2 source state drifted before prepare".to_owned());
        }
        Ok(())
    }

    fn apply_transition_if_source(
        &mut self,
        source: &KvPageDescriptor,
        target: &KvPageDescriptor,
        transition: &KvTransitionPlan,
    ) -> Result<(), String> {
        if source != &self.source
            || target != &self.target
            || transition != &self.transition_plan()?
        {
            return Err("SLHAv2 apply received a foreign KV transition".to_owned());
        }

        let mut cache = self.cache.lock()?;
        if self.current_descriptor_locked(&cache)? != *source {
            return Err("SLHAv2 source changed at the atomic mutation boundary".to_owned());
        }
        cache
            .demote_slot(self.slot)
            .map_err(|error| format!("SLHAv2 HOT->WARM mutation failed: {error}"))?;

        match self.current_descriptor_locked(&cache) {
            Ok(current) if current == *target => Ok(()),
            result => {
                let verify_error = match result {
                    Ok(_) => "SLHAv2 HOT->WARM mutation produced the wrong target".to_owned(),
                    Err(error) => error,
                };
                let rollback = cache.promote_slot(self.slot);
                if rollback.is_err() {
                    return Err(format!(
                        "{verify_error}; immediate physical rollback also failed: {}",
                        rollback.err().unwrap_or("unknown rollback failure")
                    ));
                }
                Err(verify_error)
            }
        }
    }

    fn verify_transition(
        &self,
        target: &KvPageDescriptor,
        transition: &KvTransitionPlan,
    ) -> Result<VerificationResult, String> {
        if target != &self.target || transition != &self.transition_plan()? {
            return Ok(VerificationResult::Fail {
                detail: "SLHAv2 verification target differs from the bound transition".into(),
            });
        }
        if self.physical_matches(target)? {
            Ok(VerificationResult::Pass)
        } else {
            Ok(VerificationResult::Fail {
                detail: "SLHAv2 physical WARM representation failed exact content verification"
                    .into(),
            })
        }
    }

    fn restore_page(&mut self, source: &KvPageDescriptor) -> Result<(), String> {
        if source != &self.source {
            return Err("SLHAv2 rollback requested a foreign source page".to_owned());
        }
        let mut cache = self.cache.lock()?;
        let current = self.current_descriptor_locked(&cache)?;
        if current == *source {
            return Ok(());
        }
        if current != self.target {
            return Err("SLHAv2 rollback encountered an unrecognized physical state".to_owned());
        }
        cache
            .promote_slot(self.slot)
            .map_err(|error| format!("SLHAv2 WARM->HOT rollback failed: {error}"))?;
        if self.current_descriptor_locked(&cache)? != *source {
            return Err("SLHAv2 rollback did not restore the exact source page".to_owned());
        }
        Ok(())
    }
}

fn descriptor(
    generation: u64,
    codec_name: &'static str,
    tier: &'static str,
    epoch: RepresentationEpoch,
    precision: KvPrecision,
    semantics: SlhaKvSemanticContractV1,
) -> Result<KvPageDescriptor, String> {
    let id = RepresentationId::new(format!("slhav2.{codec_name}.{tier}"))
        .map_err(|error| error.to_string())?;
    Ok(KvPageDescriptor {
        page: KvPageId::new(generation),
        representation: RepresentationState::new(id, REPRESENTATION_SCHEMA_V1, epoch),
        precision,
        residency: KvResidency::Custom(SLHA_CACHE_RESIDENCY.to_owned()),
        key_transform_scope: semantics.key_transform_scope,
        key_encoding_pipeline: semantics.key_encoding_pipeline,
        recovery_source: semantics.recovery_source,
    })
}

fn precision_for_codec(codec_name: &str) -> KvPrecision {
    match codec_name {
        "int4" => KvPrecision::Int4,
        other => KvPrecision::Custom(format!("slhav2.{other}")),
    }
}

fn codec_name(tile: &[u8; codec::TILE_BYTES]) -> Result<&'static str, String> {
    let flags = codec::read_u16_le(tile, codec::FLAGS_OFFSET).map_err(str::to_owned)?;
    const ALLOWED_FLAGS: u16 = codec::FLAG_WARM
        | codec::FLAG_NF4
        | codec::FLAG_MIXED
        | codec::FLAG_TQ3
        | codec::FLAG_TQ3_NOCORR
        | codec::FLAG_MIX3;
    let unknown = flags & !ALLOWED_FLAGS;
    if unknown != 0 {
        return Err(format!(
            "SLHAv2 tile carries unsupported flag bits 0x{unknown:04x}"
        ));
    }
    codec::validate_codec(flags).map_err(|error| error.to_string())?;
    let no_corr = codec::has_flag(flags, codec::FLAG_TQ3_NOCORR);
    let name = if codec::has_flag(flags, codec::FLAG_MIXED) {
        "mixed"
    } else if codec::has_flag(flags, codec::FLAG_TQ3) {
        if no_corr {
            "tq3-nocorr"
        } else {
            "tq3"
        }
    } else if codec::has_flag(flags, codec::FLAG_MIX3) {
        if no_corr {
            "mix3-nocorr"
        } else {
            "mix3"
        }
    } else if codec::has_flag(flags, codec::FLAG_NF4) {
        "nf4"
    } else {
        "int4"
    };
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticxxx::kv::boolean_admission::{
        BooleanKvCapacityPreflightControllerV1, BooleanKvTransitionPreflightV2,
        KvCapacityObservationV1,
    };
    use elasticxxx::resource::{
        AdmissibleTransition, CapabilityRequirement, Invariant, LogicalResourceId,
        ObservationSignalId, ResourceClassId, ResourceSpec,
    };
    use elasticxxx::{lower, FirstGroundedPlanner, Runtime, RuntimeConfig, RuntimeMode};
    use std::time::{Duration, Instant};

    fn tile(seed: u8) -> [u8; codec::TILE_BYTES] {
        let mut tile = [0u8; codec::TILE_BYTES];
        for (index, byte) in tile.iter_mut().enumerate() {
            *byte = seed.wrapping_add(index as u8);
        }
        tile[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2].copy_from_slice(&0u16.to_le_bytes());
        tile
    }

    fn resource() -> (ResourceSpec, EirResource) {
        let spec = ResourceSpec::builder(
            ResourceClassId::REPRESENTATIONAL,
            LogicalResourceId::new("slhav2-real-kv").unwrap(),
        )
        .allow(DimensionId::REPRESENTATION)
        .preserve(Invariant::new(InvariantKind::PreserveContents))
        .preserve(Invariant::new(InvariantKind::PreserveIdentity))
        .admit(AdmissibleTransition::new(
            TransitionMechanism::Reencode,
            DimensionId::REPRESENTATION,
        ))
        .require_capability(CapabilityRequirement::new(
            TransitionMechanism::Reencode,
            DimensionId::REPRESENTATION,
        ))
        .observe(ObservationSignalId::FREE_CAPACITY)
        .build()
        .unwrap();
        let eir = lower(&spec).unwrap().resources()[0].clone();
        (spec, eir)
    }

    fn runtime(spec: ResourceSpec, eir: EirResource) -> Runtime {
        Runtime::new(RuntimeConfig {
            resource_spec: spec,
            ir_resource: eir,
            mode: RuntimeMode::Apply,
            dry_run: false,
            max_cycles: 1,
            ..RuntimeConfig::default()
        })
    }

    fn gated_plan(
        backend: &SlhaKvTransitionBackendV1,
        spec: ResourceSpec,
        observation: &KvCapacityObservationV1,
        now: Instant,
    ) -> BooleanKvTransitionPreflightV2 {
        let mut gate = BooleanKvCapacityPreflightControllerV1::new(
            spec,
            TransitionMechanism::Reencode,
            Duration::from_secs(1),
        )
        .unwrap();
        gate.validate_candidate_v2(
            backend.source(),
            backend.target().representation.clone(),
            &backend.capabilities(),
            backend.attestations(),
            KvTargetMaterialization::new(
                backend.target().key_transform_scope,
                backend.target().key_encoding_pipeline,
                backend.target().recovery_source,
            ),
            observation.planning_context(),
            observation.observations(),
            codec::WARM_PACKED_BYTES as u64,
            now,
        )
        .unwrap()
    }

    #[test]
    fn real_hot_slot_runs_from_true_capacity_guard_through_verified_commit() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let original = tile(7);
        let slot = physical.insert(original);
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle.clone(),
            slot,
            RepresentationEpoch::new(1),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        let (spec, eir) = resource();
        let now = Instant::now();
        let observation = handle.capacity_observation(now).unwrap();
        assert_eq!(
            observation
                .observations()
                .get(ObservationSignalId::FREE_CAPACITY)
                .unwrap()
                .source()
                .to_string(),
            "resource:slhav2-real-kv"
        );
        let gated = gated_plan(&backend, spec.clone(), &observation, now);
        let plan = match gated {
            BooleanKvTransitionPreflightV2::Candidate { report, plan } => {
                assert_eq!(report.evidence.truth, "true");
                assert_eq!(report.evidence.forecast_method, "current-state");
                assert_eq!(report.evidence.forecast_horizon_milliseconds, 0);
                assert!(!report.evidence.forecast_confidence_claimed);
                assert_eq!(plan, backend.transition_plan().unwrap());
                plan
            }
            BooleanKvTransitionPreflightV2::Blocked(report) => {
                panic!("fresh sufficient physical capacity was blocked: {report:?}")
            }
        };
        let source = backend.source().clone();
        let capabilities = backend.capabilities();
        let attestations = backend.attestations();
        let mut actuator =
            TransactionalKvPageV1::new(backend, &eir, source, plan, &capabilities, attestations)
                .unwrap();

        let result = runtime(spec, eir.clone())
            .cycle(&eir, &FirstGroundedPlanner, &(), &mut actuator)
            .unwrap();

        assert!(result.commit.is_some());
        assert!(result.rollback.is_none());
        assert_eq!(
            handle.with_cache(|cache| cache.tier(slot)).unwrap(),
            Some(PhysicalTier::Warm)
        );
        assert_eq!(
            handle
                .with_cache(|cache| cache.restorable_hot_tile(slot))
                .unwrap(),
            Some(original)
        );
    }

    #[test]
    fn false_capacity_guard_leaves_real_hot_slot_untouched() {
        let mut physical = ElasticKvCache::new(codec::TILE_BYTES, "slhav2-real-kv");
        let original = tile(11);
        let slot = physical.insert(original);
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle.clone(),
            slot,
            RepresentationEpoch::new(1),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        let (spec, _) = resource();
        let now = Instant::now();
        let observation = handle.capacity_observation(now).unwrap();
        assert_eq!(
            observation
                .planning_context()
                .get(ObservationSignalId::FREE_CAPACITY),
            Some(0.0)
        );
        let gated = gated_plan(&backend, spec, &observation, now);
        let BooleanKvTransitionPreflightV2::Blocked(report) = gated else {
            panic!("zero free capacity must block conservative WARM materialization");
        };
        assert_eq!(report.evidence.truth, "false");
        assert_eq!(
            handle.with_cache(|cache| cache.tier(slot)).unwrap(),
            Some(PhysicalTier::Hot)
        );
        assert_eq!(
            handle
                .with_cache(|cache| cache.restorable_hot_tile(slot))
                .unwrap(),
            Some(original)
        );
    }

    #[test]
    fn unknown_capacity_guard_leaves_real_hot_slot_untouched() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let original = tile(13);
        let slot = physical.insert(original);
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle.clone(),
            slot,
            RepresentationEpoch::new(1),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        let (spec, _) = resource();
        let now = Instant::now();
        let observation = KvCapacityObservationV1::unsupported(
            spec.resource_id().clone(),
            now,
            "capacity sensor unavailable",
        );
        let gated = gated_plan(&backend, spec, &observation, now);
        let BooleanKvTransitionPreflightV2::Blocked(report) = gated else {
            panic!("unsupported capacity evidence must fail closed");
        };
        assert_eq!(report.evidence.truth, "unknown");
        assert_eq!(
            handle.with_cache(|cache| cache.tier(slot)).unwrap(),
            Some(PhysicalTier::Hot)
        );
        assert_eq!(
            handle
                .with_cache(|cache| cache.restorable_hot_tile(slot))
                .unwrap(),
            Some(original)
        );
    }

    #[test]
    fn physical_capacity_observation_rejects_invalid_cache_identity() {
        let handle = SlhaKvCacheHandleV1::new(ElasticKvCache::new(4096, ""));
        let error = handle
            .capacity_observation(Instant::now())
            .expect_err("invalid physical cache identity must fail closed");
        assert!(error.contains("invalid SLHAv2 KV resource id"));
    }

    #[test]
    fn unknown_flag_bits_fail_closed_before_binding() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let mut unsupported = tile(5);
        unsupported[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2]
            .copy_from_slice(&(1_u16 << 6).to_le_bytes());
        let slot = physical.insert(unsupported);
        let handle = SlhaKvCacheHandleV1::new(physical);

        let error = match SlhaKvTransitionBackendV1::bind_hot_slot(
            handle,
            slot,
            RepresentationEpoch::new(1),
            SlhaKvSemanticContractV1::token_stable(),
        ) {
            Ok(_) => panic!("unknown SLHA flag bits must not be labeled as INT4"),
            Err(error) => error,
        };

        assert!(error.contains("unsupported flag bits 0x0040"));
    }

    #[test]
    fn slot_reuse_is_detected_before_elasticxxx_actuation() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let slot = physical.insert(tile(1));
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle.clone(),
            slot,
            RepresentationEpoch::new(1),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        handle
            .with_cache_mut(|cache| cache.write_at(slot, tile(9)).unwrap())
            .unwrap();
        let (spec, eir) = resource();
        let mut actuator = backend.into_transactional(&eir).unwrap();

        let error = runtime(spec, eir.clone())
            .cycle(&eir, &FirstGroundedPlanner, &(), &mut actuator)
            .expect_err("recycled SLHAv2 slot must fail closed");
        assert!(error.to_string().contains("recycled"));
        assert_eq!(
            handle.with_cache(|cache| cache.tier(slot)).unwrap(),
            Some(PhysicalTier::Hot)
        );
    }

    #[test]
    fn physical_backend_rollback_restores_exact_hot_bytes() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let original = tile(3);
        let slot = physical.insert(original);
        let handle = SlhaKvCacheHandleV1::new(physical);
        let mut backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle.clone(),
            slot,
            RepresentationEpoch::new(4),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        let source = backend.source().clone();
        let target = backend.target().clone();
        let transition = backend.transition_plan().unwrap();

        backend
            .apply_transition_if_source(&source, &target, &transition)
            .unwrap();
        assert_eq!(
            handle.with_cache(|cache| cache.tier(slot)).unwrap(),
            Some(PhysicalTier::Warm)
        );
        backend.restore_page(&source).unwrap();
        assert_eq!(
            handle.with_cache(|cache| cache.tier(slot)).unwrap(),
            Some(PhysicalTier::Hot)
        );
        assert_eq!(
            handle
                .with_cache(|cache| cache.restorable_hot_tile(slot))
                .unwrap(),
            Some(original)
        );
    }
}
