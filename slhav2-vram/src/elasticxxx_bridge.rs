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

use scirust_modalg::PackedBitPlane;
use elasticxxx::kv::boolean_admission::KvCapacityObservationV1;
use elasticxxx::kv::{
    CapabilitySet, ElasticWordPlaneV1, ElasticWordWidthV1, KeyEncodingPipeline, KeyTransformScope,
    KvPageDescriptor, KvPageId, KvPrecision, KvRecoverySource, KvResidency,
    KvTargetMaterialization, KvTransitionBackendV1, KvTransitionPlan, RepresentationEpoch,
    RepresentationId, RepresentationPrecisionKvBindingV1, RepresentationState,
    TransactionalKvPageV1, TransitionAttestations,
};
use elasticxxx::resource::{
    AdmissibleTransition, CapabilityRequirement, DimensionId, Invariant, InvariantKind,
    LogicalResourceId, RepresentationalDeclaration, ResourceClassId, ResourceSpec,
};
use elasticxxx::runtime::representation_payload_decision::{
    evaluate_representation_payload_v1, RepresentationPayloadDecisionV1,
};
use elasticxxx::runtime::representation_payload_selector::{
    select_minimum_payload_v1, RepresentationPayloadCandidateV1, RepresentationPayloadMinimumV1,
};
use elasticxxx::runtime::representation_payload_stability::{
    RepresentationPayloadStabilityControllerV1, RepresentationPayloadStableDecisionV1,
};
use elasticxxx::runtime::representation_payload_stability_trace::RepresentationPayloadStabilityTraceV1;
use elasticxxx::{
    representation_precision_floor_signal, BooleanRepresentationPrecisionPreplannerV1,
    BooleanRepresentationPrecisionReportV2, EirResource, InvariantCheck, ObservationSnapshot, Plan,
    RepresentationPrecisionCandidateV1, RuntimeError, TransitionMechanism, VerificationResult,
};

use crate::codec;
use crate::elastic_cache::{ElasticKvCache, PhysicalTier};

/// Legacy version of the SLHAv2 -> ElasticXxx capacity/transaction consumer contract.
pub const SLHAV2_ELASTICXXX_KV_BRIDGE_V1: u16 = 1;
/// Additive ELANG8a bridge version: current Elastic facade + fixed-width
/// representation/precision evidence composed with the physical KV plan.
pub const SLHAV2_ELASTICXXX_KV_BRIDGE_V2: u16 = 2;
/// Versioned read-only slot control-plane projection.
pub const SLHAV2_ELASTIC_WORD_CONTROL_V1: &str = "slhav2.elastic-word-control@1.0.0";
/// Lossless baseline width for the five independent v1 slot metadata fields.
pub const SLHAV2_ELASTIC_WORD_CONTROL_BITS_V1: u16 = 512;
/// Number of u64 lanes in the v1 SLHAv2 control word.
pub const SLHAV2_ELASTIC_WORD_CONTROL_LANES_V1: usize = 8;

/// Versioned dense physical-slot control-plane projection.
pub const SLHAV2_ELASTIC_WORD_DENSE_CONTROL_V2: &str = "slhav2.elastic-word-dense-control@2.0.0";
/// Lossless v2 dense width: one generation lane plus one Boolean state bitfield.
pub const SLHAV2_ELASTIC_WORD_DENSE_CONTROL_BITS_V2: u16 = 128;
/// Every dense v2 word contains exactly two native u64 lanes.
pub const SLHAV2_ELASTIC_WORD_DENSE_CONTROL_LANES_V2: usize = 2;

/// Dense v2 state bit: the physical slot currently owns a logical KV value.
pub const SLHA_SLOT_PRESENT_BIT_V2: u64 = 1 << 0;
/// Dense v2 mutually-exclusive HOT tier bit.
pub const SLHA_SLOT_HOT_BIT_V2: u64 = 1 << 1;
/// Dense v2 mutually-exclusive WARM tier bit.
pub const SLHA_SLOT_WARM_BIT_V2: u64 = 1 << 2;
/// Dense v2 mutually-exclusive COLD tier bit.
pub const SLHA_SLOT_COLD_BIT_V2: u64 = 1 << 3;
/// Dense v2 mutually-exclusive PINNED tier bit.
pub const SLHA_SLOT_PINNED_BIT_V2: u64 = 1 << 4;
/// All state bits currently defined by the v2 schema.
pub const SLHA_SLOT_STATE_MASK_V2: u64 = SLHA_SLOT_PRESENT_BIT_V2
    | SLHA_SLOT_HOT_BIT_V2
    | SLHA_SLOT_WARM_BIT_V2
    | SLHA_SLOT_COLD_BIT_V2
    | SLHA_SLOT_PINNED_BIT_V2;

/// Versioned hybrid Boolean control-plane projection.
pub const SLHAV2_ELASTIC_WORD_HYBRID_CONTROL_V3: &str = "slhav2.elastic-word-hybrid-control@3.0.0";
/// Generation remains one exact W64 lane per physical slot.
pub const SLHAV2_ELASTIC_WORD_HYBRID_GENERATION_BITS_V3: u16 = 64;
/// Hybrid state uses one presence and two tier bitplanes.
pub const SLHAV2_ELASTIC_WORD_HYBRID_BOOLEAN_PLANES_V3: usize = 3;
/// Versioned structural accounting across the retained SLHAv2 control profiles.
pub const SLHAV2_ELASTIC_CONTROL_PROFILE_ACCOUNTING_V4: &str =
    "slhav2.elastic-control-profile-accounting@4.0.0";

/// Exact ElasticXxx revision pinned by the optional Cargo dependencies.
pub const ELASTICXXX_BE14D_CONTRACT_REVISION: &str = "4900a4ae5dfc0311bf854e20bb7e644b40342384";
/// Exact ElasticXxx ELANG7/ELANG8a source revision qualified by this consumer.
pub const ELASTICXXX_ELANG8A_CONTRACT_REVISION: &str = ELASTICXXX_BE14D_CONTRACT_REVISION;

const REPRESENTATION_SCHEMA_V1: u32 = 1;
const BACKEND_NAME: &str = "slhav2-elastic-kv-cache-v1";
const SLHA_CACHE_RESIDENCY: &str = "slhav2-host-cache";

/// Hybrid W64 + Boolean-bitplane snapshot of the physical slot control plane.
///
/// Slot identity is implicit in the position. Generation uses one exact W64
/// lane per slot. Presence and the two-bit tier code are stored as three packed
/// Boolean bitplanes shared by all slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlhaElasticHybridControlPlaneV3 {
    slot_count: usize,
    generations: ElasticWordPlaneV1,
    presence: PackedBitPlane,
    tier_low: PackedBitPlane,
    tier_high: PackedBitPlane,
}

impl SlhaElasticHybridControlPlaneV3 {
    /// Number of physical slot positions represented, including holes.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Exact W64 generation plane.
    #[must_use]
    pub const fn generations(&self) -> &ElasticWordPlaneV1 {
        &self.generations
    }

    /// Packed presence bitplane.
    #[must_use]
    pub fn presence_words(&self) -> &[u64] {
        self.presence.words()
    }

    /// Packed low tier bitplane.
    #[must_use]
    pub fn tier_low_words(&self) -> &[u64] {
        self.tier_low.words()
    }

    /// Packed high tier bitplane.
    #[must_use]
    pub fn tier_high_words(&self) -> &[u64] {
        self.tier_high.words()
    }

    /// Exact payload bits in the generation and Boolean backing planes.
    ///
    /// This excludes Rust Vec/struct headers and makes no DRAM/cache-line claim.
    #[must_use]
    pub fn payload_bits(&self) -> usize {
        self.generations.as_lanes().len() * 64
            + self.presence.backing_bits()
            + self.tier_low.backing_bits()
            + self.tier_high.backing_bits()
    }

    /// Decode one slot without reconstructing per-slot metadata objects.
    pub fn slot_state(&self, slot: usize) -> Result<Option<(u64, PhysicalTier)>, String> {
        if slot >= self.slot_count {
            return Err(format!(
                "SLHAv2 hybrid control slot {slot} out of range for {} slots",
                self.slot_count
            ));
        }
        let present = self
            .presence
            .get(slot)
            .expect("slot range checked against bitplane length");
        let low = self
            .tier_low
            .get(slot)
            .expect("slot range checked against bitplane length");
        let high = self
            .tier_high
            .get(slot)
            .expect("slot range checked against bitplane length");
        if !present {
            if low || high {
                return Err(format!(
                    "SLHAv2 hybrid absent slot {slot} carries non-zero tier bits"
                ));
            }
            return Ok(None);
        }

        let code = u8::from(low) | (u8::from(high) << 1);
        let tier = match code {
            0 => PhysicalTier::Hot,
            1 => PhysicalTier::Warm,
            2 => PhysicalTier::Cold,
            3 => PhysicalTier::Pinned,
            _ => unreachable!("two Boolean tier planes encode only 0..=3"),
        };
        let generation = self
            .generations
            .word(slot)
            .map_err(|error| error.to_string())?[0];
        Ok(Some((generation, tier)))
    }
}

/// Stable identity of one retained SLHAv2 control-plane representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SlhaElasticControlProfileV4 {
    SparseW512,
    DenseW128,
    HybridW64Boolean,
}

impl SlhaElasticControlProfileV4 {
    /// Stable cross-repository profile identifier.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::SparseW512 => "slhav2-control-sparse-w512",
            Self::DenseW128 => "slhav2-control-dense-w128",
            Self::HybridW64Boolean => "slhav2-control-hybrid-w64-boolean",
        }
    }
}

/// One domain-owned profile plus exact structural payload evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlhaElasticControlProfileCandidateV4 {
    profile: SlhaElasticControlProfileV4,
    payload_bits: usize,
}

impl SlhaElasticControlProfileCandidateV4 {
    #[must_use]
    pub const fn profile(self) -> SlhaElasticControlProfileV4 {
        self.profile
    }

    #[must_use]
    pub const fn profile_id(self) -> &'static str {
        self.profile.id()
    }

    #[must_use]
    pub const fn payload_bits(self) -> usize {
        self.payload_bits
    }
}

/// Exact backing-payload accounting for the three retained SLHAv2 control
/// representations.
///
/// These values count only representation payload bits. They deliberately
/// exclude Rust Vec/struct headers, allocator slack, cache-line effects and
/// physical DRAM/HBM traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlhaElasticControlProfileAccountingV4 {
    slot_count: usize,
    present_slots: usize,
    sparse_w512_payload_bits: usize,
    dense_w128_payload_bits: usize,
    hybrid_w64_boolean_payload_bits: usize,
}

impl SlhaElasticControlProfileAccountingV4 {
    #[must_use]
    pub const fn slot_count(self) -> usize {
        self.slot_count
    }

    #[must_use]
    pub const fn present_slots(self) -> usize {
        self.present_slots
    }

    #[must_use]
    pub const fn absent_slots(self) -> usize {
        self.slot_count - self.present_slots
    }

    /// Return all retained profiles in stable identity order.
    ///
    /// This is candidate evidence only. The order is not a preference and no
    /// runtime profile is selected by this method.
    #[must_use]
    pub const fn candidates(self) -> [SlhaElasticControlProfileCandidateV4; 3] {
        [
            SlhaElasticControlProfileCandidateV4 {
                profile: SlhaElasticControlProfileV4::SparseW512,
                payload_bits: self.sparse_w512_payload_bits,
            },
            SlhaElasticControlProfileCandidateV4 {
                profile: SlhaElasticControlProfileV4::DenseW128,
                payload_bits: self.dense_w128_payload_bits,
            },
            SlhaElasticControlProfileCandidateV4 {
                profile: SlhaElasticControlProfileV4::HybridW64Boolean,
                payload_bits: self.hybrid_w64_boolean_payload_bits,
            },
        ]
    }

    /// Ask the generic ElasticXxx selector for the exact minimum structural
    /// payload set across the three SLHAv2-owned control profiles.
    ///
    /// This method converts only stable profile identity + structural payload
    /// bits. It does not supply semantic admissibility, transition cost,
    /// quality evidence, stability evidence or actuation authority.
    pub fn minimum_payload_profiles_v5(self) -> Result<RepresentationPayloadMinimumV1, String> {
        let candidates = self
            .candidates()
            .into_iter()
            .map(|candidate| {
                let bits = u64::try_from(candidate.payload_bits())
                    .map_err(|_| "SLHAv2 profile payload bits do not fit u64".to_owned())?;
                RepresentationPayloadCandidateV1::new(candidate.profile_id(), bits)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        select_minimum_payload_v1(candidates).map_err(|error| error.to_string())
    }

    /// Interpret exact structural payload evidence for the current SLHAv2
    /// control profile using the generic ElasticXxx decision contract.
    ///
    /// A current profile that already belongs to the exact minimum set is held,
    /// including ties. A non-current unique minimum becomes a planning-only
    /// transition candidate; a non-current tie remains ambiguous. This method
    /// does not perform semantic admission, stability gating or actuation.
    pub fn payload_decision_v6(
        self,
        current: SlhaElasticControlProfileV4,
    ) -> Result<RepresentationPayloadDecisionV1, String> {
        let candidates = self
            .candidates()
            .into_iter()
            .map(|candidate| {
                let bits = u64::try_from(candidate.payload_bits())
                    .map_err(|_| "SLHAv2 profile payload bits do not fit u64".to_owned())?;
                RepresentationPayloadCandidateV1::new(candidate.profile_id(), bits)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        evaluate_representation_payload_v1(current.id(), candidates)
            .map_err(|error| error.to_string())
    }

    /// Apply ElasticXxx's generic transition-stability gate to one structural
    /// profile decision.
    ///
    /// Hold and ambiguous structural minima do not consume stability state.
    /// A unique non-current minimum may be Deferred or Admitted by the generic
    /// representation/Reencode stability policy. Even Admitted carries no
    /// SLHAv2 codec/residency actuation authority.
    pub fn stable_payload_decision_v7(
        self,
        current: SlhaElasticControlProfileV4,
        controller: &mut RepresentationPayloadStabilityControllerV1,
        observations: &ObservationSnapshot,
        now: Instant,
    ) -> Result<RepresentationPayloadStableDecisionV1, String> {
        let candidates = self
            .candidates()
            .into_iter()
            .map(|candidate| {
                let bits = u64::try_from(candidate.payload_bits())
                    .map_err(|_| "SLHAv2 profile payload bits do not fit u64".to_owned())?;
                RepresentationPayloadCandidateV1::new(candidate.profile_id(), bits)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        controller
            .evaluate(current.id(), candidates, observations, now)
            .map_err(|error| error.to_string())
    }

    /// Capture read-only stability-gated profile evidence without returning
    /// the single-use permit itself.
    ///
    /// This helper may evaluate the generic stability gate exactly like V7,
    /// but immediately converts the result into an authority-free trace.
    pub fn stable_payload_trace_v8(
        self,
        current: SlhaElasticControlProfileV4,
        controller: &mut RepresentationPayloadStabilityControllerV1,
        observations: &ObservationSnapshot,
        now: Instant,
    ) -> Result<RepresentationPayloadStabilityTraceV1, String> {
        let decision = self.stable_payload_decision_v7(current, controller, observations, now)?;
        Ok(RepresentationPayloadStabilityTraceV1::capture(&decision))
    }

    /// V1 sparse layout: one explicit 512-bit word per present slot.
    #[must_use]
    pub const fn sparse_w512_payload_bits(self) -> usize {
        self.sparse_w512_payload_bits
    }

    /// V2 dense layout: one 128-bit word per physical slot position.
    #[must_use]
    pub const fn dense_w128_payload_bits(self) -> usize {
        self.dense_w128_payload_bits
    }

    /// V3 hybrid layout: one W64 generation lane per slot plus three packed
    /// Boolean bitplanes shared across the slot domain.
    #[must_use]
    pub const fn hybrid_w64_boolean_payload_bits(self) -> usize {
        self.hybrid_w64_boolean_payload_bits
    }
}

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

    /// Snapshot present SLHAv2 physical-slot control metadata as one flat
    /// ElasticWord plane.
    ///
    /// The v1 lossless baseline uses W512 with eight u64 lanes per present slot:
    /// `[slot, generation, tier, resident_bytes, backing_bytes, 0, 0, 0]`.
    /// Width is carried once by the plane. This method is read-only and grants
    /// no mutation, paging, eviction, codec or residency-transition authority.
    pub fn elastic_word_control_plane(&self) -> Result<ElasticWordPlaneV1, String> {
        let cache = self.lock()?;
        let counts = cache.counts();
        let present_slots = counts
            .0
            .checked_add(counts.1)
            .and_then(|value| value.checked_add(counts.2))
            .and_then(|value| value.checked_add(counts.3))
            .ok_or_else(|| "SLHAv2 present-slot count overflow".to_owned())?;
        let lane_capacity = present_slots
            .checked_mul(SLHAV2_ELASTIC_WORD_CONTROL_LANES_V1)
            .ok_or_else(|| "SLHAv2 ElasticWord lane capacity overflow".to_owned())?;
        let mut lanes = Vec::with_capacity(lane_capacity);

        for (slot, generation, tier, resident_bytes, backing_bytes) in cache.slot_control_metadata()
        {
            lanes.push(
                u64::try_from(slot).map_err(|_| "SLHAv2 slot index does not fit u64".to_owned())?,
            );
            lanes.push(generation);
            lanes.push(physical_tier_code(tier));
            lanes.push(
                u64::try_from(resident_bytes)
                    .map_err(|_| "SLHAv2 resident byte count does not fit u64".to_owned())?,
            );
            lanes.push(
                u64::try_from(backing_bytes)
                    .map_err(|_| "SLHAv2 backing byte count does not fit u64".to_owned())?,
            );
            lanes.extend_from_slice(&[0, 0, 0]);
        }

        let width = ElasticWordWidthV1::from_bits(SLHAV2_ELASTIC_WORD_CONTROL_BITS_V1)
            .map_err(|error| error.to_string())?;
        ElasticWordPlaneV1::new(width, lanes).map_err(|error| error.to_string())
    }

    /// Snapshot every physical slot position into a dense W128 Boolean control plane.
    ///
    /// Word index is the stable physical slot identity. Each word is:
    ///
    /// `[generation_or_zero, state_bitfield]`
    ///
    /// An absent slot is exactly `[0, 0]`. A present slot sets PRESENT and
    /// exactly one tier bit. Resident/backing byte counts are not repeated in
    /// the word: they are re-derived from SLHAv2's authoritative tier invariants
    /// and compared with the physical slot before the snapshot is accepted.
    ///
    /// This is read-only evidence. It does not authorize residency mutation,
    /// codec changes, eviction, compaction or runtime promotion.
    pub fn elastic_word_dense_control_plane_v2(&self) -> Result<ElasticWordPlaneV1, String> {
        let cache = self.lock()?;
        let slot_count = cache.slot_control_metadata_dense().len();
        let lane_capacity = slot_count
            .checked_mul(SLHAV2_ELASTIC_WORD_DENSE_CONTROL_LANES_V2)
            .ok_or_else(|| "SLHAv2 dense ElasticWord lane capacity overflow".to_owned())?;
        let mut lanes = Vec::with_capacity(lane_capacity);

        for metadata in cache.slot_control_metadata_dense() {
            match metadata {
                None => lanes.extend_from_slice(&[0, 0]),
                Some((generation, tier, resident_bytes, backing_bytes)) => {
                    let (expected_resident, expected_backing) = derived_bytes_for_tier_v2(tier);
                    if resident_bytes != expected_resident || backing_bytes != expected_backing {
                        return Err(format!(
                            "SLHAv2 tier {tier:?} byte accounting drifted: resident={resident_bytes} backing={backing_bytes}, expected resident={expected_resident} backing={expected_backing}"
                        ));
                    }
                    lanes.push(generation);
                    lanes.push(dense_state_bits_v2(tier));
                }
            }
        }

        let width = ElasticWordWidthV1::from_bits(SLHAV2_ELASTIC_WORD_DENSE_CONTROL_BITS_V2)
            .map_err(|error| error.to_string())?;
        ElasticWordPlaneV1::new(width, lanes).map_err(|error| error.to_string())
    }

    /// Snapshot every physical slot into a hybrid W64 + Boolean control plane.
    ///
    /// Slot identity is implicit by position. Generation remains full-width u64.
    /// Presence and tier are packed into three shared Boolean bitplanes, so no
    /// per-slot state lane is repeated. Resident/backing bytes are derived from
    /// the authoritative tier invariants and verified before publication.
    pub fn elastic_word_hybrid_control_plane_v3(
        &self,
    ) -> Result<SlhaElasticHybridControlPlaneV3, String> {
        let cache = self.lock()?;
        let slot_count = cache.slot_control_metadata_dense().len();
        let mut generations = vec![0_u64; slot_count];
        let mut presence = PackedBitPlane::zeroed(slot_count);
        let mut tier_low = PackedBitPlane::zeroed(slot_count);
        let mut tier_high = PackedBitPlane::zeroed(slot_count);

        for (slot, metadata) in cache.slot_control_metadata_dense().enumerate() {
            let Some((generation, tier, resident_bytes, backing_bytes)) = metadata else {
                continue;
            };
            let (expected_resident, expected_backing) = derived_bytes_for_tier_v2(tier);
            if resident_bytes != expected_resident || backing_bytes != expected_backing {
                return Err(format!(
                    "SLHAv2 tier {tier:?} byte accounting drifted: resident={resident_bytes} backing={backing_bytes}, expected resident={expected_resident} backing={expected_backing}"
                ));
            }

            generations[slot] = generation;
            presence
                .set(slot, true)
                .map_err(|error| format!("SLHAv2 presence bitplane update failed: {error}"))?;
            let tier_code = physical_tier_code(tier);
            if tier_code & 1 != 0 {
                tier_low
                    .set(slot, true)
                    .map_err(|error| format!("SLHAv2 tier-low bitplane update failed: {error}"))?;
            }
            if tier_code & 2 != 0 {
                tier_high
                    .set(slot, true)
                    .map_err(|error| format!("SLHAv2 tier-high bitplane update failed: {error}"))?;
            }
        }

        let width = ElasticWordWidthV1::from_bits(SLHAV2_ELASTIC_WORD_HYBRID_GENERATION_BITS_V3)
            .map_err(|error| error.to_string())?;
        let generations =
            ElasticWordPlaneV1::new(width, generations).map_err(|error| error.to_string())?;
        Ok(SlhaElasticHybridControlPlaneV3 {
            slot_count,
            generations,
            presence,
            tier_low,
            tier_high,
        })
    }

    /// Compute exact representation-payload accounting for the retained V1/V2/V3
    /// control-plane layouts over the current physical slot domain.
    ///
    /// This report is evidence only. It does not select or promote a
    /// representation because the narrowest payload depends on slot count and
    /// sparsity, while transition/runtime costs require separate measurement.
    pub fn elastic_control_profile_accounting_v4(
        &self,
    ) -> Result<SlhaElasticControlProfileAccountingV4, String> {
        let cache = self.lock()?;
        let dense = cache.slot_control_metadata_dense();
        let slot_count = dense.len();
        let present_slots = dense.filter(Option::is_some).count();
        let bitmap_words = slot_count.div_ceil(64);

        let sparse_w512_payload_bits = present_slots
            .checked_mul(usize::from(SLHAV2_ELASTIC_WORD_CONTROL_BITS_V1))
            .ok_or_else(|| "SLHAv2 sparse W512 payload accounting overflow".to_owned())?;
        let dense_w128_payload_bits = slot_count
            .checked_mul(usize::from(SLHAV2_ELASTIC_WORD_DENSE_CONTROL_BITS_V2))
            .ok_or_else(|| "SLHAv2 dense W128 payload accounting overflow".to_owned())?;
        let hybrid_generation_bits = slot_count
            .checked_mul(usize::from(SLHAV2_ELASTIC_WORD_HYBRID_GENERATION_BITS_V3))
            .ok_or_else(|| "SLHAv2 hybrid generation payload accounting overflow".to_owned())?;
        let hybrid_boolean_bits = bitmap_words
            .checked_mul(SLHAV2_ELASTIC_WORD_HYBRID_BOOLEAN_PLANES_V3)
            .and_then(|words| words.checked_mul(64))
            .ok_or_else(|| "SLHAv2 hybrid Boolean payload accounting overflow".to_owned())?;
        let hybrid_w64_boolean_payload_bits = hybrid_generation_bits
            .checked_add(hybrid_boolean_bits)
            .ok_or_else(|| "SLHAv2 hybrid total payload accounting overflow".to_owned())?;

        Ok(SlhaElasticControlProfileAccountingV4 {
            slot_count,
            present_slots,
            sparse_w512_payload_bits,
            dense_w128_payload_bits,
            hybrid_w64_boolean_payload_bits,
        })
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

    /// Return the fixed-width representation/precision candidate exposed by
    /// this physical transition, when the codec semantics support that contract.
    ///
    /// Today only uniform signed INT4 is represented by ElasticXxx's fixed-width
    /// `declared-bits-per-scalar` contract. NF4/MIXED/TQ3/MIX3 remain explicit
    /// custom precisions and therefore return `None` rather than receiving a
    /// fabricated bit width.
    pub fn fixed_width_precision_candidate(
        &self,
    ) -> Result<Option<RepresentationPrecisionCandidateV1>, String> {
        if self.target.precision != KvPrecision::Int4 {
            return Ok(None);
        }
        RepresentationPrecisionCandidateV1::new(
            "slhav2-int4-warm-rank-0",
            0,
            self.target.representation.id.clone(),
            self.target.representation.schema_version,
            TransitionMechanism::Reencode,
            4,
        )
        .map(Some)
    }

    /// Build the BE14e preplanner for this exact fixed-width physical transition.
    ///
    /// The declaration reuses the cache's real logical resource identity and
    /// exact source/target representation contracts. It adds only the precision
    /// floor observation required by the generic ElasticXxx preplanner.
    pub fn fixed_width_precision_preplanner(
        &self,
    ) -> Result<Option<BooleanRepresentationPrecisionPreplannerV1>, String> {
        let Some(candidate) = self.fixed_width_precision_candidate()? else {
            return Ok(None);
        };
        let resource_id = {
            let cache = self.cache.lock()?;
            LogicalResourceId::new(cache.resource_id())
                .map_err(|error| format!("invalid SLHAv2 KV resource id for ElasticXxx: {error}"))?
        };
        let spec = ResourceSpec::builder(ResourceClassId::REPRESENTATIONAL, resource_id)
            .allow(DimensionId::REPRESENTATION)
            .observe(representation_precision_floor_signal())
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
            .build()
            .map_err(|error| error.to_string())?;
        let declaration = RepresentationalDeclaration::new(
            spec,
            [
                (
                    self.source.representation.id.clone(),
                    self.source.representation.schema_version,
                ),
                (
                    self.target.representation.id.clone(),
                    self.target.representation.schema_version,
                ),
            ],
        )
        .map_err(|error| error.to_string())?;
        BooleanRepresentationPrecisionPreplannerV1::new(declaration, vec![candidate]).map(Some)
    }

    /// Compose one actual BE14e fixed-width selection with this exact physical
    /// KV transition. This does not actuate the cache and does not replace the
    /// capacity guard or trusted transaction validation.
    pub fn bind_fixed_width_precision_report(
        &self,
        report: BooleanRepresentationPrecisionReportV2,
    ) -> Result<Option<RepresentationPrecisionKvBindingV1>, String> {
        let Some(candidate) = self.fixed_width_precision_candidate()? else {
            return Ok(None);
        };
        RepresentationPrecisionKvBindingV1::new(candidate, report, self.transition_plan()?)
            .map(Some)
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

const fn dense_state_bits_v2(tier: PhysicalTier) -> u64 {
    SLHA_SLOT_PRESENT_BIT_V2
        | match tier {
            PhysicalTier::Hot => SLHA_SLOT_HOT_BIT_V2,
            PhysicalTier::Warm => SLHA_SLOT_WARM_BIT_V2,
            PhysicalTier::Cold => SLHA_SLOT_COLD_BIT_V2,
            PhysicalTier::Pinned => SLHA_SLOT_PINNED_BIT_V2,
        }
}

const fn derived_bytes_for_tier_v2(tier: PhysicalTier) -> (usize, usize) {
    match tier {
        PhysicalTier::Hot | PhysicalTier::Pinned => (codec::TILE_BYTES, 0),
        PhysicalTier::Warm => (codec::WARM_PACKED_BYTES, codec::RESIDUAL_WORDS * 8),
        // COLD stores either a 128-byte HOT image or a 96-byte WARM image plus
        // its retained 32-byte residual. Both cases therefore retain one full
        // 128-byte logical tile outside the resident budget.
        PhysicalTier::Cold => (0, codec::TILE_BYTES),
    }
}

const fn physical_tier_code(tier: PhysicalTier) -> u64 {
    match tier {
        PhysicalTier::Hot => 0,
        PhysicalTier::Warm => 1,
        PhysicalTier::Cold => 2,
        PhysicalTier::Pinned => 3,
    }
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
    use elasticxxx::{
        lower, representation_precision_floor_signal, FirstGroundedPlanner, Observation,
        ObservationEpoch, ObservationSnapshot, ObservationSource, PlanningContext,
        ResourceGeneration, Runtime, RuntimeConfig, RuntimeMode,
    };
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
    fn dense_w128_control_plane_uses_word_index_as_slot_identity() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-dense-w128");
        let slot0 = physical.insert(tile(1));
        let slot1 = physical.insert(tile(2));
        let slot2 = physical.insert(tile(3));
        assert_eq!((slot0, slot1, slot2), (0, 1, 2));
        assert!(physical.clear_slot(slot1));
        physical.demote_slot(slot2).unwrap();
        let handle = SlhaKvCacheHandleV1::new(physical);

        let plane = handle.elastic_word_dense_control_plane_v2().unwrap();
        assert_eq!(
            plane.width().bits(),
            SLHAV2_ELASTIC_WORD_DENSE_CONTROL_BITS_V2
        );
        assert_eq!(plane.word_count(), 3);
        assert_eq!(
            plane.word(slot0).unwrap(),
            &[0, SLHA_SLOT_PRESENT_BIT_V2 | SLHA_SLOT_HOT_BIT_V2]
        );
        assert_eq!(plane.word(slot1).unwrap(), &[0, 0]);
        assert_eq!(
            plane.word(slot2).unwrap(),
            &[2, SLHA_SLOT_PRESENT_BIT_V2 | SLHA_SLOT_WARM_BIT_V2]
        );
    }

    #[test]
    fn dense_w128_bitfield_is_one_hot_over_present_tiers() {
        for tier in [
            PhysicalTier::Hot,
            PhysicalTier::Warm,
            PhysicalTier::Cold,
            PhysicalTier::Pinned,
        ] {
            let bits = dense_state_bits_v2(tier);
            assert_ne!(bits & SLHA_SLOT_PRESENT_BIT_V2, 0);
            assert_eq!(bits & !SLHA_SLOT_STATE_MASK_V2, 0);
            let tier_bits = bits & !SLHA_SLOT_PRESENT_BIT_V2;
            assert_eq!(tier_bits.count_ones(), 1);
        }
    }

    #[test]
    fn hybrid_w64_boolean_planes_round_trip_sparse_all_tiers() {
        let mut physical = ElasticKvCache::new(8192, "slhav2-hybrid-v3");
        let hot = physical.insert(tile(10));
        let warm = physical.insert(tile(20));
        let cold = physical.insert(tile(30));
        let pinned = physical.insert(tile(40));
        let hole = physical.insert(tile(50));

        physical.demote_slot(warm).unwrap();
        physical.demote_slot(cold).unwrap();
        physical.evict_slot(cold).unwrap();
        assert!(physical.pin(pinned));
        assert!(physical.clear_slot(hole));

        let expected = [
            (hot, Some((0, PhysicalTier::Hot))),
            (warm, Some((1, PhysicalTier::Warm))),
            (cold, Some((2, PhysicalTier::Cold))),
            (pinned, Some((3, PhysicalTier::Pinned))),
            (hole, None),
        ];

        let handle = SlhaKvCacheHandleV1::new(physical);
        let hybrid = handle.elastic_word_hybrid_control_plane_v3().unwrap();
        assert_eq!(hybrid.generations().width().bits(), 64);
        assert_eq!(hybrid.slot_count(), 5);

        for (slot, expected_state) in expected {
            assert_eq!(hybrid.slot_state(slot).unwrap(), expected_state);
        }
    }

    #[test]
    fn hybrid_generation_zero_remains_present_via_boolean_plane() {
        let mut physical = ElasticKvCache::new(1024, "slhav2-hybrid-zero");
        let slot = physical.insert(tile(7));
        assert_eq!(slot, 0);
        assert_eq!(physical.slot_generation(slot), Some(0));

        let handle = SlhaKvCacheHandleV1::new(physical);
        let hybrid = handle.elastic_word_hybrid_control_plane_v3().unwrap();
        assert_eq!(hybrid.slot_state(0).unwrap(), Some((0, PhysicalTier::Hot)));
        assert_eq!(hybrid.generations().as_lanes(), &[0]);
        assert_eq!(hybrid.presence_words()[0] & 1, 1);
    }

    #[test]
    fn scirust_bitplane_tail_is_zero_for_65_slot_hybrid_domain() {
        let mut physical = ElasticKvCache::new(
            65 * codec::TILE_BYTES,
            "slhav2-hybrid-scirust-tail",
        );
        for seed in 0..65_u8 {
            physical.insert(tile(seed));
        }

        let handle = SlhaKvCacheHandleV1::new(physical);
        let hybrid = handle.elastic_word_hybrid_control_plane_v3().unwrap();

        assert_eq!(hybrid.slot_count(), 65);
        assert_eq!(hybrid.presence_words().len(), 2);
        assert_eq!(hybrid.tier_low_words().len(), 2);
        assert_eq!(hybrid.tier_high_words().len(), 2);
        assert_eq!(hybrid.presence_words()[1], 1);
        assert_eq!(hybrid.tier_low_words()[1], 0);
        assert_eq!(hybrid.tier_high_words()[1], 0);
        assert_eq!(
            hybrid.slot_state(64).unwrap(),
            Some((64, PhysicalTier::Hot))
        );
    }

    #[test]
    fn hybrid_boolean_overhead_amortizes_to_three_bits_per_slot_at_64_slots() {
        let mut physical = ElasticKvCache::new(64 * codec::TILE_BYTES, "slhav2-hybrid-64");
        for seed in 0..64_u8 {
            physical.insert(tile(seed));
        }

        let handle = SlhaKvCacheHandleV1::new(physical);
        let hybrid = handle.elastic_word_hybrid_control_plane_v3().unwrap();
        assert_eq!(hybrid.slot_count(), 64);
        assert_eq!(hybrid.generations().as_lanes().len(), 64);
        assert_eq!(hybrid.presence_words().len(), 1);
        assert_eq!(hybrid.tier_low_words().len(), 1);
        assert_eq!(hybrid.tier_high_words().len(), 1);
        assert_eq!(hybrid.payload_bits(), 64 * 64 + 3 * 64);
    }

    #[test]
    fn profile_accounting_shows_no_universal_structural_winner() {
        // One present slot: dense W128 has the smallest declared payload because
        // the hybrid representation still pays one full 64-bit word per bitplane.
        let mut tiny = ElasticKvCache::new(1024, "slhav2-accounting-tiny");
        tiny.insert(tile(1));
        let tiny = SlhaKvCacheHandleV1::new(tiny)
            .elastic_control_profile_accounting_v4()
            .unwrap();
        assert_eq!(tiny.slot_count(), 1);
        assert_eq!(tiny.present_slots(), 1);
        assert_eq!(tiny.sparse_w512_payload_bits(), 512);
        assert_eq!(tiny.dense_w128_payload_bits(), 128);
        assert_eq!(tiny.hybrid_w64_boolean_payload_bits(), 256);

        // Sixty-four dense slots amortize the three Boolean words across the
        // whole domain: hybrid is structurally narrower than dense W128.
        let mut dense64 = ElasticKvCache::new(64 * codec::TILE_BYTES, "slhav2-accounting-dense64");
        for seed in 0..64_u8 {
            dense64.insert(tile(seed));
        }
        let dense64 = SlhaKvCacheHandleV1::new(dense64)
            .elastic_control_profile_accounting_v4()
            .unwrap();
        assert_eq!(dense64.sparse_w512_payload_bits(), 64 * 512);
        assert_eq!(dense64.dense_w128_payload_bits(), 64 * 128);
        assert_eq!(dense64.hybrid_w64_boolean_payload_bits(), 64 * 64 + 3 * 64);

        // A large sparse physical domain can make the explicit sparse W512
        // representation structurally narrower than either dense layout.
        let mut sparse = ElasticKvCache::new(4096, "slhav2-accounting-sparse");
        sparse.write_at(63, tile(9)).unwrap();
        let sparse = SlhaKvCacheHandleV1::new(sparse)
            .elastic_control_profile_accounting_v4()
            .unwrap();
        assert_eq!(sparse.slot_count(), 64);
        assert_eq!(sparse.present_slots(), 1);
        assert_eq!(sparse.sparse_w512_payload_bits(), 512);
        assert_eq!(sparse.dense_w128_payload_bits(), 8192);
        assert_eq!(sparse.hybrid_w64_boolean_payload_bits(), 4288);
    }

    #[test]
    fn generic_payload_selector_preserves_unique_and_tied_slha_minima() {
        let tiny = SlhaElasticControlProfileAccountingV4 {
            slot_count: 1,
            present_slots: 1,
            sparse_w512_payload_bits: 512,
            dense_w128_payload_bits: 128,
            hybrid_w64_boolean_payload_bits: 256,
        };
        let selected = tiny.minimum_payload_profiles_v5().unwrap();
        assert!(selected.is_unique());
        assert_eq!(selected.minimum_payload_bits(), 128);
        assert_eq!(
            selected.profiles()[0].profile_id(),
            SlhaElasticControlProfileV4::DenseW128.id()
        );

        let three_dense = SlhaElasticControlProfileAccountingV4 {
            slot_count: 3,
            present_slots: 3,
            sparse_w512_payload_bits: 1536,
            dense_w128_payload_bits: 384,
            hybrid_w64_boolean_payload_bits: 384,
        };
        let tied = three_dense.minimum_payload_profiles_v5().unwrap();
        assert!(!tied.is_unique());
        assert_eq!(tied.minimum_payload_bits(), 384);
        assert_eq!(
            tied.profiles()
                .iter()
                .map(RepresentationPayloadCandidateV1::profile_id)
                .collect::<Vec<_>>(),
            vec![
                SlhaElasticControlProfileV4::DenseW128.id(),
                SlhaElasticControlProfileV4::HybridW64Boolean.id(),
            ]
        );
    }

    #[test]
    fn generic_payload_decision_holds_current_exact_minimum_and_ties() {
        let tiny = SlhaElasticControlProfileAccountingV4 {
            slot_count: 1,
            present_slots: 1,
            sparse_w512_payload_bits: 512,
            dense_w128_payload_bits: 128,
            hybrid_w64_boolean_payload_bits: 256,
        };
        assert!(matches!(
            tiny.payload_decision_v6(SlhaElasticControlProfileV4::DenseW128)
                .unwrap(),
            RepresentationPayloadDecisionV1::HoldCurrentMinimum { .. }
        ));

        let three_dense = SlhaElasticControlProfileAccountingV4 {
            slot_count: 3,
            present_slots: 3,
            sparse_w512_payload_bits: 1536,
            dense_w128_payload_bits: 384,
            hybrid_w64_boolean_payload_bits: 384,
        };
        assert!(matches!(
            three_dense
                .payload_decision_v6(SlhaElasticControlProfileV4::DenseW128)
                .unwrap(),
            RepresentationPayloadDecisionV1::HoldCurrentMinimum { .. }
        ));
    }

    #[test]
    fn generic_payload_decision_preserves_unique_target_and_ambiguity() {
        let tiny = SlhaElasticControlProfileAccountingV4 {
            slot_count: 1,
            present_slots: 1,
            sparse_w512_payload_bits: 512,
            dense_w128_payload_bits: 128,
            hybrid_w64_boolean_payload_bits: 256,
        };
        let unique = tiny
            .payload_decision_v6(SlhaElasticControlProfileV4::SparseW512)
            .unwrap();
        match unique {
            RepresentationPayloadDecisionV1::UniqueTransitionCandidate { target, .. } => {
                assert_eq!(
                    target.profile_id(),
                    SlhaElasticControlProfileV4::DenseW128.id()
                );
            }
            other => panic!("expected unique payload target, observed {other:?}"),
        }

        let three_dense = SlhaElasticControlProfileAccountingV4 {
            slot_count: 3,
            present_slots: 3,
            sparse_w512_payload_bits: 1536,
            dense_w128_payload_bits: 384,
            hybrid_w64_boolean_payload_bits: 384,
        };
        assert!(matches!(
            three_dense
                .payload_decision_v6(SlhaElasticControlProfileV4::SparseW512)
                .unwrap(),
            RepresentationPayloadDecisionV1::AmbiguousMinimum { .. }
        ));
    }

    #[test]
    fn stable_payload_decision_uses_generic_stability_without_actuation() {
        use elasticxxx::{TransitionStabilityPolicyV1, TransitionStabilityStatusV1};
        use std::time::Duration;

        let report = SlhaElasticControlProfileAccountingV4 {
            slot_count: 64,
            present_slots: 1,
            sparse_w512_payload_bits: 512,
            dense_w128_payload_bits: 8192,
            hybrid_w64_boolean_payload_bits: 4288,
        };
        let policy = TransitionStabilityPolicyV1::new(
            TransitionMechanism::Reencode,
            DimensionId::REPRESENTATION,
            None,
            Some(Duration::from_secs(10)),
            None,
        )
        .unwrap();
        let mut controller = RepresentationPayloadStabilityControllerV1::new(policy).unwrap();
        let now = Instant::now();
        let observations = ObservationSnapshot::new(now, vec![]);

        let first = report
            .stable_payload_decision_v7(
                SlhaElasticControlProfileV4::DenseW128,
                &mut controller,
                &observations,
                now,
            )
            .unwrap();
        let permit = match first {
            RepresentationPayloadStableDecisionV1::Admitted {
                decision, permit, ..
            } => {
                assert!(matches!(
                    decision,
                    RepresentationPayloadDecisionV1::UniqueTransitionCandidate { .. }
                ));
                permit
            }
            other => panic!("expected admitted planning decision, observed {other:?}"),
        };
        controller.record_commit(permit, now).unwrap();

        let later = now + Duration::from_secs(1);
        let later_observations = ObservationSnapshot::new(later, vec![]);
        let deferred = SlhaElasticControlProfileAccountingV4 {
            slot_count: 1,
            present_slots: 1,
            sparse_w512_payload_bits: 512,
            dense_w128_payload_bits: 128,
            hybrid_w64_boolean_payload_bits: 256,
        }
        .stable_payload_decision_v7(
            SlhaElasticControlProfileV4::SparseW512,
            &mut controller,
            &later_observations,
            later,
        )
        .unwrap();

        match deferred {
            RepresentationPayloadStableDecisionV1::Deferred { stability, .. } => {
                assert_eq!(
                    stability.status,
                    TransitionStabilityStatusV1::CooldownActive
                );
            }
            other => panic!("expected cooldown-deferred decision, observed {other:?}"),
        }
    }

    #[test]
    fn stable_payload_trace_retains_explanation_without_permit_authority() {
        use elasticxxx::runtime::representation_payload_stability_trace::RepresentationPayloadStabilityOutcomeV1;
        use elasticxxx::TransitionStabilityPolicyV1;
        use std::time::Duration;

        let report = SlhaElasticControlProfileAccountingV4 {
            slot_count: 64,
            present_slots: 1,
            sparse_w512_payload_bits: 512,
            dense_w128_payload_bits: 8192,
            hybrid_w64_boolean_payload_bits: 4288,
        };
        let policy = TransitionStabilityPolicyV1::new(
            TransitionMechanism::Reencode,
            DimensionId::REPRESENTATION,
            None,
            Some(Duration::from_secs(10)),
            None,
        )
        .unwrap();
        let mut controller = RepresentationPayloadStabilityControllerV1::new(policy).unwrap();
        let now = Instant::now();
        let observations = ObservationSnapshot::new(now, vec![]);

        let trace = report
            .stable_payload_trace_v8(
                SlhaElasticControlProfileV4::DenseW128,
                &mut controller,
                &observations,
                now,
            )
            .unwrap();

        assert_eq!(
            trace.outcome(),
            RepresentationPayloadStabilityOutcomeV1::Admitted
        );
        assert_eq!(
            trace.decision().unique_target_profile_id(),
            Some(SlhaElasticControlProfileV4::SparseW512.id())
        );
        assert_eq!(trace.permit_generation(), Some(0));
        assert!(!trace.carries_authority());
        assert_eq!(controller.stability_gate().generation(), 0);
    }

    #[test]
    fn stable_payload_decision_preserves_tie_without_touching_stability() {
        use elasticxxx::TransitionStabilityPolicyV1;
        use std::time::Duration;

        let report = SlhaElasticControlProfileAccountingV4 {
            slot_count: 3,
            present_slots: 3,
            sparse_w512_payload_bits: 1536,
            dense_w128_payload_bits: 384,
            hybrid_w64_boolean_payload_bits: 384,
        };
        let policy = TransitionStabilityPolicyV1::new(
            TransitionMechanism::Reencode,
            DimensionId::REPRESENTATION,
            None,
            Some(Duration::from_secs(10)),
            None,
        )
        .unwrap();
        let mut controller = RepresentationPayloadStabilityControllerV1::new(policy).unwrap();
        let now = Instant::now();
        let observations = ObservationSnapshot::new(now, vec![]);

        let result = report
            .stable_payload_decision_v7(
                SlhaElasticControlProfileV4::SparseW512,
                &mut controller,
                &observations,
                now,
            )
            .unwrap();

        assert!(matches!(
            result,
            RepresentationPayloadStableDecisionV1::Ambiguous { .. }
        ));
        assert_eq!(controller.stability_gate().generation(), 0);
    }

    #[test]
    fn generic_payload_selector_does_not_mutate_cache_state() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-profile-selector-readonly");
        let slot = physical.insert(tile(17));
        physical.demote_slot(slot).unwrap();
        let before = physical.slot_generation(slot).unwrap();

        let handle = SlhaKvCacheHandleV1::new(physical);
        let report = handle.elastic_control_profile_accounting_v4().unwrap();
        let _ = report.minimum_payload_profiles_v5().unwrap();

        let after = handle
            .with_cache(|cache| cache.slot_generation(slot).unwrap())
            .unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn profile_candidates_are_stable_and_do_not_imply_preference() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-profile-candidates");
        physical.write_at(7, tile(1)).unwrap();
        physical.write_at(15, tile(2)).unwrap();

        let report = SlhaKvCacheHandleV1::new(physical)
            .elastic_control_profile_accounting_v4()
            .unwrap();
        let candidates = report.candidates();

        assert_eq!(
            candidates.map(SlhaElasticControlProfileCandidateV4::profile_id),
            [
                "slhav2-control-sparse-w512",
                "slhav2-control-dense-w128",
                "slhav2-control-hybrid-w64-boolean",
            ]
        );
        assert_eq!(
            candidates.map(SlhaElasticControlProfileCandidateV4::payload_bits),
            [
                report.sparse_w512_payload_bits(),
                report.dense_w128_payload_bits(),
                report.hybrid_w64_boolean_payload_bits(),
            ]
        );
    }

    #[test]
    fn profile_accounting_tracks_holes_without_treating_them_as_present() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-accounting-holes");
        physical.write_at(7, tile(1)).unwrap();
        physical.write_at(15, tile(2)).unwrap();
        assert!(physical.clear_slot(7));

        let report = SlhaKvCacheHandleV1::new(physical)
            .elastic_control_profile_accounting_v4()
            .unwrap();
        assert_eq!(report.slot_count(), 16);
        assert_eq!(report.present_slots(), 1);
        assert_eq!(report.absent_slots(), 15);
        assert_eq!(report.sparse_w512_payload_bits(), 512);
        assert_eq!(report.dense_w128_payload_bits(), 2048);
        assert_eq!(report.hybrid_w64_boolean_payload_bits(), 16 * 64 + 3 * 64);
    }

    #[test]
    fn dense_w128_derives_exact_physical_byte_accounting_for_all_tiers() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-dense-bytes");
        let hot = physical.insert(tile(10));
        let warm = physical.insert(tile(20));
        let cold = physical.insert(tile(30));
        let pinned = physical.insert(tile(40));

        physical.demote_slot(warm).unwrap();
        physical.demote_slot(cold).unwrap();
        physical.evict_slot(cold).unwrap();
        assert!(physical.pin(pinned));

        for (slot, tier) in [
            (hot, PhysicalTier::Hot),
            (warm, PhysicalTier::Warm),
            (cold, PhysicalTier::Cold),
            (pinned, PhysicalTier::Pinned),
        ] {
            assert_eq!(physical.tier(slot), Some(tier));
            let actual_resident = physical
                .slot_control_metadata()
                .find(|(index, _, _, _, _)| *index == slot)
                .map(|(_, _, _, resident, backing)| (resident, backing))
                .unwrap();
            assert_eq!(actual_resident, derived_bytes_for_tier_v2(tier));
        }

        let handle = SlhaKvCacheHandleV1::new(physical);
        let plane = handle.elastic_word_dense_control_plane_v2().unwrap();
        assert_eq!(plane.word_count(), 4);
        assert_eq!(
            plane.word(hot).unwrap()[1],
            SLHA_SLOT_PRESENT_BIT_V2 | SLHA_SLOT_HOT_BIT_V2
        );
        assert_eq!(
            plane.word(warm).unwrap()[1],
            SLHA_SLOT_PRESENT_BIT_V2 | SLHA_SLOT_WARM_BIT_V2
        );
        assert_eq!(
            plane.word(cold).unwrap()[1],
            SLHA_SLOT_PRESENT_BIT_V2 | SLHA_SLOT_COLD_BIT_V2
        );
        assert_eq!(
            plane.word(pinned).unwrap()[1],
            SLHA_SLOT_PRESENT_BIT_V2 | SLHA_SLOT_PINNED_BIT_V2
        );
    }

    #[test]
    fn dense_w128_generation_zero_is_unambiguous_for_present_slot() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-dense-generation-zero");
        let slot = physical.insert(tile(5));
        assert_eq!(slot, 0);
        assert_eq!(physical.slot_generation(slot), Some(0));
        let handle = SlhaKvCacheHandleV1::new(physical);

        let plane = handle.elastic_word_dense_control_plane_v2().unwrap();
        assert_eq!(plane.word(0).unwrap()[0], 0);
        assert_ne!(plane.word(0).unwrap()[1] & SLHA_SLOT_PRESENT_BIT_V2, 0);
    }

    #[test]
    fn dense_w128_can_expand_without_reintroducing_redundant_fields() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-dense-expand");
        physical.insert(tile(9));
        let handle = SlhaKvCacheHandleV1::new(physical);
        let baseline = handle.elastic_word_dense_control_plane_v2().unwrap();

        let expanded = baseline
            .reference_repack_zero_extended(ElasticWordWidthV1::from_bits(512).unwrap())
            .unwrap();
        assert_eq!(expanded.width().bits(), 512);
        assert_eq!(expanded.word_count(), 1);
        assert_eq!(&expanded.word(0).unwrap()[..2], baseline.word(0).unwrap());
        assert!(expanded.word(0).unwrap()[2..]
            .iter()
            .all(|value| *value == 0));
    }

    #[test]
    fn elastic_word_control_plane_preserves_slot_identity_generation_tier_and_bytes() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-control-plane");
        let slot0 = physical.insert(tile(7));
        let slot1 = physical.insert(tile(11));
        physical.demote_slot(slot0).unwrap();
        let handle = SlhaKvCacheHandleV1::new(physical);

        let plane = handle.elastic_word_control_plane().unwrap();
        assert_eq!(plane.width().bits(), SLHAV2_ELASTIC_WORD_CONTROL_BITS_V1);
        assert_eq!(plane.word_count(), 2);
        assert_eq!(
            plane.word(0).unwrap(),
            &[
                slot0 as u64,
                0,
                1,
                codec::WARM_PACKED_BYTES as u64,
                (codec::RESIDUAL_WORDS * 8) as u64,
                0,
                0,
                0,
            ]
        );
        assert_eq!(
            plane.word(1).unwrap(),
            &[slot1 as u64, 1, 0, codec::TILE_BYTES as u64, 0, 0, 0, 0,]
        );
    }

    #[test]
    fn elastic_word_control_plane_keeps_sparse_physical_slot_identity() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-control-plane-sparse");
        let slot0 = physical.insert(tile(1));
        let slot1 = physical.insert(tile(2));
        assert!(physical.clear_slot(slot0));
        let handle = SlhaKvCacheHandleV1::new(physical);

        let plane = handle.elastic_word_control_plane().unwrap();
        assert_eq!(plane.word_count(), 1);
        assert_eq!(plane.word(0).unwrap()[0], slot1 as u64);
        assert_eq!(plane.word(0).unwrap()[1], 1);
    }

    #[test]
    fn elastic_word_control_plane_can_expand_without_losing_v1_fields() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-control-plane-expand");
        physical.insert(tile(3));
        let handle = SlhaKvCacheHandleV1::new(physical);
        let baseline = handle.elastic_word_control_plane().unwrap();

        let expanded = baseline
            .reference_repack_zero_extended(ElasticWordWidthV1::from_bits(1024).unwrap())
            .unwrap();
        assert_eq!(expanded.width().bits(), 1024);
        assert_eq!(expanded.word_count(), 1);
        assert_eq!(&expanded.word(0).unwrap()[..8], baseline.word(0).unwrap());
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

    #[test]
    fn int4_consumer_binds_selected_precision_evidence_to_exact_physical_kv_plan() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let slot = physical.insert(tile(17));
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle,
            slot,
            RepresentationEpoch::new(3),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();

        let candidate = backend
            .fixed_width_precision_candidate()
            .unwrap()
            .expect("uniform INT4 has an honest fixed-width precision contract");
        assert_eq!(candidate.declared_precision_bits(), 4);
        assert_eq!(candidate.target(), &backend.target().representation.id);
        assert_eq!(
            candidate.target_schema_version(),
            backend.target().representation.schema_version
        );

        let preplanner = backend
            .fixed_width_precision_preplanner()
            .unwrap()
            .expect("INT4 should expose BE14e preplanning");
        let now = Instant::now();
        let signal = representation_precision_floor_signal();
        let context = PlanningContext::new().observe(signal.clone(), 4.0);
        let observations = ObservationSnapshot::new(
            now,
            vec![Observation::from_source(
                ObservationSource::runtime("slhav2-elang8a-int4"),
                signal,
                4.0,
                now,
            )],
        );
        let report = preplanner
            .screen_with_trace(
                &backend.source().representation,
                &backend.capabilities(),
                &context,
                &observations,
                now,
                ObservationEpoch::new(1),
                ResourceGeneration::new(1),
            )
            .unwrap();
        let binding = backend
            .bind_fixed_width_precision_report(report)
            .unwrap()
            .expect("selected INT4 precision evidence should compose with KV plan");

        assert_eq!(binding.candidate().declared_precision_bits(), 4);
        assert_eq!(binding.kv_plan(), &backend.transition_plan().unwrap());
        assert_eq!(
            binding.kv_plan().representation.to,
            backend.target().representation
        );
    }

    #[test]
    fn int4_precision_floor_above_four_bits_does_not_bind_to_kv_transition() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let slot = physical.insert(tile(19));
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle,
            slot,
            RepresentationEpoch::new(1),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        let preplanner = backend.fixed_width_precision_preplanner().unwrap().unwrap();
        let now = Instant::now();
        let signal = representation_precision_floor_signal();
        let context = PlanningContext::new().observe(signal.clone(), 8.0);
        let observations = ObservationSnapshot::new(
            now,
            vec![Observation::from_source(
                ObservationSource::runtime("slhav2-elang8a-int4"),
                signal,
                8.0,
                now,
            )],
        );
        let report = preplanner
            .screen_with_trace(
                &backend.source().representation,
                &backend.capabilities(),
                &context,
                &observations,
                now,
                ObservationEpoch::new(1),
                ResourceGeneration::new(1),
            )
            .unwrap();
        let error = backend
            .bind_fixed_width_precision_report(report)
            .expect_err("8-bit floor must not authorize a 4-bit candidate");
        assert!(error.contains("was not selected"));
    }

    #[test]
    fn non_uniform_slha_codecs_do_not_fabricate_fixed_width_precision_candidates() {
        for (index, flag) in [
            codec::FLAG_NF4,
            codec::FLAG_MIXED,
            codec::FLAG_TQ3,
            codec::FLAG_MIX3,
        ]
        .into_iter()
        .enumerate()
        {
            let mut encoded = tile(23 + index as u8);
            encoded[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2]
                .copy_from_slice(&flag.to_le_bytes());
            let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
            let slot = physical.insert(encoded);
            let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
                SlhaKvCacheHandleV1::new(physical),
                slot,
                RepresentationEpoch::new(1),
                SlhaKvSemanticContractV1::token_stable(),
            )
            .unwrap();
            assert!(backend.fixed_width_precision_candidate().unwrap().is_none());
            assert!(backend
                .fixed_width_precision_preplanner()
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn elasticxxx_manifest_pin_and_bridge_revision_cannot_silently_drift() {
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains("package = \"memorithm-elastic\""));
        assert!(manifest.contains(ELASTICXXX_ELANG8A_CONTRACT_REVISION));
        assert_eq!(
            ELASTICXXX_BE14D_CONTRACT_REVISION,
            ELASTICXXX_ELANG8A_CONTRACT_REVISION
        );
        assert_eq!(SLHAV2_ELASTICXXX_KV_BRIDGE_V1, 1);
        assert_eq!(SLHAV2_ELASTICXXX_KV_BRIDGE_V2, 2);
    }

    #[test]
    fn int4_capacity_and_precision_guards_converge_on_one_physical_transaction() {
        let mut physical = ElasticKvCache::new(4096, "slhav2-real-kv");
        let original = tile(31);
        let slot = physical.insert(original);
        let handle = SlhaKvCacheHandleV1::new(physical);
        let backend = SlhaKvTransitionBackendV1::bind_hot_slot(
            handle.clone(),
            slot,
            RepresentationEpoch::new(5),
            SlhaKvSemanticContractV1::token_stable(),
        )
        .unwrap();
        let (spec, eir) = resource();
        let now = Instant::now();

        let capacity = handle.capacity_observation(now).unwrap();
        let capacity_gated = gated_plan(&backend, spec.clone(), &capacity, now);
        let capacity_plan = match capacity_gated {
            BooleanKvTransitionPreflightV2::Candidate { report, plan } => {
                assert_eq!(report.evidence.truth, "true");
                plan
            }
            BooleanKvTransitionPreflightV2::Blocked(report) => {
                panic!("sufficient source-bound capacity unexpectedly blocked: {report:?}")
            }
        };

        let precision_preplanner = backend
            .fixed_width_precision_preplanner()
            .unwrap()
            .expect("INT4 should expose the fixed-width precision contract");
        let precision_signal = representation_precision_floor_signal();
        let precision_context = PlanningContext::new().observe(precision_signal.clone(), 4.0);
        let precision_observations = ObservationSnapshot::new(
            now,
            vec![Observation::from_source(
                ObservationSource::runtime("slhav2-elang8a-composed"),
                precision_signal,
                4.0,
                now,
            )],
        );
        let precision_report = precision_preplanner
            .screen_with_trace(
                &backend.source().representation,
                &backend.capabilities(),
                &precision_context,
                &precision_observations,
                now,
                ObservationEpoch::new(7),
                ResourceGeneration::new(11),
            )
            .unwrap();
        let precision_binding = backend
            .bind_fixed_width_precision_report(precision_report)
            .unwrap()
            .unwrap();
        assert_eq!(precision_binding.kv_plan(), &capacity_plan);

        let source = backend.source().clone();
        let capabilities = backend.capabilities();
        let attestations = backend.attestations();
        let mut actuator = TransactionalKvPageV1::new(
            backend,
            &eir,
            source,
            capacity_plan,
            &capabilities,
            attestations,
        )
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
}
