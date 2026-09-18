//! Versioned, read-only bridge from the SLHAv2 physical KV cache to ElasticXxx.
//!
//! SLHAv2 remains the owner of tile layout, residency and physical cache
//! mutation. This module exposes only current resident-budget headroom as the
//! source-bound `free-capacity` evidence consumed by ElasticXxx BE14d. It does
//! not ask ElasticXxx to mutate the cache and it does not bypass SLHAv2's
//! transactional cache controller.

use std::fmt;
use std::time::Instant;

use elasticxxx_core::resource::LogicalResourceId;
use elasticxxx_kv::boolean_admission::KvCapacityObservationV1;

use crate::elastic_cache::ElasticKvCache;

/// Wire/semantic version of this SLHAv2 -> ElasticXxx evidence adapter.
pub const ELASTICXXX_KV_CAPACITY_BRIDGE_SCHEMA_VERSION: u16 = 1;

/// Exact ElasticXxx merge qualified by this adapter revision.
pub const ELASTICXXX_KV_CAPACITY_BRIDGE_REVISION: &str = "107123123fd4e3430e1ee7ed2f74fdd91268f3a2";

/// Fail-closed adapter errors. No error value authorizes planning or actuation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElasticXxxKvCapacityBridgeError {
    /// The existing SLHAv2 cache identity is not a valid ElasticXxx logical id.
    InvalidResourceId(String),
    /// The host `usize` capacity could not be represented as `u64`.
    CapacityOutOfRange,
}

impl fmt::Display for ElasticXxxKvCapacityBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResourceId(message) => {
                write!(
                    f,
                    "SLHAv2 KV resource id is invalid for ElasticXxx: {message}"
                )
            }
            Self::CapacityOutOfRange => {
                f.write_str("SLHAv2 free KV capacity does not fit the ElasticXxx u64 contract")
            }
        }
    }
}

impl std::error::Error for ElasticXxxKvCapacityBridgeError {}

/// Publish the cache's current resident-budget headroom as source-bound
/// ElasticXxx BE14d evidence.
///
/// `free_bytes()` is SLHAv2's own physical resident-budget accounting. This
/// bridge neither infers freed HBM/DRAM traffic nor claims that any future
/// representation transition will succeed. ElasticXxx still evaluates the
/// Boolean guard and SLHAv2 remains authoritative for physical mutation and
/// rollback.
pub fn kv_capacity_observation(
    cache: &ElasticKvCache,
    observed_at: Instant,
) -> Result<KvCapacityObservationV1, ElasticXxxKvCapacityBridgeError> {
    let resource = LogicalResourceId::new(cache.resource_id())
        .map_err(|error| ElasticXxxKvCapacityBridgeError::InvalidResourceId(error.to_string()))?;
    let free_capacity_bytes = u64::try_from(cache.free_bytes())
        .map_err(|_| ElasticXxxKvCapacityBridgeError::CapacityOutOfRange)?;
    Ok(KvCapacityObservationV1::measured(
        resource,
        free_capacity_bytes,
        observed_at,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticxxx_core::resource::ObservationSignalId;

    #[test]
    fn bridge_uses_exact_cache_identity_and_physical_headroom() {
        let now = Instant::now();
        let mut cache = ElasticKvCache::new(1024, "slha-kv-bridge");
        cache.insert([0_u8; crate::codec::TILE_BYTES]);

        let evidence = kv_capacity_observation(&cache, now).unwrap();
        assert_eq!(
            evidence
                .planning_context()
                .get(ObservationSignalId::FREE_CAPACITY),
            Some(896.0)
        );
        let observation = evidence
            .observations()
            .get(ObservationSignalId::FREE_CAPACITY)
            .unwrap();
        assert_eq!(observation.source().to_string(), "resource:slha-kv-bridge");
        assert!(observation.is_valid());
        assert_eq!(observation.value(), 896.0);
    }

    #[test]
    fn invalid_existing_cache_identity_fails_closed() {
        let cache = ElasticKvCache::new(1024, "");
        let error = kv_capacity_observation(&cache, Instant::now()).unwrap_err();
        assert!(matches!(
            error,
            ElasticXxxKvCapacityBridgeError::InvalidResourceId(_)
        ));
    }
}
