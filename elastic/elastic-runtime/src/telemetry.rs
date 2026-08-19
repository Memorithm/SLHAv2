//! Structured elastic telemetry.
//!
//! The mission requires distinguishing at least:
//! theoretical encoded bytes, allocator bytes, physically resident bytes,
//! GPU bytes, process RAM/VRAM, fragmentation/overhead. This struct carries
//! the canonical fields; backends fill what they can measure and leave the
//! rest unset (`None`), never fabricating numbers.

/// Elastic resource telemetry snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElasticTelemetry {
    /// Logical context tokens (workload descriptor, not a memory unit).
    pub logical_tokens: Option<u64>,
    /// Tokens currently resident in fast memory.
    pub resident_tokens: Option<u64>,
    /// Tokens held in the compressed (WARM) representation.
    pub compressed_tokens: Option<u64>,
    /// Tokens offloaded to slow storage.
    pub offloaded_tokens: Option<u64>,
    /// Tokens pinned against eviction.
    pub pinned_tokens: Option<u64>,

    /// Raw (uncompressed) KV bytes the workload would occupy.
    pub kv_bytes_raw: Option<u64>,
    /// Actual physically allocated KV bytes.
    pub kv_bytes_actual: Option<u64>,
    /// `kv_bytes_actual / kv_bytes_raw` (when both are known).
    pub compression_ratio: Option<f64>,

    /// Total VRAM.
    pub vram_total: Option<u64>,
    /// Available VRAM.
    pub vram_available: Option<u64>,
    /// VRAM pressure in `[0,1]`.
    pub vram_pressure: Option<f64>,

    /// Total RAM.
    pub ram_total: Option<u64>,
    /// Available RAM.
    pub ram_available: Option<u64>,
    /// RAM pressure in `[0,1]`.
    pub ram_pressure: Option<f64>,

    /// Allocator free bytes.
    pub allocator_free: Option<u64>,
    /// Largest contiguous free block.
    pub allocator_largest_free_block: Option<u64>,
    /// Fragmentation estimate in `[0,1]` (1 = maximally fragmented).
    pub fragmentation: Option<f64>,

    /// Adaptation counters.
    pub promotions: u64,
    /// Adaptation counters.
    pub demotions: u64,
    /// Adaptation counters.
    pub compressions: u64,
    /// Adaptation counters.
    pub decompressions: u64,
    /// Adaptation counters.
    pub offloads: u64,
    /// Adaptation counters.
    pub restores: u64,
    /// Adaptation counters.
    pub prefetches: u64,
    /// Adaptation counters.
    pub evictions: u64,

    /// Compression latency (seconds).
    pub compression_latency: Option<f64>,
    /// Restore latency (seconds).
    pub restore_latency: Option<f64>,
    /// Transition cost (seconds).
    pub transition_cost: Option<f64>,

    /// Throughput (tokens/s).
    pub throughput_tokens_per_second: Option<f64>,
    /// Request latency (seconds).
    pub request_latency: Option<f64>,
}

impl ElasticTelemetry {
    /// Create an empty snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Recompute derived fields (`compression_ratio`) from the raw values.
    pub fn refresh(&mut self) {
        if let (Some(actual), Some(raw)) = (self.kv_bytes_actual, self.kv_bytes_raw) {
            if raw > 0 {
                self.compression_ratio = Some(actual as f64 / raw as f64);
            }
        }
    }

    /// Record an adaptation of the given kind (increments the counter).
    pub fn record(&mut self, kind: &str) {
        match kind {
            "promote" => self.promotions += 1,
            "demote" => self.demotions += 1,
            "compress" => self.compressions += 1,
            "decompress" => self.decompressions += 1,
            "offload" => self.offloads += 1,
            "restore" => self.restores += 1,
            "prefetch" => self.prefetches += 1,
            "evict" => self.evictions += 1,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_ratio_derived() {
        let mut t = ElasticTelemetry::new();
        t.kv_bytes_raw = Some(1000);
        t.kv_bytes_actual = Some(250);
        t.refresh();
        assert_eq!(t.compression_ratio, Some(0.25));
    }

    #[test]
    fn counters_record() {
        let mut t = ElasticTelemetry::new();
        t.record("demote");
        t.record("demote");
        t.record("promote");
        assert_eq!(t.demotions, 2);
        assert_eq!(t.promotions, 1);
    }
}
