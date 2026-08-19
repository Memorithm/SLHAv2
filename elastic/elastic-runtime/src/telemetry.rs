//! Structured elastic telemetry.
//!
//! Backends fill measured fields and leave unknown values as `None`. Derived
//! values are recomputed from their current inputs on every [`refresh`] call;
//! stale ratios/pressures are never retained when an input disappears or is
//! invalid.

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
    /// `kv_bytes_actual / kv_bytes_raw` (when both are known and raw > 0).
    pub compression_ratio: Option<f64>,

    /// Total VRAM.
    pub vram_total: Option<u64>,
    /// Available VRAM.
    pub vram_available: Option<u64>,
    /// Derived VRAM pressure in `[0,1]`.
    pub vram_pressure: Option<f64>,

    /// Total RAM.
    pub ram_total: Option<u64>,
    /// Available RAM.
    pub ram_available: Option<u64>,
    /// Derived RAM pressure in `[0,1]`.
    pub ram_pressure: Option<f64>,

    /// Allocator free bytes.
    pub allocator_free: Option<u64>,
    /// Largest contiguous free block.
    pub allocator_largest_free_block: Option<u64>,
    /// Derived fragmentation estimate in `[0,1]` (`0` = one contiguous free
    /// block). Unknown when allocator geometry is invalid or unavailable.
    pub fragmentation: Option<f64>,

    /// Promotion count.
    pub promotions: u64,
    /// Demotion count.
    pub demotions: u64,
    /// Compression count.
    pub compressions: u64,
    /// Decompression count.
    pub decompressions: u64,
    /// Offload count.
    pub offloads: u64,
    /// Restore count.
    pub restores: u64,
    /// Prefetch count.
    pub prefetches: u64,
    /// Eviction count.
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

    /// Recompute every derived field from the current raw observations.
    pub fn refresh(&mut self) {
        self.compression_ratio = match (self.kv_bytes_actual, self.kv_bytes_raw) {
            (Some(actual), Some(raw)) if raw > 0 => Some(actual as f64 / raw as f64),
            _ => None,
        };
        self.vram_pressure = pressure(self.vram_total, self.vram_available);
        self.ram_pressure = pressure(self.ram_total, self.ram_available);
        self.fragmentation = match (
            self.allocator_free,
            self.allocator_largest_free_block,
        ) {
            (Some(0), Some(0)) => Some(0.0),
            (Some(free), Some(largest)) if free > 0 && largest <= free => {
                Some(1.0 - largest as f64 / free as f64)
            }
            _ => None,
        };
    }

    /// Record an adaptation kind. Counters saturate rather than wrapping or
    /// panicking in debug builds after extreme runtimes.
    pub fn record(&mut self, kind: &str) {
        let counter = match kind {
            "promote" => Some(&mut self.promotions),
            "demote" => Some(&mut self.demotions),
            "compress" => Some(&mut self.compressions),
            "decompress" => Some(&mut self.decompressions),
            "offload" => Some(&mut self.offloads),
            "restore" => Some(&mut self.restores),
            "prefetch" => Some(&mut self.prefetches),
            "evict" => Some(&mut self.evictions),
            _ => None,
        };
        if let Some(counter) = counter {
            *counter = counter.saturating_add(1);
        }
    }
}

fn pressure(total: Option<u64>, available: Option<u64>) -> Option<f64> {
    match (total, available) {
        (Some(total), Some(available)) if total > 0 && available <= total => {
            Some((total - available) as f64 / total as f64)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_derives_current_ratios_and_pressures() {
        let mut telemetry = ElasticTelemetry::new();
        telemetry.kv_bytes_raw = Some(1000);
        telemetry.kv_bytes_actual = Some(250);
        telemetry.vram_total = Some(1000);
        telemetry.vram_available = Some(250);
        telemetry.ram_total = Some(2000);
        telemetry.ram_available = Some(1000);
        telemetry.allocator_free = Some(1000);
        telemetry.allocator_largest_free_block = Some(600);
        telemetry.refresh();
        assert_eq!(telemetry.compression_ratio, Some(0.25));
        assert_eq!(telemetry.vram_pressure, Some(0.75));
        assert_eq!(telemetry.ram_pressure, Some(0.5));
        assert_eq!(telemetry.fragmentation, Some(0.4));
    }

    #[test]
    fn refresh_clears_stale_derived_values_when_inputs_become_invalid() {
        let mut telemetry = ElasticTelemetry::new();
        telemetry.kv_bytes_raw = Some(100);
        telemetry.kv_bytes_actual = Some(50);
        telemetry.vram_total = Some(100);
        telemetry.vram_available = Some(20);
        telemetry.refresh();
        assert!(telemetry.compression_ratio.is_some());
        assert!(telemetry.vram_pressure.is_some());

        telemetry.kv_bytes_raw = Some(0);
        telemetry.vram_available = Some(101);
        telemetry.refresh();
        assert_eq!(telemetry.compression_ratio, None);
        assert_eq!(telemetry.vram_pressure, None);
    }

    #[test]
    fn counters_saturate() {
        let mut telemetry = ElasticTelemetry::new();
        telemetry.demotions = u64::MAX;
        telemetry.record("demote");
        assert_eq!(telemetry.demotions, u64::MAX);
        telemetry.record("promote");
        assert_eq!(telemetry.promotions, 1);
    }
}
