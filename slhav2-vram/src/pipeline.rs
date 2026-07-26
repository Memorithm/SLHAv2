use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::memory::VramMemoryPool;
use crate::traits::{DeviceEngine, DevicePointer, VramResult};

/// An asynchronous execution pipeline for batched low-rank MatMul +
/// TurboQuant operations.
///
/// The pipeline manages:
/// - A memory pool for scratch allocations
/// - Operation submission and sequencing
/// - Lazy synchronization for batched execution
pub struct LowRankPipeline {
    pool: Arc<VramMemoryPool>,
    ops_submitted: AtomicU64,
}

impl LowRankPipeline {
    /// Create a new pipeline backed by the given memory pool.
    ///
    /// All device operations use the pool's underlying engine.
    pub fn new(pool: VramMemoryPool) -> Self {
        LowRankPipeline {
            pool: Arc::new(pool),
            ops_submitted: AtomicU64::new(0),
        }
    }

    pub fn engine(&self) -> &dyn DeviceEngine {
        self.pool.engine()
    }

    pub fn pool(&self) -> &VramMemoryPool {
        self.pool.as_ref()
    }

    /// Submit a low-rank MatMul operation on the given pointers.
    pub fn submit_matmul(
        &self,
        input: &DevicePointer,
        weights: &DevicePointer,
        output: &DevicePointer,
        dim_m: usize,
        dim_n: usize,
        dim_k: usize,
    ) -> VramResult<()> {
        self.pool
            .engine()
            .launch_lowrank_matmul(input, weights, output, dim_m, dim_n, dim_k)?;
        self.ops_submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Convenience: allocate from pool, copy input, submit matmul, copy result
    /// back, free scratch buffers.
    pub fn run_inference(
        &self,
        host_input: &[f32],
        host_weights: &[u8],
        host_output: &mut [f32],
        dim_m: usize,
        dim_n: usize,
        dim_k: usize,
    ) -> VramResult<()> {
        let input_bytes = host_input.len() * 4;
        let output_bytes = host_output.len() * 4;

        let d_input = self.pool.allocate(input_bytes)?;
        let d_weights = self.pool.allocate(host_weights.len())?;
        let d_output = self.pool.allocate(output_bytes)?;

        let engine = self.pool.engine();

        let input_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(host_input.as_ptr() as *const u8, input_bytes)
        };

        engine.copy_to_device(input_u8, &d_input)?;
        engine.copy_to_device(host_weights, &d_weights)?;

        self.submit_matmul(&d_input, &d_weights, &d_output, dim_m, dim_n, dim_k)?;

        engine.synchronize()?;

        let mut output_u8 = vec![0u8; output_bytes];
        engine.copy_to_host(&d_output, &mut output_u8)?;

        let output_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output_u8.as_ptr() as *const f32, host_output.len())
        };
        host_output.copy_from_slice(output_f32);

        self.pool.free(d_input)?;
        self.pool.free(d_weights)?;
        self.pool.free(d_output)?;

        Ok(())
    }

    /// Number of matmul operations submitted since creation.
    pub fn ops_submitted(&self) -> u64 {
        self.ops_submitted.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cpu_ref::CpuRefBackend;

    #[test]
    fn pipeline_submit_and_sync() {
        let pool = VramMemoryPool::new(
            Box::new(CpuRefBackend::new(256)),
            64 * 1024,
        )
        .unwrap();
        let pipeline = LowRankPipeline::new(pool);

        let m = 4;
        let n = 8;
        let k = 16;

        let input = vec![1.0f32; m * k];

        let packed_bytes = n * (k / 2);
        let scales_bytes = n * (k / 16) * 4;
        let mut weights_int4 = vec![0u8; packed_bytes + scales_bytes];

        // Nibble=1 → signed_val=1, scale=1.0
        for byte in weights_int4[..packed_bytes].iter_mut() {
            *byte = 0x11; // both nibbles = 1 (signed_val=1)
        }
        let scale_start = packed_bytes;
        let scale_bytes = &mut weights_int4[scale_start..];
        for chunk in scale_bytes.chunks_mut(4) {
            chunk.copy_from_slice(&1.0f32.to_ne_bytes());
        }

        let mut output = vec![0.0f32; m * n];

        pipeline
            .run_inference(&input, &weights_int4, &mut output, m, n, k)
            .unwrap();

        assert!(output.iter().any(|&v| v != 0.0));
        assert_eq!(pipeline.ops_submitted(), 1);
    }
}
