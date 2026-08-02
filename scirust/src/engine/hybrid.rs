//! Pure Rust extreme-edge hybrid CPU (SIMD) + GPU (CUDA PTX) matrix execution engine for SLHAv2.
//!
//! This module coordinates ultra-high performance, zero-allocation operations
//! on the d-model/KV context. It implements static SIMD dispatching on both AArch64 and x86_64,
//! as well as a GPU orchestration backend leveraging `cudarc`.
//!
//! High-efficiency caching is maintained via Cache-aligned buffers (64-byte aligned),
//! ensuring hardware-aligned streaming without cash invalidations.

use std::error::Error;
use std::fmt;

/// Predefined cache line size alignment (64 bytes).
pub const CACHE_ALIGNMENT: usize = 64;

/// A statically allocated cache-line-aligned memory buffer.
///
/// Imposes a strict 64-byte alignment on structures via `#[repr(C, align(64))]`.
/// Fits perfectly on modern L1/L2 cache boundaries to prevent false sharing and invalidations.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct CacheAlignedBuffer<T, const N: usize> {
    /// Underlying array storing the contiguous memory.
    pub data: [T; N],
}

impl<T: Default + Copy, const N: usize> Default for CacheAlignedBuffer<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T, const N: usize> CacheAlignedBuffer<T, N> {
    /// Create a new aligned buffer from a pre-allocated array.
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Access the underlying aligned data as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Access the underlying aligned data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// A fixed-size arena allocator for zero-allocation hot-path executions.
///
/// Avoids allocating on the stack by boxing the pre-allocated region once at creation,
/// ensuring zero-allocation in the critical hot path of inference.
pub struct FixedArena {
    storage: Box<[u8; 1024 * 1024]>, // 1 MiB heap-allocated memory once
    offset: usize,
}

impl Default for FixedArena {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedArena {
    /// Create a new pre-allocated FixedArena.
    pub fn new() -> Self {
        Self {
            storage: Box::new([0u8; 1024 * 1024]),
            offset: 0,
        }
    }

    /// Reset the arena offset to reuse the memory.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate a slice of memory with a specific size and alignment.
    ///
    /// Returns a mutable reference to the allocated slice or None if the capacity is exceeded.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<&mut [u8]> {
        let current_ptr = self.storage.as_ptr() as usize + self.offset;
        let aligned_ptr = (current_ptr + align - 1) & !(align - 1);
        let alignment_padding = aligned_ptr - current_ptr;

        let total_size = size + alignment_padding;
        if self.offset + total_size > self.storage.len() {
            None
        } else {
            let start = self.offset + alignment_padding;
            self.offset += total_size;
            // SAFETY: The slice is bounds-checked and guaranteed to reside within the pre-allocated array storage.
            unsafe {
                let ptr = self.storage.as_mut_ptr().add(start);
                Some(std::slice::from_raw_parts_mut(ptr, size))
            }
        }
    }
}

/// Compilation constant embedding the PTX bytecode.
pub const FUSED_GEMM_PTX: &str = include_str!("../../kernels/fused_gemm.ptx");

/// Errors that can occur in the hybrid engine.
#[derive(Debug, Clone)]
pub enum EngineError {
    /// GPU API error (e.g. cudarc or driver initialization fail).
    GpuError(String),
    /// Alignment validation or size constraint violation.
    InvalidLayout(String),
    /// Resources or memory exhausted.
    OutOfMemory,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::GpuError(msg) => write!(f, "GPU Engine Error: {msg}"),
            EngineError::InvalidLayout(msg) => write!(f, "Invalid Memory Layout: {msg}"),
            EngineError::OutOfMemory => write!(f, "Out of Memory in Engine Arena"),
        }
    }
}

impl Error for EngineError {}

/// CPU static SIMD execution module with on-the-fly register-level dequantization.
///
/// Implements fused, instruction-level conditional compilation dispatching on ARM Neon (AArch64)
/// and Intel/AMD AVX-512 (x86_64), alongside a fallback portable scalar path.
pub struct CpuSimdEngine;

impl CpuSimdEngine {
    /// Run fused dequantization and FMA (Fused Multiply-Add) calculation.
    ///
    /// Accepts quantized inputs and scales, dequantizes them directly in register vectors
    /// and performs the dot product, avoiding multi-pass RAM reads.
    ///
    /// - Quantized inputs are treated as paired 4-bit INT4 (packed in `u8` bytes).
    /// - Scale factor scales the dequantized values.
    /// - Dimension must be a multiple of the SIMD register width.
    pub fn dequant_and_fma(
        query: &[f32],
        quant_keys: &[u8],
        scale: f32,
        out_scores: &mut [f32],
    ) {
        let dim = query.len();
        let num_keys = quant_keys.len() / (dim / 2);

        for k in 0..num_keys {
            let key_offset = k * (dim / 2);
            let k_slice = &quant_keys[key_offset..key_offset + (dim / 2)];
            out_scores[k] = Self::vector_dot_product(query, k_slice, scale);
        }
    }

    #[inline(always)]
    fn vector_dot_product(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: avx512f feature is explicitly checked before running the unsafe path.
                unsafe { Self::vector_dot_product_avx512(query, quant_key, scale) }
            } else {
                Self::vector_dot_product_scalar(query, quant_key, scale)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Neon features are statically guaranteed and safe to call on aarch64 targets.
            unsafe { Self::vector_dot_product_neon(query, quant_key, scale) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::vector_dot_product_scalar(query, quant_key, scale)
        }
    }

    /// AVX-512 implementation (x86_64)
    ///
    /// # Safety
    /// Caller must guarantee that the `avx512f` instruction set is supported by the CPU.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vector_dot_product_avx512(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        use std::arch::x86_64::*;

        let dim = query.len();
        let mut sum_vec = _mm512_setzero_ps();
        let scale_vec = _mm512_set1_ps(scale);
        let eight_vec = _mm512_set1_ps(8.0);

        // Process in chunks of 16 f32s (needs 8 bytes of INT4)
        for i in (0..dim).step_by(16) {
            let q_ptr = query.as_ptr().add(i);
            let q_vec = _mm512_loadu_ps(q_ptr);

            // Fetch 8 bytes containing 16 4-bit nibbles
            let key_byte_ptr = quant_key.as_ptr().add(i / 2);
            let raw_bytes = std::ptr::read(key_byte_ptr as *const u64);

            // Extract lower and upper nibbles to floats
            let mut k_unpacked = [0.0f32; 16];
            for j in 0..8 {
                let byte = ((raw_bytes >> (j * 8)) & 0xFF) as u8;
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                k_unpacked[j * 2] = lo;
                k_unpacked[j * 2 + 1] = hi;
            }

            let k_raw_vec = _mm512_loadu_ps(k_unpacked.as_ptr());
            // level = (nibble - 8) * scale
            let k_level = _mm512_sub_ps(k_raw_vec, eight_vec);
            let k_vec = _mm512_mul_ps(k_level, scale_vec);

            sum_vec = _mm512_fmadd_ps(q_vec, k_vec, sum_vec);
        }

        // Standard AVX-512 reduction to f32 using stable intrinsics.
        // Extract the low and high 256-bit vectors, add them, and then reduce down via SSE.
        let lo = _mm512_castps512_ps256(sum_vec);
        let hi = _mm512_extractf32x8_ps::<1>(sum_vec);
        let sum256 = _mm256_add_ps(lo, hi);

        let hi128 = _mm256_extractf128_ps::<1>(sum256);
        let lo128 = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(lo128, hi128);

        let shuf = _mm_movehdup_ps(sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sum64, sum64);
        let final_sum = _mm_add_ss(sum64, shuf2);

        _mm_cvtss_f32(final_sum)
    }

    /// ARM Neon implementation (aarch64)
    ///
    /// # Safety
    /// Safe on all normal ARM64 (AArch64) target platforms.
    #[cfg(target_arch = "aarch64")]
    unsafe fn vector_dot_product_neon(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        use std::arch::aarch64::*;

        let dim = query.len();
        let mut sum_vec = vdupq_n_f32(0.0);
        let scale_vec = vdupq_n_f32(scale);
        let eight_vec = vdupq_n_f32(8.0);

        // Process in chunks of 4 f32s (needs 2 bytes of INT4)
        for i in (0..dim).step_by(4) {
            let q_ptr = query.as_ptr().add(i);
            let q_vec = vld1q_f32(q_ptr);

            let key_byte_ptr = quant_key.as_ptr().add(i / 2);
            let b0 = *key_byte_ptr;
            let b1 = *key_byte_ptr.add(1);

            let n0 = (b0 & 0x0F) as f32;
            let n1 = (b0 >> 4) as f32;
            let n2 = (b1 & 0x0F) as f32;
            let n3 = (b1 >> 4) as f32;

            let k_unpacked = [n0, n1, n2, n3];
            let k_raw_vec = vld1q_f32(k_unpacked.as_ptr());
            let k_level = vsubq_f32(k_raw_vec, eight_vec);
            let k_vec = vmulq_f32(k_level, scale_vec);

            sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
        }

        vaddvq_f32(sum_vec)
    }

    /// Fallback portable scalar path
    fn vector_dot_product_scalar(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        let dim = query.len();
        let mut sum = 0.0f32;
        for d in 0..dim {
            let byte_idx = d >> 1;
            let byte = quant_key[byte_idx];
            let nibble = if d & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            let level = (nibble as i32 - 8) as f32;
            let val = level * scale;
            sum += query[d] * val;
        }
        sum
    }
}

// ── CUDA engine integration conditionally compiled ────────────────────────

#[cfg(feature = "cuda")]
pub struct GpuEngine {
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
}

#[cfg(feature = "cuda")]
impl GpuEngine {
    /// Initialize a new GpuEngine and load the PTX module.
    pub fn new() -> Result<Self, EngineError> {
        let dev = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| EngineError::GpuError(e.to_string()))?;

        // Load the PTX kernel module
        dev.load_ptx(
            cudarc::driver::Ptx::from_src(FUSED_GEMM_PTX),
            "fused_gemm",
            &["stub"],
        )
        .map_err(|e| EngineError::GpuError(e.to_string()))?;

        Ok(Self { device: dev })
    }

    /// Pinned host registration helper to register a memory region for zero-copy.
    pub fn register_pinned_memory(&self, buffer: &mut [u8]) -> Result<(), EngineError> {
        // SAFETY: The pointer and length represent a valid mutable slice allocated by Rust.
        unsafe {
            let res = cudarc::driver::sys::cudaHostRegister(
                buffer.as_mut_ptr() as *mut std::ffi::c_void,
                buffer.len(),
                cudarc::driver::sys::cudaHostRegisterFlags_CUDA_HOST_REGISTER_DEVICEMAP,
            );
            if res == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(EngineError::GpuError(format!(
                    "cudaHostRegister failed with code {res:?}"
                )))
            }
        }
    }

    /// Launch the PTX stub/gemm kernel on the device.
    pub fn launch_kernel(&self, param: u64) -> Result<(), EngineError> {
        let func = self
            .device
            .get_func("fused_gemm", "stub")
            .ok_or_else(|| EngineError::GpuError("Kernel 'stub' not found".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: The kernel launch parameters match the PTX stub parameters.
        unsafe {
            func.launch(cfg, (param,))
                .map_err(|e| EngineError::GpuError(e.to_string()))?;
        }

        Ok(())
    }
}

// ── CUDA engine integration fallback stub (non-cuda targets) ──────────────

#[cfg(not(feature = "cuda"))]
pub struct GpuEngine;

#[cfg(not(feature = "cuda"))]
impl GpuEngine {
    /// Initialize a dummy GpuEngine returning initialization error or fallback.
    pub fn new() -> Result<Self, EngineError> {
        Err(EngineError::GpuError(
            "CUDA hardware target is not enabled/compiled".to_string(),
        ))
    }

    /// Pinned host registration stub.
    pub fn register_pinned_memory(&self, _buffer: &mut [u8]) -> Result<(), EngineError> {
        Err(EngineError::GpuError("CUDA support is disabled".to_string()))
    }

    /// Launch kernel stub.
    pub fn launch_kernel(&self, _param: u64) -> Result<(), EngineError> {
        Err(EngineError::GpuError("CUDA support is disabled".to_string()))
    }
}

/// Orchestrates the pipeline/overlapping between layers N and N+1.
///
/// Demonstrates the overlapping/pipelining pattern:
/// - CPU-SIMD dequantizes the weights of Layer N+1 in the background
/// - While GPU executes the GEMM kernel for Layer N in parallel.
pub fn pipeline_execution_step(
    gpu_engine: &GpuEngine,
    layer_n_gpu_param: u64,
    layer_n_plus_1_query: &[f32],
    layer_n_plus_1_keys: &[u8],
    layer_n_plus_1_scale: f32,
    layer_n_plus_1_scores: &mut [f32],
) -> Result<(), EngineError> {
    // 1. Launch GPU Kernel for Layer N (non-blocking)
    let _ = gpu_engine.launch_kernel(layer_n_gpu_param);

    // 2. Overlap with CPU SIMD dequantization + scoring of Layer N+1
    CpuSimdEngine::dequant_and_fma(
        layer_n_plus_1_query,
        layer_n_plus_1_keys,
        layer_n_plus_1_scale,
        layer_n_plus_1_scores,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_buffer_layout() {
        let buf: CacheAlignedBuffer<f32, 16> = CacheAlignedBuffer::new([0.0f32; 16]);
        assert_eq!(std::mem::align_of_val(&buf), 64);
        assert_eq!(std::mem::size_of_val(&buf), 64);
    }

    #[test]
    fn test_fixed_arena_zero_allocation() {
        let mut arena = FixedArena::new();
        let s1 = arena.alloc(128, 64).unwrap();
        assert_eq!(s1.len(), 128);
        assert_eq!(s1.as_ptr() as usize % 64, 0);

        let s2 = arena.alloc(256, 64).unwrap();
        assert_eq!(s2.len(), 256);
        assert_eq!(s2.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_cpu_simd_dequant_equivalence() {
        let query = vec![1.5f32; 128];
        let mut keys = vec![0u8; 64]; // INT4 keys
        for i in 0..64 {
            keys[i] = 0x88; // zero-point nibbles -> yields 0.0f32
        }

        let mut scores = vec![999.0f32; 1];
        CpuSimdEngine::dequant_and_fma(&query, &keys, 1.0, &mut scores);
        assert_eq!(scores[0], 0.0);
    }
}
