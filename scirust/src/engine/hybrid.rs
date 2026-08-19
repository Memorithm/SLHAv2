//! Experimental hybrid CPU SIMD + CUDA research engine.
//!
//! This module is deliberately quarantined from the production GPU path.
//! Production CUDA execution lives in `slhav2-vram`; this module exists for
//! experiments and must fail closed on malformed layouts or absent kernels.
//! The historical no-op PTX stub is not shipped.

use std::error::Error;
use std::fmt;

/// Cache-line alignment used by host-side scratch buffers.
pub const CACHE_ALIGNMENT: usize = 64;

/// Cache-line-aligned fixed-size buffer.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct CacheAlignedBuffer<T, const N: usize> {
    /// Underlying contiguous array.
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
    /// Create a buffer from an existing array.
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Borrow the underlying array as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the underlying array as a slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Fixed 1 MiB scratch arena. Allocation happens once at construction; hot
/// path allocations only bump an offset.
pub struct FixedArena {
    storage: Box<[u8; 1024 * 1024]>,
    offset: usize,
}

impl Default for FixedArena {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self {
            storage: Box::new([0u8; 1024 * 1024]),
            offset: 0,
        }
    }

    /// Reuse all arena storage.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate `size` bytes at `align` alignment.
    ///
    /// Invalid alignment, integer overflow, zero-sized allocations and
    /// exhaustion return `None`; no unchecked arithmetic is performed.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<&mut [u8]> {
        if size == 0 || align == 0 || !align.is_power_of_two() {
            return None;
        }
        let base = self.storage.as_ptr() as usize;
        let current = base.checked_add(self.offset)?;
        let aligned = current.checked_add(align - 1)? & !(align - 1);
        let padding = aligned.checked_sub(current)?;
        let total = size.checked_add(padding)?;
        let end = self.offset.checked_add(total)?;
        if end > self.storage.len() {
            return None;
        }
        let start = self.offset.checked_add(padding)?;
        self.offset = end;
        Some(&mut self.storage[start..start + size])
    }
}

/// The removed legacy PTX stub. Kept as an empty compatibility constant so
/// callers cannot accidentally execute fake work.
pub const FUSED_GEMM_PTX: &str = "";

/// Errors produced by the experimental engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// CUDA/driver/module error.
    GpuError(String),
    /// Invalid query/key/output/alignment layout.
    InvalidLayout(String),
    /// Scratch/device memory exhausted.
    OutOfMemory,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuError(msg) => write!(f, "GPU Engine Error: {msg}"),
            Self::InvalidLayout(msg) => write!(f, "Invalid Memory Layout: {msg}"),
            Self::OutOfMemory => f.write_str("Out of Memory in Engine Arena"),
        }
    }
}

impl Error for EngineError {}

/// CPU INT4 scoring engine with architecture-specific SIMD acceleration.
pub struct CpuSimdEngine;

impl CpuSimdEngine {
    /// Compatibility wrapper for historical callers.
    ///
    /// Invalid input fails closed: the supplied output is zeroed and no panic
    /// escapes. New code should use [`Self::try_dequant_and_fma`] to receive a
    /// structured error.
    pub fn dequant_and_fma(query: &[f32], quant_keys: &[u8], scale: f32, out_scores: &mut [f32]) {
        out_scores.fill(0.0);
        let _ = Self::try_dequant_and_fma(query, quant_keys, scale, out_scores);
    }

    /// Validate and score all packed INT4 keys.
    ///
    /// Returns the number of scores written. A key occupies
    /// `ceil(query.len()/2)` bytes. The key buffer must contain a whole number
    /// of keys and the output must be large enough for all of them.
    pub fn try_dequant_and_fma(
        query: &[f32],
        quant_keys: &[u8],
        scale: f32,
        out_scores: &mut [f32],
    ) -> Result<usize, EngineError> {
        let dim = query.len();
        if dim == 0 {
            return Err(EngineError::InvalidLayout(
                "query dimension must be non-zero".to_string(),
            ));
        }
        if !scale.is_finite() {
            return Err(EngineError::InvalidLayout(
                "scale must be finite".to_string(),
            ));
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::InvalidLayout(
                "query contains non-finite values".to_string(),
            ));
        }
        let packed_per_key = dim.div_ceil(2);
        if quant_keys.len() % packed_per_key != 0 {
            return Err(EngineError::InvalidLayout(format!(
                "packed key buffer length {} is not a multiple of {packed_per_key}",
                quant_keys.len()
            )));
        }
        let num_keys = quant_keys.len() / packed_per_key;
        if out_scores.len() < num_keys {
            return Err(EngineError::InvalidLayout(format!(
                "output has {} scores but {num_keys} are required",
                out_scores.len()
            )));
        }

        for (k, out) in out_scores.iter_mut().take(num_keys).enumerate() {
            let start = k * packed_per_key;
            let key = &quant_keys[start..start + packed_per_key];
            *out = Self::vector_dot_product(query, key, scale)?;
        }
        Ok(num_keys)
    }

    #[inline]
    fn vector_dot_product(query: &[f32], quant_key: &[u8], scale: f32) -> Result<f32, EngineError> {
        let dim = query.len();
        let needed = dim.div_ceil(2);
        if dim == 0 || quant_key.len() != needed {
            return Err(EngineError::InvalidLayout(
                "query/key dimensions do not match".to_string(),
            ));
        }

        // SIMD implementations process complete vectors only. Odd or
        // non-register-multiple dimensions use the bounds-checked scalar path.
        if dim.is_multiple_of(16) {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: feature detection and exact packed-key validation
                    // above satisfy the implementation preconditions.
                    return Ok(unsafe { Self::vector_dot_product_avx512(query, quant_key, scale) });
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        if dim.is_multiple_of(4) {
            // SAFETY: Advanced SIMD is part of the AArch64 baseline and the
            // validated key has two nibbles for every query element.
            return Ok(unsafe { Self::vector_dot_product_neon(query, quant_key, scale) });
        }

        Ok(Self::vector_dot_product_scalar(query, quant_key, scale))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vector_dot_product_avx512(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        use std::arch::x86_64::*;

        let mut sum = _mm512_setzero_ps();
        let scale_vec = _mm512_set1_ps(scale);
        let eight = _mm512_set1_ps(8.0);
        for i in (0..query.len()).step_by(16) {
            let q = _mm512_loadu_ps(query.as_ptr().add(i));
            let raw = std::ptr::read_unaligned(quant_key.as_ptr().add(i / 2) as *const u64);
            let mut unpacked = [0.0f32; 16];
            for j in 0..8 {
                let byte = ((raw >> (j * 8)) & 0xff) as u8;
                unpacked[2 * j] = (byte & 0x0f) as f32;
                unpacked[2 * j + 1] = (byte >> 4) as f32;
            }
            let levels = _mm512_sub_ps(_mm512_loadu_ps(unpacked.as_ptr()), eight);
            let values = _mm512_mul_ps(levels, scale_vec);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(q, values));
        }

        let lo = _mm512_castps512_ps256(sum);
        let hi = _mm512_extractf32x8_ps::<1>(sum);
        let sum256 = _mm256_add_ps(lo, hi);
        let lo128 = _mm256_castps256_ps128(sum256);
        let hi128 = _mm256_extractf128_ps::<1>(sum256);
        let sum128 = _mm_add_ps(lo128, hi128);
        let pair = _mm_add_ps(sum128, _mm_movehdup_ps(sum128));
        _mm_cvtss_f32(_mm_add_ss(pair, _mm_movehl_ps(pair, pair)))
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn vector_dot_product_neon(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        use std::arch::aarch64::*;

        let mut sum = vdupq_n_f32(0.0);
        let scale_vec = vdupq_n_f32(scale);
        let eight = vdupq_n_f32(8.0);
        for i in (0..query.len()).step_by(4) {
            let q = vld1q_f32(query.as_ptr().add(i));
            let b0 = *quant_key.as_ptr().add(i / 2);
            let b1 = *quant_key.as_ptr().add(i / 2 + 1);
            let unpacked = [
                (b0 & 0x0f) as f32,
                (b0 >> 4) as f32,
                (b1 & 0x0f) as f32,
                (b1 >> 4) as f32,
            ];
            let levels = vsubq_f32(vld1q_f32(unpacked.as_ptr()), eight);
            sum = vfmaq_f32(sum, q, vmulq_f32(levels, scale_vec));
        }
        vaddvq_f32(sum)
    }

    fn vector_dot_product_scalar(query: &[f32], quant_key: &[u8], scale: f32) -> f32 {
        query
            .iter()
            .enumerate()
            .map(|(d, &q)| {
                let byte = quant_key[d >> 1];
                let nibble = if d & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                q * (nibble as i32 - 8) as f32 * scale
            })
            .sum()
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaHostRegister(ptr: *mut std::ffi::c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut std::ffi::c_void) -> i32;
}

#[cfg(feature = "cuda")]
const CUDA_HOST_REGISTER_DEVICEMAP: u32 = 0x02;

/// Experimental CUDA engine. The production backend is `slhav2-vram`.
#[cfg(feature = "cuda")]
pub struct GpuEngine {
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
}

#[cfg(feature = "cuda")]
impl GpuEngine {
    /// Load an explicitly supplied non-empty PTX module and kernel.
    pub fn new_with_ptx(ptx_src: &'static str, kernel_name: &'static str) -> Result<Self, EngineError> {
        use cudarc::nvrtc::Ptx;
        if ptx_src.trim().is_empty() || kernel_name.trim().is_empty() {
            return Err(EngineError::GpuError(
                "explicit non-empty PTX and kernel name are required".to_string(),
            ));
        }
        let device = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| EngineError::GpuError(format!("{e:?}")))?;
        device
            .load_ptx(Ptx::from_src(ptx_src), "fused_gemm", &[kernel_name])
            .map_err(|e| EngineError::GpuError(format!("{e:?}")))?;
        Ok(Self { device })
    }

    /// Legacy constructor: fails closed because the bundled stub was removed.
    pub fn new() -> Result<Self, EngineError> {
        Err(EngineError::GpuError(
            "GpuEngine::new has no production kernel; use new_with_ptx or slhav2-vram"
                .to_string(),
        ))
    }

    /// Register non-empty host storage and return an RAII unregister guard.
    pub fn register_pinned_memory<'a>(
        &self,
        buffer: &'a mut [u8],
    ) -> Result<PinnedHostRegion<'a>, EngineError> {
        if buffer.is_empty() {
            return Err(EngineError::InvalidLayout(
                "cannot pin an empty host region".to_string(),
            ));
        }
        let result = unsafe {
            cudaHostRegister(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                CUDA_HOST_REGISTER_DEVICEMAP,
            )
        };
        if result != 0 {
            return Err(EngineError::GpuError(format!(
                "cudaHostRegister failed with code {result}"
            )));
        }
        Ok(PinnedHostRegion {
            base: buffer.as_mut_ptr(),
            len: buffer.len(),
            _lifetime: std::marker::PhantomData,
        })
    }

    /// Launch the explicitly loaded kernel.
    pub fn launch_kernel(&self, kernel_name: &str, param: u64) -> Result<(), EngineError> {
        use cudarc::driver::LaunchAsync;
        if kernel_name.trim().is_empty() {
            return Err(EngineError::InvalidLayout(
                "kernel name must be non-empty".to_string(),
            ));
        }
        let func = self
            .device
            .get_func("fused_gemm", kernel_name)
            .ok_or_else(|| EngineError::GpuError(format!("kernel '{kernel_name}' not found")))?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (param,))
                .map_err(|e| EngineError::GpuError(format!("{e:?}")))?;
        }
        Ok(())
    }
}

/// RAII ownership of a CUDA host registration.
#[cfg(feature = "cuda")]
pub struct PinnedHostRegion<'a> {
    base: *mut u8,
    len: usize,
    _lifetime: std::marker::PhantomData<&'a mut [u8]>,
}

#[cfg(feature = "cuda")]
impl Drop for PinnedHostRegion<'_> {
    fn drop(&mut self) {
        let _ = unsafe { cudaHostUnregister(self.base.cast()) };
    }
}

#[cfg(feature = "cuda")]
impl PinnedHostRegion<'_> {
    pub fn as_ptr(&self) -> *const u8 {
        self.base
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Non-CUDA fail-closed placeholder.
#[cfg(not(feature = "cuda"))]
pub struct GpuEngine;

#[cfg(not(feature = "cuda"))]
impl GpuEngine {
    pub fn new() -> Result<Self, EngineError> {
        Err(EngineError::GpuError("CUDA support is disabled".to_string()))
    }

    pub fn new_with_ptx(
        _ptx_src: &'static str,
        _kernel_name: &'static str,
    ) -> Result<Self, EngineError> {
        Err(EngineError::GpuError("CUDA support is disabled".to_string()))
    }

    pub fn register_pinned_memory(&self, _buffer: &mut [u8]) -> Result<(), EngineError> {
        Err(EngineError::GpuError("CUDA support is disabled".to_string()))
    }

    pub fn launch_kernel(&self, _kernel_name: &str, _param: u64) -> Result<(), EngineError> {
        Err(EngineError::GpuError("CUDA support is disabled".to_string()))
    }
}

/// Experimental pipeline step. The GPU launch is validated first; CPU scoring
/// then runs through the fail-closed compatibility wrapper.
pub fn pipeline_execution_step(
    gpu_engine: &GpuEngine,
    kernel_name: &str,
    layer_n_gpu_param: u64,
    layer_n_plus_1_query: &[f32],
    layer_n_plus_1_keys: &[u8],
    layer_n_plus_1_scale: f32,
    layer_n_plus_1_scores: &mut [f32],
) -> Result<(), EngineError> {
    gpu_engine.launch_kernel(kernel_name, layer_n_gpu_param)?;
    CpuSimdEngine::try_dequant_and_fma(
        layer_n_plus_1_query,
        layer_n_plus_1_keys,
        layer_n_plus_1_scale,
        layer_n_plus_1_scores,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_alignment_is_stable() {
        let buf = CacheAlignedBuffer::new([0.0f32; 16]);
        assert_eq!(std::mem::align_of_val(&buf), CACHE_ALIGNMENT);
        assert_eq!(std::mem::size_of_val(&buf), CACHE_ALIGNMENT);
    }

    #[test]
    fn fixed_arena_rejects_bad_alignment_and_overflow() {
        let mut arena = FixedArena::new();
        assert!(arena.alloc(1, 0).is_none());
        assert!(arena.alloc(1, 3).is_none());
        assert!(arena.alloc(128, 64).is_some());
        arena.reset();
        let s = arena.alloc(128, 64).unwrap();
        assert_eq!(s.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn zero_dimension_and_short_output_fail_without_panicking() {
        let mut out = [9.0f32; 1];
        assert!(CpuSimdEngine::try_dequant_and_fma(&[], &[], 1.0, &mut out).is_err());
        CpuSimdEngine::dequant_and_fma(&[], &[], 1.0, &mut out);
        assert_eq!(out, [0.0]);

        let q = [1.0f32; 16];
        let keys = [0x88u8; 16]; // two keys, eight bytes each
        assert!(CpuSimdEngine::try_dequant_and_fma(&q, &keys, 1.0, &mut out).is_err());
    }

    #[test]
    fn scalar_and_simd_contract_scores_zero_point() {
        let query = [1.5f32; 128];
        let keys = [0x88u8; 64];
        let mut scores = [999.0f32; 1];
        assert_eq!(
            CpuSimdEngine::try_dequant_and_fma(&query, &keys, 1.0, &mut scores).unwrap(),
            1
        );
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn odd_dimension_is_supported_safely_by_scalar_path() {
        let query = [1.0f32; 3];
        let keys = [0x98u8, 0x08u8]; // levels 0,1,0 for first three nibbles
        let mut score = [0.0f32; 1];
        CpuSimdEngine::try_dequant_and_fma(&query, &keys, 1.0, &mut score).unwrap();
        assert_eq!(score[0], 1.0);
    }
}
