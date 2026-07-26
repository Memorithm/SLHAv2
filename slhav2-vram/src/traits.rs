use std::fmt;
use std::sync::Arc;

/// Opaque handle to a device memory allocation.
///
/// Each backend interprets the `raw` field privately:
/// - CUDA: `CUdeviceptr` (device memory address) as `u64`
/// - CPU:  byte offset within the internal allocation arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePointer {
    pub(crate) raw: u64,
    pub size: usize,
}

impl DevicePointer {
    pub fn null() -> Self {
        DevicePointer { raw: 0, size: 0 }
    }

    pub fn is_null(&self) -> bool {
        self.raw == 0
    }
}

/// Errors originating from VRAM operations.
#[derive(Debug, Clone)]
pub enum VramError {
    AllocationFailed(String),
    DeallocationFailed(String),
    CopyToDeviceFailed(String),
    CopyToHostFailed(String),
    KernelLaunchFailed(String),
    SynchronizationFailed(String),
    BackendNotAvailable(String),
    PoolExhausted(String),
    InvalidPointer(String),
    CudaDriver(String),
}

impl fmt::Display for VramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VramError::AllocationFailed(msg) => write!(f, "VRAM allocation failed: {msg}"),
            VramError::DeallocationFailed(msg) => write!(f, "VRAM deallocation failed: {msg}"),
            VramError::CopyToDeviceFailed(msg) => write!(f, "Host-to-device copy failed: {msg}"),
            VramError::CopyToHostFailed(msg) => write!(f, "Device-to-host copy failed: {msg}"),
            VramError::KernelLaunchFailed(msg) => write!(f, "Kernel launch failed: {msg}"),
            VramError::SynchronizationFailed(msg) => write!(f, "Synchronization failed: {msg}"),
            VramError::BackendNotAvailable(msg) => write!(f, "Backend not available: {msg}"),
            VramError::PoolExhausted(msg) => write!(f, "Memory pool exhausted: {msg}"),
            VramError::InvalidPointer(msg) => write!(f, "Invalid pointer: {msg}"),
            VramError::CudaDriver(msg) => write!(f, "CUDA driver error: {msg}"),
        }
    }
}

impl std::error::Error for VramError {}

/// Result alias for VRAM operations.
pub type VramResult<T> = Result<T, VramError>;

/// Hardware-agnostic device engine trait.
///
/// Every backend (CPU reference, CUDA, Vulkan, Metal, etc.) implements this
/// trait to provide a uniform interface for VRAM management and kernel dispatch.
pub trait DeviceEngine: Send + Sync {
    /// Human-readable backend name (e.g. "cpu-ref", "cuda-sm89").
    fn name(&self) -> &'static str;

    /// Allocate `size_bytes` of device memory.
    fn allocate(&self, size_bytes: usize) -> VramResult<DevicePointer>;

    /// Free a previously allocated device pointer.
    fn free(&self, ptr: DevicePointer) -> VramResult<()>;

    /// Copy bytes from host memory to device memory.
    fn copy_to_device(&self, src: &[u8], dst: &DevicePointer) -> VramResult<()>;

    /// Copy bytes from device memory to host memory.
    fn copy_to_host(&self, src: &DevicePointer, dst: &mut [u8]) -> VramResult<()>;

    /// Block until all pending operations on this engine complete.
    fn synchronize(&self) -> VramResult<()>;

    /// Launch the low-rank matrix-multiply kernel with TurboQuant dequantization.
    ///
    /// Computes: `output[M, N] = dequant(weights[N, K]) · input[M, K]`
    ///
    /// # Parameters
    /// - `input`:  device pointer to `[M, K]` float32 row-major input
    /// - `weight_lowrank`: device pointer to packed INT4 weights `[N, K/2]`
    /// - `output`: device pointer to `[M, N]` float32 row-major output
    /// - `dim_m`: number of rows in input and output
    /// - `dim_n`: number of columns in output (hidden dimension)
    /// - `dim_k`: number of columns in input (latent dimension, must equal D_C)
    fn launch_lowrank_matmul(
        &self,
        input: &DevicePointer,
        weight_lowrank: &DevicePointer,
        output: &DevicePointer,
        dim_m: usize,
        dim_n: usize,
        dim_k: usize,
    ) -> VramResult<()>;

    /// Return total and available device memory in bytes, if available.
    fn memory_info(&self) -> VramResult<(usize, usize)> {
        Err(VramError::BackendNotAvailable(
            "memory_info not implemented by this backend".into(),
        ))
    }
}

/// Type-erased, reference-counted device engine.
pub type DynDeviceEngine = Arc<dyn DeviceEngine>;
