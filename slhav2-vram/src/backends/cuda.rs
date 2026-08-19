use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::traits::{DeviceAllocation, DeviceEngine};

// ── Global CUDA backing allocation counter ───────────────────────────────────

static CUDA_BACKING_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// ── Opaque CUDA handle types ──────────────────────────────────────────────

#[repr(C)]
pub struct CUctx_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUmod_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUfunc_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUstream_st {
    _private: [u8; 0],
}

pub type CUcontext = *mut CUctx_st;
pub type CUmodule = *mut CUmod_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUstream = *mut CUstream_st;
pub type CUdevice = i32;
pub type CUdeviceptr = u64;
pub type CUresult = i32;

const CUDA_SUCCESS: i32 = 0;

// ── Error type ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CudaError(pub String);

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error: {}", self.0)
    }
}

impl Error for CudaError {}

/// Human-readable name for a common CUDA driver error code (the `CUDA_ERROR_*`
/// enum). Unknown codes fall back to a numeric message.
fn cu_result_name(code: CUresult) -> &'static str {
    match code {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        5 => "CUDA_ERROR_PROFILER_DISABLED",
        6 => "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
        7 => "CUDA_ERROR_PROFILER_ALREADY_STARTED",
        8 => "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        202 => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
        205 => "CUDA_ERROR_OUT_OF_MEMORY",
        206 => "CUDA_ERROR_INVALID_POINTER",
        207 => "CUDA_ERROR_INVALID_MEMORY_BLOCK",
        208 => "CUDA_ERROR_INVALID_PTX",
        209 => "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
        218 => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
        219 => "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        220 => "CUDA_ERROR_OPERATING_SYSTEM",
        300 => "CUDA_ERROR_INVALID_HANDLE",
        400 => "CUDA_ERROR_NOT_FOUND",
        500 => "CUDA_ERROR_NOT_READY",
        600 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        700 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        701 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        702 => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
        703 => "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
        704 => "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
        705 => "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
        708 => "CUDA_ERROR_LAUNCH_FAILED",
        709 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        710 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        711 => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
        715 => "CUDA_ERROR_UNKNOWN",
        800 => "CUDA_ERROR_INVALID_RESOURCE_HANDLE",
        _ => "CUDA_ERROR_UNKNOWN_CODE",
    }
}

fn cuda_check(result: CUresult, op: &'static str) -> Result<(), CudaError> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError(format!(
            "{op} failed: {} ({result})",
            cu_result_name(result)
        )))
    }
}

// ── Shared engine inner ───────────────────────────────────────────────────

struct CudaInner {
    context: CUcontext,
    device: CUdevice,
}

impl Drop for CudaInner {
    fn drop(&mut self) {
        // SAFETY: cuCtxDestroy is safe when the context handle is valid and
        // no other resources depend on it. The Rc ensures no allocations or
        // modules outlive this destruction.
        unsafe { cuCtxDestroy_v2(self.context) };
    }
}

// ── Allocation ────────────────────────────────────────────────────────────

pub struct CudaAllocation {
    ptr: CUdeviceptr,
    len: usize,
    owner: Rc<CudaInner>,
}

impl CudaAllocation {
    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl DeviceAllocation for CudaAllocation {
    fn size(&self) -> usize {
        self.len
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        let ctx = self.owner.context;
        // SAFETY: cuCtxSetCurrent and cuMemFree_v2 are safe when the CUDA
        // driver is initialized and the context handle is valid. The Rc
        // guarantees the engine still exists.
        let freed = unsafe {
            cuCtxSetCurrent(ctx);
            cuMemFree_v2(self.ptr)
        };
        // Only count the free when it actually succeeded, so the backing
        // allocation counter never desyncs from reality.
        if freed == CUDA_SUCCESS {
            CUDA_BACKING_ALLOCATIONS.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

// ── Module and function ownership ─────────────────────────────────────────

pub struct CudaModule {
    module: CUmodule,
    owner: Rc<CudaInner>,
}

impl CudaModule {
    pub fn load_ptx(engine: &CudaEngine, ptx_bytes: &[u8]) -> Result<Rc<CudaModule>, CudaError> {
        engine.load_ptx(ptx_bytes)
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        let ctx = self.owner.context;
        // SAFETY: cuCtxSetCurrent and cuModuleUnload are safe when the context
        // and module handles are valid. The Rc ensures no CudaFunction
        // outlives the module (each function holds an Rc<CudaModule>), so the
        // module is only unloaded once its last function is dropped too.
        let _ = unsafe {
            cuCtxSetCurrent(ctx);
            cuModuleUnload(self.module)
        };
    }
}

impl CudaModule {
    /// Resolve a kernel by name. Takes `&Rc<Self>` so the returned
    /// [`CudaFunction`] can clone the module handle and keep it (and the
    /// underlying `CUmodule`) alive for as long as the function is used —
    /// preventing a dangling-module use-after-free.
    pub fn get_function(self: &Rc<Self>, name: &str) -> Result<CudaFunction, CudaError> {
        // SAFETY: NUL-terminated C string for the kernel name.
        let cname =
            CString::new(name).map_err(|_| CudaError("kernel name contains null byte".into()))?;
        let mut func: CUfunction = std::ptr::null_mut();
        // SAFETY: cuModuleGetFunction reads a pointer into the NUL-terminated
        // name string and writes to the function output pointer.
        unsafe {
            cuda_check(
                cuModuleGetFunction(&mut func as *mut CUfunction, self.module, cname.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(CudaFunction {
            func,
            module: Rc::clone(self),
        })
    }

    pub fn raw_module(&self) -> CUmodule {
        self.module
    }
}

/// A CUDA stream (RAII). Streams allow async copies + kernel launches that
/// overlap; the engine's convenience paths use the null stream, while
/// [`CudaEngine::score_tiles_on_stream`] uses these for the persistent
/// scoring pipeline.
pub struct CudaStream {
    stream: CUstream,
    owner: Rc<CudaInner>,
}

impl CudaStream {
    pub fn new(engine: &CudaEngine) -> Result<Self, CudaError> {
        let mut stream: CUstream = std::ptr::null_mut();
        // SAFETY: `engine.inner.context` is a live CUDA context owned by the
        // shared engine (the `Rc` is cloned into `Self` below before this
        // handle can be used), and `cuStreamCreate` writes into the valid
        // output pointer with the context current on this thread.
        unsafe {
            cuda_check(cuCtxSetCurrent(engine.inner.context), "cuCtxSetCurrent")?;
            cuda_check(
                cuStreamCreate(&mut stream as *mut CUstream, 0),
                "cuStreamCreate",
            )?;
        }
        Ok(Self {
            stream,
            owner: Rc::clone(&engine.inner),
        })
    }

    pub fn raw(&self) -> CUstream {
        self.stream
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        let ctx = self.owner.context;
        // SAFETY: the context handle is kept alive by the shared `Rc`, and
        // `cuStreamSynchronize` only touches the live stream owned by `Self`.
        unsafe {
            cuda_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            cuda_check(cuStreamSynchronize(self.stream), "cuStreamSynchronize")
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let ctx = self.owner.context;
        // SAFETY: cuCtxSetCurrent + cuStreamDestroy are safe with valid
        // handles; the Rc keeps the engine alive.
        let _ = unsafe {
            cuCtxSetCurrent(ctx);
            cuStreamDestroy(self.stream)
        };
    }
}

/// A kernel function handle. Owns an `Rc<CudaModule>` so the module (and its
/// `CUfunction`) cannot be unloaded while the function is still in use — this
/// removes the dangling-module use-after-free that a bare `CUmodule` value
/// would allow if the module were dropped first.
pub struct CudaFunction {
    func: CUfunction,
    module: Rc<CudaModule>,
}

impl CudaFunction {
    pub fn raw_func(&self) -> CUfunction {
        self.func
    }
}

impl Clone for CudaFunction {
    fn clone(&self) -> Self {
        Self {
            func: self.func,
            module: Rc::clone(&self.module),
        }
    }
}

// ── Engine ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct CudaEngine {
    inner: Rc<CudaInner>,
}

impl fmt::Debug for CudaEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaEngine")
            .field("device", &self.inner.device)
            .finish()
    }
}

impl CudaEngine {
    pub fn new() -> Result<Self, CudaError> {
        // SAFETY: cuInit is the first CUDA call. It initializes the driver.
        unsafe {
            cuda_check(cuInit(0), "cuInit")?;
        }

        let mut count: i32 = 0;
        // SAFETY: cuDeviceGetCount writes to a valid i32.
        unsafe {
            cuda_check(cuDeviceGetCount(&mut count as *mut i32), "cuDeviceGetCount")?;
        }
        if count < 1 {
            return Err(CudaError("no CUDA-capable device found".into()));
        }

        let device: CUdevice = 0;
        let mut context: CUcontext = std::ptr::null_mut();
        // SAFETY: cuCtxCreate_v2 creates a new CUDA context on the specified
        // device and writes the handle to the output pointer.
        unsafe {
            cuda_check(
                cuCtxCreate_v2(&mut context as *mut CUcontext, 0, device),
                "cuCtxCreate_v2",
            )?;
        }

        Ok(Self {
            inner: Rc::new(CudaInner { context, device }),
        })
    }

    pub fn backing_allocation_count() -> usize {
        CUDA_BACKING_ALLOCATIONS.load(Ordering::SeqCst)
    }

    pub fn copy_to_device_at(
        &self,
        src: &[u8],
        dst: &mut CudaAllocation,
        dst_offset_bytes: usize,
    ) -> Result<(), CudaError> {
        let end = dst_offset_bytes
            .checked_add(src.len())
            .ok_or_else(|| CudaError("copy_to_device_at: offset + length overflow".into()))?;
        if end > dst.len {
            return Err(CudaError(format!(
                "copy_to_device_at: offset {} + size {} exceeds allocation size {}",
                dst_offset_bytes,
                src.len(),
                dst.len
            )));
        }
        // SAFETY: `self.inner.context` is kept alive by the `Rc` in
        // `CudaAllocation`/`Self`, and the copy bounds were checked above
        // (offset + length fits the destination allocation).
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }
        // SAFETY: `dst.ptr` is a live device allocation of `dst.len` bytes
        // (checked above), and `src` is a valid Rust slice of `src.len` bytes.
        unsafe {
            cuda_check(
                cuMemcpyHtoD_v2(
                    dst.ptr + dst_offset_bytes as u64,
                    src.as_ptr() as *const libc::c_void,
                    src.len() as u64,
                ),
                "cuMemcpyHtoD_v2",
            )
        }
    }

    pub fn copy_to_host_at(
        &self,
        src: &CudaAllocation,
        src_offset_bytes: usize,
        dst: &mut [u8],
    ) -> Result<(), CudaError> {
        let end = src_offset_bytes
            .checked_add(dst.len())
            .ok_or_else(|| CudaError("copy_to_host_at: offset + length overflow".into()))?;
        if end > src.len {
            return Err(CudaError(format!(
                "copy_to_host_at: offset {} + size {} exceeds allocation size {}",
                src_offset_bytes,
                dst.len(),
                src.len
            )));
        }
        // SAFETY: context kept alive by the `Rc`; copy bounds checked above.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }
        // SAFETY: `src.ptr` is a live device allocation of `src.len` bytes
        // (checked above), and `dst` is a valid mutable Rust slice.
        unsafe {
            cuda_check(
                cuMemcpyDtoH_v2(
                    dst.as_ptr() as *mut libc::c_void,
                    src.ptr + src_offset_bytes as u64,
                    dst.len() as u64,
                ),
                "cuMemcpyDtoH_v2",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn score_tiles_at(
        &self,
        q_coarse_dev: &CudaAllocation,
        q_coarse_offset: usize,
        q_sign_dev: &CudaAllocation,
        q_sign_offset: usize,
        tiles_dev: &CudaAllocation,
        tiles_offset: usize,
        scores_dev: &CudaAllocation,
        scores_offset: usize,
        num_tiles: i32,
        kernel: &CudaFunction,
    ) -> Result<(), CudaError> {
        if num_tiles <= 0 {
            return Ok(());
        }

        let q_coarse_ptr = q_coarse_dev.ptr + q_coarse_offset as u64;
        let q_sign_ptr = q_sign_dev.ptr + q_sign_offset as u64;
        let tiles_ptr = tiles_dev.ptr + tiles_offset as u64;
        let scores_ptr = scores_dev.ptr + scores_offset as u64;

        let grid_dim = (num_tiles as usize).div_ceil(256) as u32;
        let block_dim = 256u32;

        let args: [*mut libc::c_void; 5] = [
            &q_coarse_ptr as *const u64 as *mut libc::c_void,
            &q_sign_ptr as *const u64 as *mut libc::c_void,
            &tiles_ptr as *const u64 as *mut libc::c_void,
            &scores_ptr as *const u64 as *mut libc::c_void,
            &num_tiles as *const i32 as *mut libc::c_void,
        ];

        // SAFETY: `kernel.func` is a live CUfunction kept alive by
        // `CudaFunction`'s `Rc<CudaModule>`; the argument array has exactly the
        // five by-reference parameters the kernel expects.
        unsafe {
            cuda_check(
                cuLaunchKernel(
                    kernel.func,
                    grid_dim,
                    1,
                    1,
                    block_dim,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
        }

        Ok(())
    }

    pub fn load_ptx(&self, ptx_bytes: &[u8]) -> Result<Rc<CudaModule>, CudaError> {
        // cuModuleLoadData requires the PTX data to remain valid for the
        // duration of the call. Append a NUL terminator to be safe.
        let mut ptx = ptx_bytes.to_vec();
        ptx.push(0);

        let mut module: CUmodule = std::ptr::null_mut();
        // SAFETY: cuModuleLoadData parses the PTX image and creates a module.
        // The data pointer must be valid for the duration of the call (it is).
        unsafe {
            cuda_check(
                cuModuleLoadData(
                    &mut module as *mut CUmodule,
                    ptx.as_ptr() as *const libc::c_void,
                ),
                "cuModuleLoadData",
            )?;
        }

        Ok(Rc::new(CudaModule {
            module,
            owner: Rc::clone(&self.inner),
        }))
    }

    pub fn score_tiles(
        &self,
        q_coarse_dev: &CudaAllocation,
        q_sign_dev: &CudaAllocation,
        tiles_dev: &CudaAllocation,
        scores_dev: &CudaAllocation,
        num_tiles: i32,
        kernel: &CudaFunction,
    ) -> Result<(), CudaError> {
        if num_tiles <= 0 {
            return Ok(());
        }
        let n = num_tiles as usize;

        // Bounds-check every buffer against the kernel's read/write ranges
        // before launching: the kernel reads `n` 128-byte tiles and writes `n`
        // f32 scores; q_coarse is D_C f32s and q_sign RESIDUAL_WORDS u64s.
        let tile_bytes = n
            .checked_mul(crate::codec::TILE_BYTES)
            .ok_or_else(|| CudaError("score_tiles: tile byte count overflow".into()))?;
        let score_bytes = n
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaError("score_tiles: score byte count overflow".into()))?;
        if tile_bytes > tiles_dev.len {
            return Err(CudaError(format!(
                "score_tiles: {} tiles need {} bytes but the tile allocation has {}",
                n, tile_bytes, tiles_dev.len
            )));
        }
        if score_bytes > scores_dev.len {
            return Err(CudaError(format!(
                "score_tiles: {} scores need {} bytes but the score allocation has {}",
                n, score_bytes, scores_dev.len
            )));
        }
        let q_coarse_bytes = crate::codec::D_C * core::mem::size_of::<f32>();
        if q_coarse_bytes > q_coarse_dev.len {
            return Err(CudaError(format!(
                "score_tiles: q_coarse needs {q_coarse_bytes} bytes but the allocation has {}",
                q_coarse_dev.len
            )));
        }
        let q_sign_bytes = crate::codec::RESIDUAL_WORDS * core::mem::size_of::<u64>();
        if q_sign_bytes > q_sign_dev.len {
            return Err(CudaError(format!(
                "score_tiles: q_sign needs {q_sign_bytes} bytes but the allocation has {}",
                q_sign_dev.len
            )));
        }

        let grid_dim = (num_tiles as usize).div_ceil(256) as u32;
        let block_dim = 256u32;

        let args: [*mut libc::c_void; 5] = [
            &q_coarse_dev.ptr as *const u64 as *mut libc::c_void,
            &q_sign_dev.ptr as *const u64 as *mut libc::c_void,
            &tiles_dev.ptr as *const u64 as *mut libc::c_void,
            &scores_dev.ptr as *const u64 as *mut libc::c_void,
            &num_tiles as *const i32 as *mut libc::c_void,
        ];

        // SAFETY: cuLaunchKernel launches the kernel with the given grid/block
        // dimensions and kernel arguments. All device pointers must be valid
        // allocations (guaranteed by the allocation types and the bounds checks
        // above).
        unsafe {
            cuda_check(
                cuLaunchKernel(
                    kernel.func,
                    grid_dim,
                    1,
                    1,
                    block_dim,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
        }

        Ok(())
    }

    /// Same as [`Self::score_tiles`] but launches on `stream`, enabling
    /// overlap of H2D copies, the kernel and D2H copies. The caller is
    /// responsible for `stream.synchronize()` (or an event) before reading the
    /// scores. Bounds-checks are identical to [`Self::score_tiles`].
    #[allow(clippy::too_many_arguments)]
    pub fn score_tiles_on_stream(
        &self,
        q_coarse_dev: &CudaAllocation,
        q_sign_dev: &CudaAllocation,
        tiles_dev: &CudaAllocation,
        scores_dev: &CudaAllocation,
        num_tiles: i32,
        kernel: &CudaFunction,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        if num_tiles <= 0 {
            return Ok(());
        }
        let n = num_tiles as usize;

        let tile_bytes = n
            .checked_mul(crate::codec::TILE_BYTES)
            .ok_or_else(|| CudaError("score_tiles_on_stream: tile byte count overflow".into()))?;
        let score_bytes = n
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaError("score_tiles_on_stream: score byte count overflow".into()))?;
        if tile_bytes > tiles_dev.len {
            return Err(CudaError(format!(
                "score_tiles_on_stream: {} tiles need {} bytes but the tile allocation has {}",
                n, tile_bytes, tiles_dev.len
            )));
        }
        if score_bytes > scores_dev.len {
            return Err(CudaError(format!(
                "score_tiles_on_stream: {} scores need {} bytes but the score allocation has {}",
                n, score_bytes, scores_dev.len
            )));
        }
        let q_coarse_bytes = crate::codec::D_C * core::mem::size_of::<f32>();
        if q_coarse_bytes > q_coarse_dev.len {
            return Err(CudaError(format!(
                "score_tiles_on_stream: q_coarse needs {q_coarse_bytes} bytes but the allocation has {}",
                q_coarse_dev.len
            )));
        }
        let q_sign_bytes = crate::codec::RESIDUAL_WORDS * core::mem::size_of::<u64>();
        if q_sign_bytes > q_sign_dev.len {
            return Err(CudaError(format!(
                "score_tiles_on_stream: q_sign needs {q_sign_bytes} bytes but the allocation has {}",
                q_sign_dev.len
            )));
        }

        let grid_dim = (num_tiles as usize).div_ceil(256) as u32;
        let block_dim = 256u32;

        let args: [*mut libc::c_void; 5] = [
            &q_coarse_dev.ptr as *const u64 as *mut libc::c_void,
            &q_sign_dev.ptr as *const u64 as *mut libc::c_void,
            &tiles_dev.ptr as *const u64 as *mut libc::c_void,
            &scores_dev.ptr as *const u64 as *mut libc::c_void,
            &num_tiles as *const i32 as *mut libc::c_void,
        ];

        // SAFETY: `kernel.func` is a live CUfunction kept alive by
        // `CudaFunction`'s `Rc<CudaModule>`; `stream.raw()` is a live stream;
        // the argument array has exactly the five by-reference parameters the
        // kernel expects; and every device pointer was bounds-checked above.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
            cuda_check(
                cuLaunchKernel(
                    kernel.func,
                    grid_dim,
                    1,
                    1,
                    block_dim,
                    1,
                    1,
                    0,
                    stream.raw(),
                    args.as_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
        }

        Ok(())
    }

    /// Async host→device copy on `stream`. Returns immediately; the copy
    /// completes before any subsequent op enqueued on the same stream.
    pub fn copy_to_device_async(
        &self,
        src: &[u8],
        dst: &mut CudaAllocation,
        dst_offset: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let end = dst_offset
            .checked_add(src.len())
            .ok_or_else(|| CudaError("copy_to_device_async: offset + length overflow".into()))?;
        if end > dst.len {
            return Err(CudaError(format!(
                "copy_to_device_async: offset {dst_offset} + size {} exceeds allocation size {}",
                src.len(),
                dst.len
            )));
        }
        // SAFETY: bounds checked above; `src` is a valid Rust slice; the
        // context and stream are kept alive by their owning `Rc`s.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
            cuda_check(
                cuMemcpyHtoDAsync_v2(
                    dst.ptr + dst_offset as u64,
                    src.as_ptr() as *const libc::c_void,
                    src.len() as u64,
                    stream.raw(),
                ),
                "cuMemcpyHtoDAsync_v2",
            )
        }
    }

    /// Async device→host copy on `stream`. Returns immediately; call
    /// `stream.synchronize()` before reading `dst`.
    pub fn copy_to_host_async(
        &self,
        src: &CudaAllocation,
        src_offset: usize,
        dst: &mut [u8],
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let end = src_offset
            .checked_add(dst.len())
            .ok_or_else(|| CudaError("copy_to_host_async: offset + length overflow".into()))?;
        if end > src.len {
            return Err(CudaError(format!(
                "copy_to_host_async: offset {src_offset} + size {} exceeds allocation size {}",
                dst.len(),
                src.len
            )));
        }
        // SAFETY: bounds checked above; `dst` is a valid mutable Rust slice;
        // the context and stream are kept alive by their owning `Rc`s.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
            cuda_check(
                cuMemcpyDtoHAsync_v2(
                    dst.as_ptr() as *mut libc::c_void,
                    src.ptr + src_offset as u64,
                    dst.len() as u64,
                    stream.raw(),
                ),
                "cuMemcpyDtoHAsync_v2",
            )
        }
    }
}

impl DeviceEngine for CudaEngine {
    type Alloc = CudaAllocation;
    type Error = CudaError;

    fn allocate(&self, size: usize) -> Result<CudaAllocation, CudaError> {
        // SAFETY: Set the current context, then allocate.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }

        let mut ptr: CUdeviceptr = 0;
        // SAFETY: cuMemAlloc_v2 writes the device pointer to the output.
        unsafe {
            cuda_check(
                cuMemAlloc_v2(&mut ptr as *mut CUdeviceptr, size as u64),
                "cuMemAlloc_v2",
            )?;
        }

        CUDA_BACKING_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);

        Ok(CudaAllocation {
            ptr,
            len: size,
            owner: Rc::clone(&self.inner),
        })
    }

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &mut CudaAllocation,
        dst_offset: usize,
    ) -> Result<(), CudaError> {
        if dst_offset
            .checked_add(src.len())
            .is_none_or(|end| end > dst.len)
        {
            return Err(CudaError(format!(
                "copy_to_device: offset {} + size {} exceeds allocation size {}",
                dst_offset,
                src.len(),
                dst.len
            )));
        }
        // SAFETY: the context is kept alive by the `Rc` shared with the
        // allocation; copy bounds were checked above.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }
        // SAFETY: `dst.ptr` is a live device allocation of `dst.len` bytes
        // (checked above), and `src` is a valid Rust slice.
        unsafe {
            cuda_check(
                cuMemcpyHtoD_v2(
                    dst.ptr + dst_offset as u64,
                    src.as_ptr() as *const libc::c_void,
                    src.len() as u64,
                ),
                "cuMemcpyHtoD_v2",
            )
        }
    }

    fn copy_to_host(
        &self,
        src: &CudaAllocation,
        src_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), CudaError> {
        if src_offset
            .checked_add(dst.len())
            .is_none_or(|end| end > src.len)
        {
            return Err(CudaError(format!(
                "copy_to_host: offset {} + size {} exceeds allocation size {}",
                src_offset,
                dst.len(),
                src.len
            )));
        }
        // SAFETY: the context is kept alive by the `Rc`; bounds checked above.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }
        // SAFETY: `src.ptr` is a live device allocation of `src.len` bytes
        // (checked above), and `dst` is a valid mutable Rust slice.
        unsafe {
            cuda_check(
                cuMemcpyDtoH_v2(
                    dst.as_ptr() as *mut libc::c_void,
                    src.ptr + src_offset as u64,
                    dst.len() as u64,
                ),
                "cuMemcpyDtoH_v2",
            )
        }
    }

    fn set_device(&self) -> Result<(), CudaError> {
        // SAFETY: cuCtxSetCurrent makes this engine's context current on the
        // calling thread.
        unsafe { cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent") }
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: cuCtxSynchronize blocks until all pending operations on the
        // current context complete.
        unsafe { cuda_check(cuCtxSynchronize(), "cuCtxSynchronize") }
    }
}

// The engine is NOT Send or Sync. It contains an Rc<CudaInner> which is not
// Sync, and CUDA contexts are thread-bound.

// ── FFI declarations ──────────────────────────────────────────────────────

#[allow(non_snake_case, dead_code)]
#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDriverGetVersion(version: *mut i32) -> CUresult;
    fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    fn cuDeviceGetName(name: *mut i8, len: i32, dev: CUdevice) -> CUresult;
    fn cuDeviceComputeCapability(major: *mut i32, minor: *mut i32, dev: CUdevice) -> CUresult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, size: u64) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const libc::c_void, count: u64) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut libc::c_void, src: CUdeviceptr, count: u64) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, data: *const libc::c_void) -> CUresult;
    fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const core::ffi::c_char,
    ) -> CUresult;
    fn cuModuleUnload(module: CUmodule) -> CUresult;
    fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    fn cuStreamDestroy(stream: CUstream) -> CUresult;
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const libc::c_void,
        count: u64,
        stream: CUstream,
    ) -> CUresult;
    fn cuMemcpyDtoHAsync_v2(
        dst: *mut libc::c_void,
        src: CUdeviceptr,
        count: u64,
        stream: CUstream,
    ) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem: u32,
        stream: CUstream,
        kernel_params: *const *mut libc::c_void,
        extra: *mut libc::c_void,
    ) -> CUresult;
}
