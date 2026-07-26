use std::cell::RefCell;
use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::rc::Rc;

use crate::traits::{DeviceAllocation, DeviceEngine};

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

fn cuda_check(result: CUresult, op: &'static str) -> Result<(), CudaError> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError(format!("{op} failed with code {result}")))
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
        unsafe {
            cuCtxSetCurrent(ctx);
            cuMemFree_v2(self.ptr);
        }
    }
}

// ── Module and function ownership ─────────────────────────────────────────

pub struct CudaModule {
    module: CUmodule,
    owner: Rc<CudaInner>,
}

impl CudaModule {
    pub fn load_ptx(engine: &CudaEngine, ptx_bytes: &[u8]) -> Result<Self, CudaError> {
        engine.load_ptx(ptx_bytes)
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        let ctx = self.owner.context;
        // SAFETY: cuCtxSetCurrent and cuModuleUnload are safe when the context
        // and module handles are valid. All CudaFunctions are dropped before
        // their CudaModule due to Rust drop order (fields dropped in declaration
        // order, and CudaModule holds the module reference).
        unsafe {
            cuCtxSetCurrent(ctx);
            cuModuleUnload(self.module);
        }
    }
}

impl CudaModule {
    pub fn get_function(&self, name: &str) -> Result<CudaFunction, CudaError> {
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
            _module: self.module,
        }) // borrow-check: keep module reference
    }

    pub fn raw_module(&self) -> CUmodule {
        self.module
    }
}

pub struct CudaFunction {
    func: CUfunction,
    _module: CUmodule,
}

impl CudaFunction {
    pub fn raw_func(&self) -> CUfunction {
        self.func
    }
}

// ── Engine ────────────────────────────────────────────────────────────────

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

    pub fn load_ptx(&self, ptx_bytes: &[u8]) -> Result<CudaModule, CudaError> {
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

        Ok(CudaModule {
            module,
            owner: Rc::clone(&self.inner),
        })
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

        let grid_dim = ((num_tiles as usize + 255) / 256) as u32;
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
        // allocations (guaranteed by the allocation types).
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
            .map_or(true, |end| end > dst.len)
        {
            return Err(CudaError(format!(
                "copy_to_device: offset {} + size {} exceeds allocation size {}",
                dst_offset,
                src.len(),
                dst.len
            )));
        }
        // SAFETY: Set context then copy host→device.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }
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
            .map_or(true, |end| end > src.len)
        {
            return Err(CudaError(format!(
                "copy_to_host: offset {} + size {} exceeds allocation size {}",
                src_offset,
                dst.len(),
                src.len
            )));
        }
        // SAFETY: Set context then copy device→host.
        unsafe {
            cuda_check(cuCtxSetCurrent(self.inner.context), "cuCtxSetCurrent")?;
        }
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
    fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const i8) -> CUresult;
    fn cuModuleUnload(module: CUmodule) -> CUresult;
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
