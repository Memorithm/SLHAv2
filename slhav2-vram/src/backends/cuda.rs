use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::ptr::NonNull;

use crate::codec;
use crate::traits::{DeviceAllocation, DeviceEngine};

// Re-export for convenience
pub use self::ffi::CudaFunction;

pub enum CudaAllocation {
    Owned {
        ptr: NonNull<u8>,
        size: usize,
        ctx: CudaContext,
    },
    Borrowed,
    Invalid,
}

impl CudaAllocation {
    pub fn ptr(&self) -> *mut u8 {
        match self {
            CudaAllocation::Owned { ptr, .. } => ptr.as_ptr(),
            _ => std::ptr::null_mut(),
        }
    }
}

impl DeviceAllocation for CudaAllocation {
    fn size(&self) -> usize {
        match self {
            CudaAllocation::Owned { size, .. } => *size,
            _ => 0,
        }
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        if let CudaAllocation::Owned { ptr, size, ctx } = self {
            let p = ptr.as_ptr() as *mut libc::c_void;
            let _ctx = *ctx;
            let _ = unsafe { ffi::cuMemFree_v2(p) };
        }
    }
}

#[derive(Clone, Copy)]
pub struct CudaContext {
    pub device: i32,
}

pub struct CudaEngine {
    pub ctx: CudaContext,
    pub ptx_bytes: Vec<u8>,
    functions: Vec<CudaFunction>,
}

impl fmt::Debug for CudaEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaEngine")
            .field("ctx", &self.ctx)
            .field("ptx_bytes", &self.ptx_bytes.len())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct CudaError(pub String);

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error: {}", self.0)
    }
}

impl Error for CudaError {}

impl CudaEngine {
    pub fn new() -> Result<Self, CudaError> {
        unsafe {
            ffi::cuInit(0).to_result().map_err(|e| {
                CudaError(format!("cuInit failed (err {e}): is the driver installed?"))
            })?;
        }

        let mut count: i32 = 0;
        unsafe {
            ffi::cuDeviceGetCount(&mut count as *mut i32)
                .to_result()
                .map_err(|e| CudaError(format!("cuDeviceGetCount failed: {e}")))?;
        }
        if count < 1 {
            return Err(CudaError("no CUDA-capable device found".into()));
        }

        let device = 0;
        let mut ctx: ffi::CUcontext = std::ptr::null_mut();
        let result = unsafe {
            ffi::cuCtxCreate_v2(
                &mut ctx as *mut ffi::CUcontext,
                0,
                device as ffi::CUdevice,
            )
        };
        result.to_result().map_err(|e| CudaError(format!("cuCtxCreate failed: {e}")))?;

        Ok(Self {
            ctx: CudaContext { device },
            ptx_bytes: Vec::new(),
            functions: Vec::new(),
        })
    }

    pub fn load_ptx(&mut self, ptx: &[u8]) -> Result<CudaFunction, CudaError> {
        let module_ptr = unsafe {
            ffi::cuModuleLoadData(
                ptx.as_ptr() as *const libc::c_void,
            )
        };
        module_ptr
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
        let grid_dim = ((num_tiles as usize + 255) / 256) as i32;
        let block_dim = 256i32;

        let params: [*mut libc::c_void; 5] = [
            &q_coarse_dev.ptr() as *const *mut u8 as *mut libc::c_void,
            &q_sign_dev.ptr() as *const *mut u8 as *mut libc::c_void,
            &tiles_dev.ptr() as *const *mut u8 as *mut libc::c_void,
            &scores_dev.ptr() as *const *mut u8 as *mut libc::c_void,
            &num_tiles as *const i32 as *mut libc::c_void,
        ];

        unsafe {
            ffi::cuLaunchKernel(
                kernel.func,
                grid_dim as u32,
                1,
                1,
                block_dim as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_ptr(),
                std::ptr::null_mut(),
            )
            .to_result()?;
        }

        Ok(())
    }

    pub fn launch_kernel_1d(
        &self,
        func: ffi::CUfunction,
        grid_x: u32,
        block_x: u32,
        args: &[*mut libc::c_void],
        shared_mem: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuLaunchKernel(
                func,
                grid_x,
                1,
                1,
                block_x,
                1,
                1,
                shared_mem,
                std::ptr::null_mut(),
                args.as_ptr(),
                std::ptr::null_mut(),
            )
            .to_result()
            .map_err(|e| CudaError(format!("cuLaunchKernel failed: {e}")))
        }
    }
}

impl DeviceEngine for CudaEngine {
    type Alloc = CudaAllocation;
    type Error = CudaError;

    fn allocate(&self, size: usize) -> Result<CudaAllocation, CudaError> {
        let mut ptr: ffi::CUdeviceptr = 0;
        unsafe {
            ffi::cuMemAlloc_v2(&mut ptr as *mut ffi::CUdeviceptr, size as u64)
                .to_result()
                .map_err(|e| CudaError(format!("cuMemAlloc({size}) failed: {e}")))?;
        }
        Ok(CudaAllocation::Owned {
            ptr: NonNull::new(ptr as *mut u8).unwrap(),
            size,
            ctx: self.ctx,
        })
    }

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &mut CudaAllocation,
        dst_offset: usize,
    ) -> Result<(), CudaError> {
        let dst_ptr = match dst {
            CudaAllocation::Owned { ptr, .. } => ptr.as_ptr() as ffi::CUdeviceptr,
            _ => return Err(CudaError("copy_to_device: dst is not owned".into())),
        };
        unsafe {
            ffi::cuMemcpyHtoD_v2(
                dst_ptr + dst_offset as ffi::CUdeviceptr,
                src.as_ptr() as *const libc::c_void,
                src.len() as u64,
            )
            .to_result()
            .map_err(|e| CudaError(format!("cuMemcpyHtoD failed: {e}")))
        }
    }

    fn copy_to_host(
        &self,
        src: &CudaAllocation,
        src_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), CudaError> {
        let src_ptr = match src {
            CudaAllocation::Owned { ptr, .. } => ptr.as_ptr() as ffi::CUdeviceptr,
            _ => return Err(CudaError("copy_to_host: src is not owned".into())),
        };
        unsafe {
            ffi::cuMemcpyDtoH_v2(
                dst.as_ptr() as *mut libc::c_void,
                src_ptr + src_offset as ffi::CUdeviceptr,
                dst.len() as u64,
            )
            .to_result()
            .map_err(|e| CudaError(format!("cuMemcpyDtoH failed: {e}")))
        }
    }

    fn set_device(&self) -> Result<(), CudaError> {
        unsafe {
            ffi::cuCtxSetCurrent(self.ctx as *mut libc::c_void)
        };
        Ok(())
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        unsafe {
            ffi::cuCtxSynchronize()
                .to_result()
                .map_err(|e| CudaError(format!("cuCtxSynchronize failed: {e}")))
        }
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        let ctx = self.ctx;
        unsafe {
            ffi::cuCtxDestroy_v2(ctx);
        }
    }
}

unsafe impl Send for CudaEngine {}
unsafe impl Sync for CudaEngine {}

mod ffi {
    #![allow(non_camel_case_types, dead_code)]

    use std::os::raw::c_void;

    pub type CUresult = i32;
    pub type CUdevice = i32;
    pub type CUcontext = *mut c_void;
    pub type CUmodule = *mut c_void;
    pub type CUfunction = *mut c_void;
    pub type CUdeviceptr = u64;

    pub const CUDA_SUCCESS: i32 = 0;

    pub trait ToResult {
        fn to_result(&self) -> Result<(), i32>;
    }

    impl ToResult for i32 {
        fn to_result(&self) -> Result<(), i32> {
            if *self == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(*self)
            }
        }
    }

    extern "C" {
        pub fn cuInit(flags: u32) -> CUresult;
        pub fn cuDeviceGetCount(count: *mut i32) -> CUresult;
        pub fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
        pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
        pub fn cuCtxSynchronize() -> CUresult;
        pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
        pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, size: u64) -> CUresult;
        pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
        pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, count: u64) -> CUresult;
        pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, count: u64) -> CUresult;
        pub fn cuModuleLoadData(module: *mut CUmodule, data: *const c_void) -> CUresult;
        pub fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const u8) -> CUresult;
        pub fn cuLaunchKernel(
            f: CUfunction,
            grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
            block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
            shared_mem: u32,
            stream: *mut c_void,
            kernel_params: *const *mut c_void,
            extra: *mut c_void,
        ) -> CUresult;
        pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    }

    pub struct CudaFunction {
        pub module: CUmodule,
        pub func: CUfunction,
        pub name: Vec<u8>,
    }

    impl CudaFunction {
        pub fn load(module_data: *const c_void, name: &str) -> Result<Self, i32> {
            let mut module: CUmodule = std::ptr::null_mut();
            let result = unsafe { cuModuleLoadData(&mut module as *mut CUmodule, module_data) };
            result.to_result()?;

            let name_bytes = name.as_bytes();
            let mut func: CUfunction = std::ptr::null_mut();
            let result = unsafe {
                cuModuleGetFunction(
                    &mut func as *mut CUfunction,
                    module,
                    name_bytes.as_ptr(),
                )
            };
            result.to_result().map_err(|e| {
                unsafe { cuModuleUnload(module) };
                e
            })?;

            Ok(CudaFunction {
                module,
                func,
                name: name_bytes.to_vec(),
            })
        }
    }

    impl Drop for CudaFunction {
        fn drop(&mut self) {
            unsafe { cuModuleUnload(self.module) };
        }
    }
}
