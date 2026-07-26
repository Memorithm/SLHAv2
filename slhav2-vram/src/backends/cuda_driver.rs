#![allow(clippy::missing_transmute_annotations)]
//! Raw CUDA Driver API FFI via runtime `dlopen`/`dlsym`.
//!
//! Zero external crate dependencies beyond `libc`. Loads `libcuda.so.1`
//! at runtime so compilation succeeds even without the NVIDIA driver;
//! the CUDA backend simply fails at construction time with a clear error.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::sync::Mutex;

use crate::traits::{DeviceEngine, DevicePointer, VramError, VramResult};

// ── CUDA Driver API type aliases ───────────────────────────────

type CUdevice = i32;
type CUdeviceptr = u64;

#[repr(C)] struct CUctx_st;
type CUcontext = *mut CUctx_st;
#[repr(C)] struct CUmod_st;
type CUmodule = *mut CUmod_st;
#[repr(C)] struct CUfunc_st;
type CUfunction = *mut CUfunc_st;
#[repr(C)] struct CUstream_st;
type CUstream = *mut CUstream_st;

const CUDA_SUCCESS: i32 = 0;

fn check(result: i32, op: &str) -> VramResult<()> {
    match result {
        CUDA_SUCCESS => Ok(()),
        code => Err(VramError::CudaDriver(format!("{op} returned {code}"))),
    }
}

// ── Dynamically-loaded function table ──────────────────────────

struct CudaTable {
    _lib_handle: *mut libc::c_void,
    cu_init: extern "C" fn(u32) -> i32,
    cu_device_get: extern "C" fn(*mut CUdevice, i32) -> i32,
    cu_device_get_name: extern "C" fn(*mut i8, i32, CUdevice) -> i32,
    cu_device_total_mem: extern "C" fn(*mut usize, CUdevice) -> i32,
    cu_ctx_create: extern "C" fn(*mut CUcontext, u32, CUdevice) -> i32,
    cu_ctx_synchronize: extern "C" fn() -> i32,
    cu_module_load_data: extern "C" fn(*mut CUmodule, *const libc::c_void) -> i32,
    cu_module_get_function: extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> i32,
    cu_mem_alloc: extern "C" fn(*mut CUdeviceptr, usize) -> i32,
    cu_mem_free: extern "C" fn(CUdeviceptr) -> i32,
    cu_memcpy_htod: extern "C" fn(CUdeviceptr, *const libc::c_void, usize) -> i32,
    cu_memcpy_dtoh: extern "C" fn(*mut libc::c_void, CUdeviceptr, usize) -> i32,
    cu_stream_create: extern "C" fn(*mut CUstream, u32) -> i32,
    cu_launch_kernel: extern "C" fn(
        CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream,
        *mut *mut libc::c_void, *mut *mut libc::c_void,
    ) -> i32,
    cu_mem_get_info: extern "C" fn(*mut usize, *mut usize) -> i32,
}

macro_rules! load_sym {
    ($tbl:expr, $lib:expr, $name:literal => $field:ident) => {{
        let cname = CString::new($name).unwrap();
        let ptr = unsafe { libc::dlsym($lib, cname.as_ptr()) };
        if ptr.is_null() {
            let err = unsafe {
                CStr::from_ptr(libc::dlerror()).to_string_lossy().into_owned()
            };
            return Err(VramError::CudaDriver(format!("dlsym({}): {err}", $name)));
        }
        unsafe {
            use std::ptr::addr_of_mut;
            std::ptr::write(
                addr_of_mut!((*$tbl).$field),
                std::mem::transmute::<*mut libc::c_void, _>(ptr),
            );
        }
    }};
}

impl CudaTable {
    fn load() -> VramResult<Self> {
        let lib_name = CString::new("libcuda.so.1").unwrap();
        let lib_handle = unsafe { libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW) };
        if lib_handle.is_null() {
            let err = unsafe {
                CStr::from_ptr(libc::dlerror()).to_string_lossy().into_owned()
            };
            return Err(VramError::CudaDriver(format!(
                "dlopen(libcuda.so.1): {err}. Is the NVIDIA driver installed?"
            )));
        }

        let mut t = MaybeUninit::<CudaTable>::uninit();
        let tp = t.as_mut_ptr();

        unsafe { std::ptr::write(&mut (*tp)._lib_handle, lib_handle); }

        load_sym!(tp, lib_handle, "cuInit" => cu_init);
        load_sym!(tp, lib_handle, "cuDeviceGet" => cu_device_get);
        load_sym!(tp, lib_handle, "cuDeviceGetName" => cu_device_get_name);
        load_sym!(tp, lib_handle, "cuDeviceTotalMem" => cu_device_total_mem);
        load_sym!(tp, lib_handle, "cuCtxCreate_v2" => cu_ctx_create);
        load_sym!(tp, lib_handle, "cuCtxSynchronize" => cu_ctx_synchronize);
        load_sym!(tp, lib_handle, "cuModuleLoadData" => cu_module_load_data);
        load_sym!(tp, lib_handle, "cuModuleGetFunction" => cu_module_get_function);
        load_sym!(tp, lib_handle, "cuMemAlloc_v2" => cu_mem_alloc);
        load_sym!(tp, lib_handle, "cuMemFree_v2" => cu_mem_free);
        load_sym!(tp, lib_handle, "cuMemcpyHtoD_v2" => cu_memcpy_htod);
        load_sym!(tp, lib_handle, "cuMemcpyDtoH_v2" => cu_memcpy_dtoh);
        load_sym!(tp, lib_handle, "cuStreamCreate" => cu_stream_create);
        load_sym!(tp, lib_handle, "cuLaunchKernel" => cu_launch_kernel);
        load_sym!(tp, lib_handle, "cuMemGetInfo_v2" => cu_mem_get_info);

        Ok(unsafe { t.assume_init() })
    }
}

impl Drop for CudaTable {
    fn drop(&mut self) {
        unsafe { libc::dlclose(self._lib_handle); }
    }
}

// ── Backend ────────────────────────────────────────────────────

pub struct CudaDriverBackend {
    tbl: CudaTable,
    _ctx: CUcontext,
    _module: CUmodule,
    kernel: CUfunction,
    _stream: CUstream,
    _ptx: CString,
    _device_name: String,
    _total_vram: usize,
    allocations: Mutex<HashMap<u64, CUdeviceptr>>,
}

impl CudaDriverBackend {
    pub fn new(device_id: i32) -> VramResult<Self> {
        let tbl = CudaTable::load()?;

        check((tbl.cu_init)(0), "cuInit")?;

        let mut dev: CUdevice = 0;
        check((tbl.cu_device_get)(&mut dev, device_id), "cuDeviceGet")?;

        let mut ctx: CUcontext = std::ptr::null_mut();
        check((tbl.cu_ctx_create)(&mut ctx, 0x00, dev), "cuCtxCreate")?;

        let mut name_buf = [0i8; 256];
        (tbl.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev);
        let device_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut total_vram: usize = 0;
        check((tbl.cu_device_total_mem)(&mut total_vram, dev), "cuDeviceTotalMem")?;

        let ptx = include_str!("../../kernels/lowrank_turboquant.ptx");
        let ptx_cstr = CString::new(ptx)
            .map_err(|_| VramError::KernelLaunchFailed("PTX contains NUL".into()))?;

        let mut module: CUmodule = std::ptr::null_mut();
        check(
            (tbl.cu_module_load_data)(&mut module, ptx_cstr.as_ptr() as *const libc::c_void),
            "cuModuleLoadData",
        )?;

        let kname = CString::new("lowrank_turboquant_matmul").unwrap();
        let mut kernel: CUfunction = std::ptr::null_mut();
        check(
            (tbl.cu_module_get_function)(&mut kernel, module, kname.as_ptr()),
            "cuModuleGetFunction",
        )?;

        let mut stream: CUstream = std::ptr::null_mut();
        check((tbl.cu_stream_create)(&mut stream, 0x01), "cuStreamCreate")?;

        Ok(CudaDriverBackend {
            tbl,
            _ctx: ctx,
            _module: module,
            kernel,
            _stream: stream,
            _ptx: ptx_cstr,
            _device_name: device_name,
            _total_vram: total_vram,
            allocations: Mutex::new(HashMap::new()),
        })
    }
}

impl DeviceEngine for CudaDriverBackend {
    fn name(&self) -> &'static str {
        "cuda-sm89"
    }

    fn allocate(&self, size_bytes: usize) -> VramResult<DevicePointer> {
        let mut dptr: CUdeviceptr = 0;
        check((self.tbl.cu_mem_alloc)(&mut dptr, size_bytes), "cuMemAlloc")?;
        self.allocations.lock().unwrap().insert(dptr, dptr);
        Ok(DevicePointer { raw: dptr, size: size_bytes })
    }

    fn free(&self, ptr: DevicePointer) -> VramResult<()> {
        let dptr = self.allocations.lock().unwrap().remove(&ptr.raw)
            .ok_or_else(|| VramError::InvalidPointer(format!("{:#x} not tracked", ptr.raw)))?;
        check((self.tbl.cu_mem_free)(dptr), "cuMemFree")
    }

    fn copy_to_device(&self, src: &[u8], dst: &DevicePointer) -> VramResult<()> {
        check(
            (self.tbl.cu_memcpy_htod)(dst.raw, src.as_ptr() as *const libc::c_void, src.len()),
            "cuMemcpyHtoD",
        )
    }

    fn copy_to_host(&self, src: &DevicePointer, dst: &mut [u8]) -> VramResult<()> {
        check(
            (self.tbl.cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut libc::c_void, src.raw, dst.len()),
            "cuMemcpyDtoH",
        )
    }

    fn synchronize(&self) -> VramResult<()> {
        check((self.tbl.cu_ctx_synchronize)(), "cuCtxSynchronize")
    }

    fn launch_lowrank_matmul(
        &self,
        input: &DevicePointer,
        weight_lowrank: &DevicePointer,
        output: &DevicePointer,
        dim_m: usize,
        dim_n: usize,
        dim_k: usize,
    ) -> VramResult<()> {
        let scale_offset = dim_n * (dim_k / 2);
        let scale_ptr = weight_lowrank.raw + scale_offset as u64;

        let m = dim_m as i32;
        let n = dim_n as i32;
        let k = dim_k as i32;

        let grid_x = dim_m.div_ceil(128) as u32;
        let grid_y = dim_n.div_ceil(128) as u32;

        let params: [*mut libc::c_void; 7] = [
            &input.raw as *const _ as *mut libc::c_void,
            &weight_lowrank.raw as *const _ as *mut libc::c_void,
            &scale_ptr as *const _ as *mut libc::c_void,
            &output.raw as *const _ as *mut libc::c_void,
            &m as *const _ as *mut libc::c_void,
            &n as *const _ as *mut libc::c_void,
            &k as *const _ as *mut libc::c_void,
        ];

        check(
            (self.tbl.cu_launch_kernel)(
                self.kernel,
                grid_x, grid_y, 1,
                256, 1, 1,
                0,
                self._stream,
                params.as_ptr() as *mut *mut libc::c_void,
                std::ptr::null_mut(),
            ),
            "cuLaunchKernel",
        )
    }

    fn memory_info(&self) -> VramResult<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        check(
            (self.tbl.cu_mem_get_info)(&mut free, &mut total),
            "cuMemGetInfo",
        )?;
        Ok((total, free))
    }
}

impl Drop for CudaDriverBackend {
    fn drop(&mut self) {
        let _ = (self.tbl.cu_ctx_synchronize)();
    }
}

unsafe impl Send for CudaDriverBackend {}
unsafe impl Sync for CudaDriverBackend {}
