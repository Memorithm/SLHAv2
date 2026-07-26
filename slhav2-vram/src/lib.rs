//! slhav2-vram — Hardware-agnostic VRAM management and execution engine for SLHAv2.
//!
//! Provides a unified abstraction over GPU/CPU device memory with:
//! - `DeviceEngine` trait: allocate, free, transfer, and launch compute kernels
//! - `VramMemoryPool`: slab/arena allocator that eliminates per-allocation latency
//! - `LowRankPipeline`: asynchronous execution pipeline for batched MatMul + TurboQuant
//! - `CpuRefBackend`: pure Rust reference backend (ground truth for validation)
//! - `CudaDriverBackend`: CUDA Driver API backend with PTX dynamic loading (feature = "cuda")

pub mod backends;
pub mod memory;
pub mod pipeline;
pub mod traits;

pub use memory::VramMemoryPool;
pub use pipeline::LowRankPipeline;
pub use traits::{DeviceEngine, DevicePointer, DynDeviceEngine, VramError, VramResult};
