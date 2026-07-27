pub mod cpu;
#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
pub mod cuda;
