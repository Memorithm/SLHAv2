use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/slha_score.cu");
    println!("cargo:rerun-if-env-changed=SLHAV2_CUDA_ARCH");

    if cfg!(feature = "cuda") {
        compile_kernels();
    }
}

fn compile_kernels() {
    let kernel_src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("kernels")
        .join("slha_score.cu");

    if !kernel_src.exists() {
        panic!("CUDA kernel source not found at {}", kernel_src.display());
    }

    let arch = env::var("SLHAV2_CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let ptx_out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("slha_score.ptx");

    let status = Command::new("nvcc")
        .args(["-ptx", &format!("-arch={arch}"), "-O3", "-o"])
        .arg(&ptx_out)
        .arg(&kernel_src)
        .status()
        .unwrap_or_else(|e| {
            panic!("failed to execute nvcc: {e}");
        });

    if !status.success() {
        panic!("nvcc exited with code {:?}", status.code());
    }

    println!("cargo::warning=PTX compiled -> {}", ptx_out.display());
}
