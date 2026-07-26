use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Re-run if the kernel source changes
    println!("cargo:rerun-if-changed=kernels/slha_score.cu");

    // Only compile PTX when the CUDA feature is active
    if cfg!(feature = "cuda") && cfg!(target_os = "linux") {
        compile_kernels();
    }
}

fn compile_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let arch = env::var("SLHAV2_CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let kernel_src = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").unwrap(),
    )
    .join("kernels")
    .join("slha_score.cu");

    let ptx_out = out_dir.join("slha_score.ptx");

    if !kernel_src.exists() {
        println!("cargo:warning=Kernel source not found: {}", kernel_src.display());
        return;
    }

    // Check if nvcc is available
    let nvcc_ok = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !nvcc_ok {
        println!("cargo:warning=nvcc not found; skipping PTX compilation");
        println!("cargo:warning=Pre-compiled PTX will be required at runtime");
        return;
    }

    let status = Command::new("nvcc")
        .args([
            "-ptx",
            &format!("-arch={arch}"),
            "-O3",
            "-o",
        ])
        .arg(&ptx_out)
        .arg(&kernel_src)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=PTX compiled -> {}", ptx_out.display());
        }
        Ok(s) => {
            println!("cargo:warning=nvcc exited with code {:?}", s.code());
        }
        Err(e) => {
            println!("cargo:warning=nvcc execution failed: {e}");
        }
    }
}
