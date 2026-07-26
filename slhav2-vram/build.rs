use std::path::Path;
use std::process::Command;

/// Attempt to compile the CUDA kernel to PTX using the system nvcc.
/// If nvcc is unavailable, skip silently (the .ptx must exist from a prior build
/// or be included in the source tree).
fn compile_kernels() {
    let kernel_src = Path::new("kernels/lowrank_turboquant.cu");
    let ptx_out = Path::new("kernels/lowrank_turboquant.ptx");

    if !kernel_src.exists() {
        return;
    }

    let nvcc_ok = match Command::new("nvcc").arg("--version").output() {
        Ok(out) => out.status.success(),
        Err(_) => false,
    };
    if !nvcc_ok {
        return;
    }

    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "-arch=sm_89",
            "-O3",
            "--use_fast_math",
            "-o",
        ])
        .arg(ptx_out)
        .arg(kernel_src)
        .status();

    match status {
        Ok(s) => {
            if s.success() {
                println!("cargo:warning=PTX compiled successfully -> {}", ptx_out.display());
            } else {
                println!("cargo:warning=nvcc exited with code {:?}", s.code());
                println!("cargo:warning=using pre-existing PTX if available");
            }
        }
        Err(e) => {
            println!("cargo:warning=Failed to execute nvcc: {e}");
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/lowrank_turboquant.cu");
    compile_kernels();
}
