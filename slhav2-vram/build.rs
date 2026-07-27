use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(slhav2_cuda_ptx)");
    println!("cargo:rerun-if-changed=kernels/slha_score.cu");
    println!("cargo:rerun-if-env-changed=SLHAV2_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=SLHAV2_NVCC");
    println!("cargo:rerun-if-env-changed=SLHAV2_PTX_PATH");
    println!("cargo:rerun-if-env-changed=SLHAV2_REQUIRE_NVCC");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    match prepare_ptx() {
        Ok(ptx_out) => {
            println!("cargo:rustc-cfg=slhav2_cuda_ptx");
            println!("cargo:warning=CUDA PTX available -> {}", ptx_out.display());
        }
        Err(error) => {
            if env_flag("SLHAV2_REQUIRE_NVCC") {
                panic!(
                    "CUDA PTX is required but could not be produced: {error}. \
                     Install nvcc, set SLHAV2_NVCC, or provide SLHAV2_PTX_PATH."
                );
            }

            println!(
                "cargo:warning=CUDA PTX unavailable: {error}. \
                 CUDA runtime targets are disabled for this build. \
                 Set SLHAV2_REQUIRE_NVCC=1 to make this condition fatal."
            );
        }
    }
}

fn prepare_ptx() -> Result<PathBuf, String> {
    let out_dir = required_path("OUT_DIR")?;
    let ptx_out = out_dir.join("slha_score.ptx");

    if let Some(source) = env::var_os("SLHAV2_PTX_PATH") {
        let source = PathBuf::from(source);
        validate_input_file(&source, "SLHAV2_PTX_PATH")?;

        fs::copy(&source, &ptx_out).map_err(|error| {
            format!(
                "failed to copy PTX from {} to {}: {error}",
                source.display(),
                ptx_out.display()
            )
        })?;

        validate_output_file(&ptx_out)?;
        return Ok(ptx_out);
    }

    compile_with_nvcc(&ptx_out)?;
    validate_output_file(&ptx_out)?;
    Ok(ptx_out)
}

fn compile_with_nvcc(ptx_out: &Path) -> Result<(), String> {
    let manifest_dir = required_path("CARGO_MANIFEST_DIR")?;
    let kernel_src = manifest_dir.join("kernels").join("slha_score.cu");
    validate_input_file(&kernel_src, "CUDA kernel source")?;

    let architecture = env::var("SLHAV2_CUDA_ARCH").unwrap_or_else(|_| String::from("sm_89"));
    let nvcc = env::var("SLHAV2_NVCC").unwrap_or_else(|_| String::from("nvcc"));

    let output = Command::new(&nvcc)
        .args(["-ptx", &format!("-arch={architecture}"), "-O3", "-o"])
        .arg(ptx_out)
        .arg(&kernel_src)
        .output()
        .map_err(|error| format!("failed to execute {nvcc}: {error}"))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    Err(format!(
        "{nvcc} exited with status {}. stdout: {} stderr: {}",
        output.status,
        stdout.trim(),
        stderr.trim()
    ))
}

fn required_path(name: &str) -> Result<PathBuf, String> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("required Cargo environment variable {name} is missing"))
}

fn validate_input_file(path: &Path, label: &str) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("{label} not found at {}", path.display()))
    }
}

fn validate_output_file(path: &Path) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("PTX output {} is unreadable: {error}", path.display()))?;

    if metadata.len() == 0 {
        Err(format!("PTX output {} is empty", path.display()))
    } else {
        Ok(())
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
