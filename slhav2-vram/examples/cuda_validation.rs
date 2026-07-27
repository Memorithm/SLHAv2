//! CUDA validation example — run under compute-sanitizer:
//!   cargo run --features cuda --example cuda_validation
//!   compute-sanitizer --tool memcheck target/debug/examples/cuda_validation

#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
use slhav2_vram::codec;
#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
use slhav2_vram::mem::tile::SerializedTile;

#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
fn make_test_tile(scale: f32, warm: bool) -> SerializedTile {
    let mut tile = SerializedTile::zeroed();
    tile.set_scale(scale);
    tile.set_dynamic_lambda(0.1);
    tile.set_group_scales(&[200u8; 8]);
    if warm {
        tile.set_flags(codec::FLAG_WARM);
    } else {
        tile.set_flags(0);
    }
    for d in 0..codec::LATENT_BYTES {
        let lo = (d as u8) & 0x0F;
        let hi = ((d as u8) >> 4) & 0x0F;
        tile.as_bytes_mut()[d] = (hi << 4) | lo;
    }
    tile
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature.");
    eprintln!("  cargo run --features cuda --example cuda_validation");
    std::process::exit(0);
}

#[cfg(all(feature = "cuda", not(slhav2_cuda_ptx)))]
fn main() {
    eprintln!("CUDA validation unavailable: no real PTX was generated.");
    eprintln!("Install nvcc, set SLHAV2_NVCC, or provide SLHAV2_PTX_PATH.");
    eprintln!("Set SLHAV2_REQUIRE_NVCC=1 to make the build fail immediately.");
    std::process::exit(1);
}

#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
fn run_validation() -> Result<(), Box<dyn std::error::Error>> {
    use slhav2_vram::backends::cuda::{CudaEngine, CudaModule};
    use slhav2_vram::traits::DeviceEngine;

    eprintln!("Initializing CUDA driver API...");
    let engine = CudaEngine::new()?;

    let tiles: Vec<SerializedTile> = (0..32)
        .map(|i| make_test_tile(0.5 + i as f32 * 0.1, i % 4 == 0))
        .collect();

    let q_coarse: Vec<f32> = (0..codec::D_C).map(|i| (i as f32).sin() * 0.5).collect();
    let q_sign: Vec<u64> = vec![0xDEAD_BEEF_CAFE_FACEu64; codec::RESIDUAL_WORDS];

    // Reference scores
    let ref_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q_coarse, &q_sign)).collect();
    eprintln!("  Reference (CPU) scores (first 4): {:?}", &ref_scores[..4]);

    // Load PTX
    let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/slha_score.ptx"));
    eprintln!("  Loading kernel from PTX ({} bytes)", ptx.len());
    let module = CudaModule::load_ptx(&engine, ptx)?;
    let kernel = module.get_function("slha_score_kernel")?;

    // Allocate GPU memory
    let total_tile_bytes = tiles.len() * codec::TILE_BYTES;
    let mut q_coarse_dev = engine.allocate(codec::D_C * 4)?;
    let mut q_sign_dev = engine.allocate(codec::RESIDUAL_WORDS * 8)?;
    let mut tiles_dev = engine.allocate(total_tile_bytes)?;
    let scores_dev = engine.allocate(tiles.len() * 4)?;

    eprintln!("  Allocated GPU memory ({} B tiles)", total_tile_bytes);

    // Copy data to GPU
    engine.copy_to_device(
        &codec::f32_slice_to_le_bytes(&q_coarse),
        &mut q_coarse_dev,
        0,
    )?;
    engine.copy_to_device(&codec::u64_slice_to_le_bytes(&q_sign), &mut q_sign_dev, 0)?;

    let mut tiles_buf = vec![0u8; total_tile_bytes];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        tiles_buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine.copy_to_device(&tiles_buf, &mut tiles_dev, 0)?;
    eprintln!("  Copied data to GPU");

    // Launch kernel
    engine.score_tiles(
        &q_coarse_dev,
        &q_sign_dev,
        &tiles_dev,
        &scores_dev,
        tiles.len() as i32,
        &kernel,
    )?;
    engine.synchronize()?;
    eprintln!("  Kernel launched and synchronized");

    // Read back scores
    let mut scores_buf = vec![0u8; tiles.len() * 4];
    engine.copy_to_host(&scores_dev, 0, &mut scores_buf)?;
    let gpu_scores =
        codec::le_bytes_to_f32_vec(&scores_buf).map_err(|e| format!("score conversion: {e}"))?;

    eprintln!("  GPU scores (first 4): {:?}", &gpu_scores[..4]);

    // Compare
    let mut max_abs = 0.0f32;
    let mut mismatches = 0;
    for (i, (&gpu, &ref_s)) in gpu_scores.iter().zip(ref_scores.iter()).enumerate() {
        let diff = (gpu - ref_s).abs();
        max_abs = max_abs.max(diff);
        if diff > 1e-3 {
            if mismatches < 5 {
                eprintln!("  MISMATCH tile {i}: gpu={gpu}  ref={ref_s}  diff={diff:.6e}");
            }
            mismatches += 1;
        }
    }

    if mismatches > 0 {
        eprintln!("FAIL: {mismatches}/{mismatches} mismatches, max_abs={max_abs:.6e}");
        return Err(format!("{mismatches} mismatches, max_abs={max_abs:.6e}").into());
    }

    eprintln!(
        "PASS: all {} scores match, max_abs={max_abs:.6e}",
        tiles.len()
    );
    Ok(())
}

#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
fn main() {
    match run_validation() {
        Ok(()) => {
            eprintln!("PASS");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            std::process::exit(1);
        }
    }
}
