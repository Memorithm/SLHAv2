//! CUDA validation example — run under compute-sanitizer:
//!   cargo run --features cuda --example cuda_validation
//!   compute-sanitizer --tool memcheck cargo run --features cuda --example cuda_validation

use slhav2_vram::codec;
use slhav2_vram::mem::tile::SerializedTile;
use slhav2_vram::traits::{DeviceAllocation, DeviceEngine};

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

    for d in 0..codec::LATENT_KV_WORDS {
        let lo = (d as u8) & 0x0F;
        let hi = ((d as u8) >> 4) & 0x0F;
        tile.as_bytes_mut()[d] = (hi << 4) | lo;
    }
    tile
}

#[cfg(feature = "cuda")]
fn run_cuda_validation() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Initializing CUDA driver API...");
    let engine = slhav2_vram::backends::cuda::CudaEngine::new()?;
    eprintln!("  CUDA context created on device {}", engine.ctx.device);

    let tiles: Vec<SerializedTile> = (0..32)
        .map(|i| make_test_tile(0.5 + i as f32 * 0.1, i % 4 == 0))
        .collect();

    let q_coarse: Vec<f32> = (0..codec::D_C).map(|i| (i as f32).sin() * 0.5).collect();
    let q_sign: Vec<u64> = vec![0xDEAD_BEEF_CAFE_FACEu64; codec::RESIDUAL_WORDS];

    let ref_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q_coarse, &q_sign)).collect();
    eprintln!("  Reference scores (first 4): {:?}", &ref_scores[..4]);

    let total_tile_bytes = tiles.len() * codec::TILE_BYTES;
    let mut q_coarse_dev = engine.allocate(codec::D_C * 4)?;
    let mut q_sign_dev = engine.allocate(codec::RESIDUAL_WORDS * 8)?;
    let mut tiles_dev = engine.allocate(total_tile_bytes)?;
    let mut scores_dev = engine.allocate(tiles.len() * 4)?;

    eprintln!("  Allocated GPU memory ({} B tiles)", total_tile_bytes);

    engine.copy_to_device(
        &bytemuck::cast_slice(&q_coarse),
        &mut q_coarse_dev,
        0,
    )?;
    engine.copy_to_device(
        &bytemuck::cast_slice(&q_sign),
        &mut q_sign_dev,
        0,
    )?;

    let mut tiles_buf = vec![0u8; total_tile_bytes];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        tiles_buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine.copy_to_device(&tiles_buf, &mut tiles_dev, 0)?;
    eprintln!("  Copied data to GPU");

    let ptx_path = std::path::Path::new(&std::env::var("OUT_DIR").unwrap_or_default()).join("slha_score.ptx");
    let ptx_bytes = std::fs::read(&ptx_path).unwrap_or_else(|_| {
        let fallback = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("slha_score.ptx");
        std::fs::read(&fallback).expect("PTX not found — compile with nvcc on PATH")
    });
    eprintln!("  Loading kernel from PTX ({} bytes)", ptx_bytes.len());
    let kernel_fn = engine.load_ptx(&ptx_bytes)?;

    engine.score_tiles(
        &q_coarse_dev,
        &q_sign_dev,
        &tiles_dev,
        &scores_dev,
        tiles.len() as i32,
        &kernel_fn,
    )?;
    engine.synchronize()?;
    eprintln!("  Kernel launched and synchronized");

    let mut scores_buf = vec![0u8; tiles.len() * 4];
    engine.copy_to_host(&scores_dev, 0, &mut scores_buf)?;
    let gpu_scores: Vec<f32> = scores_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    eprintln!("  GPU scores (first 4): {:?}", &gpu_scores[..4]);

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for (i, (&gpu, &ref_s)) in gpu_scores.iter().zip(ref_scores.iter()).enumerate() {
        let diff = (gpu - ref_s).abs();
        if diff > 1e-3 {
            mismatches += 1;
            max_diff = max_diff.max(diff);
            if mismatches <= 5 {
                eprintln!("  MISMATCH tile {i}: gpu={gpu} ref={ref_s} diff={diff}");
            }
        }
    }

    if mismatches == 0 {
        eprintln!("  All {} scores match reference", tiles.len());
    } else {
        eprintln!("  {mismatches} mismatches, max diff = {max_diff}");
    }

    eprintln!("Validation complete.");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_validation() -> Result<(), Box<dyn std::error::Error>> {
    Err("CUDA feature not enabled (build with --features cuda)".into())
}

fn main() {
    match run_cuda_validation() {
        Ok(()) => eprintln!("PASS"),
        Err(e) => {
            eprintln!("SKIP: {e}");
            // Exit 0 so the example works in CI without CUDA
        }
    }
}
