#![cfg(feature = "cuda")]

//! Hardware-gated CUDA integration tests.
//! Requires:  `cargo test --features cuda --test cuda -- --ignored`
//! Requires:  `SLHAV2_TEST_CUDA=1` env var
//! Requires:  nvcc-compiled PTX in OUT_DIR

use slhav2_vram::codec;
use slhav2_vram::mem::tile::SerializedTile;
use slhav2_vram::traits::{DeviceAllocation, DeviceEngine};

fn has_cuda_hardware() -> bool {
    std::env::var("SLHAV2_TEST_CUDA").map(|v| v == "1").unwrap_or(false)
}

fn cuda_init() -> Option<slhav2_vram::backends::cuda::CudaEngine> {
    if !has_cuda_hardware() {
        return None;
    }
    match slhav2_vram::backends::cuda::CudaEngine::new() {
        Ok(eng) => Some(eng),
        Err(e) => {
            eprintln!("CUDA init skipped: {e}");
            None
        }
    }
}

fn load_kernel(engine: &slhav2_vram::backends::cuda::CudaEngine) -> Option<slhav2_vram::backends::cuda::CudaFunction> {
    let ptx_bytes = std::fs::read(
        std::path::Path::new(&std::env::var("OUT_DIR").unwrap_or_default()).join("slha_score.ptx"),
    )
    .or_else(|_| {
        // Fallback: try from CARGO_MANIFEST_DIR
        std::fs::read(
            std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default())
                .join("kernels")
                .join("slha_score.ptx"),
        )
    })
    .ok()?;

    engine.load_ptx(&ptx_bytes).ok()
}

fn make_test_tiles(n: usize) -> Vec<SerializedTile> {
    let mut tiles = Vec::with_capacity(n);
    for i in 0..n {
        let mut t = SerializedTile::zeroed();
        let scale = 0.5 + (i as f32) * 0.1;
        t.set_scale(scale);
        t.set_dynamic_lambda(0.05);
        t.set_group_scales(&[128u8; 8]);
        t.set_flags(0);

        for d in 0..codec::LATENT_KV_WORDS {
            let lo = ((i + d) as u8) & 0x0F;
            let hi = (((i + d) >> 4) as u8) & 0x0F;
            t.as_bytes_mut()[d] = (hi << 4) | lo;
        }
        tiles.push(t);
    }
    tiles
}

fn cast_to_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    let byte_len = slice.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) }
}

#[test]
#[ignore]
fn test_cuda_allocate_and_copy() {
    let engine = match cuda_init() {
        Some(e) => e,
        None => return,
    };
    let mut dev_alloc = engine.allocate(256).unwrap();
    assert!(dev_alloc.size() >= 256);

    let src = b"hello, cuda world!";
    engine.copy_to_device(src, &mut dev_alloc, 0).unwrap();

    let mut dst = vec![0u8; src.len()];
    engine.copy_to_host(&dev_alloc, 0, &mut dst).unwrap();
    assert_eq!(&dst, src);
}

#[test]
#[ignore]
fn test_cuda_score_against_cpu() {
    let engine = match cuda_init() {
        Some(e) => e,
        None => return,
    };

    let kernel_fn = match load_kernel(&engine) {
        Some(k) => k,
        None => {
            eprintln!("PTX not found; skipping");
            return;
        }
    };

    let tiles = make_test_tiles(16);
    let q_coarse: Vec<f32> = (0..codec::D_C).map(|i| (i as f32) * 0.01).collect();
    let q_sign: Vec<u64> = vec![0xDEADBEEF_CAFEBABEu64; codec::RESIDUAL_WORDS];

    let ref_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q_coarse, &q_sign)).collect();

    let total_tile_bytes = tiles.len() * codec::TILE_BYTES;
    let mut q_coarse_dev = engine.allocate(codec::D_C * 4).unwrap();
    let mut q_sign_dev = engine.allocate(codec::RESIDUAL_WORDS * 8).unwrap();
    let mut tiles_dev = engine.allocate(total_tile_bytes).unwrap();
    let mut scores_dev = engine.allocate(tiles.len() * 4).unwrap();

    engine.copy_to_device(cast_to_bytes(&q_coarse), &mut q_coarse_dev, 0).unwrap();
    engine.copy_to_device(cast_to_bytes(&q_sign), &mut q_sign_dev, 0).unwrap();

    let mut tiles_buf = vec![0u8; total_tile_bytes];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        tiles_buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine.copy_to_device(&tiles_buf, &mut tiles_dev, 0).unwrap();

    engine.score_tiles(
        &q_coarse_dev,
        &q_sign_dev,
        &tiles_dev,
        &scores_dev,
        tiles.len() as i32,
        &kernel_fn,
    ).unwrap();
    engine.synchronize().unwrap();

    let mut scores_buf = vec![0u8; tiles.len() * 4];
    engine.copy_to_host(&scores_dev, 0, &mut scores_buf).unwrap();
    let gpu_scores: Vec<f32> = scores_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    for (i, (&gpu, &ref_s)) in gpu_scores.iter().zip(ref_scores.iter()).enumerate() {
        let diff = (gpu - ref_s).abs();
        assert!(diff < 1e-3, "tile {i}: gpu {gpu} vs ref {ref_s}, diff {diff}");
    }
}

#[test]
#[ignore]
fn test_cuda_large_batch() {
    let engine = match cuda_init() {
        Some(e) => e,
        None => return,
    };

    let kernel_fn = match load_kernel(&engine) {
        Some(k) => k,
        None => {
            eprintln!("PTX not found; skipping");
            return;
        }
    };

    let num_tiles = 1024;
    let tiles = make_test_tiles(num_tiles);
    let q_coarse: Vec<f32> = vec![0.01; codec::D_C];
    let q_sign: Vec<u64> = vec![0; codec::RESIDUAL_WORDS];

    let ref_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q_coarse, &q_sign)).collect();

    let total = num_tiles * codec::TILE_BYTES;
    let mut q_coarse_dev = engine.allocate(codec::D_C * 4).unwrap();
    let mut q_sign_dev = engine.allocate(codec::RESIDUAL_WORDS * 8).unwrap();
    let mut tiles_dev = engine.allocate(total).unwrap();
    let mut scores_dev = engine.allocate(num_tiles * 4).unwrap();

    engine.copy_to_device(cast_to_bytes(&q_coarse), &mut q_coarse_dev, 0).unwrap();
    engine.copy_to_device(cast_to_bytes(&q_sign), &mut q_sign_dev, 0).unwrap();

    let mut tiles_buf = vec![0u8; total];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        tiles_buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine.copy_to_device(&tiles_buf, &mut tiles_dev, 0).unwrap();

    engine.score_tiles(
        &q_coarse_dev,
        &q_sign_dev,
        &tiles_dev,
        &scores_dev,
        num_tiles as i32,
        &kernel_fn,
    ).unwrap();
    engine.synchronize().unwrap();

    let mut scores_buf = vec![0u8; num_tiles * 4];
    engine.copy_to_host(&scores_dev, 0, &mut scores_buf).unwrap();
    let gpu_scores: Vec<f32> = scores_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    for (i, (&gpu, &ref_s)) in gpu_scores.iter().zip(ref_scores.iter()).enumerate() {
        let diff = (gpu - ref_s).abs();
        assert!(diff < 2e-3, "tile {i}: gpu {gpu} vs ref {ref_s}, diff {diff}");
    }
}
