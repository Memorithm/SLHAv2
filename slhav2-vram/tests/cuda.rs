#![cfg(feature = "cuda")]

//! CUDA hardware integration tests.
//! Requires: `SLHAV2_REQUIRE_CUDA=1 cargo test --features cuda -- --ignored --nocapture --test-threads=1`

use scirust::SciRustSlhaTile;

use slhav2_vram::backends::cuda::{CudaEngine, CudaModule};
use slhav2_vram::codec;
use slhav2_vram::mem::tile::SerializedTile;
use slhav2_vram::traits::{DeviceAllocation, DeviceEngine};

fn check_required() {
    if std::env::var("SLHAV2_REQUIRE_CUDA")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return;
    }
    panic!("Set SLHAV2_REQUIRE_CUDA=1 to run CUDA tests");
}

fn ptx_bytes() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/slha_score.ptx"))
}

fn make_q_coarse(val: f32) -> Vec<f32> {
    vec![val; codec::D_C]
}

fn make_q_sign(val: u64) -> Vec<u64> {
    vec![val; codec::RESIDUAL_WORDS]
}

fn make_int4_tile(
    latent_byte: u8,
    scale: f32,
    lambda: f32,
    residual: &[u64; codec::RESIDUAL_WORDS],
    flags: u16,
) -> SerializedTile {
    let mut t = SerializedTile::zeroed();
    for b in t.latent_mut().iter_mut() {
        *b = latent_byte;
    }
    t.set_residual(residual);
    t.set_scale(scale);
    t.set_dynamic_lambda(lambda);
    t.set_group_scales(&[255u8; 8]);
    t.set_flags(flags);
    t
}

fn make_tile_from_scirust(st: &SciRustSlhaTile) -> SerializedTile {
    let mut t = SerializedTile::zeroed();
    t.latent_mut().copy_from_slice(&st.latent_kv);
    t.set_residual(&st.residual_bitmap);
    t.set_scale(st.scale);
    t.set_dynamic_lambda(st.dynamic_lambda);
    t.set_group_scales(&st.group_scales);
    t.set_flags(st.flags);
    t
}

fn score_tiles_gpu(
    engine: &CudaEngine,
    module: &CudaModule,
    tiles: &[SerializedTile],
    q_coarse: &[f32],
    q_sign: &[u64],
) -> Vec<f32> {
    let num_tiles = tiles.len();
    let total_tile_bytes = num_tiles * codec::TILE_BYTES;

    let mut q_coarse_dev = engine.allocate(q_coarse.len() * 4).unwrap();
    let mut q_sign_dev = engine.allocate(q_sign.len() * 8).unwrap();
    let mut tiles_dev = engine.allocate(total_tile_bytes).unwrap();
    let mut scores_dev = engine.allocate(num_tiles * 4).unwrap();

    engine
        .copy_to_device(
            &codec::f32_slice_to_le_bytes(q_coarse),
            &mut q_coarse_dev,
            0,
        )
        .unwrap();
    engine
        .copy_to_device(&codec::u64_slice_to_le_bytes(q_sign), &mut q_sign_dev, 0)
        .unwrap();

    let mut tiles_buf = vec![0u8; total_tile_bytes];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        tiles_buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine
        .copy_to_device(&tiles_buf, &mut tiles_dev, 0)
        .unwrap();

    let kernel = module
        .get_function("slha_score_kernel")
        .expect("kernel function lookup");
    engine
        .score_tiles(
            &q_coarse_dev,
            &q_sign_dev,
            &tiles_dev,
            &scores_dev,
            num_tiles as i32,
            &kernel,
        )
        .unwrap();
    engine.synchronize().unwrap();

    let mut scores_buf = vec![0u8; num_tiles * 4];
    engine
        .copy_to_host(&scores_dev, 0, &mut scores_buf)
        .unwrap();
    codec::le_bytes_to_f32_vec(&scores_buf).unwrap()
}

fn calc_metrics(gpu: &[f32], cpu: &[f32]) -> (f32, f32, f32) {
    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    let max_rel = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(g, c)| {
            let denom = c.abs().max(1e-10);
            (g - c).abs() / denom
        })
        .fold(0.0f32, f32::max);

    let dot: f32 = gpu.iter().zip(cpu.iter()).map(|(g, c)| g * c).sum();
    let gpu_norm: f32 = gpu.iter().map(|g| g * g).sum::<f32>().sqrt();
    let cpu_norm: f32 = cpu.iter().map(|c| c * c).sum::<f32>().sqrt();
    let cos_sim = if gpu_norm > 0.0 && cpu_norm > 0.0 {
        dot / (gpu_norm * cpu_norm)
    } else {
        1.0
    };

    (max_abs, max_rel, cos_sim)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_device_metadata() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    // If we got here without panicking, CUDA is available.
    // (No public API to query device metadata currently.)
    drop(engine);
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_64_mib_roundtrip() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let size = 64 * 1024 * 1024; // 64 MiB
    let mut dev = engine.allocate(size).unwrap();
    assert_eq!(dev.size(), size);

    let src: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    engine.copy_to_device(&src, &mut dev, 0).unwrap();

    let mut dst = vec![0u8; size];
    engine.copy_to_host(&dev, 0, &mut dst).unwrap();
    assert_eq!(src, dst, "64 MiB round-trip mismatch");
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_int4_zero_point_parity() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    // Test all three zero-point cases
    for &(byte_val, expected_level) in &[(0x88u8, 0.0f32), (0x00, -8.0), (0xFF, 7.0)] {
        let q = make_q_coarse(1.0);
        let qs = make_q_sign(0);
        let mut tile = make_int4_tile(byte_val, 1.0, 0.0, &[0; 4], codec::FLAG_HOT);
        let cpu_score = tile.score(&q, &qs);

        let gpu_scores = score_tiles_gpu(&engine, &module, &[tile], &q, &qs);
        let gpu_score = gpu_scores[0];

        let diff = (gpu_score - cpu_score).abs();
        assert!(
            diff < 1e-4,
            "byte 0x{byte_val:02X}: gpu {gpu_score} vs cpu {cpu_score}, diff {diff}"
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_hot_score_parity() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    let q = make_q_coarse(0.01);
    let qs = make_q_sign(0xDEAD_BEEF_CAFE_FACE);

    let mut tiles = Vec::new();
    for i in 0..32 {
        let latent_byte = (i as u8).wrapping_mul(0x11);
        let tile = make_int4_tile(
            latent_byte,
            0.5 + i as f32 * 0.05,
            0.05,
            &[0xDEAD_BEEF_CAFE_FACEu64; 4],
            codec::FLAG_HOT,
        );
        tiles.push(tile);
    }

    let cpu_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q, &qs)).collect();
    let gpu_scores = score_tiles_gpu(&engine, &module, &tiles, &q, &qs);

    let (max_abs, max_rel, cos_sim) = calc_metrics(&gpu_scores, &cpu_scores);

    eprintln!(
        "HOT: max_abs={:.6e}  max_rel={:.6e}  cos_sim={:.10}",
        max_abs, max_rel, cos_sim
    );
    assert!(max_abs < 1e-3, "HOT max abs error {max_abs:.6e}");
    assert!(cos_sim > 0.9999, "HOT cosine similarity {cos_sim}");
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_warm_score_parity() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    let q = make_q_coarse(0.01);
    let qs = make_q_sign(0);

    let mut tiles = Vec::new();
    for i in 0..32 {
        let latent_byte = (i as u8).wrapping_mul(0x22);
        let tile = make_int4_tile(
            latent_byte,
            0.5 + i as f32 * 0.05,
            0.05,
            &[0; 4],
            codec::FLAG_WARM,
        );
        tiles.push(tile);
    }

    let cpu_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q, &qs)).collect();
    let gpu_scores = score_tiles_gpu(&engine, &module, &tiles, &q, &qs);

    let (max_abs, max_rel, cos_sim) = calc_metrics(&gpu_scores, &cpu_scores);

    eprintln!(
        "WARM: max_abs={:.6e}  max_rel={:.6e}  cos_sim={:.10}",
        max_abs, max_rel, cos_sim
    );
    assert!(max_abs < 1e-3, "WARM max abs error {max_abs:.6e}");
    assert!(cos_sim > 0.9999);
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_nf4_score_parity() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    let q = make_q_coarse(0.5);
    let qs = make_q_sign(0x1234_5678_9ABC_DEF0);

    let mut tiles = Vec::new();
    for i in 0..32 {
        let mut latent = [0u8; codec::LATENT_BYTES];
        for j in 0..codec::LATENT_BYTES {
            let lo = ((i + j) * 2) % 16;
            let hi = ((i + j) * 2 + 1) % 16;
            latent[j] = (hi << 4) | lo;
        }
        let mut t = SerializedTile::zeroed();
        t.latent_mut().copy_from_slice(&latent);
        t.set_residual(&[0x1234_5678_9ABC_DEF0u64; 4]);
        t.set_scale(0.5 + i as f32 * 0.1);
        t.set_dynamic_lambda(0.1);
        t.set_group_scales(&[200u8; 8]);
        t.set_flags(codec::FLAG_NF4);
        tiles.push(t);
    }

    let cpu_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q, &qs)).collect();
    let gpu_scores = score_tiles_gpu(&engine, &module, &tiles, &q, &qs);

    let (max_abs, max_rel, cos_sim) = calc_metrics(&gpu_scores, &cpu_scores);

    eprintln!(
        "NF4: max_abs={:.6e}  max_rel={:.6e}  cos_sim={:.10}",
        max_abs, max_rel, cos_sim
    );
    assert!(max_abs < 1e-3);
    assert!(cos_sim > 0.9999);
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_deterministic_1024_tiles() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    let q = make_q_coarse(0.01);
    let qs = make_q_sign(0xDEAD_BEEF_CAFE_FACE);

    let mut tiles = Vec::with_capacity(1024);
    for i in 0..1024 {
        let tile = make_int4_tile(
            (i as u8).wrapping_mul(0x33),
            0.5 + (i % 20) as f32 * 0.05,
            if i % 3 == 0 { 0.05 } else { 0.1 },
            &[0xDEAD_BEEF_CAFE_FACEu64; 4],
            if i % 5 == 0 {
                codec::FLAG_WARM
            } else {
                codec::FLAG_HOT
            },
        );
        tiles.push(tile);
    }

    let cpu_scores: Vec<f32> = tiles.iter().map(|t| t.score(&q, &qs)).collect();
    let gpu_scores = score_tiles_gpu(&engine, &module, &tiles, &q, &qs);

    let (max_abs, max_rel, cos_sim) = calc_metrics(&gpu_scores, &cpu_scores);

    eprintln!(
        "1024 tiles: max_abs={:.6e}  max_rel={:.6e}  cos_sim={:.10}",
        max_abs, max_rel, cos_sim
    );

    // Find worst mismatch
    let worst = gpu_scores
        .iter()
        .zip(cpu_scores.iter())
        .enumerate()
        .max_by(|a, b| {
            let da = (a.1 .0 - a.1 .1).abs();
            let db = (b.1 .0 - b.1 .1).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    eprintln!(
        "Worst mismatch: tile {}  gpu={}  cpu={}  diff={:.6e}",
        worst.0,
        worst.1 .0,
        worst.1 .1,
        (worst.1 .0 - worst.1 .1).abs()
    );

    assert!(max_abs < 1e-3, "max abs error {max_abs:.6e}");
    assert!(cos_sim > 0.9999);
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_one_thousand_launches() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    let q = make_q_coarse(0.01);
    let qs = make_q_sign(0xCAFE_FACE_DEAD_BEEF);
    let tile = make_int4_tile(0x88, 1.0, 0.0, &[0; 4], codec::FLAG_HOT);

    for _ in 0..1000 {
        let scores = score_tiles_gpu(&engine, &module, &[tile.clone()], &q, &qs);
        assert!(
            scores[0].abs() < 1e-4,
            "launch {}: expected 0, got {}",
            _,
            scores[0]
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_backend_lifecycle() {
    check_required();
    // Create and drop the engine repeatedly to detect resource leaks.
    for _ in 0..100 {
        let engine = CudaEngine::new().expect("CudaEngine::new");
        drop(engine);
    }
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_alloc_reuse_after_drop() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

    // Allocate, write, free, then allocate again and verify the new
    // allocation is writable.
    for _ in 0..50 {
        let mut a = engine.allocate(256).unwrap();
        let src = vec![0xABu8; 256];
        engine.copy_to_device(&src, &mut a, 0).unwrap();
        drop(a);

        let mut b = engine.allocate(256).unwrap();
        engine.copy_to_device(&src, &mut b, 0).unwrap();

        let mut dst = vec![0u8; 256];
        engine.copy_to_host(&b, 0, &mut dst).unwrap();
        assert_eq!(src, dst);
    }
}
