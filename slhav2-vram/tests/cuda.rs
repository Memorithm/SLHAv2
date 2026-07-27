#![cfg(feature = "cuda")]
#![allow(
    clippy::needless_range_loop,
    dead_code,
    clippy::cloned_ref_to_slice_refs
)]

//! CUDA hardware integration tests.
//! Requires: `SLHAV2_REQUIRE_CUDA=1 cargo test --features cuda -- --ignored --nocapture --test-threads=1`

use scirust::SciRustSlhaTile;

use slhav2_vram::backends::cuda::{CudaAllocation, CudaEngine, CudaModule};
use slhav2_vram::codec;
use slhav2_vram::mem::arena::DeviceArena;
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
    let scores_dev = engine.allocate(num_tiles * 4).unwrap();

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
            let denom = c.abs().max(1.0e-6);
            (g - c).abs() / denom
        })
        .fold(0.0f32, f32::max);

    let dot: f32 = gpu.iter().zip(cpu.iter()).map(|(g, c)| g * c).sum();
    let gpu_norm: f32 = gpu.iter().map(|g| g * g).sum::<f32>().sqrt();
    let cpu_norm: f32 = cpu.iter().map(|c| c * c).sum::<f32>().sqrt();
    let cos_sim = if gpu_norm > 0.0 && cpu_norm > 0.0 {
        (dot / (gpu_norm * cpu_norm)).clamp(-1.0, 1.0)
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
    for &(byte_val, _expected_level) in &[(0x88u8, 0.0f32), (0x00, -8.0), (0xFF, 7.0)] {
        let q = make_q_coarse(1.0);
        let qs = make_q_sign(0);
        let tile = make_int4_tile(byte_val, 1.0, 0.0, &[0; 4], codec::FLAG_HOT);
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
            let lo = (((i + j) * 2) % 16) as u8;
            let hi = (((i + j) * 2 + 1) % 16) as u8;
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

    for i in 0..1000 {
        let scores = score_tiles_gpu(&engine, &module, &[tile.clone()], &q, &qs);
        assert!(
            scores[0].abs() < 1e-4,
            "launch {i}: expected 0, got {}",
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
fn test_cuda_real_backing_arena() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let initial_count = CudaEngine::backing_allocation_count();

    let capacity: usize = 16 * 1024 * 1024;
    let backing = engine.allocate(capacity).expect("16 MiB allocation");
    assert_eq!(
        CudaEngine::backing_allocation_count(),
        initial_count + 1,
        "count after one allocation"
    );

    let mut arena: DeviceArena<CudaAllocation> = DeviceArena::new(backing, capacity as u64);

    let mut slices = Vec::new();
    for _ in 0..1000 {
        let s = arena.allocate(128).expect("arena suballocation");
        assert_eq!(s.offset % 256, 0, "offset not 256-byte aligned");
        slices.push(s);
    }

    let mut ranges: Vec<(u64, u64)> = slices
        .iter()
        .map(|s| (s.offset, s.offset + s.size))
        .collect();
    ranges.sort_by_key(|r| r.0);
    for w in ranges.windows(2) {
        assert!(
            w[0].1 <= w[1].0,
            "overlapping ranges: {:?} and {:?}",
            w[0],
            w[1]
        );
    }
    for r in &ranges {
        assert!(r.1 <= arena.capacity(), "range exceeds capacity: {:?}", r);
    }

    assert_eq!(
        CudaEngine::backing_allocation_count(),
        initial_count + 1,
        "count still initial+1 after suballocations"
    );

    for s in &slices {
        arena.free(s);
    }

    assert_eq!(arena.live_allocations(), 0);
    assert_eq!(arena.free_bytes(), arena.capacity());

    assert_eq!(arena.free_bytes(), arena.capacity());
    assert_eq!(arena.free_bytes(), capacity as u64);

    drop(arena);

    assert_eq!(
        CudaEngine::backing_allocation_count(),
        initial_count,
        "count returned to initial after arena drop"
    );
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_arena_scoring_integration() {
    check_required();
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");
    let kernel = module
        .get_function("slha_score_kernel")
        .expect("kernel function lookup");

    let q_coarse = make_q_coarse(0.01);
    let q_sign = make_q_sign(0xDEAD_BEEF_CAFE_FACE);

    let mut tiles = Vec::new();
    for i in 0..32 {
        let tile = make_int4_tile(
            (i as u8).wrapping_mul(0x11),
            0.5 + i as f32 * 0.05,
            0.05,
            &[0xDEAD_BEEF_CAFE_FACEu64; 4],
            if i % 3 == 0 {
                codec::FLAG_WARM
            } else {
                codec::FLAG_HOT
            },
        );
        tiles.push(tile);
    }

    let q_coarse_arr: &[f32; codec::D_C] = q_coarse[..].try_into().expect("D_C length");
    let q_sign_arr: &[u64; codec::RESIDUAL_WORDS] =
        q_sign[..].try_into().expect("RESIDUAL_WORDS length");

    let reference: Vec<f32> = tiles
        .iter()
        .map(|t| {
            let st = t.to_slha_tile();
            st.compute_score_scalar(q_coarse_arr, q_sign_arr)
        })
        .collect();

    let q_coarse_bytes = codec::f32_slice_to_le_bytes(&q_coarse);
    let q_sign_bytes = codec::u64_slice_to_le_bytes(&q_sign);
    let total_tile_bytes = tiles.len() * codec::TILE_BYTES;
    let mut tiles_buf = vec![0u8; total_tile_bytes];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        tiles_buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    let scores_bytes_len = tiles.len() * 4;

    let capacity =
        (q_coarse_bytes.len() + q_sign_bytes.len() + total_tile_bytes + scores_bytes_len + 4096)
            as u64;
    let backing = engine
        .allocate(capacity as usize)
        .expect("arena backing allocation");
    let mut arena: DeviceArena<CudaAllocation> = DeviceArena::new(backing, capacity);

    let q_coarse_slice = arena
        .allocate(q_coarse_bytes.len() as u64)
        .expect("q_coarse arena slice");
    let q_sign_slice = arena
        .allocate(q_sign_bytes.len() as u64)
        .expect("q_sign arena slice");
    let tiles_slice = arena
        .allocate(total_tile_bytes as u64)
        .expect("tiles arena slice");
    let scores_slice = arena
        .allocate(scores_bytes_len as u64)
        .expect("scores arena slice");

    engine
        .copy_to_device_at(
            &q_coarse_bytes,
            arena.backing_mut(),
            q_coarse_slice.offset as usize,
        )
        .expect("copy q_coarse to device at offset");

    engine
        .copy_to_device_at(
            &q_sign_bytes,
            arena.backing_mut(),
            q_sign_slice.offset as usize,
        )
        .expect("copy q_sign to device at offset");

    engine
        .copy_to_device_at(&tiles_buf, arena.backing_mut(), tiles_slice.offset as usize)
        .expect("copy tiles to device at offset");

    engine
        .score_tiles_at(
            arena.backing(),
            q_coarse_slice.offset as usize,
            arena.backing(),
            q_sign_slice.offset as usize,
            arena.backing(),
            tiles_slice.offset as usize,
            arena.backing(),
            scores_slice.offset as usize,
            tiles.len() as i32,
            &kernel,
        )
        .expect("score_tiles_at kernel launch");
    engine.synchronize().expect("CUDA synchronize");

    let mut scores_bytes = vec![0u8; scores_bytes_len];
    engine
        .copy_to_host_at(
            arena.backing(),
            scores_slice.offset as usize,
            &mut scores_bytes,
        )
        .expect("copy scores from device at offset");

    let gpu_scores = codec::le_bytes_to_f32_vec(&scores_bytes).expect("decode scores");

    assert_eq!(gpu_scores.len(), reference.len(), "score count mismatch");

    for (i, s) in gpu_scores.iter().enumerate() {
        assert!(s.is_finite(), "non-finite GPU score at index {i}: {s}");
    }

    let tolerance: f32 = 1e-3;
    let (max_abs, max_rel, cos_sim) = calc_metrics(&gpu_scores, &reference);

    let worst = gpu_scores
        .iter()
        .zip(reference.iter())
        .enumerate()
        .max_by(|a, b| {
            let da = (a.1 .0 - a.1 .1).abs();
            let db = (b.1 .0 - b.1 .1).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();

    eprintln!(
        "Arena scoring: max_abs={:.6e}  max_rel={:.6e}  cos_sim={:.10}",
        max_abs, max_rel, cos_sim
    );
    eprintln!(
        "Worst mismatch: index {}  gpu={}  cpu={}  diff={:.6e}",
        worst.0,
        worst.1 .0,
        worst.1 .1,
        (worst.1 .0 - worst.1 .1).abs()
    );

    assert!(
        max_abs < tolerance,
        "arena scoring max_abs {:.6e} exceeds tolerance {tolerance}",
        max_abs
    );
    assert!(cos_sim > 0.9999, "arena scoring cos_sim {cos_sim}");
}

#[test]
#[ignore = "requires an NVIDIA CUDA GPU"]
fn test_cuda_alloc_reuse_after_drop() {
    let engine = CudaEngine::new().expect("CudaEngine::new");
    let _module = CudaModule::load_ptx(&engine, ptx_bytes()).expect("PTX load");

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
