use slhav2_vram::backends::cpu::CpuEngine;
use slhav2_vram::mem::tile::SerializedTile;
use slhav2_vram::pipeline::{score_tiles_cpu, ScoringInput};
use slhav2_vram::traits::{DeviceAllocation, DeviceEngine};
use slhav2_vram::codec;

fn make_test_tile_int4(warm: bool) -> SerializedTile {
    let mut tile = SerializedTile::zeroed();
    // Fill latent with alternating INT4 values
    for d in 0..64 {
        // byte = (nib_hi << 4) | nib_lo
        let lo = ((d as u8) & 0x0F).min(0x0F);
        let hi = ((d as u8) >> 4).min(0x0F);
        tile.as_bytes_mut()[d] = (hi << 4) | lo;
    }
    tile.set_scale(1.0);
    tile.set_dynamic_lambda(0.05);
    tile.set_group_scales(&[128u8; 8]);
    if warm {
        tile.set_flags(codec::FLAG_WARM);
    }
    tile
}

fn make_test_query() -> (Vec<f32>, Vec<u64>) {
    let q_coarse: Vec<f32> = (0..codec::D_C).map(|i| (i as f32) * 0.01).collect();
    let q_sign = vec![0xABCDEF0123456789u64; codec::RESIDUAL_WORDS];
    (q_coarse, q_sign)
}

fn reference_score_int4_host(
    q_coarse: &[f32],
    latent_kv: &[u8],
    scale: f32,
    group_scales: &[u8],
) -> f32 {
    let mut sum = 0.0;
    for d in 0..codec::D_C {
        let byte = latent_kv[d >> 1];
        let nib = if (d & 1) != 0 { byte >> 4 } else { byte & 0x0F };
        let level = (nib as i32 - 8) as f32;
        let gs = scale * (group_scales[d / 16] as f32) * (1.0 / 255.0);
        sum += q_coarse[d] * level * gs;
    }
    sum
}

#[test]
fn test_cpu_engine_allocate() {
    let engine = CpuEngine::new();
    let alloc = engine.allocate(1024).unwrap();
    assert_eq!(alloc.size(), 1024);
}

#[test]
fn test_cpu_engine_copy_roundtrip() {
    let engine = CpuEngine::new();
    let mut alloc = engine.allocate(16).unwrap();
    let src = b"hello cpu world!";
    engine.copy_to_device(src, &mut alloc, 0).unwrap();

    let mut dst = vec![0u8; src.len()];
    engine.copy_to_host(&alloc, 0, &mut dst).unwrap();
    assert_eq!(&dst, src);
}

#[test]
fn test_cpu_score_int4_warm() {
    let tile = make_test_tile_int4(true);
    let (q_coarse, q_sign) = make_test_query();
    let score = tile.score(&q_coarse, &q_sign);
    let ref_score = reference_score_int4_host(
        &q_coarse,
        &tile.as_bytes()[..codec::LATENT_KV_WORDS],
        tile.scale(),
        tile.group_scales(),
    );
    assert!((score - ref_score).abs() < 1e-4);
}

#[test]
fn test_cpu_score_int4_hot() {
    let tile = make_test_tile_int4(false);
    let (q_coarse, q_sign) = make_test_query();
    let score = tile.score(&q_coarse, &q_sign);

    let ref_coarse = reference_score_int4_host(
        &q_coarse,
        &tile.as_bytes()[..codec::LATENT_KV_WORDS],
        tile.scale(),
        tile.group_scales(),
    );
    // With q_sign matching residual (all zeros -> diff), ham = popcount(q_sign ^ 0) = lots
    // Since residual is all zeros in our test tile, ham = popcount of each q_sign word
    let ham_ref: u32 = q_sign.iter().map(|x| x.count_ones()).sum();
    let expected = ref_coarse + tile.dynamic_lambda() * (256.0 - 2.0 * ham_ref as f32);
    assert!((score - expected).abs() < 1e-4);
}

#[test]
fn test_cpu_pipeline_batch() {
    let engine = CpuEngine::new();
    let tiles: Vec<SerializedTile> = (0..4).map(|i| {
        let mut t = make_test_tile_int4(i % 2 == 0);
        t.set_scale(1.0 + i as f32 * 0.1);
        t
    }).collect();
    let (q_coarse, q_sign) = make_test_query();
    let mut scores = vec![0.0f32; tiles.len()];

    score_tiles_cpu(ScoringInput {
        engine: &engine,
        tiles: &tiles,
        q_coarse: &q_coarse,
        q_sign: &q_sign,
        scores: &mut scores,
    });

    for (i, t) in tiles.iter().enumerate() {
        let expected = t.score(&q_coarse, &q_sign);
        assert!((scores[i] - expected).abs() < 1e-4);
    }
}

#[test]
fn test_tile_nf4_flag() {
    let mut tile = SerializedTile::zeroed();
    tile.set_scale(1.0);
    tile.set_group_scales(&[200u8; 8]);
    tile.set_flags(codec::FLAG_NF4);

    // Fill latent with NF4 indices
    for nib_idx in 0..64 {
        tile.as_bytes_mut()[nib_idx] = 0x77; // 0x7 in both nibbles
    }

    let (q_coarse, q_sign) = make_test_query();
    let score = tile.score(&q_coarse, &q_sign);

    // NF4 value for 0x7 = 0.0 (index 7 in codebook)
    let expected: f32 = 0.0;
    assert!((score - expected).abs() < 1e-4);
}

#[test]
fn test_int4_zero_point_not_twos_complement() {
    // Verify: nibble 0x8 -> 0, not -8 (two's complement would be -8)
    let mut tile = SerializedTile::zeroed();
    tile.set_scale(1.0);
    tile.set_group_scales(&[255u8; 8]);

    // Set all nibbles to 0x8
    for byte in tile.as_bytes_mut()[..codec::LATENT_KV_WORDS].iter_mut() {
        *byte = 0x88;
    }

    let q_coarse: Vec<f32> = vec![1.0; codec::D_C];
    let q_sign = vec![0u64; codec::RESIDUAL_WORDS];
    let score = tile.score(&q_coarse, &q_sign);
    // 0x8 -> level = 0, so score should be 0
    assert!((score - 0.0).abs() < 1e-4,
        "nibble 0x8 should decode to 0 (zero-point), got {score}");

    // Set all nibbles to 0x0 -> level = -8
    for byte in tile.as_bytes_mut()[..codec::LATENT_KV_WORDS].iter_mut() {
        *byte = 0x00;
    }
    let score2 = tile.score(&q_coarse, &q_sign);
    let expected2: f32 = q_coarse.iter().sum::<f32>() * (-8.0) * (255.0 / 255.0);
    assert!((score2 - expected2).abs() < 1e-2,
        "nibble 0x0 should decode to -8, got {score2} vs {expected2}");
}
