use scirust::SciRustSlhaTile;

use slhav2_vram::backends::cpu::CpuEngine;
use slhav2_vram::codec;
use slhav2_vram::mem::tile::SerializedTile;
use slhav2_vram::pipeline::{score_tiles_cpu, ScoringInput};
use slhav2_vram::traits::{DeviceAllocation, DeviceEngine};

fn ref_tile(
    latent: [u8; codec::LATENT_BYTES],
    scale: f32,
    lambda: f32,
    residual: [u64; codec::RESIDUAL_WORDS],
    flags: u16,
    group_scales: [u8; codec::N_GROUP_SCALES],
) -> SciRustSlhaTile {
    SciRustSlhaTile {
        latent_kv: latent,
        residual_bitmap: residual,
        scale,
        dynamic_lambda: lambda,
        residual_sigma: 0.0,
        token_id: 0,
        position: 0,
        head_id: 0,
        flags,
        group_scales,
    }
}

fn q_coarse_all(val: f32) -> [f32; codec::D_C] {
    [val; codec::D_C]
}

fn q_sign_all(val: u64) -> [u64; codec::RESIDUAL_WORDS] {
    [val; codec::RESIDUAL_WORDS]
}

fn fill_latent(val: u8) -> [u8; codec::LATENT_BYTES] {
    [val; codec::LATENT_BYTES]
}

fn fill_latent_seq() -> [u8; codec::LATENT_BYTES] {
    let mut l = [0u8; codec::LATENT_BYTES];
    for i in 0..codec::LATENT_BYTES {
        l[i] = i as u8;
    }
    l
}

#[test]
fn test_int4_zero_point_0x88() {
    // 0x88 → both nibbles = 8 → level = 0 → score should be 0
    let latent = fill_latent(0x88);
    let gs = [255u8; 8];
    let q = q_coarse_all(1.0);
    let qs = q_sign_all(0);

    let st = ref_tile(latent, 1.0, 0.0, [0; 4], codec::FLAG_HOT, gs);
    let expected = st.compute_score_scalar(&q, &qs);

    assert!(expected.abs() < 1e-6, "scirust: 0x88 → 0, got {expected}");

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&fill_latent(0x88));
    stile.set_scale(1.0);
    stile.set_group_scales(&[255u8; 8]);
    let got = stile.score(&q, &qs);
    assert!((got - 0.0).abs() < 1e-4, "slhav2-vram: 0x88 → 0, got {got}");
    assert!((got - expected).abs() < 1e-4, "parity: {got} vs {expected}");
}

#[test]
fn test_int4_zero_point_0x00() {
    // 0x00 → both nibbles = 0 → level = -8 each
    let latent = fill_latent(0x00);
    let gs = [255u8; 8];
    let q = q_coarse_all(1.0);
    let qs = q_sign_all(0);

    let st = ref_tile(latent, 1.0, 0.0, [0; 4], codec::FLAG_HOT, gs);
    let expected = st.compute_score_scalar(&q, &qs);
    let expected_numerical: f32 = codec::D_C as f32 * (-8.0);
    assert!((expected - expected_numerical).abs() < 1e-4);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&fill_latent(0x00));
    stile.set_scale(1.0);
    stile.set_group_scales(&[255u8; 8]);
    let got = stile.score(&q, &qs);
    assert!((got - expected).abs() < 1e-4, "parity: {got} vs {expected}");
}

#[test]
fn test_int4_zero_point_0xff() {
    // 0xFF → both nibbles = 0xF → level = 7 each
    let latent = fill_latent(0xFF);
    let gs = [255u8; 8];
    let q = q_coarse_all(1.0);
    let qs = q_sign_all(0);

    let st = ref_tile(latent, 1.0, 0.0, [0; 4], codec::FLAG_HOT, gs);
    let expected = st.compute_score_scalar(&q, &qs);
    let expected_numerical: f32 = codec::D_C as f32 * 7.0;
    assert!((expected - expected_numerical).abs() < 1e-4);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&fill_latent(0xFF));
    stile.set_scale(1.0);
    stile.set_group_scales(&[255u8; 8]);
    let got = stile.score(&q, &qs);
    assert!((got - expected).abs() < 1e-4, "parity: {got} vs {expected}");
}

#[test]
fn test_int4_hot_parity() {
    let latent = fill_latent_seq();
    let residual = q_sign_all(0xDEAD_BEEF_CAFE_FACE);
    let gs = [128, 150, 100, 200, 180, 90, 210, 140];

    let st = ref_tile(latent, 0.75, 0.05, residual, codec::FLAG_HOT, gs);
    let q = q_coarse_all(0.01);
    let qs = q_sign_all(0xDEAD_BEEF_CAFE_FACE); // match residual → zero hamming
    let expected = st.compute_score_scalar(&q, &qs);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&latent);
    stile.set_residual(&residual);
    stile.set_scale(0.75);
    stile.set_dynamic_lambda(0.05);
    stile.set_group_scales(&gs);
    let got = stile.score(&q, &qs);

    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4,
        "INT4 HOT parity: {got} vs {expected}, diff {diff}"
    );
}

#[test]
fn test_int4_warm_parity() {
    let latent = fill_latent_seq();
    let gs = [128, 150, 100, 200, 180, 90, 210, 140];
    let residual = [0u64; 4];

    let st = ref_tile(latent, 0.75, 0.0, residual, codec::FLAG_WARM, gs);
    let q = q_coarse_all(0.01);
    let qs = q_sign_all(0);
    let expected = st.compute_score_scalar(&q, &qs);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&latent);
    stile.set_scale(0.75);
    stile.set_dynamic_lambda(0.05); // should be ignored in WARM mode
    stile.set_group_scales(&gs);
    stile.set_flags(codec::FLAG_WARM);
    let got = stile.score(&q, &qs);

    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4,
        "INT4 WARM parity: {got} vs {expected}, diff {diff}"
    );
}

#[test]
fn test_nf4_hot_parity() {
    let mut latent = [0u8; codec::LATENT_BYTES];
    // Fill with NF4 codeword indices 0..15 repeating
    for i in 0..codec::LATENT_BYTES {
        let lo = ((i * 2) % 16) as u8;
        let hi = ((i * 2 + 1) % 16) as u8;
        latent[i] = (hi << 4) | lo;
    }
    let residual = q_sign_all(0x1234_5678_9ABC_DEF0);
    let gs = [200u8; 8];

    let mut st = ref_tile(latent, 1.0, 0.1, residual, codec::FLAG_NF4, gs);
    st.flags |= codec::FLAG_NF4;
    let q = q_coarse_all(0.5);
    let qs = q_sign_all(0x1234_5678_9ABC_DEF0);
    let expected = st.compute_score_scalar(&q, &qs);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&latent);
    stile.set_residual(&residual);
    stile.set_scale(1.0);
    stile.set_dynamic_lambda(0.1);
    stile.set_group_scales(&gs);
    stile.set_flags(codec::FLAG_NF4);
    let got = stile.score(&q, &qs);

    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4,
        "NF4 HOT parity: {got} vs {expected}, diff {diff}"
    );
}

#[test]
fn test_nf4_warm_parity() {
    let mut latent = [0u8; codec::LATENT_BYTES];
    for i in 0..codec::LATENT_BYTES {
        let lo = ((i * 2 + 3) % 16) as u8;
        let hi = ((i * 2 + 5) % 16) as u8;
        latent[i] = (hi << 4) | lo;
    }
    let gs = [200u8; 8];

    let mut st = ref_tile(
        latent,
        1.0,
        0.0,
        [0; 4],
        codec::FLAG_WARM | codec::FLAG_NF4,
        gs,
    );
    st.flags |= codec::FLAG_NF4;
    let q = q_coarse_all(0.3);
    let qs = q_sign_all(0);
    let expected = st.compute_score_scalar(&q, &qs);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&latent);
    stile.set_scale(1.0);
    stile.set_group_scales(&gs);
    stile.set_flags(codec::FLAG_WARM | codec::FLAG_NF4);
    let got = stile.score(&q, &qs);

    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4,
        "NF4 WARM parity: {got} vs {expected}, diff {diff}"
    );
}

#[test]
fn test_flag_precedence_nf4_takes_priority() {
    // When both INT4 (no flags) and NF4 are set, NF4 should be used.
    // The scirust dequant_at checks is_nf4() first.
    let latent = fill_latent(0x88); // 0x88 → INT4 level 0, NF4 codebook index 8
    let gs = [255u8; 8];
    let q = q_coarse_all(1.0);
    let qs = q_sign_all(0);

    let mut st = ref_tile(latent, 1.0, 0.0, [0; 4], codec::FLAG_NF4, gs);
    st.flags = codec::FLAG_NF4;
    let expected = st.compute_score_scalar(&q, &qs);

    // NF4: nibble 0x8 → codebook index 8 → value 0.0421
    let expected_nf4: f32 = codec::D_C as f32 * 0.0421;
    assert!((expected - expected_nf4).abs() < 1e-4);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&fill_latent(0x88));
    stile.set_scale(1.0);
    stile.set_group_scales(&[255u8; 8]);
    stile.set_flags(codec::FLAG_NF4);
    let got = stile.score(&q, &qs);
    assert!((got - expected).abs() < 1e-4, "parity: {got} vs {expected}");
}

#[test]
fn test_cpu_backend_basics() {
    let engine = CpuEngine::new();
    let alloc = engine.allocate(1024).unwrap();
    assert_eq!(alloc.size(), 1024);

    let mut dst = engine.allocate(16).unwrap();
    let src = b"hello cpu world!";
    engine.copy_to_device(src, &mut dst, 0).unwrap();

    let mut buf = vec![0u8; src.len()];
    engine.copy_to_host(&dst, 0, &mut buf).unwrap();
    assert_eq!(&buf, src);
}

#[test]
fn test_pipeline_batch_parity() {
    let engine = CpuEngine::new();

    let tiles: Vec<SerializedTile> = (0..8)
        .map(|i| {
            let mut t = SerializedTile::zeroed();
            let val = if i % 2 == 0 { 0x88 } else { 0x12 };
            t.latent_mut().copy_from_slice(&fill_latent(val));
            t.set_scale(1.0 + i as f32 * 0.1);
            t.set_group_scales(&[128u8; 8]);
            if i % 3 == 0 {
                t.set_flags(codec::FLAG_WARM);
            }
            t
        })
        .collect();

    let q = q_coarse_all(0.01);
    let qs = q_sign_all(0xABCDEF0123456789);
    let mut scores = vec![0.0f32; tiles.len()];

    score_tiles_cpu(ScoringInput {
        engine: &engine,
        tiles: &tiles,
        q_coarse: &q,
        q_sign: &qs,
        scores: &mut scores,
    });

    for (i, t) in tiles.iter().enumerate() {
        let expected = t.score(&q, &qs);
        assert!(
            (scores[i] - expected).abs() < 1e-4,
            "tile {i}: pipeline {} vs direct {expected}",
            scores[i]
        );
    }
}

#[test]
fn test_slha_tile_to_serialized_parity() {
    // Build a SciRustSlhaTile, compute score, then copy bytes into
    // SerializedTile and verify the score matches.
    let residual = [0xDEAD_BEEF_CAFE_FACEu64, 0, 0, 0];
    let gs = [128, 150, 100, 200, 180, 90, 210, 140];
    let latent = fill_latent_seq();

    let st = ref_tile(latent, 0.75, 0.05, residual, codec::FLAG_NF4, gs);
    let q = q_coarse_all(0.01);
    let qs = q_sign_all(0xDEAD_BEEF_CAFE_FACE);
    let expected = st.compute_score_scalar(&q, &qs);

    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&st.latent_kv);
    stile.set_residual(&st.residual_bitmap);
    stile.set_scale(st.scale);
    stile.set_dynamic_lambda(st.dynamic_lambda);
    stile.set_group_scales(&st.group_scales);
    stile.set_flags(st.flags);
    let got = stile.score(&q, &qs);

    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4,
        "SciRustSlhaTile→SerializedTile parity: {got} vs {expected}, diff {diff}"
    );
}
