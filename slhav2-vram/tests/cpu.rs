#![allow(clippy::needless_range_loop)]

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

/// Signature of a scirust per-codec quantizer.
type QuantizeFn = fn(
    &[f32; scirust::D_C],
) -> (
    [u8; scirust::LATENT_BYTES],
    f32,
    [u8; scirust::attention::slha_v2::N_GROUPS],
);

/// Build a real codec tile through scirust's quantizer, then verify the vram
/// serialized decode matches the scirust scalar score exactly.
fn codec_tile_parity(quantize: QuantizeFn, codec_flag: u16, warm: bool, nocorr: bool) {
    use scirust::attention::slha_v2::{FLAG_TQ3_NOCORR, FLAG_WARM, LATENT_BYTES};
    // A steep, GPT-2-like latent spectrum — the motivating input for the
    // mixed/TQ3/MIX3 codecs.
    let mut v = [0.0f32; scirust::D_C];
    let mut rng = scirust::rng::Rng::new(0xC0DEC);
    for (d, x) in v.iter_mut().enumerate() {
        let amp = 37.0 * ((d + 1) as f32).powf(-0.9);
        *x = amp * rng.next_gaussian();
    }

    let (packed, global, gs) = quantize(&v);
    let mut flags = codec_flag;
    if warm {
        flags |= FLAG_WARM;
    }
    if nocorr {
        flags |= FLAG_TQ3_NOCORR;
    }

    // Reference: scirust scalar score on the same tile.
    let residual = [0xDEAD_BEEF_CAFE_FACEu64, 0x1234, 0xCAFE, 0xFACE];
    let mut ref_t = SciRustSlhaTile {
        latent_kv: packed,
        residual_bitmap: residual,
        scale: global,
        dynamic_lambda: 0.37,
        residual_sigma: 0.0,
        token_id: 0,
        position: 0,
        head_id: 0,
        flags,
        group_scales: gs,
    };
    if nocorr {
        // CCOS masks the plane when paging it out; emulate that too.
        let n = ref_t.separable_corr_bytes();
        ref_t.latent_kv[LATENT_BYTES - n..].fill(0);
    }
    let q = q_coarse_all(0.01);
    let qs = q_sign_all(0xDEAD_BEEF_CAFE_FACE);
    let expected = ref_t.compute_score_scalar(&q, &qs);

    // Serialized round-trip.
    let mut stile = SerializedTile::zeroed();
    stile.latent_mut().copy_from_slice(&ref_t.latent_kv);
    stile.set_residual(&residual);
    stile.set_scale(ref_t.scale);
    stile.set_dynamic_lambda(ref_t.dynamic_lambda);
    stile.set_group_scales(&ref_t.group_scales);
    stile.set_flags(flags);
    let got = stile.score(&q, &qs);

    let diff = (got - expected).abs();
    assert!(
        diff < 1e-4,
        "codec {:#06x} warm={warm} nocorr={nocorr} parity: {got} vs {expected}, diff {diff}",
        codec_flag
    );
}

#[test]
fn test_mixed_codec_hot_parity() {
    codec_tile_parity(
        scirust::attention::slha_v2::quantize_latent_mixed,
        codec::FLAG_MIXED,
        false,
        false,
    );
}

#[test]
fn test_mixed_codec_warm_parity() {
    codec_tile_parity(
        scirust::attention::slha_v2::quantize_latent_mixed,
        codec::FLAG_MIXED,
        true,
        false,
    );
}

#[test]
fn test_tq3_codec_hot_parity() {
    codec_tile_parity(
        scirust::attention::slha_v2::quantize_latent_tq3,
        codec::FLAG_TQ3,
        false,
        false,
    );
}

#[test]
fn test_tq3_codec_nocorr_parity() {
    codec_tile_parity(
        scirust::attention::slha_v2::quantize_latent_tq3,
        codec::FLAG_TQ3,
        false,
        true,
    );
}

#[test]
fn test_mix3_codec_hot_parity() {
    codec_tile_parity(
        scirust::attention::slha_v2::quantize_latent_mix3,
        codec::FLAG_MIX3,
        false,
        false,
    );
}

#[test]
fn test_mix3_codec_nocorr_parity() {
    codec_tile_parity(
        scirust::attention::slha_v2::quantize_latent_mix3,
        codec::FLAG_MIX3,
        false,
        true,
    );
}

#[test]
fn test_unknown_codec_combination_is_rejected_not_misdecoded() {
    // Two mutually-exclusive codec flags set together must be rejected, not
    // silently decoded (the old behaviour would have fallen through to INT4
    // and produced a wrong score).
    let stile = {
        let mut t = SerializedTile::zeroed();
        t.latent_mut().copy_from_slice(&fill_latent_seq());
        t.set_scale(1.0);
        t.set_group_scales(&[128u8; 8]);
        t.set_flags(codec::FLAG_NF4 | codec::FLAG_MIXED);
        t
    };
    let q = q_coarse_all(1.0);
    let qs = q_sign_all(0);
    assert!(
        stile.try_score(&q, &qs).is_err(),
        "NF4|MIXED must be rejected"
    );

    // NOCORR without a TQ3/MIX3 codec is also invalid.
    let mut t2 = SerializedTile::zeroed();
    t2.set_flags(codec::FLAG_TQ3_NOCORR);
    assert!(codec::validate_codec(t2.flags()).is_err());
}

#[test]
fn test_pipeline_copy_glue_cpu_engine() {
    // The pipeline glue (copy_tiles_to_gpu / copy_scores_from_gpu) must round-
    // trip through a CPU engine and preserve scores exactly.
    use slhav2_vram::pipeline::{copy_scores_from_gpu, copy_tiles_to_gpu};

    let engine = CpuEngine::new();
    let mut tiles_dev = engine.allocate(16 * codec::TILE_BYTES).unwrap();

    let mut tiles: Vec<SerializedTile> = (0..4)
        .map(|i| {
            let mut t = SerializedTile::zeroed();
            t.latent_mut()
                .copy_from_slice(&fill_latent(0x80 | (i as u8)));
            t.set_scale(1.0 + i as f32 * 0.25);
            t.set_group_scales(&[200u8; 8]);
            t
        })
        .collect();
    tiles[1].set_flags(codec::FLAG_WARM);

    copy_tiles_to_gpu(&engine, &tiles, &mut tiles_dev, 0).unwrap();

    let q = q_coarse_all(0.01);
    let qs = q_sign_all(0);

    // Score via the CPU backend on the device allocation.
    let mut scores = vec![0.0f32; tiles.len()];
    {
        let mut dev_tiles: Vec<SerializedTile> = tiles_dev
            .data()
            .as_chunks::<{ codec::TILE_BYTES }>()
            .0
            .iter()
            .map(SerializedTile::from_bytes)
            .collect();
        dev_tiles.truncate(tiles.len());
        score_tiles_cpu(ScoringInput {
            engine: &engine,
            tiles: &dev_tiles,
            q_coarse: &q,
            q_sign: &qs,
            scores: &mut scores,
        });
    }

    // copy_scores_from_gpu returns the same values.
    let mut scores_dev = engine.allocate(tiles.len() * 4).unwrap();
    let mut bytes = vec![0u8; tiles.len() * 4];
    for (i, s) in scores.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
    }
    engine.copy_to_device(&bytes, &mut scores_dev, 0).unwrap();
    let round = copy_scores_from_gpu(&engine, &scores_dev, 0, tiles.len()).unwrap();
    for (a, b) in scores.iter().zip(round.iter()) {
        assert!((a - b).abs() < 1e-6, "score copy round-trip: {a} vs {b}");
    }
}

#[test]
fn test_pipeline_copy_overflow_rejected() {
    // The checked byte-count arithmetic must reject an absurd tile count
    // rather than silently overflowing (guarded by `expect`, which panics with
    // a clear message).
    use slhav2_vram::pipeline::{copy_scores_from_gpu, copy_tiles_to_gpu};

    let engine = CpuEngine::new();
    let mut tiles_dev = engine.allocate(128).unwrap();
    let huge: Vec<SerializedTile> = Vec::new();
    // Empty slice: no overflow, no-op success.
    copy_tiles_to_gpu(&engine, &huge, &mut tiles_dev, 0).unwrap();

    // An absurd score count must panic (overflow) rather than allocate.
    let scores_dev = engine.allocate(4).unwrap();
    let result = std::panic::catch_unwind(|| {
        copy_scores_from_gpu(&engine, &scores_dev, 0, usize::MAX / 2 + 1)
    });
    assert!(result.is_err(), "overflowing score count must panic");
}
