//! Randomized **property / fuzz** tests — dependency-free, driven by the crate's
//! own deterministic RNG (so failures are reproducible). They assert invariants
//! over thousands of random inputs rather than a single hand-picked case:
//!
//! - SIMD paths (AVX2 / AVX-512) ≡ scalar on random INT4 tiles;
//! - `compute_score` never produces NaN/inf on bounded-random INT4 **and** NF4
//!   tiles (robustness fuzz);
//! - WARM == coarse-only;
//! - softmax is a probability distribution;
//! - INT4(MX) and NF4 dequantisation stay within their per-group error bound.

use scirust::attention::slha_v2::{
    quantize_latent_grouped, quantize_latent_nf4, SciRustSlhaTile, D_C, D_S, FLAG_NF4, FLAG_WARM,
    GROUP_DIM, LATENT_BYTES, NF4_CODEBOOK, N_GROUPS, RESIDUAL_WORDS,
};
use scirust::metrics::softmax_into;
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection, D_K};

fn rand_q(rng: &mut Rng) -> [f32; D_C] {
    let mut q = [0.0f32; D_C];
    rng.fill_gaussian(&mut q);
    q
}

fn rand_q_sign(rng: &mut Rng) -> [u64; RESIDUAL_WORDS] {
    let mut s = [0u64; RESIDUAL_WORDS];
    for w in s.iter_mut() {
        *w = rng.next_u64();
    }
    s
}

/// A random *uniform-INT4* tile with random per-group scales; HOT or WARM.
fn rand_int4_tile(rng: &mut Rng, warm: bool) -> SciRustSlhaTile {
    let mut latent_kv = [0u8; LATENT_BYTES];
    for b in latent_kv.iter_mut() {
        *b = rng.next_u64() as u8;
    }
    let mut group_scales = [0u8; N_GROUPS];
    for g in group_scales.iter_mut() {
        *g = (rng.next_u64() as u8) | 1; // 1..=255, never a zero scale
    }
    SciRustSlhaTile {
        latent_kv,
        residual_bitmap: rand_q_sign(rng),
        scale: rng.next_unit() * 2.0 + 0.01,
        dynamic_lambda: rng.next_gaussian() * 0.1,
        residual_sigma: rng.next_unit(),
        token_id: 0,
        position: 0,
        head_id: 0,
        flags: if warm { FLAG_WARM } else { 0 },
        group_scales,
    }
}

#[test]
fn prop_simd_paths_equal_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = std::is_x86_feature_detected!("avx2");
        let has_avx512 = std::is_x86_feature_detected!("avx512f");
        if !has_avx2 && !has_avx512 {
            return;
        }
        let mut rng = Rng::new(0xF00D);
        for it in 0..1000 {
            let tile = rand_int4_tile(&mut rng, it % 3 == 0);
            let q = rand_q(&mut rng);
            let qs = rand_q_sign(&mut rng);
            let s = tile.compute_score_scalar(&q, &qs);
            let tol = 1e-3 * (1.0 + s.abs());
            if has_avx2 {
                let a = unsafe { tile.compute_score_avx2(&q, &qs) };
                assert!((s - a).abs() <= tol, "avx2 iter {it}: {s} vs {a}");
            }
            if has_avx512 {
                let a = unsafe { tile.compute_score_avx512(&q, &qs) };
                assert!((s - a).abs() <= tol, "avx512 iter {it}: {s} vs {a}");
            }
        }
    }
}

#[test]
fn prop_score_is_always_finite() {
    // Robustness fuzz over INT4 and NF4 tiles: a valid tile never yields NaN/inf.
    let mut rng = Rng::new(0xBEEF);
    for it in 0..3000 {
        let mut tile = rand_int4_tile(&mut rng, it % 5 == 0);
        if it % 2 == 0 {
            tile.flags |= FLAG_NF4;
        }
        let s = tile.compute_score(&rand_q(&mut rng), &rand_q_sign(&mut rng));
        assert!(s.is_finite(), "non-finite score at iter {it}: {s}");
    }
}

#[test]
fn prop_warm_equals_coarse_only() {
    let mut rng = Rng::new(7);
    for _ in 0..500 {
        let hot = rand_int4_tile(&mut rng, false);
        let q = rand_q(&mut rng);
        let qs = rand_q_sign(&mut rng);
        let mut warm = hot;
        warm.flags |= FLAG_WARM;
        let mut hot_lambda0 = hot;
        hot_lambda0.dynamic_lambda = 0.0;
        let sw = warm.compute_score(&q, &qs);
        let s0 = hot_lambda0.compute_score(&q, &qs);
        assert!((sw - s0).abs() <= 1e-4 * (1.0 + sw.abs()), "{sw} vs {s0}");
    }
}

#[test]
fn prop_softmax_is_a_distribution() {
    let mut rng = Rng::new(11);
    for _ in 0..400 {
        let n = 1 + (rng.next_u64() as usize % 64);
        let mut s = vec![0.0f32; n];
        for x in s.iter_mut() {
            *x = rng.next_gaussian() * (rng.next_unit() * 20.0); // varied magnitudes
        }
        let mut w = vec![0.0f32; n];
        softmax_into(&s, rng.next_unit() * 2.0, &mut w);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax sum = {sum}");
        assert!(
            w.iter().all(|&p| (0.0..=1.0).contains(&p)),
            "softmax weight out of [0,1]"
        );
    }
}

#[test]
fn prop_dequant_error_within_bound() {
    let mut rng = Rng::new(99);
    for _ in 0..300 {
        let mut v = [0.0f32; D_C];
        for x in v.iter_mut() {
            *x = rng.next_gaussian() * (rng.next_unit() * 5.0 + 0.1);
        }

        // Uniform INT4 (MX): error ≤ one quantisation step (the per-group scale).
        let (latent_kv, scale, group_scales) = quantize_latent_grouped(&v);
        let t = SciRustSlhaTile {
            latent_kv,
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: 0,
            group_scales,
        };
        let dq = t.dequant_latent();
        for d in 0..D_C {
            let eff = scale * (group_scales[d / GROUP_DIM] as f32 / 255.0);
            assert!((dq[d] - v[d]).abs() <= eff + 1e-5, "int4 dim {d}");
        }

        // NF4: error ≤ ~half the widest codebook gap (~0.15) times the group absmax.
        let (lk, sc, gs) = quantize_latent_nf4(&v);
        let mut t2 = t;
        t2.latent_kv = lk;
        t2.scale = sc;
        t2.group_scales = gs;
        t2.flags |= FLAG_NF4;
        let dq2 = t2.dequant_latent();
        for d in 0..D_C {
            let eff = sc * (gs[d / GROUP_DIM] as f32 / 255.0);
            assert!((dq2[d] - v[d]).abs() <= 0.16 * eff + 1e-5, "nf4 dim {d}");
        }
    }
}

// --- Round 2: extra invariants -------------------------------------------------

#[test]
fn prop_rng_and_encode_are_deterministic() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..256 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    // Same key + projection -> byte-identical tile (reproducible encoding).
    let proj = Projection::new(1);
    let (_q, toks) = generate(5, 4, 0.4);
    let t1 = build_tile(&proj, &toks[0], 0, false);
    let t2 = build_tile(&proj, &toks[0], 0, false);
    assert_eq!(t1.latent_kv, t2.latent_kv);
    assert_eq!(t1.residual_bitmap, t2.residual_bitmap);
    assert_eq!(t1.group_scales, t2.group_scales);
    assert_eq!(t1.scale.to_bits(), t2.scale.to_bits());
    assert_eq!(t1.dynamic_lambda.to_bits(), t2.dynamic_lambda.to_bits());
}

#[test]
fn prop_sign_bits_negation_flips_all_bits() {
    // sign(Z·(-v)) is the bitwise complement of sign(Z·v): all D_S signs flip.
    let proj = Projection::new(2);
    let mut rng = Rng::new(3);
    for _ in 0..200 {
        let mut v = [0.0f32; D_K];
        rng.fill_gaussian(&mut v);
        let mut nv = v;
        for x in nv.iter_mut() {
            *x = -*x;
        }
        let a = proj.sign_bits(&v);
        let b = proj.sign_bits(&nv);
        let ham: u32 = (0..RESIDUAL_WORDS)
            .map(|w| (a[w] ^ b[w]).count_ones())
            .sum();
        assert_eq!(ham, D_S as u32, "negation should flip every sign bit");
    }
}

#[test]
fn prop_residual_term_bounded_by_lambda_ds() {
    // HOT - WARM = λ·(d_s - 2·popcount) ∈ [-|λ|·d_s, |λ|·d_s].
    let mut rng = Rng::new(4);
    for _ in 0..500 {
        let tile = rand_int4_tile(&mut rng, false);
        let q = rand_q(&mut rng);
        let qs = rand_q_sign(&mut rng);
        let mut warm = tile;
        warm.flags |= FLAG_WARM;
        let residual = tile.compute_score(&q, &qs) - warm.compute_score(&q, &qs);
        let bound = tile.dynamic_lambda.abs() * D_S as f32;
        assert!(residual.abs() <= bound + 1e-2, "{residual} exceeds {bound}");
    }
}

#[test]
fn prop_nf4_codebook_is_well_formed() {
    assert_eq!(NF4_CODEBOOK.len(), 16);
    assert_eq!(NF4_CODEBOOK[0], -1.0);
    assert_eq!(NF4_CODEBOOK[15], 1.0);
    for i in 1..16 {
        assert!(
            NF4_CODEBOOK[i] > NF4_CODEBOOK[i - 1],
            "not ascending at {i}"
        );
    }
    for i in 0..16 {
        assert!(
            (NF4_CODEBOOK[i] + NF4_CODEBOOK[15 - i]).abs() < 1e-6,
            "not symmetric at {i}"
        );
    }
}

#[test]
fn prop_degenerate_tiles_stay_finite() {
    // Zero latent / zero scale / extreme λ / all-ones query sign must stay finite
    // (no panic, no NaN/inf) for both INT4 and NF4.
    let mut base = rand_int4_tile(&mut Rng::new(1), false);
    base.latent_kv = [0u8; LATENT_BYTES];
    base.scale = 0.0;
    base.dynamic_lambda = 1.0e6;
    base.group_scales = [1u8; N_GROUPS];
    let q = [1.0e3f32; D_C];
    let qs = [u64::MAX; RESIDUAL_WORDS];
    assert!(base.compute_score(&q, &qs).is_finite());
    let mut nf4 = base;
    nf4.flags |= FLAG_NF4;
    assert!(nf4.compute_score(&q, &qs).is_finite());
}
