//! Integration tests: these *prove* the core SLHA v2 claims, they don't just
//! exercise the code.

use scirust::attention::slha_v2::{D_S, FLAG_WARM, RESIDUAL_WORDS};
use scirust::metrics::{dot, spearman};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection, D_K};

#[inline]
fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Eq. (2.3) binary core: `d_s - 2·popcount(a ^ b)` must equal the signed dot
/// product of the two ±1 sign vectors. Verified against brute force.
#[test]
fn hamming_identity_matches_sign_dot() {
    let mut rng = Rng::new(7);
    for _ in 0..2000 {
        let mut a = [0u64; RESIDUAL_WORDS];
        let mut b = [0u64; RESIDUAL_WORDS];
        for w in 0..RESIDUAL_WORDS {
            a[w] = rng.next_u64();
            b[w] = rng.next_u64();
        }
        let hamming: u32 = (0..RESIDUAL_WORDS)
            .map(|w| (a[w] ^ b[w]).count_ones())
            .sum();
        let formula = D_S as i32 - 2 * hamming as i32;

        let mut brute = 0i32;
        for s in 0..D_S {
            let sa = if (a[s >> 6] >> (s & 63)) & 1 == 1 {
                -1
            } else {
                1
            };
            let sb = if (b[s >> 6] >> (s & 63)) & 1 == 1 {
                -1
            } else {
                1
            };
            brute += sa * sb;
        }
        assert_eq!(formula, brute);
    }
}

/// The fused score must equal the hand-computed eq. (2.3).
#[test]
fn score_matches_equation_2_3() {
    let proj = Projection::new(3);
    let (q, toks) = generate(4, 1, 0.4);
    let tile = build_tile(&proj, &toks[0], 0, false);
    let q_sign = proj.sign_bits(&q);

    let got = tile.compute_score(&q, &q_sign);

    let k = tile.dequant_latent();
    let coarse = dot(&q, &k);
    let hamming: u32 = (0..RESIDUAL_WORDS)
        .map(|w| (q_sign[w] ^ tile.residual_bitmap[w]).count_ones())
        .sum();
    let manual = coarse + tile.dynamic_lambda * (D_S as f32 - 2.0 * hamming as f32);

    assert!((got - manual).abs() < 1e-4, "{got} vs {manual}");
}

/// CCOS WARM mode (spec §4): freeing the residual must reduce the score to the
/// latent-base-only term, i.e. exactly HOT with λ = 0.
#[test]
fn warm_mode_drops_the_binary_term() {
    let proj = Projection::new(2);
    let (q, toks) = generate(2, 1, 0.5);
    let q_sign = proj.sign_bits(&q);

    let hot = build_tile(&proj, &toks[0], 0, false);
    let mut warm = hot;
    warm.flags |= FLAG_WARM;
    let mut hot_lambda0 = hot;
    hot_lambda0.dynamic_lambda = 0.0;

    let s_warm = warm.compute_score(&q, &q_sign);
    let s_hot0 = hot_lambda0.compute_score(&q, &q_sign);
    let s_hot = hot.compute_score(&q, &q_sign);

    assert!((s_warm - s_hot0).abs() < 1e-5, "WARM != coarse-only");
    assert!((s_hot - s_warm).abs() > 0.0, "residual contributed nothing");
}

/// The 1-bit sign-LSH residual must track the *direction* (cosine) of the true
/// inner product it approximates. This is the empirical justification for the
/// whole binary-correction idea.
#[test]
fn sign_lsh_residual_tracks_cosine() {
    let proj = Projection::new(11);
    let mut rng = Rng::new(99);
    let n = 400;

    let mut q = [0.0f32; D_K];
    rng.fill_gaussian(&mut q);
    let nq = norm(&q);
    for x in q.iter_mut() {
        *x /= nq; // unit query
    }
    let q_sign = proj.sign_bits(&q);

    // Probe with e's spanning the full cosine range (controlled alignment with
    // q) — the proper way to evaluate a sign-LSH cosine estimator. (The harder
    // near-orthogonal regime is reported separately by the `measure` example.)
    let mut est = Vec::with_capacity(n);
    let mut cos = Vec::with_capacity(n);
    for _ in 0..n {
        let mut e = [0.0f32; D_K];
        rng.fill_gaussian(&mut e);
        let align = (rng.next_unit() * 2.0 - 1.0) * (D_K as f32).sqrt();
        for i in 0..D_K {
            e[i] += align * q[i];
        }
        let b = proj.sign_bits(&e);
        let hamming: u32 = (0..RESIDUAL_WORDS)
            .map(|w| (q_sign[w] ^ b[w]).count_ones())
            .sum();
        est.push(D_S as f32 - 2.0 * hamming as f32);
        cos.push(dot(&q, &e) / norm(&e));
    }

    let sp = spearman(&est, &cos);
    assert!(sp > 0.85, "Spearman(residual, cosine) = {sp} — too low");
}

/// End-to-end: with a meaningful residual energy, the HOT score (latent + 1-bit
/// residual) must rank context tokens at least as well as WARM (latent only).
#[test]
fn hot_ranks_at_least_as_well_as_warm() {
    let proj = Projection::new(5);
    let (q, toks) = generate(123, 256, 0.4);
    let q_sign = proj.sign_bits(&q);

    let mut s_true = Vec::with_capacity(toks.len());
    let mut s_hot = Vec::with_capacity(toks.len());
    let mut s_warm = Vec::with_capacity(toks.len());
    for (i, t) in toks.iter().enumerate() {
        s_true.push(dot(&q, &t.k_real));
        let hot = build_tile(&proj, t, i as u32, false);
        let mut warm = hot;
        warm.flags |= FLAG_WARM;
        s_hot.push(hot.compute_score(&q, &q_sign));
        s_warm.push(warm.compute_score(&q, &q_sign));
    }

    let sp_hot = spearman(&s_hot, &s_true);
    let sp_warm = spearman(&s_warm, &s_true);

    assert!(sp_hot > 0.8, "HOT Spearman {sp_hot} unexpectedly low");
    // HOT uses strictly more information than WARM; allow a tiny tolerance.
    assert!(
        sp_hot + 0.02 >= sp_warm,
        "HOT ({sp_hot}) ranked worse than WARM ({sp_warm})"
    );
}
