//! Unit tests for the SLHA v2 micro-kernel: layout, codec round-trips and the
//! per-ISA SIMD ≡ scalar equivalence matrix.

use super::*;
use std::mem::{align_of, size_of};

#[test]
fn tile_is_exactly_128_bytes_zero_padding() {
    // Default align(64) (= two 64-byte lines) on every 64-byte-line part,
    // including all our targets (x86-64 and AArch64/Neoverse — the Thor
    // measures 64 B). build.rs bumps to align(128) only on a native
    // 128-byte-line host (cfg cache_line_128). Size is 128 B with no padding
    // either way.
    let expected_align = if cfg!(cache_line_128) { 128 } else { 64 };
    assert_eq!(align_of::<SciRustSlhaTile>(), expected_align);
    assert_eq!(size_of::<SciRustSlhaTile>(), 128);

    // Sum of field sizes == struct size  =>  no padding anywhere.
    let field_bytes = LATENT_BYTES            // latent_kv
        + RESIDUAL_WORDS * 8                  // residual_bitmap
        + 4 + 4 + 4                           // scale, dynamic_lambda, residual_sigma
        + 4 + 4                               // token_id, position
        + 2 + 2                               // head_id, flags
        + 8; // group_scales
    assert_eq!(field_bytes, 128);
    assert_eq!(field_bytes, size_of::<SciRustSlhaTile>());
}

fn tile_from(
    latent_kv: [u8; LATENT_BYTES],
    scale: f32,
    group_scales: [u8; N_GROUPS],
) -> SciRustSlhaTile {
    SciRustSlhaTile {
        latent_kv,
        residual_bitmap: [0; RESIDUAL_WORDS],
        scale,
        dynamic_lambda: 0.0,
        residual_sigma: 0.0,
        token_id: 0,
        position: 0,
        head_id: 0,
        flags: FLAG_HOT,
        group_scales,
    }
}

#[test]
fn int4_dequant_round_trips_signed_values() {
    // A vector with both signs must survive quantise -> dequantise within
    // one quantisation step, and crucially keep negative values negative.
    let mut v = [0.0f32; D_C];
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i as f32) - 64.0) / 16.0; // spans negative and positive
    }
    let (packed, scale) = quantize_latent(&v);
    // [255; N_GROUPS] makes every group's effective scale == the global scale,
    // i.e. exactly the single-scale behaviour.
    let tile = tile_from(packed, scale, [255; N_GROUPS]);
    let dq = tile.dequant_latent();
    // At least one strictly-negative reconstructed value (zero-point works).
    assert!(
        dq.iter().any(|&x| x < 0.0),
        "no negative values reconstructed"
    );
    for d in 0..D_C {
        assert!(
            (dq[d] - v[d]).abs() <= scale + 1e-6,
            "dim {d}: |{} - {}| > step {scale}",
            dq[d],
            v[d]
        );
    }
}

#[test]
fn grouped_int4_beats_single_on_spread_variance() {
    // Per-group magnitudes spanning orders of magnitude (like PCA components
    // ordered by eigenvalue): a single global scale crushes the small
    // groups, per-group scaling does not.
    let mut v = [0.0f32; D_C];
    let mut rng = crate::rng::Rng::new(5);
    for g in 0..N_GROUPS {
        let amp = 10f32.powi(-(g as i32)); // 1, 0.1, 0.01, ...
        for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
            v[d] = amp * rng.next_gaussian();
        }
    }
    let sq_err = |t: &SciRustSlhaTile| -> f32 {
        let dq = t.dequant_latent();
        (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
    };
    let (p1, s1) = quantize_latent(&v);
    let e_single = sq_err(&tile_from(p1, s1, [255; N_GROUPS]));
    let (p2, s2, gs2) = quantize_latent_grouped(&v);
    let e_grouped = sq_err(&tile_from(p2, s2, gs2));
    assert!(
        e_grouped < e_single * 0.5,
        "grouped err {e_grouped} not clearly < single err {e_single}"
    );
}

#[test]
fn nf4_beats_uniform_int4_on_gaussian_latent() {
    // NF4's normal-quantile codebook should reconstruct Gaussian latent
    // values more accurately than uniform INT4 at the same 4-bit budget.
    let mut v = [0.0f32; D_C];
    let mut rng = crate::rng::Rng::new(8);
    for x in v.iter_mut() {
        *x = rng.next_gaussian();
    }
    let sq_err = |t: &SciRustSlhaTile| -> f32 {
        let dq = t.dequant_latent();
        (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
    };

    let (p1, s1, g1) = quantize_latent_grouped(&v);
    let e_uniform = sq_err(&tile_from(p1, s1, g1)); // FLAG_HOT -> uniform

    let (p2, s2, g2) = quantize_latent_nf4(&v);
    let mut nf4 = tile_from(p2, s2, g2);
    nf4.flags |= FLAG_NF4;
    let e_nf4 = sq_err(&nf4);

    assert!(
        e_nf4 < e_uniform,
        "NF4 err {e_nf4} not < uniform {e_uniform}"
    );
}

/// A steep, GPT-2-like latent spectrum (the measured motivation for the
/// codec): per-dim std ~ 37·(d+1)^-0.9, i.e. λ0/λ63 ≈ 40× in std.
fn steep_latent(seed: u64) -> [f32; D_C] {
    let mut rng = crate::rng::Rng::new(seed);
    let mut v = [0.0f32; D_C];
    for (d, x) in v.iter_mut().enumerate() {
        let amp = 37.0 * ((d + 1) as f32).powf(-0.9);
        *x = amp * rng.next_gaussian();
    }
    v
}

#[test]
fn mixed_dequant_roundtrips_and_drops_tail() {
    let v = steep_latent(11);
    let (packed, global, gs) = quantize_latent_mixed(&v);
    let mut tile = tile_from(packed, global, gs);
    tile.flags |= FLAG_MIXED;
    let dq = tile.dequant_latent();
    // 8-bit head: error within one 8-bit step.
    let eff_hi = global * (gs[0] as f32 / 255.0);
    for d in 0..MIXED_HI_DIMS {
        assert!(
            (dq[d] - v[d]).abs() <= eff_hi + 1e-6,
            "hi dim {d}: |{} - {}| > step {eff_hi}",
            dq[d],
            v[d]
        );
    }
    // 4-bit body: error within one step of its group.
    for d in MIXED_HI_DIMS..MIXED_DIMS {
        let g = 1 + (d - MIXED_HI_DIMS) / GROUP_DIM;
        let eff = global * (gs[g] as f32 / 255.0);
        assert!(
            (dq[d] - v[d]).abs() <= eff + 1e-6,
            "lo dim {d}: |{} - {}| > step {eff}",
            dq[d],
            v[d]
        );
    }
    // Dropped tail decodes to exactly 0.
    for d in MIXED_DIMS..D_C {
        assert_eq!(dq[d], 0.0, "tail dim {d} must decode to 0");
    }
}

#[test]
fn mixed_beats_uniform_int4_on_steep_spectrum() {
    // On the steep spectrum the codec was built for, the 8-bit head must
    // cut the reconstruction error decisively — including the price of the
    // dropped tail (which carries ~no energy at this decay).
    let sq_err = |t: &SciRustSlhaTile, v: &[f32; D_C]| -> f32 {
        let dq = t.dequant_latent();
        (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
    };
    let (mut worse, mut total) = (0, 0);
    for seed in 0..8u64 {
        let v = steep_latent(100 + seed);
        let (p1, s1, g1) = quantize_latent_grouped(&v);
        let e_uniform = sq_err(&tile_from(p1, s1, g1), &v);
        let (p2, s2, g2) = quantize_latent_mixed(&v);
        let mut mixed = tile_from(p2, s2, g2);
        mixed.flags |= FLAG_MIXED;
        let e_mixed = sq_err(&mixed, &v);
        total += 1;
        if e_mixed >= e_uniform * 0.5 {
            worse += 1;
        }
    }
    assert_eq!(
        worse, 0,
        "mixed did not halve the uniform error on {worse}/{total} steep latents"
    );
}

#[test]
fn tq3_dequant_roundtrips_within_quarter_step() {
    // The same 16 values tiled across every group ⇒ identical per-group
    // absmax ⇒ every gs byte is 255 and the effective step is exactly the
    // global scale: the strict quarter-step bound of the corrected grid
    // must hold on every dim.
    let mut pattern = [0.0f32; GROUP_DIM];
    let mut rng = crate::rng::Rng::new(17);
    for x in pattern.iter_mut() {
        *x = rng.next_gaussian();
    }
    let mut v = [0.0f32; D_C];
    for (d, x) in v.iter_mut().enumerate() {
        *x = pattern[d % GROUP_DIM];
    }
    let (packed, global, gs) = quantize_latent_tq3(&v);
    assert_eq!(gs, [255u8; N_GROUPS], "equal-spread groups must share gs");
    let mut tile = tile_from(packed, global, gs);
    tile.flags |= FLAG_TQ3;
    let dq = tile.dequant_latent();
    for d in 0..D_C {
        assert!(
            (dq[d] - v[d]).abs() <= TQ3_CORRECTION * global + 1e-5,
            "dim {d}: |{} - {}| > quarter step {}",
            dq[d],
            v[d],
            TQ3_CORRECTION * global
        );
    }
}

#[test]
fn tq3_roundtrips_on_steep_spectrum() {
    // On the realistic steep spectrum, per-group scaling plus the u8
    // scale rounding keeps every dim within one grid step of the truth
    // (same bound convention as the grouped INT4 test).
    let v = steep_latent(19);
    let (packed, global, gs) = quantize_latent_tq3(&v);
    let mut tile = tile_from(packed, global, gs);
    tile.flags |= FLAG_TQ3;
    let dq = tile.dequant_latent();
    for d in 0..D_C {
        let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
        assert!(
            (dq[d] - v[d]).abs() <= eff + 1e-6,
            "dim {d}: |{} - {}| > step {eff}",
            dq[d],
            v[d]
        );
    }
}

#[test]
fn tq3_positive_values_do_not_collapse() {
    // Regression guard for the upstream TurboQuant defect this port
    // fixes: a mis-sized grid (15 levels clamped to 8) decoded every
    // positive input to 0. Both signs must survive the round trip.
    let mut v = [0.0f32; D_C];
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i as f32) - 64.0) / 16.0; // spans negative and positive
    }
    let (packed, global, gs) = quantize_latent_tq3(&v);
    let mut tile = tile_from(packed, global, gs);
    tile.flags |= FLAG_TQ3;
    let dq = tile.dequant_latent();
    let mut pos = 0;
    for d in 0..D_C {
        let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
        // Any value at least one step from zero must keep its sign.
        if v[d] >= eff {
            assert!(dq[d] > 0.0, "dim {d}: positive {} decoded {}", v[d], dq[d]);
            pos += 1;
        }
        if v[d] <= -eff {
            assert!(dq[d] < 0.0, "dim {d}: negative {} decoded {}", v[d], dq[d]);
        }
    }
    assert!(pos > 0, "test vector must exercise the positive half");
}

#[test]
fn tq3_error_matches_int4_grouped_resolution() {
    // 3-bit + 1-bit correction spends the same 4 bits/dim as INT4, and
    // both grids share the same worst-case error (0.25 step). On a
    // Gaussian latent TQ3 pays a measured ~1.3–1.6× MSE penalty for its
    // missing zero level (near-zero mass always costs ≥ 0.25 step) — the
    // documented trade for the separable correction plane. Guard that
    // the penalty stays in that class and never explodes.
    let sq_err = |t: &SciRustSlhaTile, v: &[f32; D_C]| -> f32 {
        let dq = t.dequant_latent();
        (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
    };
    for seed in 0..8u64 {
        let mut v = [0.0f32; D_C];
        let mut rng = crate::rng::Rng::new(200 + seed);
        for x in v.iter_mut() {
            *x = rng.next_gaussian();
        }
        let (p1, s1, g1) = quantize_latent_grouped(&v);
        let e_int4 = sq_err(&tile_from(p1, s1, g1), &v);
        let (p2, s2, g2) = quantize_latent_tq3(&v);
        let mut tq3 = tile_from(p2, s2, g2);
        tq3.flags |= FLAG_TQ3;
        let e_tq3 = sq_err(&tq3, &v);
        assert!(
            e_tq3 <= e_int4 * 2.0,
            "seed {seed}: TQ3 err {e_tq3} way above INT4 err {e_int4}"
        );
    }
}

#[test]
fn tq3_nocorr_decodes_the_bare_grid() {
    // With FLAG_TQ3_NOCORR the decoder must ignore the correction plane
    // (even a zeroed one, which would otherwise bias every dim by
    // −TQ3_CORRECTION): the error bound relaxes to half a step, and the
    // corrected tile must reconstruct at least as well on aggregate.
    let v = steep_latent(29);
    let (packed, global, gs) = quantize_latent_tq3(&v);
    let mut with_corr = tile_from(packed, global, gs);
    with_corr.flags |= FLAG_TQ3;
    let mut nocorr = with_corr;
    nocorr.flags |= FLAG_TQ3_NOCORR;
    // CCOS masks the plane when paging it out; emulate that too.
    for b in nocorr.latent_kv[TQ3_CODE_BYTES..].iter_mut() {
        *b = 0;
    }
    let (mut e_corr, mut e_bare) = (0.0f32, 0.0f32);
    for d in 0..D_C {
        let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
        let err = (nocorr.dequant_at(d) - v[d]).abs();
        assert!(
            err <= 0.5 * eff + 1e-5,
            "dim {d}: bare-grid error {err} > half step {}",
            0.5 * eff
        );
        e_corr += (with_corr.dequant_at(d) - v[d]).powi(2);
        e_bare += (nocorr.dequant_at(d) - v[d]).powi(2);
    }
    assert!(
        e_corr < e_bare,
        "correction must help on aggregate: {e_corr} !< {e_bare}"
    );
}

/// TQ3 equivalence fixture: a query, its sign bits, and TQ3 tiles built
/// via `quantize_latent_tq3` from both data shapes (Gaussian scenario
/// tokens and the steep GPT-2-like spectrum), sweeping all four
/// HOT/WARM × corr/nocorr flag combinations. Shared by the dispatch and
/// per-ISA score-equivalence tests.
fn tq3_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
    use crate::scenario::{generate, Projection};
    let proj = Projection::new(9);
    let (q, toks) = generate(123, 32, 0.4);
    let q_sign = proj.sign_bits(&q);
    let mut tiles = Vec::new();
    let mut push = |v: &[f32; D_C], bitmap: [u64; RESIDUAL_WORDS], i: usize| {
        let (packed, global, gs) = quantize_latent_tq3(v);
        let mut tile = tile_from(packed, global, gs);
        tile.residual_bitmap = bitmap;
        tile.dynamic_lambda = 0.37;
        tile.position = i as u32;
        tile.flags = FLAG_TQ3
            | if i.is_multiple_of(2) { FLAG_WARM } else { 0 }
            | if (i / 2).is_multiple_of(2) {
                FLAG_TQ3_NOCORR
            } else {
                0
            };
        tiles.push(tile);
    };
    for (i, t) in toks.iter().enumerate() {
        push(&t.k_coarse, proj.sign_bits(&t.e), i);
    }
    for seed in 0..8u64 {
        push(&steep_latent(300 + seed), [!0, 0, !0, 0], seed as usize);
    }
    (q, q_sign, tiles)
}

#[test]
fn mix3_dequant_roundtrips_and_drops_tail() {
    let v = steep_latent(41);
    let (packed, global, gs) = quantize_latent_mix3(&v);
    let mut tile = tile_from(packed, global, gs);
    tile.flags |= FLAG_MIX3;
    let dq = tile.dequant_latent();
    // 8-bit head: error within one 8-bit step (identical to mixed).
    let eff_hi = global * (gs[0] as f32 / 255.0);
    for d in 0..MIXED_HI_DIMS {
        assert!(
            (dq[d] - v[d]).abs() <= eff_hi + 1e-6,
            "hi dim {d}: |{} - {}| > step {eff_hi}",
            dq[d],
            v[d]
        );
    }
    // TQ3 body: within one grid step of its group (same convention as
    // the grouped/TQ3 tests; the corrected bound is quarter-step when
    // gs rounding is exact).
    for d in MIXED_HI_DIMS..MIXED_DIMS {
        let g = 1 + (d - MIXED_HI_DIMS) / GROUP_DIM;
        let eff = global * (gs[g] as f32 / 255.0);
        assert!(
            (dq[d] - v[d]).abs() <= eff + 1e-6,
            "lo dim {d}: |{} - {}| > step {eff}",
            dq[d],
            v[d]
        );
    }
    // Dropped tail decodes to exactly 0.
    for d in MIXED_DIMS..D_C {
        assert_eq!(dq[d], 0.0, "tail dim {d} must decode to 0");
    }
}

#[test]
fn mix3_stays_in_the_mixed_error_class_and_beats_uniform() {
    // The whole point of the synthesis: on the steep spectrum the codec
    // was built for, MIX3 must (a) clearly beat uniform INT4 (the 8-bit
    // head does the work, like mixed), and (b) stay in the mixed codec's
    // error class — its body spends the same 4 bits/dim (3-bit grid +
    // 1-bit correction), paying only the zero-free-grid penalty.
    let sq_err = |t: &SciRustSlhaTile, v: &[f32; D_C]| -> f32 {
        let dq = t.dequant_latent();
        (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
    };
    for seed in 0..8u64 {
        let v = steep_latent(300 + seed);
        let (p0, s0, g0) = quantize_latent_grouped(&v);
        let e_uniform = sq_err(&tile_from(p0, s0, g0), &v);
        let (p1, s1, g1) = quantize_latent_mixed(&v);
        let mut mixed = tile_from(p1, s1, g1);
        mixed.flags |= FLAG_MIXED;
        let e_mixed = sq_err(&mixed, &v);
        let (p2, s2, g2) = quantize_latent_mix3(&v);
        let mut mix3 = tile_from(p2, s2, g2);
        mix3.flags |= FLAG_MIX3;
        let e_mix3 = sq_err(&mix3, &v);
        assert!(
            e_mix3 < e_uniform,
            "seed {seed}: MIX3 err {e_mix3} not < uniform {e_uniform}"
        );
        assert!(
            e_mix3 <= e_mixed * 2.0,
            "seed {seed}: MIX3 err {e_mix3} way above mixed {e_mixed}"
        );
    }
}

#[test]
fn mix3_nocorr_decodes_the_bare_grid() {
    // With FLAG_TQ3_NOCORR the body decodes without correction (head
    // untouched): body error relaxes to half a step, and the corrected
    // tile must reconstruct at least as well on aggregate.
    let v = steep_latent(43);
    let (packed, global, gs) = quantize_latent_mix3(&v);
    let mut with_corr = tile_from(packed, global, gs);
    with_corr.flags |= FLAG_MIX3;
    let mut nocorr = with_corr;
    nocorr.flags |= FLAG_TQ3_NOCORR;
    for b in nocorr.latent_kv[MIX3_CORR_OFF..].iter_mut() {
        *b = 0;
    }
    let (mut e_corr, mut e_bare) = (0.0f32, 0.0f32);
    for d in MIXED_HI_DIMS..MIXED_DIMS {
        let g = 1 + (d - MIXED_HI_DIMS) / GROUP_DIM;
        let eff = global * (gs[g] as f32 / 255.0);
        let err = (nocorr.dequant_at(d) - v[d]).abs();
        assert!(
            err <= 0.5 * eff + 1e-5,
            "dim {d}: bare-grid error {err} > half step {}",
            0.5 * eff
        );
        e_corr += (with_corr.dequant_at(d) - v[d]).powi(2);
        e_bare += (nocorr.dequant_at(d) - v[d]).powi(2);
    }
    // Head decode identical with and without the flag.
    for d in 0..MIXED_HI_DIMS {
        assert_eq!(
            with_corr.dequant_at(d).to_bits(),
            nocorr.dequant_at(d).to_bits()
        );
    }
    assert!(
        e_corr < e_bare,
        "correction must help on aggregate: {e_corr} !< {e_bare}"
    );
}

/// MIX3 equivalence fixture: a Gaussian query/sign pair and MIX3 tiles
/// built via `quantize_latent_mix3` from the steep GPT-2-like spectrum
/// the codec was designed for, sweeping all four HOT/WARM × corr/nocorr
/// flag combinations. Shared by the dispatch and per-ISA
/// score-equivalence tests.
fn mix3_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
    let mut rng = crate::rng::Rng::new(53);
    let mut q = [0.0f32; D_C];
    rng.fill_gaussian(&mut q);
    let q_sign = [
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
    ];
    let mut tiles = Vec::new();
    for seed in 0..8u64 {
        let (packed, global, gs) = quantize_latent_mix3(&steep_latent(600 + seed));
        for warm in [false, true] {
            for nocorr in [false, true] {
                let mut tile = tile_from(packed, global, gs);
                tile.dynamic_lambda = 0.37;
                tile.residual_bitmap = [!0, rng.next_u64(), !0, rng.next_u64()];
                tile.flags = FLAG_MIX3
                    | if warm { FLAG_WARM } else { 0 }
                    | if nocorr { FLAG_TQ3_NOCORR } else { 0 };
                tiles.push(tile);
            }
        }
    }
    (q, q_sign, tiles)
}

#[test]
fn mix3_score_paths_agree() {
    // Replaces `mix3_tiles_route_to_the_scalar_path`: MIX3 tiles now take
    // a SIMD path where the CPU offers one (AVX-512/AVX2 on x86-64, NEON
    // on aarch64) and fall back to scalar elsewhere, so whatever path the
    // dispatcher selects is checked for *equivalence* with the scalar
    // reference (up to float reassociation), not bit-exactness — HOT and
    // WARM, with and without the correction plane.
    let (q, q_sign, tiles) = mix3_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        let a = tile.compute_score(&q, &q_sign);
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#08b}): scalar {s} vs dispatch {a}",
            tile.flags
        );
    }
}

#[test]
fn avx2_mix3_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("avx2 unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = mix3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx2 checked just above.
            let a = unsafe { tile.compute_score_avx2_mix3(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs avx2 {a}",
                tile.flags
            );
        }
    }
}

#[test]
fn avx512_mix3_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = mix3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx512f checked just above.
            let a = unsafe { tile.compute_score_avx512_mix3(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs avx512 {a}",
                tile.flags
            );
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_mix3_path_matches_scalar() {
    let (q, q_sign, tiles) = mix3_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        // SAFETY: NEON is always available on aarch64.
        let a = unsafe { tile.compute_score_neon_mix3(&q, &q_sign) };
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#08b}): scalar {s} vs neon {a}",
            tile.flags
        );
    }
}

#[test]
fn tq3_score_paths_agree() {
    // Replaces `tq3_tiles_route_to_the_scalar_path`: TQ3 tiles now take a
    // SIMD path where the CPU offers one (AVX-512/AVX2 on x86-64, NEON on
    // aarch64) and fall back to scalar elsewhere, so whatever path the
    // dispatcher selects is checked for *equivalence* with the scalar
    // reference (up to float reassociation), not bit-exactness — HOT and
    // WARM, with and without the correction plane.
    let (q, q_sign, tiles) = tq3_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        let a = tile.compute_score(&q, &q_sign);
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#07b}): scalar {s} vs dispatch {a}",
            tile.flags
        );
    }
}

#[test]
fn avx2_tq3_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("avx2 unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = tq3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx2 checked just above.
            let a = unsafe { tile.compute_score_avx2_tq3(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#07b}): scalar {s} vs avx2 {a}",
                tile.flags
            );
        }
    }
}

#[test]
fn avx512_tq3_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = tq3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx512f checked just above.
            let a = unsafe { tile.compute_score_avx512_tq3(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#07b}): scalar {s} vs avx512 {a}",
                tile.flags
            );
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_tq3_path_matches_scalar() {
    let (q, q_sign, tiles) = tq3_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        // SAFETY: NEON is always available on aarch64.
        let a = unsafe { tile.compute_score_neon_tq3(&q, &q_sign) };
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#07b}): scalar {s} vs neon {a}",
            tile.flags
        );
    }
}

/// NF4 equivalence fixture: a query, its sign bits, and NF4 tiles built
/// via `quantize_latent_nf4` from both data shapes (Gaussian scenario
/// tokens and the steep GPT-2-like spectrum), sweeping HOT/WARM. Shared
/// by the dispatch and per-ISA score-equivalence tests.
fn nf4_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
    use crate::scenario::{generate, Projection};
    let proj = Projection::new(9);
    let (q, toks) = generate(123, 32, 0.4);
    let q_sign = proj.sign_bits(&q);
    let mut tiles = Vec::new();
    let mut push = |v: &[f32; D_C], bitmap: [u64; RESIDUAL_WORDS], i: usize| {
        let (packed, global, gs) = quantize_latent_nf4(v);
        let mut tile = tile_from(packed, global, gs);
        tile.residual_bitmap = bitmap;
        tile.dynamic_lambda = 0.37;
        tile.position = i as u32;
        tile.flags = FLAG_NF4 | if i.is_multiple_of(2) { FLAG_WARM } else { 0 };
        tiles.push(tile);
    };
    for (i, t) in toks.iter().enumerate() {
        push(&t.k_coarse, proj.sign_bits(&t.e), i);
    }
    for seed in 0..8u64 {
        push(&steep_latent(400 + seed), [!0, 0, !0, 0], seed as usize);
    }
    (q, q_sign, tiles)
}

#[test]
fn nf4_score_paths_agree() {
    // NF4 tiles take a SIMD path where the CPU offers one (AVX-512/AVX2
    // on x86-64, NEON on aarch64) and fall back to scalar elsewhere, so
    // whatever path the dispatcher selects is checked for *equivalence*
    // with the scalar reference (up to float reassociation), not
    // bit-exactness — HOT and WARM.
    let (q, q_sign, tiles) = nf4_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        let a = tile.compute_score(&q, &q_sign);
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#08b}): scalar {s} vs dispatch {a}",
            tile.flags
        );
    }
}

#[test]
fn avx2_nf4_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("avx2 unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = nf4_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx2 checked just above.
            let a = unsafe { tile.compute_score_avx2_nf4(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs avx2 {a}",
                tile.flags
            );
        }
    }
}

#[test]
fn avx512_nf4_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = nf4_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx512f checked just above.
            let a = unsafe { tile.compute_score_avx512_nf4(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs avx512 {a}",
                tile.flags
            );
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_nf4_path_matches_scalar() {
    let (q, q_sign, tiles) = nf4_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        // SAFETY: NEON is always available on aarch64.
        let a = unsafe { tile.compute_score_neon_nf4(&q, &q_sign) };
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#08b}): scalar {s} vs neon {a}",
            tile.flags
        );
    }
}

/// Mixed-precision equivalence fixture: a Gaussian query/sign pair and
/// mixed tiles built via `quantize_latent_mixed` from the steep
/// GPT-2-like spectrum the codec was designed for, sweeping HOT/WARM.
/// Shared by the dispatch and per-ISA score-equivalence tests.
fn mixed_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
    let mut rng = crate::rng::Rng::new(33);
    let mut q = [0.0f32; D_C];
    rng.fill_gaussian(&mut q);
    let q_sign = [
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
    ];
    let mut tiles = Vec::new();
    for seed in 0..8u64 {
        let (packed, global, gs) = quantize_latent_mixed(&steep_latent(500 + seed));
        for warm in [false, true] {
            let mut tile = tile_from(packed, global, gs);
            tile.dynamic_lambda = 0.37;
            tile.residual_bitmap = [!0, rng.next_u64(), !0, rng.next_u64()];
            tile.flags = FLAG_MIXED | if warm { FLAG_WARM } else { 0 };
            tiles.push(tile);
        }
    }
    (q, q_sign, tiles)
}

#[test]
fn mixed_score_paths_agree() {
    // Replaces `mixed_tiles_route_to_the_scalar_path`: mixed tiles now
    // take a SIMD path where the CPU offers one (AVX-512/AVX2 on x86-64,
    // NEON on aarch64) and fall back to scalar elsewhere, so whatever
    // path the dispatcher selects is checked for *equivalence* with the
    // scalar reference (up to float reassociation), not bit-exactness —
    // HOT and WARM.
    let (q, q_sign, tiles) = mixed_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        let a = tile.compute_score(&q, &q_sign);
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#08b}): scalar {s} vs dispatch {a}",
            tile.flags
        );
    }
}

#[test]
fn avx2_mixed_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("avx2 unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = mixed_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx2 checked just above.
            let a = unsafe { tile.compute_score_avx2_mixed(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs avx2 {a}",
                tile.flags
            );
        }
    }
}

#[test]
fn avx512_mixed_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f unavailable — skipping equivalence check");
            return;
        }
        let (q, q_sign, tiles) = mixed_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: avx512f checked just above.
            let a = unsafe { tile.compute_score_avx512_mixed(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs avx512 {a}",
                tile.flags
            );
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_mixed_path_matches_scalar() {
    let (q, q_sign, tiles) = mixed_equivalence_fixture();
    for (i, tile) in tiles.iter().enumerate() {
        let s = tile.compute_score_scalar(&q, &q_sign);
        // SAFETY: NEON is always available on aarch64.
        let a = unsafe { tile.compute_score_neon_mixed(&q, &q_sign) };
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i} (flags {:#08b}): scalar {s} vs neon {a}",
            tile.flags
        );
    }
}

#[test]
fn avx2_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("avx2 unavailable — skipping equivalence check");
            return;
        }
        use crate::scenario::{build_tile, generate, Projection};
        let proj = Projection::new(9);
        let (q, toks) = generate(123, 64, 0.4);
        let q_sign = proj.sign_bits(&q);
        for (i, t) in toks.iter().enumerate() {
            // Alternate HOT / WARM to cover both branches.
            let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = unsafe { tile.compute_score_avx2(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i}: scalar {s} vs avx2 {a}"
            );
        }
    }
}

#[test]
fn avx512_path_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f unavailable — skipping equivalence check");
            return;
        }
        use crate::scenario::{build_tile, generate, Projection};
        let proj = Projection::new(9);
        let (q, toks) = generate(123, 64, 0.4);
        let q_sign = proj.sign_bits(&q);
        for (i, t) in toks.iter().enumerate() {
            let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = unsafe { tile.compute_score_avx512(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i}: scalar {s} vs avx512 {a}"
            );
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_path_matches_scalar() {
    use crate::scenario::{build_tile, generate, Projection};
    let proj = Projection::new(9);
    let (q, toks) = generate(123, 64, 0.4);
    let q_sign = proj.sign_bits(&q);
    for (i, t) in toks.iter().enumerate() {
        // Alternate HOT / WARM to cover both branches.
        let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
        let s = tile.compute_score_scalar(&q, &q_sign);
        let a = unsafe { tile.compute_score_neon(&q, &q_sign) };
        assert!(
            (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
            "tile {i}: scalar {s} vs neon {a}"
        );
    }
}

/// The public `hamming_distance` dispatcher (whatever SIMD path the running
/// CPU selects) must equal a brute-force per-bit count, over random inputs.
#[test]
fn hamming_distance_matches_bruteforce() {
    let mut rng = crate::rng::Rng::new(0x4D31);
    for _ in 0..4000 {
        let a = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let b = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let brute: u32 = (0..D_S)
            .map(|s| (((a[s >> 6] >> (s & 63)) ^ (b[s >> 6] >> (s & 63))) & 1) as u32)
            .sum();
        assert_eq!(hamming_distance(&a, &b), brute);
    }
}

/// On CPUs that advertise it, the AVX-512 VPOPCNTDQ path must be bit-exact
/// with the scalar reduction. Compile-checked on every x86-64 build; only
/// *executed* where the feature is present (skipped on this bench otherwise).
#[test]
#[cfg(target_arch = "x86_64")]
fn vpopcntdq_hamming_matches_scalar() {
    if !(std::is_x86_feature_detected!("avx512vpopcntdq")
        && std::is_x86_feature_detected!("avx512vl"))
    {
        eprintln!("avx512vpopcntdq+vl unavailable — skipping (compile-checked only)");
        return;
    }
    let mut rng = crate::rng::Rng::new(0x5E42);
    for _ in 0..4000 {
        let a = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let b = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        // SAFETY: features checked just above.
        let simd = unsafe { hamming_vpopcntdq(&a, &b) };
        assert_eq!(simd, hamming_scalar(&a, &b));
    }
}
