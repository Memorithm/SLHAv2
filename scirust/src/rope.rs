//! Rotary position embedding (RoPE) for the pre-RoPE projection study
//! (plan axis **A1**, after ShadowKV / KVQuant).
//!
//! The synthetic prototype has no real transformer, so there is no RoPE in the
//! data path. To study the **A1** lever — *project the key before RoPE rather
//! than after* — we need a position-dependent rotation that mixes channels the
//! way RoPE does in a real model. This module provides exactly that: a standard
//! paired-channel rotation, `θ_i(pos) = pos · base^{−2i/d}`, applied in place to
//! `(v[2i], v[2i+1])`.
//!
//! RoPE is **orthogonal**: `⟨RoPE_p(a), RoPE_p(b)⟩ = ⟨a, b⟩` (same position).
//! That is the property the A1 measurement relies on — see
//! `examples/pre_rope_projection.rs` and the improvement plan
//! (`docs/SLHAv2_schema_plan.pdf`, axis A1).
//!
//! See: ShadowKV (arXiv 2410.21465), KVQuant pre-RoPE (2401.18079).

/// Default RoPE base (θ frequency decay), as in GPT-J / Llama.
pub const ROPE_BASE: f32 = 10000.0;

/// Apply RoPE in place to `v` (length `d`, must be even) at sequence position
/// `pos`, with frequency base `base`. Rotates each channel pair `(2i, 2i+1)`
/// by `θ_i = pos · base^{−2i/d}`. `pos = 0` is the identity.
pub fn rope(v: &mut [f32], pos: u32, base: f32) {
    let d = v.len();
    assert!(d.is_multiple_of(2), "rope needs an even dimension, got {d}");
    let p = pos as f32;
    if p == 0.0 {
        return;
    }
    let inv_d = 1.0 / d as f32;
    let mut i = 0usize;
    while i < d {
        // θ_i = pos · base^{-(2i)/d} ; pair index i/2 maps to exponent -2*(i/2)/d.
        let freq = base.powf(-(2.0 * (i as f32) * inv_d / 2.0));
        let theta = p * freq;
        let (c, s) = (theta.cos(), theta.sin());
        let a = v[i];
        let b = v[i + 1];
        v[i] = a * c - b * s;
        v[i + 1] = a * s + b * c;
        i += 2;
    }
}

/// Convenience: copy `src` and rotate the copy by `pos` (leaving `src` intact).
pub fn rope_copy(src: &[f32], pos: u32, base: f32) -> Vec<f32> {
    let mut out = src.to_vec();
    rope(&mut out, pos, base);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_zero_is_identity() {
        let mut rng = crate::rng::Rng::new(3);
        let d = 64;
        let mut v = vec![0.0f32; d];
        rng.fill_gaussian(&mut v);
        let orig = v.clone();
        rope(&mut v, 0, ROPE_BASE);
        assert_eq!(v, orig);
    }

    /// RoPE is orthogonal: the L2 norm is preserved at every position.
    #[test]
    fn rope_preserves_norm() {
        let mut rng = crate::rng::Rng::new(5);
        let d = 128;
        let mut v = vec![0.0f32; d];
        rng.fill_gaussian(&mut v);
        let n0: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for pos in [1u32, 7, 100, 4096] {
            let mut w = v.clone();
            rope(&mut w, pos, ROPE_BASE);
            let n1: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n0 - n1).abs() <= 1e-4 * n0, "pos {pos}: norm {n0} -> {n1}");
        }
    }

    /// Same-position orthogonality of the dot product: rotating both vectors
    /// by the same angle leaves their dot unchanged.
    #[test]
    fn rope_preserves_same_position_dot() {
        let mut rng = crate::rng::Rng::new(9);
        let d = 64;
        let mut a = vec![0.0f32; d];
        let mut b = vec![0.0f32; d];
        rng.fill_gaussian(&mut a);
        rng.fill_gaussian(&mut b);
        let dot0: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let pos = 123u32;
        let ra = rope_copy(&a, pos, ROPE_BASE);
        let rb = rope_copy(&b, pos, ROPE_BASE);
        let dot1: f32 = ra.iter().zip(&rb).map(|(x, y)| x * y).sum();
        assert!(
            (dot0 - dot1).abs() <= 1e-3 * (1.0 + dot0.abs()),
            "dot not preserved: {dot0} -> {dot1}"
        );
    }

    /// A single pair rotated by a known angle matches the analytic rotation.
    #[test]
    fn rope_pair_matches_analytic() {
        let mut v = vec![1.0f32, 0.0];
        // d=2 ⇒ freq = base^0 = 1 ⇒ θ = pos.
        rope(&mut v, 1, ROPE_BASE); // θ = 1 rad
        assert!((v[0] - 1.0_f32.cos()).abs() <= 1e-5, "v[0] {}", v[0]);
        assert!((v[1] - 1.0_f32.sin()).abs() <= 1e-5, "v[1] {}", v[1]);
    }
}
