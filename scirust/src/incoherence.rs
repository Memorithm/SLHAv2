//! Incoherence processing for the sign-LSH residual (plan axis **A2**).
//!
//! The 1-bit residual `B = sign(Z·E)` loses resolution on directions whose
//! energy is concentrated on a few channels (outliers): a single sign-LSH
//! hyperplane then captures almost the same bit regardless of where the query
//! sits, and the §7.1 Spearman of the binary core stalls around 0.67. QuIP#
//! and Palu fix this for weight quantisation with **incoherence processing** —
//! a randomised Hadamard transform (RHT) that flattens the energy spectrum so
//! every channel carries roughly equal mass, before quantising.
//!
//! Here we apply the same idea to the *residual* of the SLHA key split. The RHT
//! is `H·D` where `D` is a fixed diagonal of ±1 signs and `H` is the orthonormal
//! Walsh–Hadamard matrix, realised in-place by the fast transform (FWHT,
//! O(d·log d), no floating multiply on the ±1 diagonal). It is applied
//! identically to the residual `E` and to the query `Q` before the sign-LSH.
//!
//! Because the RHT is **orthogonal**, `⟨RHT·E, RHT·Q⟩ = ⟨E, Q⟩`: the signed
//! dot product the Hamming distance approximates is unchanged, the fused score
//! (eq. 2.3) is preserved, and the coarse latent path is untouched. Only the
//! *resolution* of the 1-bit residual improves.
//!
//! See: QuIP# (arXiv 2402.04396), Palu fused Walsh-Hadamard (2407.21118),
//! NSNQuant (2505.18231). Grounded in the improvement plan
//! (`docs/SLHAv2_schema_plan.pdf`, axis A2).

/// Smallest power of two `≥ n` (FWHT operates on power-of-two lengths). The
/// RHT pads the input with zeros up to this length; zero padding contributes
/// nothing to the dot product, so `⟨RHT·E, RHT·Q⟩ = ⟨E, Q⟩` still holds.
pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place orthonormal Fast Walsh–Hadamard Transform. `v.len()` must be a
/// power of two. Scales by `1/√n` so the transform is orthogonal
/// (`H·Hᵀ = I`); applying it twice recovers the input.
///
/// Uses the iterative Cooley–Tukey pattern: log₂(n) stages, each pairing
/// indices that differ only in bit `s`. Pure adds/subtracts plus one final
/// scale — no FP multiply in the inner loop, which is the whole point of the
/// Hadamard basis for incoherence processing.
pub fn fwht_inplace(v: &mut [f32]) {
    let n = v.len();
    assert!(
        n.is_power_of_two(),
        "fwht needs a power-of-two length, got {n}"
    );
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += h << 1;
        }
        h <<= 1;
    }
    // Orthonormalise: H_n / √n.
    let inv = (n as f32).sqrt().recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Fixed randomised Hadamard transform: `H·D` with `D` a deterministic diagonal
/// of ±1 signs. `D` randomises the basis (so the flattening is non-trivial and
/// data-independent); `H` spreads the energy across all channels.
///
/// Build with [`HadamardIncoherence::new`] and apply with
/// [`HadamardIncoherence::transform_into`], which zero-pads a length-`d` input
/// to `d_pad = next_pow2(d)`, multiplies by the diagonal, then runs the FWHT.
pub struct HadamardIncoherence {
    /// ±1 diagonal, length `d_pad`.
    diag: Vec<f32>,
    d: usize,
    d_pad: usize,
}

impl HadamardIncoherence {
    /// Build the RHT for inputs of length `d`. `d_pad` is `next_pow2(d)`; the
    /// diagonal signs (length `d_pad`) are drawn deterministically from `seed`.
    pub fn new(d: usize, seed: u64) -> Self {
        let d_pad = next_pow2(d);
        let mut rng = crate::rng::Rng::new(seed);
        let mut diag = vec![1.0f32; d_pad];
        // ±1 with the prototype's PRNG — data-independent, reproducible.
        for x in diag.iter_mut() {
            *x = if rng.next_unit() < 0.5 { -1.0 } else { 1.0 };
        }
        HadamardIncoherence { diag, d, d_pad }
    }

    /// Padded length the caller must size the sign-LSH `Z` for.
    pub fn d_pad(&self) -> usize {
        self.d_pad
    }

    /// Original (un-padded) input length.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Apply `H·D` to a length-`d` input, writing the length-`d_pad` transformed
    /// vector into `out`. `out.len()` must equal [`Self::d_pad`]. The input is
    /// copied into `out[..d]`, the tail is zeroed, then `D` and the FWHT run
    /// in place. The transform is orthogonal: `||out|| == ||in||`.
    pub fn transform_into(&self, v: &[f32], out: &mut [f32]) {
        assert_eq!(v.len(), self.d, "input length must be d");
        assert_eq!(out.len(), self.d_pad, "out must be d_pad");
        out[..self.d].copy_from_slice(v);
        for x in out[self.d..].iter_mut() {
            *x = 0.0;
        }
        for (i, x) in out.iter_mut().enumerate() {
            *x *= self.diag[i];
        }
        fwht_inplace(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The orthonormal FWHT is its own inverse: applying it twice recovers the
    /// input (up to float rounding).
    #[test]
    fn fwht_is_orthogonal_inverse() {
        let mut rng = crate::rng::Rng::new(11);
        let n = 128;
        let mut v = vec![0.0f32; n];
        rng.fill_gaussian(&mut v);
        let orig = v.clone();
        fwht_inplace(&mut v);
        fwht_inplace(&mut v);
        for i in 0..n {
            assert!(
                (v[i] - orig[i]).abs() <= 1e-4 * (1.0 + orig[i].abs()),
                "idx {i}: {} != {}",
                v[i],
                orig[i]
            );
        }
    }

    /// FWHT preserves the L2 norm (orthonormal).
    #[test]
    fn fwht_preserves_norm() {
        let mut rng = crate::rng::Rng::new(12);
        let n = 64;
        let mut v = vec![0.0f32; n];
        rng.fill_gaussian(&mut v);
        let n0: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        fwht_inplace(&mut v);
        let n1: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - n1).abs() <= 1e-4 * n0, "norm {n0} -> {n1}");
    }

    /// `next_pow2` is the smallest power of two ≥ n.
    #[test]
    fn next_pow2_correct() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(160), 256);
        assert_eq!(next_pow2(257), 512);
    }

    /// The full RHT (H·D) is orthogonal: `||RHT·v|| == ||v||`, and two inputs
    /// keep their dot product after the transform.
    #[test]
    fn rht_is_orthogonal_and_dot_preserving() {
        let d = 160;
        let rht = HadamardIncoherence::new(d, 99);
        let dp = rht.d_pad();
        assert_eq!(dp, 256);

        let mut rng = crate::rng::Rng::new(7);
        let mut a = vec![0.0f32; d];
        let mut b = vec![0.0f32; d];
        rng.fill_gaussian(&mut a);
        rng.fill_gaussian(&mut b);

        let dot_in: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let mut ta = vec![0.0f32; dp];
        let mut tb = vec![0.0f32; dp];
        rht.transform_into(&a, &mut ta);
        rht.transform_into(&b, &mut tb);

        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nta: f32 = ta.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (na - nta).abs() <= 1e-4 * na,
            "norm not preserved: {na} -> {nta}"
        );

        let dot_out: f32 = ta.iter().zip(&tb).map(|(x, y)| x * y).sum();
        assert!(
            (dot_in - dot_out).abs() <= 1e-3 * (1.0 + dot_in.abs()),
            "dot not preserved: {dot_in} -> {dot_out}"
        );
    }

    /// The whole point of incoherence processing: when the residual energy is
    /// concentrated on a few channels (outliers), the flattened RHT spectrum
    /// has lower peak-to-mean ratio — every channel carries comparable mass.
    #[test]
    fn rht_flattens_concentrated_energy() {
        let d = 128;
        let rht = HadamardIncoherence::new(d, 3);
        let dp = rht.d_pad();
        // Energy concentrated on 3 of 128 channels (typical outlier pattern).
        let mut e = vec![0.0f32; d];
        e[0] = 10.0;
        e[1] = -9.0;
        e[40] = 8.0;

        let peak_to_mean = |v: &[f32]| -> f32 {
            let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            v.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / mean.max(1e-12)
        };
        let p_in = peak_to_mean(&e);

        let mut t = vec![0.0f32; dp];
        rht.transform_into(&e, &mut t);
        let p_out = peak_to_mean(&t);

        assert!(
            p_out < p_in * 0.25,
            "RHT did not flatten: peak/mean {p_in:.1} -> {p_out:.1}"
        );
    }
}
