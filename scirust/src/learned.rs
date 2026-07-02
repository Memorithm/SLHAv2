//! Learned low-rank projection for SLHA v2.
//!
//! The synthetic prototype in [`crate::scenario`] *assumes* the low-rank base
//! is captured ideally. This module removes that assumption: it **learns** the
//! projection by PCA (the optimal linear rank-`D_C` reconstruction, by
//! Eckart–Young) on sample keys, then measures how much energy a real rank-`D_C`
//! projection actually keeps.
//!
//! ## Latent whitening (INT4-friendliness)
//! PCA latent components have wildly different variances (the eigenvalue
//! spectrum). A single per-tile INT4 scale is then dominated by the top
//! component and crushes the small ones. We therefore optionally **whiten** the
//! latent — store `h_k = (e_k·K) / s_k` with `s_k = sqrt(λ_k)` so every
//! component has ~unit variance — and **de-whiten the query** — `q_k =
//! (e_k·Q)·s_k`. The product `Σ q_k h_k = Σ (e_k·Q)(e_k·K)` is **identical** to
//! the un-whitened score, so whitening is mathematically neutral on the score
//! while making the INT4 latent far more accurate.
//!
//! Dimensions: keys live in `R^d` (`d > D_C`); the latent is `R^{D_C}`; the
//! sign-LSH residual `Z` maps `R^d -> D_S` bits. Everything still feeds the
//! *unchanged* fixed-size tile and kernel.

use crate::attention::slha_v2::{
    quantize_latent, quantize_latent_grouped, quantize_latent_mixed, quantize_latent_nf4,
    quantize_latent_tq3, LatentCodec, SciRustSlhaTile, D_C, D_S, FLAG_MIXED, FLAG_NF4, FLAG_TQ3,
    FLAG_WARM, N_GROUPS, RESIDUAL_WORDS,
};
use crate::incoherence::HadamardIncoherence;
use crate::linalg::jacobi_eigh;
use crate::rng::Rng;

/// A PCA-learned projection plus a fixed random sign-LSH `Z` (`D_S × d_pad`).
pub struct LearnedModel {
    pub d: usize,
    /// Padded length the sign-LSH `Z` operates on (`next_pow2(d)` when an RHT
    /// is applied for incoherence processing, else `d`).
    d_pad: usize,
    /// Top-`D_C` principal eigenvectors, row-major `D_C × d`.
    evec: Vec<f32>,
    /// Per-component scale `s_k` (`sqrt(λ_k)` if whitening, else `1.0`).
    scale: Vec<f32>,
    pub z: Vec<f32>,
    /// Optional incoherence transform (plan axis A2) applied to the residual
    /// and the query before the sign-LSH. Orthogonal ⇒ score-preserving.
    rht: Option<HadamardIncoherence>,
    /// Fraction of total variance retained by the top-`D_C` subspace.
    pub captured_energy: f32,
    pub whiten: bool,
}

impl LearnedModel {
    /// Fit by PCA on `train` keys (each length `d`). `Z` is a fixed random
    /// Johnson–Lindenstrauss projection. With `whiten`, latent components are
    /// normalised to unit variance (score-preserving — see module docs).
    /// No incoherence processing — see [`Self::fit_with`].
    pub fn fit(train: &[Vec<f32>], d: usize, seed: u64, whiten: bool) -> Self {
        Self::fit_with(train, d, seed, whiten, false)
    }

    /// As [`Self::fit`], optionally enabling incoherence processing (plan axis
    /// A2): with `rht = true` the residual and query are transformed by a
    /// fixed randomised Hadamard transform before the sign-LSH. Orthogonal ⇒
    /// the fused score is unchanged; only the 1-bit residual resolution
    /// improves (see [`crate::incoherence`]).
    pub fn fit_with(train: &[Vec<f32>], d: usize, seed: u64, whiten: bool, rht: bool) -> Self {
        assert!(d > D_C, "need d > D_C for a non-trivial residual");
        let n = train.len().max(1);

        // Empirical (uncentered) covariance, row-major d×d.
        let mut cov = vec![0.0f32; d * d];
        for key in train {
            for i in 0..d {
                let ki = key[i];
                let row = i * d;
                for j in 0..d {
                    cov[row + j] += ki * key[j];
                }
            }
        }
        let inv = 1.0 / n as f32;
        for c in cov.iter_mut() {
            *c *= inv;
        }
        Self::fit_from_cov(cov, d, seed, whiten, rht)
    }

    /// Fit on the pooled second moment of **keys and queries** (first step of
    /// the plan's §1.3 score-aware objective). The coarse score is `⟨P·q, P·k⟩`,
    /// so query energy outside `span(P)` is lost even when the keys reconstruct
    /// perfectly — and real query distributions carry such energy (measured on
    /// held-out GPT-2 layer-6 activations: a keys-only PCA keeps 69.6% of the
    /// real queries' energy, and pooling raises the float coarse-score
    /// attention-output cosine from 0.958 to 0.971). With `queries` empty this
    /// is exactly [`Self::fit_with`] (same covariance, bit-identical).
    ///
    /// `captured_energy` is the *pooled* (keys + queries) captured fraction.
    pub fn fit_joint(
        keys: &[Vec<f32>],
        queries: &[Vec<f32>],
        d: usize,
        seed: u64,
        whiten: bool,
        rht: bool,
    ) -> Self {
        assert!(d > D_C, "need d > D_C for a non-trivial residual");
        let n = (keys.len() + queries.len()).max(1);

        let mut cov = vec![0.0f32; d * d];
        for v in keys.iter().chain(queries) {
            for i in 0..d {
                let vi = v[i];
                let row = i * d;
                for j in 0..d {
                    cov[row + j] += vi * v[j];
                }
            }
        }
        let inv = 1.0 / n as f32;
        for c in cov.iter_mut() {
            *c *= inv;
        }
        Self::fit_from_cov(cov, d, seed, whiten, rht)
    }

    /// Shared tail of the `fit*` constructors: eigendecompose a d×d second
    /// moment, keep the top-`D_C` subspace, seed `Z` (+ optional RHT).
    fn fit_from_cov(cov: Vec<f32>, d: usize, seed: u64, whiten: bool, rht: bool) -> Self {
        let (eigvals, eigvecs) = jacobi_eigh(&cov, d);

        let mut idx: Vec<usize> = (0..d).collect();
        idx.sort_by(|&a, &b| eigvals[b].total_cmp(&eigvals[a]));

        let total: f64 = eigvals.iter().map(|&x| x.max(0.0)).sum();
        let kept: f64 = idx[..D_C].iter().map(|&i| eigvals[i].max(0.0)).sum();
        let captured_energy = if total > 0.0 {
            (kept / total) as f32
        } else {
            1.0
        };

        // Floor the whitening scale so near-zero eigenvalues don't blow up
        // pure-noise components (cap the dynamic range at ~sqrt(1e3)).
        let lambda_top = eigvals[idx[0]].max(1e-12);
        let floor = lambda_top * 1e-3;

        let mut evec = vec![0.0f32; D_C * d];
        let mut scale = vec![1.0f32; D_C];
        for (k, &ei) in idx[..D_C].iter().enumerate() {
            for i in 0..d {
                evec[k * d + i] = eigvecs[i * d + ei] as f32;
            }
            if whiten {
                scale[k] = (eigvals[ei].max(floor) as f32).sqrt();
            }
        }

        // Incoherence transform: sizes the sign-LSH Z at D_S × d_pad.
        let rht_obj = if rht {
            Some(HadamardIncoherence::new(
                d,
                seed ^ 0xA2_A2_A2_A2_A2_A2_A2_A2,
            ))
        } else {
            None
        };
        let d_pad = rht_obj.as_ref().map_or(d, |r| r.d_pad());

        let mut rng = Rng::new(seed);
        let mut z = vec![0.0f32; D_S * d_pad];
        rng.fill_gaussian(&mut z);

        LearnedModel {
            d,
            d_pad,
            evec,
            scale,
            z,
            rht: rht_obj,
            captured_energy,
            whiten,
        }
    }

    /// Build a model from an arbitrary (e.g. SGD-learned) projection `p`
    /// (`D_C × d`, row-major), with no per-component whitening. `Z` is seeded
    /// like [`Self::fit`], so two models built with the same `seed` share the
    /// residual projection and differ only in `P`. No incoherence processing
    /// — see [`Self::from_projection_with`].
    pub fn from_projection(p: Vec<f32>, d: usize, seed: u64) -> Self {
        Self::from_projection_with(p, d, seed, false)
    }

    /// As [`Self::from_projection`], optionally enabling incoherence
    /// processing (plan axis A2).
    pub fn from_projection_with(p: Vec<f32>, d: usize, seed: u64, rht: bool) -> Self {
        assert_eq!(p.len(), D_C * d, "projection must be D_C×d");
        let rht_obj = if rht {
            Some(HadamardIncoherence::new(
                d,
                seed ^ 0xA2_A2_A2_A2_A2_A2_A2_A2,
            ))
        } else {
            None
        };
        let d_pad = rht_obj.as_ref().map_or(d, |r| r.d_pad());
        let mut rng = Rng::new(seed);
        let mut z = vec![0.0f32; D_S * d_pad];
        rng.fill_gaussian(&mut z);
        LearnedModel {
            d,
            d_pad,
            evec: p,
            scale: vec![1.0f32; D_C],
            z,
            rht: rht_obj,
            captured_energy: f32::NAN, // not the projector of a single covariance
            whiten: false,
        }
    }

    /// The projection matrix `P` (`D_C × d`, row-major) — e.g. to warm-start
    /// [`train_projection`] from this model's PCA solution.
    pub fn projection(&self) -> &[f32] {
        &self.evec
    }

    #[inline]
    fn evec_dot(&self, k: usize, v: &[f32]) -> f32 {
        let row = &self.evec[k * self.d..(k + 1) * self.d];
        let mut acc = 0.0f32;
        for i in 0..self.d {
            acc += row[i] * v[i];
        }
        acc
    }

    /// Whitened latent `h_k = (e_k·key) / s_k` (length `D_C`).
    pub fn latent(&self, key: &[f32]) -> [f32; D_C] {
        let mut h = [0.0f32; D_C];
        for (k, hk) in h.iter_mut().enumerate() {
            *hk = self.evec_dot(k, key) / self.scale[k];
        }
        h
    }

    /// De-whitened query coarse `q_k = (e_k·Q)·s_k` (length `D_C`), consumed
    /// directly by `compute_score`.
    pub fn query_coarse(&self, q: &[f32]) -> [f32; D_C] {
        let mut h = [0.0f32; D_C];
        for (k, hk) in h.iter_mut().enumerate() {
            *hk = self.evec_dot(k, q) * self.scale[k];
        }
        h
    }

    /// Reconstruction `K_coarse_i = Σ_k h_k · s_k · e_k[i]` (length `d`).
    pub fn reconstruct(&self, h: &[f32; D_C]) -> Vec<f32> {
        let mut r = vec![0.0f32; self.d];
        for k in 0..D_C {
            let w = h[k] * self.scale[k];
            let row = &self.evec[k * self.d..(k + 1) * self.d];
            for i in 0..self.d {
                r[i] += w * row[i];
            }
        }
        r
    }

    /// Packed sign bits of `Z · v` (`v` length `d`). With incoherence
    /// processing enabled the effective projection is `Z · RHT`, realised by
    /// transforming `v` (zero-padded to `d_pad`) before the dot — the result is
    /// `sign(Z · (RHT·v))`. Orthogonal RHT ⇒ the Hamming distance still
    /// approximates the signed dot product of the underlying vectors.
    pub fn sign_bits(&self, v: &[f32]) -> [u64; RESIDUAL_WORDS] {
        assert_eq!(v.len(), self.d, "input length must be d");
        let mut buf = vec![0.0f32; self.d_pad];
        if let Some(rht) = &self.rht {
            rht.transform_into(v, &mut buf);
        } else {
            // No RHT: d_pad == d, straight copy.
            buf.copy_from_slice(v);
        }
        let mut out = [0u64; RESIDUAL_WORDS];
        for s in 0..D_S {
            let row = &self.z[s * self.d_pad..(s + 1) * self.d_pad];
            let mut acc = 0.0f32;
            for i in 0..self.d_pad {
                acc += row[i] * buf[i];
            }
            if acc < 0.0 {
                out[s >> 6] |= 1u64 << (s & 63);
            }
        }
        out
    }

    /// Encode a key into a tile. The residual `E = key - reconstruct(latent)` is
    /// computed from the *un-quantised* latent; the quantisation error is a
    /// separate, smaller approximation folded into the coarse term. `codec`
    /// selects the latent quantiser.
    pub fn encode_with(
        &self,
        key: &[f32],
        pos: u32,
        warm: bool,
        codec: LatentCodec,
    ) -> SciRustSlhaTile {
        let h = self.latent(key);
        let (latent, scale, group_scales, codec_flag) = match codec {
            LatentCodec::Int4Single => {
                let (l, s) = quantize_latent(&h);
                (l, s, [255u8; N_GROUPS], 0)
            }
            LatentCodec::Int4Grouped => {
                let (l, s, gs) = quantize_latent_grouped(&h);
                (l, s, gs, 0)
            }
            LatentCodec::Nf4 => {
                let (l, s, gs) = quantize_latent_nf4(&h);
                (l, s, gs, FLAG_NF4)
            }
            LatentCodec::Mixed => {
                let (l, s, gs) = quantize_latent_mixed(&h);
                (l, s, gs, FLAG_MIXED)
            }
            LatentCodec::Tq3 => {
                let (l, s, gs) = quantize_latent_tq3(&h);
                (l, s, gs, FLAG_TQ3)
            }
        };
        let recon = self.reconstruct(&h);
        let mut e = vec![0.0f32; self.d];
        for i in 0..self.d {
            e[i] = key[i] - recon[i];
        }
        let bitmap = self.sign_bits(&e);
        let sigma_e = (e.iter().map(|x| x * x).sum::<f32>() / self.d as f32).sqrt();
        let lambda = sigma_e * (std::f32::consts::PI / (2.0 * D_S as f32)).sqrt();
        let flags = if warm { FLAG_WARM } else { 0 } | codec_flag;
        SciRustSlhaTile {
            latent_kv: latent,
            residual_bitmap: bitmap,
            scale,
            dynamic_lambda: lambda,
            residual_sigma: sigma_e,
            token_id: pos,
            position: pos,
            head_id: 0,
            flags,
            group_scales,
        }
    }

    /// Encode a key into a tile (per-group micro-scaled INT4 latent).
    pub fn encode(&self, key: &[f32], pos: u32, warm: bool) -> SciRustSlhaTile {
        self.encode_with(key, pos, warm, LatentCodec::Int4Grouped)
    }
}

/// Fraction of total key variance captured by a rank-`rank` PCA on `train`
/// (each key length `d`). Eckart–Young: the top-`rank` principal subspace is
/// the optimal rank-`rank` linear reconstruction. Exposed so callers can probe
/// the spectrum at *any* rank — e.g. the A1 study measures how RoPE inflates
/// the effective rank by comparing `captured_energy_at(pre_rope, d, k)` vs
/// `captured_energy_at(post_rope, d, k)` across `k` (see
/// `examples/pre_rope_projection.rs`).
pub fn captured_energy_at(train: &[Vec<f32>], d: usize, rank: usize) -> f32 {
    assert!(rank <= d, "rank must be ≤ d");
    let n = train.len().max(1);
    let mut cov = vec![0.0f32; d * d];
    for key in train {
        for i in 0..d {
            let ki = key[i];
            let row = i * d;
            for j in 0..d {
                cov[row + j] += ki * key[j];
            }
        }
    }
    let inv = 1.0 / n as f32;
    for c in cov.iter_mut() {
        *c *= inv;
    }
    let (eigvals, _eigvecs) = jacobi_eigh(&cov, d);
    let mut sorted: Vec<f64> = eigvals.iter().map(|&x| x.max(0.0)).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = sorted.iter().sum();
    if total <= 0.0 {
        return 1.0;
    }
    let kept: f64 = sorted[..rank].iter().sum();
    (kept / total) as f32
}

/// Generate `n` keys in `R^d` from a factor model: `r` random unit factor
/// directions with geometrically decaying strength `decay^i`, plus isotropic
/// `noise`. Lower `decay` / higher `noise` ⇒ flatter spectrum ⇒ a rank-`D_C`
/// projection captures less ⇒ larger residual.
pub fn gen_keys(seed: u64, n: usize, d: usize, r: usize, decay: f32, noise: f32) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);

    let mut factors = vec![vec![0.0f32; d]; r];
    for f in factors.iter_mut() {
        rng.fill_gaussian(f);
        let nrm = f.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in f.iter_mut() {
            *x /= nrm;
        }
    }
    let strengths: Vec<f32> = (0..r).map(|i| decay.powi(i as i32)).collect();

    let mut keys = Vec::with_capacity(n);
    for _ in 0..n {
        let mut k = vec![0.0f32; d];
        for fi in 0..r {
            let g = rng.next_gaussian() * strengths[fi];
            let fv = &factors[fi];
            for i in 0..d {
                k[i] += g * fv[i];
            }
        }
        for i in 0..d {
            k[i] += noise * rng.next_gaussian();
        }
        keys.push(k);
    }
    keys
}

/// Train a projection `P` (`D_C × d`) by SGD to minimise the **score** error
/// `E_{Q,K}[(⟨Q,K⟩ − ⟨PQ,PK⟩)²]` over *independent* query/key samples — a
/// task-aware objective, unlike PCA which only minimises key reconstruction and
/// ignores the query distribution.
///
/// Gradient (closed form, per sample): with `a = Pq`, `b = Pk`,
/// `r = ⟨q,k⟩ − ⟨a,b⟩`, then `∂r²/∂P = −2r (b qᵀ + a kᵀ)`.
///
/// Returns `(P, loss_history)` (mean per-sample loss per epoch). Warm-start
/// `init_p` from a PCA fit to measure the task-aware improvement on top of it.
pub fn train_projection(
    qs: &[Vec<f32>],
    ks: &[Vec<f32>],
    init_p: Vec<f32>,
    epochs: usize,
    lr: f32,
    batch: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        init_p.len().is_multiple_of(D_C),
        "projection must be D_C×d (row-major)"
    );
    let d = init_p.len() / D_C;
    let mut p = init_p;
    let mut rng = Rng::new(seed);
    let (nq, nk) = (qs.len(), ks.len());
    let mut a = vec![0.0f32; D_C];
    let mut b = vec![0.0f32; D_C];
    let mut grad = vec![0.0f32; D_C * d];
    let steps = (nq.max(nk) / batch).max(1);
    let mut history = Vec::with_capacity(epochs);

    for ep in 0..epochs {
        // Linear learning-rate decay to 0 — stabilises the late epochs of this
        // non-convex (quartic-in-P) objective.
        let cur_lr = lr * (1.0 - ep as f32 / epochs as f32);
        let mut epoch_loss = 0.0f64;
        for _ in 0..steps {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
            let mut batch_loss = 0.0f32;
            for _ in 0..batch {
                let q = &qs[(rng.next_u64() as usize) % nq];
                let k = &ks[(rng.next_u64() as usize) % nk];
                let mut qk = 0.0f32;
                for j in 0..d {
                    qk += q[j] * k[j];
                }
                for row in 0..D_C {
                    let pr = &p[row * d..(row + 1) * d];
                    let mut sa = 0.0f32;
                    let mut sb = 0.0f32;
                    for j in 0..d {
                        sa += pr[j] * q[j];
                        sb += pr[j] * k[j];
                    }
                    a[row] = sa;
                    b[row] = sb;
                }
                let mut ab = 0.0f32;
                for row in 0..D_C {
                    ab += a[row] * b[row];
                }
                let r = qk - ab;
                batch_loss += r * r;
                let c = -2.0 * r;
                for row in 0..D_C {
                    let gr = &mut grad[row * d..(row + 1) * d];
                    let (ai, bi) = (a[row], b[row]);
                    for j in 0..d {
                        gr[j] += c * (bi * q[j] + ai * k[j]);
                    }
                }
            }
            let step = cur_lr / batch as f32;
            for (pi, gi) in p.iter_mut().zip(&grad) {
                *pi -= step * gi;
            }
            epoch_loss += batch_loss as f64;
        }
        history.push((epoch_loss / (steps * batch) as f64) as f32);
    }
    (p, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::slha_v2::{hamming_distance, LatentCodec};
    use crate::metrics::{dot, spearman};
    use crate::rope::{rope_copy, ROPE_BASE};

    fn run(decay: f32, whiten: bool) -> (f32, f32, f32) {
        let d = 160;
        let train = gen_keys(1, 600, d, 200, decay, 0.02);
        let model = LearnedModel::fit(&train, d, 7, whiten);

        let eval = gen_keys(2, 256, d, 200, decay, 0.02);
        let q = &gen_keys(3, 1, d, 200, decay, 0.02)[0];
        let q_coarse = model.query_coarse(q);
        let q_sign = model.sign_bits(q);

        let mut s_true = Vec::new();
        let mut s_hot = Vec::new();
        let mut s_warm = Vec::new();
        for (i, key) in eval.iter().enumerate() {
            s_true.push(dot(q, key));
            let hot = model.encode(key, i as u32, false);
            let mut warm = hot;
            warm.flags |= FLAG_WARM;
            s_hot.push(hot.compute_score(&q_coarse, &q_sign));
            s_warm.push(warm.compute_score(&q_coarse, &q_sign));
        }
        (
            model.captured_energy,
            spearman(&s_hot, &s_true),
            spearman(&s_warm, &s_true),
        )
    }

    #[test]
    fn pca_captures_dominant_subspace_and_hot_beats_warm() {
        // Default (non-whitened) latent — see `whitening_does_not_help`.
        let (captured, sp_hot, sp_warm) = run(0.95, false);
        assert!(captured > 0.8, "captured energy {captured} too low");
        assert!(sp_hot > 0.8, "HOT Spearman {sp_hot} too low with learned P");
        assert!(sp_hot + 0.03 >= sp_warm, "HOT {sp_hot} << WARM {sp_warm}");
    }

    #[test]
    fn whitening_does_not_help_score_preserving_int4() {
        // Empirical finding: whitening the PCA latent does NOT improve ranking
        // (it hurts). A single-scale INT4 already allocates resolution to the
        // high-variance components that dominate the score; whitening then
        // re-amplifies their quantisation error through the de-whitened query.
        // The score itself is identical either way (whitening is neutral on it).
        let (_, hot_whiten, _) = run(0.9, true);
        let (_, hot_naive, _) = run(0.9, false);
        assert!(
            hot_naive + 0.02 >= hot_whiten,
            "whitening unexpectedly helped: naive {hot_naive} vs whiten {hot_whiten}"
        );
    }

    #[test]
    fn grouped_int4_does_not_hurt_end_to_end() {
        // Per-group MX scaling halves latent reconstruction error (see the
        // slha_v2 unit test); end-to-end it must be at least as good as a single
        // scale. (The end-to-end gain is small here because the score is
        // dominated by high-variance components a single scale already handles.)
        let d = 160;
        let train = gen_keys(1, 600, d, 200, 0.93, 0.02);
        let model = LearnedModel::fit(&train, d, 7, false);
        let eval = gen_keys(2, 256, d, 200, 0.93, 0.02);
        let q = &gen_keys(3, 1, d, 200, 0.93, 0.02)[0];
        let q_coarse = model.query_coarse(q);
        let q_sign = model.sign_bits(q);

        let hot_sp = |codec: LatentCodec| {
            let mut st = Vec::new();
            let mut sh = Vec::new();
            for (i, key) in eval.iter().enumerate() {
                st.push(dot(q, key));
                sh.push(
                    model
                        .encode_with(key, i as u32, false, codec)
                        .compute_score(&q_coarse, &q_sign),
                );
            }
            spearman(&sh, &st)
        };
        let single = hot_sp(LatentCodec::Int4Single);
        let grouped = hot_sp(LatentCodec::Int4Grouped);
        assert!(
            grouped + 0.02 >= single,
            "grouped {grouped} worse than single {single}"
        );
    }

    #[test]
    fn attention_output_is_high_fidelity_and_hot_ge_warm() {
        // The softmax-weighted value output is far more robust to score error
        // than raw ranking. Cosine(out_true, out_hot) must be high, HOT >= WARM.
        use crate::attention::slha_v2::FLAG_WARM;
        use crate::metrics::{cosine, softmax_into};
        use crate::rng::Rng;

        let (d, dv, n) = (160usize, 48usize, 200usize);
        let train = gen_keys(1, 600, d, 200, 0.9, 0.02);
        let model = LearnedModel::fit(&train, d, 7, false);
        let keys = gen_keys(2, n, d, 200, 0.9, 0.02);
        let mut rng = Rng::new(5);
        let values: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut v = vec![0.0f32; dv];
                rng.fill_gaussian(&mut v);
                v
            })
            .collect();
        let tiles: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| model.encode(k, i as u32, false))
            .collect();

        let mut q = vec![0.0f32; d];
        rng.fill_gaussian(&mut q);
        let qc = model.query_coarse(&q);
        let qs = model.sign_bits(&q);
        let scale = 1.0 / (d as f32).sqrt();

        let s_true: Vec<f32> = keys.iter().map(|k| dot(&q, k)).collect();
        let s_hot: Vec<f32> = tiles.iter().map(|t| t.compute_score(&qc, &qs)).collect();
        let s_warm: Vec<f32> = tiles
            .iter()
            .map(|t| {
                let mut w = *t;
                w.flags |= FLAG_WARM;
                w.compute_score(&qc, &qs)
            })
            .collect();

        let agg = |s: &[f32]| -> Vec<f32> {
            let mut w = vec![0.0f32; n];
            softmax_into(s, scale, &mut w);
            let mut o = vec![0.0f32; dv];
            for (wi, v) in w.iter().zip(&values) {
                for j in 0..dv {
                    o[j] += wi * v[j];
                }
            }
            o
        };
        let ot = agg(&s_true);
        let ch = cosine(&ot, &agg(&s_hot));
        let cw = cosine(&ot, &agg(&s_warm));
        assert!(ch > 0.9, "HOT attention-output cosine {ch} too low");
        assert!(ch + 0.02 >= cw, "HOT {ch} < WARM {cw}");
    }

    #[test]
    fn train_projection_reduces_score_loss() {
        // Optimiser sanity: from a deliberately bad (random) projection, the
        // task-aware SGD must substantially reduce E[(⟨Q,K⟩ − ⟨PQ,PK⟩)²].
        // (The decisive "learned beats PCA" result — ~0.16 vs ~0.86 WARM
        // Spearman on clean data — is shown by the `learn_projection` example.)
        let d = 160;
        let gen = |seed: u64, n: usize| -> Vec<Vec<f32>> {
            let mut rng = Rng::new(seed);
            (0..n)
                .map(|_| {
                    let mut v = vec![0.0f32; d];
                    rng.fill_gaussian(&mut v);
                    v
                })
                .collect()
        };
        let qs = gen(1, 256);
        let ks = gen(2, 256);

        let mut rng = Rng::new(3);
        let mut p = vec![0.0f32; D_C * d];
        for x in p.iter_mut() {
            *x = rng.next_gaussian() * 0.05; // deliberately poor starting point
        }

        // Kept light (it runs in debug under CI); a wrong-sign / no-op gradient
        // would fail this even with a lenient threshold.
        let (_p, hist) = train_projection(&qs, &ks, p, 40, 2.0e-3, 64, 7);
        let (l0, l1) = (hist[0], hist[hist.len() - 1]);
        assert!(
            l1 < 0.85 * l0,
            "score loss {l0:.2} -> {l1:.2} did not drop enough"
        );
    }

    /// Plan axis **A2** — the headline result, in the regime QuIP# actually
    /// targets: a **strong common (outlier) direction blinds the plain
    /// sign-LSH** — the bits are dominated by `sign(Z·strong)` which is nearly
    /// identical across samples, so the Hamming distance carries little of the
    /// structured signal that drives the true `⟨E_q, E_j⟩` ranking. The
    /// randomised Hadamard transform spreads the common energy across all bits,
    /// so the per-sample structured component contributes comparably per bit,
    /// and the **binary core** (residual term alone) ranks materially better.
    ///
    /// NB: this is *not* the "independent outliers" regime — there the dominant
    /// signal already separates the samples and flattening only adds noise (the
    /// transform is neutral-to-harmful). Incoherence processing helps exactly
    /// when a dominant component would otherwise blind the hash.
    #[test]
    fn rht_improves_binary_core_on_outlier_blinded_residuals() {
        let d = 128; // power-of-two ⇒ d_pad == d, clean Z parity
        let mut rng = Rng::new(2026);

        // Only sign_bits is exercised, so the projection P is irrelevant; both
        // models share the same Z (same seed), differing only in the RHT.
        let p = vec![0.0f32; D_C * d];
        let m_plain = LearnedModel::from_projection(p.clone(), d, 42);
        let m_rht = LearnedModel::from_projection_with(p, d, 42, true);
        assert!(m_rht.rht.is_some(), "RHT not enabled");

        // Strong common direction on a few channels (the outlier) + a unique
        // low-rank structured signal that determines the ranking.
        let chans = [0usize, 1, 40, 41, 80];
        let mut common = vec![0.0f32; d];
        for &c in &chans {
            common[c] = 10.0;
        }
        let ndir = 3;
        let mut dirs = vec![vec![0.0f32; d]; ndir];
        for dir in dirs.iter_mut() {
            rng.fill_gaussian(dir);
            let nrm = dir.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in dir.iter_mut() {
                *x /= nrm;
            }
        }
        let mk = |rng: &mut Rng| -> Vec<f32> {
            let mut e = common.clone();
            for dir in &dirs {
                let a = 2.0 * rng.next_gaussian();
                for i in 0..d {
                    e[i] += a * dir[i];
                }
            }
            e
        };
        let n = 512;
        let residuals: Vec<Vec<f32>> = (0..n).map(|_| mk(&mut rng)).collect();
        let eq = mk(&mut rng);

        let s_true: Vec<f32> = residuals.iter().map(|e| dot(&eq, e)).collect();
        let binary_ranking = |m: &LearnedModel| -> Vec<f32> {
            let qs = m.sign_bits(&eq);
            residuals
                .iter()
                .map(|e| {
                    let es = m.sign_bits(e);
                    // d_s − 2·Hamming = signed dot of the two ±1 sign vectors.
                    D_S as f32 - 2.0 * hamming_distance(&qs, &es) as f32
                })
                .collect()
        };

        let sp_plain = spearman(&binary_ranking(&m_plain), &s_true);
        let sp_rht = spearman(&binary_ranking(&m_rht), &s_true);
        assert!(
            sp_rht >= sp_plain + 0.15,
            "RHT did not robustly improve the blinded binary core: \
             plain {sp_plain:.3} vs rht {sp_rht:.3} (need +0.15)"
        );
    }

    /// Plan axis **A2** — the orthogonality invariant that *does* hold
    /// universally: the RHT touches only the sign-LSH path, never the coarse
    /// latent, so **WARM** (coarse-only score) is identical with or without the
    /// transform. HOT can move either way — it improves on outlier-blinded
    /// residuals and can regress on well-conditioned ones (where the dominant
    /// structure is already in the coarse term) — so A2 is opt-in, applied
    /// where the residual has a dominant component, not by default.
    #[test]
    fn rht_preserves_warm_coarse_path() {
        let d = 160;
        let train = gen_keys(1, 600, d, 200, 0.9, 0.02);
        let model_plain = LearnedModel::fit(&train, d, 7, false);
        let model_rht = LearnedModel::fit_with(&train, d, 7, false, true);
        // Same PCA ⇒ same captured energy ⇒ same coarse base.
        assert_eq!(model_plain.captured_energy, model_rht.captured_energy);

        let eval = gen_keys(2, 256, d, 200, 0.9, 0.02);
        let q = &gen_keys(3, 1, d, 200, 0.9, 0.02)[0];

        let warm_spearman = |m: &LearnedModel| -> f32 {
            let qc = m.query_coarse(q);
            let qs = m.sign_bits(q); // unused by WARM, but keeps parity
            let mut s_true = Vec::new();
            let mut s_warm = Vec::new();
            for (i, key) in eval.iter().enumerate() {
                s_true.push(dot(q, key));
                let mut warm = m.encode(key, i as u32, false);
                warm.flags |= FLAG_WARM;
                s_warm.push(warm.compute_score(&qc, &qs));
            }
            spearman(&s_warm, &s_true)
        };

        let warm_plain = warm_spearman(&model_plain);
        let warm_rht = warm_spearman(&model_rht);
        // Coarse path is RHT-independent ⇒ WARM ranking is bit-for-bit the same.
        assert!(
            (warm_rht - warm_plain).abs() <= 1e-6,
            "WARM not preserved under RHT: {warm_plain:.6} -> {warm_rht:.6}"
        );
    }

    /// Plan axis **A1** — the robust ShadowKV mechanism, reproducible across
    /// seeds: RoPE mixes channels and destroys the low-rank structure of the
    /// keys, so a PCA captures markedly more energy on **pre-RoPE** keys than
    /// on **post-RoPE** keys at any rank where the pre-RoPE energy is
    /// concentrated. This is the measured root of the §7.8 "the projection is
    /// the ceiling" finding and the reason to project before rotation.
    ///
    /// Measured at `d = 160`, rank 32 — where the rank inflation is dramatic
    /// (pre-RoPE ≈ 99%, post-RoPE ≈ 68%) and the Jacobi eigendecomposition
    /// stays fast for CI. The full rank-`D_C` picture at `d = 256` is in
    /// `examples/pre_rope_projection.rs`.
    ///
    /// (The *consequence* — a robust WARM-ranking lift — is NOT assertable on
    /// this synthetic factor model: the gain is seed-dependent and within
    /// noise because the model is too compressible even post-RoPE. See that
    /// example for the honest, measured report; a robust lift needs real LLM
    /// key distributions, i.e. the Phase 3 / A7 integration.)
    #[test]
    fn rope_destroys_low_rank_projection_ceiling() {
        let (d, r, noise, n, span, rank) = (160, 8, 0.02, 256, 8u32, 32);
        for seed in [10u64, 11] {
            let k_pre = gen_keys(seed, n, d, r, 0.9, noise);
            // Long-context span: position j sits at j·span, so few samples still
            // exercise large RoPE angles (where the rank inflation shows).
            let positions: Vec<u32> = (0..n as u32).map(|i: u32| i * span).collect();
            let k_post: Vec<Vec<f32>> = k_pre
                .iter()
                .zip(&positions)
                .map(|(k, &p)| rope_copy(k, p, ROPE_BASE))
                .collect();
            let pre = captured_energy_at(&k_pre, d, rank);
            let post = captured_energy_at(&k_post, d, rank);
            assert!(
                pre > post + 0.20,
                "seed {seed}: RoPE did not inflate the rank \
                 (pre {pre:.3} vs post {post:.3} at rank {rank})"
            );
        }
    }

    #[test]
    fn fit_joint_with_no_queries_equals_fit_with() {
        let d = 160;
        let keys = gen_keys(3, 200, d, d, 0.9, 0.02);
        let a = LearnedModel::fit_with(&keys, d, 7, false, false);
        let b = LearnedModel::fit_joint(&keys, &[], d, 7, false, false);
        assert_eq!(a.projection(), b.projection(), "projection differs");
        assert_eq!(a.captured_energy, b.captured_energy);
    }

    /// Keys live in dims 0..130; queries carry most of their energy in dims
    /// 130..160 that the keys never touch. A keys-only PCA cannot keep those
    /// directions (its covariance is zero there); the joint fit must.
    #[test]
    fn fit_joint_keeps_query_only_directions() {
        let (d, n) = (160usize, 300usize);
        let mut rng = Rng::new(42);
        let mut keys = Vec::with_capacity(n);
        let mut queries = Vec::with_capacity(n);
        for _ in 0..n {
            let mut k = vec![0.0f32; d];
            rng.fill_gaussian(&mut k[..130]);
            keys.push(k);

            let mut q = vec![0.0f32; d];
            rng.fill_gaussian(&mut q[..20]);
            let mut tail = vec![0.0f32; 30];
            rng.fill_gaussian(&mut tail);
            for (i, t) in tail.iter().enumerate() {
                q[130 + i] = 3.0 * t;
            }
            queries.push(q);
        }
        // Mean captured query energy ||P·q||²/||q||² (scale = 1: latent = e·q).
        let captured_q = |m: &LearnedModel| -> f32 {
            let mut acc = 0.0f32;
            for q in &queries {
                let h = m.latent(q);
                let num: f32 = h.iter().map(|x| x * x).sum();
                let den: f32 = q.iter().map(|x| x * x).sum();
                acc += num / den;
            }
            acc / queries.len() as f32
        };
        let keys_only = LearnedModel::fit_with(&keys, d, 9, false, false);
        let joint = LearnedModel::fit_joint(&keys, &queries, d, 9, false, false);
        let (ko, jo) = (captured_q(&keys_only), captured_q(&joint));
        assert!(
            ko < 0.3,
            "keys-only PCA unexpectedly kept query energy: {ko}"
        );
        assert!(jo > 0.8, "joint fit failed to keep query energy: {jo}");
    }
}
