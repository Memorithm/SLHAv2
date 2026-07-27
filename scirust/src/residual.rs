//! Plan axis **A4** — multi-bit / multi-round sign-LSH residual, at a **fixed
//! 256-bit budget**.
//!
//! The §2.2 residual is 256 sign-LSH bits — **1 bit per hyperplane**. Two
//! levers from the literature improve its resolution without growing the
//! residual (the 128-byte tile invariant is sacred, so the total residual width
//! stays `D_S = 256` bits — we only change how those bits are *spent*):
//!
//! - **Multi-bit (quantised) residual** (QINCo / NSNQuant): fewer hyperplanes,
//!   `b` bits each. At high residual energy `ρ` the 1-bit sign saturates (the
//!   hash correlation is bounded); a 2-/4-bit quantised projection carries
//!   finer magnitude, so the residual term ranks better there.
//! - **Multi-round LSH** (Reformer): `K` independent 1-bit hashes of `D_S/K`
//!   bits, averaged. Reduces the per-round variance (and the false-negative rate
//!   Reformer targets for retrieval); at fixed total bits the scoring tradeoff
//!   is measured here, not assumed.
//!
//! Both are studied **above the tile** (like A1/A2): the kernel and the 128-byte
//! layout are untouched. The integration path is a **graduated Soft-Paging**
//! (HOT2 = full `b`-bit residual → HOT1 = keep only the MSB of each code, i.e.
//! the sign = the current 1-bit residual → WARM = coarse only): a bit-masking
//! reinterpretation of the same 32 bytes, O(1) like the existing WARM page-out.
//!
//! See: Reformer LSH (arXiv 2001.04451), RBE/BEBR (1802.06466, 2302.08714),
//! QINCo (ICML 2024), NSNQuant (2505.18231).

use crate::attention::slha_v2::{hamming_distance, D_S, RESIDUAL_WORDS};
use crate::rng::Rng;

/// A row-normalised Gaussian JL projection `Z ∈ ℝ^{n_planes × d}` (each row unit
/// norm). Shared by the 1-bit and `b`-bit schemes so they differ only in how
/// they spend the bits.
fn jl_projection(seed: u64, n_planes: usize, d: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut z = vec![0.0f32; n_planes * d];
    rng.fill_gaussian(&mut z);
    for p in 0..n_planes {
        let row = &mut z[p * d..(p + 1) * d];
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }
    z
}

/// **1-bit sign-LSH residual** (the §2.2 scheme, generalised to `n_bits`
/// hyperplanes). `dot_estimate = n_bits − 2·Hamming` is the signed dot of the
/// two ±1 sign vectors — the binary core of eq. (2.3).
pub struct BinaryResidual {
    z: Vec<f32>,
    n_bits: usize,
    d: usize,
}

impl BinaryResidual {
    /// `n_bits` hyperplanes over `R^d`. Use `D_S` for the §2.2 256-bit residual.
    pub fn new(seed: u64, n_bits: usize, d: usize) -> Self {
        Self {
            z: jl_projection(seed, n_bits, d),
            n_bits,
            d,
        }
    }

    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    fn project(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.d);
        let mut p = vec![0.0f32; self.n_bits];
        for i in 0..self.n_bits {
            let row = &self.z[i * self.d..(i + 1) * self.d];
            let mut acc = 0.0f32;
            for j in 0..self.d {
                acc += row[j] * v[j];
            }
            p[i] = acc;
        }
        p
    }

    /// Pack `sign(Z·v)` into bits (bit = 1 iff the projection is negative).
    pub fn encode(&self, v: &[f32]) -> Vec<u64> {
        let p = self.project(v);
        let words = self.n_bits.div_ceil(64);
        let mut out = vec![0u64; words];
        for (i, &pi) in p.iter().enumerate() {
            if pi < 0.0 {
                out[i >> 6] |= 1u64 << (i & 63);
            }
        }
        out
    }

    /// Signed-dot estimate `n_bits − 2·Hamming(a, b)` (the binary core).
    pub fn dot_estimate(&self, a: &[u64], b: &[u64]) -> f32 {
        // Hamming over the min number of full shared words; extra tail bits in
        // the last word are zero-padded by encode, so they agree trivially.
        let words = self.n_bits.div_ceil(64);
        let mut h = 0u32;
        for w in 0..words {
            let aw = if w < a.len() { a[w] } else { 0 };
            let bw = if w < b.len() { b[w] } else { 0 };
            h += (aw ^ bw).count_ones();
        }
        self.n_bits as f32 - 2.0 * h as f32
    }
}

/// Symmetric uniform quantiser over `[-L, L]` with `2^bits` levels (centred,
/// no zero level). `quant(p) → code`, `dequant(code) → level`.
struct UniformQuant {
    bits: u32,
    levels: Vec<f32>, // length 2^bits, ascending
    scale: f32,       // multiply dequant by `scale` to recover project units
}

impl UniformQuant {
    fn new(bits: u32, scale: f32) -> Self {
        let n = 1usize << bits;
        // Span ±L = 1.5 standard deviations (residual projections are ~N(0,1)
        // after dividing by `scale`). Centred levels: step = 2L/n.
        let l = 1.5f32;
        let step = 2.0 * l / n as f32;
        let levels: Vec<f32> = (0..n).map(|i| -l + step * (i as f32 + 0.5)).collect();
        UniformQuant {
            bits,
            levels,
            scale,
        }
    }

    fn quant(&self, p: f32) -> u32 {
        let n = self.levels.len() as i32;
        // nearest level by index. Levels are *centred*: `levels[i] = -L +
        // step·(i+0.5)`, so cell `i` covers `[-L+step·i, -L+step·(i+1))` and the
        // nearest-level index is `floor((p+L)/step)` — NOT `round`, which would
        // shift by half a step and pick the level *above* the nearest one (and
        // flips the sign at `bits=1`). Clamp to the valid range.
        let l = 1.5f32;
        let step = 2.0 * l / n as f32;
        let idx = ((p + l) / step).floor() as i32;
        idx.clamp(0, n - 1) as u32
    }

    fn dequant(&self, code: u32) -> f32 {
        self.levels[code as usize] * self.scale
    }
}

/// **`b`-bit quantised residual**: `n_planes = D_S / bits` hyperplanes, each
/// projection quantised to `2^bits` levels. `dot_estimate = Σ dequant(q_i)·
/// dequant(k_i)` estimates `(n_planes/d)·⟨q, k⟩` — ranking-preserving (the
/// constant factor is the same for all keys, so Spearman is unaffected).
pub struct QuantResidual {
    z: Vec<f32>,
    n_planes: usize,
    d: usize,
    q: UniformQuant,
}

impl QuantResidual {
    /// `bits` bits per hyperplane, `n_planes = D_S / bits` hyperplanes (so the
    /// total residual width stays `D_S = 256`). `sigma` is the residual
    /// per-dimension std used to normalise projections to ~N(0,1) before
    /// quantising — estimate it once from the training residuals (it equals the
    /// std of `z·v` for unit-norm `z`, since `Var(z·v) = σ_v²`).
    pub fn new(seed: u64, bits: u32, d: usize, sigma: f32) -> Self {
        assert!((1..=8).contains(&bits), "bits per plane must be 1..=8");
        assert!(
            D_S.is_multiple_of(bits as usize),
            "D_S must be divisible by bits for a clean fixed-bit budget"
        );
        let n_planes = D_S / bits as usize;
        Self {
            z: jl_projection(seed, n_planes, d),
            n_planes,
            d,
            q: UniformQuant::new(bits, sigma),
        }
    }

    pub fn n_planes(&self) -> usize {
        self.n_planes
    }

    pub fn bits(&self) -> u32 {
        self.q.bits
    }

    /// Encode `v` to one quantised code per hyperplane.
    pub fn encode(&self, v: &[f32]) -> Vec<u32> {
        debug_assert_eq!(v.len(), self.d);
        let mut out = vec![0u32; self.n_planes];
        for i in 0..self.n_planes {
            let row = &self.z[i * self.d..(i + 1) * self.d];
            let mut acc = 0.0f32;
            for j in 0..self.d {
                acc += row[j] * v[j];
            }
            out[i] = self.q.quant(acc / self.q.scale);
        }
        out
    }

    /// Dot estimate `Σ_i dequant(q_i)·dequant(k_i)` ≈ `(n_planes/d)·⟨q, k⟩`.
    pub fn dot_estimate(&self, a: &[u32], b: &[u32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.n_planes {
            s += self.q.dequant(a[i]) * self.q.dequant(b[i]);
        }
        s
    }
}

/// **Multi-round 1-bit LSH** (Reformer): `K` independent `BinaryResidual`s of
/// `D_S / K` bits each. The combined estimate is the mean of the per-round
/// signed-dot estimates — variance-reduced at fixed total bits (`D_S`).
pub struct MultiRoundResidual {
    rounds: Vec<BinaryResidual>,
}

impl MultiRoundResidual {
    /// `k_rounds` independent 1-bit hashes, `D_S / k_rounds` bits each.
    pub fn new(seed: u64, k_rounds: usize, d: usize) -> Self {
        assert!(k_rounds >= 1 && D_S.is_multiple_of(k_rounds));
        let bits_per_round = D_S / k_rounds;
        let rounds: Vec<BinaryResidual> = (0..k_rounds)
            .map(|i| BinaryResidual::new(seed.wrapping_add((i as u64) * 0x9E37), bits_per_round, d))
            .collect();
        MultiRoundResidual { rounds }
    }

    pub fn k_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Per-round packed bits, one `Vec<u64>` per round.
    pub fn encode(&self, v: &[f32]) -> Vec<Vec<u64>> {
        self.rounds.iter().map(|r| r.encode(v)).collect()
    }

    /// Mean of the per-round `bits_per_round − 2·Hamming` estimates.
    pub fn dot_estimate(&self, a: &[Vec<u64>], b: &[Vec<u64>]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.rounds.len() {
            s += self.rounds[i].dot_estimate(&a[i], &b[i]);
        }
        s / self.rounds.len() as f32
    }
}

/// Convenience: the §2.2 reference scheme — 1-bit, `D_S` hyperplanes, packed
/// into the tile's `RESIDUAL_WORDS` words (so it interoperates with
/// [`crate::attention::slha_v2::hamming_distance`]).
pub fn binary_residual_tile(seed: u64, d: usize) -> BinaryResidual {
    assert_eq!(D_S, RESIDUAL_WORDS * 64);
    BinaryResidual::new(seed, D_S, d)
}

/// Spearman-style ranking check is left to the example/tests via
/// `metrics::spearman`; here we only expose the estimators. The Hamming helper
/// below mirrors the tile kernel for the 256-bit, 4-word case so a `BinaryResidual`
/// with `n_bits == D_S` agrees with the SIMD path bit-for-bit.
pub fn binary_dot_via_tile_kernel(a: &[u64; RESIDUAL_WORDS], b: &[u64; RESIDUAL_WORDS]) -> f32 {
    D_S as f32 - 2.0 * hamming_distance(a, b) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{dot, spearman};
    use crate::rng::Rng;

    /// The 1-bit estimator is monotone in the true dot: a clearly-positive true
    /// dot gives a larger estimate than a clearly-negative one, robustly.
    #[test]
    fn binary_estimate_ranks_the_true_dot() {
        let d = 128;
        let r = BinaryResidual::new(7, D_S, d);
        let mut rng = Rng::new(11);
        // A fixed query; keys = query·a + orthogonal noise, sweeping a from
        // negative to positive so the true dot spans both signs.
        let mut q = vec![0.0f32; d];
        rng.fill_gaussian(&mut q);
        let mut noise = vec![0.0f32; d];
        rng.fill_gaussian(&mut noise);
        // Make `noise` orthogonal to `q`.
        let proj = dot(&q, &noise) / dot(&q, &q);
        for i in 0..d {
            noise[i] -= proj * q[i];
        }
        let nrm = dot(&noise, &noise).sqrt().max(1e-9);
        for x in noise.iter_mut() {
            *x /= nrm;
        }
        let qn = dot(&q, &q).sqrt();
        let qb = r.encode(&q);
        let mut true_dot = Vec::new();
        let mut est = Vec::new();
        for i in 0..64 {
            let a = -1.0 + 2.0 * (i as f32) / 63.0;
            let mut k = vec![0.0f32; d];
            for j in 0..d {
                k[j] = a * q[j] / qn + 3.0 * noise[j];
            }
            true_dot.push(dot(&q, &k));
            est.push(r.dot_estimate(&qb, &r.encode(&k)));
        }
        let sp = spearman(&est, &true_dot);
        assert!(
            sp > 0.8,
            "binary estimate should rank the true dot, sp={sp}"
        );
    }

    /// The generalised `BinaryResidual` with `n_bits == D_S` agrees bit-for-bit
    /// with the tile-kernel Hamming path (256-bit, 4 words).
    #[test]
    fn binary_residual_matches_tile_kernel_at_d_s() {
        let d = 128;
        let r = binary_residual_tile(0x5EED, d);
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let mut a = vec![0.0f32; d];
            let mut b = vec![0.0f32; d];
            rng.fill_gaussian(&mut a);
            rng.fill_gaussian(&mut b);
            let ea = r.encode(&a);
            let eb = r.encode(&b);
            // encode returns exactly RESIDUAL_WORDS words at n_bits == D_S.
            let mut wa = [0u64; RESIDUAL_WORDS];
            let mut wb = [0u64; RESIDUAL_WORDS];
            wa.copy_from_slice(&ea);
            wb.copy_from_slice(&eb);
            let via_struct = r.dot_estimate(&ea, &eb);
            let via_kernel = binary_dot_via_tile_kernel(&wa, &wb);
            assert!(
                (via_struct - via_kernel).abs() <= 1e-6,
                "struct {via_struct} != kernel {via_kernel}"
            );
        }
    }

    /// `BinaryResidual::dot_estimate` must Hamming-compare only the first
    /// `n_bits` bits even when `n_bits` is **not** a multiple of 64. `encode`
    /// zero-pads the tail word; the struct path XORs full words, so the padding
    /// must agree trivially (zero in both) and not inflate the Hamming count.
    /// Pins the tail-padding contract.
    #[test]
    fn binary_residual_tail_word_padding_when_n_bits_not_multiple_of_64() {
        let d = 96;
        // 200 bits ⇒ 3 full words + a 200−192 = 8-bit tail in word 3.
        let n_bits = 200;
        assert_eq!(n_bits % 64, 8, "guard: pick a non-multiple of 64");
        let r = BinaryResidual::new(0xA11CE, n_bits, d);
        let mut rng = Rng::new(99);
        for _ in 0..256 {
            let mut a = vec![0.0f32; d];
            let mut b = vec![0.0f32; d];
            rng.fill_gaussian(&mut a);
            rng.fill_gaussian(&mut b);
            let ea = r.encode(&a);
            let eb = r.encode(&b);
            // Brute-force Hamming over ONLY the first n_bits bits.
            let mut h = 0u32;
            for i in 0..n_bits {
                let ba = (ea[i >> 6] >> (i & 63)) & 1;
                let bb = (eb[i >> 6] >> (i & 63)) & 1;
                if ba != bb {
                    h += 1;
                }
            }
            let brute = n_bits as f32 - 2.0 * h as f32;
            let via_struct = r.dot_estimate(&ea, &eb);
            assert!(
                (via_struct - brute).abs() <= 1e-6,
                "n_bits={n_bits}: struct {via_struct} != brute {brute}"
            );
        }
    }

    /// `QuantResidual` keeps the fixed-bit budget: `n_planes · bits == D_S`.
    #[test]
    fn quant_residual_keeps_fixed_bit_budget() {
        for bits in [1u32, 2, 4, 8] {
            let q = QuantResidual::new(1, bits, 128, 1.0);
            assert_eq!(q.n_planes() * q.bits() as usize, D_S, "bits={bits}");
        }
    }

    /// `UniformQuant::quant` must pick the **nearest** level (the level whose
    /// value is closest to `p`), for every bit width — and, for `bits=1`, preserve
    /// the sign of `p` within the ±L span. This pins the half-step bug: a centred
    /// grid needs `floor`, not `round` (round shifts by half a cell and picks the
    /// level above the nearest one, flipping the sign at `bits=1`).
    #[test]
    fn quant_picks_nearest_level_and_preserves_sign() {
        for bits in [1u32, 2, 4, 8] {
            let q = UniformQuant::new(bits, 1.0);
            // Dense sweep across and beyond the [-L, L] span (in normalised units).
            let mut i = -300;
            while i <= 300 {
                let p = i as f32 * 0.01;
                let code = q.quant(p);
                let picked = q.levels[code as usize];
                // The picked level must be a nearest level (no other level is
                // strictly closer). Ties (exactly on a cell boundary) are allowed
                // to round either way, so we require ≤, not <.
                for (j, &lvl) in q.levels.iter().enumerate() {
                    assert!(
                        (picked - p).abs() <= (lvl - p).abs() + 1e-6,
                        "bits={bits} p={p}: picked level {picked} (code {code}) is farther than level {lvl} (code {j})"
                    );
                }
                if bits == 1 && p.abs() < 1.5 {
                    // Within the span the 1-bit code must carry the sign of p
                    // (the whole point of the sign-LSH residual at bits=1).
                    assert!(
                        p == 0.0 || picked.signum() == p.signum(),
                        "bits=1 p={p}: sign flipped (picked {picked})"
                    );
                }
                i += 1;
            }
        }
    }

    /// The quantised estimator is ranking-preserving (monotone in the true dot)
    /// — the precondition for it to be a usable residual term.
    #[test]
    fn quant_estimate_ranks_the_true_dot() {
        let d = 128;
        let mut rng = Rng::new(21);
        // `base` kept at full Gaussian norm so its projection onto a unit-norm
        // random `z` is ~N(0,1) — a usable signal. (Normalising `base` to unit
        // norm would shrink the projection to O(1/√d) and bury it.)
        let mut base = vec![0.0f32; d];
        rng.fill_gaussian(&mut base);
        let mut keys = Vec::new();
        for i in 0..64 {
            let a = -1.0 + 2.0 * (i as f32) / 63.0;
            let mut k = vec![0.0f32; d];
            for j in 0..d {
                k[j] = a * base[j] + 0.1 * rng.next_gaussian();
            }
            keys.push(k);
        }
        // Calibrate `sigma` = per-dim std of the residual population so the
        // projections land in the quantiser's [-1.5, 1.5] window.
        let mut sum2 = 0.0f32;
        for k in &keys {
            for &x in k {
                sum2 += x * x;
            }
        }
        let sigma = (sum2 / (keys.len() * d) as f32).sqrt();
        let q = QuantResidual::new(9, 2, d, sigma);
        let qb = q.encode(&base);
        let mut true_dot = Vec::new();
        let mut est = Vec::new();
        for k in &keys {
            true_dot.push(dot(&base, k));
            est.push(q.dot_estimate(&qb, &q.encode(k)));
        }
        let sp = spearman(&est, &true_dot);
        assert!(sp > 0.7, "quant estimate should rank the true dot, sp={sp}");
    }

    /// `QuantResidual` at `bits=8` (the widest code, `D_S/8 = 32` planes) still
    /// ranks the true dot — a weak, seed-robust floor that catches a sign-flip
    /// or a quantiser collapse (e.g. the half-step bug regressing the MSB),
    /// without over-claiming a ranking *gain* (the plan's honest stance: fewer
    /// planes cost directional resolution).
    #[test]
    fn quant_estimate_ranks_the_true_dot_at_bits8() {
        let d = 128;
        let mut rng = Rng::new(77);
        let mut base = vec![0.0f32; d];
        rng.fill_gaussian(&mut base);
        let mut keys = Vec::new();
        for i in 0..64 {
            let a = -1.0 + 2.0 * (i as f32) / 63.0;
            let mut k = vec![0.0f32; d];
            for j in 0..d {
                k[j] = a * base[j] + 0.1 * rng.next_gaussian();
            }
            keys.push(k);
        }
        let mut sum2 = 0.0f32;
        for k in &keys {
            for &x in k {
                sum2 += x * x;
            }
        }
        let sigma = (sum2 / (keys.len() * d) as f32).sqrt();
        let q = QuantResidual::new(9, 8, d, sigma);
        let qb = q.encode(&base);
        let mut true_dot = Vec::new();
        let mut est = Vec::new();
        for k in &keys {
            true_dot.push(dot(&base, k));
            est.push(q.dot_estimate(&qb, &q.encode(k)));
        }
        let sp = spearman(&est, &true_dot);
        assert!(
            sp > 0.3,
            "8-bit quant estimate should rank the true dot (weak floor), sp={sp}"
        );
    }

    /// Multi-round averaging is ranking-preserving and uses the fixed budget.
    #[test]
    fn multiround_keeps_budget_and_ranks() {
        let d = 128;
        let mr = MultiRoundResidual::new(5, 4, d);
        assert_eq!(mr.k_rounds() * (D_S / 4), D_S);
        let mut rng = Rng::new(2);
        let mut base = vec![0.0f32; d];
        rng.fill_gaussian(&mut base);
        let nrm = dot(&base, &base).sqrt().max(1e-9);
        for x in base.iter_mut() {
            *x /= nrm;
        }
        let mb = mr.encode(&base);
        let mut true_dot = Vec::new();
        let mut est = Vec::new();
        for i in 0..64 {
            let a = -1.0 + 2.0 * (i as f32) / 63.0;
            let mut k = vec![0.0f32; d];
            for j in 0..d {
                k[j] = a * base[j] + 0.3 * rng.next_gaussian();
            }
            true_dot.push(dot(&base, &k));
            est.push(mr.dot_estimate(&mb, &mr.encode(&k)));
        }
        let sp = spearman(&est, &true_dot);
        assert!(sp > 0.7, "multi-round should rank the true dot, sp={sp}");
    }

    /// `MultiRoundResidual` with `k=1` is a single `D_S`-bit hash ⇒ it must
    /// agree **bit-for-bit** with a `BinaryResidual::new(seed, D_S, d)` built
    /// from the same seed (k=1 uses `seed.wrapping_add(0)` = `seed`). Pins the
    /// degenerate-round equivalence and the seed-derivation contract.
    #[test]
    fn multiround_k1_equals_binary_residual_d_s() {
        let d = 128;
        let mr = MultiRoundResidual::new(0xC0FFEE, 1, d);
        let bin = BinaryResidual::new(0xC0FFEE, D_S, d);
        let mut rng = Rng::new(5);
        for _ in 0..64 {
            let mut a = vec![0.0f32; d];
            let mut b = vec![0.0f32; d];
            rng.fill_gaussian(&mut a);
            rng.fill_gaussian(&mut b);
            let ma = mr.encode(&a);
            let mb = mr.encode(&b);
            // k=1 ⇒ exactly one round, packed into RESIDUAL_WORDS words.
            assert_eq!(ma.len(), 1);
            let via_mr = mr.dot_estimate(&ma, &mb);
            let via_bin = bin.dot_estimate(&bin.encode(&a), &bin.encode(&b));
            assert!(
                (via_mr - via_bin).abs() <= 1e-6,
                "k=1 must equal BinaryResidual(D_S): mr={via_mr} bin={via_bin}"
            );
        }
    }

    /// `MultiRoundResidual::new` panics when `D_S` is not divisible by
    /// `k_rounds` (the fixed-bit budget is sacred). Pins the divisibility
    /// contract — e.g. `k=3` is rejected (256 is not a multiple of 3).
    #[test]
    #[should_panic]
    fn multiround_rejects_non_divisor_k() {
        // 256 % 3 != 0 ⇒ must panic in the constructor.
        let _ = MultiRoundResidual::new(1, 3, 128);
    }

    /// Plan axis **A4** — the *robust* headline: at a fixed 256-bit residual
    /// budget, a `b`-bit quantised residual has a markedly lower dot-estimate
    /// **relative-L2 error** (`rel_l2 = ||est−true||/||true||`) than the 1-bit
    /// sign-LSH, across residual energy levels and seeds (3 decays × 6 seeds =
    /// **18 combos**). Multi-bit buys **magnitude** resolution (the plan's
    /// "meilleur HOT à rho élevé" — at high residual energy the 1-bit sign
    /// saturates and loses magnitude). This is robust; the *ranking* (Spearman)
    /// gain is NOT (see the `multibit_residual` example for the honest,
    /// seed-fragile report). The enforced bound is the one stated below
    /// (`THRESHOLD`); the *typical* reduction is larger and grows at high-ρ.
    #[test]
    fn multibit_robustly_reduces_dot_estimate_rel_l2() {
        let d = 128;
        // Enforced bound: a < THRESHOLD·b guarantees a reduction factor >
        // 1/THRESHOLD. Set to the worst robust ratio observed across the 18
        // combos (with margin) — the test asserts ONLY this, not the larger
        // typical/peak figures the CHANGELOG reports separately.
        const THRESHOLD: f32 = 0.30; // ⇒ enforced reduction > ~3.3×
        for &decay in &[0.99f32, 0.95, 0.80] {
            for seed in [10u64, 11, 12, 13, 14, 15] {
                let train = crate::learned::gen_keys(seed, 512, d, 256, decay, 0.02);
                let eval = crate::learned::gen_keys(seed + 100, 512, d, 256, decay, 0.02);
                let q = &crate::learned::gen_keys(seed + 200, 1, d, 256, decay, 0.02)[0];

                let mut sum2 = 0.0f32;
                let mut n = 0usize;
                for r in &train {
                    for &x in r {
                        sum2 += x * x;
                        n += 1;
                    }
                }
                let sigma = (sum2 / n as f32).sqrt();

                let b1 = BinaryResidual::new(seed, D_S, d);
                let b2 = QuantResidual::new(seed, 2, d, sigma);
                let b4 = QuantResidual::new(seed, 4, d, sigma);
                let q1 = b1.encode(q);
                let q2 = b2.encode(q);
                let q4 = b4.encode(q);

                let mut s_true = Vec::new();
                let mut s1 = Vec::new();
                let mut s2 = Vec::new();
                let mut s4 = Vec::new();
                for e in &eval {
                    s_true.push(dot(q, e));
                    s1.push(b1.dot_estimate(&q1, &b1.encode(e)));
                    s2.push(b2.dot_estimate(&q2, &b2.encode(e)));
                    s4.push(b4.dot_estimate(&q4, &b4.encode(e)));
                }
                // The metric is relative-L2 error (||est−true||/||true||), NOT
                // mean-squared error — named accordingly.
                let rel1 = crate::metrics::rel_l2(&s_true, &s1);
                let rel2 = crate::metrics::rel_l2(&s_true, &s2);
                let rel4 = crate::metrics::rel_l2(&s_true, &s4);
                // 2-bit and 4-bit both beat 1-bit on magnitude, robustly. The
                // 1-bit error is large (the sign carries no magnitude); multi-bit
                // quantises it. The enforced reduction is > ~3.3× at every
                // (decay, seed) probed; the typical reduction is larger and
                // reaches ×200+ at high-ρ (see the example, not asserted here).
                // (No Spearman assertion here — that gain is seed-fragile: fewer
                // planes cost directional resolution.)
                assert!(
                    rel2 < rel1 * THRESHOLD,
                    "decay {decay} seed {seed}: 2-bit rel-L2 {rel2:.2} not < {THRESHOLD}·1-bit {rel1:.2}"
                );
                assert!(
                    rel4 < rel1 * THRESHOLD,
                    "decay {decay} seed {seed}: 4-bit rel-L2 {rel4:.2} not < {THRESHOLD}·1-bit {rel1:.2}"
                );
            }
        }
    }
}
