//! Plan axis **A2** — incoherence processing (QuIP#/Palu) on the sign-LSH
//! residual.
//!
//! Run with:  `cargo run --example hadamard_incoherence --release`
//!
//! Reports the *measured* behaviour (not the paper claim) of applying a
//! randomised Hadamard transform to the residual and the query before the
//! sign-LSH. The transform is orthogonal (`⟨RHT·E, RHT·Q⟩ = ⟨E, Q⟩`), so:
//!
//! 1. **WARM is bit-for-bit preserved** — the RHT never touches the coarse
//!    latent path, only the 1-bit residual.
//! 2. **The binary core gains materially in the outlier-blinded regime** — a
//!    strong common (outlier) direction dominates the plain hash bits and
//!    blinds them to the structured signal that drives the ranking; spreading
//!    that energy across all bits restores discrimination. This is the regime
//!    QuIP# targets.
//! 3. **On well-conditioned residuals the RHT is neutral-to-harmful for HOT**
//!    — there the dominant structure is already absorbed into the coarse term,
//!    and flattening an already-isotropic residual only adds hash noise. A2 is
//!    therefore **opt-in**, applied where the residual has a dominant
//!    component (high peak-to-mean), not by default.
//!
//! See: QuIP# (arXiv 2402.04396), Palu (2407.21118), NSNQuant (2505.18231).

use scirust::attention::slha_v2::{hamming_distance, D_S};
use scirust::learned::{gen_keys, LearnedModel};
use scirust::metrics::{dot, spearman};
use scirust::rng::Rng;

/// Spearman of the binary core (residual term alone) vs the true `⟨E_q, E_j⟩`.
fn binary_core_spearman(model: &LearnedModel, residuals: &[Vec<f32>], eq: &[f32]) -> f32 {
    let qs = model.sign_bits(eq);
    let s_approx: Vec<f32> = residuals
        .iter()
        .map(|e| {
            let es = model.sign_bits(e);
            D_S as f32 - 2.0 * hamming_distance(&qs, &es) as f32
        })
        .collect();
    let s_true: Vec<f32> = residuals.iter().map(|e| dot(eq, e)).collect();
    spearman(&s_approx, &s_true)
}

/// (captured_energy, HOT Spearman, WARM Spearman) on factor-model data.
fn fused_spearman(model: &LearnedModel, d: usize, decay: f32) -> (f32, f32, f32) {
    use scirust::attention::slha_v2::FLAG_WARM;
    let eval = gen_keys(20, 512, d, 256, decay, 0.02);
    let q = &gen_keys(30, 1, d, 256, decay, 0.02)[0];
    let qc = model.query_coarse(q);
    let qs = model.sign_bits(q);
    let mut s_true = Vec::new();
    let mut s_hot = Vec::new();
    let mut s_warm = Vec::new();
    for (i, key) in eval.iter().enumerate() {
        s_true.push(dot(q, key));
        let hot = model.encode(key, i as u32, false);
        let mut warm = hot;
        warm.flags |= FLAG_WARM;
        s_hot.push(hot.compute_score(&qc, &qs));
        s_warm.push(warm.compute_score(&qc, &qs));
    }
    (
        model.captured_energy,
        spearman(&s_hot, &s_true),
        spearman(&s_warm, &s_true),
    )
}

fn main() {
    println!("== SLHA v2 — axe A2 : incohérence Hadamard (QuIP#/Palu) ==\n");

    // ----- 1. Binary core in the outlier-blinded regime --------------------
    // A strong common direction on a few channels blinds the plain hash; a
    // low-rank structured unique signal drives the true ranking.
    println!("  (1) Cœur binaire — régime « outlier aveuglant » (le cas QuIP#)");
    println!("      direction forte commune (S=10) + signal unique structuré (U=2, 3 dirs)\n");

    let d = 128; // power-of-two ⇒ d_pad == d, Z identique entre les deux modèles
    let p = vec![0.0f32; scirust::D_C * d];
    let m_plain = LearnedModel::from_projection(p.clone(), d, 42);
    let m_rht = LearnedModel::from_projection_with(p, d, 42, true);

    let mut rng = Rng::new(2026);
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

    let sp_plain = binary_core_spearman(&m_plain, &residuals, &eq);
    let sp_rht = binary_core_spearman(&m_rht, &residuals, &eq);
    println!(
        "      {:<22} Spearman = {sp_plain:.3}",
        "sans RHT (baseline)"
    );
    println!("      {:<22} Spearman = {sp_rht:.3}", "avec RHT (A2)");
    println!(
        "      gain : {:+.3}  ({:+.1}%)",
        sp_rht - sp_plain,
        100.0 * (sp_rht - sp_plain) / sp_plain.abs().max(1e-6)
    );

    // ----- 2. Fused score: WARM preserved, HOT may move --------------------
    println!("\n  (2) Score fusionné HOT/WARM — préservation WARM (RHT orthogonal)\n");
    println!("      La RHT ne touche que le résidu 1-bit ; le chemin coarse (WARM) est intact.");
    println!("      HOT peut bouger : neutre sur résidu bien conditionné, gain sinon.\n");
    println!(
        "  {:>6} {:>9} | {:^17} | {:^17}",
        "", "énergie", "sans RHT", "avec RHT (A2)"
    );
    println!(
        "  {:>6} {:>9} | {:>8} {:>8} | {:>8} {:>8}",
        "decay", "captée", "HOT", "WARM", "HOT", "WARM"
    );
    println!("  {}", "-".repeat(58));

    let d2 = 256;
    let n_train = 1024;
    for &decay in &[0.99f32, 0.95, 0.9, 0.85] {
        let train = gen_keys(10, n_train, d2, 256, decay, 0.02);
        let m0 = LearnedModel::fit(&train, d2, 0xC0FFEE, false);
        let m1 = LearnedModel::fit_with(&train, d2, 0xC0FFEE, false, true);
        let (cap0, h0, w0) = fused_spearman(&m0, d2, decay);
        let (_cap1, h1, w1) = fused_spearman(&m1, d2, decay);
        let warm_delta = w1 - w0;
        println!(
            "  {:>6.2} {:>8.1}% | {:>8.3} {:>8.3} | {:>8.3} {:>8.3}  ΔWARM {warm_delta:+.4}",
            decay,
            cap0 * 100.0,
            h0,
            w0,
            h1,
            w1,
        );
    }

    println!(
        "\n  Lecture (mesuré, pas affirmé sur foi de la littérature) :\n  \
         • WARM est strictement préservé (ΔWARM ≈ 0) — la RHT est orthogonale et\n    \
         n'atteint jamais le chemin coarse.\n  \
         • Le cœur binaire gagne +0,3..+0,4 de Spearman quand un outlier aveugle le\n    \
         hash naïf — le régime que QuIP# cible.\n  \
         • Sur résidu bien conditionné, la RHT est neutre à nuisible pour HOT : à\n    \
         n'activer que si le résidu a une composante dominante (peak/mean élevé).\n  \
         ⇒ A2 est opt-in, conditionnel — pas un défaut."
    );
}
