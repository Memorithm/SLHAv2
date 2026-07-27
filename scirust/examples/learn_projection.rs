//! Learned (task-aware) low-rank projection vs PCA.
//!
//! Run with:  `cargo run --example learn_projection --release`
//!
//! PCA picks the top-`D_C` directions of the **key** covariance — it ignores
//! the query distribution. When queries and keys emphasise *different*
//! subspaces, a projection trained to preserve the **score** `⟨Q,K⟩` beats PCA.
//!
//! Adversarial-but-clear setup: a shared factor basis split in two halves.
//! - Half A (factors 0..128): **high key variance**, low query weight.
//! - Half B (factors 128..256): moderate key variance, **high query weight**.
//!
//! PCA keeps half A (highest key variance) and almost misses the score, which
//! lives in half B. We evaluate **WARM** (coarse/latent only) to isolate the
//! projection from the 1-bit residual.

use scirust::attention::slha_v2::FLAG_WARM;
use scirust::learned::{train_projection, LearnedModel};
use scirust::metrics::{cosine, dot, softmax_into, spearman};
use scirust::rng::Rng;

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Modified Gram–Schmidt: make the factor basis orthonormal, so the key
/// covariance's eigenvectors are exactly the factors and PCA's top-`D_C` cut is
/// a clean split by `s_k`.
fn orthonormalize(factors: &mut [Vec<f32>], d: usize) {
    for i in 0..factors.len() {
        let (prev, rest) = factors.split_at_mut(i);
        let fi = &mut rest[0];
        for fj in prev.iter() {
            let dotij: f32 = fi.iter().zip(fj).map(|(a, b)| a * b).sum();
            for t in 0..d {
                fi[t] -= dotij * fj[t];
            }
        }
        normalize(fi);
    }
}

fn gen(seed: u64, n: usize, d: usize, factors: &[Vec<f32>], s: &[f32]) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let mut v = vec![0.0f32; d];
            for (j, fj) in factors.iter().enumerate() {
                let g = rng.next_gaussian() * s[j];
                for i in 0..d {
                    v[i] += g * fj[i];
                }
            }
            v
        })
        .collect()
}

fn attn_output(weights: &[f32], values: &[Vec<f32>], dv: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dv];
    for (w, v) in weights.iter().zip(values) {
        for j in 0..dv {
            out[j] += w * v[j];
        }
    }
    out
}

/// Returns (WARM Spearman, HOT Spearman, HOT output cosine).
fn evaluate(
    m: &LearnedModel,
    q_eval: &[Vec<f32>],
    k_eval: &[Vec<f32>],
    d: usize,
) -> (f32, f32, f32) {
    let dv = 64;
    let mut rng = Rng::new(123);
    let values: Vec<Vec<f32>> = (0..k_eval.len())
        .map(|_| {
            let mut v = vec![0.0f32; dv];
            rng.fill_gaussian(&mut v);
            v
        })
        .collect();
    let hot: Vec<_> = k_eval
        .iter()
        .enumerate()
        .map(|(i, k)| m.encode(k, i as u32, false))
        .collect();
    let scale = 1.0 / (d as f32).sqrt();

    let (mut warm_sp, mut hot_sp, mut cos) = (0.0f32, 0.0f32, 0.0f32);
    for q in q_eval {
        let qc = m.query_coarse(q);
        let qs = m.sign_bits(q);
        let s_true: Vec<f32> = k_eval.iter().map(|k| dot(q, k)).collect();
        let s_hot: Vec<f32> = hot.iter().map(|t| t.compute_score(&qc, &qs)).collect();
        let s_warm: Vec<f32> = hot
            .iter()
            .map(|t| {
                let mut w = *t;
                w.flags |= FLAG_WARM;
                w.compute_score(&qc, &qs)
            })
            .collect();
        warm_sp += spearman(&s_warm, &s_true);
        hot_sp += spearman(&s_hot, &s_true);

        let mut wt = vec![0.0f32; s_true.len()];
        let mut wh = vec![0.0f32; s_hot.len()];
        softmax_into(&s_true, scale, &mut wt);
        softmax_into(&s_hot, scale, &mut wh);
        cos += cosine(
            &attn_output(&wt, &values, dv),
            &attn_output(&wh, &values, dv),
        );
    }
    let n = q_eval.len() as f32;
    (warm_sp / n, hot_sp / n, cos / n)
}

fn main() {
    let d = 256;
    let r = d;
    let half = d / 2;

    let mut rng = Rng::new(1);
    let mut factors: Vec<Vec<f32>> = (0..r)
        .map(|_| {
            let mut f = vec![0.0f32; d];
            rng.fill_gaussian(&mut f);
            normalize(&mut f);
            f
        })
        .collect();
    orthonormalize(&mut factors, d);

    // Half A: high key variance, low query weight. Half B: the opposite.
    let s_k: Vec<f32> = (0..r).map(|j| if j < half { 1.0 } else { 0.5 }).collect();
    let s_q: Vec<f32> = (0..r).map(|j| if j < half { 0.1 } else { 1.0 }).collect();

    let k_train = gen(10, 2000, d, &factors, &s_k);
    let q_train = gen(11, 2000, d, &factors, &s_q);
    let k_eval = gen(20, 512, d, &factors, &s_k);
    let q_eval = gen(21, 64, d, &factors, &s_q);

    let seed = 0xC0FFEE;
    println!("== Projection APPRISE (task-aware) vs PCA ==\n");
    println!("  d={d}, D_C=128 ; clés fortes sur la moitié A, requêtes fortes sur la moitié B");
    println!("  (PCA garde A par variance ; le score vit dans B)\n");

    let pca = LearnedModel::fit(&k_train, d, seed, false);
    let epochs = 200;
    let (p_learned, hist) = train_projection(
        &q_train,
        &k_train,
        pca.projection().to_vec(), // warm-start from PCA (d inferred from its length)
        epochs,
        2.0e-3, // decays linearly to 0 inside train_projection
        64,
        7,
    );
    let learned = LearnedModel::from_projection(p_learned, d, seed);

    print!("  SGD score-loss :");
    for &e in &[0, epochs / 4, epochs / 2, 3 * epochs / 4, epochs - 1] {
        print!(" {:.2}", hist[e]);
    }
    println!(
        "  (époques 0/{}/{}/{}/{})\n",
        epochs / 4,
        epochs / 2,
        3 * epochs / 4,
        epochs - 1
    );

    let (pca_w, pca_h, pca_c) = evaluate(&pca, &q_eval, &k_eval, d);
    let (lr_w, lr_h, lr_c) = evaluate(&learned, &q_eval, &k_eval, d);

    println!(
        "  {:>10} | {:>12} | {:>11} | {:>11}",
        "", "WARM Spear", "HOT Spear", "sortie cos"
    );
    println!("  {}", "-".repeat(52));
    println!(
        "  {:>10} | {:>12.3} | {:>11.3} | {:>11.3}",
        "PCA", pca_w, pca_h, pca_c
    );
    println!(
        "  {:>10} | {:>12.3} | {:>11.3} | {:>11.3}",
        "Apprise", lr_w, lr_h, lr_c
    );

    println!(
        "\n  Lecture : WARM (coarse seul) isole la projection. PCA, qui optimise la\n  \
         reconstruction des clés, garde la moitié A (forte variance) et rate le\n  \
         score qui vit dans B -> WARM Spearman faible. La projection apprise sur\n  \
         le SCORE réalloue vers B. Le résidu 1-bit (HOT) rattrape une partie de\n  \
         l'écart, mais ne le supprime pas."
    );
}
