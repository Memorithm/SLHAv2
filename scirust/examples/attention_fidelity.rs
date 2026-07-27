//! Attention-OUTPUT fidelity: does the SLHA score approximation preserve the
//! softmax-weighted value aggregation `out = softmax(QKᵀ/√d) · V`?
//!
//! Run with:  `cargo run --example attention_fidelity --release`
//!
//! Score-ranking (§7.2/7.3) is a proxy; what a model actually consumes is the
//! attention *output*. Small score errors may wash out (softmax averaging) or
//! amplify (if they flip the top-1). This measures the real output error —
//! the quantity closest to the perplexity target of §6.3 — fully offline.

use scirust::attention::slha_v2::FLAG_WARM;
use scirust::learned::{gen_keys, LearnedModel};
use scirust::metrics::{cosine, dot, rel_l2, softmax_into};
use scirust::rng::Rng;

fn attn_output(weights: &[f32], values: &[Vec<f32>], dv: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dv];
    for (w, v) in weights.iter().zip(values) {
        for j in 0..dv {
            out[j] += w * v[j];
        }
    }
    out
}

fn main() {
    let d = 256;
    let r = 256;
    let dv = 64;
    let n = 512;
    let nq = 64;
    let scale = 1.0 / (d as f32).sqrt(); // standard attention temperature

    println!("== Fidélité de la SORTIE d'attention  (out = softmax(QKᵀ/√d) · V) ==\n");
    println!("  d={d}, contexte N={n}, d_v={dv}, {nq} requêtes, échelle softmax 1/√d\n");
    println!(
        "  {:>6} {:>8} | {:^17} | {:^17}",
        "decay", "captée", "HOT  cos / relL2", "WARM cos / relL2"
    );
    println!("  {}", "-".repeat(52));

    for &decay in &[0.99f32, 0.95, 0.90, 0.80] {
        let train = gen_keys(10, 1024, d, r, decay, 0.02);
        let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);

        let keys = gen_keys(20, n, d, r, decay, 0.02);
        let mut rng = Rng::new(77);
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

        let (mut hc, mut he, mut wc, mut we) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for _ in 0..nq {
            let mut q = vec![0.0f32; d];
            rng.fill_gaussian(&mut q);
            let qc = model.query_coarse(&q);
            let qs = model.sign_bits(&q);

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

            let mut wt = vec![0.0f32; n];
            let mut wh = vec![0.0f32; n];
            let mut ww = vec![0.0f32; n];
            softmax_into(&s_true, scale, &mut wt);
            softmax_into(&s_hot, scale, &mut wh);
            softmax_into(&s_warm, scale, &mut ww);

            let ot = attn_output(&wt, &values, dv);
            let oh = attn_output(&wh, &values, dv);
            let ow = attn_output(&ww, &values, dv);

            hc += cosine(&ot, &oh);
            he += rel_l2(&ot, &oh);
            wc += cosine(&ot, &ow);
            we += rel_l2(&ot, &ow);
        }
        let k = nq as f32;
        println!(
            "  {:>6.2} {:>7.1}% | {:>7.3} / {:>6.3} | {:>7.3} / {:>6.3}",
            decay,
            model.captured_energy * 100.0,
            hc / k,
            he / k,
            wc / k,
            we / k
        );
    }

    println!(
        "\n  Lecture : la sortie d'attention est plus robuste que le ranking brut —\n  \
         le softmax moyenne les valeurs, donc des erreurs de score sur des jetons\n  \
         de poids voisin se compensent. HOT ≥ WARM ; l'écart se voit surtout quand\n  \
         le spectre s'aplatit (decay faible) et que le résidu 1-bit compte le plus."
    );
}
