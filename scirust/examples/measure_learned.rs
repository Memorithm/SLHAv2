//! Measurement prototype with a **learned** (PCA) low-rank projection.
//!
//! Run with:  `cargo run --example measure_learned --release`
//!
//! Unlike `measure` (which hand-sets the residual energy `rho`), here the
//! projection is learned by PCA on sample keys, so the captured energy — and
//! hence the effective `rho` — is *determined by the data spectrum* at rank
//! `D_C = 128`. PCA is the optimal linear rank-`D_C` reconstruction, so this is
//! the honest "with a trained low-rank base" picture.

use scirust::attention::slha_v2::{
    LatentCodec, SciRustSlhaTile, D_C, FLAG_WARM, GROUP_DIM, N_GROUPS,
};
use scirust::learned::{gen_keys, LearnedModel};
use scirust::metrics::{dot, spearman, topk_overlap};

/// Returns (HOT Spearman, HOT top16, WARM Spearman, WARM top16).
fn evaluate(model: &LearnedModel, decay: f32, codec: LatentCodec) -> (f32, f32, f32, f32) {
    let d = model.d;
    let eval = gen_keys(20, 512, d, 256, decay, 0.02);
    let q = &gen_keys(30, 1, d, 256, decay, 0.02)[0];
    let q_coarse = model.query_coarse(q);
    let q_sign = model.sign_bits(q);

    let mut s_true = Vec::new();
    let mut s_hot = Vec::new();
    let mut s_warm = Vec::new();
    for (i, key) in eval.iter().enumerate() {
        s_true.push(dot(q, key));
        let hot: SciRustSlhaTile = model.encode_with(key, i as u32, false, codec);
        let mut warm = hot;
        warm.flags |= FLAG_WARM;
        s_hot.push(hot.compute_score(&q_coarse, &q_sign));
        s_warm.push(warm.compute_score(&q_coarse, &q_sign));
    }
    (
        spearman(&s_hot, &s_true),
        topk_overlap(&s_true, &s_hot, 16),
        spearman(&s_warm, &s_true),
        topk_overlap(&s_true, &s_warm, 16),
    )
}

fn main() {
    let d = 256;
    let r = 256;
    let n_train = 1024;

    println!("== SLHA v2 — mesure avec projection APPRISE (PCA) ==\n");
    println!("  d_model = {d}, latent D_C = 128, résidu d_s = 256 bits");
    println!(
        "  P = top-128 vecteurs propres de la covariance des clés (PCA), latent INT4 non whitené\n"
    );

    println!(
        "  {:>6} {:>9} {:>6} | {:^17} | {:^17}",
        "", "énergie", "", "HOT (latent+résidu)", "WARM (latent seul)"
    );
    println!(
        "  {:>6} {:>9} {:>6} | {:>8} {:>8} | {:>8} {:>8}",
        "decay", "captée", "rho~", "Spear", "top16", "Spear", "top16"
    );
    println!("  {}", "-".repeat(62));

    for &decay in &[0.99f32, 0.97, 0.93, 0.85] {
        let train = gen_keys(10, n_train, d, r, decay, 0.02);
        let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);
        let rho_eff = (1.0 - model.captured_energy).max(0.0).sqrt();
        let (h_sp, h_tk, w_sp, w_tk) = evaluate(&model, decay, LatentCodec::Int4Grouped);
        println!(
            "  {:>6.2} {:>8.1}% {:>6.2} | {:>8.3} {:>8.3} | {:>8.3} {:>8.3}",
            decay,
            model.captured_energy * 100.0,
            rho_eff,
            h_sp,
            h_tk,
            w_sp,
            w_tk
        );
    }

    // --- Codecs du latent 4 bits : uniforme simple / groupé (MX) / NF4 -------
    println!("\n  Codec du latent 4 bits (decay = 0.93, même tuile 128 o) :");
    let train = gen_keys(10, n_train, d, r, 0.93, 0.02);
    let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);
    for (label, codec) in [
        ("INT4 unique", LatentCodec::Int4Single),
        ("INT4 groupé", LatentCodec::Int4Grouped),
        ("NF4 groupé", LatentCodec::Nf4),
        ("TQ3 TurboQuant", LatentCodec::Tq3),
        ("MIX3 tête+corps", LatentCodec::Mix3),
    ] {
        let (h_sp, h_tk, w_sp, _w_tk) = evaluate(&model, 0.93, codec);
        println!("    {label:<12} -> HOT Spearman {h_sp:.3} (top16 {h_tk:.3})   WARM {w_sp:.3}");
    }

    // INT8 reference: what 8 bits would buy (2× latent bytes -> 192-byte tile).
    // WARM / coarse-only (the residual is codec-independent), same eval set.
    {
        let eval = gen_keys(20, 512, d, r, 0.93, 0.02);
        let q = &gen_keys(30, 1, d, r, 0.93, 0.02)[0];
        let qc = model.query_coarse(q);
        let mut s_true = Vec::new();
        let mut s_int8 = Vec::new();
        for key in &eval {
            s_true.push(dot(q, key));
            let h = model.latent(key);
            let mut deq = [0.0f32; D_C];
            for g in 0..N_GROUPS {
                let (lo, hi) = (g * GROUP_DIM, (g + 1) * GROUP_DIM);
                let mx = h[lo..hi].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                for i in lo..hi {
                    deq[i] = (h[i] / s).round().clamp(-127.0, 127.0) * s;
                }
            }
            s_int8.push(dot(&qc, &deq));
        }
        println!(
            "    {:<12} -> WARM Spearman {:.3}   (* 2× octets latent : tuile 192 o)",
            "INT8 réf*",
            spearman(&s_int8, &s_true)
        );
    }

    println!(
        "\n  Lecture : NF4 et le groupage (MX) tiennent dans la tuile 128 o et réduisent\n  \
         l'erreur de reconstruction, mais le gain end-to-end est marginal. Surprise\n  \
         (réf INT8) : DOUBLER les bits ne change quasiment rien au WARM (~0,61) — le\n  \
         plafond du terme coarse n'est PAS la quantification mais la PROJECTION\n  \
         bas-rang elle-même. Vrais leviers : meilleure projection (§7.7) + résidu 1-bit."
    );
}
