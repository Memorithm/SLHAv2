//! λ calibration / semantic-drift (ΔP) study.
//!
//! Run with:  `cargo run -p scirust --release --example calibrate_lambda`
//!
//! The fused score is `coarse + λ·r`, where `r = d_s − 2·popcount(q_sign ⊕ B)`
//! is the signed sign-LSH dot (independent of λ) and the spec sets
//! `λ = σ_E·√(π/(2·d_s))` (§3.2 — flagged there as an *unvalidated heuristic*).
//!
//! Against an FP reference `⟨Q,K⟩`, the optimal global multiplier on that λ has a
//! closed form (least squares): `α* = Σ rt·(λr) / Σ (λr)²` with `rt = ⟨Q,K⟩ −
//! coarse`. If `α* ≈ 1` the formula's constant is right (freeze it); otherwise
//! `C_emp = α*·√(π/(2·d_s))` is the corrected constant. We also report the
//! attention-output drift `ΔP_out = 1 − cos` at the formula's λ.

use scirust::attention::slha_v2::{D_S, FLAG_WARM};
use scirust::metrics::{cosine, dot, softmax_into};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection, D_K};

/// Per-query data kept for the output-drift pass: (coarse, λ·r, true, values).
type PerQuery = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<Vec<f32>>);

fn attn_out(w: &[f32], v: &[Vec<f32>], dv: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; dv];
    for (wi, vi) in w.iter().zip(v) {
        for j in 0..dv {
            o[j] += wi * vi[j];
        }
    }
    o
}

fn main() {
    let proj = Projection::new(0xCA11B);
    let c_formula = (std::f32::consts::PI / (2.0 * D_S as f32)).sqrt();
    let dv = 64usize;
    let n = 256usize;
    let nq = 16usize;
    let scale = 1.0 / (D_K as f32).sqrt();

    println!("== Calibration de λ (dérive ΔP vs référence FP) ==");
    println!("  C_formula = √(π/(2·d_s)) = {c_formula:.5}   (λ_tuile = C·σ_E)\n");
    println!(
        "  {:>5} | {:>8} {:>9} | {:>9} {:>9} | {:>8}",
        "rho", "α*(LS)", "C_emp", "RMSE@1", "RMSE@α*", "Δout@1"
    );
    println!("  {}", "-".repeat(60));

    for &rho in &[0.1f32, 0.2, 0.3, 0.5, 0.7] {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let mut sse1 = 0.0f64;
        let mut cnt = 0u64;
        // Per query: (coarse[], λr[], true[], values[]) — kept for output drift.
        let mut per_q: Vec<PerQuery> = Vec::new();

        for qi in 0..nq {
            let (q, toks) = generate(1000 + qi as u64, n, rho);
            let q_sign = proj.sign_bits(&q);
            let mut rngv = Rng::new(7000 + qi as u64);
            let values: Vec<Vec<f32>> = (0..n)
                .map(|_| {
                    let mut v = vec![0.0f32; dv];
                    rngv.fill_gaussian(&mut v);
                    v
                })
                .collect();

            let mut coarse_v = vec![0.0f32; n];
            let mut lamr_v = vec![0.0f32; n];
            let mut true_v = vec![0.0f32; n];
            for (i, tok) in toks.iter().enumerate() {
                let hot = build_tile(&proj, tok, i as u32, false);
                let mut warm = hot;
                warm.flags |= FLAG_WARM;
                let coarse = warm.compute_score(&q, &q_sign);
                let lamr = hot.compute_score(&q, &q_sign) - coarse; // = λ_formula · r
                let truth = dot(&q, &tok.k_real);
                let rt = truth - coarse;

                num += (rt * lamr) as f64;
                den += (lamr * lamr) as f64;
                let e1 = rt - lamr; // err at α=1 (formula λ)
                sse1 += (e1 * e1) as f64;
                cnt += 1;

                coarse_v[i] = coarse;
                lamr_v[i] = lamr;
                true_v[i] = truth;
            }
            per_q.push((coarse_v, lamr_v, true_v, values));
        }

        let alpha = if den > 0.0 { (num / den) as f32 } else { 0.0 };
        let c_emp = alpha * c_formula;
        let rmse1 = (sse1 / cnt as f64).sqrt() as f32;

        let mut sse_a = 0.0f64;
        let mut dout1 = 0.0f32;
        for (cv, lr, tv, vals) in &per_q {
            for i in 0..cv.len() {
                let e = tv[i] - (cv[i] + alpha * lr[i]);
                sse_a += (e * e) as f64;
            }
            // Output drift at α = 1 (formula λ).
            let mut wt = vec![0.0f32; cv.len()];
            let mut wa = vec![0.0f32; cv.len()];
            softmax_into(tv, scale, &mut wt);
            let slha1: Vec<f32> = (0..cv.len()).map(|i| cv[i] + lr[i]).collect();
            softmax_into(&slha1, scale, &mut wa);
            dout1 += 1.0 - cosine(&attn_out(&wt, vals, dv), &attn_out(&wa, vals, dv));
        }
        let rmse_a = (sse_a / cnt as f64).sqrt() as f32;
        dout1 /= nq as f32;

        println!(
            "  {:>5.2} | {:>8.3} {:>9.5} | {:>9.4} {:>9.4} | {:>8.4}",
            rho, alpha, c_emp, rmse1, rmse_a, dout1
        );
    }

    println!(
        "\n  Lecture : α*(LS) = multiplicateur optimal sur la λ de la formule.\n  \
         α* ≈ 1 ⇒ la formule est calibrée (à figer) ; sinon la constante corrigée\n  \
         est C_emp = α*·C_formula. RMSE@1 vs RMSE@α* = marge gagnée en optimisant λ ;\n  \
         Δout@1 = dérive de la SORTIE d'attention (1−cos) avec la formule.\n  \
         (Référence FP, données synthétiques, projections aléatoires.)"
    );
}
