//! Measurement prototype for SLHA v2.
//!
//! Run with:  `cargo run --example measure --release`
//!
//! It reports, on synthetic data with a controllable residual energy `rho`:
//!   * how well the 1-bit sign-LSH core tracks the true cosine,
//!   * how well the full SLHA score (HOT) and the latent-only score (WARM)
//!     approximate the true full-precision score, by ranking (Spearman),
//!     magnitude (Pearson) and top-k overlap.
//!
//! All numbers are from random (untrained) projections — they measure the
//! quantisation + 1-bit-residual machinery, not the quality of a *learned*
//! low-rank projection (that is training-dependent and out of scope here).

use std::mem::{align_of, size_of};
use std::time::Instant;

use scirust::attention::slha_v2::SciRustSlhaTile;
use scirust::metrics::{dot, pearson, spearman, topk_overlap};
use scirust::scenario::{build_tile, generate, Projection, D_K};

fn main() {
    println!("== SLHA v2 — prototype de mesure (projections aléatoires) ==\n");

    // --- Tile layout facts ---------------------------------------------------
    println!("Tuile SciRustSlhaTile :");
    println!("  size_of  = {} octets", size_of::<SciRustSlhaTile>());
    println!("  align_of = {} octets", align_of::<SciRustSlhaTile>());
    println!(
        "  lignes de cache (64o) par tuile = {}\n",
        size_of::<SciRustSlhaTile>() / 64
    );

    let proj = Projection::new(0xC0FFEE);

    // --- 1) sign-LSH residual core fidelity (vs cosine) ----------------------
    {
        let (q, toks) = generate(0xA11CE, 512, 0.5);
        let q_sign = proj.sign_bits(&q);
        let mut est = Vec::new();
        let mut cosv = Vec::new();
        for t in &toks {
            let b = proj.sign_bits(&t.e);
            let hamming: u32 = (0..q_sign.len())
                .map(|w| (q_sign[w] ^ b[w]).count_ones())
                .sum();
            est.push(256.0f32 - 2.0 * hamming as f32);
            let ne = dot(&t.e, &t.e).sqrt();
            let nq = dot(&q, &q).sqrt();
            cosv.push(dot(&q, &t.e) / (nq * ne));
        }
        println!("1) Cœur binaire sign-LSH  (d_s = 256 bits)");
        println!(
            "   Spearman(résidu, cos θ) = {:.3}   Pearson = {:.3}\n",
            spearman(&est, &cosv),
            pearson(&est, &cosv)
        );
    }

    // --- 2) End-to-end HOT vs WARM across residual energy --------------------
    println!("2) Score complet vs vérité terrain FP  (N = 512 jetons/contexte)");
    println!("   rho = ||e|| / ||k_real||   (part d'énergie que le bas-rang rate)\n");
    println!(
        "   {:>5} | {:^27} | {:^27}",
        "", "HOT (latent + résidu)", "WARM (latent seul)"
    );
    println!(
        "   {:>5} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "rho", "Spear", "Pears", "top16", "Spear", "Pears", "top16"
    );
    println!("   {}", "-".repeat(66));

    for &rho in &[0.05f32, 0.1, 0.2, 0.3, 0.5, 0.7] {
        let (q, toks) = generate(0xBEEF, 512, rho);
        let q_sign = proj.sign_bits(&q);

        let mut s_true = Vec::new();
        let mut s_hot = Vec::new();
        let mut s_warm = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            s_true.push(dot(&q, &t.k_real));
            let hot = build_tile(&proj, t, i as u32, false);
            let mut warm = hot;
            warm.flags |= scirust::attention::slha_v2::FLAG_WARM;
            s_hot.push(hot.compute_score(&q, &q_sign));
            s_warm.push(warm.compute_score(&q, &q_sign));
        }

        println!(
            "   {:>5.2} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3}",
            rho,
            spearman(&s_hot, &s_true),
            pearson(&s_hot, &s_true),
            topk_overlap(&s_true, &s_hot, 16),
            spearman(&s_warm, &s_true),
            pearson(&s_warm, &s_true),
            topk_overlap(&s_true, &s_warm, 16),
        );
    }
    println!();

    // --- 3) Throughput: scalar vs AVX2 vs AVX-512 ----------------------------
    {
        let (q, toks) = generate(1, 4096, 0.3);
        let q_sign = proj.sign_bits(&q);
        let tiles: Vec<SciRustSlhaTile> = toks
            .iter()
            .enumerate()
            .map(|(i, t)| build_tile(&proj, t, i as u32, false))
            .collect();
        let iters = 200usize;
        let scores = (iters * tiles.len()) as f64;

        let mut sink = 0.0f32;
        let t0 = Instant::now();
        for _ in 0..iters {
            for tile in &tiles {
                sink += tile.compute_score_scalar(&q, &q_sign);
            }
        }
        let scal = scores / t0.elapsed().as_secs_f64() / 1e6;

        println!("3) Débit  (checksum {sink:.0})");
        println!("   scalaire : {scal:>6.1} M scores/s   (1×)");

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                let mut s = 0.0f32;
                let t = Instant::now();
                for _ in 0..iters {
                    for tile in &tiles {
                        s += unsafe { tile.compute_score_avx2(&q, &q_sign) };
                    }
                }
                std::hint::black_box(s);
                let r = scores / t.elapsed().as_secs_f64() / 1e6;
                println!("   AVX2     : {r:>6.1} M scores/s   (×{:.2})", r / scal);
            }
            if std::is_x86_feature_detected!("avx512f") {
                let mut s = 0.0f32;
                let t = Instant::now();
                for _ in 0..iters {
                    for tile in &tiles {
                        s += unsafe { tile.compute_score_avx512(&q, &q_sign) };
                    }
                }
                std::hint::black_box(s);
                let r = scores / t.elapsed().as_secs_f64() / 1e6;
                println!("   AVX-512  : {r:>6.1} M scores/s   (×{:.2})", r / scal);
            }
        }
    }

    println!("\n   D_K = {D_K}. Toutes les valeurs sont reproductibles (graines fixes).");
}
