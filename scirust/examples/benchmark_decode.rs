//! Proxy de débit de **décodage bout-en-bout** — ⚠️ PAS un vrai LLM.
//!
//! Lancement :  `cargo run --example benchmark_decode --release`
//!             `cargo run --example benchmark_decode --release -- 16384`  (contexte custom)
//!
//! Les §7 mesurent la fidélité et le débit au niveau du **score**
//! (`compute_score`). Ici on enchaîne un **cycle de décodage d'attention
//! complet** — multi-têtes : scores → softmax → agrégation des valeurs `V` — pour
//! estimer un *ordre de grandeur* en **tokens/seconde**, comparé à une référence
//! pleine précision (clés `f32` non compressées).
//!
//! C'est un **PROXY** : pas de vrai modèle, pas de perplexité. Le `V` est en FP32
//! et identique des deux côtés ; l'écart vient donc du calcul des scores et de
//! l'empreinte mémoire des clés (SLHA **128 o/token** vs `f32` **512 o/token**).
//! Le facteur **dépend fortement du matériel** (calcul vs bande passante) — à
//! confirmer sur un vrai modèle (voir les réserves d'honnêteté, §6–7).

// Boucles numériques indexées : plus proches des maths que des chaînes d'itérateurs.
#![allow(clippy::needless_range_loop)]

use scirust::attention::slha_v2::SciRustSlhaTile;
use scirust::metrics::{dot, softmax_into};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection, D_K};
use std::time::Instant;

/// `out = Σ_i weights[i] · V[i]` (agrégation softmax·V, FP32, identique partout).
fn aggregate_v(weights: &[f32], values: &[f32], d_v: usize, out: &mut [f32]) {
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for (i, &w) in weights.iter().enumerate() {
        let row = &values[i * d_v..(i + 1) * d_v];
        for j in 0..d_v {
            out[j] += w * row[j];
        }
    }
}

fn main() {
    let context_len: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);
    let n_heads = 32usize;
    let d_head = D_K; // 128
    let d_v = 128usize;
    let iters = 20usize;
    let scale = 1.0 / (d_head as f32).sqrt();

    println!("== Proxy de décodage bout-en-bout (⚠️ PAS un vrai LLM) ==\n");
    println!("  têtes={n_heads}, d_head={d_head}, d_v={d_v}, contexte={context_len}, {iters} itérations\n");

    // ── Contexte KV partagé : tuiles HOT + clés f32 non compressées + valeurs V ──
    let proj = Projection::new(0x00DE_C0DE);
    let (_, toks) = generate(0x00DE_C0DE, context_len, 0.3);
    let tiles: Vec<SciRustSlhaTile> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();
    let keys_fp: Vec<[f32; 128]> = toks.iter().map(|t| t.k_real).collect();

    let mut rngv = Rng::new(0x7);
    let mut values = vec![0.0f32; context_len * d_v];
    rngv.fill_gaussian(&mut values);

    // Une requête (+ ses bits de signe) par tête.
    let mut rngq = Rng::new(0x11);
    let mut queries: Vec<[f32; 128]> = Vec::with_capacity(n_heads);
    for _ in 0..n_heads {
        let mut q = [0.0f32; 128];
        rngq.fill_gaussian(&mut q);
        queries.push(q);
    }
    let qsigns: Vec<_> = queries.iter().map(|q| proj.sign_bits(q)).collect();

    let mut scores = vec![0.0f32; context_len];
    let mut weights = vec![0.0f32; context_len];
    let mut out = vec![0.0f32; d_v];

    // ── SLHA : score fusionné sur tuile 128 o (dispatch SIMD) ──
    let mut sink = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..iters {
        for h in 0..n_heads {
            let (q, qs) = (&queries[h], &qsigns[h]);
            for (s, tile) in scores.iter_mut().zip(&tiles) {
                *s = tile.compute_score(q, qs);
            }
            softmax_into(&scores, scale, &mut weights);
            aggregate_v(&weights, &values, d_v, &mut out);
            sink += out[0];
        }
    }
    let slha_dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    // ── Référence pleine précision : produit scalaire sur clés f32 non compressées ──
    let mut sink_fp = 0.0f32;
    let t1 = Instant::now();
    for _ in 0..iters {
        for h in 0..n_heads {
            let q = &queries[h];
            for (s, k) in scores.iter_mut().zip(&keys_fp) {
                *s = dot(q, k);
            }
            softmax_into(&scores, scale, &mut weights);
            aggregate_v(&weights, &values, d_v, &mut out);
            sink_fp += out[0];
        }
    }
    let fp_dt = t1.elapsed().as_secs_f64();
    std::hint::black_box(sink_fp);

    let slha_tps = iters as f64 / slha_dt;
    let fp_tps = iters as f64 / fp_dt;
    let speedup = slha_tps / fp_tps;
    let slha_kb = context_len * 128 / 1024;
    let fp_kb = context_len * d_head * 4 / 1024;

    println!(
        "  empreinte clés/contexte : SLHA {slha_kb} Ko (128 o/token) · FP f32 {fp_kb} Ko ({} o/token)\n",
        d_head * 4
    );
    println!("  {:<24} {:>10} {:>12}", "", "tok/s", "ms/token");
    println!("  {}", "-".repeat(48));
    println!(
        "  {:<24} {:>10.1} {:>12.2}",
        "SLHA (tuile 128 o)",
        slha_tps,
        1e3 * slha_dt / iters as f64
    );
    println!(
        "  {:<24} {:>10.1} {:>12.2}",
        "Réf FP (clés f32)",
        fp_tps,
        1e3 * fp_dt / iters as f64
    );
    println!("  {:<24} {:>9.2}×", "Accélération", speedup);

    println!(
        "\n  Lecture : proxy bout-en-bout (scores + softmax + agrégation V), PAS un vrai\n  \
         LLM ni une mesure de perplexité. Le V est FP32 et identique des deux côtés,\n  \
         donc l'écart vient du calcul des scores et de l'empreinte mémoire des clés\n  \
         (SLHA 128 o vs f32 512 o/token). Le facteur dépend du matériel (calcul vs\n  \
         bande passante) ; sous forte pression mémoire (long contexte) l'avantage de\n  \
         l'empreinte réduite ressort davantage. À confirmer sur un vrai modèle (§6–7)."
    );
}
