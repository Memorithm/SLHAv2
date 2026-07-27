//! CCOS Soft-Paging demo (§4) — elastic KV-cache under a memory budget.
//!
//! Run with:  `cargo run -p scirust --release --example ccos_softpaging`
//!
//! Streams a context into an `ElasticKvCache`, enforcing a byte budget. Two
//! regimes: (A) budget that only triggers HOT→WARM paging (no token dropped) —
//! we measure the attention-output fidelity vs an all-HOT reference; (B) a tight
//! budget that also evicts (→COLD) to stay bounded.

use scirust::attention::slha_v2::SciRustSlhaTile;
use scirust::ccos::{ElasticKvCache, PageOutPolicy};
use scirust::metrics::{cosine, dot, softmax_into};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection, D_K};

fn attn_out(scores: &[f32], values: &[Vec<f32>], scale: f32, dv: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; scores.len()];
    softmax_into(scores, scale, &mut w);
    let mut o = vec![0.0f32; dv];
    for (wi, vi) in w.iter().zip(values) {
        for j in 0..dv {
            o[j] += wi * vi[j];
        }
    }
    o
}

fn main() {
    let proj = Projection::new(0xC0501);
    let n = 8192usize;
    let dv = 64usize;
    let scale = 1.0 / (D_K as f32).sqrt();

    let (q, toks) = generate(1, n, 0.3);
    let q_sign = proj.sign_bits(&q);
    let tiles: Vec<SciRustSlhaTile> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();
    let mut rngv = Rng::new(9);
    let values: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut v = vec![0.0f32; dv];
            rngv.fill_gaussian(&mut v);
            v
        })
        .collect();
    let naive = n * 128;

    println!("== CCOS Soft-Paging (§4) — cache KV élastique ==");
    println!(
        "  contexte {n} tuiles · naïf tout-HOT = {} Ko\n",
        naive / 1024
    );

    // --- A) Paging-only budget (no eviction): measure output fidelity ---------
    // Recommended default policy: hybrid (page by σ_E, evict by age).
    let budget_a = n * 112; // between WARM(96) and HOT(128) totals -> pages, never evicts
    let mut cache = ElasticKvCache::with_budget(budget_a);
    for (i, t) in tiles.iter().enumerate() {
        cache.insert(*t);
        if i % 256 == 0 {
            cache.enforce_budget();
        }
    }
    cache.enforce_budget();
    let (hot, warm, cold) = cache.counts();

    // Fidelity: cache (mix HOT/WARM) vs all-HOT reference, over all n tokens.
    let s_ref: Vec<f32> = tiles.iter().map(|t| t.compute_score(&q, &q_sign)).collect();
    let s_cache: Vec<f32> = (0..n).map(|i| cache.score(i, &q, &q_sign)).collect();
    let s_true: Vec<f32> = toks.iter().map(|t| dot(&q, &t.k_real)).collect();
    let out_ref = attn_out(&s_ref, &values, scale, dv);
    let out_cache = attn_out(&s_cache, &values, scale, dv);
    let out_true = attn_out(&s_true, &values, scale, dv);

    println!("  A) Budget paging-seul = {} Ko :", budget_a / 1024);
    println!("     HOT={hot} · WARM={warm} · COLD={cold}");
    println!(
        "     footprint élastique = {} Ko  ({:.0}% du naïf)",
        cache.live_bytes() / 1024,
        100.0 * cache.live_bytes() as f32 / naive as f32
    );
    println!(
        "     sortie d'attention — cos(cache, tout-HOT) = {:.4} ; cos(cache, FP) = {:.4}",
        cosine(&out_ref, &out_cache),
        cosine(&out_true, &out_cache)
    );

    // --- B) Tight budget that forces eviction (→COLD) -------------------------
    // Pure-causal alternative policy (paging by age); eviction is by age either
    // way, so under this much pressure the two policies converge.
    let budget_b = n * 40;
    let mut cache_b = ElasticKvCache::new(budget_b, PageOutPolicy::OldestFirst);
    for (i, t) in tiles.iter().enumerate() {
        cache_b.insert(*t);
        if i % 256 == 0 {
            cache_b.enforce_budget();
        }
    }
    cache_b.enforce_budget();
    let (h2, w2, c2) = cache_b.counts();
    println!(
        "\n  B) Budget serré = {} Ko (force l'éviction) :",
        budget_b / 1024
    );
    println!("     HOT={h2} · WARM={w2} · COLD={c2}");
    println!(
        "     footprint élastique = {} Ko ≤ budget {} Ko  (slots COLD recyclables)",
        cache_b.live_bytes() / 1024,
        budget_b / 1024
    );

    println!(
        "\n  Lecture : (A) le Soft-Paging HOT→WARM borne la mémoire en libérant 32 o\n  \
         de résidu par tuile, sans I/O ni perte de jeton — la sortie reste fidèle.\n  \
         (B) sous forte pression, l'éviction →COLD borne le footprint (au prix du\n  \
         contexte le plus ancien ; un vrai CCOS le snapshoterait dans l'EventLog)."
    );
}
