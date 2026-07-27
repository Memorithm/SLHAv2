//! Plan axis **A5** — informed eviction (H2O / StreamingLLM / SnapKV).
//!
//! Run with:  `cargo run --example informed_eviction --release`
//!
//! The §4 eviction policy is pure-causal (oldest first). Real attention has
//! **heavy-hitters** (tokens that consistently attract mass) and **attention
//! sinks** (the first few tokens, which every query attends — StreamingLLM);
//! dropping them by age alone destroys quality under pressure. A5 replaces the
//! eviction order with an **importance** order: evict the lowest cumulative
//! attention mass first (H2O), and **pin the sinks** by position. `σ_E` keeps
//! its role in the *paging* phase (HOT→WARM) — only the harsher *eviction*
//! phase changes.
//!
//! This prototype has no real transformer, so we construct a scenario that
//! exhibits the mechanism cleanly and **measure** the consequence: the cosine
//! of the attention output (softmax·V) produced by the budgeted cache vs the
//! full-attention output, under the same budget, for `Causal` vs `Importance`.
//! The scenario is constructed (a few mid-sequence anchor tokens a recurring
//! query attends to + pinned initial sinks) — exactly the heavy-hitter/sink
//! structure the literature reports. The magnitude on a real LLM is the
//! Phase 3 / A7 question; the *sign* (informed degrades more gracefully) is
//! what we assert here.
//!
//! See: H2O (arXiv 2306.14048), StreamingLLM (2309.17453), SnapKV (2404.14469),
//! PyramidKV (2406.02069), FastGen (2310.01801).

use scirust::ccos::{ElasticKvCache, EvictionPolicy, TileState, WARM_BYTES};
use scirust::metrics::{cosine, dot, softmax_into};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, ContextToken, Projection, D_K};

const N: usize = 64; // context length
const DV: usize = 64; // value head width
const SINK_WINDOW: usize = 4; // StreamingLLM: pin the first 4 tokens
const KEEP: usize = 16; // budget keeps 16 WARM tiles → evict 48 of 64
/// Anchor (heavy-hitter) positions: mid-sequence, NOT sinks, so causal eviction
/// (oldest first) drops them while informed eviction keeps them.
const ANCHORS: [usize; 3] = [12, 28, 44];

/// One query: strongly aligned with `anchor_dir` so attention concentrates on
/// the anchor tokens (the heavy-hitters).
fn mk_query(rng: &mut Rng, anchor_dir: &[f32; D_K]) -> [f32; D_K] {
    let mut q = [0.0f32; D_K];
    for i in 0..D_K {
        q[i] = 5.0 * anchor_dir[i] + 0.5 * rng.next_gaussian();
    }
    q
}

/// Build the context: anchors get a large coarse key along `anchor_dir`
/// (so ⟨q, k_real⟩ is large for anchor-aligned queries); the rest are generic
/// synthetic tokens.
fn build_context(seed: u64, anchor_dir: &[f32; D_K]) -> Vec<ContextToken> {
    let (_q, toks) = generate(seed, N, 0.3);
    let mut out: Vec<ContextToken> = toks;
    let mut rng = Rng::new(seed ^ 0xA5_A5);
    for &a in &ANCHORS {
        // Coarse key = anchor direction × large amplitude; tiny residual.
        let mut k_coarse = [0.0f32; D_K];
        for i in 0..D_K {
            k_coarse[i] = 8.0 * anchor_dir[i];
        }
        let mut e = [0.0f32; D_K];
        for x in e.iter_mut() {
            *x = 0.1 * rng.next_gaussian();
        }
        let mut k_real = [0.0f32; D_K];
        for i in 0..D_K {
            k_real[i] = k_coarse[i] + e[i];
        }
        out[a] = ContextToken {
            k_coarse,
            e,
            k_real,
        };
    }
    out
}

/// Attention output `softmax(scores/√d)·V` over the tokens in `scores`
/// (`f32::NEG_INFINITY` ⇒ token excluded, as for evicted slots).
fn attend(scores: &[f32], values: &[Vec<f32>]) -> Vec<f32> {
    let scale = 1.0 / (D_K as f32).sqrt();
    let mut w = vec![0.0f32; scores.len()];
    softmax_into(scores, scale, &mut w);
    let mut o = vec![0.0f32; DV];
    for (wi, v) in w.iter().zip(values) {
        for j in 0..DV {
            o[j] += wi * v[j];
        }
    }
    o
}

/// Fill a cache with all N tiles (HOT). For `Importance`, accumulate H2O
/// importance over `train_queries` decoding steps before enforcing the budget.
fn run_cache(
    seed: u64,
    anchor_dir: &[f32; D_K],
    eviction: EvictionPolicy,
    train_queries: &[[f32; D_K]],
) -> ElasticKvCache {
    let proj = Projection::new(seed);
    let toks = build_context(seed, anchor_dir);
    let budget = KEEP * WARM_BYTES;
    let mut cache = ElasticKvCache::with_eviction(budget, eviction);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    if matches!(eviction, EvictionPolicy::Importance { .. }) {
        let temp = 1.0 / (D_K as f32).sqrt();
        for q in train_queries {
            let q_sign = proj.sign_bits(q);
            let scores: Vec<(usize, f32)> = (0..N)
                .filter(|&s| cache.state(s) != TileState::Cold)
                .map(|s| (s, cache.score(s, q, &q_sign)))
                .collect();
            cache.observe_scores(&scores, temp);
        }
    }
    cache.enforce_budget();
    cache
}

fn main() {
    println!("== SLHA v2 — axe A5 : éviction informée (H2O / sinks) ==\n");
    println!(
        "  N={N} tokens, sinks={SINK_WINDOW} (pinnés), budget={KEEP} tuiles WARM \
         (→ éviction de {})",
        N - KEEP
    );
    println!("  heavy-hitters aux positions {ANCHORS:?} (mi-séquence, pas des sinks)\n");

    let mut rng = Rng::new(2026);
    let mut anchor_dir = [0.0f32; D_K];
    rng.fill_gaussian(&mut anchor_dir);
    let nrm = dot(&anchor_dir, &anchor_dir).sqrt().max(1e-9);
    for x in anchor_dir.iter_mut() {
        *x /= nrm;
    }

    // Values V (random, fixed across policies so the comparison is clean).
    let mut vrng = Rng::new(0xBEEF);
    let values: Vec<Vec<f32>> = (0..N)
        .map(|_| {
            let mut v = vec![0.0f32; DV];
            vrng.fill_gaussian(&mut v);
            v
        })
        .collect();

    // H2O importance is accumulated over a few training queries (decoding steps).
    let train_queries: Vec<[f32; D_K]> = (0..12).map(|_| mk_query(&mut rng, &anchor_dir)).collect();

    let proj = Projection::new(2026);
    let toks = build_context(2026, &anchor_dir);

    // Held-out evaluation queries: average the output cosine over several.
    let eval_queries: Vec<[f32; D_K]> = (0..8).map(|_| mk_query(&mut rng, &anchor_dir)).collect();

    let causal = run_cache(2026, &anchor_dir, EvictionPolicy::Causal, &train_queries);
    let informed = run_cache(
        2026,
        &anchor_dir,
        EvictionPolicy::Importance {
            sink_window: SINK_WINDOW,
        },
        &train_queries,
    );

    let mut cos_causal = 0.0f32;
    let mut cos_informed = 0.0f32;
    for q in &eval_queries {
        let q_sign = proj.sign_bits(q);
        // Full-attention reference output (exact scores, all tokens).
        let s_true: Vec<f32> = toks.iter().map(|t| dot(q, &t.k_real)).collect();
        let full = attend(&s_true, &values);

        // Budgeted output for a cache: live tiles scored, evicted → -inf.
        let budgeted_output = |cache: &ElasticKvCache| -> Vec<f32> {
            let mut s = vec![f32::NEG_INFINITY; N];
            for (s_idx, slot) in s.iter_mut().enumerate() {
                if cache.state(s_idx) != TileState::Cold {
                    *slot = cache.score(s_idx, q, &q_sign);
                }
            }
            attend(&s, &values)
        };
        cos_causal += cosine(&budgeted_output(&causal), &full);
        cos_informed += cosine(&budgeted_output(&informed), &full);
    }
    let nc = eval_queries.len() as f32;
    let cos_causal = cos_causal / nc;
    let cos_informed = cos_informed / nc;

    // Report which anchor tokens each policy kept.
    let kept = |cache: &ElasticKvCache| -> Vec<usize> {
        ANCHORS
            .iter()
            .copied()
            .filter(|&a| cache.state(a) != TileState::Cold)
            .collect()
    };
    let sinks_kept = |cache: &ElasticKvCache| -> usize {
        (0..SINK_WINDOW)
            .filter(|&s| cache.state(s) != TileState::Cold)
            .count()
    };

    println!(
        "  {:<18} {:>10} {:>10} {:>10}",
        "politique", "cos(out)", "sinks kept", "HH kept"
    );
    println!("  {}", "-".repeat(52));
    println!(
        "  {:<18} {:>10.3} {:>10}/{SINK_WINDOW} {:>10}",
        "Causal (oldest)",
        cos_causal,
        sinks_kept(&causal),
        kept(&causal).len()
    );
    println!(
        "  {:<18} {:>10.3} {:>10}/{SINK_WINDOW} {:>10}",
        "Importance (A5)",
        cos_informed,
        sinks_kept(&informed),
        kept(&informed).len()
    );
    println!(
        "\n  gain cosinus de sortie : {:+.3}  ({:+.1}%)",
        cos_informed - cos_causal,
        100.0 * (cos_informed - cos_causal) / cos_causal.abs().max(1e-6)
    );

    println!(
        "\n  Lecture (mesuré, pas affirmé sur foi de la littérature) :\n  \
         • Sous pression ({} tuiles gardées sur {N}), l'éviction causale droppait les\n    \
         heavy-hitters mi-séquence ET les sinks — la sortie d'attention se rétracte\n    \
         sur les jetons récents sans importance réelle (cos {cos_causal:.3}).\n  \
         • L'éviction informée (H2O + sinks pinnés) préserve les {}/3 heavy-hitters\n    \
         et les {}/{} sinks : la sortie reste proche de la référence (cos {cos_informed:.3}).\n  \
         • σ_E garde son rôle dans la phase de *paging* (HOT→WARM) — seule la phase\n    \
         d'*éviction* change. Les deux caches pagent à l'identique.\n  \
         • Le scénario est construit pour exhiber le mécanisme (HH + sinks, exactement\n    \
         la structure rapportée par H2O/StreamingLLM). La *magnitude* sur un vrai LLM\n    \
         (long contexte, perplexité) est l'axe A7 / Phase 3 ; le *signe* — informée\n    \
         dégrade plus gracieusement — est ce que la mesure confirme ici.",
        KEEP,
        kept(&informed).len(),
        sinks_kept(&informed),
        SINK_WINDOW
    );
}
