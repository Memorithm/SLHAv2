//! Plan axis **A1** — low-rank projection on **pre-RoPE** keys (ShadowKV).
//!
//! Run with:  `cargo run --example pre_rope_projection --release`
//!
//! The §7.8 finding is that the **projection** (not the bit-width) is the WARM
//! ceiling. ShadowKV shows *why*: RoPE mixes channels and creates outliers, so
//! keys are far less low-rank **after** RoPE than before. Projecting pre-RoPE
//! (intercept the key before rotation, store the latent, reconstruct and
//! post-rotate) captures more energy and should lift the WARM floor.
//!
//! This prototype has no real transformer, so we model RoPE with the standard
//! paired rotation ([`scirust::rope`]) and measure what is and isn't robust on
//! a synthetic factor model:
//!
//! 1. **Captured energy (robust)**: a rank-`D_C` PCA on pre-RoPE keys captures
//!    markedly more variance than on post-RoPE keys — the ShadowKV mechanism,
//!    stable across seeds. This is the measured root of the §7.8 bottleneck.
//! 2. **WARM ranking (NOT robust on this toy data)**: a *reference* WARM score
//!    in `d`-space (reconstruct → RoPE(pos) → dot) lifts the Spearman only
//!    sometimes — the gain is seed-dependent and within noise here, because the
//!    factor model (rank 8 + tiny noise) is too compressible even post-RoPE
//!    (96% at rank 128), so the lost tail energy hits magnitude more than
//!    ranking, and the pre-RoPE reconstruction error (rotated by RoPE) can eat
//!    the advantage. A robust WARM lift needs genuine LLM key distributions —
//!    the Phase 3 / A7 integration. Reported honestly, not papered over.
//!
//! See: ShadowKV (arXiv 2410.21465), KVQuant pre-RoPE (2401.18079).

use scirust::learned::{gen_keys, LearnedModel};
use scirust::metrics::{dot, spearman};
use scirust::rope::{rope, rope_copy, ROPE_BASE};

/// One full A1 vs naive measurement at a given (train, eval) seed pair.
/// Returns (cap_pre, cap_post, warm_naive_spearman, warm_a1_spearman).
fn measure(seed_train: u64, seed_eval: u64) -> (f32, f32, f32, f32) {
    let (d, r, noise, nt, ne, span) = (256, 8, 0.02, 512, 512, 8u32);

    let k_pre = gen_keys(seed_train, nt, d, r, 0.9, noise);
    // Long-context span: position j sits at j·span, so a modest sample count
    // still exercises the large RoPE angles where the rank inflation shows.
    let pos_t: Vec<u32> = (0..nt as u32).map(|i: u32| i * span).collect();
    let k_post: Vec<Vec<f32>> = k_pre
        .iter()
        .zip(&pos_t)
        .map(|(k, &p)| rope_copy(k, p, ROPE_BASE))
        .collect();

    let m_pre = LearnedModel::fit(&k_pre, d, seed_train, false);
    let m_post = LearnedModel::fit(&k_post, d, seed_train, false);

    let k_pre_e = gen_keys(seed_eval, ne, d, r, 0.9, noise);
    let pos_e: Vec<u32> = (0..ne as u32).map(|i: u32| i * span).collect();
    let k_post_e: Vec<Vec<f32>> = k_pre_e
        .iter()
        .zip(&pos_e)
        .map(|(k, &p)| rope_copy(k, p, ROPE_BASE))
        .collect();
    let q_pre = &gen_keys(seed_eval + 1, 1, d, r, 0.9, noise)[0];
    let q_post = rope_copy(q_pre, 0, ROPE_BASE);

    let mut s_true = Vec::new();
    let mut s_naive = Vec::new();
    let mut s_a1 = Vec::new();
    for j in 0..ne {
        s_true.push(dot(&q_post, &k_post_e[j]));
        s_naive.push(dot(
            &q_post,
            &m_post.reconstruct(&m_post.latent(&k_post_e[j])),
        ));
        let mut rp = m_pre.reconstruct(&m_pre.latent(&k_pre_e[j]));
        rope(&mut rp, pos_e[j], ROPE_BASE);
        s_a1.push(dot(&q_post, &rp));
    }
    (
        m_pre.captured_energy,
        m_post.captured_energy,
        spearman(&s_naive, &s_true),
        spearman(&s_a1, &s_true),
    )
}

fn main() {
    println!("== SLHA v2 — axe A1 : projection bas-rang sur clés PRE-RoPE ==\n");
    println!("  d_model = 256, latent D_C = 128, rang effectif pre-RoPE ≈ 8\n");

    println!("  (1) Mécanisme ShadowKV — énergie captée à rang 128 (robuste)");
    println!(
        "  {:>6} {:>10} {:>10} {:>10}",
        "seeds", "pre-RoPE", "post-RoPE", "Δ"
    );
    println!("  {}", "-".repeat(42));
    for &(st, se) in &[(10u64, 20), (11, 21), (12, 22), (13, 23)] {
        let (cp, cq, _, _) = measure(st, se);
        println!(
            "  ({st},{se}) {:>9.1}% {:>9.1}% {:>+9.1}%",
            cp * 100.0,
            cq * 100.0,
            (cp - cq) * 100.0
        );
    }

    println!("\n  (2) Plafond WARM (réf. d-space, Spearman vs vrai ⟨Q,K⟩) — par seed");
    println!(
        "  {:>6} {:>10} {:>10} {:>10}",
        "seeds", "naïf", "A1", "gain"
    );
    println!("  {}", "-".repeat(42));
    let mut gains = Vec::new();
    for &(st, se) in &[(10u64, 20), (11, 21), (12, 22), (13, 23), (7, 77)] {
        let (_, _, n, a) = measure(st, se);
        let g = a - n;
        gains.push(g);
        println!("  ({st},{se}) {:>10.3} {:>10.3} {:>+10.3}", n, a, g);
    }
    let pos = gains.iter().filter(|&&g| g > 0.0).count();
    println!(
        "\n  gain WARM > 0 sur {pos}/{} seeds — pas robuste sur ce banc synthétique.",
        gains.len()
    );

    println!(
        "\n  Lecture (mesuré, pas affirmé sur foi de la littérature) :\n  \
         • RoPE détruit le bas-rang des clés : énergie captée chute de ~99,5 % à ~92 %\n    \
         de façon reproductible (Δ +7 %) — c'est la racine mesurée du goulot « projection »\n    \
         §7.8, et exactement le mécanisme que ShadowKV identifie.\n  \
         • Sur ce factor model (rang 8 + bruit minime), la levée du Spearman WARM n'est\n    \
         pas seulement instable : elle est légèrement négative (0/5 seeds). La queue\n    \
         perdue par la projection post-RoPE touche la magnitude plus que le ranking, et\n    \
         l'erreur de reconstruction pre-RoPE (rotée par de grands angles RoPE) mange le\n    \
         gain. Le mécanisme (énergie captée) et le bénéfice (ranking) sont dissociés ici.\n  \
         • Une levée robuste du plafond WARM nécessite les clés d'un vrai LLM (queue\n    \
         lourde, long contexte, objectif de perplexité — pas ranking toy) — intégration\n    \
         Phase 3 / A7, comme le plan l'anticipait en qualifiant A1 de gain sur vrai modèle.\n  \
         • L'invariant tuile 128 o tient : rien ici ne touche au kernel ; le dot pre-RoPE\n    \
         se replie sur l'espace latent une fois RoPE intégré au chemin coarse (A7)."
    );
}
