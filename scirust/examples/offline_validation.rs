//! Phase 0 — OFFLINE validation of SLHA v2 (the GO/NO-GO, no LLM integration).
//!
//! Question: with a **learned** low-rank projection (not the random JL used by
//! the §7 studies), does the SLHA score preserve the *attention output*
//! `out = softmax(QKᵀ/√d)·V` on a realistic key distribution? And by how much
//! does the attention weight distribution move (a **perplexity proxy**)?
//!
//! Run (realistic-synthetic keys — produces a first number immediately):
//! ```text
//! cargo run --release --example offline_validation
//! ```
//! Run on REAL activations dumped from an HF model (see
//! `scripts/dump_activations.py`, which writes `k.bin` / `v.bin` / `q.bin`):
//! ```text
//! cargo run --release --example offline_validation -- --dump path/to/dir
//! ```
//! Validate a **held-out** projection (trained elsewhere, e.g. by
//! `train_on_real_activations`) instead of re-fitting on the scored keys:
//! ```text
//! cargo run --release --example offline_validation -- --weights proj.slhw
//! ```
//! Fitting the projection on the very keys you then score is optimistic — the
//! basis has already seen the test set. `--weights` loads a projection fitted on
//! a *different* key set, giving the honest, non-optimistic number.
//!
//! `--codec {grouped|nf4|mixed|tq3}` selects the latent quantiser (default:
//! grouped INT4). `mixed` stores the top 8 latent dims at 8-bit and the next
//! 112 at 4-bit in the same 64 bytes — built for the steep spectra of real
//! keys, where uniform INT4's 16 levels cannot span the outlier direction.
//! `tq3` is the TurboQuant port: 3-bit grid + separable 1-bit correction
//! plane in the same 64 bytes (see docs/TURBOQUANT.md).
//!
//! This is a PROXY, not a real perplexity: it isolates the attention layer with
//! cached activations. It exists to give a cheap, quantified GO/NO-GO *before*
//! the expensive llama.cpp integration (see PLAN.md, Phase 0).

// Numeric loops read closer to the math with indexing.
#![allow(clippy::needless_range_loop)]

use scirust::attention::slha_v2::{LatentCodec, D_C};
use scirust::learned::{gen_keys, LearnedModel};
use scirust::metrics::{cosine, dot, rel_l2, softmax_into};
use scirust::rng::Rng;
use scirust::weights;

// GO/NO-GO thresholds (tune per model / context; documented in PLAN.md §1).
const GO_COSINE: f32 = 0.98; // attention-output cosine vs FP32
const GO_KL: f32 = 0.03; // mean KL(w_true‖w_slha) of attention weights

/// One key distribution to evaluate (real dump, or one synthetic spectrum).
struct Regime {
    label: String,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    d: usize,
    dv: usize,
}

fn attn_output(scores: &[f32], values: &[Vec<f32>], scale: f32, dv: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; scores.len()];
    softmax_into(scores, scale, &mut w);
    let mut out = vec![0.0f32; dv];
    for (wi, vi) in w.iter().zip(values) {
        for j in 0..dv {
            out[j] += wi * vi[j];
        }
    }
    out
}

/// KL(p‖q) over the two softmax weight vectors — how far SLHA moves the
/// attention distribution. A stand-in for the downstream perplexity impact.
fn kl(scores_true: &[f32], scores_slha: &[f32], scale: f32) -> f32 {
    let mut p = vec![0.0f32; scores_true.len()];
    let mut q = vec![0.0f32; scores_slha.len()];
    softmax_into(scores_true, scale, &mut p);
    softmax_into(scores_slha, scale, &mut q);
    let mut acc = 0.0f32;
    for (pi, qi) in p.iter().zip(&q) {
        if *pi > 1e-9 {
            acc += pi * (pi / qi.max(1e-9)).ln();
        }
    }
    acc
}

/// Measure attention-output fidelity and weight-distribution KL over the
/// queries. With `preloaded = Some(m)` the held-out projection `m` is scored
/// as-is (train/test separation); otherwise a projection is fitted on the
/// regime's own keys (optimistic). Returns `(cosine, rel_l2, kl, captured)`;
/// `captured` is NaN for a preloaded projection (energy is not persisted).
fn evaluate(
    r: &Regime,
    preloaded: Option<&LearnedModel>,
    warm: bool,
    rht: bool,
    codec: LatentCodec,
) -> (f32, f32, f32, f32) {
    let fitted;
    let model = match preloaded {
        Some(m) => m,
        None => {
            fitted = LearnedModel::fit_with(&r.keys, r.d, 0x0005_C0FF, false, rht);
            &fitted
        }
    };
    let scale = 1.0 / (r.d as f32).sqrt();
    let tiles: Vec<_> = r
        .keys
        .iter()
        .enumerate()
        .map(|(i, k)| model.encode_with(k, i as u32, warm, codec))
        .collect();

    let (mut cos, mut rl2, mut klsum) = (0.0f32, 0.0f32, 0.0f32);
    for q in &r.queries {
        let qc = model.query_coarse(q);
        let qs = model.sign_bits(q);
        let s_true: Vec<f32> = r.keys.iter().map(|k| dot(q, k)).collect();
        let s_slha: Vec<f32> = tiles.iter().map(|t| t.compute_score(&qc, &qs)).collect();
        let o_true = attn_output(&s_true, &r.values, scale, r.dv);
        let o_slha = attn_output(&s_slha, &r.values, scale, r.dv);
        cos += cosine(&o_true, &o_slha);
        rl2 += rel_l2(&o_true, &o_slha);
        klsum += kl(&s_true, &s_slha, scale);
    }
    let n = r.queries.len() as f32;
    (cos / n, rl2 / n, klsum / n, model.captured_energy)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let dump = opt("--dump");

    // Latent codec for the tile encode (plan axis: INT4 vs NF4 vs mixed 8/4-bit).
    let codec = match opt("--codec").as_deref() {
        None | Some("grouped") => LatentCodec::Int4Grouped,
        Some("nf4") => LatentCodec::Nf4,
        Some("mixed") => LatentCodec::Mixed,
        Some("tq3") => LatentCodec::Tq3,
        Some(other) => panic!("--codec {other}: expected grouped | nf4 | mixed | tq3"),
    };

    // Optional held-out projection: score it as-is instead of re-fitting.
    let preloaded = opt("--weights")
        .map(|path| weights::load(&path).unwrap_or_else(|e| panic!("--weights {path}: {e}")));

    let (source, regimes) = build_regimes(dump);

    println!("== Phase 0 — validation OFFLINE de SLHA v2 (projection APPRISE) ==\n");
    println!("  Source : {source}");
    if let Some(m) = preloaded.as_ref() {
        println!(
            "  Projection : held-out (--weights, d={}), NON réajustée sur les clés testées.",
            m.d
        );
    } else {
        println!("  Projection : réajustée sur les clés testées (optimiste ; cf. --weights).");
    }
    println!(
        "  Codec latent : {}",
        match codec {
            LatentCodec::Int4Grouped => "INT4 groupé (défaut)",
            LatentCodec::Nf4 => "NF4",
            LatentCodec::Mixed => "MIXTE 8/4-bit (tête 8 dims @8b)",
            LatentCodec::Tq3 => "TQ3 TurboQuant (3-bit + correction 1-bit)",
            LatentCodec::Int4Single => "INT4 simple",
        }
    );
    println!();
    println!(
        "  {:<12} {:<5} {:>4} | {:>8} {:>8} {:>9} {:>8} | verdict",
        "régime", "état", "RHT", "cos↑", "relL2↓", "KL(ppl)↓", "captée"
    );
    println!("  {}", "-".repeat(78));

    let mut any_hot_go = false;
    for r in &regimes {
        if let Some(m) = preloaded.as_ref() {
            assert_eq!(
                m.d, r.d,
                "projection --weights entraînée pour d={} mais le régime « {} » a d={}",
                m.d, r.label, r.d
            );
        }
        // A held-out projection carries one baked-in RHT setting ("wts"); without
        // one we sweep RHT off/on to compare the incoherence transform (axis A2).
        let cases: Vec<(&str, Option<&LearnedModel>, bool)> = match preloaded.as_ref() {
            Some(m) => vec![("wts", Some(m), false)],
            None => vec![("off", None, false), ("on", None, true)],
        };
        for &warm in &[false, true] {
            for &(rht_label, pm, rht) in &cases {
                let (cos, rl2, klv, captured) = evaluate(r, pm, warm, rht, codec);
                let go = !warm && cos >= GO_COSINE && klv <= GO_KL;
                any_hot_go |= go;
                let captured_col = if captured.is_nan() {
                    "—".to_string()
                } else {
                    format!("{:.1}%", captured * 100.0)
                };
                println!(
                    "  {:<12} {:<5} {:>4} | {:>8.4} {:>8.4} {:>9.4} {:>8} | {}",
                    r.label,
                    if warm { "WARM" } else { "HOT" },
                    rht_label,
                    cos,
                    rl2,
                    klv,
                    captured_col,
                    if warm {
                        "—"
                    } else if go {
                        "GO ✅"
                    } else {
                        "NO-GO ❌"
                    },
                );
            }
        }
    }

    println!("\n  Seuils GO (HOT) : cos ≥ {GO_COSINE}, KL ≤ {GO_KL}.");
    println!(
        "  VERDICT : {}",
        if any_hot_go {
            "régime HOT viable trouvé (≥1 config sous les seuils) — GO pour la Phase 1."
        } else {
            "aucun régime HOT sous les seuils — ajuster (d_c/d_s/λ, RHT, résidu multi-bit) AVANT d'intégrer."
        }
    );
    println!(
        "\n  Honnêteté : (1) PROXY couche-isolée, pas une vraie perplexité ; (2) sans\n  \
         `--dump`, clés synthétiques réalistes — branche `scripts/dump_activations.py`\n  \
         sur un vrai modèle pour le chiffre réel ; (3) projection APPRISE (PCA),\n  \
         contrairement aux études §7 (JL aléatoire) ; (4) sans `--weights`, la\n  \
         projection est réajustée sur les clés testées (optimiste) — passe une\n  \
         projection tenue à l'écart pour le chiffre honnête."
    );
}

/// Build the regimes: one from a real activation dump, or several synthetic
/// spectra (flatter decay ⇒ harder for a low-rank base).
fn build_regimes(dump: Option<String>) -> (String, Vec<Regime>) {
    if let Some(dir) = dump {
        let (keys, d) = load_bin(&format!("{dir}/k.bin"));
        let (values, dv) = load_bin(&format!("{dir}/v.bin"));
        let (queries, dq) = load_bin(&format!("{dir}/q.bin"));
        assert_eq!(d, dq, "q and k must share the key dimension");
        assert!(
            d > D_C,
            "SLHA compresses a key of dim > {D_C} into a {D_C}-dim latent + residual; \
             got d={d}. Dump a wider key representation (e.g. pre-projection width)."
        );
        let source = format!(
            "activations RÉELLES ({dir}) — d={d}, n={}, d_v={dv}, {} requêtes",
            keys.len(),
            queries.len()
        );
        return (
            source,
            vec![Regime {
                label: "réel".into(),
                keys,
                values,
                queries,
                d,
                dv,
            }],
        );
    }

    // Realistic synthetic: spectral decay + outlier noise (learned::gen_keys),
    // the repo's model of real key distributions. d=256 > D_C=128 so the
    // low-rank base + residual pipeline is exercised (as in the §7 learned study).
    let (d, dv, n, nq) = (256usize, 64usize, 512usize, 64usize);
    let mut rng = Rng::new(7);
    let values: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut v = vec![0.0f32; dv];
            rng.fill_gaussian(&mut v);
            v
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| {
            let mut q = vec![0.0f32; d];
            rng.fill_gaussian(&mut q);
            q
        })
        .collect();
    let regimes: Vec<Regime> = [0.99f32, 0.95, 0.90, 0.80]
        .iter()
        .map(|&decay| Regime {
            label: format!("decay={decay:.2}"),
            keys: gen_keys(20, n, d, d, decay, 0.02),
            values: values.clone(),
            queries: queries.clone(),
            d,
            dv,
        })
        .collect();
    let source = format!(
        "synthétique RÉALISTE (gen_keys, spectre décroissant) — d={d}, n={n}, d_v={dv}, {nq} requêtes"
    );
    (source, regimes)
}

/// Load a `.bin` matrix written by `scripts/dump_activations.py`:
/// `[u32 magic=0x534C4841][u32 rows][u32 cols][f32 rows*cols row-major, LE]`.
fn load_bin(path: &str) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert!(bytes.len() >= 12, "{path}: truncated header");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(
        magic, 0x534C_4841,
        "{path}: bad magic (not an SLHA activation dump)"
    );
    let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 12 + rows * cols * 4, "{path}: size mismatch");
    let mut out = Vec::with_capacity(rows);
    let mut off = 12;
    for _ in 0..rows {
        let mut row = vec![0.0f32; cols];
        for c in row.iter_mut() {
            *c = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
        }
        out.push(row);
    }
    (out, cols)
}
