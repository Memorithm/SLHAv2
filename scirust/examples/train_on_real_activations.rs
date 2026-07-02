//! Phase 1 (§1.1) — train a **learned** projection on real (or realistic) keys,
//! save it to a portable weights file ([`scirust::weights`]), and reload it to
//! prove the round-trip. The saved `.slhw` file is the reusable projection
//! artifact an LLM integration would ship (train once, load everywhere).
//!
//! Run (realistic synthetic keys):
//! ```text
//! cargo run --release --example train_on_real_activations
//! ```
//! On REAL activations (see `scripts/dump_activations.py`, uses `k.bin`):
//! ```text
//! cargo run --release --example train_on_real_activations -- --dump DIR --rht --out proj.slhw
//! ```
//! With `--joint` the projection is fitted on the pooled second moment of the
//! keys **and** the real queries (`q.bin`, plan §1.3): the coarse score is
//! `⟨Pq, Pk⟩`, so query energy outside the key subspace is otherwise lost —
//! measured on GPT-2, a keys-only PCA keeps just ~70% of real-query energy:
//! ```text
//! cargo run --release --example train_on_real_activations -- --dump DIR --joint --out proj.slhw
//! ```
//! With `--sgd` the (joint-)PCA solution is then **refined by score-aware SGD**
//! ([`scirust::learned::train_projection`]) on the real `q.bin`/`k.bin` pairs —
//! it minimises the *score* error `E[(⟨q,k⟩ − ⟨Pq,Pk⟩)²]`, not reconstruction.
//! Defaults match the synthetic study (`learn_projection`): 200 epochs,
//! lr 2e-3 (linear decay), batch 64; override with `--sgd-epochs N`,
//! `--sgd-lr X`, `--sgd-batch N`.
//!
//! ⚠️ **On real activations this lever is a measured negative** (see
//! `docs/TURBOQUANT.md` §3quater): the synthetic lr (2e-3) diverges to NaN on
//! GPT-2 magnitudes — real dumps need `--sgd-lr 1e-6` to converge — and even
//! then the refinement overfits the train-set score error and *degrades*
//! held-out fidelity below plain joint-PCA. The flag is kept for reproducibility
//! (it converges on synthetic data), not because it helps on real dumps.
//! ```text
//! cargo run --release --example train_on_real_activations -- --dump DIR --joint --sgd --out proj.slhw
//! ```

// Numeric loops read closer to the math with indexing.
#![allow(clippy::needless_range_loop)]

use scirust::attention::slha_v2::D_C;
use scirust::learned::{gen_keys, train_projection, LearnedModel};
use scirust::metrics::{cosine, dot, softmax_into};
use scirust::rng::Rng;
use scirust::weights;

/// Sampling seed of the SGD refinement — same as the synthetic
/// `learn_projection` study, so the two runs are directly comparable.
const SGD_SEED: u64 = 7;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let dump = opt("--dump");
    let rht = args.iter().any(|a| a == "--rht");
    let joint = args.iter().any(|a| a == "--joint");
    let sgd = args.iter().any(|a| a == "--sgd");
    let sgd_epochs: usize = opt("--sgd-epochs")
        .map(|v| v.parse().expect("--sgd-epochs: entier attendu"))
        .unwrap_or(200);
    let sgd_lr: f32 = opt("--sgd-lr")
        .map(|v| v.parse().expect("--sgd-lr: flottant attendu"))
        .unwrap_or(2.0e-3);
    let sgd_batch: usize = opt("--sgd-batch")
        .map(|v| v.parse().expect("--sgd-batch: entier attendu"))
        .unwrap_or(64);
    let out = opt("--out").unwrap_or_else(|| {
        std::env::temp_dir()
            .join("slha_projection.slhw")
            .to_string_lossy()
            .into_owned()
    });
    let seed = 0x0000_5107_u64;

    // Training keys: real dump or realistic synthetic (spectral decay + outliers).
    let (keys, d, source) = match &dump {
        Some(dir) => {
            let (k, d) = load_k_bin(&format!("{dir}/k.bin"));
            (k, d, format!("activations RÉELLES {dir}/k.bin"))
        }
        None => {
            assert!(
                !joint && !sgd,
                "--joint / --sgd nécessitent --dump (ils apprennent sur les VRAIES requêtes q.bin)"
            );
            let d = 256usize;
            let k = gen_keys(1, 2000, d, d, 0.9, 0.02);
            (k, d, "synthétique réaliste (gen_keys)".to_string())
        }
    };
    assert!(
        d > D_C,
        "SLHA needs a key dim > {D_C}; got d={d} (dump a wider key representation)."
    );
    // Real queries for the joint (score-aware) objective and/or the SGD refine.
    let queries: Vec<Vec<f32>> = if joint || sgd {
        let dir = dump.as_ref().unwrap();
        let (q, dq) = load_k_bin(&format!("{dir}/q.bin"));
        assert_eq!(dq, d, "q.bin and k.bin must share the key dimension");
        q
    } else {
        Vec::new()
    };

    println!("== Phase 1 — entraînement + sauvegarde d'une projection APPRISE ==\n");
    println!("  clés d'entraînement : {source} — n={}, d={d}", keys.len());
    println!(
        "  objectif             : {}",
        if joint {
            format!(
                "sous-espace JOINT clés+requêtes (--joint, §1.3) — {} requêtes",
                queries.len()
            )
        } else {
            "clés seules (PCA)".to_string()
        }
    );
    println!(
        "  incohérence RHT (A2) : {}",
        if rht { "activée" } else { "désactivée" }
    );
    println!(
        "  raffinage SGD (§1.3) : {}\n",
        if sgd {
            format!(
                "activé — {sgd_epochs} époques, lr {sgd_lr}, batch {sgd_batch}, graine {SGD_SEED}"
            )
        } else {
            "désactivé".to_string()
        }
    );

    // Train (PCA on keys, or on the pooled keys+queries second moment).
    let model = if joint {
        LearnedModel::fit_joint(&keys, &queries, d, seed, false, rht)
    } else {
        LearnedModel::fit_with(&keys, d, seed, false, rht)
    };
    println!(
        "  énergie captée par le sous-espace top-{D_C} : {:.2}%{}",
        model.captured_energy * 100.0,
        if joint {
            "  (énergie POOLÉE clés+requêtes)"
        } else {
            ""
        }
    );

    // Optional score-aware SGD refinement, warm-started from the PCA solution.
    // Same Z seed ⇒ only the projection P differs from the PCA model.
    let model = if sgd {
        let (p, hist) = train_projection(
            &queries,
            &keys,
            model.projection().to_vec(),
            sgd_epochs,
            sgd_lr,
            sgd_batch,
            SGD_SEED,
        );
        println!(
            "  SGD score-aware        : perte {:.4} → {:.4} (min {:.4}) sur {} époques",
            hist.first().copied().unwrap_or(f32::NAN),
            hist.last().copied().unwrap_or(f32::NAN),
            hist.iter().copied().fold(f32::INFINITY, f32::min),
            hist.len()
        );
        LearnedModel::from_projection_with(p, d, seed, rht)
    } else {
        model
    };

    // Save → reload → prove the round-trip is exact.
    weights::save(&out, &model, seed, rht).expect("save weights");
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    let reloaded = weights::load(&out).expect("load weights");
    let identical = model.projection() == reloaded.projection();
    println!("  poids écrits          : {out}  ({bytes} o)");
    println!(
        "  rechargement          : projection {}\n",
        if identical {
            "identique ✅"
        } else {
            "DIFFÉRENTE ❌"
        }
    );

    // Quick fidelity sanity: attention-output cosine of the trained projection
    // vs full precision, HOT, on a held-out query/value set.
    let (dv, nq, nctx) = (64usize, 32usize, keys.len().min(512));
    let ctx = &keys[..nctx];
    let mut rng = Rng::new(9);
    let values: Vec<Vec<f32>> = (0..nctx)
        .map(|_| {
            let mut v = vec![0.0f32; dv];
            rng.fill_gaussian(&mut v);
            v
        })
        .collect();
    let tiles: Vec<_> = ctx
        .iter()
        .enumerate()
        .map(|(i, k)| model.encode(k, i as u32, false))
        .collect();
    let scale = 1.0 / (d as f32).sqrt();
    let mut cos = 0.0f32;
    for _ in 0..nq {
        let mut q = vec![0.0f32; d];
        rng.fill_gaussian(&mut q);
        let qc = model.query_coarse(&q);
        let qs = model.sign_bits(&q);
        let s_true: Vec<f32> = ctx.iter().map(|k| dot(&q, k)).collect();
        let s_slha: Vec<f32> = tiles.iter().map(|t| t.compute_score(&qc, &qs)).collect();
        cos += cosine(
            &attn(&s_true, &values, scale, dv),
            &attn(&s_slha, &values, scale, dv),
        );
    }
    println!(
        "  fidélité de sortie HOT (cos vs FP32) : {:.4}",
        cos / nq as f32
    );
    println!(
        "\n  → Projection réutilisable prête. Charge-la ailleurs via `scirust::weights::load`\n  \
         (ou lance `offline_validation --dump …` pour le GO/NO-GO complet sur activations réelles)."
    );
}

fn attn(scores: &[f32], values: &[Vec<f32>], scale: f32, dv: usize) -> Vec<f32> {
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

/// Load a `k.bin` matrix (same format as `scripts/dump_activations.py`):
/// `[u32 magic=0x534C4841][u32 rows][u32 cols][f32 rows*cols row-major, LE]`.
fn load_k_bin(path: &str) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert!(bytes.len() >= 12, "{path}: truncated header");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic, 0x534C_4841, "{path}: bad magic");
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
