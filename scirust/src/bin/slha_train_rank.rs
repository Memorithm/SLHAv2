//! Atomic ranking-aware trainer for the SLHA projection.
//!
//! Reads the offline ranking dataset produced by `slha_rank_dataset`, trains one
//! projection per layer with the selected objective, and publishes the result
//! atomically. Argument validation is FAIL-CLOSED: an incompatible combination
//! is an error before any weight is written, never a silent fallback.
//!
//! Baseline logits are used here, offline, as training labels only. Nothing this
//! binary produces lets the inference path read a baseline score.

use scirust::attention::slha_v2::D_C;
use scirust::learned::LearnedModel;
use scirust::ranking::{parse_layer_set, train_ranking, Geometry, Objective, Row, TrainConfig};
use scirust::weights;
use std::collections::BTreeMap;
use std::io::Write;

const MAGIC: u32 = 0x534C_4841;
const HDR: usize = 40;
const MAX_TOPK: usize = 64;

struct Args {
    objective: String,
    top_k: usize,
    layers: String,
    seed: u64,
    lr: f32,
    epochs: usize,
    steps: usize,
    w_pairwise: f32,
    w_listwise: f32,
    w_l2: f32,
    w_geometry: f32,
    margin: f32,
    max_negatives: usize,
    negative_policy: String,
    dataset: String,
    split: String,
    initial_weights: String,
    output: String,
    training_manifest: String,
    validation_manifest: String,
    n_layers: usize,
    batch: usize,
    max_keys: usize,
}

fn fail(msg: impl AsRef<str>) -> ! {
    eprintln!("ERROR: {}", msg.as_ref());
    std::process::exit(2);
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut m: BTreeMap<String, String> = BTreeMap::new();
    let mut i = 0;
    while i < raw.len() {
        let k = raw[i].clone();
        if !k.starts_with("--") {
            fail(format!("unexpected positional argument {k:?}"));
        }
        let v = raw
            .get(i + 1)
            .unwrap_or_else(|| fail(format!("{k} requires a value")))
            .clone();
        if m.insert(k.clone(), v).is_some() {
            fail(format!("{k} given more than once"));
        }
        i += 2;
    }
    let get = |k: &str, d: Option<&str>| -> String {
        m.get(k)
            .cloned()
            .or_else(|| d.map(str::to_string))
            .unwrap_or_else(|| fail(format!("missing required argument {k}")))
    };
    let num = |k: &str, d: Option<&str>| -> f64 {
        let s = get(k, d);
        s.parse::<f64>()
            .unwrap_or_else(|_| fail(format!("{k}: expected a number, got {s:?}")))
    };
    let a = Args {
        objective: get("--objective", None),
        top_k: num("--top-k", Some("0")) as usize,
        layers: get("--layers", None),
        seed: num("--seed", Some("7")) as u64,
        lr: num("--learning-rate", Some("1e-9")) as f32,
        epochs: num("--epochs", Some("2")) as usize,
        steps: num("--steps", Some("0")) as usize,
        w_pairwise: num("--pairwise-weight", Some("1.0")) as f32,
        w_listwise: num("--listwise-weight", Some("0.5")) as f32,
        w_l2: num("--l2-weight", Some("0.1")) as f32,
        w_geometry: num("--geometry-weight", Some("0.25")) as f32,
        margin: num("--margin", Some("1.0")) as f32,
        max_negatives: num("--max-negatives", Some("8")) as usize,
        negative_policy: get("--negative-policy", Some("boundary-then-seeded")),
        dataset: get("--dataset", None),
        split: get("--split-chunks", None),
        initial_weights: get("--initial-weights", None),
        output: get("--output", None),
        training_manifest: get("--training-manifest", None),
        validation_manifest: get("--validation-manifest", Some("")),
        n_layers: num("--n-layers", Some("28")) as usize,
        batch: num("--batch", Some("16")) as usize,
        max_keys: num("--max-keys", Some("256")) as usize,
    };
    validate(&a, &m);
    a
}

/// Fail-closed compatibility checks. Every rejection happens before any file is
/// created, so an invalid invocation can never leave a partial variant behind.
fn validate(a: &Args, given: &BTreeMap<String, String>) {
    let known = [
        "l2".to_string(),
        "pairwise-topk".to_string(),
        "listwise".to_string(),
        "hybrid".to_string(),
    ];
    if !known.contains(&a.objective) {
        fail(format!(
            "unknown --objective {:?}; expected one of {}",
            a.objective,
            known.join(", ")
        ));
    }
    let topk_family = a.objective == "pairwise-topk" || a.objective == "hybrid";
    // l2 must not be given top-k-specific parameters
    if !topk_family {
        for k in [
            "--top-k",
            "--margin",
            "--max-negatives",
            "--negative-policy",
        ] {
            if given.contains_key(k) {
                fail(format!("--objective {} does not accept {k}", a.objective));
            }
        }
    }
    if a.objective == "l2" {
        for k in [
            "--pairwise-weight",
            "--listwise-weight",
            "--geometry-weight",
        ] {
            if given.contains_key(k) {
                fail(format!("--objective l2 does not accept {k}"));
            }
        }
    }
    if topk_family {
        if a.top_k == 0 {
            fail("--top-k must be at least 1 for this objective");
        }
        if a.top_k > MAX_TOPK {
            fail(format!(
                "--top-k {} exceeds the maximum {MAX_TOPK}",
                a.top_k
            ));
        }
        if a.max_negatives == 0 {
            fail("--max-negatives must be at least 1");
        }
        if a.negative_policy != "boundary-then-seeded" {
            fail(format!(
                "unknown --negative-policy {:?}; only boundary-then-seeded is implemented",
                a.negative_policy
            ));
        }
    }
    for (k, v) in [
        ("--learning-rate", a.lr),
        ("--pairwise-weight", a.w_pairwise),
        ("--listwise-weight", a.w_listwise),
        ("--l2-weight", a.w_l2),
        ("--geometry-weight", a.w_geometry),
        ("--margin", a.margin),
    ] {
        if !v.is_finite() {
            fail(format!("{k} must be finite, got {v}"));
        }
        if v < 0.0 {
            fail(format!("{k} must not be negative, got {v}"));
        }
    }
    if a.margin <= 0.0 && (a.objective == "pairwise-topk" || a.objective == "hybrid") {
        fail("--margin must be strictly positive");
    }
    if a.epochs == 0 || a.batch == 0 || a.max_keys == 0 {
        fail("--epochs, --batch and --max-keys must all be at least 1");
    }
    match parse_layer_set(&a.layers, a.n_layers) {
        Ok(v) if v.is_empty() => fail("--layers selects no layer; an empty scope is rejected"),
        Ok(_) => {}
        Err(e) => fail(format!("--layers: {e}")),
    }
    if !std::path::Path::new(&a.dataset).is_dir() {
        fail(format!("--dataset {:?} is not a directory", a.dataset));
    }
    if !std::path::Path::new(&format!("{}/rank_dataset_manifest.json", a.dataset)).exists() {
        fail(
            "--dataset has no rank_dataset_manifest.json; refusing to train on an unmanifested set",
        );
    }
    if !std::path::Path::new(&a.initial_weights).is_dir() {
        fail(format!(
            "--initial-weights {:?} is not a directory",
            a.initial_weights
        ));
    }
    for c in a.split.split(',') {
        if c.trim().parse::<usize>().is_err() {
            fail(format!("--split-chunks: {c:?} is not a chunk index"));
        }
    }
}

struct LayerRows {
    q: Vec<f32>,
    keys: Vec<Vec<f32>>, // per chunk
    b: Vec<f32>,
    nvis: Vec<i32>,
    chunk: Vec<i32>,
    q_dim: usize,
    key_dim: usize,
    rows: usize,
}

fn read_layer(path: &str) -> Result<LayerRows, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if raw.len() < HDR {
        return Err(format!("{path}: truncated header"));
    }
    let u32at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    let u64at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap()) as usize;
    if u32at(0) != MAGIC as usize || u32at(4) != 2 {
        return Err(format!("{path}: bad magic or version"));
    }
    let q_dim = u32at(12);
    let rows = u64at(16);
    let n_chunks = u64at(24);
    let key_dim = u32at(32);
    let mut o = HDR;
    let mut key_rows = Vec::with_capacity(n_chunks);
    for _ in 0..n_chunks {
        key_rows.push(u64at(o));
        o += 8;
    }
    o += rows * 4; // head
    o += rows * 4; // gqa
    o += rows * 8; // token
    let mut nvis = Vec::with_capacity(rows);
    for i in 0..rows {
        nvis.push(i32::from_le_bytes(
            raw[o + i * 4..o + i * 4 + 4].try_into().unwrap(),
        ));
    }
    o += rows * 4;
    let mut chunk = Vec::with_capacity(rows);
    for i in 0..rows {
        chunk.push(i32::from_le_bytes(
            raw[o + i * 4..o + i * 4 + 4].try_into().unwrap(),
        ));
    }
    o += rows * 4;
    let f32s = |o: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| f32::from_le_bytes(raw[o + i * 4..o + i * 4 + 4].try_into().unwrap()))
            .collect()
    };
    let q = f32s(o, rows * q_dim);
    o += rows * q_dim * 4;
    let total: usize = nvis.iter().map(|&v| v as usize).sum();
    let b = f32s(o, total);
    o += total * 4;
    o += total * 4; // slha scores, not needed for training
    let mut keys = Vec::with_capacity(n_chunks);
    for &n in &key_rows {
        keys.push(f32s(o, n * key_dim));
        o += n * key_dim * 4;
    }
    if o != raw.len() {
        return Err(format!("{path}: trailing bytes ({o} != {})", raw.len()));
    }
    Ok(LayerRows {
        q,
        keys,
        b,
        nvis,
        chunk,
        q_dim,
        key_dim,
        rows,
    })
}

fn sha256_file(path: &str) -> String {
    let out = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum");
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string()
}

fn main() {
    let a = parse_args();
    let scope = parse_layer_set(&a.layers, a.n_layers).unwrap();
    let split: Vec<i32> = a
        .split
        .split(',')
        .map(|c| c.trim().parse::<i32>().unwrap())
        .collect();

    let base_obj = Objective::parse(&a.objective, a.top_k).unwrap_or_else(|e| fail(e));
    let objective = match base_obj {
        Objective::PairwiseTopK { k, .. } => Objective::PairwiseTopK {
            k,
            tau: a.margin,
            negatives: a.max_negatives,
        },
        Objective::Hybrid {
            k,
            t_teacher,
            t_student,
            ..
        } => Objective::Hybrid {
            k,
            tau: a.margin,
            negatives: a.max_negatives,
            t_teacher,
            t_student,
            w_pairwise: a.w_pairwise,
            w_listwise: a.w_listwise,
            w_l2: a.w_l2,
        },
        other => other,
    };

    // stage directory; published only after full validation
    let stage = format!("{}.stage.{}", a.output, std::process::id());
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).unwrap_or_else(|e| fail(format!("cannot create stage: {e}")));

    let mut per_layer = Vec::new();
    let mut loss_history: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut rows_seen_total = 0u64;
    let mut pair_total = 0u64;
    let t0 = std::time::Instant::now();

    for layer in 0..a.n_layers {
        let src = format!("{}/layer-{layer:03}.slhw", a.initial_weights);
        let init = match weights::load(&src) {
            Ok(m) => m,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&stage);
                fail(format!("cannot load {src}: {e}"));
            }
        };
        // seed and RHT flag live in the weights header; preserve them exactly
        let hdr = std::fs::read(&src).unwrap();
        let wseed = u64::from_le_bytes(hdr[12..20].try_into().unwrap());
        let wrht = hdr[20] == 1;
        let d = init.d;

        let trained = if scope.contains(&layer) {
            let path = format!("{}/rank-layer-{layer:03}.bin", a.dataset);
            let data = match read_layer(&path) {
                Ok(v) => v,
                Err(e) => {
                    let _ = std::fs::remove_dir_all(&stage);
                    fail(e);
                }
            };
            if data.q_dim != d || data.key_dim != d {
                let _ = std::fs::remove_dir_all(&stage);
                fail(format!(
                    "layer {layer}: dataset dim {}/{} != weight dim {d}",
                    data.q_dim, data.key_dim
                ));
            }
            let mut starts = Vec::with_capacity(data.rows + 1);
            let mut acc = 0usize;
            for &n in &data.nvis {
                starts.push(acc);
                acc += n as usize;
            }
            starts.push(acc);
            let rows: Vec<Row<'_>> = (0..data.rows)
                .filter(|&i| split.contains(&data.chunk[i]))
                .filter(|&i| {
                    let c = data.chunk[i] as usize;
                    let nv = data.nvis[i] as usize;
                    c < data.keys.len() && data.keys[c].len() >= nv * data.key_dim
                })
                .map(|i| {
                    let c = data.chunk[i] as usize;
                    let nv = data.nvis[i] as usize;
                    Row {
                        q: &data.q[i * data.q_dim..(i + 1) * data.q_dim],
                        keys: &data.keys[c][..nv * data.key_dim],
                        baseline: &data.b[starts[i]..starts[i + 1]],
                        n_visible: nv,
                        d,
                    }
                })
                .collect();
            let cfg = TrainConfig {
                objective: objective.clone(),
                geometry: Geometry {
                    weight: if a.objective == "l2" {
                        0.0
                    } else {
                        a.w_geometry
                    },
                },
                epochs: a.epochs,
                lr: a.lr,
                batch: a.batch,
                seed: a.seed,
                max_keys: a.max_keys,
            };
            let (p, hist) = train_ranking(&rows, init.projection().to_vec(), &cfg);
            rows_seen_total += hist.rows_seen;
            pair_total += hist.pairwise_comparisons;
            loss_history.insert(layer, hist.epoch_loss.clone());
            if p.iter().any(|v| !v.is_finite()) {
                let _ = std::fs::remove_dir_all(&stage);
                fail(format!("layer {layer}: trained projection is not finite"));
            }
            LearnedModel::from_projection_with(p, d, wseed, wrht)
        } else {
            init
        };

        let dst = format!("{stage}/layer-{layer:03}.slhw");
        if let Err(e) = weights::save(&dst, &trained, wseed, wrht) {
            let _ = std::fs::remove_dir_all(&stage);
            fail(format!("cannot write {dst}: {e}"));
        }
        per_layer.push((layer, sha256_file(&dst)));
        assert_eq!(trained.projection().len(), D_C * d);
    }

    // ---- validation before publication ----
    if per_layer.len() != a.n_layers {
        let _ = std::fs::remove_dir_all(&stage);
        fail("staged weight count does not match the expected layer count");
    }
    let agg_input: String = per_layer
        .iter()
        .map(|(l, h)| format!("{l}:{h}"))
        .collect::<Vec<_>>()
        .join(",");
    let aggregate = {
        let mut c = std::process::Command::new("sha256sum")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("sha256sum");
        c.stdin
            .as_mut()
            .unwrap()
            .write_all(agg_input.as_bytes())
            .unwrap();
        let o = c.wait_with_output().unwrap();
        String::from_utf8_lossy(&o.stdout)
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string()
    };

    let loss_json = {
        let mut t = String::from("{");
        for (i, (l, v)) in loss_history.iter().enumerate() {
            if i > 0 {
                t.push(',');
            }
            t.push_str(&format!("\"{l}\":{v:?}"));
        }
        t.push('}');
        t
    };
    let per_layer_json = per_layer
        .iter()
        .map(|(l, h)| format!("\"{l}\":\"{h}\""))
        .collect::<Vec<_>>()
        .join(",");
    let mut lines: Vec<String> = Vec::new();
    lines.push("{".into());
    lines.push("  \"schema\": \"slha_rank_training_manifest_v1\",".into());
    lines.push(format!("  \"objective\": \"{}\",", a.objective));
    lines.push(format!("  \"top_k\": {},", a.top_k));
    lines.push(format!("  \"layers\": \"{}\",", a.layers));
    lines.push(format!("  \"trained_layers\": {:?},", scope));
    lines.push(format!("  \"seed\": {},", a.seed));
    lines.push(format!("  \"learning_rate\": {},", a.lr));
    lines.push(format!("  \"epochs\": {},", a.epochs));
    lines.push(format!("  \"batch\": {},", a.batch));
    lines.push(format!("  \"max_keys\": {},", a.max_keys));
    lines.push(format!("  \"margin\": {},", a.margin));
    lines.push(format!("  \"max_negatives\": {},", a.max_negatives));
    lines.push(format!("  \"negative_policy\": \"{}\",", a.negative_policy));
    lines.push(format!("  \"w_pairwise\": {},", a.w_pairwise));
    lines.push(format!("  \"w_listwise\": {},", a.w_listwise));
    lines.push(format!("  \"w_l2\": {},", a.w_l2));
    lines.push(format!("  \"w_geometry\": {},", a.w_geometry));
    lines.push(format!("  \"dataset\": \"{}\",", a.dataset));
    lines.push(format!("  \"split_chunks\": {:?},", split));
    lines.push(format!("  \"initial_weights\": \"{}\",", a.initial_weights));
    lines.push(format!("  \"source_rows_processed\": {},", rows_seen_total));
    lines.push(format!("  \"pairwise_comparisons\": {},", pair_total));
    lines.push(format!(
        "  \"optimizer_steps\": {},",
        if a.steps > 0 {
            a.epochs * a.steps
        } else {
            a.epochs * (rows_seen_total as usize / a.batch).max(1)
        }
    ));
    lines.push(format!("  \"steps_override\": {},", a.steps));
    lines.push(format!(
        "  \"wall_time_seconds\": {:.3},",
        t0.elapsed().as_secs_f64()
    ));
    lines.push(format!("  \"loss_history\": {},", loss_json));
    lines.push(format!("  \"per_layer_sha256\": {{{}}},", per_layer_json));
    lines.push(format!("  \"aggregate_sha256\": \"{}\",", aggregate));
    lines.push("  \"valid\": true".into());
    lines.push("}".into());
    let manifest = lines.join("\n") + "\n";
    std::fs::write(format!("{stage}/manifest.json"), &manifest)
        .unwrap_or_else(|e| fail(format!("cannot write manifest: {e}")));
    std::fs::write(&a.training_manifest, &manifest)
        .unwrap_or_else(|e| fail(format!("cannot write training manifest: {e}")));
    if !a.validation_manifest.is_empty() {
        std::fs::write(&a.validation_manifest, &manifest).ok();
    }

    // ---- atomic publication ----
    if std::path::Path::new(&a.output).exists() {
        let prev = format!("{}.prev.{}", a.output, std::process::id());
        std::fs::rename(&a.output, &prev)
            .unwrap_or_else(|e| fail(format!("cannot move aside: {e}")));
        match std::fs::rename(&stage, &a.output) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&prev);
            }
            Err(e) => {
                let _ = std::fs::rename(&prev, &a.output);
                let _ = std::fs::remove_dir_all(&stage);
                fail(format!("cannot publish: {e}"));
            }
        }
    } else {
        std::fs::rename(&stage, &a.output).unwrap_or_else(|e| fail(format!("cannot publish: {e}")));
    }
    println!("PUBLISHED {} aggregate={}", a.output, aggregate);
}
