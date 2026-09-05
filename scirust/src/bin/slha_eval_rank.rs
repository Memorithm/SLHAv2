//! Held-out mechanistic evaluator for the frozen SLHA-LR1 candidate.
//!
//! This binary never trains. It reads the offline rank dataset, compares the
//! recorded control scores with a frozen candidate projection on the exact same
//! validation rows, and writes machine-readable ranking/geometry evidence.
//! Baseline logits remain offline labels only.

use scirust::attention::slha_v2::LatentCodec;
use scirust::metrics::spearman;
use scirust::rank_dataset::read_layer;
use scirust::weights;
use std::collections::BTreeMap;
use std::path::Path;

const EXPECTED_TOP_K: usize = 16;
const EXPECTED_LAYERS: usize = 6;
const EXPECTED_STORAGE_SLOTS: usize = 17;
const EXPECTED_POPULATED: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const EXPECTED_SPLIT: [usize; 4] = [13, 14, 15, 16];

struct Args {
    dataset: String,
    weights_dir: String,
    output: String,
    contract: String,
    split: Vec<usize>,
    top_k: usize,
    n_layers: usize,
    codec: String,
}

fn fail(msg: impl AsRef<str>) -> ! {
    eprintln!("ERROR: {}", msg.as_ref());
    std::process::exit(2)
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut values = BTreeMap::<String, String>::new();
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].clone();
        if !key.starts_with("--") {
            fail(format!("unexpected positional argument {key:?}"));
        }
        let value = raw
            .get(i + 1)
            .unwrap_or_else(|| fail(format!("{key} requires a value")))
            .clone();
        if values.insert(key.clone(), value).is_some() {
            fail(format!("{key} given more than once"));
        }
        i += 2;
    }

    let known = [
        "--dataset",
        "--weights-dir",
        "--output",
        "--contract",
        "--split-chunks",
        "--top-k",
        "--n-layers",
        "--codec",
    ];
    for key in values.keys() {
        if !known.contains(&key.as_str()) {
            fail(format!("unknown option {key}"));
        }
    }

    let get = |key: &str, default: Option<&str>| -> String {
        values
            .get(key)
            .cloned()
            .or_else(|| default.map(str::to_owned))
            .unwrap_or_else(|| fail(format!("missing required option {key}")))
    };
    let parse_usize = |key: &str, default: &str| -> usize {
        let text = get(key, Some(default));
        text.parse::<usize>().unwrap_or_else(|_| {
            fail(format!(
                "{key}: expected non-negative integer, got {text:?}"
            ))
        })
    };

    let split_text = get("--split-chunks", Some("13,14,15,16"));
    let split = split_text
        .split(',')
        .map(|value| {
            value.trim().parse::<usize>().unwrap_or_else(|_| {
                fail(format!(
                    "--split-chunks: expected comma-separated chunk ids, got {split_text:?}"
                ))
            })
        })
        .collect::<Vec<_>>();

    let args = Args {
        dataset: get("--dataset", None),
        weights_dir: get("--weights-dir", None),
        output: get("--output", None),
        contract: get("--contract", None),
        split,
        top_k: parse_usize("--top-k", "16"),
        n_layers: parse_usize("--n-layers", "6"),
        codec: get("--codec", Some("mixed")),
    };
    validate_args(&args);
    args
}

fn validate_args(args: &Args) {
    if args.top_k != EXPECTED_TOP_K {
        fail(format!(
            "LR1 evaluator is frozen at top_k={EXPECTED_TOP_K}; got {}",
            args.top_k
        ));
    }
    if args.n_layers != EXPECTED_LAYERS {
        fail(format!(
            "LR1 evaluator is frozen at n_layers={EXPECTED_LAYERS}; got {}",
            args.n_layers
        ));
    }
    if args.codec != "mixed" {
        fail(format!(
            "LR1 evaluator is frozen at codec=mixed; got {:?}",
            args.codec
        ));
    }
    if args.split != EXPECTED_SPLIT {
        fail(format!(
            "LR1 validation split is frozen at {:?}; got {:?}",
            EXPECTED_SPLIT, args.split
        ));
    }
    if !Path::new(&args.dataset).is_dir() {
        fail(format!("--dataset {:?} is not a directory", args.dataset));
    }
    if !Path::new(&args.weights_dir).is_dir() {
        fail(format!(
            "--weights-dir {:?} is not a directory",
            args.weights_dir
        ));
    }
    let dataset_manifest = format!("{}/rank_dataset_manifest.json", args.dataset);
    let weights_manifest = format!("{}/manifest.json", args.weights_dir);
    for path in [
        args.contract.as_str(),
        dataset_manifest.as_str(),
        weights_manifest.as_str(),
    ] {
        if !Path::new(path).is_file() {
            fail(format!("required manifest/file is missing: {path}"));
        }
    }
}

fn sha256_file(path: &str) -> Result<String, String> {
    let output = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| format!("cannot execute sha256sum for {path}: {e}"))?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {path}"));
    }
    let hash = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_owned();
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("invalid sha256sum output for {path}"));
    }
    Ok(hash)
}

fn git_head() -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| format!("cannot execute git rev-parse HEAD: {e}"))?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed".into());
    }
    let head = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if head.len() != 40 || !head.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("invalid git HEAD {head:?}"));
    }
    Ok(head)
}

fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    indices.truncate(k.min(values.len()));
    indices
}

fn stddev(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

#[derive(Default)]
struct Summary {
    rows: u64,
    topk_overlap_sum: f64,
    pair_score_sum: f64,
    pair_count: u64,
    spearman_sum: f64,
    geometry_ratio_sum: f64,
    geometry_log_error_sum: f64,
    geometry_rows: u64,
}

impl Summary {
    fn observe(&mut self, baseline: &[f32], scores: &[f32], k: usize) -> Result<(), String> {
        if baseline.len() != scores.len() || baseline.is_empty() {
            return Err("metric row has empty or mismatched score vectors".into());
        }
        if baseline.iter().chain(scores).any(|v| !v.is_finite()) {
            return Err("metric row contains non-finite score".into());
        }

        let base_top = topk_indices(baseline, k);
        let score_top = topk_indices(scores, k);
        let k_eff = base_top.len();
        let overlap = base_top
            .iter()
            .filter(|index| score_top.contains(index))
            .count();
        self.topk_overlap_sum += overlap as f64 / k_eff as f64;

        let mut is_top = vec![false; baseline.len()];
        for &index in &base_top {
            is_top[index] = true;
        }
        for &top in &base_top {
            for other in 0..scores.len() {
                if is_top[other] {
                    continue;
                }
                self.pair_count += 1;
                self.pair_score_sum += if scores[top] > scores[other] {
                    1.0
                } else if scores[top] == scores[other] {
                    0.5
                } else {
                    0.0
                };
            }
        }

        let row_spearman = spearman(scores, baseline);
        if !row_spearman.is_finite() {
            return Err("metric row produced non-finite Spearman correlation".into());
        }
        self.spearman_sum += f64::from(row_spearman);
        let base_std = stddev(baseline);
        let score_std = stddev(scores);
        if base_std > 0.0 && score_std > 0.0 {
            let ratio = score_std / base_std;
            self.geometry_ratio_sum += ratio;
            self.geometry_log_error_sum += ratio.ln().abs();
            self.geometry_rows += 1;
        }
        self.rows += 1;
        Ok(())
    }

    fn topk_overlap(&self) -> f64 {
        self.topk_overlap_sum / self.rows as f64
    }

    fn pair_accuracy(&self) -> f64 {
        if self.pair_count == 0 {
            0.0
        } else {
            self.pair_score_sum / self.pair_count as f64
        }
    }

    fn mean_spearman(&self) -> f64 {
        self.spearman_sum / self.rows as f64
    }

    fn mean_stddev_ratio(&self) -> f64 {
        if self.geometry_rows == 0 {
            0.0
        } else {
            self.geometry_ratio_sum / self.geometry_rows as f64
        }
    }

    fn mean_abs_log_stddev_ratio(&self) -> f64 {
        if self.geometry_rows == 0 {
            0.0
        } else {
            self.geometry_log_error_sum / self.geometry_rows as f64
        }
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let args = parse_args();
    let commit = git_head().unwrap_or_else(|e| fail(e));
    let contract_sha = sha256_file(&args.contract).unwrap_or_else(|e| fail(e));
    let dataset_manifest = format!("{}/rank_dataset_manifest.json", args.dataset);
    let dataset_manifest_sha = sha256_file(&dataset_manifest).unwrap_or_else(|e| fail(e));
    let weights_manifest = format!("{}/manifest.json", args.weights_dir);
    let weights_manifest_sha = sha256_file(&weights_manifest).unwrap_or_else(|e| fail(e));

    let mut control = Summary::default();
    let mut candidate = Summary::default();
    let mut per_layer_rows = Vec::<(usize, u64, String)>::new();

    for layer in 0..args.n_layers {
        let dataset_path = format!("{}/rank-layer-{layer:03}.bin", args.dataset);
        let weight_path = format!("{}/layer-{layer:03}.slhw", args.weights_dir);
        if !Path::new(&dataset_path).is_file() {
            fail(format!("missing rank dataset layer file {dataset_path}"));
        }
        if !Path::new(&weight_path).is_file() {
            fail(format!("missing candidate weight file {weight_path}"));
        }

        let rows = read_layer(&dataset_path).unwrap_or_else(|e| fail(e));
        if rows.layer != layer as u32 {
            fail(format!(
                "dataset layer header mismatch: file layer {layer}, header {}",
                rows.layer
            ));
        }
        rows.validate_chunk_layout(EXPECTED_STORAGE_SLOTS, &EXPECTED_POPULATED)
            .unwrap_or_else(|e| fail(format!("layer {layer}: {e}")));
        let model = weights::load(&weight_path).unwrap_or_else(|e| fail(e));
        if model.d != rows.q_dim || model.d != rows.key_dim {
            fail(format!(
                "layer {layer}: candidate d={} does not match dataset q/key dimensions {}/{}",
                model.d, rows.q_dim, rows.key_dim
            ));
        }

        let mut layer_rows = 0u64;
        for index in rows.indices_for_chunks(&args.split) {
            let row = rows.row(index).unwrap_or_else(|e| fail(e));
            if row.n_visible < args.top_k {
                fail(format!(
                    "layer {layer} row {index}: n_visible={} is smaller than frozen top_k={}",
                    row.n_visible, args.top_k
                ));
            }
            control
                .observe(row.baseline, row.control_scores, args.top_k)
                .unwrap_or_else(|e| fail(format!("layer {layer} row {index} control: {e}")));

            let query_coarse = model.query_coarse(row.q);
            let query_sign = model.sign_bits(row.q);
            let mut scores = Vec::with_capacity(row.n_visible);
            for (position, key) in row.keys.chunks_exact(row.key_dim).enumerate() {
                let tile = model.encode_with(key, position as u32, false, LatentCodec::Mixed);
                scores.push(tile.compute_score(&query_coarse, &query_sign));
            }
            if scores.len() != row.n_visible {
                fail(format!(
                    "layer {layer} row {index}: candidate scored {} keys, expected {}",
                    scores.len(),
                    row.n_visible
                ));
            }
            candidate
                .observe(row.baseline, &scores, args.top_k)
                .unwrap_or_else(|e| fail(format!("layer {layer} row {index} candidate: {e}")));
            layer_rows += 1;
        }
        if layer_rows == 0 {
            fail(format!(
                "layer {layer}: frozen validation chunks {:?} contain no rows",
                args.split
            ));
        }
        let weight_sha = sha256_file(&weight_path).unwrap_or_else(|e| fail(e));
        per_layer_rows.push((layer, layer_rows, weight_sha));
    }

    if control.rows != candidate.rows || control.rows == 0 {
        fail(format!(
            "control/candidate row accounting mismatch: control={} candidate={}",
            control.rows, candidate.rows
        ));
    }

    let per_layer_json = per_layer_rows
        .iter()
        .map(|(layer, rows, sha)| {
            format!("    {{\"layer\":{layer},\"rows\":{rows},\"weights_sha256\":\"{sha}\"}}")
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let split_json = args
        .split
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");

    let report = format!(
        concat!(
            "{{\n",
            "  \"schema\":\"slha_lr1_rank_validation_v1\",\n",
            "  \"status\":\"MECHANISTIC_VALIDATION_ONLY_NOT_QUALITY_PROMOTION\",\n",
            "  \"candidate_id\":\"slha-lr1-pairwise-top16-all-layers-v1\",\n",
            "  \"slhav2_commit\":\"{}\",\n",
            "  \"contract\":{{\"path\":\"{}\",\"sha256\":\"{}\"}},\n",
            "  \"dataset\":{{\"path\":\"{}\",\"manifest_sha256\":\"{}\",\"storage_slots\":17,\"populated_chunks\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\"validation_chunks\":[{}]}},\n",
            "  \"candidate_weights\":{{\"path\":\"{}\",\"manifest_sha256\":\"{}\"}},\n",
            "  \"configuration\":{{\"top_k\":{},\"layers\":{},\"codec\":\"mixed\"}},\n",
            "  \"rows\":{},\n",
            "  \"control\":{{\"source\":\"recorded_dataset_control_scores\",\"mean_topk_set_recall\":{:.12},\"topk_vs_rest_pair_accuracy\":{:.12},\"mean_spearman\":{:.12},\"mean_score_stddev_ratio\":{:.12},\"mean_abs_log_score_stddev_ratio\":{:.12}}},\n",
            "  \"candidate\":{{\"source\":\"frozen_weights_rescored_with_deployable_mixed_codec_math\",\"mean_topk_set_recall\":{:.12},\"topk_vs_rest_pair_accuracy\":{:.12},\"mean_spearman\":{:.12},\"mean_score_stddev_ratio\":{:.12},\"mean_abs_log_score_stddev_ratio\":{:.12}}},\n",
            "  \"delta_candidate_minus_control\":{{\"mean_topk_set_recall\":{:.12},\"topk_vs_rest_pair_accuracy\":{:.12},\"mean_spearman\":{:.12}}},\n",
            "  \"per_layer\":[\n{}\n  ],\n",
            "  \"limitations\":[\"Offline mechanistic validation only; not perplexity or end-to-end quality evidence.\",\"Baseline logits are offline labels and are not available to deployable external-K inference.\",\"No hyperparameter, top-k, layer, objective, codec, or split search is performed by this binary.\"]\n",
            "}}\n"
        ),
        commit,
        json_escape(&args.contract),
        contract_sha,
        json_escape(&args.dataset),
        dataset_manifest_sha,
        split_json,
        json_escape(&args.weights_dir),
        weights_manifest_sha,
        args.top_k,
        args.n_layers,
        control.rows,
        control.topk_overlap(),
        control.pair_accuracy(),
        control.mean_spearman(),
        control.mean_stddev_ratio(),
        control.mean_abs_log_stddev_ratio(),
        candidate.topk_overlap(),
        candidate.pair_accuracy(),
        candidate.mean_spearman(),
        candidate.mean_stddev_ratio(),
        candidate.mean_abs_log_stddev_ratio(),
        candidate.topk_overlap() - control.topk_overlap(),
        candidate.pair_accuracy() - control.pair_accuracy(),
        candidate.mean_spearman() - control.mean_spearman(),
        per_layer_json,
    );

    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| fail(format!("cannot create output directory: {e}")));
        }
    }
    let temp = format!("{}.tmp.{}", args.output, std::process::id());
    std::fs::write(&temp, report.as_bytes())
        .unwrap_or_else(|e| fail(format!("cannot write temporary report: {e}")));
    std::fs::rename(&temp, &args.output)
        .unwrap_or_else(|e| fail(format!("cannot publish report atomically: {e}")));

    println!(
        "LR1_VALIDATION rows={} top16_control={:.6} top16_candidate={:.6} delta={:+.6} output={}",
        control.rows,
        control.topk_overlap(),
        candidate.topk_overlap(),
        candidate.topk_overlap() - control.topk_overlap(),
        args.output
    );
}
