//! Frozen trainer for SLHA-LR1 Stage A.
//!
//! Unlike the historical generic ranking trainer, this binary is deliberately
//! non-configurable with respect to objective, top-k, layers and optimiser
//! hyperparameters. It exists so the current TinyStories LR1 experiment can be
//! executed without weakening historical dataset/binding invariants.

use scirust::learned::LearnedModel;
use scirust::lr1_contract;
use scirust::rank_dataset::read_layer;
use scirust::ranking::{train_ranking, Geometry, Objective, Row, TrainConfig};
use scirust::weights;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;

const N_LAYERS: usize = 6;
const TOP_K: usize = 16;
const MARGIN: f32 = 1.0;
const NEGATIVES: usize = 8;
const SEED: u64 = 7;
const LEARNING_RATE: f32 = 1.0e-9;
const EPOCHS: usize = 2;
const BATCH: usize = 16;
const MAX_KEYS: usize = 256;
const GEOMETRY_WEIGHT: f32 = 0.25;
const STORAGE_SLOTS: usize = 17;
const POPULATED_CHUNKS: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const TRAINING_CHUNKS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

struct Args {
    dataset: String,
    initial_weights: String,
    output: String,
    contract: String,
    source_manifest: String,
    training_manifest: String,
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
        "--initial-weights",
        "--output",
        "--contract",
        "--source-manifest",
        "--training-manifest",
    ];
    for key in values.keys() {
        if !known.contains(&key.as_str()) {
            fail(format!(
                "unknown option {key}; LR1 hyperparameters are intentionally not configurable"
            ));
        }
    }
    let get = |key: &str| {
        values
            .get(key)
            .cloned()
            .unwrap_or_else(|| fail(format!("missing required option {key}")))
    };
    let args = Args {
        dataset: get("--dataset"),
        initial_weights: get("--initial-weights"),
        output: get("--output"),
        contract: get("--contract"),
        source_manifest: get("--source-manifest"),
        training_manifest: get("--training-manifest"),
    };
    validate_args(&args);
    args
}

fn validate_args(args: &Args) {
    for dir in [&args.dataset, &args.initial_weights] {
        if !Path::new(dir).is_dir() {
            fail(format!("required directory does not exist: {dir}"));
        }
    }
    let dataset_manifest = format!("{}/rank_dataset_manifest.json", args.dataset);
    let initial_manifest = format!("{}/manifest.json", args.initial_weights);
    for file in [
        args.contract.as_str(),
        args.source_manifest.as_str(),
        dataset_manifest.as_str(),
        initial_manifest.as_str(),
    ] {
        if !Path::new(file).is_file() {
            fail(format!("required input file does not exist: {file}"));
        }
    }
    if Path::new(&args.output).exists() {
        fail(format!(
            "--output {:?} already exists; LR1 never overwrites a prior training attempt",
            args.output
        ));
    }
    if Path::new(&args.training_manifest).exists() {
        fail(format!(
            "--training-manifest {:?} already exists; refusing to overwrite evidence",
            args.training_manifest
        ));
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

fn sha256_bytes(bytes: &[u8]) -> Result<String, String> {
    let mut child = std::process::Command::new("sha256sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot execute sha256sum: {e}"))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| "sha256sum stdin unavailable".to_string())?
        .write_all(bytes)
        .map_err(|e| format!("cannot write sha256sum stdin: {e}"))?;
    let output = child
        .wait_with_output()
        .map_err(|e| format!("cannot wait for sha256sum: {e}"))?;
    if !output.status.success() {
        return Err("sha256sum failed".into());
    }
    let hash = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_owned();
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid sha256sum output".into());
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

fn weight_seed_rht(path: &str) -> Result<(u64, bool), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if bytes.len() < 24 {
        return Err(format!("{path}: truncated weights header"));
    }
    let seed = u64::from_le_bytes(bytes[12..20].try_into().expect("validated seed range"));
    let rht = match bytes[20] {
        0 => false,
        1 => true,
        other => return Err(format!("{path}: invalid RHT flag {other}")),
    };
    Ok((seed, rht))
}

fn main() {
    let args = parse_args();
    lr1_contract::validate_file(&args.contract)
        .unwrap_or_else(|e| fail(format!("LR1 contract semantic validation failed: {e}")));
    let commit = git_head().unwrap_or_else(|e| fail(e));
    let contract_sha = sha256_file(&args.contract).unwrap_or_else(|e| fail(e));
    let source_manifest_sha = sha256_file(&args.source_manifest).unwrap_or_else(|e| fail(e));
    let dataset_manifest = format!("{}/rank_dataset_manifest.json", args.dataset);
    let dataset_manifest_sha = sha256_file(&dataset_manifest).unwrap_or_else(|e| fail(e));
    let initial_manifest = format!("{}/manifest.json", args.initial_weights);
    let initial_manifest_sha = sha256_file(&initial_manifest).unwrap_or_else(|e| fail(e));

    let mut initial_layer_hashes = Vec::with_capacity(N_LAYERS);
    for layer in 0..N_LAYERS {
        let path = format!("{}/layer-{layer:03}.slhw", args.initial_weights);
        if !Path::new(&path).is_file() {
            fail(format!("missing initial layer weights: {path}"));
        }
        initial_layer_hashes.push(sha256_file(&path).unwrap_or_else(|e| fail(e)));
    }
    let initial_hash_json = initial_layer_hashes
        .iter()
        .enumerate()
        .map(|(layer, hash)| format!("    {{\"layer\":{layer},\"sha256\":\"{hash}\"}}"))
        .collect::<Vec<_>>()
        .join(",\n");

    let stage = format!("{}.stage.{}", args.output, std::process::id());
    if Path::new(&stage).exists() {
        std::fs::remove_dir_all(&stage)
            .unwrap_or_else(|e| fail(format!("cannot clear stale stage directory: {e}")));
    }
    std::fs::create_dir_all(&stage)
        .unwrap_or_else(|e| fail(format!("cannot create stage directory: {e}")));

    if let Some(parent) = Path::new(&args.training_manifest).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| fail(format!("cannot create evidence directory: {e}")));
        }
    }

    // Publish immutable input lineage before the first optimiser call.
    let startup = format!(
        concat!(
            "{{\n",
            "  \"schema\":\"slha_lr1_training_startup_v1\",\n",
            "  \"candidate_id\":\"slha-lr1-pairwise-top16-all-layers-v1\",\n",
            "  \"slhav2_commit\":\"{}\",\n",
            "  \"contract_sha256\":\"{}\",\n",
            "  \"contract_semantically_validated\":true,\n",
            "  \"source_manifest_sha256\":\"{}\",\n",
            "  \"rank_dataset_manifest_sha256\":\"{}\",\n",
            "  \"initial_weights_manifest_sha256\":\"{}\",\n",
            "  \"initial_weights_per_layer\":[\n{}\n  ],\n",
            "  \"rank_dataset_storage_slots\":17,\n",
            "  \"populated_chunks\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\n",
            "  \"training_chunks\":[1,2,3,4,5,6,7,8,9,10,11,12],\n",
            "  \"objective\":\"pairwise-topk\",\n",
            "  \"top_k\":16,\n",
            "  \"layers\":\"all\",\n",
            "  \"margin\":1.0,\n",
            "  \"max_negatives\":8,\n",
            "  \"negative_policy\":\"boundary-then-seeded\",\n",
            "  \"seed\":7,\n",
            "  \"learning_rate\":1e-9,\n",
            "  \"epochs\":2,\n",
            "  \"batch\":16,\n",
            "  \"max_keys\":256,\n",
            "  \"geometry_weight\":0.25,\n",
            "  \"short_context_policy\":\"retain_existing_objective_semantics\",\n",
            "  \"input_frozen_before_optimizer\":true\n",
            "}}\n"
        ),
        commit,
        contract_sha,
        source_manifest_sha,
        dataset_manifest_sha,
        initial_manifest_sha,
        initial_hash_json,
    );
    std::fs::write(format!("{stage}/startup_lineage.json"), startup.as_bytes())
        .unwrap_or_else(|e| fail(format!("cannot write startup lineage: {e}")));
    std::fs::write(
        format!("{}.startup.json", args.training_manifest),
        startup.as_bytes(),
    )
    .unwrap_or_else(|e| fail(format!("cannot write startup lineage sidecar: {e}")));

    let objective = Objective::PairwiseTopK {
        k: TOP_K,
        tau: MARGIN,
        negatives: NEGATIVES,
    };
    let mut output_hashes = Vec::with_capacity(N_LAYERS);
    let mut layer_rows = Vec::with_capacity(N_LAYERS);
    let mut layer_pairs = Vec::with_capacity(N_LAYERS);
    let mut histories = Vec::<(usize, Vec<f32>)>::with_capacity(N_LAYERS);

    for layer in 0..N_LAYERS {
        let dataset_path = format!("{}/rank-layer-{layer:03}.bin", args.dataset);
        let initial_path = format!("{}/layer-{layer:03}.slhw", args.initial_weights);
        let data = read_layer(&dataset_path).unwrap_or_else(|e| fail(e));
        if data.layer != layer as u32 {
            fail(format!(
                "dataset layer mismatch: file {layer}, header {}",
                data.layer
            ));
        }
        data.validate_chunk_layout(STORAGE_SLOTS, &POPULATED_CHUNKS)
            .unwrap_or_else(|e| fail(format!("layer {layer}: {e}")));
        let initial = weights::load(&initial_path).unwrap_or_else(|e| fail(e));
        if initial.d != data.q_dim || initial.d != data.key_dim {
            fail(format!(
                "layer {layer}: initial d={} does not match dataset q/key dimensions {}/{}",
                initial.d, data.q_dim, data.key_dim
            ));
        }

        let rows = data
            .indices_for_chunks(&TRAINING_CHUNKS)
            .map(|index| {
                let row = data.row(index).unwrap_or_else(|e| fail(e));
                Row {
                    q: row.q,
                    keys: row.keys,
                    baseline: row.baseline,
                    n_visible: row.n_visible,
                    d: row.key_dim,
                }
            })
            .collect::<Vec<_>>();
        if rows.is_empty() {
            fail(format!(
                "layer {layer}: frozen training chunks contain no rows"
            ));
        }

        let config = TrainConfig {
            objective: objective.clone(),
            geometry: Geometry {
                weight: GEOMETRY_WEIGHT,
            },
            epochs: EPOCHS,
            lr: LEARNING_RATE,
            batch: BATCH,
            seed: SEED,
            max_keys: MAX_KEYS,
            l2_scale: 1.0,
        };
        let (projection, history) = train_ranking(&rows, initial.projection().to_vec(), &config);
        if projection.iter().any(|value| !value.is_finite())
            || history.epoch_loss.iter().any(|value| !value.is_finite())
        {
            fail(format!(
                "layer {layer}: training produced non-finite values"
            ));
        }

        let (weight_seed, rht) = weight_seed_rht(&initial_path).unwrap_or_else(|e| fail(e));
        let trained = LearnedModel::from_projection_with(projection, initial.d, weight_seed, rht);
        let output_path = format!("{stage}/layer-{layer:03}.slhw");
        weights::save(&output_path, &trained, weight_seed, rht)
            .unwrap_or_else(|e| fail(format!("cannot save {output_path}: {e}")));
        output_hashes.push(sha256_file(&output_path).unwrap_or_else(|e| fail(e)));
        layer_rows.push(history.rows_seen);
        layer_pairs.push(history.pairwise_comparisons);
        histories.push((layer, history.epoch_loss));
    }

    let aggregate_input = output_hashes
        .iter()
        .enumerate()
        .map(|(layer, hash)| format!("{layer}:{hash}"))
        .collect::<Vec<_>>()
        .join(",");
    let aggregate_sha = sha256_bytes(aggregate_input.as_bytes()).unwrap_or_else(|e| fail(e));
    let per_layer = output_hashes
        .iter()
        .enumerate()
        .map(|(layer, hash)| {
            format!(
                "    {{\"layer\":{layer},\"initial_sha256\":\"{}\",\"output_sha256\":\"{hash}\",\"rows_seen\":{},\"pairwise_comparisons\":{}}}",
                initial_layer_hashes[layer], layer_rows[layer], layer_pairs[layer]
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let history_json = histories
        .iter()
        .map(|(layer, values)| {
            format!(
                "    \"{layer}\":[{}]",
                values
                    .iter()
                    .map(|value| format!("{value:.9}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let manifest = format!(
        concat!(
            "{{\n",
            "  \"schema\":\"slha_lr1_training_manifest_v1\",\n",
            "  \"candidate_id\":\"slha-lr1-pairwise-top16-all-layers-v1\",\n",
            "  \"slhav2_commit\":\"{}\",\n",
            "  \"contract_sha256\":\"{}\",\n",
            "  \"contract_semantically_validated\":true,\n",
            "  \"source_manifest_sha256\":\"{}\",\n",
            "  \"rank_dataset_manifest_sha256\":\"{}\",\n",
            "  \"initial_weights_manifest_sha256\":\"{}\",\n",
            "  \"objective\":\"pairwise-topk\",\n",
            "  \"top_k\":16,\n",
            "  \"layers\":\"all\",\n",
            "  \"margin\":1.0,\n",
            "  \"max_negatives\":8,\n",
            "  \"negative_policy\":\"boundary-then-seeded\",\n",
            "  \"seed\":7,\n",
            "  \"learning_rate\":1e-9,\n",
            "  \"epochs\":2,\n",
            "  \"steps_override\":0,\n",
            "  \"batch\":16,\n",
            "  \"max_keys\":256,\n",
            "  \"geometry_weight\":0.25,\n",
            "  \"rank_dataset_storage_slots\":17,\n",
            "  \"populated_chunks\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\n",
            "  \"training_chunks\":[1,2,3,4,5,6,7,8,9,10,11,12],\n",
            "  \"short_context_policy\":\"retain_existing_objective_semantics\",\n",
            "  \"per_layer\":[\n{}\n  ],\n",
            "  \"loss_history\":{{\n{}\n  }},\n",
            "  \"aggregate_sha256\":\"{}\",\n",
            "  \"valid\":true\n",
            "}}\n"
        ),
        commit,
        contract_sha,
        source_manifest_sha,
        dataset_manifest_sha,
        initial_manifest_sha,
        per_layer,
        history_json,
        aggregate_sha,
    );
    std::fs::write(format!("{stage}/manifest.json"), manifest.as_bytes())
        .unwrap_or_else(|e| fail(format!("cannot write staged manifest: {e}")));
    std::fs::write(&args.training_manifest, manifest.as_bytes())
        .unwrap_or_else(|e| fail(format!("cannot write training manifest: {e}")));

    std::fs::rename(&stage, &args.output)
        .unwrap_or_else(|e| fail(format!("cannot atomically publish LR1 weights: {e}")));
    println!(
        "LR1_PUBLISHED output={} aggregate={} manifest={}",
        args.output, aggregate_sha, args.training_manifest
    );
}
