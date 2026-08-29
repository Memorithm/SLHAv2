# Real autoregressive inference — baseline vs physical SLHA K

This document describes the first engine integration in which llama.cpp does
not retain a context-sized persistent K tensor. It is deliberately separate
from the historical `tilestore + replace` diagnostic path, which still owns the
ordinary llama.cpp K cache and materialises baseline `Q*K` before overwriting
its logits.

Reference engine: llama.cpp tag `b9860`, commit
`fdb1db877c526ec90f668eca1b858da5dba85560`.

## What `SLHA_EXTERNAL_K=1` changes

With the environment contract accepted by the shim:

- llama.cpp keeps its ordinary V cache and cell/sequence metadata;
- the persistent GGML K payload is replaced by a constant-size, type-valid
  sentinel rather than a context-sized tensor;
- current K passes through llama.cpp's ordinary model/RoPE path and is encoded
  at the existing K-write seam into the bounded SLHA tile store;
- non-Flash attention obtains K-side logits from the SLHA score operation
  directly; baseline `ggml_mul_mat(k, q)` is not built in that branch;
- reset invalidates all external tiles through the existing KV-clear hook;
- stale tiles in removed/reused positions are not authoritative: llama.cpp's
  own KV cell metadata/mask remains authoritative and a reused slot is
  overwritten by the next K write.

The mode is fail-closed. It currently requires:

```text
SLHA_EXTERNAL_K=1
SLHA_KV_MODE=tilestore
SLHA_SCORE_MODE=replace
SLHA_SCORE_LAYERS=all
parallel sequences = 1
FlashAttention = off
```

It rejects paired-baseline diagnostics (`SLHA_SCORE_ORACLE`, oracle metrics,
scale fitting and rank-dataset collection). Non-zero sequence shifts/division
are rejected because transformed compressed K is not yet implemented. Context
state serialization is not claimed as supported by this milestone.

## Projection weights

Physical-K inference needs the same per-layer `.slhw` projection weights used
by the existing direct compressed-score experiments. The repository does not
ship model-specific projection weights.

For the historically exercised Qwen2.5-1.5B-Instruct Q8_0 path, use a separate
calibration corpus from evaluation/generation data and the existing fail-closed
pipeline:

```bash
WORK=/tmp/slha-llama \
  integration/llama.cpp/scripts/prepare_calibration.sh

CALIB_DIR=/tmp/slha-llama/calibration \
DATA_FILE=/tmp/slha-llama/wiki.train.raw \
WORK=/tmp/slha-llama \
  integration/llama.cpp/build_and_roundtrip.sh collect

MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF \
MODEL_FILE=qwen2.5-1.5b-instruct-q8_0.gguf \
WORK=/tmp/slha-llama \
CALIB_DIR=/tmp/slha-llama/calibration \
WEIGHTS_DIR=/tmp/slha-llama/weights \
  integration/llama.cpp/scripts/train_layer_weights.sh mixed
```

The default calibration policy is `reject`: non-finite or structurally invalid
calibration data aborts before training. Do not use the optional `drop-row`
research/recovery policy for a production-quality comparison unless that choice
is explicitly recorded in the experiment provenance.

## One real baseline command

Use an explicit local GGUF path. The runner builds one patched llama.cpp binary
and uses that same binary for both control arms.

```bash
bash integration/llama.cpp/run_real_inference.sh baseline \
  --model /absolute/path/qwen2.5-1.5b-instruct-q8_0.gguf \
  --prompt 'Explain why deterministic experiments need provenance.' \
  --max-tokens 64 \
  --context-size 2048 \
  --threads 4 \
  --seed 1 \
  --gpu-layers 0 \
  --output-json /tmp/slha-real/baseline.json
```

## One real physical-SLHA command

Use the exact same model, prompt, generation parameters and hardware:

```bash
bash integration/llama.cpp/run_real_inference.sh external \
  --model /absolute/path/qwen2.5-1.5b-instruct-q8_0.gguf \
  --weights-dir /tmp/slha-llama/weights \
  --codec mixed \
  --prompt 'Explain why deterministic experiments need provenance.' \
  --max-tokens 64 \
  --context-size 2048 \
  --threads 4 \
  --seed 1 \
  --gpu-layers 0 \
  --output-json /tmp/slha-real/external.json
```

The runner forces greedy sampling (`--temp 0`), one sequence and FlashAttention
off. CPU-only (`--gpu-layers 0`) is the initial reference configuration; a GPU
configuration must be recorded explicitly rather than silently inherited.

## What the JSON report means

Observed fields include:

- exact SLHAv2 and llama.cpp commits;
- model SHA-256 and file bytes;
- model file type/quantization string when llama.cpp emits it;
- prompt SHA-256;
- context, generated-token limit, thread count, seed and cache types;
- host platform, CPU, RAM and GPU inventory when applicable;
- `/usr/bin/time` maximum process RSS and wall/user/system time;
- llama.cpp KV allocation log lines as raw evidence;
- llama.cpp prompt/decode throughput only when the corresponding engine timing
  line is present;
- physical SLHA tile-store owned allocation (`logical_tile_bytes`, backing
  vector capacity and validity-map capacity);
- strict replacement counters and coverage for the external arm;
- log path and SHA-256.

Fields that are not measured by this runner are `null`, including TTFT,
p50/p95 per-token latency, model-weight *resident* bytes, separated runtime
overhead bytes, per-op SLHA compression/score costs and paired quality metrics.
They are intentionally not inferred from wall-clock time or model file size.

A physical-SLHA run is rejected unless the process exits successfully, a valid
`SLHA_EXTERNAL_K_STORE` line is present, `SLHA_REPLACE_SUMMARY` is present and
its strict `valid=true` gate passes.

## PR boundaries

PR 1 establishes the real engine lifecycle and physical external-K path plus
this single-run evidence surface. The paired baseline/SLHA comparison harness,
objective token/logit/perplexity comparison and detailed latency/memory
breakdown belong to PR 2. Connecting `slhav2-vram::ElasticKvCache` HOT/WARM/COLD
to a live llama.cpp decode and measuring transitions belongs to PR 3.

No memory saving, throughput improvement or quality/fidelity result should be
reported for the physical path until both commands have been executed against
the same real model/workload and the resulting immutable reports are attached
to the experiment record.
