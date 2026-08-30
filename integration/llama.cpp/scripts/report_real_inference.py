#!/usr/bin/env python3
"""Build a measured JSON report from one llama.cpp real-inference run.

The parser deliberately leaves unavailable metrics as null. It never converts
model file size or tile format size into process-resident-memory claims.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


def parse_time_value(text: str, name: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}=([^\n]+)$", text, re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_perf_tps(log: str) -> tuple[float | None, float | None]:
    """Return (prompt/prefill TPS, decode TPS) from exact llama perf lines."""
    prompt_tps: float | None = None
    decode_tps: float | None = None
    token_rate = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second", re.I)

    for line in log.splitlines():
        lowered = line.lower()
        if "tokens per second" not in lowered:
            continue
        match = token_rate.search(line)
        if not match:
            continue
        value = float(match.group(1))
        if "prompt eval time" in lowered:
            prompt_tps = value
        elif "eval time" in lowered:
            # Explicitly exclude prompt-eval lines. The old ad-hoc regex matched
            # "prompt eval time" as decode because it contains "eval time".
            decode_tps = value

    return prompt_tps, decode_tps


def parse_external_store(log: str) -> dict[str, Any] | None:
    matches = re.findall(r"^SLHA_EXTERNAL_K_STORE\s+(.+)$", log, re.MULTILINE)
    if not matches:
        return None
    result: dict[str, Any] = {}
    for item in matches[-1].split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if value in ("true", "false"):
            result[key] = value == "true"
            continue
        try:
            result[key] = int(value)
        except ValueError:
            result[key] = value
    return result


def nanoseconds_to_milliseconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) / 1_000_000.0
    return None


def parse_replace_summary(log: str) -> dict[str, Any] | None:
    marker = "SLHA_REPLACE_SUMMARY\n"
    pos = log.rfind(marker)
    if pos < 0:
        return None

    result: dict[str, Any] = {}
    for line in log[pos + len(marker) :].splitlines():
        if line.startswith("layer_"):
            continue
        if not line or "=" not in line:
            if result:
                break
            continue
        key, value = line.split("=", 1)
        if " " in key or key.startswith("["):
            break
        if value in ("true", "false"):
            result[key] = value == "true"
            continue
        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def first_text(log: str, pattern: str) -> str | None:
    match = re.search(pattern, log, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def cpu_model() -> str | None:
    try:
        text = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in text.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return None


def mem_total_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def gpu_inventory(gpu_layers: int) -> list[str] | None:
    if gpu_layers == 0:
        return []
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in output.splitlines() if line.strip()]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    log = Path(args.log).read_text(errors="replace")
    time_text = Path(args.time).read_text(errors="replace") if Path(args.time).exists() else ""
    prompt_tps, decode_tps = parse_perf_tps(log)
    external_store = parse_external_store(log)

    max_rss = parse_time_value(time_text, "max_rss_kb")
    kv_lines = [
        line.strip()
        for line in log.splitlines()
        if "KV buffer size" in line or "KV self size" in line
    ]

    return {
        "schema_version": 1,
        "mode": args.mode,
        "valid_process_exit": args.exit_code == 0,
        "process_exit_code": args.exit_code,
        "provenance": {
            "slhav2_commit": args.slhav2_commit,
            "llama_cpp_commit": args.llama_commit,
            "model_path": os.path.abspath(args.model),
            "model_sha256": args.model_sha256,
            "model_bytes": args.model_bytes,
            "quantization_from_engine_log": first_text(log, r"file type\s*=\s*([^\n]+)"),
            "prompt_sha256": args.prompt_sha256,
            "context_size": args.context_size,
            "max_tokens": args.max_tokens,
            "threads": args.threads,
            "seed": args.seed,
            "gpu_layers": args.gpu_layers,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "codec": args.codec if args.mode in ("external", "ccos") else None,
            "weights_dir": os.path.abspath(args.weights_dir) if args.weights_dir else None,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "ram_total_kb": mem_total_kb(),
            "gpu_inventory": gpu_inventory(args.gpu_layers),
        },
        "memory": {
            "max_process_rss_kb": int(max_rss) if max_rss is not None else None,
            "engine_kv_allocation_lines": kv_lines,
            "external_k_store": external_store,
            "weights_resident_bytes": None,
            "runtime_overhead_bytes": None,
            "baseline_kv_bytes": None,
            "slha_kv_bytes": None,
        },
        "performance": {
            "wall_elapsed_s": parse_time_value(time_text, "elapsed_s"),
            "user_cpu_s": parse_time_value(time_text, "user_s"),
            "system_cpu_s": parse_time_value(time_text, "sys_s"),
            "prefill_tokens_per_second_from_engine": prompt_tps,
            "decode_tokens_per_second_from_engine": decode_tps,
            "time_to_first_token_ms": None,
            "p50_token_latency_ms": None,
            "p95_token_latency_ms": None,
            "slha_compression_cost_ms": nanoseconds_to_milliseconds(
                external_store.get("compression_ns") if external_store else None
            ),
            "slha_score_cost_ms": nanoseconds_to_milliseconds(
                external_store.get("score_ns") if external_store else None
            ),
            "slha_budget_enforcement_cost_ms": nanoseconds_to_milliseconds(
                external_store.get("budget_ns") if external_store else None
            ),
        },
        "slha_replace_summary": parse_replace_summary(log),
        "quality": {
            "next_token_logits": None,
            "token_agreement": None,
            "perplexity": None,
            "note": (
                "Generation is real; paired quality metrics are intentionally deferred "
                "to the comparison harness rather than inferred from this single run."
            ),
        },
        "artifacts": {
            "log_path": os.path.abspath(args.log),
            "log_sha256": args.log_sha256,
            "time_path": os.path.abspath(args.time),
        },
        "limitations": [
            "PR1 records real autoregressive execution and process RSS but does not infer model-weight residency from file size.",
            "TTFT and p50/p95 token latency require per-token instrumentation and remain null here.",
            "CCOS cache-owned residency counters are distinct from process RSS and model-weight residency.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "external", "ccos"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-bytes", type=int, required=True)
    parser.add_argument("--prompt-sha256", required=True)
    parser.add_argument("--slhav2-commit", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, required=True)
    parser.add_argument("--cache-type-k", required=True)
    parser.add_argument("--cache-type-v", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log-sha256", required=True)
    parser.add_argument("--weights-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
