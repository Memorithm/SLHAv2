#!/usr/bin/env python3
"""Compare paired real-model baseline and SLHA evaluations.

The two arms must have identical model/workload provenance. Full next-token
logits are compared only while both autoregressive contexts are identical: all
rows through the first generated-token divergence, or all common rows when no
divergence occurs. Metrics after a token divergence would compare different
contexts and are deliberately excluded from the logits error statistics.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import platform
import re
import statistics
from pathlib import Path
from typing import Any


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_time_file(path: str) -> dict[str, float | int | None]:
    text = Path(path).read_text(errors="replace")
    values: dict[str, float | int | None] = {
        "max_rss_kb": None,
        "elapsed_s": None,
        "user_s": None,
        "sys_s": None,
    }
    for key in tuple(values):
        match = re.search(rf"^{re.escape(key)}=([^\n]+)$", text, re.MULTILINE)
        if not match:
            continue
        try:
            parsed = float(match.group(1))
        except ValueError:
            continue
        values[key] = int(parsed) if key == "max_rss_kb" else parsed
    return values


def parse_key_values_after_marker(log: str, marker: str) -> dict[str, Any] | None:
    pos = log.rfind(marker + "\n")
    if pos < 0:
        return None
    result: dict[str, Any] = {}
    for line in log[pos + len(marker) + 1 :].splitlines():
        if not line or "=" not in line:
            if result:
                break
            continue
        if line.startswith("layer_"):
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
    return result or None


def parse_external_store(log: str) -> dict[str, Any] | None:
    matches = re.findall(r"^SLHA_EXTERNAL_K_STORE\s+(.+)$", log, re.MULTILINE)
    if not matches:
        return None
    # The store is reported once after construction and again at shutdown.
    # Only the last snapshot contains the complete cumulative runtime counters.
    result: dict[str, Any] = {}
    for item in matches[-1].split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if value in ("true", "false"):
            result[key] = value == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def nanoseconds_to_milliseconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) / 1_000_000.0
    return None


def unit_bytes(value: float, unit: str) -> int:
    factors = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return int(round(value * factors[unit]))


def parse_kv_components(log: str) -> dict[str, int | None]:
    """Parse explicit llama.cpp K/V component lines only; never infer a split."""
    result: dict[str, int | None] = {"k_bytes": None, "v_bytes": None}
    for key, label in (("k_bytes", "K"), ("v_bytes", "V")):
        matches = re.findall(
            rf"(?:^|[,;])\s*{label}(?:\s*\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB)",
            log,
            re.MULTILINE,
        )
        if matches:
            result[key] = sum(unit_bytes(float(value), unit) for value, unit in matches)
    return result


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def compare_logits(
    baseline_path: str,
    external_path: str,
    rows: int,
    columns: int,
) -> dict[str, Any] | None:
    if rows <= 0 or columns <= 0:
        return None
    expected = rows * columns * 4
    baseline_size = os.path.getsize(baseline_path)
    external_size = os.path.getsize(external_path)
    if baseline_size < expected or external_size < expected:
        raise ValueError(
            f"logits files are too short for {rows}x{columns} rows: "
            f"baseline={baseline_size}, external={external_size}, need={expected}"
        )

    total = 0
    sum_abs = 0.0
    sum_sq = 0.0
    baseline_sq = 0.0
    max_abs = 0.0
    argmax_agree = 0

    row_bytes = columns * 4
    with open(baseline_path, "rb") as left, open(external_path, "rb") as right:
        for _ in range(rows):
            a = array.array("f")
            b = array.array("f")
            a.frombytes(left.read(row_bytes))
            b.frombytes(right.read(row_bytes))
            if len(a) != columns or len(b) != columns:
                raise ValueError("short logits row")
            a_argmax = max(range(columns), key=a.__getitem__)
            b_argmax = max(range(columns), key=b.__getitem__)
            argmax_agree += int(a_argmax == b_argmax)
            for av, bv in zip(a, b):
                if not math.isfinite(av) or not math.isfinite(bv):
                    raise ValueError("non-finite value in logits evidence")
                diff = float(bv) - float(av)
                adiff = abs(diff)
                max_abs = max(max_abs, adiff)
                sum_abs += adiff
                sum_sq += diff * diff
                baseline_sq += float(av) * float(av)
                total += 1

    rmse = math.sqrt(sum_sq / total) if total else None
    relative_l2 = math.sqrt(sum_sq / baseline_sq) if baseline_sq > 0.0 else None
    return {
        "comparable_rows": rows,
        "columns": columns,
        "values_compared": total,
        "mean_absolute_error": sum_abs / total if total else None,
        "max_absolute_error": max_abs,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "argmax_agreement_rows": argmax_agree,
        "argmax_agreement_ratio": argmax_agree / rows if rows else None,
        "baseline_sha256": sha256_file(baseline_path),
        "external_sha256": sha256_file(external_path),
    }


def require_equal(baseline: dict[str, Any], external: dict[str, Any], keys: list[str]) -> None:
    mismatches = []
    for key in keys:
        if baseline.get(key) != external.get(key):
            mismatches.append(f"{key}: {baseline.get(key)!r} != {external.get(key)!r}")
    if mismatches:
        raise ValueError("paired run configuration mismatch: " + "; ".join(mismatches))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline = read_json(args.baseline_json)
    external = read_json(args.external_json)
    require_equal(
        baseline,
        external,
        [
            "engine",
            "decoder_only",
            "n_vocab",
            "context_size",
            "threads",
            "gpu_layers",
            "cache_type_k",
            "cache_type_v",
            "prompt_tokens",
        ],
    )

    baseline_tokens = [int(v) for v in baseline.get("generated_tokens", [])]
    external_tokens = [int(v) for v in external.get("generated_tokens", [])]
    divergence = first_divergence(baseline_tokens, external_tokens)
    common_rows = min(len(baseline_tokens), len(external_tokens))
    if divergence is None:
        comparable_rows = common_rows
        common_prefix = common_rows
    elif divergence < common_rows:
        # Row `divergence` was computed from an identical preceding context and
        # is therefore still a valid next-token logits comparison.
        comparable_rows = divergence + 1
        common_prefix = divergence
    else:
        comparable_rows = common_rows
        common_prefix = common_rows

    aligned = min(len(baseline_tokens), len(external_tokens))
    aligned_agreements = sum(
        1 for left, right in zip(baseline_tokens, external_tokens) if left == right
    )

    n_vocab = int(baseline["n_vocab"])
    logits = compare_logits(
        args.baseline_logits,
        args.external_logits,
        comparable_rows,
        n_vocab,
    )

    baseline_time = parse_time_file(args.baseline_time)
    external_time = parse_time_file(args.external_time)
    baseline_log = Path(args.baseline_log).read_text(errors="replace")
    external_log = Path(args.external_log).read_text(errors="replace")
    baseline_kv = parse_kv_components(baseline_log)
    external_kv = parse_kv_components(external_log)
    external_store = parse_external_store(external_log)
    replace_summary = parse_key_values_after_marker(external_log, "SLHA_REPLACE_SUMMARY")
    runtime_cost = parse_key_values_after_marker(external_log, "SLHA_RUNTIME_COST_SUMMARY")

    baseline_decode = [float(v) for v in baseline.get("timing", {}).get("decode_step_ms", [])]
    external_decode = [float(v) for v in external.get("timing", {}).get("decode_step_ms", [])]

    slha_store_bytes = None
    slha_peak_physical_bytes = None
    external_backend = external_store.get("backend") if external_store else None
    if external_store is not None:
        for key in ("allocated_bytes", "allocation_bytes", "owned_bytes"):
            value = external_store.get(key)
            if isinstance(value, int):
                slha_store_bytes = value
                break
        peak_resident = external_store.get("peak_resident_bytes")
        peak_offloaded = external_store.get("peak_offloaded_bytes")
        if isinstance(peak_resident, int) and isinstance(peak_offloaded, int):
            slha_peak_physical_bytes = peak_resident + peak_offloaded

    baseline_ppl = read_json(args.baseline_perplexity) if args.baseline_perplexity else None
    external_ppl = read_json(args.external_perplexity) if args.external_perplexity else None
    if (baseline_ppl is None) != (external_ppl is None):
        raise ValueError("perplexity evidence must be supplied for both arms or neither")

    perplexity: dict[str, Any] | None = None
    if baseline_ppl is not None and external_ppl is not None:
        if baseline_ppl.get("corpus_sha256") != external_ppl.get("corpus_sha256"):
            raise ValueError("perplexity corpus mismatch")
        perplexity = {
            "corpus_sha256": baseline_ppl.get("corpus_sha256"),
            "baseline": baseline_ppl.get("perplexity"),
            "external": external_ppl.get("perplexity"),
            "absolute_delta": (
                float(external_ppl["perplexity"]) - float(baseline_ppl["perplexity"])
                if baseline_ppl.get("perplexity") is not None
                and external_ppl.get("perplexity") is not None
                else None
            ),
        }

    baseline_tps = baseline.get("timing", {}).get("decode_tokens_per_second")
    external_tps = external.get("timing", {}).get("decode_tokens_per_second")
    speed_ratio = None
    if isinstance(baseline_tps, (int, float)) and baseline_tps > 0 and isinstance(external_tps, (int, float)):
        speed_ratio = external_tps / baseline_tps

    baseline_rss = baseline_time.get("max_rss_kb")
    external_rss = external_time.get("max_rss_kb")
    rss_delta = None
    if isinstance(baseline_rss, int) and isinstance(external_rss, int):
        rss_delta = external_rss - baseline_rss

    return {
        "schema_version": 1,
        "experiment": (
            "llama.cpp baseline vs physical SLHA external-K (CCOS elastic)"
            if external_backend == "ccos_elastic"
            else "llama.cpp baseline vs physical SLHA external-K"
        ),
        "provenance": {
            "slhav2_commit": args.slhav2_commit,
            "llama_cpp_commit": args.llama_commit,
            "model_path": os.path.abspath(args.model),
            "model_sha256": args.model_sha256,
            "prompt_sha256": args.prompt_sha256,
            "context_size": baseline["context_size"],
            "threads": baseline["threads"],
            "gpu_layers": baseline["gpu_layers"],
            "cache_type_k": baseline["cache_type_k"],
            "cache_type_v": baseline["cache_type_v"],
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "quality": {
            "baseline_generated_tokens": len(baseline_tokens),
            "external_generated_tokens": len(external_tokens),
            "aligned_positions": aligned,
            "aligned_token_agreements": aligned_agreements,
            "aligned_token_agreement_ratio": aligned_agreements / aligned if aligned else None,
            "common_prefix_tokens": common_prefix,
            "first_divergence_index": divergence,
            "text_exact_match": baseline.get("generated_text") == external.get("generated_text"),
            "baseline_text": baseline.get("generated_text"),
            "external_text": external.get("generated_text"),
            "next_token_logits": logits,
            "perplexity": perplexity,
        },
        "memory": {
            "baseline_max_process_rss_kb": baseline_rss,
            "external_max_process_rss_kb": external_rss,
            "external_minus_baseline_rss_kb": rss_delta,
            "baseline_engine_k_bytes": baseline_kv["k_bytes"],
            "baseline_engine_v_bytes": baseline_kv["v_bytes"],
            "external_engine_k_sentinel_bytes": external_kv["k_bytes"],
            "external_engine_v_bytes": external_kv["v_bytes"],
            "external_slha_backend": external_backend,
            "external_slha_store_allocated_bytes": slha_store_bytes,
            "external_slha_peak_physical_bytes": slha_peak_physical_bytes,
            "external_slha_store": external_store,
        },
        "performance": {
            "baseline_prefill_ms": baseline.get("timing", {}).get("prefill_ms"),
            "external_prefill_ms": external.get("timing", {}).get("prefill_ms"),
            "baseline_ttft_ms": baseline.get("timing", {}).get("ttft_ms"),
            "external_ttft_ms": external.get("timing", {}).get("ttft_ms"),
            "baseline_decode_tokens_per_second": baseline_tps,
            "external_decode_tokens_per_second": external_tps,
            "external_over_baseline_decode_tps": speed_ratio,
            "baseline_p50_decode_ms": percentile(baseline_decode, 0.50),
            "baseline_p95_decode_ms": percentile(baseline_decode, 0.95),
            "external_p50_decode_ms": percentile(external_decode, 0.50),
            "external_p95_decode_ms": percentile(external_decode, 0.95),
            "slha_runtime_cost": runtime_cost,
            "slha_compression_ms": nanoseconds_to_milliseconds(
                external_store.get("compression_ns") if external_store else None
            ),
            "slha_score_ms": nanoseconds_to_milliseconds(
                external_store.get("score_ns") if external_store else None
            ),
            "slha_budget_enforcement_ms": nanoseconds_to_milliseconds(
                external_store.get("budget_ns") if external_store else None
            ),
        },
        "validity": {
            "same_prompt_tokens": baseline.get("prompt_tokens") == external.get("prompt_tokens"),
            "slha_replace_summary": replace_summary,
            "external_replace_valid": replace_summary.get("valid") if replace_summary else None,
            "external_backend": external_backend,
            "ccos_enabled": external_backend == "ccos_elastic",
            "ccos_dense_no_cold": (
                external_store.get("peak_cold_slots") == 0
                if external_backend == "ccos_elastic" and external_store else None
            ),
            "ccos_budget_failures": (
                external_store.get("budget_failures")
                if external_backend == "ccos_elastic" and external_store else None
            ),
            "logits_context_rule": (
                "Logits are compared through the first divergent generated token only; "
                "later rows are excluded because autoregressive contexts differ."
            ),
        },
        "artifacts": {
            "baseline_eval_sha256": sha256_file(args.baseline_json),
            "external_eval_sha256": sha256_file(args.external_json),
            "baseline_log_sha256": sha256_file(args.baseline_log),
            "external_log_sha256": sha256_file(args.external_log),
        },
        "limitations": [
            "RSS is process-level peak RSS and is not decomposed into weight residency and allocator overhead.",
            "K/V byte components are populated only when llama.cpp prints an explicit K/V split; otherwise they remain null.",
            "Perplexity remains null unless paired real perplexity evidence is supplied.",
            "CCOS resident/offloaded byte counters describe cache-owned representations only and are not process RSS.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--external-json", required=True)
    parser.add_argument("--baseline-logits", required=True)
    parser.add_argument("--external-logits", required=True)
    parser.add_argument("--baseline-time", required=True)
    parser.add_argument("--external-time", required=True)
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--external-log", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--prompt-sha256", required=True)
    parser.add_argument("--slhav2-commit", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-perplexity")
    parser.add_argument("--external-perplexity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
