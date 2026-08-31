#!/usr/bin/env python3
"""Parse one real llama.cpp perplexity arm into strict machine-readable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_final_estimate(log: str) -> tuple[float, float]:
    matches = re.findall(
        r"Final estimate:\s*PPL\s*=\s*([0-9.eE+\-]+)\s*\+/-\s*([0-9.eE+\-]+)",
        log,
    )
    if not matches:
        raise ValueError("llama-perplexity final estimate not found")
    ppl = float(matches[-1][0])
    uncertainty = float(matches[-1][1])
    if not math.isfinite(ppl) or ppl <= 0.0:
        raise ValueError(f"invalid perplexity value: {ppl!r}")
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError(f"invalid perplexity uncertainty: {uncertainty!r}")
    return ppl, uncertainty


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


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    log = Path(args.log).read_text(errors="replace")
    ppl, uncertainty = parse_final_estimate(log)
    corpus_size = os.path.getsize(args.corpus)
    if corpus_size <= 0:
        raise ValueError("perplexity corpus is empty")

    replace_summary = parse_key_values_after_marker(log, "SLHA_REPLACE_SUMMARY")
    external_store = parse_external_store(log)
    external_replace_valid: bool | None = None
    external_backend: str | None = None

    if args.mode == "external":
        if replace_summary is None or replace_summary.get("valid") is not True:
            raise ValueError("external perplexity did not produce a valid SLHA_REPLACE_SUMMARY")
        if external_store is None or external_store.get("valid") is not True:
            raise ValueError("external perplexity did not produce a valid SLHA_EXTERNAL_K_STORE")
        external_replace_valid = True
        backend = external_store.get("backend")
        external_backend = str(backend) if backend is not None else None

    return {
        "schema_version": 1,
        "mode": args.mode,
        "engine": "llama.cpp",
        "llama_cpp_commit": args.llama_commit,
        "model_sha256": args.model_sha256,
        "corpus_path": os.path.abspath(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "corpus_bytes": corpus_size,
        "context_size": args.context_size,
        "batch_size": args.context_size,
        "parallel": 1,
        "chunks_requested": args.chunks,
        "threads": args.threads,
        "gpu_layers": args.gpu_layers,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "perplexity": ppl,
        "uncertainty": uncertainty,
        "external_replace_valid": external_replace_valid,
        "external_backend": external_backend,
        "replace_summary": replace_summary if args.mode == "external" else None,
        "external_store": external_store if args.mode == "external" else None,
        "log_sha256": sha256_file(args.log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "external"), required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, required=True)
    parser.add_argument("--cache-type-k", required=True)
    parser.add_argument("--cache-type-v", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.context_size <= 0 or args.chunks <= 0 or args.threads <= 0 or args.gpu_layers < 0:
        parser.error("context-size, chunks and threads must be positive; gpu-layers must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
