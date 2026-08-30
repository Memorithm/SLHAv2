from pathlib import Path


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    begin = text.find(start)
    if begin < 0:
        raise RuntimeError(f"missing start marker: {start!r}")
    finish = text.find(end, begin)
    if finish < 0:
        raise RuntimeError(f"missing end marker: {end!r}")
    return text[:begin] + replacement + text[finish:]


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}, got {text.count(old)}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Paired comparison: consume the final physical-store snapshot and surface
# CCOS counters without turning cache-owned bytes into process-RSS claims.
# ---------------------------------------------------------------------------
p = Path("integration/llama.cpp/scripts/compare_real_eval.py")
s = p.read_text()
s = replace_between(
    s,
    "def parse_external_store(log: str) -> dict[str, Any] | None:\n",
    "\ndef unit_bytes(value: float, unit: str) -> int:\n",
    '''def parse_external_store(log: str) -> dict[str, Any] | None:\n    matches = re.findall(r"^SLHA_EXTERNAL_K_STORE\\s+(.+)$", log, re.MULTILINE)\n    if not matches:\n        return None\n    # The store is reported once after construction and again at shutdown.\n    # Only the last snapshot contains the complete cumulative runtime counters.\n    result: dict[str, Any] = {}\n    for item in matches[-1].split():\n        if "=" not in item:\n            continue\n        key, value = item.split("=", 1)\n        if value in ("true", "false"):\n            result[key] = value == "true"\n        else:\n            try:\n                result[key] = int(value)\n            except ValueError:\n                result[key] = value\n    return result\n\n\ndef nanoseconds_to_milliseconds(value: Any) -> float | None:\n    if isinstance(value, (int, float)):\n        return float(value) / 1_000_000.0\n    return None\n\n'''
)
s = replace_once(
    s,
    '''    slha_store_bytes = None\n    if external_store is not None:\n        for key in ("allocated_bytes", "allocation_bytes", "owned_bytes"):\n            value = external_store.get(key)\n            if isinstance(value, int):\n                slha_store_bytes = value\n                break\n''',
    '''    slha_store_bytes = None\n    slha_peak_physical_bytes = None\n    external_backend = external_store.get("backend") if external_store else None\n    if external_store is not None:\n        for key in ("allocated_bytes", "allocation_bytes", "owned_bytes"):\n            value = external_store.get(key)\n            if isinstance(value, int):\n                slha_store_bytes = value\n                break\n        peak_resident = external_store.get("peak_resident_bytes")\n        peak_offloaded = external_store.get("peak_offloaded_bytes")\n        if isinstance(peak_resident, int) and isinstance(peak_offloaded, int):\n            slha_peak_physical_bytes = peak_resident + peak_offloaded\n'''
)
s = replace_once(
    s,
    '        "experiment": "llama.cpp baseline vs physical SLHA external-K",\n',
    '''        "experiment": (\n            "llama.cpp baseline vs physical SLHA external-K (CCOS elastic)"\n            if external_backend == "ccos_elastic"\n            else "llama.cpp baseline vs physical SLHA external-K"\n        ),\n'''
)
s = replace_once(
    s,
    '''            "external_slha_store_allocated_bytes": slha_store_bytes,\n            "external_slha_store": external_store,\n''',
    '''            "external_slha_backend": external_backend,\n            "external_slha_store_allocated_bytes": slha_store_bytes,\n            "external_slha_peak_physical_bytes": slha_peak_physical_bytes,\n            "external_slha_store": external_store,\n'''
)
s = replace_once(
    s,
    '''            "slha_runtime_cost": runtime_cost,\n''',
    '''            "slha_runtime_cost": runtime_cost,\n            "slha_compression_ms": nanoseconds_to_milliseconds(\n                external_store.get("compression_ns") if external_store else None\n            ),\n            "slha_score_ms": nanoseconds_to_milliseconds(\n                external_store.get("score_ns") if external_store else None\n            ),\n            "slha_budget_enforcement_ms": nanoseconds_to_milliseconds(\n                external_store.get("budget_ns") if external_store else None\n            ),\n'''
)
s = replace_once(
    s,
    '''            "external_replace_valid": replace_summary.get("valid") if replace_summary else None,\n            "logits_context_rule": (\n''',
    '''            "external_replace_valid": replace_summary.get("valid") if replace_summary else None,\n            "external_backend": external_backend,\n            "ccos_enabled": external_backend == "ccos_elastic",\n            "ccos_dense_no_cold": (\n                external_store.get("peak_cold_slots") == 0\n                if external_backend == "ccos_elastic" and external_store else None\n            ),\n            "ccos_budget_failures": (\n                external_store.get("budget_failures")\n                if external_backend == "ccos_elastic" and external_store else None\n            ),\n            "logits_context_rule": (\n'''
)
s = replace_once(
    s,
    '            "CCOS HOT/WARM/COLD transitions are outside this PR and are not represented here.",\n',
    '            "CCOS resident/offloaded byte counters describe cache-owned representations only and are not process RSS.",\n'
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Single-run report: same final-snapshot rule and explicit CCOS mode/costs.
# ---------------------------------------------------------------------------
p = Path("integration/llama.cpp/scripts/report_real_inference.py")
s = p.read_text()
s = replace_between(
    s,
    "def parse_external_store(log: str) -> dict[str, Any] | None:\n",
    "\ndef parse_replace_summary(log: str) -> dict[str, Any] | None:\n",
    '''def parse_external_store(log: str) -> dict[str, Any] | None:\n    matches = re.findall(r"^SLHA_EXTERNAL_K_STORE\\s+(.+)$", log, re.MULTILINE)\n    if not matches:\n        return None\n    result: dict[str, Any] = {}\n    for item in matches[-1].split():\n        if "=" not in item:\n            continue\n        key, value = item.split("=", 1)\n        if value in ("true", "false"):\n            result[key] = value == "true"\n            continue\n        try:\n            result[key] = int(value)\n        except ValueError:\n            result[key] = value\n    return result\n\n\ndef nanoseconds_to_milliseconds(value: Any) -> float | None:\n    if isinstance(value, (int, float)):\n        return float(value) / 1_000_000.0\n    return None\n\n'''
)
s = replace_once(
    s,
    '''    prompt_tps, decode_tps = parse_perf_tps(log)\n\n    max_rss = parse_time_value(time_text, "max_rss_kb")\n''',
    '''    prompt_tps, decode_tps = parse_perf_tps(log)\n    external_store = parse_external_store(log)\n\n    max_rss = parse_time_value(time_text, "max_rss_kb")\n'''
)
s = replace_once(
    s,
    '            "codec": args.codec if args.mode == "external" else None,\n',
    '            "codec": args.codec if args.mode in ("external", "ccos") else None,\n'
)
s = replace_once(
    s,
    '            "external_k_store": parse_external_store(log),\n',
    '            "external_k_store": external_store,\n'
)
s = replace_once(
    s,
    '''            "slha_compression_cost_ms": None,\n            "slha_score_cost_ms": None,\n''',
    '''            "slha_compression_cost_ms": nanoseconds_to_milliseconds(\n                external_store.get("compression_ns") if external_store else None\n            ),\n            "slha_score_cost_ms": nanoseconds_to_milliseconds(\n                external_store.get("score_ns") if external_store else None\n            ),\n            "slha_budget_enforcement_cost_ms": nanoseconds_to_milliseconds(\n                external_store.get("budget_ns") if external_store else None\n            ),\n'''
)
s = replace_once(
    s,
    '            "CCOS HOT/WARM/COLD is not connected to the llama.cpp external-K path in PR1.",\n',
    '            "CCOS cache-owned residency counters are distinct from process RSS and model-weight residency.",\n'
)
s = replace_once(
    s,
    '    parser.add_argument("--mode", choices=("baseline", "external"), required=True)\n',
    '    parser.add_argument("--mode", choices=("baseline", "external", "ccos"), required=True)\n'
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Runner: explicit CCOS switch while preserving historical external-K mode.
# ---------------------------------------------------------------------------
p = Path("integration/llama.cpp/run_real_pair.sh")
s = p.read_text()
s = replace_once(
    s,
    '''CACHE_TYPE_K="f16"\nCACHE_TYPE_V="f16"\n''',
    '''CACHE_TYPE_K="f16"\nCACHE_TYPE_V="f16"\nCCOS=0\nCCOS_BUDGET_BYTES=""\nCCOS_IMPORTANCE_TEMPERATURE=""\n'''
)
s = replace_once(
    s,
    '''        --cache-type-k) CACHE_TYPE_K="${2:?missing value for --cache-type-k}"; shift 2 ;;\n        --cache-type-v) CACHE_TYPE_V="${2:?missing value for --cache-type-v}"; shift 2 ;;\n''',
    '''        --cache-type-k) CACHE_TYPE_K="${2:?missing value for --cache-type-k}"; shift 2 ;;\n        --cache-type-v) CACHE_TYPE_V="${2:?missing value for --cache-type-v}"; shift 2 ;;\n        --ccos) CCOS=1; shift ;;\n        --ccos-budget-bytes) CCOS_BUDGET_BYTES="${2:?missing value for --ccos-budget-bytes}"; shift 2 ;;\n        --ccos-importance-temperature) CCOS_IMPORTANCE_TEMPERATURE="${2:?missing value for --ccos-importance-temperature}"; shift 2 ;;\n'''
)
s = replace_once(
    s,
    '''compgen -G "$WEIGHTS_DIR/layer-*.slhw" >/dev/null || {\n    echo "ERROR: no layer-*.slhw files in $WEIGHTS_DIR" >&2\n    exit 2\n}\n''',
    '''compgen -G "$WEIGHTS_DIR/layer-*.slhw" >/dev/null || {\n    echo "ERROR: no layer-*.slhw files in $WEIGHTS_DIR" >&2\n    exit 2\n}\nif [[ "$CCOS" -ne 1 && ( -n "$CCOS_BUDGET_BYTES" || -n "$CCOS_IMPORTANCE_TEMPERATURE" ) ]]; then\n    echo "ERROR: CCOS budget/temperature options require --ccos" >&2\n    exit 2\nfi\n'''
)
s = replace_once(
    s,
    '''printf 'GPU layers    : %s\\n' "$GPU_LAYERS"\n''',
    '''printf 'GPU layers    : %s\\n' "$GPU_LAYERS"\nprintf 'external back.: %s\\n' "$([[ "$CCOS" -eq 1 ]] && echo ccos_elastic || echo vector)"\nprintf 'CCOS budget   : %s\\n' "${CCOS_BUDGET_BYTES:-default-full-HOT}"\nprintf 'CCOS temp.    : %s\\n' "${CCOS_IMPORTANCE_TEMPERATURE:-default-1.0}"\n'''
)
s = replace_once(
    s,
    '''    unset SLHA_RANK_DATASET_DIR SLHA_WEIGHTS_DIR SLHA_CODEC\n''',
    '''    unset SLHA_RANK_DATASET_DIR SLHA_WEIGHTS_DIR SLHA_CODEC\n    unset SLHA_CCOS SLHA_CCOS_BUDGET_BYTES SLHA_CCOS_IMPORTANCE_TEMPERATURE\n'''
)
s = replace_once(
    s,
    '''        export SLHA_WEIGHTS_DIR="$WEIGHTS_DIR"\n        export SLHA_CODEC="$CODEC"\n''',
    '''        export SLHA_WEIGHTS_DIR="$WEIGHTS_DIR"\n        export SLHA_CODEC="$CODEC"\n        if [[ "$CCOS" -eq 1 ]]; then\n            export SLHA_CCOS=1\n            [[ -z "$CCOS_BUDGET_BYTES" ]] || export SLHA_CCOS_BUDGET_BYTES="$CCOS_BUDGET_BYTES"\n            [[ -z "$CCOS_IMPORTANCE_TEMPERATURE" ]] || \\\n                export SLHA_CCOS_IMPORTANCE_TEMPERATURE="$CCOS_IMPORTANCE_TEMPERATURE"\n        fi\n'''
)
s = replace_once(
    s,
    '''python3 - "$OUTPUT_DIR/comparison.json" <<'PY'\nimport json, sys\np = sys.argv[1]\nr = json.load(open(p))\n''',
    '''python3 - "$OUTPUT_DIR/comparison.json" "$CCOS" <<'PY'\nimport json, sys\np = sys.argv[1]\nccos_requested = sys.argv[2] == "1"\nr = json.load(open(p))\n'''
)
s = replace_once(
    s,
    '''if valid is not True:\n    raise SystemExit(f"external SLHA replace summary is not valid: {valid!r}")\nq = r["quality"]\n''',
    '''if valid is not True:\n    raise SystemExit(f"external SLHA replace summary is not valid: {valid!r}")\nvalidity = r.get("validity", {})\nbackend = validity.get("external_backend")\nif ccos_requested:\n    if backend != "ccos_elastic":\n        raise SystemExit(f"CCOS was requested but measured backend is {backend!r}")\n    if validity.get("ccos_dense_no_cold") is not True:\n        raise SystemExit("dense CCOS run observed a COLD slot")\n    if validity.get("ccos_budget_failures") != 0:\n        raise SystemExit(\n            f"CCOS budget failures are non-zero: {validity.get('ccos_budget_failures')!r}"\n        )\nq = r["quality"]\n'''
)
s = replace_once(
    s,
    '''print(f"external peak RSS KB: {mem['external_max_process_rss_kb']}")\nprint(f"report              : {p}")\n''',
    '''print(f"external peak RSS KB: {mem['external_max_process_rss_kb']}")\nprint(f"external backend    : {validity.get('external_backend')}")\nif ccos_requested:\n    store = mem.get("external_slha_store") or {}\n    print(f"CCOS peak resident  : {store.get('peak_resident_bytes')} bytes")\n    print(f"CCOS peak offloaded : {store.get('peak_offloaded_bytes')} bytes")\n    print(f"CCOS HOT/WARM/COLD  : {store.get('peak_hot_slots')}/{store.get('peak_warm_slots')}/{store.get('peak_cold_slots')}")\n    print(f"CCOS compression ms : {pf.get('slha_compression_ms')}")\n    print(f"CCOS score ms       : {pf.get('slha_score_ms')}")\n    print(f"CCOS budget ms      : {pf.get('slha_budget_enforcement_ms')}")\nprint(f"report              : {p}")\n'''
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Tests: prove the final snapshot wins and time conversion is exact.
# ---------------------------------------------------------------------------
p = Path("integration/llama.cpp/tests/test_compare_real_eval.py")
s = p.read_text()
s = replace_once(
    s,
    '''        external_log.write_text(\n            "SLHA_EXTERNAL_K_STORE allocated_bytes=1234 capacity=64\\n"\n            "SLHA_REPLACE_SUMMARY\\n"\n            "callbacks=2\\nvalid=true\\n\\n"\n        )\n        assert mod.parse_external_store(external_log.read_text())["allocated_bytes"] == 1234\n''',
    '''        external_log.write_text(\n            "SLHA_EXTERNAL_K_STORE valid=true backend=vector allocated_bytes=1234 capacity=64\\n"\n            "SLHA_EXTERNAL_K_STORE valid=true backend=ccos_elastic peak_resident_bytes=960 peak_offloaded_bytes=32 peak_cold_slots=0 budget_failures=0 compression_ns=2500000 score_ns=4000000 budget_ns=500000\\n"\n            "SLHA_REPLACE_SUMMARY\\n"\n            "callbacks=2\\nvalid=true\\n\\n"\n        )\n        parsed_store = mod.parse_external_store(external_log.read_text())\n        assert parsed_store["backend"] == "ccos_elastic"\n        assert parsed_store["peak_resident_bytes"] == 960\n        assert parsed_store["budget_failures"] == 0\n        assert mod.nanoseconds_to_milliseconds(parsed_store["compression_ns"]) == 2.5\n'''
)
p.write_text(s)

p = Path("integration/llama.cpp/tests/test_real_inference_report.py")
s = p.read_text()
s = replace_once(
    s,
    '''SLHA_EXTERNAL_K_STORE valid=true layers=28 capacity=2048 tile_bytes=128 logical_tile_bytes=7340032 tile_backing_capacity_bytes=7340159 validity_backing_capacity_bytes=57344\nSLHA_REPLACE_SUMMARY\n''',
    '''SLHA_EXTERNAL_K_STORE valid=true backend=vector layers=28 capacity=2048 tile_bytes=128 logical_tile_bytes=7340032 tile_backing_capacity_bytes=7340159 validity_backing_capacity_bytes=57344\nSLHA_EXTERNAL_K_STORE valid=true backend=ccos_elastic layers=28 capacity=2048 tile_bytes=128 logical_tile_bytes=7340032 peak_resident_bytes=4096 peak_offloaded_bytes=128 peak_cold_slots=0 budget_failures=0 compression_ns=1500000 score_ns=2750000 budget_ns=250000\nSLHA_REPLACE_SUMMARY\n'''
)
s = replace_once(
    s,
    '''    assert store["valid"] is True\n    assert store["layers"] == 28\n    assert store["logical_tile_bytes"] == 7340032\n''',
    '''    assert store["valid"] is True\n    assert store["backend"] == "ccos_elastic"\n    assert store["layers"] == 28\n    assert store["logical_tile_bytes"] == 7340032\n    assert store["peak_resident_bytes"] == 4096\n    assert module.nanoseconds_to_milliseconds(store["score_ns"]) == 2.75\n'''
)
p.write_text(s)


# ---------------------------------------------------------------------------
# Official real-model smoke: exercise the Rust CCOS backend, not the legacy
# vector path, while leaving budget tightening to a separately measured sweep.
# ---------------------------------------------------------------------------
p = Path(".github/workflows/llama-real-model-smoke.yml")
s = p.read_text()
s = replace_once(
    s,
    '    name: TinyStories 15M real autoregressive external-K\n',
    '    name: TinyStories 15M real autoregressive CCOS external-K\n'
)
s = replace_once(
    s,
    '''            --codec mixed \\\n            --output-dir "$RUNNER_TEMP/slha-real-smoke/evidence"\n''',
    '''            --codec mixed \\\n            --ccos \\\n            --output-dir "$RUNNER_TEMP/slha-real-smoke/evidence"\n'''
)
s = replace_once(
    s,
    '''          assert r["memory"]["external_slha_store"] is not None, r["memory"]\n          print(json.dumps(r, sort_keys=True))\n''',
    '''          store = r["memory"]["external_slha_store"]\n          assert store is not None, r["memory"]\n          assert store.get("backend") == "ccos_elastic", store\n          assert store.get("peak_resident_bytes", 0) > 0, store\n          assert store.get("score_calls", 0) > 0, store\n          assert store.get("compression_ns", 0) > 0, store\n          assert store.get("peak_cold_slots") == 0, store\n          assert store.get("budget_failures") == 0, store\n          assert r["validity"]["ccos_dense_no_cold"] is True, r["validity"]\n          print(json.dumps(r, sort_keys=True))\n'''
)
p.write_text(s)
