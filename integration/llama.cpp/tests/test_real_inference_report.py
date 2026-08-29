#!/usr/bin/env python3

import importlib.util
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_real_inference.py"
spec = importlib.util.spec_from_file_location("report_real_inference", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> None:
    log = """llama_model_loader: - kv  11: general.file_type u32 = 7
print_info: file type = Q8_0
llama_kv_cache: CPU KV buffer size = 56.00 MiB
llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens (10.00 ms per token, 100.00 tokens per second)
llama_perf_context_print:        eval time = 200.00 ms / 5 runs   (40.00 ms per token, 25.00 tokens per second)
SLHA_EXTERNAL_K_STORE valid=true layers=28 capacity=2048 tile_bytes=128 logical_tile_bytes=7340032 tile_backing_capacity_bytes=7340159 validity_backing_capacity_bytes=57344
SLHA_REPLACE_SUMMARY
callbacks=28
active_expected_vectors=100
active_replaced_vectors=100
active_expected_logits=200
active_replaced_logits=200
failed_vectors=0
fallback_vectors=0
error_code=0
active_coverage=1
valid=true
layer_0_success=1 layer_0_fail=0
"""
    prompt_tps, decode_tps = module.parse_perf_tps(log)
    assert prompt_tps == 100.0
    assert decode_tps == 25.0

    store = module.parse_external_store(log)
    assert store is not None
    assert store["valid"] is True
    assert store["layers"] == 28
    assert store["logical_tile_bytes"] == 7340032

    summary = module.parse_replace_summary(log)
    assert summary is not None
    assert summary["valid"] is True
    assert summary["failed_vectors"] == 0
    assert summary["active_coverage"] == 1

    assert module.parse_perf_tps("prompt eval time = no rate\n") == (None, None)
    assert module.parse_external_store("ordinary baseline\n") is None
    assert module.parse_replace_summary("ordinary baseline\n") is None

    with tempfile.TemporaryDirectory() as td:
        time_path = Path(td) / "time.txt"
        time_path.write_text("max_rss_kb=12345\nelapsed_s=1.25\n")
        text = time_path.read_text()
        assert module.parse_time_value(text, "max_rss_kb") == 12345.0
        assert module.parse_time_value(text, "elapsed_s") == 1.25
        assert module.parse_time_value(text, "missing") is None

    print("real_inference_report_tests: ok")


if __name__ == "__main__":
    main()
