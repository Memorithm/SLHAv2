#!/usr/bin/env python3
from __future__ import annotations

import array
import importlib.util
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_real_eval.py"
spec = importlib.util.spec_from_file_location("compare_real_eval", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_f32(path: Path, rows: list[list[float]]) -> None:
    values = array.array("f")
    for row in rows:
        values.extend(row)
    path.write_bytes(values.tobytes())


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        baseline_logits = root / "baseline.f32"
        external_logits = root / "external.f32"
        # Generated token divergence is at row 1. Row 1 remains scientifically
        # comparable because both runs entered it with the same preceding
        # context. Row 2 must be excluded because the contexts have diverged.
        write_f32(
            baseline_logits,
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [1000.0, 0.0, 0.0, 0.0],
            ],
        )
        write_f32(
            external_logits,
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 2.0, 1.0],
                [-1000.0, 0.0, 0.0, 0.0],
            ],
        )
        metrics = mod.compare_logits(str(baseline_logits), str(external_logits), 2, 4)
        assert metrics is not None
        assert metrics["comparable_rows"] == 2
        assert metrics["values_compared"] == 8
        assert metrics["argmax_agreement_rows"] == 1
        assert metrics["max_absolute_error"] == 1.0

        assert mod.first_divergence([3, 1, 9], [3, 2, 8]) == 1
        assert mod.first_divergence([3, 1], [3, 1]) is None
        assert mod.first_divergence([3], [3, 1]) == 1
        assert mod.percentile([10.0, 20.0, 30.0], 0.50) == 20.0
        assert mod.percentile([], 0.95) is None

        kv = mod.parse_kv_components(
            "K (f16): 2.00 MiB, V (f16): 3.00 MiB\n"
        )
        assert kv["k_bytes"] == 2 * 1024 * 1024
        assert kv["v_bytes"] == 3 * 1024 * 1024

        external_log = root / "external.log"
        external_log.write_text(
            "SLHA_EXTERNAL_K_STORE allocated_bytes=1234 capacity=64\n"
            "SLHA_REPLACE_SUMMARY\n"
            "callbacks=2\nvalid=true\n\n"
        )
        assert mod.parse_external_store(external_log.read_text())["allocated_bytes"] == 1234
        assert mod.parse_key_values_after_marker(
            external_log.read_text(), "SLHA_REPLACE_SUMMARY"
        )["valid"] is True

    print("test_compare_real_eval: ok")


if __name__ == "__main__":
    main()
