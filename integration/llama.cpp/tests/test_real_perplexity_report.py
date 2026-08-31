#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_real_perplexity.py"
spec = importlib.util.spec_from_file_location("report_real_perplexity", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def args_for(root: Path, mode: str, log: Path) -> argparse.Namespace:
    corpus = root / "holdout.txt"
    corpus.write_text("A distinct held-out story used only for evaluation.\n" * 16)
    return argparse.Namespace(
        mode=mode,
        log=str(log),
        corpus=str(corpus),
        model_sha256="a" * 64,
        llama_commit="b" * 40,
        context_size=128,
        chunks=2,
        threads=2,
        gpu_layers=0,
        cache_type_k="f16",
        cache_type_v="f16",
        output=str(root / f"{mode}.json"),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        baseline_log = root / "baseline.log"
        baseline_log.write_text(
            "perplexity_v2: computing over 2 chunks, n_ctx=128, batch_size=128, n_seq=1\n"
            "Final estimate: PPL = 12.3456 +/- 0.12345\n"
        )
        baseline = mod.build_report(args_for(root, "baseline", baseline_log))
        assert baseline["perplexity"] == 12.3456
        assert baseline["uncertainty"] == 0.12345
        assert baseline["batch_size"] == 128
        assert baseline["parallel"] == 1
        assert baseline["external_replace_valid"] is None

        external_log = root / "external.log"
        external_log.write_text(
            "perplexity_v2: computing over 2 chunks, n_ctx=128, batch_size=128, n_seq=1\n"
            "Final estimate: PPL = 13.0000 +/- 0.20000\n"
            "SLHA_EXTERNAL_K_STORE valid=true backend=ccos_elastic peak_resident_bytes=4096 peak_cold_slots=0 budget_failures=0\n"
            "SLHA_REPLACE_SUMMARY\n"
            "callbacks=8\n"
            "active_coverage=1\n"
            "failed_vectors=0\n"
            "fallback_vectors=0\n"
            "n_stream=1\n"
            "valid=true\n\n"
        )
        external = mod.build_report(args_for(root, "external", external_log))
        assert external["perplexity"] == 13.0
        assert external["external_replace_valid"] is True
        assert external["external_backend"] == "ccos_elastic"
        assert external["replace_summary"]["valid"] is True

        assert mod.parse_final_estimate("Final estimate: PPL = 1.25 +/- 0.01\n") == (1.25, 0.01)
        for invalid in (
            "no final estimate here",
            "Final estimate: PPL = nan +/- 0.1",
            "Final estimate: PPL = -1.0 +/- 0.1",
        ):
            try:
                mod.parse_final_estimate(invalid)
            except (ValueError, OverflowError):
                pass
            else:
                raise AssertionError(f"expected invalid final estimate: {invalid!r}")

        bad_external = root / "bad-external.log"
        bad_external.write_text(
            "Final estimate: PPL = 10.0 +/- 0.1\n"
            "SLHA_REPLACE_SUMMARY\nvalid=false\n\n"
        )
        try:
            mod.build_report(args_for(root, "external", bad_external))
        except ValueError as exc:
            assert "valid SLHA_REPLACE_SUMMARY" in str(exc)
        else:
            raise AssertionError("external invalid replacement summary was accepted")

    print("test_real_perplexity_report: ok")


if __name__ == "__main__":
    main()
