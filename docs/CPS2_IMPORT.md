# CPS-2 development evidence import

Status: destination-owned synthetic verifier, 2026-09-30. This slice does not
change the LR1 candidate, SLHA scorer, codecs, cache layout or runtime defaults.

## Immutable upstream scope

- KVLab producer: `265a2a65120b8f6f06c3cbb6604cf821ceda94d0` (PR #175).
- FLAT producer: `ad1634fc922f6223dd3a83ac84154a82b1a35562` (CPS-1).
- Upstream schema: `kvlab.cps2-compact-quality/v1`.
- Upstream protocol: `docs/CPS2_COMPACT_PRESELECTION_PROTOCOL.md` in that KVLab tree.
- Producer command: `cargo run --manifest-path rust/bkv_scan/Cargo.toml --bin cps2_compact_panel`.

The verifier accepts only the two frozen N=5, D=2 non-causal fixtures:
`aligned_coordinate` and `omitted_dominant_coordinate`. Each has five arms and
five query rows. The accepted complete panel therefore has exactly 50 unique
(case, arm, query-row) observations. This is not a real-model trace format.

## Destination checks

The zero-dependency `scirust::kvlab_cps2` module checks the exact 21-column header,
source/schema assertions, complete coverage, original key positions, deterministic
row-scoped random control, candidate budget, reference top-2 membership, hit count,
recall, selected density, logical selector components and numerical pair counts.
Missing rows, duplicate rows, unknown fields/arms/cases and non-finite diagnostics
are rejected. Input is limited to 65,536 bytes; the command reads at most that
limit plus one byte before rejection.

An independent scalar oracle re-evaluates the published synthetic formulas:

```text
aligned: Q = [1, 0], K_j = [j, 0], V_j = [j, 4-j]
adverse: Q = [1, 1], K_0 = [0, 20], K_j = [j, 0] for j > 0
projection = [0], candidate budget = 2, reference k = 2
random seed = 0x43505332
```

Because V has coordinates j and 4-j, the maximum output-coordinate error is
the absolute difference of the full and selected weighted means of j. The LSE
error is minus the logarithm of retained mass. These are independent fixture
identities, not a second general attention implementation.

Verification tolerances are declared before retaining the producer output:
absolute tolerance 1e-12 for printed mass/density diagnostics and 4e-6 for O/LSE
errors (the producer subtracts f32 outputs, while this oracle evaluates in f64).
The all-accept control must report exactly zero O/LSE error. These tolerances are
specific to the frozen bounded fixture, not SLHAv2 model-quality acceptance rules.

The negative fixture remains negative. Correctly reported loss of the dominant
key must be retained and accepted as a diagnostic; it must not be relabeled a
successful quality result or tuned away.

## Execution and evidence

The retained CSV must come from the actual pinned upstream Rust executable,
not a hand-written reproduction. Run the producer twice and require byte-identical
output. Record the raw CSV SHA-256, source identities, toolchain and RemoteOps run
in `docs/CPS2_IMPORT_EVIDENCE.md` when that execution has completed.

Destination commands:

```bash
cargo --locked fmt --all --check
cargo --locked clippy -p scirust --all-targets -- -D warnings
cargo --locked test -p scirust --test kvlab_cps2
cargo --locked run -p scirust --example cps2_inspect -- \
  scirust/tests/fixtures/kvlab_cps2_v1.csv \
  265a2a65120b8f6f06c3cbb6604cf821ceda94d0
```

The existing complete repository CI, including MSRV and aarch64 checks, remains
required before merge. No dependency or CI gate is removed or weakened.

## Trust and promotion boundary

A matching revision string is an asserted provenance field, not cryptographic
proof of origin. The parser verifies content against the frozen formulas; it
does not attest a worker or authenticate an execution. Preserve the original
artifact and execution provenance alongside the report.

All four upstream measurement/promotion flags must remain false. The returned
report is read-only and `permits_runtime_promotion()` always returns false.
It is not consumed by the SLHA scorer, cache or ElasticXxx controller.

This bridge demonstrates a destination-verifiable scientific diagnostic only.
It provides no PPL/NLL improvement, physical KV memory saving, latency, bandwidth,
GPU acceleration or model-quality claim. Full numerical K/V and the dense
fallback remain authoritative. Real-model capture and destination qualification
require their own frozen identities and protocol, without accessing or retuning
the protected LR1 holdout.
