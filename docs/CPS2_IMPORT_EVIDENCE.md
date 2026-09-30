# CPS-2 destination import execution

Scope: synthetic diagnostic only; no runtime promotion.

- Executed at UTC: `2026-09-30T20:36:04Z`.
- SLHAv2 public CPU worker run: `36773552599`, attempt `1`.
- Worker source revision: `d8890c3d521df23746be52b641db442a4d2081d9`.
- RemoteOps qualification plan: `c6816d87b28d5011e2e8304ea6ff21cd44baaeb4`.
- Destination checkout before formatting: `7f57822411aa869df5e87690dee9972aa105b172`.
- KVLab revision: `265a2a65120b8f6f06c3cbb6604cf821ceda94d0`.
- FLAT revision: `ad1634fc922f6223dd3a83ac84154a82b1a35562`.
- Raw CSV SHA-256: `2457b70e8e75cf29db366f8b11761efa084f1da3bec9ed49133c6436ac2c7216`.
- Raw fixture: `scirust/tests/fixtures/kvlab_cps2_v1.csv`.
- Repetition: two executions of the actual upstream Rust binary; byte-identical output.
- Complete observations: 50; input header: 21 columns.
- Execution backend: GitHub-hosted Linux CPU runner.
- Architecture: `x86_64`; compiler: `rustc 1.98.1 (48a229cea 2026-09-01)`.

Formatting, strict default-feature Clippy and all 14 destination regression tests passed before this record was written. Full repository CI remains a separate merge gate.

The record is unsigned provenance, not worker attestation. No model or protected holdout was accessed. No Thor execution, GPU, quality, memory, traffic or performance result is claimed.

## Actual destination command output

```text
schema=slha.cps2-development-import/v1
asserted_kvlab_revision=265a2a65120b8f6f06c3cbb6604cf821ceda94d0
rows=50
scope=unsigned_synthetic_diagnostic
runtime_promotion=false
case,arm,rows,mean_top_k_recall,mean_retained_mass,max_output_error
aligned_coordinate,all_accept,5,1.000000000000,1.000000000000,0.000000000
aligned_coordinate,compact_projected,5,1.000000000000,0.870530303812,0.279117346
aligned_coordinate,full_score_topk,5,1.000000000000,0.870530303812,0.279117346
aligned_coordinate,recent_tail,5,1.000000000000,0.870530303812,0.279117346
aligned_coordinate,matched_random,5,0.300000000000,0.271043778911,1.720883131
omitted_dominant_coordinate,all_accept,5,1.000000000000,1.000000000000,0.000000000
omitted_dominant_coordinate,compact_projected,5,0.500000000000,0.000000153935,3.731058121
omitted_dominant_coordinate,full_score_topk,5,1.000000000000,0.999999937768,0.000000160
omitted_dominant_coordinate,recent_tail,5,0.500000000000,0.000000153935,3.731058121
omitted_dominant_coordinate,matched_random,5,0.500000000000,0.799999906466,1.731058002
```
