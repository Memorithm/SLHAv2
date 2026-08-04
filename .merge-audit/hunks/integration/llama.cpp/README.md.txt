file=integration/llama.cpp/README.md
conflict_count=7

### conflict lines 1-30
>000001: <<<<<<< HEAD
>000002: # SLHA v2 × llama.cpp — K-cache round-trip and compressed-score quality gates
>000003: 
>000004: This directory implements and measures real-LLM quality-path integrations for
>000005: SLHA v2 on llama.cpp. Two milestones are recorded here:
>000006: 
>000007: 1. **K-cache round-trip** — every K vector is encoded to a 128-byte SLHA tile
>000008:    and decoded back to the original K dimension *before* it is stored in
>000009:    llama.cpp's normal KV cache. Attention is untouched.
>000010: 2. **Direct compressed-score replacement path** — a custom GGML operation
>000011:    replaces the attention logits with scores computed directly from the
>000012:    compressed SLHA tiles, after llama.cpp has already materialised the baseline
>000013:    Q·K product.
>000014: 
>000015: > Neither milestone is a fused attention kernel or a physically compressed
>000016: > KV-cache implementation. No KV-cache memory reduction and no attention-speed
>000017: > gain is claimed.
>000018: =======
>000019: # SLHA v2 × llama.cpp — compressed-score (fused-QK) integration
>000020: 
>000021: This directory implements and measures the real-LLM quality path for SLHA v2:
>000022: the **fused-QK** mode encodes every K vector to a 128-byte SLHA tile and
>000023: **replaces the QK^T attention scores** with SLHA scores computed directly on
>000024: the tiles (no K reconstruction). It also keeps the earlier round-trip path
>000025: (encode → decode K before storage) as a measured reference.
>000026: 
>000027: > This is **not** the final fused-score bandwidth phase: the fused node is a
>000028: > CPU GGML callback, so it does not yet claim any KV memory reduction or
>000029: > attention-speed gain. This milestone measures *quality* (perplexity).
>000030: >>>>>>> refs/remotes/origin/pr71
 000031: 
 000032: ## Status
 000033: 
 000034: Implemented and measured:

### conflict lines 41-298
 000037: * Per-layer K activation collection mode (writes `layer-N-k.bin`).
 000038: * Automated per-layer projection training script (outputs `layer-NNN.slhw` +
 000039:   `manifest.json`).
 000040: * SLHA K round-trip callback using `slha_encode_key` + `slha_decode_key`.
>000041: <<<<<<< HEAD
>000042: * Shadow-score quality gate: `SLHA_SCORE_MODE=shadow` compares baseline Q·K
>000043:   logits with direct SLHA scores while leaving attention output unchanged.
>000044: * **Strict direct compressed-score replacement**: `SLHA_SCORE_MODE=replace`
>000045:   substitutes the logits and fails closed unless every active vector was
>000046:   replaced.
>000047: * Reproducible build/apply/measure scripts.
>000048: 
>000049: ## The direct compressed-score replacement path
>000050: 
>000051: What it **does**:
>000052: 
>000053: * A custom GGML operation is inserted after `build_attn_mha` computes `kq`.
>000054:   It overwrites the attention logits with scores computed directly from the
>000055:   compressed SLHA tile side-store.
>000056: * It is **fail-closed**: the run is rejected unless every active vector was
>000057:   replaced, with no failures and no fallbacks.
>000058: * Active logits and padding logits are accounted for **separately**. Padding
>000059:   and inactive-stream positions are excluded from the count of logits directly
>000060:   computed by SLHA.
>000061: 
>000062: What it **does not** do:
>000063: 
>000064: * **Baseline Q·K is still materialised by llama.cpp.** The custom operation
>000065:   replaces logits *after* the baseline matrix multiplication; it does not avoid
>000066:   that matrix multiplication.
>000067: * No fused attention kernel is implemented.
>000068: * No physical K-cache memory reduction is claimed. The compressed tiles live in
>000069:   a side-store; the normal KV cache is unchanged.
>000070: * Strict replace mode currently supports **one parallel sequence** only
>000071:   (`n_stream == 1`).
>000072: 
>000073: ## Provenance of the recorded measurements
>000074: 
>000075: ```text
>000076: SLHAv2 implementation commit : 6361dfdbcd30660bf2d623fe19029938dd209cd7
>000077: llama.cpp commit             : fdb1db877c526ec90f668eca1b858da5dba85560 (tag b9860)
>000078: measurement branch           : claude/compressed-score-quality-gate-qiqbm5
>000079: ```
>000080: 
>000081: The original feature-branch commits (canonical compressed-score C API, shadow
>000082: score gate, richer shadow metrics, strict fail-closed replacement) were
>000083: **squash-merged through PR #57** into implementation commit `6361dfd`. The
>000084: strict replacement implementation was therefore measured from `6361dfd`; the
>000085: pre-merge SHAs no longer exist in this repository. The documentation commit
>000086: recording these results is necessarily later than the implementation commit it
>000087: measures.
>000088: 
>000089: ## Headline results
>000090: 
>000091: Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 test, mixed codec. All runs used
>000092: `chunks = 12`, `context length = 512`, `batch size = 512`,
>000093: `parallel sequences = 1`, `threads = 4`, `flash attention = off`, verified from
>000094: each run's own runtime output.
>000095: 
>000096: | Metric | Value |
>000097: | --- | ---: |
>000098: | Unpatched baseline mean PPL (rung A, n=3) | 11.8779 |
>000099: | Pass-through control mean PPL (rung D, n=3) | 11.8831 |
>000100: | Strict replace mean PPL (rung F, n=3) | 16.9173 |
>000101: | Strict replace sample standard deviation | 0.0525 |
>000102: | Primary gap vs pass-through control (F − D) | +5.0342 PPL |
>000103: | Primary relative gap | +42.364 % |
>000104: | Active coverage | 1 |
>000105: | Failed vectors | 0 |
>000106: | Fallback vectors | 0 |
>000107: | Padding logits audited | 170688 |
>000108: | Padding nonzero count | 0 |
>000109: | Strict replace seconds per pass | ≈ 9.40 |
>000110: | Strict replace tokens per second | ≈ 54.5 |
>000111: | Pass-through control seconds per pass | ≈ 3.02 |
>000112: | Pass-through control tokens per second | ≈ 169.5 |
>000113: 
>000114: The direct compressed-score replacement path produced a large observed PPL
>000115: degradation relative to the pass-through integration control. The effect was
>000116: far larger than the run-to-run variability observed in the control ladder. Its
>000117: precise cause has not yet been isolated.
>000118: 
>000119: The observed quality gap applies to the current model, mixed codec, projection
>000120: weights, score implementation and evaluation protocol. It is **not** attributed
>000121: solely to quantization. The experiment does not determine whether the gap
>000122: originates from codec quantization, projection training, score scaling, query
>000123: preparation, attention temperature, layer-specific error accumulation, padding
>000124: behaviour, tile representation or another integration effect.
>000125: 
>000126: ## Six-rung control ladder
>000127: 
>000128: Three complete runs per rung, identical protocol. All 18 accepted runs exited
>000129: with status `0`.
>000130: 
>000131: | Rung | Description | run 1 | run 2 | run 3 | mean | sample sd | spread |
>000132: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
>000133: | A | unpatched baseline | 11.8837 | 11.8737 | 11.8764 | 11.8779 | 0.0052 | 0.0100 |
>000134: | B | patched llama.cpp, SLHA fully off | 11.8770 | 11.8766 | 11.8873 | 11.8803 | 0.0061 | 0.0107 |
>000135: | C | tilestore K-write hook only | 11.8785 | 11.8829 | 11.8787 | 11.8800 | 0.0025 | 0.0044 |
>000136: | D | padaudit pass-through custom operation | 11.8775 | 11.8768 | 11.8949 | 11.8831 | 0.0103 | 0.0181 |
>000137: | E | padzero | 11.8811 | 11.8872 | 11.8762 | 11.8815 | 0.0055 | 0.0110 |
>000138: | F | strict compressed-score replacement | 16.8573 | 16.9396 | 16.9549 | 16.9173 | 0.0525 | 0.0976 |
>000139: 
>000140: The **observed unpatched-baseline run-to-run spread = 0.0100 PPL across three
>000141: runs**. This is an observed variability figure, not a formally characterised
>000142: numerical noise floor.
>000143: 
>000144: ### Contrasts
>000145: 
>000146: Difference of means with propagated standard error. All intervals are
>000147: **approximate exploratory intervals based on three independent runs per group**
>000148: (t = 2.776, df = 4). **Statistical power is limited.**
>000149: 
>000150: | Contrast | Δ PPL | Δ % | approximate 95 % CI | Interpretation |
>000151: | --- | ---: | ---: | --- | --- |
>000152: | B − A, patch/build | +0.0024 | +0.020 % | [−0.0104, +0.0151] | not resolved at this sample size |
>000153: | C − B, tilestore hook | −0.0003 | −0.002 % | [−0.0108, +0.0102] | not resolved at this sample size |
>000154: | D − C, pass-through custom operation | +0.0030 | +0.026 % | [−0.0139, +0.0199] | not resolved at this sample size |
>000155: | E − D, padding zeroing | −0.0016 | −0.013 % | [−0.0202, +0.0171] | not resolved at this sample size |
>000156: | F − D, direct score substitution | +5.0342 | +42.364 % | [+4.9485, +5.1199] | large resolved effect |
>000157: | F − A, secondary comparison | +5.0393 | +42.426 % | [+4.9548, +5.1239] | large resolved effect |
>000158: 
>000159: "Not resolved at this sample size" is **not** a claim of equivalence or of a
>000160: proven absent effect.
>000161: 
>000162: Numerical PPL results show run-to-run variation even in **unpatched**
>000163: llama.cpp. The production counters were fully deterministic across repeats; the
>000164: numerical variation was not instrumentally isolated.
>000165: 
>000166: ## Strict replacement counters
>000167: 
>000168: Identical across all three accepted runs, imported programmatically from the
>000169: immutable run logs:
>000170: 
>000171: ```text
>000172: callbacks                = 1456
>000173: active_expected_vectors  = 2065056
>000174: active_replaced_vectors  = 2065056
>000175: active_expected_logits   = 1056965952
>000176: active_replaced_logits   = 1056965952
>000177: padding_vectors          = 672
>000178: padding_logits           = 170688
>000179: inactive_stream_vectors  = 0
>000180: inactive_stream_logits   = 0
>000181: failed_vectors           = 0
>000182: fallback_vectors         = 0
>000183: missing_tile             = 0
>000184: query_prep_fail          = 0
>000185: score_fail               = 0
>000186: nonfinite_score          = 0
>000187: unsupported_shape        = 0
>000188: unsupported_stride       = 0
>000189: error_code               = 0
>000190: n_stream                 = 1
>000191: active_coverage          = 1
>000192: valid                    = true
>000193: ```
>000194: 
>000195: `build_and_roundtrip.sh replace` rejects the run unless these conditions hold,
>000196: so incomplete or fallback runs are never reported as results.
>000197: 
>000198: ## Padding audit
>000199: 
>000200: Deterministic raw-logit audit over the padded region `k >= n_written`,
>000201: identical across all six diagnostic runs:
>000202: 
>000203: ```text
>000204: audited_padded_vectors           = 672
>000205: audited_padded_logits            = 170688
>000206: max_abs_padded_baseline_logit    = 0
>000207: nonzero_padded_baseline_logits   = 0
>000208: nonfinite_padded_baseline_logits = 0
>000209: ```
>000210: 
>000211: `audited_padded_logits` cross-checks exactly against the strict-replacement
>000212: `padding_logits` counter.
>000213: 
>000214: For this pinned model and protocol, all audited padded baseline Q·K logits were
>000215: exactly zero. Writing `0.0f` to those positions is therefore an arithmetic
>000216: no-op.
>000217: 
>000218: The empirical `padzero − padaudit` comparison (−0.0016 PPL, approximate 95 % CI
>000219: [−0.0202, +0.0171], not resolved at this sample size) is **supporting
>000220: evidence, not the primary proof**.
>000221: 
>000222: This exact-zero result must **not** be generalised to other models, other
>000223: llama.cpp versions or other KV-cache implementations.
>000224: 
>000225: ## Throughput
>000226: 
>000227: | Path | seconds per pass | tokens per second |
>000228: | --- | ---: | ---: |
>000229: | padaudit pass-through control | ≈ 3.02 | ≈ 169.5 |
>000230: | strict compressed-score replacement | ≈ 9.40 | ≈ 54.5 |
>000231: 
>000232: Strict replacement is approximately **3.11× slower by seconds per pass**. This
>000233: is a quality-gate implementation using a custom CPU operation — not a fused
>000234: kernel and not an optimised performance benchmark.
>000235: 
>000236: ## Calibration preprocessing (out-of-tree)
>000237: 
>000238: The projection weights used by the strict replacement runs were **not** trained
>000239: directly from the raw collected dumps. A deterministic out-of-tree row-removal
>000240: step was required first:
>000241: 
>000242: ```text
>000243: 27 total rows removed
>000244: layers 1–27 affected
>000245: one removed row per affected layer
>000246: row index 0 or 1
>000247: all removed rows contained NaN
>000248: zero removed rows contained positive or negative infinity
>000249: 6146 raw rows to 6145 clean rows per affected layer
>000250: layer 0 unchanged
>000251: ```
>000252: 
>000253: The rule is deterministic and limited to removing complete rows containing any
>000254: non-finite value. No finite value was imputed, replaced, clamped or otherwise
>000255: modified.
>000256: 
>000257: ```text
>000258: sanitization script SHA-256:
>000259: 4f0d7e3f0bbd557afd99085ca899d45bc4916bd0a9e4d014e26f7775b9ba32a4
>000260: ```
>000261: 
>000262: The position and pattern are consistent with a warmup-like callback, but the
>000263: exact upstream origin was not instrumentally proven.
>000264: 
>000265: The weights were generated after the squash merge from implementation commit
>000266: `6361dfdbcd30660bf2d623fe19029938dd209cd7` using an out-of-tree deterministic
>000267: non-finite-row removal step.
>000268: 
>000269: Per-layer raw and cleaned hashes, aggregate calibration hashes, all weight-file
>000270: hashes, the manifest hash and the exact training command are recorded in
>000271: [`results/measurements.json`](results/measurements.json).
>000272: =======
>000273: * **Fused-QK mode** (`SLHA_KV_MODE=fused`): every K vector is encoded to a
>000274:   128-byte tile at `cpy_k` time, and the QK^T attention scores are **replaced**
>000275:   by SLHA scores computed directly on the tiles (`slha_prepare_query` +
>000276:   `slha_process_tile`). The softmax then operates on the SLHA scores; V is
>000277:   untouched. This is the "score compressed tiles directly" path the round-trip
>000278:   NO-GO explicitly left open.
>000279: 
>000280: Measured conclusions:
>000281: 
>000282: * **Round-trip (K reconstruction): NO-GO** — ΔPPL ≈ +40 % (see
>000283:   [`results/measurements.json`](results/measurements.json)).
>000284: * **Fused-QK: NO-GO** — the SLHA scores do not preserve the attention
>000285:   distribution well enough to replace QK^T. Measured on the same config
>000286:   (Qwen2.5-1.5B Q8_0, WikiText-2 test, 12 chunks, 512 ctx, 4 threads):
>000287:   PPL = **19830** vs baseline **11.88**, with score-diagnostic cos ≈ 0.68 and
>000288:   KL ≈ 5.2 on deep layers. The fused path is *functional* (it compiles, runs,
>000289:   and demonstrably replaces the scores — see the diagnostics), but the
>000290:   quality gate fails massively.
>000291: 
>000292: The fused-QK result points to the open problem: the per-layer projections are
>000293: trained on K alone (`fit_with`), and the resulting SLHA scores are correlated
>000294: with the true QK^T (cos ≈ 0.68) but with a very different softmax
>000295: distribution (KL ≈ 5). Closing the gap would require joint K/Q calibration,
>000296: score-scale calibration, or a much larger latent/residual budget — all future
>000297: work, not validated here.
>000298: >>>>>>> refs/remotes/origin/pr71
 000299: 
 000300: > **Update.** The out-of-tree sanitization above is no longer required: the
 000301: > production pipeline now validates calibration and fails closed on any
 000302: > non-finite row. See **Calibration integrity policy** below.

### conflict lines 456-473
 000452:   integration/llama.cpp/scripts/train_layer_weights.sh mixed
 000453: 
 000454: # 7. Round-trip perplexity
 000455: SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip
>000456: <<<<<<< HEAD
>000457: 
>000458: # 8. Shadow-score quality gate (attention unchanged)
>000459: SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh shadow
>000460: 
>000461: # 9. Strict direct compressed-score replacement (fails closed)
>000462: SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh replace
>000463: =======
>000464: SLHA_CODEC=mix3 WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip
>000465: 
>000466: # 8. Fused-QK perplexity (scores computed directly on the tiles)
>000467: SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh fused
>000468: 
>000469: # 9. Fused-QK score diagnostics (cos / KL per layer, opt-in)
>000470: SLHA_KV_MODE=fused SLHA_FUSED_DIAG=1 SLHA_WEIGHTS_DIR=/tmp/slha-llama/weights \
>000471:   llama.cpp/build/bin/llama-perplexity -m qwen2.5-1.5b-instruct-q8_0.gguf \
>000472:   -f wiki.test.raw --chunks 2 -t 4 -fa off
>000473: >>>>>>> refs/remotes/origin/pr71
 000474: ```
 000475: 
 000476: All scripts pin the llama.cpp tag (`b9860`) and verify the commit hash before
 000477: building.

### conflict lines 483-510
 000479: ## Runtime interface
 000480: 
 000481: Modes are selected via environment variables:
 000482: 
>000483: <<<<<<< HEAD
>000484: | Variable           | Values                              |
>000485: |--------------------|-------------------------------------|
>000486: | `SLHA_KV_MODE`     | `off` / `passthrough` / `collect` / `roundtrip` / `tilestore` |
>000487: | `SLHA_SCORE_MODE`  | `off` (default) / `shadow` / `replace` |
>000488: | `SLHA_CODEC`       | `mixed` (default) / `mix3` / `grouped` / `nf4` / `tq3` |
>000489: | `SLHA_WEIGHTS_DIR` | directory with `layer-NNN.slhw` and `manifest.json` |
>000490: | `SLHA_CALIBRATION_NONFINITE_POLICY` | `reject` (default) / `drop-row` (research/recovery only) |
>000491: | `SLHA_CALIBRATION_MIN_ROWS` | minimum rows a layer must retain under `drop-row` (default 1) |
>000492: 
>000493: Shadow and replace modes require `SLHA_KV_MODE=tilestore` so the K tiles are
>000494: encoded at the K-cache write seam. Both force `--flash-attn off --parallel 1`
>000495: so the baseline logits are materialised and the tile-store positions stay
>000496: contiguous within a single sequence. Replace mode additionally pins
>000497: `--batch-size 512`.
>000498: =======
>000499: | Variable          | Values                              |
>000500: |-------------------|-------------------------------------|
>000501: | `SLHA_KV_MODE`    | `off` / `passthrough` / `collect` / `roundtrip` / `scorediag` / `fused` |
>000502: | `SLHA_CODEC`      | `mixed` (default) / `mix3` / `grouped` / `nf4` / `tq3` |
>000503: | `SLHA_WEIGHTS_DIR`| directory with `layer-NNN.slhw` and `manifest.json` |
>000504: 
>000505: `fused` and `scorediag` require the standard (non-flash) attention path: the
>000506: build script passes `-fa off` for these modes because the SLHA custom node is a
>000507: GGML op on that path (flash attention would bypass it).
>000508: 
>000509: ## Measured results
>000510: >>>>>>> refs/remotes/origin/pr71
 000511: 
 000512: ## Research: layerwise SLHA score-quality diagnosis
 000513: 
 000514: <<<<<<< HEAD

### conflict lines 514-531
 000510: >>>>>>> refs/remotes/origin/pr71
 000511: 
 000512: ## Research: layerwise SLHA score-quality diagnosis
 000513: 
>000514: <<<<<<< HEAD
>000515: This section diagnoses **which layers** the direct compressed-score replacement
>000516: path degrades and **which score distortion** predicts that degradation. It is a
>000517: diagnostic milestone — the production score mathematics are unchanged.
>000518: =======
>000519: | Mode        |      PPL | ΔPPL absolute | ΔPPL relative | Notes           |
>000520: | ----------- | -------: | ------------: | ------------: | --------------- |
>000521: | baseline    | 11.8753  |             — |             — | original        |
>000522: | passthrough | 11.8753  |          0.00 |          0.0% | hook sanity     |
>000523: | mixed       | 16.5976  |          4.72 |         39.8% | round-trip      |
>000524: | mix3        | 16.6460  |          4.77 |         40.2% | round-trip      |
>000525: | fused       | 19830.9  |       19819.0 |    ~167000 %  | fused-QK (this run) |
>000526: 
>000527: The fused row is the measurement made on this machine (aarch64, 4 threads) with
>000528: `SLHA_CODEC=mixed`, `-fa off`, 12 chunks. It is a hard NO-GO: the SLHA scores
>000529: do not preserve the attention distribution (see the score diagnostics: cos ≈
>000530: 0.68, KL ≈ 5 on deep layers).
>000531: >>>>>>> refs/remotes/origin/pr71
 000532: 
 000533: Provenance: SLHAv2 `23e27c0`, llama.cpp `fdb1db877c526ec90f668eca1b858da5dba85560`,
 000534: Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 test, mixed codec, weights from the
 000535: corrected PR #59 pipeline. Full data + hashes:

### conflict lines 540-1179
 000536: [`results/layerwise_score_gap.json`](results/layerwise_score_gap.json).
 000537: 
 000538: ### Experimental layer-mask interface
 000539: 
>000540: <<<<<<< HEAD
>000541: `SLHA_SCORE_LAYERS` selects which layers use direct SLHA score replacement;
>000542: unselected layers pass baseline Q·K through unchanged.
>000543: 
>000544: | Spec | Meaning | | Spec | Meaning |
>000545: | --- | --- | --- | --- | --- |
>000546: | `all` | every layer (default) | | `0-6` | inclusive range |
>000547: | `none` | no layer (only valid empty) | | `0-3,7,12-14` | combined |
>000548: | `7` / `3,7,12` | single / list | | (invalid) | fails closed → `valid=false` |
>000549: 
>000550: Parsing is strict: negatives, out-of-range ids (checked against the model layer
>000551: count), malformed ranges, and an empty spec are rejected; duplicates are
>000552: de-duplicated deterministically. `SLHA_SCORE_MASK_SUMMARY` reports the requested,
>000553: resolved, and executed masks plus per-selected-layer coverage.
>000554: 
>000555: ### Method
>000556: 
>000557: Screening ran at **4 chunks** (reduced from 12 — this session's host was
>000558: materially slower); the pass-through control used the same 4 chunks, so every
>000559: delta is consistent. Deltas are versus the **pass-through custom-op control**
>000560: (B). One run per configuration (screening).
>000561: 
>000562: ### Controls and headline
>000563: 
>000564: | Control | mean PPL (4 chunks) | Δ vs B |
>000565: | --- | ---: | ---: |
>000566: | A unpatched baseline | 9.3852 | — |
>000567: | B pass-through custom op | 9.3852 | 0.0000 |
>000568: | C all-layer replacement | 13.5421 | **+4.1569 (+44.3%)** |
>000569: 
>000570: The custom op is inert (A = B); all-layer replacement reproduces PR #58's ~+42 %
>000571: gap at the reduced chunk count.
>000572: 
>000573: **The degradation is distributed and super-additive.** The five most-damaging
>000574: single layers are 5 (+0.577), 0 (+0.253), 12 (+0.150), 8 (+0.122), 24 (+0.121);
>000575: the five least are 21 (+0.020), 27 (+0.007), 18 (+0.006), 25 (−0.057), 3
>000576: (−0.065) — two layers *improve* PPL alone. No single layer dominates: the largest
>000577: is 0.58 of the 4.16 total. The **sum of the 28 single-layer deltas is 2.245**,
>000578: but all-layer replacement is **4.157** — a **1.85× super-additive** amplification:
>000579: errors compound through the residual stream.
>000580: 
>000581: Cumulative prefixes confirm progressive, mid-network-weighted accumulation:
>000582: 
>000583: | Prefix | ΔPPL | | Prefix | ΔPPL |
>000584: | --- | ---: | --- | --- | ---: |
>000585: | `0` | +0.253 | | `0-13` | +2.853 |
>000586: | `0-3` | +0.393 | | `0-20` | +3.773 |
>000587: | `0-6` | +1.546 | | `0-27` | +4.157 |
>000588: 
>000589: The largest increments fall in the middle of the stack (layers ~4–20).
>000590: Cumulative **suffixes** (from the top of the stack) confirm the front-loading:
>000591: `14-27` (last 14 layers) is only +0.579 while `0-13` (first 14) is +2.853 — the
>000592: **first half causes ~5× the damage of the second half**. The four quartiles
>000593: `0-6 / 7-13 / 14-20 / 21-27` contribute **+1.55 / +0.49 / +0.45 / +0.15**: damage
>000594: is concentrated in the **first quartile** and tapers toward the output.
>000595: 
>000596: ### Best predictor of PPL damage (exploratory, 28 layers)
>000597: 
>000598: Correlating single-layer PPL damage against per-layer raw-score shadow metrics:
>000599: 
>000600: | Metric | Pearson | Spearman |
>000601: | --- | ---: | ---: |
>000602: | **top-1 attention agreement** | **−0.42** | **−0.51** |
>000603: | top-5 overlap | −0.42 | −0.40 |
>000604: | MAE | +0.37 | +0.50 |
>000605: | cosine | +0.01 | −0.34 |
>000606: | relative-L2 | −0.01 | +0.17 |
>000607: 
>000608: **Top-1 agreement is the best predictor** (Spearman −0.51): damage tracks how
>000609: often SLHA's *argmax* key differs from baseline's, not the overall score
>000610: correlation. Raw-score cosine (~0.99 everywhere) does **not** predict damage.
>000611: With only 28 layers these are exploratory (moderate |ρ|).
>000612: 
>000613: ### Score-path semantics (audit)
>000614: 
>000615: The op replaces logits after the raw `kq = Q·K` and **before** `soft_max_ext`, so
>000616: for both paths the `1/sqrt(head_dim)` scale, causal mask, RoPE positional
>000617: information, GQA head mapping, and (absent) softcap are **identical**. Qwen2.5 has
>000618: no attention logit softcap; there is no additive positional bias; baseline and
>000619: SLHA use identical active KV lengths (padded positions are exactly zero and
>000620: masked). **Only two things differ:** the score approximation itself, and that the
>000621: *same fixed* `1/sqrt(head_dim)` scale is applied to SLHA scores whose magnitude
>000622: is not calibrated to Q·K. If `E[|slha|] ≠ E[|Q·K|]` the effective softmax
>000623: temperature differs — the leading mechanism hypothesis for the damage, consistent
>000624: with top-1 (argmax) agreement being the best damage predictor.
>000625: 
>000626: ### Limitations
>000627: 
>000628: Single-run screening at a reduced chunk count (not a formal noise floor);
>000629: 28-layer correlations are exploratory. The finer post-softmax / affine /
>000630: per-head / position diagnostics were left to out-of-tree instrumentation
>000631: (archived + hashed in the results JSON); that instrumentation segfaulted on this
>000632: host and did not emit metrics, so the post-softmax analysis rests on the
>000633: committed raw-score shadow metrics (top-1 agreement being the argmax/attention
>000634: selection signal) and the score-path audit rather than on measured softmax
>000635: divergences. Absolute PPLs are not comparable to the 12-chunk PR #58 numbers, but
>000636: within-experiment deltas are. The root cause is not proven: the
>000637: temperature-mismatch hypothesis needs an end-to-end scale experiment, and
>000638: super-additivity means single-layer screening understates joint damage.
>000639: That end-to-end scale experiment was subsequently run and **rejected** the
>000640: temperature hypothesis — see
>000641: [Research: score-temperature (magnitude) calibration](#research-score-temperature-magnitude-calibration)
>000642: below.
>000643: 
>000644: ## Research: score-temperature (magnitude) calibration
>000645: 
>000646: PR #60 left one mechanism hypothesis open: that the direct compressed-score
>000647: degradation is primarily a **layer-dependent score-magnitude mismatch**,
>000648: equivalent to an incorrect softmax temperature. This section tests it end to end
>000649: and **rejects it**. Production score mathematics are unchanged; scaling is a
>000650: strict, default-off (`a = 1.0`) experimental knob.
>000651: 
>000652: Provenance: llama.cpp `fdb1db877c526ec90f668eca1b858da5dba85560` (tag b9860),
>000653: Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 raw test, mixed codec, weights from the
>000654: PR #59 pipeline. Full data, per-layer tables and hashes:
>000655: [`results/score_temperature_calibration.json`](results/score_temperature_calibration.json).
>000656: 
>000657: ### The knob, and exactly what it is
>000658: 
>000659: The op replaces logits after the raw `kq = Q·Kᵀ` (`llama-graph.cpp:2451`) and
>000660: **before** `soft_max_ext` (`:2500`). For Qwen2.5 there is no logit softcap, no
>000661: `kq_b`, no ALiBi and no sinks, and `kq_scale = 1/√128`, so
>000662: 
>000663: ```
>000664: baseline:  P = softmax( kq_scale · (Q·Kᵀ) + mask )
>000665: replace:   P̂ = softmax( kq_scale ·  S_SLHA + mask )
>000666: scaled:    P̃ = softmax( (kq_scale · a_layer) · S_SLHA + mask )
>000667: ```
>000668: 
>000669: `a_layer` multiplies the **effective inverse temperature**, applied exactly once
>000670: to the raw score; the `1/√head_dim` factor is applied afterwards by
>000671: `soft_max_ext` and is never applied twice. Because softmax is shift-invariant per
>000672: row, `a·s = a·(s − s̄_row) + (a−1)·s̄_row` and the row-constant term is discarded —
>000673: so scaling the raw score *is* scaling the row-centred logit, and a global sweep
>000674: probes the softmax-relevant temperature directly.
>000675: 
>000676: ### Determinism had to be fixed first
>000677: 
>000678: The replacement path contained a data race: `dst` was zero-initialised by flat
>000679: element range but written by vector range, and a ggml custom op has no internal
>000680: barrier, so one worker could blank rows another had already filled. Three
>000681: **numerically identical** replacement configurations returned
>000682: 13.4196 / 13.4068 /
>000683: 13.4237 — a spread of 0.0169 PPL.
>000684: After the fix they return bit-identical values. Every measurement below was taken
>000685: with the fixed binary, so differences between configurations are signal, not noise.
>000686: 
>000687: | identity-equivalent configuration | PPL (4 chunks) |
>000688: | --- | ---: |
>000689: | id_replace_noscale | 13.4162 |
>000690: | id_replace_g1p0 | 13.4162 |
>000691: | id_replace_file1p0 | 13.4162 |
>000692: 
>000693: The pass-through custom-op control and the SLHA-off baseline both give
>000694: 9.3852 — the op itself is inert. Unscaled replacement gives
>000695: 13.4162, a gap of **4.0310 (42.95%)**.
>000696: 
>000697: ### Offline fit: the magnitudes are already calibrated
>000698: 
>000699: Shadow mode streams sufficient statistics over causally-unmasked positions
>000700: (`k ≤ t`, clamped to written tiles). The pair count per layer matches the analytic
>000701: prediction Σ(t+1)·heads·chunks **exactly**, proving no vector was skipped and that
>000702: `t` is the true token position.
>000703: 
>000704: | estimator | min | median | max |
>000705: | --- | ---: | ---: | ---: |
>000706: | OLS through origin | 0.9828 | 0.9968 | 1.0036 |
>000707: | robust median-ratio | 0.9886 | 0.9886 | 1.0116 |
>000708: | variance matching | 0.9999 | 1.0097 | 1.0260 |
>000709: | slope with free intercept | 0.9963 | 1.0002 | 1.0079 |
>000710: | Pearson r(b,s) | 0.9795 | 0.9908 | 1.0000 |
>000711: 
>000712: Fitting with a free intercept — the softmax-relevant form, since a constant
>000713: per-row offset cancels — puts every layer within
>000714: **0.79%** of identity, tighter than the
>000715: through-origin fit. So the near-unit scale is not an artifact of forcing the fit
>000716: through the origin. Applying the best-fit scale removes only
>000717: **0.38%** of the score's squared error
>000718: (pair-weighted; 1.00% equally
>000719: weighted across layers, at most 5.88% for
>000720: any single layer).
>000721: 
>000722: The fit is not dominated by a degenerate subpopulation: each of the 12 heads
>000723: carries exactly 1/12 of the samples with per-head scales inside a ~3% band,
>000724: position-bucket counts follow the causal prediction with per-bucket scales ≈ 1.0,
>000725: near-zero scores carry a vanishing share of the OLS denominator, and there are no
>000726: non-finite pairs. Two caveats are recorded rather than glossed: a **pooled**
>000727: cross-layer OLS is meaningless here because layer 0 alone carries
>000728: 99.91% of Σs², and the robust estimator is quantised to
>000729: its 0.01-dex histogram bin (≈2.3%), so its digits beyond ~1% are not meaningful.
>000730: 
>000731: **Per-layer structure is not resolved.** The split-half disagreement (max 0.0256) is AS LARGE AS or LARGER than the entire per-layer spread of the fitted scale (0.0208), so the apparent per-layer structure is NOT resolved: the per-layer scales are consistent with all layers sharing a scale of ~1.0 plus estimation noise. This makes the fitted per-layer scale files perturbations of the identity rather than a genuine search over per-layer temperatures, and it is why they are reported alongside, not instead of, the direct global sweep. Rank correlation between the two disjoint halves is 0.354.
>000732: 
>000733: ### End-to-end: no temperature recovers the gap
>000734: 
>000735: Every measured point, at 4 chunks. `recovered_gap = (unscaled − config) / (unscaled − pass-through)`.
>000736: 
>000737: | global scale a | PPL | recovered gap |
>000738: | ---: | ---: | ---: |
>000739: | 0.40 | 269.9862 | -6364.92% |
>000740: | 0.50 | 69.4502 | -1390.08% |
>000741: | 0.60 | 26.8259 | -332.66% |
>000742: | 0.70 | 17.6512 | -105.06% |
>000743: | 0.80 | 14.6611 | -30.88% |
>000744: | 0.90 | 13.8585 | -10.97% |
>000745: | 0.92 | 13.7724 | -8.84% |
>000746: | 0.94 | 13.7056 | -7.18% |
>000747: | 0.96 | 13.5451 | -3.20% |
>000748: | 0.98 | 13.4530 | -0.91% |
>000749: | 0.99 | 13.5311 | -2.85% |
>000750: | 1.00 | 13.4162 | 0.00% |
>000751: | 1.01 | 13.6068 | -4.73% |
>000752: | 1.02 | 13.4988 | -2.05% |
>000753: | 1.04 | 13.5626 | -3.63% |
>000754: | 1.06 | 13.4889 | -1.80% |
>000755: | 1.08 | 13.4590 | -1.06% |
>000756: | 1.10 | 13.5629 | -3.64% |
>000757: | 1.20 | 13.8681 | -11.21% |
>000758: | 1.30 | 14.0534 | -15.81% |
>000759: | 1.50 | 15.2022 | -44.31% |
>000760: | 1.75 | 17.2626 | -95.42% |
>000761: | 2.00 | 19.8388 | -159.33% |
>000762: 
>000763: A clean U-curve with its minimum at **a ≈ 1.0**. Both directions are worse:
>000764: sharpening degrades steadily, flattening degrades catastrophically (a = 0.40 →
>000765: 269.99),
>000766: which is what rank-preservation predicts: flattening destroys the selectivity the
>000767: model depends on while buying nothing back.
>000768: 
>000769: **Resolution floor.** The binary is deterministic, yet PPL(a) is not smooth at
>000770: fine scale: a least-squares quadratic through the local window
>000771: [0.90, 1.10] leaves a residual roughness of
>000772: **0.0584 PPL RMS** (max 0.1143). That is not
>000773: measurement noise — identical configurations give bit-identical results — but genuine
>000774: chaotic sensitivity of the forward pass to tiny attention perturbations. It sets the
>000775: resolution for any recovered-gap claim at about
>000776: **1.45%** of the gap. The smooth
>000777: component of the curve has its minimum at a = 1.037 with curvature
>000778: d²PPL/da² ≈ 40.4; no measured point recovers any of the gap.
>000779: 
>000780: | fitted-scale strategy | PPL | recovered gap | manifest |
>000781: | --- | ---: | ---: | --- |
>000782: | `sc_robust_0_6` | 13.4196 | -0.08% | 26239a1bcb0c |
>000783: | `sc_robust_all` | 13.4434 | -0.67% | 12799dc02ca2 |
>000784: | `sc_global_robust` | 13.4766 | -1.50% | 5858defc1dfa |
>000785: | `sc_ols_0_13` | 13.4821 | -1.63% | 34c2920dc7f3 |
>000786: | `sc_ols_all` | 13.5006 | -2.09% | 820c9a847428 |
>000787: | `sc_ols_0_20` | 13.5125 | -2.39% | f910a38c8a09 |
>000788: | `sc_ols_0_6` | 13.5192 | -2.56% | 6a95ebf3689e |
>000789: | `sc_var_all` | 13.5216 | -2.61% | ae9af62c6611 |
>000790: | `sc_global_ols` | 13.5497 | -3.31% | b4c6fa1ba027 |
>000791: 
>000792: ### Twelve-chunk validation (3 repetitions each)
>000793: 
>000794: | configuration | mean PPL | sample stdev | spread | recovered gap | reps |
>000795: | --- | ---: | ---: | ---: | ---: | ---: |
>000796: | `v_passthrough` | 11.8644 | 0.0000 | 0.0000 | — | 3 |
>000797: | `v_replace_noscale` | 16.8855 | 0.0000 | 0.0000 | 0.00% | 3 |
>000798: | `v_replace_g1p0` | 16.8855 | 0.0000 | 0.0000 | 0.00% | 3 |
>000799: | `v_best_global` | 16.8855 | 0.0000 | 0.0000 | 0.00% | 3 |
>000800: | `v_best_perlayer` | 16.9073 | 0.0000 | 0.0000 | -0.43% | 3 |
>000801: | `v_best_early` | 16.9138 | 0.0000 | 0.0000 | -0.56% | 3 |
>000802: 
>000803: ### Why this excludes the whole family, not just the points tested
>000804: 
>000805: Softmax normalises over one `(layer, head, query)` row, so a per-layer, per-head,
>000806: per-query-row or context-length-dependent positive scale is **constant within the
>000807: normalisation group**. Every such scale is exactly rank-preserving, and the global
>000808: sweep measures the family's mean directly. The residual bounds its spread:
>000809: attributing **100%** of each layer's score residual to per-row gain jitter — the
>000810: most generous possible magnitude hypothesis — gives σ_a ≈ 0.070, which
>000811: priced against the measured sweep curvature
>000812: (d²PPL/da² ≈ 40.4) costs only
>000813: ≈ 0.100 PPL, about
>000814: 2.5% of the gap.
>000815: Explaining the whole gap would need a jitter several times larger than the one
>000816: actually measured. The scale family is excluded numerically, not merely unsampled.
>000817: 
>000818: ### Conclusion
>000819: 
>000820: Per-layer and global multiplicative calibration recovered less than 1% of the PPL gap, while fitted scales remained close to identity and scaling removed less than approximately 1% of raw-score error. No positive rescaling that is constant within a softmax row -- per-layer, per-head, per-query-row, or context-length-dependent -- can close the gap: every such rescaling is exactly rank-preserving, the measured optimum of the global sweep sits at a ~ 1.0, and even attributing 100% of the score residual to per-row gain jitter prices that whole family at only a few percent of the gap. The quality gap is therefore dominated by score distortion that is NOT constant within a softmax row. This experiment does not further decompose that residual into reordering versus order-preserving gap error.
>000821: 
>000822: Twelve-chunk validation repeats were **bit-identical** (sample stdev 0.0000 across
>000823: three repetitions of every configuration), and determinism was confirmed on the
>000824: genuinely scaled write path as well, not only on the `a = 1.0` path that skips the
>000825: scaling loop.
>000826: 
>000827: What this does **not** establish: it does not decompose the residual into
>000828: reordering versus order-preserving gap error. A monotone nonlinear magnitude map
>000829: is non-scalar, order-preserving, changes the softmax distribution, and is
>000830: invisible to every statistic computed here; a per-**key** magnitude error is
>000831: likewise not row-constant and therefore does change ranking. The `top1`/`top5`
>000832: figures quoted in PR #60 and reproduced by the shadow metrics are computed over
>000833: the full `n_kv` row including causally-masked keys, so they are not clean
>000834: attention-relevant rank-agreement numbers and are not used as load-bearing
>000835: evidence here.
>000836: 
>000837: ### Recommended next experiments
>000838: 
>000839: After a negative scaling result the next diagnostic should target the score
>000840: *ordering* and per-key magnitude, not another temperature sweep:
>000841: - joint Q/K projection training (optimize the compressed projection for score ordering)
>000842: - pairwise ranking loss on score pairs rather than L2 on score values
>000843: - top-k preservation loss during projection training
>000844: - per-head projection calibration
>000845: - residual correction of SLHA scores (learned low-rank correction term)
>000846: - hybrid exact top-k plus compressed tail scoring
>000847: 
>000848: None of these are implemented in this PR.
>000849: 
>000850: ### Experimental interface
>000851: 
>000852: `SLHA_SCORE_SCALE` (`"0.75"` or `"layer:0=0.91,5=0.72"`) and
>000853: `SLHA_SCORE_SCALE_FILE` (JSON `{"global":…,"layers":{…}}`) set the per-layer
>000854: scale; default `1.0` (no-op). Strict and fail-closed: only finite strictly
>000855: positive scales; zero, negative, NaN, ±Inf, malformed, duplicate/out-of-range
>000856: layer ids, and — in per-layer mode — any selected layer lacking a scale are
>000857: rejected and mark the run invalid. `SLHA_SCORE_SCALE_SUMMARY` reports the
>000858: requested and resolved scales, a manifest SHA-256, `scaled_vectors`/
>000859: `scaled_logits`, `invalid_scale` and `scale_manifest_valid`.
>000860: `SLHA_SCALE_FIT_JSON=<path>` (shadow mode) writes the offline per-layer fit.
>000861: 
>000862: ### Limitations
>000863: 
>000864: Screening ran at 4 chunks because this host is materially slower than the PR #58
>000865: host; every control used the same chunk count, so within-experiment deltas are
>000866: consistent, but absolute PPLs are not comparable to the 12-chunk PR #58 numbers.
>000867: The offline fit is off-policy — it estimates scales on baseline-conditioned
>000868: states with no cross-layer error compounding — so only the end-to-end sweep is
>000869: on-policy; that is why the sweep, not the fit, carries the conclusion. The
>000870: per-layer scale files tested are magnitude-fitted perturbations of the identity
>000871: (all within ~3%), not a PPL-optimised search of the 28-dimensional scale space.
>000872: 
>000873: ## Research: rank-transplant oracle (ranking vs order-preserving error)
>000874: 
>000875: PR #62 established that no rescaling constant within a softmax row explains the
>000876: compressed-score quality gap. That left two mechanisms unseparated: **(A) key-ranking
>000877: errors** — SLHA puts the wrong keys at the top — and **(B) order-preserving errors** in
>000878: the score gaps and shape. This section separates them with a diagnostic transplant
>000879: oracle. It is measurement only: production score mathematics are unchanged and the
>000880: oracle is default-off.
>000881: 
>000882: Provenance: source content proven byte-identical to squash merge `e8db022`,
>000883: llama-perplexity `43dc88a072b590d6…`, libllama `ea4979aa14ff6c68…`,
>000884: model `d7efb072e7724d25…`, dataset `136677b69515d194…`,
>000885: weight manifest `afe6deb0cf986015…`. Full data, per-layer tables and
>000886: hashes: [`results/rank_transplant_oracle.json`](results/rank_transplant_oracle.json).
>000887: 
>000888: > **These oracles require access to exact baseline scores and are not deployable
>000889: > replacement methods.** Every one of them reads the baseline `Q·Kᵀ` row that a compressed
>000890: > KV cache exists precisely to avoid computing. They bound what a better codec could
>000891: > achieve; they are not one. The oracle is default-off and no production path changes.
>000892: 
>000893: ### The oracles
>000894: 
>000895: At the raw `kq = Q·Kᵀ` seam, for query token `t`, the oracle rewrites the row from the
>000896: paired baseline row `B` and SLHA row `S`:
>000897: 
>000898: ```
>000899: Oracle A   baseline ranking  +  the visible SLHA value multiset
>000900: Oracle B   SLHA ranking      +  the visible baseline value multiset
>000901: top-k      baseline's top-k keys promoted; the tail keeps its relative SLHA order
>000902: ```
>000903: 
>000904: **Causal visibility is part of the definition.** Query `t` attends keys `0..t`, so the
>000905: oracle domain is `n_visible = min(n_check, t+1)`. An earlier version spanned the whole
>000906: written prefix; masked keys then consumed value-multiset slots that softmax never sees,
>000907: which diluted the transplant. At the four-chunk screening resolution, fixing it moved
>000908: Oracle A from 10.4088 to
>000909: 10.9429 PPL and the recovered fraction from
>000910: 74.6% to
>000911: 61.4% — a material change, not a rounding
>000912: difference. Every pre-fix oracle run was quarantined, not reinterpreted.
>000913: 
>000914: Values are gathered through the *other* vector's own ranking permutation, so the
>000915: transplanted multiset is bit-exact. The ordering invariant is **weak-order** preserving:
>000916: the output must be non-increasing along the reference permutation. A strict
>000917: permutation-equality check is wrong whenever the transplanted values tie —
>000918: `B = [1.0, 9.0]`,
>000919: `S = [5.0, 5.0]` gives
>000920: `out = [5.0, 5.0]`, whose rank vector
>000921: `[0, 1]` cannot equal the reference
>000922: permutation `[1, 0]` because the two values are equal.
>000923: 
>000924: ### Identity controls
>000925: 
>000926: Nothing below is interpreted unless both identity controls are exact at twelve chunks.
>000927: 
>000928: | control | PPL (12 chunks, mean of 3) | must equal |
>000929: | --- | ---: | --- |
>000930: | pass-through | 11.8644 | — |
>000931: | baseline-identity oracle | 11.8644 | pass-through |
>000932: | strict replacement | 16.8855 | — |
>000933: | SLHA-identity oracle | 16.8855 | strict replacement |
>000934: 
>000935: Both hold exactly (`controls_exact = true`). Three repetitions of each of the
>000936: nine configurations are **bit-identical**, sample standard deviation exactly zero, with
>000937: identical strict, oracle, exact-tie, invariant-check and active-domain accounting
>000938: counters (`all_deterministic = true`). Any difference would be a regression after the
>000939: PR #62 race fix and would block aggregation.
>000940: 
>000941: ### Result
>000942: 
>000943: Total gap at twelve chunks: 16.8855 − 11.8644 = **5.0211 PPL**.
>000944: 
>000945: | oracle | PPL | change vs strict | fraction of gap |
>000946: | --- | ---: | ---: | ---: |
>000947: | Oracle A — baseline ranking, visible SLHA values | 13.5756 | +3.3099 | 65.92% |
>000948: | Oracle B — SLHA ranking, visible baseline values | 19.6719 | -2.7864 | -55.49% |
>000949: | order-preserving residual (Oracle A − pass-through) | — | +1.7112 | 34.08% |
>000950: 
>000951: Restoring the baseline ordering of causally visible keys while preserving the causally
>000952: visible SLHA score-value multiset recovers 65.9% of the gap. The remaining
>000953: 1.7112 PPL is the **order-preserving residual** — error that survives with
>000954: the ranking already correct. It is *not* labelled magnitude error: PR #62 rejected the
>000955: magnitude hypothesis, and this experiment does not identify what the residual is.
>000956: 
>000957: Oracle B is reported independently and read conservatively: applying the baseline value
>000958: distribution to the incorrect SLHA key ordering **amplifies** the quality loss. This
>000959: demonstrates a strong interaction between key assignment and score geometry; it does not
>000960: prove that score geometry is irrelevant. **Oracle A and Oracle B are not an additive
>000961: decomposition** — their fractions do not sum to 1 and must not be read as a variance
>000962: partition.
>000963: 
>000964: ### How many keys have to be right
>000965: 
>000966: | restoration | PPL | absolute recovery | fraction of gap | fraction of full Oracle A | remaining residual |
>000967: | --- | ---: | ---: | ---: | ---: | ---: |
>000968: | top-1 | 14.2400 | +2.6455 | 52.69% | 79.93% | 2.3756 |
>000969: | top-16 | 13.6278 | +3.2577 | 64.88% | 98.42% | 1.7634 |
>000970: | full ranking (Oracle A) | 13.5756 | +3.3099 | 65.92% | 100.00% | 1.7112 |
>000971: 
>000972: - CONFIRMED: top-1 restoration alone reaches 79.93% of the full Oracle A recovery.
>000973: - CONFIRMED as approximate saturation: top-16 reaches 98.42% of the full Oracle A recovery.
>000974: 
>000975: The SIGN of the top-16 minus full-ranking difference reversed between the two resolutions. At four chunks top-16 was 0.0113 PPL BELOW full ranking; at twelve chunks it is 0.0522 PPL ABOVE it. The screening appearance that top-16 slightly exceeded full ranking therefore does NOT persist: at the validated resolution top-16 recovers slightly less than the full ranking, as expected. Both differences are small in absolute terms, but the reversal is why the four-chunk contrast was never promoted to a claim.
>000976: 
>000977: ### Early layers
>000978: 
>000979: | restoration | PPL | recovery | fraction of gap |
>000980: | --- | ---: | ---: | ---: |
>000981: | full-rank Oracle A, layers 0–6 | 12.4770 | +4.4085 | 87.80% |
>000982: | full-rank Oracle A, all 28 layers | 13.5756 | +3.3099 | 65.92% |
>000983: 
>000984: Correcting rankings only in early layers can outperform correcting every layer because some later-layer SLHA deviations may be compensating, while early ranking errors trigger downstream amplification. This does not mean later layers are irrelevant.
>000985: 
>000986: ### Screening versus final
>000987: 
>000988: The twelve-chunk matrix is authoritative. The four-chunk sweep is screening evidence only.
>000989: 
>000990: | metric (recovered fraction of gap) | four-chunk screening | twelve-chunk final | absolute difference | direction preserved |
>000991: | --- | ---: | ---: | ---: | :---: |
>000992: | Oracle A recovery | 61.36% | 65.92% | +4.56 pp | yes |
>000993: | Oracle B change | -37.26% | -55.49% | -18.23 pp | yes |
>000994: | top-1 recovery | 46.98% | 52.69% | +5.71 pp | yes |
>000995: | top-16 recovery | 61.64% | 64.88% | +3.24 pp | yes |
>000996: | early 0-6 recovery | 88.55% | 87.80% | -0.75 pp | yes |
>000997: 
>000998: Every direction is preserved (`all_directions_preserved = true`). The
>000999: magnitudes move by a few percentage points, which is why the four-chunk figures were never
>001000: promoted to validated claims.
>001001: 
>001002: ### Layerwise screening, including the anomalies
>001003: 
>001004: The 28 matched pairs (`U_layer_L` strict vs `L_oracleA_L` oracle, identical conditions)
>001005: are a **four-chunk screening** result; no equivalent twelve-chunk layerwise decomposition
>001006: was run. Classification precedence, applied in this order:
>001007: 
>001008: 1. `unstable_denominator` — |damage| < eps -- checked first, because every fraction below is undefined when the denominator is not resolvable
>001009: 1. `negative_damage` — damage <= -eps -- replacing this layer alone IMPROVES PPL, so a 'recovered fraction' has no meaning regardless of the oracle change
>001010: 1. `over_recovery` — damage >= eps and oracle_change > damage (fraction > 1) -- the oracle drives PPL below pass-through; the fraction is reported uncapped
>001011: 1. `negative_recovery` — damage >= eps and oracle_change < 0 -- the transplant makes this layer worse; the negative fraction is reported signed, never clipped
>001012: 1. `positive_recovery` — everything else
>001013: 
>001014: With a declared denominator guard of ε = 0.01 PPL:
>001015: 
>001016: - **negative damage** (replacing the layer alone *improves* PPL, so a fraction is meaningless): layers 3, 25
>001017: - **unstable denominator** (|damage| < ε): layer 10
>001018: - **positive damage but negative recovery** (the transplant makes the layer worse; the signed fraction is reported, never clipped): layers 1, 2, 8, 9, 15, 18, 22
>001019: - **over-recovery** (fraction > 1): none — no screening layer recovers more than 100%
>001020: 
>001021: No fraction is capped anywhere in the artifact. Layer 5 carries the largest single-layer
>001022: damage (0.5578 PPL) and the largest absolute recovery
>001023: (+0.4220 PPL), but its recovery does not exceed its damage: the
>001024: fraction is 0.7565 and the oracle result 9.5210 stays above
>001025: pass-through 9.3852. The largest fraction observed is
>001026: 0.9907 at layer 27.
>001027: 
>001028: Summing the **signed** single-layer absolute recoveries over layers 0–6 — including the
>001029: negative terms, none excluded — gives 0.3811 PPL, while restoring the group
>001030: jointly recovers 3.5693 PPL: an interaction of
>001031: **+3.1882 PPL**. Early ranking errors amplify downstream. This is a
>001032: screening figure and is labelled as such in the artifact.
>001033: 
>001034: ### Active-key rank agreement
>001035: 
>001036: All statistics are restricted to **causally visible** keys — written tile, in-stream,
>001037: finite, unmasked. They supersede the PR #60 top-1/top-5 figures, which were computed over
>001038: the full padded row and are not attention-relevant. Micro (pooled over rows) and macro
>001039: (equal weight per layer) are reported separately and never averaged together.
>001040: 
>001041: | statistic | micro | macro mean | macro median | macro min | macro max |
>001042: | --- | ---: | ---: | ---: | ---: | ---: |
>001043: | top-1 agreement | 0.7389 | 0.7389 | 0.8159 | 0.2093 | 1.0000 |
>001044: | top-2 overlap | 0.7548 | 0.7548 | 0.8004 | 0.4597 | 0.9173 |
>001045: | top-4 overlap | 0.8055 | 0.8055 | 0.8054 | 0.6361 | 0.9435 |
>001046: | top-8 overlap | 0.8384 | 0.8384 | 0.8465 | 0.7182 | 0.9506 |
>001047: | top-16 overlap | 0.8627 | 0.8627 | 0.8703 | 0.7704 | 0.9619 |
>001048: | Spearman (fractional ranks) | 0.9640 | 0.9640 | 0.9644 | 0.9267 | 0.9934 |
>001049: | Kendall tau-b | 0.8529 | 0.8529 | 0.8484 | 0.7807 | 0.9383 |
>001050: 
>001051: **Tie counts are sampled diagnostics, not exhaustive.** The metrics hook records a row only
>001052: when `(t % 16) == 0 && h < 2`, so 7,224 rows were inspected
>001053: (258 per layer across 28 layers) — 1.049% of the
>001054: 688,800 replaced vectors in a full-model run. On that sample:
>001055: 96 baseline exact-tie pairs, 70 SLHA exact-tie pairs,
>001056: 280 undefined rows and top-k boundary ties
>001057: top1=0, top2=0, top4=0, top8=0, top16=1 — a boundary tie means the
>001058: k-th and (k+1)-th baseline scores are exactly equal, so top-k membership is decided by the
>001059: deterministic index tiebreak rather than by the score. These are **not** comparable with the exhaustive oracle
>001060: counter `oracle_ties = 12511`, which counts tied comparisons over every replaced vector
>001061: during permutation construction — a different population and a different definition. Do
>001062: not divide one by the other.
>001063: 
>001064: Position accounting holds on every sampled row (`physical = included + causally_masked +
>001065: padding + inactive_stream + nonfinite`), with 0 failures.
>001066: 
>001067: ### What this does and does not establish
>001068: 
>001069: Restoring the baseline ordering of causally visible keys while preserving the causally visible SLHA score-value multiset recovered most of the measured quality gap. Correcting only a small number of highest-ranked keys captured a large fraction of that benefit, and correcting layers 0-6 prevented substantial downstream amplification.
>001070: 
>001071: The dominant measured mechanism is therefore incorrect assignment of high score values to visible keys. A substantial order-preserving residual remains, and Oracle B demonstrates that key assignment and score geometry interact strongly.
>001072: 
>001073: It does **not** establish that:
>001074: 
>001075: - ranking explains all degradation
>001076: - the residual is purely magnitude error
>001077: - later layers do not matter
>001078: - Oracle A and Oracle B partition the gap additively
>001079: 
>001080: These oracles require access to exact baseline scores and are not deployable replacement methods. Every oracle in this experiment reads the baseline Q·K row that a compressed KV cache exists precisely to avoid computing; the measurements bound what a better codec could achieve, they do not constitute one.
>001081: 
>001082: ### Limitations
>001083: 
>001084: - one model (Qwen2.5-1.5B-Instruct Q8_0), one corpus (WikiText-2 raw test), one codec configuration; nothing here is shown to generalise
>001085: - twelve chunks of 512 tokens is 6144 tokens -- enough to separate these effects, far short of a full benchmark
>001086: - the layerwise decomposition and the early-layer interaction are FOUR-CHUNK screening results; no twelve-chunk layerwise matrix was run
>001087: - the active-key statistics come from a deterministic 1-in-16 token, 2-of-12 head sample, not an exhaustive pass
>001088: - perplexity is the only quality metric; no downstream task was measured
>001089: - the order-preserving residual is quantified but not explained -- this experiment does not identify what it consists of
>001090: - two process-tree terminations remain unexplained; they were reconciled by re-running, not by root-cause analysis
>001091: 
>001092: ### Recommended next experiment
>001093: 
>001094: Constrain the codec so that the top-k baseline ordering is preserved by construction and measure how much of the 65.92% ranking recovery survives without any baseline-score access. The top-1 result (79.93% of the full ranking benefit from a single key) and the layers 0-6 result (87.80% of the gap from seven layers) together suggest the cheapest viable target: an early-layer, top-k-preserving score path. A parallel line should attack the order-preserving residual directly, since PR #62 already ruled out a per-row rescaling constant and this experiment shows the residual is 34.08% of the gap even with the ranking exactly correct.
>001095: 
>001096: ### Methodological corrections
>001097: 
>001098: Every defect found during this experiment is recorded in the artifact under
>001099: `provenance.experimental_provenance_chronological`. The two that changed a measurement or
>001100: a decision:
>001101: 
>001102: 1. **Incorrect strict-permutation invariant.** The ordering check required exact
>001103:    permutation equality, which is only valid when no two transplanted values tie. Corrected
>001104:    to a weak-order check. Observational: the quarantined runs reproduced identical PPL.
>001105: 2. **Non-causal oracle span.** The oracle spanned the whole written prefix instead of the
>001106:    causally visible one. Material: it inflated the apparent ranking contribution. All
>001107:    affected runs were quarantined and re-run, never reinterpreted.
>001108: 
>001109: Also recorded there: the four pre-lineage twelve-chunk attempts that were quarantined and
>001110: re-run once the harness recorded the full lineage block; the `running`-state PID recording
>001111: fix; and the two unexplained process-tree terminations.
>001112: 
>001113: ### Resolution
>001114: 
>001115: No single scalar "experimental resolution" is claimed. The relevant quantities have
>001116: different sources and are reported separately:
>001117: 
>001118: - deterministic PPL printing precision: 0.0001 (llama-perplexity prints the final estimate to four decimal places)
>001119: - twelve-chunk repetition spread: 0.0000 PPL maximum across all nine configurations (all bit-identical)
>001120: - screening chunk-count limitation: 4 chunks × 512 tokens = 2048 tokens evaluated
>001121: - smallest screening contrast treated as meaningful: 0.0108 PPL (layer 27 single-layer damage, the smallest damage still classified stable)
>001122: - The top-16 and full-ranking screening values differ by 0.0113 PPL, which is small relative to the coarse four-chunk screening protocol. The twelve-chunk validation is used to determine whether this difference persists. It did not persist with the same sign: at twelve chunks top-16 sits 0.0522 PPL ABOVE full ranking rather than below it. Top-16 still reaches 98.42% of the full Oracle A recovery, so approximate saturation holds, but the four-chunk contrast was too fine for its sign to be meaningful.
>001123: 
>001124: ### Execution integrity
>001125: 
>001126: Two completeness gates run independently; neither implies the other.
>001127: 
>001128: - screening: 75/75 accepted and revalidated, two agreeing reads under proven quiescence, report `526f6f5da91352f5…`
>001129: - twelve-chunk: 27/27 accepted and revalidated, two agreeing reads, report `5979a43b5253f723…`, bound by sha256 to the frozen screening report
>001130: 
>001131: Acceptance is validated from the run itself, never inferred from a results row. Each
>001132: attempt has an immutable log and parsed summary, an explicit child wait, exit and signal
>001133: status, and an atomically written state record. Twelve-chunk states additionally require
>001134: a complete lineage block and the exact declared run geometry. Resume revalidates every
>001135: accepted state before skipping it. Superseded and interrupted attempts are preserved as
>001136: provenance and never counted toward completeness.
>001137: 
>001138: On 2 occasions the process tree was unexpectedly terminated. Available evidence
>001139: did not identify an OOM, disk-capacity failure or application crash
>001140: (15,209 MB of 16,075 MB free; disk 35% used; no OOM record; no crash trace), so the cause is recorded as **unknown**. Both
>001141: were reconciled by preserving the interrupted attempt as provenance and launching a new
>001142: immutable attempt.
>001143: 
>001144: ## Earlier round-trip results
>001145: 
>001146: Recorded under the earlier protocol; see git history and
>001147: [`results/measurements.json`](results/measurements.json) for details.
>001148: 
>001149: | Mode        |      PPL | ΔPPL absolute | ΔPPL relative | Notes           |
>001150: | ----------- | -------: | ------------: | ------------: | --------------- |
>001151: | baseline    | 11.8753  |             — |             — | original        |
>001152: | passthrough | 11.8753  |          0.00 |          0.0% | hook sanity     |
>001153: | mixed       | 16.5976  |          4.72 |         39.8% | SLHA round-trip |
>001154: | mix3        | 16.6460  |          4.77 |         40.2% | SLHA round-trip |
>001155: 
>001156: The round-trip callback reconstructs K from the SLHA tile using the latent plus
>001157: a linear estimate of the sign-LSH residual. The 256-bit residual sketch is a
>001158: *score-side* correction rather than a faithful inverse of the quantization
>001159: error, and the round-trip perplexity reflects that limitation.
>001160: =======
>001161: **Round-trip:** the callback reconstructs K from the SLHA tile using the latent
>001162: plus a linear estimate of the sign-LSH residual. While the latent captures the
>001163: principal subspace of the calibration K activations, the 256-bit residual
>001164: sketch is a *score-side* correction (it preserves attention dot products well
>001165: in offline score tests) rather than a faithful inverse of the quantization
>001166: error. Restoring the exact per-vector K value from sign bits alone is not what
>001167: the residual was designed for, and the perplexity measurement reflects that
>001168: limitation.
>001169: 
>001170: **Fused-QK:** the SLHA scores are computed directly on the tiles — no K
>001171: reconstruction — yet they still fail to replace QK^T. The score diagnostics
>001172: show why: the per-layer projections are trained on K alone (`fit_with`, the
>001173: collection seam only exposes K), so the SLHA scores are correlated with the
>001174: true QK^T (cos ≈ 0.68) but with a very different *softmax distribution* (KL ≈
>001175: 5 on deep layers). Replacing the scores shifts attention mass to the wrong
>001176: tokens, which the perplexity measurement makes catastrophic. Closing the gap
>001177: would require joint K/Q calibration (collect Q at the same seam), score-scale
>001178: calibration, or a larger latent/residual budget — none validated here.
>001179: >>>>>>> refs/remotes/origin/pr71
 001180: 
 001181: ## Limitations and known issues
 001182: 
 001183: * This milestone evaluates **score quality only**. No physical K-cache memory

### conflict lines 1204-1213
 001200:   confidence intervals.
 001201: * The measured quality gap is specific to this configuration and its precise
 001202:   cause has not been isolated.
 001203: * Per-layer projections are trained on K only (`fit_with`), not joint K/Q,
>001204: <<<<<<< HEAD
>001205:   because the collection seam currently only exposes K.
>001206: =======
>001207:   because the collection seam currently only exposes K — this is the most
>001208:   likely cause of the fused-QK NO-GO.
>001209: * The round-trip reconstruction uses a spectral residual estimate; it improves
>001210:   over latent-only reconstruction but is not an exact inverse.
>001211: * The fused and scorediag nodes are CPU GGML callbacks (one thread per layer);
>001212:   they run on the standard non-flash attention path only (`-fa off`).
>001213: >>>>>>> refs/remotes/origin/pr71
 001214: * AddressSanitizer builds successfully but the runtime shadow memory could not
 001215:   be allocated in the current container (`ulimit -v` insufficient). No memory
 001216:   errors were observed in normal runs.
 001217: * Only CPU inference is exercised here; GPU is out of scope.
