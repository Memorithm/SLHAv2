# Elastic Doctrine

**Elasticity is a first-class architectural property.**

`ElasticXxx` means that `Xxx` must not be treated as a statically provisioned
resource when its optimal capacity, representation, placement or execution
strategy changes with workload and available resources.

Elastic components remove arbitrary implementation limits while respecting
genuine mathematical, model, protocol, operating-system and hardware
constraints.

Adaptation must be:
- **automatic** — the system adapts without operator intervention;
- **observable** — every decision is explainable and recorded;
- **stable** — hysteresis and anti-thrash prevent oscillation;
- **correctness-preserving** — hard constraints outrank utility;
- **reversible where possible** — demotions can be promoted back;
- **transactional when state can be corrupted** — prepare/validate/commit/
  rollback;
- **predictive where useful** — forecast before exhaustion, never wait for
  the OOM.

## The context principle

> The context size is not a configuration constant. It is a runtime resource
> managed by ElasticContext. Logical context, physical KV residency and
> SLHA-compressed residency are separate concepts. The system must
> continuously adapt the physical representation to workload and available
> resources while preserving model correctness and respecting the model's
> true positional limits.

## Not everything should become elastic

An Elastic resource is justified when:
1. its useful operating point varies over time/workload/hardware;
2. runtime observations can inform better decisions;
3. adaptation has measurable benefit;
4. adaptation cost is lower than expected benefit.

Do **not** "elasticize":
- mathematical constants;
- file-format invariants;
- ABI invariants;
- protocol constants;
- true model limits;
- true hardware limits.

## Limit classification

Every important fixed constant in the codebase is classified as one of:

| Class | Meaning | Example |
|---|---|---|
| `AlgorithmInvariant` | fixed by the algorithm/math | `D_C = 128`, tile = 128 B |
| `FormatInvariant` | fixed by a persisted format | tile field offsets |
| `AbiInvariant` | fixed by the C ABI | `SciRustSlhaTile` layout |
| `ModelLimit` | true model positional limit | RoPE max context |
| `HardwareLimit` | true device limit | VRAM size |
| `SafetyLimit` | safety-mandated | hard budget floors |
| `PolicyDefault` | tunable policy default | watermarks 0.85/0.70 |
| `ElasticTarget` | owned by the ECA | context residency |
| `LegacyArbitraryLimit` | historical arbitrary cap | `SLHA_MAX_LAYERS 128` |

`LegacyArbitraryLimit` values should normally be **removed**.

## Dependency direction

```text
elastic-core
     ↑
elastic-runtime
     ↑
elastic (facade)
     ↑
SLHAv2 controllers (ElasticContext, ElasticKvCache)
     ↑
CPU / CUDA / llama.cpp backends
     ↑
CCOS / product integrations
```

The Elastic language never depends on SLHAv2, CCOS, llama.cpp, CUDA or any
application.
