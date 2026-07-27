# SciRust — noyau de référence SLHA v2

Implémentation de référence du mécanisme d'attention **SLHA v2** (Sub-Low Rank
Hybrid Attention) décrit dans [`../SLHAv2.md`](../SLHAv2.md).

## Idée

- Base latente **bas-rang** (`d_c = 128`) stockée en **INT4 signé par groupe** (MX, 64 o + 8 octets d'échelles).
- Résidu de correction **1-bit** via sign-LSH Johnson–Lindenstrauss
  (`d_s = 256` bits, 32 o).
- Score fusionné continu + binaire (`popcount`), eq. (2.3) du paper.
- Tuile alignée cache : **128 octets exacts, zéro padding** (garanti par test).

## Build / test / mesure

```sh
cargo test -p scirust --all-features            # suite noyau : unitaires + intégration + property/fuzz + doctests
                                                 #  (Hamming, layout 128 o, zero-point, WARM, sign-LSH, Jacobi,
                                                 #   PCA, MX, NF4, sortie d'attention, SGD, SIMD≡scalaire, calibration λ,
                                                 #   CCOS Soft-Paging : page_out/evict/budget/recyclage de slots ;
                                                 #   property : fuzz SIMD≡scalaire, finitude, softmax, bornes dequant,
                                                 #   déterminisme, complément de signe, borne du résidu, codebook NF4)
cargo bench                                      # micro-benchs criterion (scalaire / AVX2 / AVX-512)
cargo run --example measure --release            # rho fixé : fidélité, HOT vs WARM, débit scalaire/AVX2/AVX-512
cargo run --example measure_learned --release    # base apprise par PCA + codecs INT4 (MX) / NF4 + réf INT8
cargo run --example bench_vs_fp16 --release       # SLHA 128 o vs clé bf16 256 o : débit & trafic mémoire
cargo run --example benchmark_decode --release    # PROXY decode bout-en-bout : tok/s SLHA vs réf f32 (multi-têtes)
cargo run --example attention_fidelity --release  # fidélité de la sortie softmax·V (proxy perplexité)
cargo run --example learn_projection --release    # projection apprise (task-aware) vs PCA
cargo run --example calibrate_lambda --release    # calibration de λ (ΔP) vs référence FP
cargo run --example cycles --release              # cycles/tuile (rdtsc) : scalaire/AVX2/AVX-512
cargo run --example ccos_softpaging --release     # cache KV élastique : Soft-Paging HOT/WARM/COLD sous budget
cargo run --example platform_report --release     # kit multi-plateforme : features SIMD, cache, débit (x86 & ARM)
cargo run --example salient_outliers --release    # étude BiLLM : canaux outliers + préservation FP saillante
cargo run --bin slha-audit                         # auto-audit : invariants tuile, équivalence SIMD, fidélité, CCOS → Markdown
cargo run --bin slha-audit -- --json               # même rapport en JSON (CI / agents) ; --out FILE, --diff PRIOR.json
./scripts/bench_device.sh                          # lance le kit + exemples §7 sur l'appareil → results_<arch>.txt
./scripts/stress_test.sh                           # harnais massif : gate qualité + 16 exemples + rapport horodaté
```

**Bibliothèque sans dépendance** : la lib n'ajoute rien à l'arbre d'un
consommateur (PRNG déterministe maison). Seuls les **benches** tirent une
`criterion` allégée (dev-dependency, sans plotters/rayon) ; les tests
property/fuzz restent eux aussi sans dépendance.

## Statut (voir §5.1 et §7 du paper)

API sûre (pas de `read_volatile`), sémantique exacte, avec des **chemins SIMD
AVX2, AVX-512 (x86_64) et NEON (aarch64)** dispatchés à l'exécution + repli
scalaire portable, chacun avec un test d'équivalence ≡ scalaire (AVX2 ~×11,5,
AVX-512 ~×14,1 vs scalaire, **banc Xeon partagé** — ratios indicatifs, dépendants
du matériel et de l'auto-vectorisation). NEON **mesuré sur Jetson Thor AGX 128**
(Neoverse-V3AE) : ~**×5,7** vs scalaire (via le kit `platform_report`). Un
chemin popcount **AVX-512 VPOPCNTDQ** existe aussi (sélection à l'exécution).

Le prototype de mesure utilise des projections **aléatoires** (non apprises) :
il valide la machinerie *quantification INT4 + résidu 1-bit + ranking*, **pas**
la qualité d'une projection bas-rang apprise (qui ne peut qu'améliorer les
chiffres). Résultat clé : HOT ≥ WARM partout, Soft-Paging quasi sans perte à
faible énergie résiduelle, gains du résidu 1-bit modérés à `d_s = 256`.

## Organisation (`src/`)

| Fichier | Rôle |
|---|---|
| `attention/slha_v2.rs` | Tuile `SciRustSlhaTile` (128 o), kernel `compute_score` (scalaire + AVX2 + AVX-512 + NEON ; popcount VPOPCNTDQ), codecs latents INT4 (MX) / NF4 |
| `ccos.rs` | Cache KV élastique `ElasticKvCache` : Soft-Paging HOT/WARM/COLD, `page_out`/`evict`/`enforce_budget` (§4) |
| `linalg.rs` | Décomposition propre symétrique (Jacobi) pour la PCA |
| `learned.rs` | Projection bas-rang : PCA + **SGD task-aware** (`train_projection`) + génération de clés |
| `scenario.rs` | Projection sign-LSH, génération de contexte à énergie résiduelle `rho` contrôlable |
| `metrics.rs` | `dot`, Pearson, Spearman, top-k overlap |
| `rng.rs` | PRNG déterministe (SplitMix64) + échantillonneur gaussien |
| `json.rs` | JSON minimal **sans dépendance** (valeur + sérialiseur compact/joli + parseur) — partagé par l'audit et `slha-mcp` |
| `audit.rs` | Auto-audit : invariants tuile, équivalence SIMD≡scalaire *live*, features/cache, fidélité, budget CCOS, déterminisme → `Json` + Markdown + `diff` |
| `bin/slha_audit.rs` | CLI **`slha-audit`** (Markdown / `--json` / `--out` / `--diff PRIOR.json`) |
| `../tests/slha.rs` | Tests d'intégration (preuves) |
| `../tests/properties.rs` | Tests randomisés property / fuzz (zéro-dépendance) |
| `../benches/kernel.rs` | Micro-benchs criterion du kernel |
| `../examples/measure.rs` | Prototype de mesure (`rho` fixé) |
| `../examples/measure_learned.rs` | Prototype avec base apprise (PCA) + INT4 groupé (MX) |
| `../examples/bench_vs_fp16.rs` | Débit / trafic mémoire : SLHA (128 o) vs clé bf16 (256 o) |
| `../examples/benchmark_decode.rs` | **Proxy** decode bout-en-bout (tok/s) : multi-têtes score+softmax+V, SLHA vs réf f32 |
| `../examples/attention_fidelity.rs` | Fidélité de la sortie `softmax·V` (proxy de perplexité) |
| `../examples/learn_projection.rs` | Projection apprise (task-aware) vs PCA |
| `../examples/calibrate_lambda.rs` | Calibration de λ (dérive ΔP) vs référence FP |
| `../examples/cycles.rs` | Cycles/tuile (TSC via rdtsc) — complète le bench ns |
| `../examples/ccos_softpaging.rs` | Démo CCOS : cache KV élastique sous budget, fidélité de la sortie |
| `../examples/platform_report.rs` | Kit multi-plateforme (x86 & ARM) : features SIMD, niveaux de cache, alignement, débit |
| `../examples/salient_outliers.rs` | Étude BiLLM : injecte des canaux outliers, mesure INT4 vs préservation FP saillante |
| `../scripts/bench_device.sh` | Lance le kit + les exemples §7 sur l'appareil → `results_<arch>.txt` |
| `../tests/calibration.rs` | Test épinglant la calibration de λ (forme + constante) |
| `../tests/ccos.rs` | Tests d'intégration du Soft-Paging (masquage résidu, budget, recyclage) |
