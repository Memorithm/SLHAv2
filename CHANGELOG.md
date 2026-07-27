# Changelog

Format basé sur [Keep a Changelog](https://keepachangelog.com/) ; versioning
[SemVer](https://semver.org/). Ce fichier décrit l'état **réel** du code.

## [Unreleased]
### Added
- **Kernels SIMD pour NF4, mixte et MIX3** (AVX2/AVX-512/NEON) : les cinq
  codecs latents décodent désormais en SIMD avec repli scalaire portable.
  NF4 : lookup du codebook 16 entrées (`permutevar8x32`×2 / `permutexvar`
  / `vqtbl4q`) ; mixte et MIX3 : tête 8-bit vectorisée + corps
  nibble/3-bit ; MIX3 réutilise le pli algébrique des kernels TQ3
  (décodage par dimension bit-identique au scalaire, `FLAG_TQ3_NOCORR`
  honoré, sûreté des fenêtres de queue documentée). Équivalence ≤ 1e-3
  testée par ISA (mesuré ~1e-6/1e-7) ; les tests de routage scalaire
  deviennent des tests d'accord de chemins. 155 tests workspace.

### Fixed
- **`LICENSE.md` n'était pas le vrai texte PolyForm NC 1.0.0** : une
  paraphrase à 11 sections (sur 15) — sans « Noncommercial Purposes »,
  « Personal Uses », « Noncommercial Organizations » ni « Definitions »
  (le terme *permitted purpose* n'y était donc jamais défini), et avec une
  clause « Violations » réécrite **plus dure** que l'originale (résiliation
  immédiate au lieu du délai de mise en conformité de 32 jours). Remplacé
  par le texte canonique (polyformproject.org), Required Notice conservé.
  Formulations resserrées : l'exclusivité CCOS porte sur l'offre
  **commerciale** (l'usage non-commercial reste du PolyForm NC standard,
  sans restriction d'environnement) ; section « Prior versions » ajoutée à
  `LICENSING.md` (le passé MIT/Apache du dépôt restait non documenté là).

### Added
- **Codec `LatentCodec::Mix3` / `FLAG_MIX3` — la synthèse post-NO-GO** :
  tête 8-bit du codec mixte (8 dims) + corps TQ3 (112 dims à 3 bits, plan de
  correction séparable de 14 o), queue lâchée — 64 o pile. **Mesuré au
  niveau du mixte sur activations GPT-2 réelles** (held-out jointe :
  cos 0,9835 vs 0,9846, KL 0,0599 vs 0,0553) là où TQ3 pur était NO-GO
  (0,79), tout en conservant le barreau de pagination CCOS à plan séparable
  (échelle 128 → 114 → 96 → 82 → 0). Barreau CCOS généralisé aux deux codecs
  (`separable_corr_bytes`, `reclaimed_correction_bytes`) ; exposé dans
  `offline_validation --codec mix3` et `measure_learned` ; décodage scalaire
  (SIMD MIX3 en suivi). Voir `docs/TURBOQUANT.md` §3ter.
- **Positionnement CCOS et périmètre commercial documentés** : SLHAv2 et
  [TurboQuant](https://github.com/CHECKUPAUTO/TurboQuant) sont des modules
  compagnons de **CCOS**, et la licence commerciale est offerte pour les
  déploiements CCOS (`LICENSING.md`, README, FAQ de
  `docs/GETTING_STARTED.md`, `docs/TURBOQUANT.md` §6). TurboQuant adopte la
  **même double licence** (PolyForm NC 1.0.0 + commerciale) dans son propre
  dépôt. `CONTRIBUTING.md` documente désormais l'exigence de CLA que
  `LICENSING.md` §4 posait sans relais contributeur ; le pointeur licence
  du README corrigé (`LICENSE.md` → `LICENSING.md` pour l'arm commercial) ;
  `slha-python/pyproject.toml` déclare enfin sa licence.
- **Barreau de pagination CCOS « correction lâchée »** (`FLAG_TQ3_NOCORR`,
  `ElasticKvCache::drop_correction`) : le plan de correction de 16 o des
  tuiles TQ3 se page indépendamment — échelle HOT 128 → HOT¬corr 112 →
  WARM 96 → WARM¬corr 80 → COLD 0, dureté strictement croissante,
  comptabilité `live_bytes` exacte, déterminisme testé. Accesseurs
  `tile()`/`nocorr_count()`.
- **Kernels SIMD dédiés au codec TQ3** (AVX2/AVX-512/NEON) : correction 1 bit
  repliée algébriquement (`code − 3,75 + corr/2`, exact en f32),
  `FLAG_TQ3_NOCORR` honoré, sûreté de la lecture de queue documentée ;
  équivalence scalaire ≤ 1e-3 testée par ISA (mesuré ~1e-6). NF4/mixte
  restent scalaires.
- **`slha.compress` (MCP) : paramètre `codec` optionnel**
  (`int4|grouped|nf4|mixed|tq3`, défaut `int4` = comportement antérieur) ;
  la réponse rapporte `codec`, `flags`, `group_scales`.
- **Validation TQ3 sur activations réelles (GPT-2 c6, WikiText-2, held-out)** :
  **NO-GO mesuré** comme codec HOT (cos 0,78–0,83, KL ≈ 0,9–1,1 contre
  0,9846/0,0553 pour le codec mixte) — la grille uniforme sans zéro
  s'effondre sur le spectre raide réel ; la projection jointe relève l'INT4
  groupé mais pas TQ3 (goulot = grille, pas sous-espace). Tableau complet et
  protocole reproductible dans `docs/TURBOQUANT.md` §3bis, post-scriptum dans
  `FINDINGS.md` §5. TQ3 reste pertinent sur distributions plates et pour son
  barreau de pagination à plans séparables.
- Docs : TQ3 dans `docs/api.md` (avec le codec mixte, absent lui aussi),
  `SLHAv2.md` §3.1/§5.1/§7.12, `paper/slhav2.tex` (énumération + table
  d'ablation), `docs/MCP.md`.
- **Codec latent TQ3 (portage TurboQuant)** : `LatentCodec::Tq3` / `FLAG_TQ3` —
  grille 3 bits symétrique (8 niveaux, sans zéro) + plan de correction de signe
  1 bit par dimension, échelles par groupe comme les autres codecs. 48 o de
  codes + 16 o de corrections = le budget latent de 64 o exactement ; la tuile
  reste 128 o, zéro padding. Décodage scalaire à ce stade (kernels SIMD
  ajoutés ensuite — cf. entrée ci-dessus). Exposé dans
  `offline_validation --codec tq3` et `measure_learned`. Mesuré au niveau des
  codecs 4 bits existants (cos HOT 0,9979–0,9999, Spearman 0,881 vs 0,884
  INT4 groupé) ; trade-off documenté (pas de niveau zéro → MSE ~1,3–1,6× INT4
  sur gaussien, testé ≤ 2×) contre un plan de correction **séparable**
  (barreau de pagination CCOS implémenté ensuite — cf. entrée ci-dessus). Voir
  [`docs/TURBOQUANT.md`](docs/TURBOQUANT.md). Le portage a mis au jour un bug
  de grille dans TurboQuant amont (valeurs positives écrasées à 0) — corrigé
  là-bas, garde-fou `tq3_positive_values_do_not_collapse` ici.

## [0.2.0] — 2026-06-30
### Fixed
- **Build cassé sur aarch64 (PR#19, `slha-c`)** : `slha_audit`/`slha_free_string`
  déclaraient `*mut i8` alors que `CString::into_raw`/`from_raw` utilisent
  `*mut c_char` (qui vaut `*mut u8` sur aarch64 — `char` non signé sur ARM).
  Corrigé en `*mut std::os::raw::c_char` (portable). La CI x86_64 ne le voyait
  pas (c_char = i8 sur x86) et son step aarch64 ne construisait que `scirust` ;
  ce step type-check désormais `scirust` + `slha-c` + `slha-mcp` contre la cible
  aarch64 (`cargo check`, pas `build` — un cross-`build` linkerait `slha-c`
  (cdylib) et `slha-mcp` (bin) et demanderait un cross-linker aarch64 absent sur
  le runner x86_64 ; `check` résout `cfg(target_arch="aarch64")` et attrape
  l'erreur de type E0308 sans linker). `slha-python` (PyO3) reste exclu (dev libs
  Python aarch64).
- **Licence des crates** : tous les manifests membres (`scirust`, `slha-mcp`,
  `slha-c`, `slha-python`) déclaraient `license = "MIT OR Apache-2.0"` alors que
  le dépôt est en double licence **PolyForm Noncommercial 1.0.0 + commerciale**.
  Alignés sur `PolyForm-Noncommercial-1.0.0`. Les sections licence
  contradictoires du `README` (MIT/Apache + liens vers `LICENSE-MIT`/
  `LICENSE-APACHE` supprimés) et la FAQ de `docs/GETTING_STARTED.md` corrigées.
- **`slha.h`** : ajout de la branche d'alignement 128 o
  (`-DSLHA_CACHE_LINE_128=1`) et du chemin MSVC (`__declspec(align)`).
- **`slha_init`** : remplace le `static mut DUMMY` (accès à static mutable,
  évité en Rust moderne) par un `NonNull::dangling()` ; cycle de vie documenté.
- **Robustesse numérique** : `metrics::softmax_into` ne produit plus de NaN/Inf
  sur entrée tout-`-inf` (retourne une distribution uniforme, comme les autres
  helpers sur entrée dégénérée). Test ajouté.
- **Claims non étayés** : « sans ralentissement » / « 8 Go VRAM → 4 Go RAM » /
  ratio 125× désormais qualifiés de projections (mesuré kernel = 2× vs clé bf16) ;
  « imperceptible » et « Raspberry Pi » retirés (aucune mesure sur modèle réel
  ni sur Pi). Statut « intégration LLM réel » repassé d'✅ à 🟡 (esquisse seule).
  Table de comparaison de `docs/GETTING_STARTED.md` : retiré les claims
  « Tient dans le cache CPU (L1/L2) » / « cache hit en 1-4 cycles » / « 200 Go/s »
  (résidence cache non mesurée — §6.1, compteurs `perf` indisponibles ; 12 Mo ne
  tiennent pas en L1/L2) et « libère 30% de mémoire » / « perte ~5% » du
  Soft-Paging (§4 mesure −25 % d'empreinte et cos 0,9995 = 0,05 % de déviation).
- **Comptes de tests** mis à jour : 85 workspace (78 `scirust` + 7 `slha-mcp`).
- **CI rouge sur master (PR#19)** : le merge PR#19 avait laissé `cargo fmt
  --all --check` cassé (`scirust/src/adapter.rs` et `slha-python/src/lib.rs`
  non formatés — le `Format` step échouait en premier, masquant tout le reste).
  `cargo fmt --all` appliqué. La CI master n'était donc plus verte depuis PR#19.
- **Step `Benchmarks compile`** : `cargo bench --workspace --no-run` échouait
  au link du « lib test » de `slha-python` en profil release/LTO — pyo3
  `extension-module` ne link pas libpython (fourni par l'interpéteur au load),
  et le LTO flaggue alors les symboles `Py_*` non résolus. `cargo build
  --workspace --all-targets` et `cargo test --workspace` (profil dev, sans LTO)
  restaient verts ; seul le profil release/bench était touché. Seul `scirust`
  ayant des benches, le step cible désormais `cargo bench -p scirust --no-run`
  (documenté dans `CONTRIBUTING.md`).
- *Note historique* : l'entrée précédente mentionnant `LICENSE-MIT` +
  `LICENSE-APACHE` à la racine est obsolète — ces fichiers ont été retirés et le
  crate re-licencié en PolyForm Noncommercial (double licence commerciale).

### Added
- **Filtre de sécurité géométrique latent** (`scirust::safety`,
  `LatentSafetyGuard`) — axe C du point 5 (roadmap d'optimisation matérielle,
  Phase 1). Classifieur ultra-léger (~200 cycles, zéro allocation) opérant
  **directement sur les vecteurs latents compressés** (`[u8; 64]`, INT4) avant
  décompression, pour bloquer injections de prompts / jailbreaks / dérives
  sémantiques avant la génération du token. Trois signaux testés dans l'ordre :
  (1) déviation angulaire — cosinus vs vecteur directeur de référence
  (magnitude-invariant, normalisé par `‖v‖`) ; (2) isolation orthogonale —
  classifieur linéaire optionnel (`with_linear_classifier`) ; (3) dérive
  glissante — moyenne du cosinus sur une fenêtre de 4 échantillons, évaluée
  seulement une fois la fenêtre pleine (évite les faux positifs au démarrage).
  Module **additif** : n'altère ni la tuile 128 o ni les kernels SIMD ; pur
  safe Rust portable (x86_64/aarch64/…) ; self-audit `slha-audit` reste 7/7.
  Tests unitaires + d'intégration + doctests ; docs dans `docs/api.md` et
  `SLHAv2.md` §5.1.
- **Pool NUMA-aware + épinglage de thread + allocation alignée**
  (`scirust::numa`) — axe A du point 5 (roadmap d'optimisation matérielle,
  Phase 2). Deux niveaux d'API :
  - **`AlignedBuffer`** — allocation heap alignée 128 o (ligne de cache, ou
    alignement configurable) via l'allocateur global `std::alloc`. **Portables,
    zéro dépendance, disponibles par défaut** sur toutes cibles (x86_64/aarch64/…)
    pour aligner les buffers chauds du chemin SIMD indépendamment du NUMA.
  - **Feature optionnelle `numa`** (Linux + `libc` en dépendance *optionnelle* —
    la configuration par défaut reste **sans dépendance externe**) : `NumaBuffer`
    (région `mmap(MAP_ANONYMOUS|MAP_PRIVATE)` page-alignée + `mbind(MPOL_BIND,
    MPOL_MF_MOVE)` best-effort vers le nœud local), `pin_current_thread_to_cpu`
    / `pin_current_thread_local` (`sched_setaffinity`), introspection
    `current_cpu`/`current_node`/`num_nodes`/`numa_available` (parsing sysfs
    `/sys/devices/system/node`). Repli gracieux hors Linux / sans la feature
    (`NumaError::Unavailable` ; `NumaBuffer` non construisible). **Intégration
    first-touch** sur l'arena KV-cache : `ElasticKvCache::pin_caller_to_local_numa`
    épingle le thread d'inférence à son CPU local avant le warm-up — les pages du
    `Vec` (non page-aligné, donc `mbind` peu fiable) atterrissent sur le bon nœud
    par first-touch, sans `mbind`. Sur Jetson Thor (mémoire unifiée, mono-NUMA),
    `numa_available()` rend `false` et l'épinglage reste utile (évite les
    migrations de thread). CI : nouveau job dédié `numa-check` (check + clippy +
    build + test + doc + cross-check aarch64, tous avec `--features numa`) ; job
    `msrv` étendu (`cargo check -p scirust --features numa --all-targets`).
    Tests d'intégration (`scirust/tests/numa.rs`) : `AlignedBuffer` (alignement,
    zero, roundtrip, rejets d'alignement invalide, len nulle) + chemin Linux réel
    best-effort (tolérant mono-nœud/permissions CI) + repli stub. Self-audit
    `slha-audit` reste 7/7. Docs : `docs/api.md`, `SLHAv2.md` §5.1.
- **Plan d'amélioration — Phase 1 (fidélité) : axes A1 et A2** implémentés
  comme modules *additifs* (aucun changement à la tuile 128 o ni aux kernels
  SIMD ; les 51 tests historiques restent verts). Voir
  `docs/SLHAv2_schema_plan.pdf` pour le plan complet.
  - **A2 — Incohérence Hadamard (QuIP#/Palu)** sur le résidu sign-LSH :
    nouveau module `scirust::incoherence` (FWHT orthonormée O(d·log d) +
    transformée randomisée `H·D` diagonale ±1). Câblage **opt-in** dans
    `LearnedModel` (`fit_with`/`from_projection_with(..., rht: bool)`) : la
    RHT est appliquée au résidu `E` et à la requête `Q` avant le sign-LSH.
    **Orthogonale ⇒ `⟨RHT·E, RHT·Q⟩ = ⟨E, Q⟩`** : le score fusionné est
    préservé, seul le résidu 1-bit gagne en résolution. **Mesuré** : dans le
    régime « outlier aveuglant » (direction forte commune + signal unique
    structuré — le cas QuIP#), le cœur binaire passe de Spearman 0,07 à 0,49
    (**+0,42**) ; **WARM est préservé bit-exact** (ΔWARM = 0,0000) car la RHT
    n'atteint jamais le chemin coarse. **Honnêtement** : sur résidu bien
    conditionné, la RHT est neutre à nuisible pour HOT → A2 est *opt-in
    conditionnel* (activer si peak/mean du résidu est élevé), pas un défaut.
    Exemple `examples/hadamard_incoherence.rs`.
  - **A1 — Projection bas-rang sur clés PRE-RoPE (ShadowKV)** : nouveau
    module `scirust::rope` (rotation RoPE standard par paires de canaux,
    orthogonale, testée). Nouvelle API publique
    `learned::captured_energy_at(train, d, rank)` (énergie captée par un PCA
    de rang k — pour sonder le spectre). **Mesuré (robuste sur 4 seeds)** :
    RoPE détruit le bas-rang des clés — énergie captée chute de ~99,5 % à
    ~92 % à rang 128 (Δ +7 %), et de 99 % à 68 % à rang 32 (Δ +30 %). C'est
    la racine mesurée du goulot « projection » du §7.8, exactement le
    mécanisme ShadowKV. **Honnêtement** : sur ce factor model synthétique, la
    levée du *Spearman WARM* n'est pas robuste (légèrement négative, 0/5
    seeds) — la queue perdue touche la magnitude plus que le ranking, et
    l'erreur de reconstruction pre-RoPE (rotée) peut manger le gain. Une
    levée robuste du plafond WARM nécessite les clés d'un vrai LLM (queue
    lourde, perplexité) — intégration Phase 3 / A7, comme le plan
    l'anticipait. Exemple `examples/pre_rope_projection.rs`.
  - +14 tests (incoherence 7, rope 4, learned A1/A2 3) → **60 tests scirust,
    70 workspace**.
- **Plan d'amélioration — Phase 2 (politique de cache) : axes A4 et A5**,
  implémentés comme modules *additifs* (la tuile 128 o et les kernels SIMD
  sont inchangés ; les 70 tests Phase 1 restent verts).
  - **A5 — Éviction informée (H2O / StreamingLLM / SnapKV)** : nouvelle
    `ccos::EvictionPolicy` (`Causal` par défaut, back-compatible, ou
    `Importance { sink_window }`) qui remplace l'éviction purement causale par
    un ordre d'**importance** : éviction des tuiles de plus faible masse
    d'attention cumulée (H2O), avec **pinnage des attention sinks** (les
    `sink_window` premiers jetons, par position — StreamingLLM). L'importance
    est accumulée par `ElasticKvCache::observe_scores` (softmax de l'attention
    sur les tuiles live, à chaque pas de décodage). `σ_E` garde son rôle dans
    la phase de *paging* (HOT→WARM via `PageOutPolicy`) — seule la phase
    d'*éviction* change. **Mesuré** (`examples/informed_eviction.rs`, scénario
    construit heavy-hitters + sinks) : sous pression (16/64 tuiles gardées),
    le cosinus de la sortie d'attention passe de **0,13 (Causal) à 0,53
    (Importance)** (+0,41) — l'éviction causale droppait les heavy-hitters
    mi-séquence ET les sinks. Le scénario est construit pour exhiber le
    mécanisme ; la magnitude sur vrai LLM est l'axe A7 / Phase 3, le *signe*
    est ce que la mesure confirme. +3 tests ccos (éviction informée + sinks,
    accumulation H2O, et `sink_window = 0` = H2O pur sans pinning).
  - **A4 — Résidu multi-bit / multi-round (QINCo / Reformer)** à **budget
    256 bits fixé** (l'invariant tuile 128 o tient : on ne change que la façon
    de *dépenser* les 256 bits du résidu). Nouveau module `scirust::residual` :
    `BinaryResidual` (1-bit généralisé), `QuantResidual` (`b`-bit, `D_S/b`
    hyperplans, quantificateur uniforme centré calibré par σ_E),
    `MultiRoundResidual` (`K` hashes 1-bit de `D_S/K` bits, moyenne). **Mesuré
    (robuste sur 18 combos decay×seed — 3 decays × 6 seeds)** : le multi-bit
    réduit l'**erreur relative L2** (`rel_l2 = ‖est−true‖/‖true‖`, *pas* une MSE)
    de l'estimateur du résidu d'un facteur **> ~3,3× (garanti par le test sur les
    18 combos)** — typique ~×4, jusqu'à **×200+ à haut-ρ (×245 mesuré au pic)**
    où le sign 1-bit sature — c'est le levier confirmé du plan (« meilleur HOT à
    rho élevé »), un gain de *magnitude*. **Honnêtement** : le gain de *rang*
    (Spearman) n'est **pas robuste** — le 1-bit×256 (plus d'hyperplans) échantillonne
    mieux la direction, et le multi-bit s'effondre sur certaines seeds (jusqu'à
    négatif) : trade-off magnitude vs direction, pas une domination. La
    graduation Soft-Paging HOT2 (`b`-bit) → HOT1 (MSB = sign 1-bit) → WARM est
    un masquage de bits des mêmes 32 o (O(1), intégration Phase 3). Exemple
    `examples/multibit_residual.rs`. +11 tests residual (budget fixe, rang 1-/2-/8-bit,
    équivalence kernel tuile, **padding du mot de queue** (n_bits non multiple de 64),
    **niveau le plus proche du quantificateur + conservation du signe à bits=1**
    (pin du bug half-step : `floor` pas `round` sur grille centrée), **k=1 ≡
    BinaryResidual(D_S) bit-pour-bit**, **k=3 rejeté (panic)**, réduction robuste
    sur 18 combos).
  - +14 tests (A4 : 11 residual, A5 : 3 ccos) → **74 tests scirust, 81 workspace**
    (hors doctests ; +3 doctests scirust).
- **Serveur MCP `slha-mcp`** (nouveau crate du workspace, **zéro dépendance
  externe** — réutilise `scirust::json`) : serveur Model Context Protocol sur
  **stdio** (JSON-RPC 2.0 délimité par lignes) qui expose le noyau et l'auto-audit
  SLHA comme **outils appelables par un agent** (Claude Code / Desktop, ou tout
  client MCP). 5 outils : `slha.audit`, `slha.explain`, `slha.compress`,
  `slha.score`, `slha.benchmark`. Branchement :
  `claude mcp add slha -- .../target/release/slha-mcp`. Guide complet
  `docs/MCP.md`. +7 tests de dispatch → **57 tests** (workspace : 50 scirust + 7
  slha-mcp).
- **Outil d'auto-audit `slha-audit`** (bin) + modules `scirust::audit` et
  `scirust::json` (JSON **sans dépendance** : valeur + sérialiseur + parseur).
  L'audit exécute tous les invariants à l'exécution — layout de tuile (128 o,
  zéro padding, alignement), **équivalence SIMD ≡ scalaire** *live*, features
  CPU + niveaux de cache, **fidélité de sortie** vs attention complète,
  **invariant de budget CCOS**, déterminisme — et rend un rapport **Markdown**
  ou **JSON** (`--json`/`--pretty`/`--out FILE`), avec **diff vs un rapport
  antérieur** (`--diff PRIOR.json`, exit ≠ 0 sur régression). Code de sortie ≠ 0
  si un contrôle échoue. +9 tests (JSON 5, audit 4) → **50 tests** (scirust).
  Réutilisé par le serveur `slha-mcp` (ci-dessus).
- **Prêt pour crates.io / docs.rs** : métadonnées de publication sur `scirust`
  (`keywords`, `categories`, `readme`, `documentation`, `rust-version`) ;
  `cargo publish -p scirust --dry-run` passe (35 fichiers, sans avertissement).
  `slha-mcp` reçoit aussi les métadonnées et une dépendance `scirust` versionnée
  (publiable une fois `scirust` sur crates.io). **MSRV = 1.89** (intrinsèques
  AVX-512 stabilisées en 1.89 ; `usize::is_multiple_of` en 1.87).
- **Fichiers de licence** `LICENSE-MIT` + `LICENSE-APACHE` à la racine (le crate
  déclarait `MIT OR Apache-2.0` sans fournir les textes ; lien `LICENSE` du
  README désormais valide). Conformité double-licence façon écosystème Rust.
- **Harnais de test massif** `scripts/stress_test.sh` : exécute la barrière
  qualité complète (fmt, clippy `-D warnings`, build debug+release, tests
  debug+release, doc, benches, cross-compile aarch64), **lance les 11 exemples**,
  vérifie le **déterminisme** de sortie, propose un mode **soak**, et **génère un
  rapport Markdown + JSON horodaté** sous `target/stress/` (auditable). Lance
  aussi `slha-audit` ; suite à **50 tests** verts.
- **Alignement adaptatif à l'hôte via `build.rs`** (`SciRustSlhaTile`, §3.1) :
  script de build sans dépendance qui sonde la **taille de ligne L1d réelle de
  l'hôte** sur une *build native* (triplet hôte == cible ; `sysfs` Linux ou
  `sysctl` macOS) et émet `cfg(cache_line_128)` pour porter la tuile à
  `align(128)` **uniquement** sur une puce à ligne de 128 o (p. ex. Apple
  Silicon). En cross-compilation, la ligne de l'hôte n'a pas de rapport avec la
  cible : le défaut sûr `align(64)` est conservé. Raffinement de portabilité,
  pas de correctness — la tuile reste **128 o sans padding** dans les deux cas,
  et sur **toutes nos cibles** (x86-64, Thor) le résultat est inchangé
  (`align(64)`). Remplace l'hypothèse « AArch64 ⇒ 128 » retirée ; test
  `tile_is_exactly_128_bytes_zero_padding` rendu cfg-aware. `build.rs` émet
  aussi `rustc-check-cfg` (zéro warning `unexpected_cfgs`).
- **Kit de mesure multi-plateforme** (`examples/platform_report.rs` +
  `scripts/bench_device.sh`) : binaire portable (x86-64 **et** AArch64) qui
  détecte les features SIMD (AVX2/AVX-512/VPOPCNTDQ ou NEON/dotprod/SVE/SVE2),
  **liste tous les niveaux de cache et leur taille de ligne**, vérifie la
  taille/alignement de tuile vs la ligne, affiche le chemin kernel dispatché,
  et mesure le débit (scalaire vs SIMD) en temps mur. A servi à produire les
  **chiffres ARM réels** sur **Jetson Thor AGX 128** (NEON 17,1 M/s vs 3,0
  scalaire = 5,7× ; toutes lignes de cache à 64 o ; `sve2` présent) ; x86 reste
  la baseline serveur.

### Changed / Corrected
- **Robustesse & finitude (nits)** :
  - `install.sh` s’exécute désormais localement après inspection — les prompts lisent
    `/dev/tty` (ou prennent le défaut affiché en non-interactif au lieu
    d'avorter sur EOF) ; au passage, le chemin « garder le dossier existant » ne
    re-clone plus en double (bug `cd` imbriqué corrigé).
  - `metrics` : `rms([])`, `topk_overlap(.., 0)` et `pearson([], [])` renvoient
    `0.0` (fini) au lieu de `NaN` (division `0/0`) ; test de finitude ajouté
    (→ **51 tests** scirust, **58** workspace).
  - Docs : caveat « ratios SIMD **indicatifs, dépendants du matériel** » près des
    chiffres x86 (banc Xeon), par symétrie avec les chiffres ARM déjà qualifiés.
- **CI durcie** : ajout de `cargo doc` (warnings = erreurs), exécution
  bout-en-bout de **`slha-audit`**, `cargo publish -p scirust --dry-run`, et un
  **job MSRV (Rust 1.89)** qui vérifie tout le workspace `--all-targets`.
- **Durcissement NaN des tris flottants** (`metrics::ranks`/`topk_overlap`,
  `learned::fit`, exemple `salient_outliers`) : `partial_cmp().unwrap()` →
  `f32/f64::total_cmp` (ordre total sans panique). Comportement identique sur
  données finies ; supprime un risque de panique sur l'API publique en cas de
  `NaN`/`Inf` en entrée. Repéré par l'audit code.
- **Statut toolchain SVE2 documenté précisément** (roadmap #1 ; paper Future
  Work, `SLHAv2.md` §7.4, `FINDINGS.md`). Vérifié sur `rustc 1.94.1` : la
  **détection** runtime `is_aarch64_feature_detected!("sve2")` est *stable*,
  mais les **intrinsèques** SVE2 (`svdot_s32`…) sont **absentes du
  `core::arch::aarch64` stable** (nightly-only, comme `std::simd`) ; la seule
  voie stable (`asm!` manuel) *compile* mais reste **invérifiable sans appareil
  SVE2** (CI x86 ; la cross-compilation ne type-checke pas la sémantique de
  l'`asm!`). On garde donc **NEON + `cnt`** comme chemin livré, mesuré et
  correct ; SVE2 reste sur la roadmap (défer *justifié par le toolchain*, pas un
  oubli). Aucun changement de code.
- **Alignement de tuile ramené à `align(64)` universel** (`SciRustSlhaTile`,
  §3.1). On avait introduit un `align(128)` conditionnel sur `aarch64` en
  supposant une ligne de cache de 128 o sur le Jetson Thor ; **la mesure de
  l'appareil l'a réfuté** (L1d/L1i/L2 = 64 o — le « 128 » d'*AGX 128* = les
  128 Go de **mémoire unifiée CPU/GPU** LPDDR5X, pas la ligne de cache).
  `align(64)` est correct et optimal sur les deux cibles
  (tuile = 2 lignes de 64 o). Un `align(128)` ne sert que sur les puces à ligne
  de 128 o (p. ex. Apple Silicon) → détection hôte en `build.rs` (**désormais
  implémenté**, cf. *Added* ci-dessus).
- **Popcount résidu vectorisé AVX-512 VPOPCNTDQ** (`hamming_distance`, eq. 2.3) :
  chemin x86-64 *branchless* qui plie les 256 bits du résidu en un seul `vpopcntq`
  (`_mm256_popcnt_epi64`), sélectionné à l'exécution (`avx512vpopcntdq`+`vl`) avec
  repli `count_ones()` (→`POPCNT`/`CNT`). Équivalence bit-à-bit garantie
  (`vpopcntdq_hamming_matches_scalar` sur CPU compatible, compile-checked sinon ;
  `hamming_distance_matches_bruteforce` partout).
- **Constante λ calibrée exposée** (`scenario::LAMBDA_C_CALIBRATED` ≈ 0,33,
  `calibrated_lambda`, `analytic_lambda_c`, §7.9) : option pour le poids du
  résidu corrigé du facteur ~4,2×. `build_tile` garde la constante **analytique**
  par défaut (conservatrice : la calibration optimise la *magnitude*, pas le
  *ranking*). Test `calibrated_lambda_needs_no_further_multiplier` (α\* ≈ 1).
- **Property-tests CCOS randomisés** (`tests/ccos.rs`) :
  `prop_enforce_budget_respects_budget_and_recycles` (300 configs) épingle les
  invariants `live_bytes ≤ budget`, cohérence octets/compteurs, et recyclage des
  slots COLD, sur les deux politiques.
- **Couche d'interfaçage CCOS** (`src/ccos.rs`, §4) : `ElasticKvCache`, un cache
  KV élastique sur **arène contiguë** qui pilote le *Soft-Paging*. Trois états
  HOT (128 o) / WARM (96 o, résidu masqué + `λ = 0`) / COLD (évincé, slot
  recyclé) ; `page_out()` masque/libère les 32 o de `residual_bitmap` en **O(1)**
  sans I/O ni allocation ; `enforce_budget()` borne l'empreinte logique sous un
  budget en octets (`PageOutPolicy::LowestImpactFirst` — plus faible `σ_E`
  d'abord — ou `OldestFirst`) puis évince si nécessaire ; `evict()` recycle le
  slot via free-list. La politique par défaut est l'**hybride** (`Default` :
  pagination par `σ_E`, éviction par ancienneté) ; `with_budget()` la construit.
  Exemple `examples/ccos_softpaging.rs` + 6 tests d'intégration
  (`tests/ccos.rs`). Mesure : pager **la moitié** des tuiles HOT→WARM laisse la
  sortie d'attention à **cos ≈ 0,9995** vs tout-HOT.
- **Calibration de λ** (`examples/calibrate_lambda.rs` + test
  `tests/calibration.rs`, §7.9) : confronte le poids du résidu à une attention
  FP de référence. La forme `λ ∝ σ_E` est **validée** (α* stable sur `rho`) ;
  la constante `√(π/(2·d_s))` **sous-pondère ~4,2×** → constante calibrée
  `C_emp ≈ 0,33` (d_s = 256). La formule analytique reste le défaut conservateur.
- **Coût en cycles** (`examples/cycles.rs`, via `rdtsc`) : ~942 cyc/tuile
  scalaire, ~89 AVX2, ~71 AVX-512 ; balayage de working-set (signal cache
  indirect — compteurs `perf` indisponibles). Complète le bench criterion (ns).

### Fixed
- **Doc & packaging.** Remplacement d'un second crate `scirust` déclaré à la
  racine dont le bench (`benches/score.rs`), la doc (`docs/api.md`) et ce
  changelog décrivaient une **API inexistante** *portée par la tuile*
  (`SciRustSlhaTile::new`, `score_safe`, `enforce_paging`, `TileState`/`TileError`)
  et une tuile de « 104 octets » (à ne pas confondre avec le gestionnaire réel
  `ccos::ElasticKvCache` ajouté ci-dessus, distinct de la tuile). La racine est
  désormais un **workspace Cargo** autour de
  l'unique crate `scirust` ; `docs/api.md` documente l'**API réelle** (tuile de
  **128 octets**, score via `compute_score`) ; le bench cassé est supprimé
  (`scirust/benches/kernel.rs`, fonctionnel, est conservé). Suppression des
  features `avx2/popcnt/neon = []` no-op (la sélection SIMD est *runtime*).

## [0.2.0] - 2026-06-16

### Added
- `SciRustSlhaTile` : tuile **128 octets**, alignée 64, **zéro padding** (latent
  64 o + résidu 32 o + métadonnées 32 o), vérifié par test.
- `compute_score` (eq. 2.3) avec dispatch à l'exécution **AVX-512 > AVX2 >
  scalaire** (x86_64) et **NEON** (aarch64) ; équivalences SIMD ≡ scalaire
  testées (property/fuzz inclus).
- Codecs latents : INT4 **signé** (zero-point), INT4 **par groupe (MX)**, **NF4**
  (codebook normal) — même tuile 128 o.
- Résidu 1-bit sign-LSH + cœur `popcount` (identité de Hamming prouvée vs réf.).
- `learned` : projection **PCA** (`jacobi_eigh`) et projection **apprise
  task-aware** par SGD (`train_projection`), qui bat la PCA sous décalage Q/K.
- Exemples : `measure`, `measure_learned`, `bench_vs_fp16`, `attention_fidelity`,
  `learn_projection`, `basic_usage`.
- Tests : unitaires + intégration + **property/fuzz** + **doctests** (30 au total).
- **criterion** benches (dev-dependency allégée, lib sans dépendance) ; **CI**
  (fmt + clippy `-D warnings` + tests + benches + cross-compile NEON).

### Fixed (par rapport au paper v1)
- Tuile : **128 octets** et non « 104 » (`align(64)` arrondit la taille ; vérifié
  empiriquement `size_of = 128`).
- Déquantification INT4 **signée** `(nibble − 8)·scale` (et non `[0, 15]·scale`).
- Retrait du `read_volatile` (qui bloquait la vectorisation) et de l'import /
  `target_feature(avx2)` trompeurs.

## [0.1.0] - 2026

### Added
- Spécification SLHA v2 (`SLHAv2.md`) et micro-noyau de référence initial.
