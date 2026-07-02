# SLHA v2 — API Reference

Référence de l'API **réelle** du crate `scirust` (vérifiée contre le code et les
tests). Pour la spécification et les mesures, voir [`../SLHAv2.md`](../SLHAv2.md)
et [`../FINDINGS.md`](../FINDINGS.md).

> ⚠️ Une version antérieure de ce fichier décrivait une API inexistante
> (`new`, `score_safe`, `TileError`, des méthodes de paging *portées par la
> tuile*…) et une tuile de « 104 octets ». **C'est faux** : voir l'API ci-dessous
> (tuile de **128 octets**, score via `compute_score`). Le Soft-Paging réel vit
> dans un gestionnaire séparé, [`ccos::ElasticKvCache`](#modules-scirust), pas
> sur la tuile. Corrigé.

## Modules (`scirust::…`)

| Module | Rôle |
|---|---|
| `attention::slha_v2` | Tuile + kernel de score fusionné (eq. 2.3), quantizers INT4/NF4/mixte/TQ3 |
| `ccos` | `ElasticKvCache` (Soft-Paging HOT/WARM/COLD, §4), `PageOutPolicy`, `TileState` |
| `metrics` | `dot`, `cosine`, `rel_l2`, `pearson`, `spearman`, `topk_overlap`, `rms`, `softmax_into` |
| `rng` | PRNG déterministe `Rng` (SplitMix64) + gaussien |
| `linalg` | `jacobi_eigh` (décomposition propre symétrique, pour la PCA) |
| `learned` | `LearnedModel` (PCA + SGD task-aware), `train_projection`, `gen_keys` |
| `scenario` | `Projection` (sign-LSH), `build_tile`, `generate` (données synthétiques) |
| `safety` | `LatentSafetyGuard` — filtre de sécurité géométrique dans l'espace latent compressé (anti-injection, anti-dérive), opère avant décompression |
| `numa` | `AlignedBuffer` (alignée, portable, zéro-dép) + politique NUMA/épinglage de thread optionnelle (feature `numa`, Linux + `libc`) |

## Constantes (`attention::slha_v2`)

```rust
pub const D_C: usize = 128;            // dim latente (INT4)
pub const D_S: usize = 256;            // bits de résidu sign-LSH
pub const LATENT_BYTES: usize = 64;    // D_C / 2
pub const RESIDUAL_WORDS: usize = 4;   // D_S / 64
pub const N_GROUPS: usize = 8;         // groupes de micro-échelles
pub const GROUP_DIM: usize = 16;       // D_C / N_GROUPS

pub const FLAG_HOT: u16 = 0;           // tuile complète (latent + résidu)
pub const FLAG_WARM: u16 = 1 << 0;     // résidu paginé : score = latent seul
pub const FLAG_NF4: u16 = 1 << 1;      // latent codé en NF4 (sinon INT4 uniforme)
pub const FLAG_MIXED: u16 = 1 << 2;    // latent mixte : 8 dims @8-bit + 112 @4-bit
pub const FLAG_TQ3: u16 = 1 << 3;      // latent TQ3 (TurboQuant) : 3 bits + correction 1 bit

pub const MIXED_HI_DIMS: usize = 8;    // dims stockées à 8 bits (codec mixte)
pub const MIXED_LO_DIMS: usize = 112;  // dims à 4 bits ; la queue (8 dims) est lâchée
pub const MIXED_DIMS: usize = 120;     // dims conservées par le codec mixte
pub const MIXED_LO_GROUPS: usize = 7;  // groupes 4 bits (gs[0] = échelle du bloc 8 bits)

pub const TQ3_CODE_BYTES: usize = 48;  // 128 dims × 3 bits (codes, flux little-endian)
pub const TQ3_CORR_BYTES: usize = 16;  // 128 dims × 1 bit (signes de correction)
pub const TQ3_HALF_RANGE: f32 = 3.5;   // grille 8 niveaux : code − 3,5 (pas de niveau zéro)
pub const TQ3_CORRECTION: f32 = 0.25;  // amplitude de la correction 1 bit, en pas de grille

pub const NF4_CODEBOOK: [f32; 16];     // 16 quantiles N(0,1) normalisés à [-1, 1]
```

## `SciRustSlhaTile` — **128 octets, align 64, zéro padding**

```rust
// align(64) par défaut ; align(128) sur un hôte natif à ligne de 128 o
// (build.rs émet cfg(cache_line_128)). Taille = 128 o dans les deux cas.
#[cfg_attr(cache_line_128, repr(C, align(128)))]
#[cfg_attr(not(cache_line_128), repr(C, align(64)))]
pub struct SciRustSlhaTile {
    pub latent_kv: [u8; 64],          // 64  base h_KV : INT4/NF4 (2/octet), mixte 8/4, ou TQ3 (3+1 bits)
    pub residual_bitmap: [u64; 4],    // 32  résidu sign-LSH (256 bits)
    pub scale: f32,                   //  4  échelle globale de déquantification
    pub dynamic_lambda: f32,          //  4  poids de correction binaire λ (eq. 3.2)
    pub residual_sigma: f32,          //  4  σ_E par tuile (recalibrage de λ)
    pub token_id: u32,                //  4
    pub position: u32,                //  4
    pub head_id: u16,                 //  2
    pub flags: u16,                   //  2  HOT / WARM / NF4 / MIXED / TQ3
    pub group_scales: [u8; 8],        //  8  micro-échelles MX : eff(g) = scale·gs[g]/255
}
```

Le test `tile_is_exactly_128_bytes_zero_padding` vérifie `size_of == 128`,
`align_of == 64` par défaut (128 sur un hôte natif à ligne de 128 o), et
l'absence de padding.

### Méthodes

```rust
impl SciRustSlhaTile {
    pub fn is_warm(&self) -> bool;     // résidu paginé ?
    pub fn is_nf4(&self) -> bool;      // latent en NF4 ?
    pub fn is_mixed(&self) -> bool;    // latent mixte 8/4-bit ?
    pub fn is_tq3(&self) -> bool;      // latent TQ3 (3 bits + correction 1 bit) ?
    pub fn group_scale(&self, d: usize) -> f32;   // scale·gs[d/16]/255
    pub fn dequant_at(&self, d: usize) -> f32;     // décode INT4 / NF4 / mixte / TQ3 selon les flags
    pub fn dequant_latent(&self) -> [f32; 128];

    /// Score fusionné (eq. 2.3) :
    ///   <q_coarse, dequant(latent)> + λ·(d_s − 2·popcount(q_sign ⊕ B))
    /// Dispatch runtime : AVX-512 > AVX2 > scalaire (x86_64), NEON (aarch64).
    /// Les tuiles NF4 et mixtes passent par le chemin scalaire ; TQ3 a ses
    /// kernels SIMD dédiés (AVX2/AVX-512/NEON). En WARM,
    /// le terme binaire est supprimé.
    pub fn compute_score(&self, q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;

    pub fn compute_score_scalar(&self, q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;

    #[cfg(target_arch = "x86_64")]  // # Safety: nécessite la feature CPU correspondante
    pub unsafe fn compute_score_avx2(&self,  q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn compute_score_avx512(&self, q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;
    // (aarch64) compute_score_neon : interne, appelée par le dispatcher.
}
```

> Il n'y a **pas** de constructeur `new` ni de `score_safe`/`Result` sur la
> tuile. Une tuile se construit par littéral de struct (cf. exemple) ou via
> `learned::LearnedModel::encode` / `scenario::build_tile`. Pour passer en WARM à
> la main : `tile.flags |= FLAG_WARM`. La machine à états HOT/WARM/COLD et le
> paging sous budget sont gérés par [`ccos::ElasticKvCache`](#modules-scirust)
> (`page_out` / `evict` / `enforce_budget`), pas par la tuile elle-même.

## Quantizers (`attention::slha_v2`)

```rust
// INT4 uniforme signé, une échelle globale. value ≈ (nibble−8)·scale.
pub fn quantize_latent(v: &[f32; 128]) -> ([u8; 64], f32);

// INT4 « micro-scaling » : une échelle par groupe de 16 dims (dans group_scales).
pub fn quantize_latent_grouped(v: &[f32; 128]) -> ([u8; 64], f32, [u8; 8]);

// NF4 (codebook normal) par groupe — même tuile, flag FLAG_NF4 requis au scoring.
pub fn quantize_latent_nf4(v: &[f32; 128]) -> ([u8; 64], f32, [u8; 8]);

// Mixte 8/4 bits : 8 dims @8-bit (échelle gs[0]) + 112 dims @4-bit, queue lâchée —
// même 64 o, flag FLAG_MIXED requis. Suppose un latent ordonné par variance (PCA).
pub fn quantize_latent_mixed(v: &[f32; 128]) -> ([u8; 64], f32, [u8; 8]);

// TQ3 (port TurboQuant) : 48 o de codes 3 bits (grille symétrique 8 niveaux,
// sans niveau zéro) + 16 o de signes de correction 1 bit (±0,25 pas) — même 64 o,
// flag FLAG_TQ3 requis. Voir docs/TURBOQUANT.md (mesures et trade-offs).
pub fn quantize_latent_tq3(v: &[f32; 128]) -> ([u8; 64], f32, [u8; 8]);
```

`LatentCodec { Int4Single, Int4Grouped, Nf4, Mixed, Tq3 }` sélectionne le codec
via `LearnedModel::encode_with(key, pos, warm, codec)`.

## `safety` — Filtre de sécurité géométrique latent

Classifieur ultra-léger opérant **directement sur les vecteurs latents compressés**
(`[u8; 64]`, 128 dims INT4) sans déquantification complète, pour détecter les anomalies
géométriques typiques des injections de prompts / jailbreaks / dérives sémantiques
**avant la phase de décompression**. Module **additif** : n'altère ni la tuile de 128 o,
ni les kernels SIMD.

```rust
use scirust::safety::{LatentSafetyGuard, SafetyResult, SafetyReason};

// `reference` calibrée sur un corpus de prompts normaux (normalisée à l'unité).
let mut guard = LatentSafetyGuard::new([1.0f32; 128], 0.5);
// ou avec classifieur linéaire entraîné (signal 2) :
//   LatentSafetyGuard::with_linear_classifier(reference, weights, bias, 0.15);

let latent_kv: [u8; 64] = /* tuile compressée */;
match guard.analyze(&latent_kv) {
    SafetyResult::Safe => { /* décompresser, générer le token */ }
    SafetyResult::Anomalous { deviation, reason } => { /* bloquer avant décompression */ }
}
```

**Trois signaux testés dans l'ordre** (le premier qui déclenche retourne son anomalie) :

1. **Déviation angulaire** (`DotProductDeviation`) — cosinus vs vecteur directeur de
   référence < `dot_threshold`. Magnitude invariant (normalisé par la norme du vecteur
   analysé). Un vecteur nul (norme indéfinie) est rangé ici.
2. **Isolation orthogonale** (`OrthogonalIsolation`) — score du classifieur linéaire
   `dot(weights, v)/‖v‖ + bias` < `orthogonal_threshold`. Optionnel (activé via
   `with_linear_classifier`).
3. **Dérive sémantique** (`ActivationDrift`) — moyenne glissante du cosinus sur une
   fenêtre de `DRIFT_WINDOW` (=4) échantillons < `drift_threshold`. N'est évaluée qu'une
   fois la fenêtre pleine (évite les faux positifs au démarrage). La bande
   `[dot_threshold, drift_threshold[` capture des vecteurs *individuellement plausibles
   mais collectivement dérivants*.

| Méthode | Rôle |
|---|---|
| `new(reference, dot_threshold)` | Guard avec référence + seuil cosinus (défaut ≈ cos 60°) |
| `with_linear_classifier(reference, weights, bias, orthogonal_threshold)` | Ajoute le signal 2 |
| `analyze(&[u8; 64])` | Analyse une tuile compressée (décode les nibbles INT4, point zéro 8) |
| `analyze_dequantized(&[f32; 128])` | Analyse un vecteur déquantisé |
| `last_cosine()` | Dernier cosinus mesuré (1.0 = alignement parfait) |

Coût : ~200 cycles/tuile (produit scalaire sur 128 dims), zéro allocation. Fonctionne
sur toutes les architectures (x86_64, aarch64, RISC-V…). Seuils et fenêtre sont des
constances internes ajustables à la compilation.

## `numa` — Allocation alignée + politique NUMA + épinglage de thread

Deux niveaux d'API. Le premier est **toujours disponible, portable, zéro
dépendance** ; le second est **optionnel** (feature `numa`, Linux + `libc`).

### `AlignedBuffer` — allocation heap alignée (portable, par défaut)

Aligne un buffer sur une ligne de cache (128 o par défaut, ou alignement
configurable) via l'allocateur global `std::alloc`. Utile pour aligner les buffers
chauds du chemin SIMD indépendamment du NUMA. Fonctionne sur toutes les cibles.

```rust
use scirust::numa::AlignedBuffer;

let mut buf = AlignedBuffer::new_aligned128(4096)?; // align 128, len 4096
buf.zero();
buf.as_mut_slice()[0] = 0xFF;
assert!(buf.is_aligned()); // adresse % 128 == 0
```

| Méthode | Rôle |
|---|---|
| `new(len, align)` / `new_aligned128(len)` | Alloue `len` octets alignés (non initialisé) |
| `zero()` | Remplit de zéros |
| `as_slice()` / `as_mut_slice()` | Accès typé |
| `is_aligned()` / `align()` / `len()` / `is_empty()` | Introspection |

`AlignedBuffer` respecte l'éthique **zéro-dépendance** du crate : il est compilé
dans la configuration par défaut, sans `libc`.

### Feature `numa` — politique NUMA + épinglage (Linux, optionnel)

Activée par `cargo build/test --features numa`. Tire en `libc` comme **dépendance
optionnelle** — la construction par défaut reste **sans dépendance externe**. Hors
Linux ou sans la feature, les fonctions rendent `NumaError::Unavailable` et
`NumaBuffer` n'est pas construisible (repli gracieux).

| Fonction | Rôle |
|---|---|
| `pin_current_thread_to_cpu(cpu)` | Épingle le thread appelant à un cœur (sched_setaffinity) |
| `pin_current_thread_local()` | Épingle au CPU courant → first-touch local. Renvoie le CPU |
| `current_cpu()` / `current_node()` | CPU / nœud NUMA du thread appelant |
| `numa_available()` / `num_nodes()` | `true` si >1 nœud ; nombre de nœuds (sysfs) |
| `migrate_to_local_node(ptr, len)` | `mbind(MPOL_BIND)` best-effort — **exige ptr page-aligné** |
| `NumaBuffer::new_local(len)` | Région `mmap` page-alignée + `mbind` sur le nœud local |

**Intégration recommandée (first-touch).** L'arena KV-cache de `ccos` est un `Vec`
(allocateur global, aligné à 16 o — **pas** page-aligné, donc `mbind` n'est pas
fiable). La stratégie sûre est le **first-touch** : épingler le thread d'inférence à
son CPU local *avant* de remplir l'arena, pour que ses pages atterrissent sur le bon
nœud sans `mbind`. Helper exposé sur le cache :

```rust
use scirust::ccos::ElasticKvCache;

let cache = ElasticKvCache::with_budget(1 << 20);
// À appeler une fois, depuis le thread d'inférence, juste avant le warm-up :
if let Some(cpu) = ElasticKvCache::pin_caller_to_local_numa() {
    // thread épinglé au CPU `cpu` → first-touch placera l'arena sur le nœud local
}
// ... puis insert / warm-up ...
```

`pin_caller_to_local_numa()` rend `None` sans la feature `numa` ou hors Linux (le
cache fonctionne alors correctement, sans garantie de localité). Pour une région
explicitement page-alignée avec `mbind`, utiliser `NumaBuffer::new_local` (chemin
pour buffers ad hoc, hors `Vec` ccos).

**Note Jetson / mémoire unifiée.** Sur une puce à mémoire unifiée (Jetson Thor AGX,
Apple Silicon) le système est mono-socket / mono-nœud : `numa_available()` rend
`false` et l'épinglage reste utile (évite les migrations de thread). Pour le zero-Copy
CPU/GPU, voir Phase 3 (`zero_copy`, à venir) — ce module est purement CPU.

## Features Cargo

Le crate **n'a pas** de features gating la compilation des chemins SIMD : la
sélection est **à l'exécution** (`std::is_x86_feature_detected!`), avec repli
scalaire portable. (Les anciennes features `avx2/popcnt/neon = []` étaient des
no-op trompeuses — supprimées.)

La bibliothèque est **sans dépendance** dans sa configuration par défaut ; `criterion`
n'est qu'une dev-dependency pour `cargo bench`. L'unique feature de compilation est
**`numa`** (voir section `numa` ci-dessus) : elle tire en `libc` (Linux uniquement,
optionnelle) et active la politique NUMA + l'épinglage de thread. `AlignedBuffer`
(allocation alignée portable) est disponible sans la feature.

## Performance (mesurée, banc partagé)

**x86-64 (Xeon) — baseline serveur :**

| Chemin | Débit (1024 tuiles) | Rapport |
|---|---|---|
| Scalaire | ~3,0 M scores/s | 1× |
| AVX2 | ~34–38 M scores/s | ~×11,5 |
| AVX-512 | ~40–42 M scores/s | ~×14,1 |

**AArch64 (Jetson Thor AGX 128, Neoverse-V3AE) — mesuré sur l'appareil :**

| Chemin | Débit | Rapport |
|---|---|---|
| Scalaire | ~3,0 M scores/s | 1× |
| NEON | ~17,1 M scores/s | **~×5,7** |

(Lignes de cache à 64 o à tous les niveaux ; `sve2` présent. **Ratios
indicatifs — ils dépendent du CPU et de l'auto-vectorisation.** Reproductible via
`cargo run --release -p scirust --example platform_report`.)

- **Mémoire :** tuile 128 o/token contre 256 o pour une clé bf16 → **2× moins
  d'octets/token**. Sur un banc Xeon AVX2 cela donne **~2,5× tokens/s** au
  niveau kernel ; sur CPU scalaire le même banc donne ~1,3×. Facteur de bout en
  bout (decode LLM) non mesuré (§7.5).
- **Fidélité :** la sortie d'attention (`softmax·V`) reste à **cosinus
  0,95–0,997** vs FP malgré un score approché (§7.6).

(Voir `SLHAv2.md` §7 pour la méthodologie et les réserves — projections
synthétiques, `perf`/perplexité hors banc.)

## Exemple minimal

```rust
use scirust::attention::slha_v2::{quantize_latent, SciRustSlhaTile, FLAG_HOT};

let mut v = [0.0f32; 128];
for (i, x) in v.iter_mut().enumerate() { *x = ((i as f32) - 64.0) / 16.0; }
let (latent_kv, scale) = quantize_latent(&v);

let tile = SciRustSlhaTile {
    latent_kv,
    residual_bitmap: [0; 4],
    scale,
    dynamic_lambda: 0.5,
    residual_sigma: 0.0,
    token_id: 0, position: 0, head_id: 0,
    flags: FLAG_HOT,
    group_scales: [255; 8], // [255;8] == échelle unique (équiv. INT4 simple)
};

let q_coarse = [0.0f32; 128];
let q_sign = [0u64; 4];
let score = tile.compute_score(&q_coarse, &q_sign); // dispatch SIMD auto
```

Voir aussi `scirust/examples/basic_usage.rs` (exemple exécutable identique).

## Build / test / bench (depuis la racine, workspace)

```sh
cargo test                 # 78 tests scirust (unitaires + intégration + property/fuzz + doctests + calibration λ + CCOS + JSON + audit)
cargo bench                # micro-benchs criterion (scalaire / AVX2 / AVX-512)
cargo run -p scirust --example basic_usage
cargo run --release -p scirust --example platform_report   # kit x86/ARM : features SIMD, cache, débit
cargo run --bin slha-audit                                 # auto-audit → Markdown (--json / --out FILE / --diff PRIOR.json)
cargo build --workspace --all-targets   # compile lib + tests + benches + exemples
```

### Auto-audit (`slha-audit`) et modules `audit` / `json`

Le module **`scirust::audit`** exécute tous les invariants du système (layout de
tuile, équivalence **SIMD ≡ scalaire** *live*, features/cache, fidélité de sortie
vs attention complète, invariant de budget CCOS, déterminisme) et renvoie un
[`json::Json`] structuré, plus un rendu Markdown (`audit::to_markdown`) et un
`audit::diff(prior, current)` (détection de régression). Le binaire
**`slha-audit`** l'expose en ligne de commande ; le module **`scirust::json`**
est un JSON minimal **sans dépendance** (réutilisé par le serveur `slha-mcp`).

```sh
cargo run --bin slha-audit -- --json --out audit.json   # rapport machine
cargo run --bin slha-audit -- --diff audit.json          # diff vs un rapport antérieur (exit ≠ 0 si changement)
```

Sur un appareil ARM (p. ex. Jetson Thor), `platform_report` (ou
`scripts/bench_device.sh`) produit les chiffres NEON réels — voir la section
Performance ci-dessus.

---
*SLHA v2 — référence d'API alignée sur le code (`scirust` v0.2.0).*
