# SLHA v2: Sub-Low Rank Hybrid Attention Co-Consciente et Linéarisée pour l'Inférence Cohérente aux Limites des Caches L1/L2/L3

**Auteurs :** Forge CHECKUPAUTO

**Statut :** Spécification de Recherche Fondamentale et d'Ingénierie Système — Édition 2026

---

## Résumé

L'inférence locale de grands modèles de langage (LLM) sur des architectures de serveurs denses ou des accélérateurs embarqués se heurte à une contrainte physique immuable : le mur de la bande passante mémoire (Memory-Bandwidth Wall). Le cache KV, en grandissant de manière linéaire avec le contexte, sature les bus d'interconnexion et provoque une sous-utilisation critique des unités de calcul vectoriel.

Nous présentons **SLHA v2** (Sub-Low Rank Hybrid Attention version 2), un mécanisme d'attention asymétrique et élastique conçu pour s'indexer précisément sur la topologie des caches L1, L2, et L3 des processeurs du marché (architectures multi-cœurs x86_64 type Xeon/Epyc et clusters ARM Neoverse/Thor). En fusionnant une compression latente de bas rang et une quantification résiduelle binaire sur 1-bit via l'infrastructure **SciRust**, SLHA v2 vise un "Soft-Paging" de la précision sémantique sans aucune allocation sur le tas, transformant la gestion du contexte en une opération déterministe au bit près, orchestrée par le noyau de système d'exploitation de contexte **CCOS**.

> **Statut (v1).** Ce document est une *spécification*, pas un rapport de résultats. Les cibles chiffrées de performance (§6) sont des **hypothèses non encore mesurées**, et l'implémentation de référence (§5) comporte des limitations explicitement recensées au §5.1.

---

## 1. Introduction & Limites du Matériel

Les pipelines d'attention conventionnels traitent le cache KV comme un flux continu de tenseurs en haute précision (FP16/BF16). Lors des phases de génération (decoding), chaque nouveau jeton exige le rechargement complet de l'historique du cache depuis la mémoire vive principale (DRAM ou VRAM) vers les registres du CPU/GPU.

Sur un processeur d'infrastructure type serveur multi-cœurs ou système sur puce (SoC) embarqué avancé :

- **Cache L1 (Data) :** Ultra-rapide (~1 cycle d'horloge) mais confiné à une capacité stricte (souvent 32 à 64 Ko par cœur).
- **Cache L2 :** Intermédiaire (~4 à 15 cycles), oscillant entre 512 Ko et 2 Mo par cœur.
- **Cache L3 / LLC (Last Level Cache) :** Partagé, offrant une bande passante élevée mais une latence accrue (~40 à 80 cycles), avec des tailles allant de 32 Mo à plusieurs centaines de Mo.

Si la topologie du cache KV ne s'aligne pas chirurgicalement sur ces strates de silicium, le processeur passe la quasi-totalité de son temps à attendre des transferts de données (Cache Misses). SLHA v2 résout ce problème en adaptant la structure même des tenseurs d'attention à la géométrie interne de ces mémoires caches.

---

## 2. Formalisation Mathématique de SLHA v2

SLHA v2 divise la représentation des Clés (K) et Valeurs (V) en deux composantes asymétriques distinctes : une base sémantique compacte et un résidu de correction binaire haute fréquence.

```
                  ┌──────────────────────────────────────────────┐
                  │          Vecteur d'Activation X_j            │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                ┌──────────────────────────────────────────────────┐
                │  Projection de Bas Rang Partagée (W_down)         │
                └────────────────────────┬─────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │  Espace Latent h_KV (Base Basse-Fidélité)│
                    └────────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼ (Soustraction de la Reconstruction)          ▼ (Projection Orthogonale Z)
    ┌──────────────────────────┐                    ┌──────────────────────────┐
    │  Erreur de Compression   │                    │  Quantification 1-Bit    │
    │        E_j^(n)           │                    │     Residual Bitmap      │
    └──────────────────────────┘                    └──────────────────────────┘
```

### 2.1 Compression Latente Multi-Têtes (MLC)

Soit **X**\_j ∈ ℝ^{d_model} le vecteur d'activation du jeton contextuel j. Nous projetons ce vecteur dans un espace latent partagé à goulot d'étranglement de dimension d_c :

```
h_KV,j = W_down · X_j ∈ ℝ^{d_c}
```

À partir de ce vecteur h_KV,j, chaque tête d'attention n extrait sa composante de clé grossière (coarse key) via une matrice d'up-projection spécifique W_up,K^(n) ∈ ℝ^{d_k × d_c} :

```
K_coarse,j^(n) = W_up,K^(n) · h_KV,j
```

### 2.2 Quantification Résiduelle 1-Bit Déterministe

Pour compenser la dérive sémantique induite par la réduction de rang sans impacter le bus mémoire, l'erreur de reconstruction **E**\_j^(n) = K_real,j^(n) - K_coarse,j^(n) est capturée par une projection aléatoire orthogonale de Johnson-Lindenstrauss. On extrait uniquement le signe du résidu sur d_s dimensions :

```
B_j^(n) = sign( Z · E_j^(n) ) ∈ {-1, 1}^{d_s}
```

Où **Z** ∈ ℝ^{d_s × d_k} est une matrice de projection fixe, stable et pseudo-aléatoire initialisée au démarrage du noyau. En mémoire, B_j^(n) est stocké sous forme de masques de bits compacts (bitmaps).

### 2.3 Équation de Score Flottant-Binaire Fusionnée

Le score d'attention non normalisé entre la requête Q_i^(n) et le jeton du cache j combine le produit scalaire continu et le produit scalaire binaire, accéléré par des opérations logiques de bas niveau :

```
Score_ij^(n) = (Q_i^(n) · W_up,K^(n)) · h_KV,j
              + λ · [ d_s - 2 · popcount( pack_sign(Q_i^(n) · Z^T) ⊕ B_j^(n) ) ]
```

Où ⊕ représente l'opérateur de OU exclusif (XOR) et popcount compte le nombre de bits à 1.

---

## 3. Topologie Matérielle et Élasticité des Caches

L'innovation de la version v2 réside dans l'agencement mémoire en **Tuiles Statiques (Hardware-Aware Tiling)** orchestrées par SciRust. Plutôt que de stocker les tenseurs dans des matrices globales déconnectées de la réalité du matériel, les structures de données adoptent des tailles modulaires calquées sur les lignes de cache de 64 octets.

### 3.1 Alignement Géométrique des Tuiles

Pour maximiser la localité spatiale et temporelle, nous définissons une structure `SciRustSlhaTile` alignée **par défaut sur 64 octets** (`align(64)`, portée à `align(128)` par `build.rs` sur un hôte natif à ligne de 128 o — cf. §3.1), dont l'empreinte mémoire est un multiple exact de la ligne de cache de 64 octets :

| Élément de la Tuile SLHA v2 | Format / Spécification | Taille Mémoire | Cible Matérielle Principale |
|---|---|---|---|
| Composante Latente (d_c = 128) | INT4 Quantifié (4 bits / échantillon) | 64 Octets | Cache L1 (Data) — Ligne complète |
| Bitmaps de Résidus (d_s = 256) | 4 mots binaires de 64 bits (u64) | 32 Octets | Registres Vectoriels AVX-512 / ARM Neon |
| Métadonnées (échelle, λ, σ_E, token, position, tête, flags, réserve) | 3×f32 + 2×u32 + 2×u16 + 8 o réservés | 32 Octets | Pagination CCOS & contrôle d'amplitude |

**Invariant Matériel (implémenté & vérifié) :** Les trois blocs totalisent **exactement 128 octets** — latent 64 o + résidu 32 o + métadonnées 32 o. Avec `#[repr(C, align(64))]`, `size_of::<SciRustSlhaTile>()` vaut **128 octets sans aucun padding** (`align_of = 64`), soit **2 lignes de cache pleines**, alignées sur 64 o donc sans chevauchement d'une 3ᵉ ligne. C'est l'alignement correct pour **les deux cibles** : x86-64 **et** AArch64/Neoverse-V3AE — ce dernier **mesuré à 64 o** sur tous les niveaux (L1d/L1i/L2) d'un **Jetson Thor AGX 128** (le « 128 » désigne les 128 Go de **mémoire unifiée CPU/GPU** LPDDR5X, **pas** la ligne de cache). *Réserve corrigée :* on avait d'abord supposé une ligne de 128 o sur le Thor et appliqué `align(128)` sur `aarch64` ; la mesure de l'appareil l'a réfuté, donc on est revenu à `align(64)` **par défaut** (un `align(128)` ne sert que sur les rares puces à ligne de 128 o, p. ex. Apple Silicon : `build.rs` **détecte désormais** la ligne réelle de l'hôte sur une *build native* — `sysfs` Linux ou `sysctl` macOS — et n'active `align(128)` que là, via `cfg(cache_line_128)` ; jamais comme hypothèse « AArch64 ⇒ 128 ». En cross-compilation la ligne de l'hôte est sans rapport avec la cible, donc le défaut 64 o est conservé). Garanti par le test `tile_is_exactly_128_bytes_zero_padding` (somme des champs == `size_of` == 128 ; `align_of == 64` par défaut, 128 sur un hôte natif à ligne de 128 o) et par la cross-compilation `aarch64-unknown-linux-gnu`. *Historique :* l'énoncé v1 « exactement 104 octets / multiple d'une ligne de cache » était faux — 104 n'est pas un multiple de 64, et l'`align(64)` arrondit la taille à 128. Les 24 octets qui n'étaient alors que du remplissage portent désormais des métadonnées utiles (σ_E, identifiants, drapeaux de pagination).

Par ailleurs, le **bloc latent INT4 occupe à lui seul exactement 64 octets**, soit **une** ligne de cache pleine pour 128 dimensions sémantiques : c'est ce sous-bloc — et non la tuile entière — qui sature l'unité arithmétique en un unique chargement de ligne. Lors d'un calcul d'attention, SciRust pré-charge les vecteurs de requêtes dans le cache L1, puis balaie les tuiles de contexte séquentiellement.

Le même budget de 64 octets admet **cinq codecs latents** interchangeables, sélectionnés par les drapeaux de la tuile (`FLAG_NF4`, `FLAG_MIXED`, `FLAG_TQ3`, `FLAG_MIX3` ; sinon INT4 uniforme, à échelle simple ou par groupe) : INT4, NF4 (§7.8), mixte 8/4 bits (8 dims à 8 bits + 112 à 4 bits, queue lâchée), **TQ3** — le port du codec KV de TurboQuant, 128 codes de 3 bits + un plan séparable de correction de signe 1 bit (§7.12, [`docs/TURBOQUANT.md`](docs/TURBOQUANT.md)) — et **MIX3**, la synthèse tête-mixte-8-bit × corps-TQ3 (plan séparable de 14 o, au niveau du mixte sur activations réelles ; `docs/TURBOQUANT.md` §3ter). L'invariant tuile 128 o est préservé dans tous les cas ; les **cinq codecs disposent de kernels SIMD dédiés** (AVX2/AVX-512/NEON, équivalence scalaire ≤ 1e-3 testée par ISA, repli scalaire portable).

> **État :** l'option « zéro gaspillage » est désormais **implémentée** dans le crate `scirust` (les 24 anciens octets de padding portent des métadonnées, test à l'appui). Reste ouverte une variante **SoA** (latent 64 o dans un flux, résidu + métadonnées dans un autre) si l'on veut ramener le balayage à **une seule** ligne de cache par tuile.

### 3.2 Calibration Dynamique du Facteur d'Échelle λ

La constante λ, qui régit le poids de la correction binaire, n'est plus fixe. SciRust évalue de manière analytique la variance de l'erreur de quantification σ_E² à chaque tick de traitement du contexte. En exploitant la trace de la projection, la valeur optimale s'ajuste en temps réel selon la formule :

```
λ = σ_E · √(π / (2 · d_s))
```

Si l'arborescence causale du code (analysée par le parser CCOS) détecte une zone hautement critique (par exemple, une signature de fonction fondamentale), σ_E augmente pour forcer le processeur à sur-pondérer la fidélité binaire.

> **Statut — calibré (§7.9).** La **forme** `λ ∝ σ_E` est **validée empiriquement** : le multiplicateur optimal (moindres carrés vs référence FP) est ~constant sur tout `rho` (`α* ≈ 4,2`). En revanche la **constante** `√(π/(2·d_s)) ≈ 0,078` **sous-pondère** le résidu d'un facteur ~4,2 ; la constante calibrée est **`C_emp ≈ 0,33`** (à `d_s = 256`). Le crate garde la formule analytique comme défaut conservateur ; l'exemple `calibrate_lambda` re-dérive `C` (l'optimum dépend des données/du modèle — cf. §7.8). Détails et tableau au **§7.9**.

---

## 4. Intégration Holistique CCOS : Politique de "Soft-Paging"

Grâce à la décomposition asymétrique de SLHA v2, le noyau CCOS applique le principe **P2 (Boundedness)** non plus par une expulsion binaire (tout ou rien), mais par une élasticité fine de la fidélité en fonction de la hiérarchie des caches matériels.

```
       [ Contexte HOT ]  ──▶  Mémoire Cache L1/L2 active
         (Latent 4-bit + Résidu 1-bit)            128 o
               │
               ▼  Pression mémoire détectée (enforce_budget)
      [ Contexte WARM ]  ──▶  Libération des Bitmaps Résiduels
         (Latent 4-bit uniquement)                 96 o  (−25 %)
               │
               ▼  Éviction causale totale
       [ Contexte COLD ] ──▶  Snapshot chiffré sur disque (EventLog)
```

| Mode | Description | Impact |
|---|---|---|
| **HOT** (Working Set Actif) | Les tuiles SLHA v2 sont complètes. Le calcul fusionné (Bas rang + Résidu) s'exécute à pleine puissance dans les caches L1/L2. La perplexité sémantique est optimale. | Fidélité maximale — 128 o |
| **WARM** (Pagination Élastique) | Lorsque la mémoire sature ou qu'un nœud s'éloigne dans l'arbre causal, CCOS appelle `page_out()`. Au lieu de décharger le nœud sur disque, le noyau remet à zéro les 32 octets de `residual_bitmap` et pose `dynamic_lambda = 0.0` + `FLAG_WARM` : le score bascule sur le terme grossier seul (eq. 2.3). La structure reste en cache L3 / DRAM ; l'empreinte chute sans aucune I/O ni allocation. | Empreinte logique 96 o (−25 %) |
| **COLD** (Archivé) | Le nœud est intégralement purgé du jeu actif (`evict()`) et son slot est recyclé au prochain `insert`. Sa trace transactionnelle resterait préservée de façon immuable dans l'EventLog (non simulé ici). | Persistance chiffrée |

**Gestionnaire de référence (`ccos::ElasticKvCache`).** Le module [`scirust/src/ccos.rs`](scirust/src/ccos.rs) implémente cette politique sur une **arène contiguë** de tuiles. `enforce_budget()` borne l'**empreinte logique** (HOT 128 o / WARM 96 o / COLD 0 o) sous un budget en octets : il page d'abord HOT→WARM selon une [`PageOutPolicy`] (`LowestImpactFirst` — les plus faibles `σ_E` d'abord, là où le résidu 1-bit compte le moins, §7.2 ; ou `OldestFirst` — distance causale), puis, si nécessaire, évince les plus anciens →COLD. Le masquage WARM est **O(1)** (zéro 32 o + un drapeau), sans allocation ; l'éviction recycle le slot via une *free-list*.

L'exemple [`ccos_softpaging`](scirust/examples/ccos_softpaging.rs) (8 192 tuiles, 1 024 Ko en tout-HOT) mesure l'effet sur la **sortie d'attention** :

| Régime | Budget | HOT / WARM / COLD | Empreinte | cos(sortie, tout-HOT) |
|---|---|---|---|---|
| **A** — paging seul | 896 Ko (112 o/tuile) | 4096 / 4096 / 0 | 896 Ko (88 %) | **0,9995** |
| **B** — éviction forcée | 320 Ko (40 o/tuile) | 0 / 3413 / 256 | 319 Ko ≤ budget | — |

Pager **la moitié** des tuiles (les plus faibles `σ_E`) HOT→WARM ne dégrade quasiment pas la sortie (cos ≈ 0,9995) : c'est le bénéfice central du Soft-Paging — borner la mémoire en libérant le résidu là où il pèse le moins, sans I/O ni perte de jeton. Sous forte pression (B), l'éviction →COLD borne le footprint au prix du contexte le plus ancien (qu'un vrai CCOS snapshoterait dans l'EventLog).

---

## 5. Implémentation Logicielle du Micro-Noyau Asymétrique

Le noyau d'inférence SLHA v2 est écrit en Rust (édition 2021) et respecte l'invariant de zéro allocation dans les boucles critiques. Le crate `scirust` **compile et passe ses tests** (`cargo test`, 7 tests verts) ; le listing ci-dessous en est le cœur. C'est une **implémentation de référence _scalaire_, correcte avant d'être rapide** : API sûre (pas de pointeurs bruts, pas d'`unsafe`), sémantique exacte du score fusionné (eq. 2.3). Le chemin SIMD explicite reste à écrire — mais il est désormais *débloqué*, l'ancien `read_volatile` (qui interdisait toute vectorisation) ayant été retiré (cf. §5.1).

Code source complet : [`scirust/src/attention/slha_v2.rs`](scirust/src/attention/slha_v2.rs)

```rust
// Constantes du modèle
pub const D_C: usize = 128;                 // dim latente (INT4) -> 64 octets
pub const D_S: usize = 256;                 // bits de résidu sign-LSH -> 32 octets
pub const LATENT_BYTES: usize = D_C / 2;    // 64
pub const RESIDUAL_WORDS: usize = D_S / 64; // 4
pub const FLAG_HOT: u16 = 0;
pub const FLAG_WARM: u16 = 1 << 0;          // résidu paginé : score = base latente seule

// 128 octets exacts, zéro padding (test à l'appui). align(64) par défaut ;
// align(128) seulement sur un hôte natif à ligne de 128 o (build.rs → cfg).
#[cfg_attr(cache_line_128, repr(C, align(128)))]
#[cfg_attr(not(cache_line_128), repr(C, align(64)))]
#[derive(Clone)]
pub struct SciRustSlhaTile {
    pub latent_kv: [u8; LATENT_BYTES],          // 64  base h_KV en INT4 signé
    pub residual_bitmap: [u64; RESIDUAL_WORDS], // 32  résidu sign-LSH (256 bits)
    pub scale: f32,            //  4  échelle de déquantification INT4
    pub dynamic_lambda: f32,   //  4  poids de correction binaire (eq. 3.2)
    pub residual_sigma: f32,   //  4  σ_E par tuile (recalibrage de λ)
    pub token_id: u32,         //  4
    pub position: u32,         //  4
    pub head_id: u16,          //  2
    pub flags: u16,            //  2  HOT / WARM
    pub _reserved: [u8; 8],    //  8  réserve -> total exact = 128
}

impl SciRustSlhaTile {
    #[inline]
    pub fn is_warm(&self) -> bool { self.flags & FLAG_WARM != 0 }

    /// Déquantification INT4 *signée* (zero-point) : (nibble − 8) · scale.
    #[inline]
    pub fn dequant_at(&self, d: usize) -> f32 {
        let byte = self.latent_kv[d >> 1];
        let nib = if d & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        ((nib as i32) - 8) as f32 * self.scale
    }

    /// Score fusionné (eq. 2.3). API sûre, sans `read_volatile`, boucle
    /// auto-vectorisable. En mode WARM, le terme binaire est ignoré.
    pub fn compute_score(&self, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        // 1. Terme continu bas-fidélité : <q_coarse, dequant(latent)>
        //    (le crate matérialise d'abord le latent via dequant_latent()
        //     pour une boucle SIMD plus propre — équivalent à l'inline ci-dessous)
        let mut coarse = 0.0f32;
        for d in 0..D_C {
            coarse += q_coarse[d] * self.dequant_at(d);
        }
        // 2. WARM : résidu paginé -> base latente seule
        if self.is_warm() {
            return coarse;
        }
        // 3. Correction binaire 1-bit : λ · (d_s − 2·popcount(q_sign ^ B))
        //    popcount(XOR) = distance de Hamming ; d_s − 2·Hamming = produit
        //    scalaire signé des deux vecteurs ±1.
        let mut hamming = 0u32;
        for w in 0..RESIDUAL_WORDS {
            hamming += (q_sign[w] ^ self.residual_bitmap[w]).count_ones();
        }
        let residual_score = D_S as f32 - 2.0 * hamming as f32;
        coarse + self.dynamic_lambda * residual_score
    }
}

/// Quantification INT4 signée, échelle symétrique par tuile.
/// value ≈ (nibble − 8) · scale, avec nibble ∈ [0, 15] -> plage signée [−8, 7].
pub fn quantize_latent(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let nib = (((v[d] / scale).round() as i32).clamp(-8, 7) + 8) as u8 & 0x0F;
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, scale)
}
```

### 5.1 État de l'implémentation de référence

Les limitations de la v1 sont **levées** dans le crate `scirust` (`cargo test` vert) :

- ✅ **`read_volatile` supprimé.** Le chemin chaud lit des `slice`s normaux : LLVM peut de nouveau auto-vectoriser et réordonner. La spécialisation SIMD n'est pas encore écrite, mais elle n'est plus **bloquée**.
- ✅ **INT4 signé (zero-point).** La déquantification est `(nibble − 8)·scale` : la base bas-rang représente désormais des valeurs négatives. Garanti par le test `int4_dequant_round_trips_signed_values`.
- ✅ **API sûre, pas de `target_feature` trompeur.** Plus d'`unsafe`, plus d'import mort, plus de gate `avx2` sans intrinsèque ; `count_ones()` se compile en `POPCNT` quand la cible le supporte, avec repli portable (ARM Neoverse/Thor inclus).
- ✅ **Tuile = 128 o sans padding** et **crate compilable + testé** : **78 tests** scirust (unitaires + intégration + property/fuzz + doctests + calibration λ + CCOS Soft-Paging), dont l'identité de Hamming `d_s − 2·popcount` prouvée contre une référence brute, l'équivalence SIMD ≡ scalaire (fuzz randomisé), la finitude des scores, et la correspondance code ↔ eq. (2.3).

**Avancées récentes & restant :**

- ✅ **Chemins SIMD : AVX2, AVX-512 (x86_64) + NEON (aarch64).** Dispatch runtime (AVX-512 > AVX2 > scalaire), chacun avec un test d'équivalence ≡ scalaire (AVX2 ×11,5, AVX-512 ×14,1, §7.4). NEON **vérifié par cross-compilation** (`aarch64-unknown-linux-gnu`) ; son équivalence runtime tourne sur matériel ARM.
- ✅ **Outillage de durcissement** : tests randomisés **property / fuzz** (équivalence SIMD, finitude des scores, lois du softmax, bornes de déquantification), micro-benchs **criterion**, et **CI** (fmt + clippy `-D warnings` + tests + compilation des benches + cross-compile NEON).
- ✅/◑ **Projection bas-rang apprise** : PCA (§7.3) **et** une projection *task-aware* entraînée par SGD (§7.7) qui bat nettement la PCA sous décalage Q/K (WARM 0,16 → 0,86). Reste l'entraînement **conjoint** avec un vrai modèle.
- ✅ **Codecs latents INT4 (MX), NF4, mixte 8/4 bits et TQ3 (port TurboQuant, §7.12)** (même tuile 128 o). Ils réduisent l'erreur de reconstruction mais le gain end-to-end est marginal ; une **référence INT8 confirme que la quantification n'est pas le goulot** — c'est la projection bas-rang (§7.8).
- ✅ **Filtre de sécurité géométrique latent** (`scirust::safety`, `LatentSafetyGuard`). Classifieur ultra-léger (~200 cycles, zéro allocation) opérant **directement sur les vecteurs latents compressés** (`[u8; 64]`, INT4) avant décompression : trois signaux — déviation angulaire (cosinus vs référence), isolation orthogonale (classifieur linéaire optionnel), dérive glissante (moyenne du cosinus sur une fenêtre pleine) — pour bloquer injections/jailbreaks/dérives avant la génération du token. Module **additif**, n'altère ni la tuile ni les kernels SIMD ; fonctionne sur toutes architectures (cf. `docs/api.md`).
- ✅ **Allocation alignée + politique NUMA + épinglage de thread** (`scirust::numa`). Deux niveaux : (1) `AlignedBuffer` — allocation heap alignée 128 o (ligne de cache), **portable, zéro dépendance**, disponible par défaut sur toutes cibles ; (2) feature optionnelle **`numa`** (Linux + `libc` en dépendance optionnelle — la configuration par défaut reste **sans dépendance externe**) : `NumaBuffer` (région `mmap` page-alignée + `mbind(MPOL_BIND)` vers le nœud local), `pin_current_thread_to_cpu`/`pin_current_thread_local` (`sched_setaffinity`), introspection CPU/nœud via sysfs. Repli gracieux hors Linux / sans la feature (`NumaError::Unavailable`). **Intégration first-touch** exposée sur l'arena KV-cache (`ElasticKvCache::pin_caller_to_local_numa`) : épingler le thread d'inférence à son CPU local avant le warm-up place les pages du `Vec` sur le bon nœud sans `mbind` (le `Vec` n'étant pas page-aligné). Sur Jetson Thor (mémoire unifiée, mono-NUMA), `numa_available()` rend `false` et l'épinglage reste utile (évite les migrations). CI : job dédié `numa-check` (check + clippy + test + doc + cross-check aarch64, tout avec `--features numa`). Phase 2 de la feuille de route d'optimisation matérielle (cf. `docs/api.md`).

---

## 6. Protocole d'Évaluation et Validation Empirique

Pour valider les gains de performance de SLHA v2 par rapport aux structures de graphes et d'attention conventionnelles, trois métriques matérielles strictes doivent être mesurées lors des prochains tests de charge sous Debian 13.

**Les valeurs ci-dessous (≥ 85 %, 3,5×–5×, ΔP < 0,04) sont des cibles de conception / hypothèses, pas des résultats mesurés.** Les §6.1–6.3 décrivent le protocole matériel (cache misses, débit, perplexité) qui reste à exécuter sous Debian 13. En revanche, des **mesures préliminaires sur prototype** (fidélité de l'approximation, HOT vs WARM) existent déjà et sont rapportées au **§7**.

### 6.1 Taux de Cache Misses L1/L2/L3

- **Objectif :** Démontrer que le tuilage statique élimine le phénomène d'attente CPU.
- **Méthodologie :** Instrumentation du binaire via les compteurs de performance matériels du processeur (`perf stat -e L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses`).
- **Cible attendue :** Une baisse de ≥ 85% des lignes de cache invalidées lors du balayage d'un contexte de 32 000 jetons.
- **Statut :** non mesuré dans ce sandbox (`perf` absent, `perf_event_paranoid = 2`). L'effet de cache est montré *indirectement* par la chute de débit avec la taille de contexte (§7.5).

### 6.2 Débit d'Inférence sous Stress Temporel

- **Objectif :** Quantifier le gain en jetons par seconde lors des phases d'écriture intensive des agents de codage.
- **Méthodologie :** Comparaison de la latence du premier jeton (Time to First Token) et du débit continu face à un cache KV FP16 non compressé.
- **Cible attendue (hypothèse) :** Accélération d'un facteur 3,5× à 5× sur les architectures cibles sans accélération matérielle externe dédiée. Le chemin **AVX2** existe désormais et donne ~×13 sur le *seul* calcul de score (§7.4) ; mais le 3,5×–5× de **bout en bout** (vs cache KV FP16, bande passante mémoire incluse) reste à mesurer sous charge réelle.
- **Mesure partielle (§7.5) :** au niveau du kernel, SLHA score **~2,5× plus de tokens/s** qu'une référence bf16, en lisant 2× moins d'octets/token. Le facteur de bout en bout (decode LLM complet) reste à mesurer.

### 6.3 Dérive de la Perplexité en Mode Dégradé (Soft-Paging Validation)

- **Objectif :** Valider que le passage du mode HOT au mode WARM (perte du résidu 1-bit) n'altère pas la capacité sémantique de l'agent à comprendre la structure globale d'un code.
- **Méthodologie :** Mesure de la perplexité du modèle sur les suites de tests CCOS (`benchmark --cycles 10000`) en basculant sélectivement les nœuds amonts en mode WARM.
- **Cible attendue (hypothèse) :** Dérive de la perplexité inférieure à ΔP < 0,04, ce qui *confirmerait* — une fois réellement mesuré — l'efficacité de la distribution sémantique de bas rang.
- **Statut :** non mesurable sans modèle ni jeu de données réels. **Proxy le plus proche : §7.6** — la fidélité de la *sortie* d'attention (cosinus 0,95–0,997) ; la dérive HOT→WARM y est faible, cohérent avec un ΔP attendu petit.

---

## 7. Résultats de Mesure Préliminaires (Prototype SciRust)

Le crate `scirust` inclut deux prototypes reproductibles — `cargo run --example measure --release` (résidu `rho` fixé à la main) et `--example measure_learned` (base bas-rang **apprise par PCA**) — qui mesurent la **qualité de l'approximation** du score SLHA sur données synthétiques.

**Portée & honnêteté méthodologique.** Dans les §7.1–7.2, les projections sont **aléatoires** (Gaussiennes) : `Z` (sign-LSH) l'est par conception, mais `W_down`/`W_up` ne sont pas apprises ; on suppose la base bas-rang *capturée idéalement* et on mesure la machinerie **quantification INT4 + résidu 1-bit + ranking**. On note `rho = ||e|| / ||k_real||` la part d'énergie que la base laisse au résidu. Le **§7.3 lève cette réserve** en apprenant la base par PCA.

### 7.1 Fidélité du cœur binaire (sign-LSH, d_s = 256)

Sur des paires couvrant toute la plage angulaire, l'estimateur 1-bit suit fidèlement le cosinus vrai (Spearman > 0,9, prouvé par test). Mais dans le **régime réaliste** (résidu quasi-orthogonal à la requête), 256 bits ne donnent qu'une résolution **modérée** : Spearman(résidu, cos θ) ≈ **0,67**. C'est une limite réelle — un seul bit par hyperplan résout mal des directions presque orthogonales.

### 7.2 Score complet vs vérité terrain FP (N = 512 jetons, requête fixe)

| rho | HOT Spearman | HOT top-16 | WARM Spearman | WARM top-16 |
|---|---|---|---|---|
| 0,05 | 0,987 | 0,875 | 0,987 | 0,875 |
| 0,10 | 0,983 | 0,750 | 0,982 | 0,750 |
| 0,20 | 0,966 | 0,750 | 0,961 | 0,688 |
| 0,30 | 0,937 | 0,625 | 0,925 | 0,562 |
| 0,50 | 0,844 | 0,500 | 0,811 | 0,438 |
| 0,70 | 0,702 | 0,375 | 0,627 | 0,250 |

**Lecture :**

- **Soft-Paging validé (§4).** À faible `rho` (≤ 0,1), HOT ≈ WARM : libérer le résidu 1-bit est **quasi sans perte** quand la base bas-rang capture l'essentiel — précisément la condition sous laquelle CCOS bascule un nœud en WARM.
- **Le résidu 1-bit aide, modestement.** HOT ≥ WARM partout ; l'apport croît avec `rho` (à 0,5 : +0,03 Spearman, +6 pts de top-16 ; à 0,7 : +0,08 Spearman, +12 pts de top-16). Effet réel, mais pas spectaculaire à 256 bits.
- **Mise en garde.** Quand la base rate beaucoup (`rho` élevé), même HOT ne récupère pas bien le top-k exact (0,375 de recouvrement top-16 à `rho` = 0,7). La fidélité finale dépend donc fortement de la qualité — apprise — de la base bas-rang.

### 7.3 Projection bas-rang apprise par PCA, et quantification par groupe

Pour lever la réserve « base idéale », `measure_learned` **apprend** la projection par PCA (meilleure reconstruction linéaire de rang `D_C`, par Eckart–Young) sur des clés synthétiques à spectre contrôlable (`d_model = 256`, latent 128, `d_s = 256`). `Z` reste aléatoire par conception. Le latent INT4 utilise désormais des **micro-échelles par groupe** (8 groupes de 16 dims, une échelle `u8` chacun, logées dans les 8 octets jadis « reserved » — la tuile reste à 128 o).

| decay | énergie captée | HOT Spearman | HOT top-16 | WARM Spearman | WARM top-16 |
|---|---|---|---|---|---|
| 0,99 | 97,6 % | 0,788 | 0,562 | 0,703 | 0,438 |
| 0,97 | 99,7 % | 0,814 | 0,375 | 0,700 | 0,312 |
| 0,93 | 99,5 % | 0,884 | 0,438 | 0,609 | 0,188 |
| 0,85 | 99,1 % | 0,905 | 0,562 | 0,871 | 0,500 |

**Lecture :**

- **HOT plafonne à 0,79–0,90**, WARM (coarse seul) à ~0,60–0,87, malgré 97–99,7 % d'énergie captée : l'énergie de reconstruction n'est pas la fidélité de *score*.
- **HOT > WARM partout**, parfois nettement (decay 0,93 : **+0,28** de Spearman) : le résidu 1-bit récupère une grande part de ce que le coarse rate.
- **Quantification par groupe (MX) : forte baisse de l'erreur de reconstruction (>2×, prouvé par test), gain end-to-end marginal** (decay 0,93 : HOT 0,879 → 0,884). Le score est dominé par les composantes de forte variance, déjà bien représentées par l'échelle unique.
- **Résultat négatif assumé : whitener le latent DÉGRADE** (0,859 → 0,750 à decay 0,95) — l'échelle INT4 alloue mieux sa résolution non whitenée.
- **Attention à l'interprétation.** On pourrait croire ce plafond dû à l'INT4 du latent ; le **§7.8 le réfute** (INT8 n'améliore pas le WARM). Le vrai facteur limitant du coarse est la **projection bas-rang**, pas la quantification.

### 7.4 Débit (scalaire vs AVX2 / AVX-512 / NEON)

Le kernel dispose de chemins **AVX2** et **AVX-512** (dispatch à l'exécution via `is_x86_feature_detected!`, ordre AVX-512 > AVX2 > scalaire, repli portable), chacun avec un **test d'équivalence** ≡ scalaire. Sur le banc partagé (**x86, Xeon** ; les ratios dépendent du CPU et de l'auto-vectorisation) :

| Chemin | Débit | Rapport |
|---|---|---|
| Scalaire (référence) | ~2,9 M scores/s | 1× |
| AVX2 | ~33,9 M scores/s | **×11,5** |
| AVX-512 (un FMA 16-wide / groupe) | ~41,6 M scores/s | **×14,1** |

Le facteur dépasse le 8×/16× « théorique » car le chemin scalaire payait aussi une déquantification INT4 *branchy* que le SIMD fusionne. AVX-512 n'ajoute que **~+23 %** sur AVX2 : le kernel est court (128 dims) et limité surtout par le dénibblage/chargement, pas par la largeur FMA. À traiter comme un ordre de grandeur sur banc partagé.

**Chemin NEON (AArch64) — désormais mesuré.** Sur un **Jetson Thor AGX 128** (Neoverse-V3AE), via le kit `platform_report` :

| Chemin | Débit | Rapport |
|---|---|---|
| Scalaire (référence) | ~3,0 M scores/s | 1× |
| NEON (dispatché) | ~17,1 M scores/s | **×5,7** |

Toutes les lignes de cache de l'appareil sont à **64 o** (L1d/L1i/L2 ; le « 128 » d'*AGX 128* = 128 Go de mémoire unifiée CPU/GPU, pas la ligne), et **`sve2` est présent** — la cible du chemin scalable de la roadmap (§8). **Statut toolchain SVE2** (vérifié sur `rustc 1.94.1`, juin 2026) : la **détection** runtime `is_aarch64_feature_detected!("sve2")` est *stable*, mais les **intrinsèques** SVE2 (`svdot_s32`…) sont **absentes du `core::arch::aarch64` stable** — nightly-only, comme `std::simd`. La seule voie stable, l'`asm!` écrit à la main, *compile* mais reste **invérifiable sans appareil SVE2** (CI x86 ; la cross-compilation ne type-checke pas la sémantique de l'`asm!`, et chaque autre chemin SIMD est épinglé par un test d'équivalence bit-à-bit). Le repli **NEON + `cnt`** reste donc le chemin **livré et correct** ; SVE2 atterrira derrière la détection de longueur de vecteur quand les intrinsèques se stabiliseront (ou via un `asm!` validé sur appareil). x86 reste la baseline serveur ; les chiffres ARM sont **mesurés sur l'appareil**, non extrapolés. (Reproductible : `cargo run --release -p scirust --example platform_report`, ou `scripts/bench_device.sh`.)

**En cycles** (exemple `cycles`, via `rdtsc` ; TSC = cycles de *référence*, pas cycles cœur) : ~**942** cyc/tuile scalaire, ~**89** AVX2, ~**71** AVX-512. Le balayage du working-set montre cyc/tuile ~plat tant que résident (68→71 de 0,25 à 16 Mo) puis **+~19 % à 128 Mo** — débordement cache visible *indirectement* (les compteurs de cache-miss `perf` restent indisponibles, §6.1).

### 7.5 Trafic mémoire & débit vs une référence bf16 (§6.2, au niveau kernel)

`bench_vs_fp16` compare le scoring d'une tuile SLHA (**128 o/token**) à un produit scalaire sur une clé **bf16** (`d_k·2 = 256` o/token), les deux en AVX2 (le comparatif isole le format mémoire, pas la chance de codegen).

| Contexte (tokens) | empreinte SLHA / bf16 | SLHA | bf16 |
|---|---|---|---|
| 8 192 | 1 / 2 Mo | 42,4 M scores/s | 17,0 M scores/s |
| 65 536 | 8 / 16 Mo | 40,1 | 16,9 |
| 262 144 | 32 / 64 Mo | 35,4 | 14,0 |
| 1 048 576 | 128 / 256 Mo | 29,9 | 13,8 |

**Lecture :**

- **~2,5× plus de tokens/s** pour SLHA sur ce banc (Xeon AVX2), à débit mémoire (GB/s) comparable : lire **2× moins d'octets/token** (+ un dénibblage INT4→f32 plus court que le décodage bf16) se convertit directement en débit. C'est la thèse du « mur de bande passante » vérifiée au niveau du kernel. **Sur CPU scalaire le même banc donne ~1,3×** (le gain dépend de l'auto-vectorisation) ; le facteur de bout en bout (decode LLM complet) reste à mesurer.
- **Le débit décroît quand le contexte grandit** (42→30 M/s pour SLHA) : effet de cache visible *indirectement*, l'empreinte plus petite de SLHA la gardant résidente plus longtemps.
- **Honnêteté :** le LLC fait 260 Mo sur ce banc — on **ne sature pas la DRAM** (GB/s mesurés ≪ pic DRAM) ; le gain vient du volume d'octets et des uops, pas d'une limite DRAM atteinte. Les **compteurs de cache matériels (§6.1) sont indisponibles** ici (`perf` absent, `perf_event_paranoid = 2`). Sur un LLC plus petit, ou en décodage réellement DRAM-bound, l'avantage de SLHA serait plus marqué.

**Proxy de décodage bout-en-bout (`examples/benchmark_decode.rs`).** Au-delà du score, un exemple enchaîne un cycle de décodage d'attention complet — multi-têtes : scores → softmax → agrégation des valeurs `V` (FP32) — pour estimer un ordre de grandeur en *tokens/s*, vs une référence à clés `f32` non compressées. Sur le banc x86 de dev (32 têtes, contexte 8 192) : **SLHA ≈ 1,6× la référence FP**. C'est un **PROXY** (pas un vrai LLM ni de perplexité) : le `V` est FP32 et identique des deux côtés, donc l'écart vient du kernel de score et de l'empreinte des clés (128 o vs 512 o/token) ; le facteur dépend du matériel (calcul vs bande passante). À confirmer sur un modèle réel (§6.3).

### 7.6 Fidélité de la *sortie* d'attention (proxy de perplexité, §6.3)

Le ranking des scores (§7.2/7.3) est un proxy ; ce qu'un modèle consomme réellement est la **sortie** d'attention `out = softmax(QKᵀ/√d)·V`. `attention_fidelity` la mesure : cosinus et erreur L2 relative entre la sortie vraie (FP) et la sortie SLHA (base apprise PCA, `d = 256`, `N = 512`, moyenne sur 64 requêtes, `d_v = 64`).

| decay | énergie captée | HOT cos / relL2 | WARM cos / relL2 |
|---|---|---|---|
| 0,99 | 97,6 % | 0,948 / 0,318 | 0,943 / 0,333 |
| 0,95 | 99,6 % | 0,989 / 0,147 | 0,988 / 0,154 |
| 0,90 | 99,4 % | 0,994 / 0,108 | 0,993 / 0,114 |
| 0,80 | 98,9 % | 0,997 / 0,078 | 0,996 / 0,082 |

**C'est le résultat le plus important du §7.** La sortie d'attention est **bien plus robuste** que le ranking ne le laissait craindre : là où le Spearman des scores plafonnait à 0,79–0,90 (§7.3), le **cosinus de la sortie atteint 0,95–0,997**. Raison : le softmax moyenne les valeurs, donc les erreurs de score entre jetons de poids voisins se compensent. C'est la métrique la plus proche de la perplexité accessible hors LLM, et elle est nettement favorable. HOT ≥ WARM partout (écart faible à ces forts `decay`, où le résidu compte peu).

### 7.7 Projection apprise *task-aware* vs PCA

La PCA choisit les directions de plus forte variance des **clés** — elle ignore la distribution des **requêtes**. Quand Q et K privilégient des sous-espaces différents, une projection entraînée à préserver le **score** `⟨Q,K⟩` (et non la reconstruction des clés) bat la PCA. `train_projection` réalise cette descente de gradient, avec un gradient en **forme close** : `a = PQ`, `b = PK`, `r = ⟨Q,K⟩ − ⟨a,b⟩`, `∂r²/∂P = −2r(b Qᵀ + a Kᵀ)`.

`learn_projection` confronte PCA et projection apprise sur un cas adverse : base de facteurs **orthonormés**, moitié A à forte variance-clé mais faible poids-requête, moitié B l'inverse. La PCA remplit son budget rang-128 avec A et **rate le score, qui vit dans B**. On évalue en **WARM** (coarse seul) pour isoler la projection du résidu.

| Projection | WARM Spearman | HOT Spearman | sortie cosinus |
|---|---|---|---|
| PCA | 0,161 | 0,325 | 0,947 |
| **Apprise** | **0,859** | **0,863** | **0,985** |

(SGD : perte de score **28,8 → 5,1**, descente lisse avec décroissance linéaire du lr.)

**Réponse à la question ouverte : oui, l'apprentissage bat nettement la PCA** (WARM 0,86 vs 0,16) lorsque Q et K diffèrent — la projection task-aware réalloue la capacité bas-rang vers les directions qui comptent pour le score. **Réserves :** données synthétiques à décalage Q/K délibéré ; quand Q et K partagent la même statistique, la PCA est déjà (quasi-)optimale pour le score (rien à gagner) ; et c'est une preuve de concept SGD sur `P` seule, pas un entraînement conjoint avec un vrai modèle.

### 7.8 Codec latent NF4 et référence INT8 : la quantification n'est pas le goulot

Le latent peut être encodé en **NF4** (codebook NormalFloat-4, 16 niveaux aux quantiles d'une gaussienne) au lieu d'INT4 uniforme — **même budget 4 bits, même tuile 128 o**. On ajoute une **référence INT8** (coarse seul, tuile hypothétique 192 o) pour isoler l'effet de la largeur de bits.

| Codec latent (decay 0,93) | HOT Spearman | WARM Spearman |
|---|---|---|
| INT4 uniforme (1 échelle) | 0,879 | 0,603 |
| INT4 groupé (MX) | 0,884 | 0,609 |
| NF4 groupé | 0,885 | 0,610 |
| **INT8** (réf, 2× octets) | — | **0,606** |

**Constat — et correction du §7.3 :** NF4 réduit l'erreur de reconstruction (test `nf4_beats_uniform_int4_on_gaussian_latent`) mais le gain end-to-end est **nul à marginal** (0,884 → 0,885). Surtout, **INT8 ne fait pas mieux qu'INT4 au WARM (~0,61)** : doubler les bits ne lève pas le plafond du terme coarse. **Le facteur limitant n'est donc PAS la quantification du latent, mais la projection bas-rang elle-même** (la part du score que la PCA laisse au résidu). Cela recadre les leviers réels : **(a)** une meilleure projection (§7.7 : apprise, WARM 0,16 → 0,86) et **(b)** le résidu 1-bit (HOT). NF4 reste utile (meilleure reconstruction, sans coût de tuile), mais n'est pas le levier de fidélité de score.

### 7.9 Calibration de λ (dérive ΔP)

`calibrate_lambda` confronte le poids du résidu `λ` à une attention **FP de référence**. En décomposant `score = coarse + λ·r` (où `r = d_s − 2·popcount` est indépendant de `λ`), le multiplicateur optimal sur la `λ` de la formule a une **forme close** (moindres carrés) : `α* = Σ rt·(λr) / Σ (λr)²`, avec `rt = ⟨Q,K⟩ − coarse`.

| rho | α* (LS) | C_emp = α*·C_formule | RMSE @formule | RMSE @opt | Δout @formule |
|---|---|---|---|---|---|
| 0,10 | 4,18 | 0,327 | 1,37 | 1,26 | 0,006 |
| 0,20 | 4,23 | 0,331 | 2,25 | 1,94 | 0,017 |
| 0,30 | 4,24 | 0,332 | 3,29 | 2,79 | 0,036 |
| 0,50 | 4,25 | 0,333 | 5,87 | 4,92 | 0,106 |
| 0,70 | 4,26 | 0,334 | 9,90 | 8,26 | 0,262 |

**Deux conclusions :**

- **La forme `λ ∝ σ_E` est validée.** `α*` est quasi constant sur tout `rho` (4,18–4,26) : le facteur `σ_E` capture bien la dépendance ; le seul degré de liberté restant est la constante.
- **La constante est ~4,2× trop petite.** `√(π/(2·d_s)) ≈ 0,078` sous-pondère le résidu ; la constante **calibrée** est **`C_emp ≈ 0,33`** (`d_s = 256`). L'optimiser réduit le RMSE de score (~15 %) et la dérive de sortie au fort `rho`.

**Figer.** La forme est **figée** (validée) ; la constante calibrée est **documentée et épinglée par test** (`lambda_calibration_is_stable_and_pinned`). Le crate **garde la formule analytique par défaut** : le facteur 4,2 est mesuré sur données *synthétiques* à projections aléatoires, et l'optimum réel dépend du modèle (cf. §7.8). `calibrate_lambda` re-dérive `C` par déploiement ; `C ≈ 0,33` est la valeur recommandée une fois validée sur modèle réel.

### 7.10 Plan d'amélioration — Phase 1 (axes A1, A2)

Le plan d'amélioration complet (12 axes, roadmap) est dans
`docs/SLHAv2_schema_plan.pdf`. Les deux axes critiques de la **Phase 1
(fidélité)** sont implémentés comme modules **additifs** — la tuile 128 o et
les kernels SIMD sont inchangés, les tests historiques restent verts.

**A2 — Incohérence Hadamard (QuIP#/Palu) sur le résidu sign-LSH.** Nouveau
module `scirust::incoherence` (FWHT orthonormée O(d·log d) + transformée
randomisée `H·D`, diagonale ±1), câblée en **opt-in** dans `LearnedModel`
(`fit_with`/`from_projection_with(..., rht = true)`). La RHT est appliquée au
résidu `E` et à la requête `Q` avant le sign-LSH ; **orthogonale ⇒
`⟨RHT·E, RHT·Q⟩ = ⟨E, Q⟩`**, le score fusionné est préservé, seul le résidu
1-bit gagne en résolution. **Mesuré** (`examples/hadamard_incoherence.rs`) :
dans le régime « outlier aveuglant » (direction forte commune + signal unique
structuré — le cas QuIP#), le **cœur binaire** passe de Spearman **0,07 à 0,49
(+0,42)** ; **WARM est préservé bit-exact** (ΔWARM = 0,0000) car la RHT
n'atteint jamais le chemin coarse. **Honnêtement** : sur résidu bien
conditionné, la RHT est neutre à nuisible pour HOT → A2 est *opt-in
conditionnel* (activer si le rapport peak/mean du résidu est élevé), **pas un
défaut**. Cible directe du §7.1 (résidu 0,67 sur directions quasi-orthogonales).

**A1 — Projection bas-rang sur clés PRE-RoPE (ShadowKV).** Nouveau module
`scirust::rope` (rotation RoPE standard par paires, orthogonale, testée) et
API publique `learned::captured_energy_at(train, d, rank)`. **Mesuré
(robuste sur plusieurs seeds)** : RoPE détruit le bas-rang des clés — énergie
captée chute de **~99,5 % à ~92 %** à rang 128 (Δ +7 %), et de **99 % à 68 %**
à rang 32 (Δ +30 %). C'est la **racine mesurée du goulot « projection » du
§7.8**, exactement le mécanisme que ShadowKV (arXiv 2410.21465) identifie.
**Honnêtement** : sur ce factor model synthétique, la levée du *Spearman WARM*
n'est **pas robuste** (légèrement négative, 0/5 seeds) — la queue perdue touche
la magnitude plus que le ranking, et l'erreur de reconstruction pre-RoPE
(rotée par de grands angles) peut manger le gain. Une levée robuste du
plafond WARM nécessite les clés d'un **vrai LLM** (queue lourde, long contexte,
objectif perplexité) — intégration **Phase 3 / A7**, comme le plan
l'anticipait en qualifiant A1 de gain sur vrai modèle. Voir
`examples/pre_rope_projection.rs`.

### 7.11 Plan d'amélioration — Phase 2 (axes A5, A4)

Les deux axes **Haute** priorité de la **Phase 2 (politique de cache)** sont
implémentés comme modules **additifs** — la tuile 128 o et les kernels SIMD
sont inchangés, les 70 tests Phase 1 restent verts.

**A5 — Éviction informée (H2O / StreamingLLM / SnapKV).** L'éviction §4 est
purement causale (plus ancien d'abord) ; or l'attention a des **heavy-hitters**
et des **attention sinks** (les premiers jetons, que toute requête attende —
StreamingLLM) que l'éviction par âge détruit. Nouvelle `ccos::EvictionPolicy`
(`Causal` par défaut, back-compatible, ou `Importance { sink_window }`) : éviction
des tuiles de **plus faible masse d'attention cumulée** (H2O, accumulée par
`ElasticKvCache::observe_scores` à chaque pas de décodage), avec **pinnage des
sinks** (les `sink_window` premiers jetons, par position). `σ_E` garde son rôle
dans la phase de *paging* (HOT→WARM via `PageOutPolicy`) — seule la phase
d'*éviction* change. **Mesuré** (`examples/informed_eviction.rs`, scénario
construit heavy-hitters mi-séquence + sinks pinnés, budget 16/64 tuiles) : le
cosinus de la sortie d'attention passe de **0,13 (Causal) à 0,53 (Importance)**
(+0,41) — l'éviction causale droppait les heavy-hitters ET les sinks ; informée
les préserve. **Honnêtement** : le scénario est construit pour exhiber le
mécanisme (exactement la structure rapportée par H2O/StreamingLLM) ; la
*magnitude* sur un vrai LLM (long contexte, perplexité) est l'axe A7 / Phase 3,
le *signe* — informée dégrade plus gracieusement — est ce que la mesure confirme.

**A4 — Résidu multi-bit / multi-round (QINCo / Reformer) à budget 256 bits
fixé.** L'invariant tuile 128 o tient : on ne change que la façon de *dépenser*
les 256 bits du résidu. Nouveau module `scirust::residual` : `BinaryResidual`
(1-bit généralisé), `QuantResidual` (`b`-bit, `D_S/b` hyperplans, quantificateur
uniforme centré calibré par σ_E), `MultiRoundResidual` (`K` hashes 1-bit de
`D_S/K` bits, moyenne — Reformer). **Mesuré (robuste sur 18 combos
decay×seed — 3 decays × 6 seeds)** : le multi-bit réduit l'**erreur relative L2**
(`rel_l2 = ‖est−true‖/‖true‖`, *pas* une MSE) de l'estimateur du résidu d'un
facteur **> ~3,3× (garanti par le test sur les 18 combos)** — typique ~×4,
jusqu'à **×200+ à haut-ρ (×245 mesuré au pic)** où le sign 1-bit sature — le levier
confirmé du plan (« meilleur HOT à rho élevé »), un gain de **magnitude**.
**Honnêtement** : le gain de *rang* (Spearman) n'est **pas robuste** — le
1-bit×256 (plus d'hyperplans) échantillonne mieux la direction, et le multi-bit
s'effondre sur certaines seeds (jusqu'à négatif) : trade-off magnitude vs
direction, pas une domination. La graduation Soft-Paging **HOT2** (`b`-bit) →
**HOT1** (MSB = sign 1-bit) → **WARM** est un masquage de bits des mêmes 32 o
(O(1), invariant tuile préservé ; le kernel lit le MSB seul en HOT1 —
intégration Phase 3). Voir `examples/multibit_residual.rs`.

### 7.12 Codec latent TQ3 (port TurboQuant : 3 bits + correction de signe 1 bit)

Le codec KV de [TurboQuant](https://github.com/CHECKUPAUTO/TurboQuant) (QJL)
tient **exactement** dans le budget latent de 64 o de la tuile : 128 dims ×
3 bits = 48 o de codes (grille symétrique à 8 niveaux `code − 3,5`, **sans
niveau zéro**) + 128 × 1 bit = 16 o de signes de correction (le bit déplace le
niveau décodé de ±0,25 pas) — implémenté sous `LatentCodec::Tq3` / `FLAG_TQ3`,
avec les mêmes échelles par groupe que les autres codecs et un décodage
**SIMD dédié** (AVX2/AVX-512/NEON, équivalence scalaire ≤ 1e-3 testée ; NF4, mixte et MIX3 ont désormais les leurs aussi). La rotation pré-quantification
de TurboQuant (PolarQuant) n'est **pas** portée : le blanchiment de
`LearnedModel` et la RHT opt-in (A2, §7.10) jouent déjà ce rôle sur le latent
et le résidu respectivement.

**Mesuré** (`offline_validation --codec tq3` et `measure_learned`, synthétique,
graines fixes) : cosinus de la sortie d'attention HOT **0,9979–0,9999** — GO
sur les quatre régimes, au niveau de l'INT4 groupé (0,9981–0,9999) ; Spearman
end-to-end **0,881** vs 0,884 pour l'INT4 groupé. Attendu : 3 + 1 = 4 bits
d'information par dimension, même erreur pire-cas de 0,25 pas. **Trade-off
honnête** : sans niveau zéro, toute valeur proche de 0 paie ≥ 0,25 pas — sur
latent gaussien la MSE vaut ≈ **1,3–1,6×** celle de l'INT4 groupé (testé,
garde-fou ≤ 2×) ; cohérent avec le §7.8 (« la quantification n'est pas le
goulot »), la différence end-to-end est dans le bruit. Ce que TQ3 achète en
échange, qu'aucun codec à nibbles ne peut offrir : ses deux plans sont
**séparables** — lâcher les 16 o de correction dégrade gracieusement vers une
tuile 3 bits pure, un état de pagination CCOS *plus fin* que HOT→WARM
(candidat : HOT → TQ3-sans-correction → WARM → COLD ; non branché dans `ccos`,
suivi en roadmap). Étude complète, portage et bug amont trouvé :
[`docs/TURBOQUANT.md`](docs/TURBOQUANT.md).

**Conclusion partielle.** Le mécanisme est **mathématiquement correct** (tests, dont les équivalences scalaire/AVX2) et **directionnellement validé** : HOT ≥ WARM, Soft-Paging quasi sans perte à faible `rho`, SIMD ×13, **~2,5× de tokens/s vs bf16** (§7.5), **sortie d'attention à cosinus 0,95–0,997** (§7.6), **projection apprise > PCA** sous décalage Q/K (§7.7), et **λ calibrée** (forme `∝σ_E` validée, constante corrigée ~4,2× → `C_emp ≈ 0,33`, §7.9). Le §7.8 corrige une idée reçue : le plafond du *score coarse* tient à la **projection bas-rang**, non à la quantification. Leviers réels : **meilleure projection** et **résidu 1-bit** (correctement pondéré une fois `λ` calibrée) ; pistes restantes : `d_s` plus grand et entraînement conjoint avec un vrai modèle.

---

## 8. Conclusion

SLHA v2 **propose** qu'en mariant la rigueur d'un système d'exploitation gérant sa mémoire au bit près (CCOS) avec des abstractions mathématiques appliquées aux limites physiques du silicium (SciRust), l'inférence locale puisse changer de paradigme : le cache KV cesserait d'être un fardeau monolithique pour devenir une structure fluide, résiliente et consciente de l'architecture qui l'héberge.

**Cette spécification (v1) progresse vers la validation.** Le crate `scirust` est compilable et testé (§5.1), plusieurs bancs reproductibles existent, un chemin **AVX2** (×13, §7.4) accélère le kernel, et les résultats (§7) confirment la correction du mécanisme et la viabilité du Soft-Paging. Enseignement clé, **corrigé par la mesure** : le plafond de fidélité du *score coarse* tient à la **projection bas-rang**, pas à la quantification du latent — NF4 et même une référence INT8 n'y changent rien (§7.8), tandis qu'une projection *apprise* le lève nettement (§7.7) et que le résidu 1-bit fait le reste (sortie d'attention à cosinus 0,95–0,997, §7.6). Côté noyau, les chemins **AVX2, AVX-512 et NEON** sont en place (§7.4). Restent à faire : (1) la validation **matérielle réelle** du §6 (compteurs de cache via `perf`, perplexité d'un vrai modèle) — non faisable dans ce sandbox, amorcée au niveau kernel en §7.5 ; (2) l'**entraînement conjoint** des projections avec un vrai modèle (le §7.7 en établit le principe).

---

*Spécification de Recherche Fondamentale et d'Ingénierie Système — Édition 2026 — Forge CHECKUPAUTO*
