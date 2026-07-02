# SLHA v2 — Rapport de synthèse des findings

Synthèse honnête de ce que l'implémentation de référence (`scirust/`) et ses
mesures ont **réellement** établi. Toutes les valeurs sont reproductibles
(graines fixes) ; détails et tableaux complets dans [`SLHAv2.md`](SLHAv2.md) §7.

> **Cadre.** Mesures sur données **synthétiques**, projections `Z` (sign-LSH)
> **aléatoires** ; sauf au §7.7, la base bas-rang est une PCA (non entraînée
> conjointement à un modèle). Pas de vrai LLM, pas de compteurs `perf` (sandbox).
> Ces résultats valident la **mécanique**, pas (encore) la qualité sur un modèle réel.
> **Exception : le §5 ci-dessous** — premières mesures sur **activations réelles**
> (GPT-2), qui ont corrigé une conclusion synthétique.

## Tableau de bord

| Question | Résultat mesuré | Réf. |
|---|---|---|
| Tuile alignée cache exacte ? | **128 o, 0 padding** (prouvé par test) | §3.1 |
| Identité de Hamming du cœur binaire ? | exacte vs référence brute | §7.1 |
| Soft-Paging HOT→WARM à faible `rho` ? | quasi sans perte (Spearman ~0,98) | §7.2 |
| Le résidu 1-bit aide-t-il ? | **HOT ≥ WARM partout**, parfois +0,28 | §7.2–7.3 |
| Fidélité de la **sortie** d'attention ? | **cosinus 0,95–0,997** vs FP | §7.6 |
| Trafic mémoire vs bf16 ? | **2× moins d'octets/token → ~2,5× tokens/s** (Xeon AVX2 ; ~1,3× sur CPU scalaire) | §7.5 |
| Débit SIMD (vs scalaire) ? | x86 : AVX2 **×11,5**, AVX-512 **×14,1** ; ARM : NEON **×5,7** (Jetson Thor) | §7.4 |
| Projection apprise vs PCA (Q≠K) ? | WARM **0,16 → 0,86** | §7.7 |
| Cache KV élastique sous budget (Soft-Paging) ? | pager **½** des tuiles HOT→WARM : sortie à **cos 0,9995** | §4 |
| **Premier chiffre RÉEL** (GPT-2 c6, held-out) ? | NO-GO **0,834** → **0,966** après 2 correctifs | **§5** |

## 1. Ce qui est validé

- **Le mécanisme est correct et implémentable.** Tuile 128 o sans gaspillage,
  score fusionné conforme à l'éq. (2.3), kernels scalaire/AVX2/AVX-512/NEON
  **prouvés équivalents** (78 tests scirust dont property/fuzz, clippy `-D warnings`, CI).
- **Le « Soft-Paging » tient — et tourne.** À faible énergie résiduelle, libérer
  le résidu 1-bit (WARM) est quasi sans perte ; le résidu redevient utile quand
  la base bas-rang laisse passer de l'énergie. La politique HOT/WARM/COLD est
  désormais implémentée de bout en bout (`ccos::ElasticKvCache`, §4) : sous un
  budget en octets, pager **la moitié** des tuiles (les plus faibles `σ_E`)
  HOT→WARM laisse la **sortie d'attention** à **cos ≈ 0,9995** vs tout-HOT.
- **La sortie d'attention est robuste** — le résultat le plus important. Même
  quand le ranking des scores plafonne (Spearman 0,79–0,90), la sortie
  `softmax·V` reste à **cosinus 0,95–0,997** : le softmax absorbe l'erreur de
  score. C'est le proxy le plus proche de la perplexité accessible hors LLM.
- **L'argument « mur de bande passante » tient au niveau kernel.** 128 o/token
  contre 256 o pour une clé bf16 → **~2,5× plus de tokens/s** à débit GB/s
  comparable (sur Xeon AVX2 ; **~1,3× sur CPU scalaire**).
- **Mesuré sur les deux cibles (x86 + ARM).** Le kit `platform_report` produit
  des chiffres x86 (AVX-512 ~×14 sur Xeon) **et** ARM **mesurés sur un Jetson
  Thor AGX 128** (Neoverse-V3AE) : NEON **~×5,7** vs scalaire. Au passage il a
  **corrigé une fausse hypothèse** : le Thor a des lignes de cache de **64 o**
  (pas 128 — le « 128 » = 128 Go de mémoire unifiée CPU/GPU), d'où le retour à
  `align(64)` **par défaut** (`build.rs` sonde désormais l'hôte et ne passe à
  `align(128)` que sur une vraie ligne de 128 o, p. ex. Apple Silicon — jamais
  comme hypothèse AArch64-wide) ; et **`sve2` est présent** (cible de la
  roadmap §8). *Statut SVE2 vérifié* (`rustc 1.94.1`) : détection runtime
  **stable**, mais intrinsèques SVE2 (`svdot_s32`…) **nightly-only** (absentes
  du `core::arch::aarch64` stable, comme `std::simd`) ; la seule voie stable
  (`asm!` manuel) est **invérifiable sans appareil SVE2** (CI x86) ⇒ on garde
  **NEON + `cnt`** comme chemin livré, mesuré et correct.

## 2. Les leviers réels (et les faux leviers)

- **Levier #1 — la projection bas-rang.** Une projection **apprise task-aware**
  (SGD minimisant l'erreur de *score*, pas la reconstruction) bat nettement la
  PCA quand requêtes et clés diffèrent (WARM 0,16 → **0,86**). La PCA optimise la
  reconstruction des clés et **ignore la distribution des requêtes**.
- **Levier #2 — le résidu 1-bit.** Il récupère une grande part de ce que le
  terme coarse rate (HOT ≫ WARM à `rho` élevé).
- **Faux levier — la largeur de bits du latent** *(sur données synthétiques)*.
  Une **référence INT8** ne fait pas mieux qu'INT4 au terme coarse (~0,61) :
  **la quantification n'est pas le goulot**, c'est la projection. NF4 et le
  groupage MX réduisent l'erreur de reconstruction mais ne déplacent quasiment
  pas le ranking end-to-end. ⚠️ **Ne généralise PAS aux activations réelles** :
  sur GPT-2 le spectre est bien plus raide que `gen_keys` et la quantification
  **devient** le goulot dominant — voir §5 (c'est le codec mixte qui le lève).
- **Largeur SIMD ≠ levier majeur ici.** AVX-512 n'ajoute que **~+23 %** sur AVX2 :
  le kernel (128 dims) est limité par le dénibblage/chargement, pas la largeur FMA.

## 3. Résultats négatifs assumés

- **Whitening du latent : dégrade** (HOT 0,859 → 0,750). L'échelle INT4 unique
  alloue mieux sa résolution non whitenée.
- **Groupage MX / NF4 : gain end-to-end marginal** malgré une meilleure
  reconstruction (le score est dominé par les composantes de forte variance).
- **INT8 : n'élève pas le plafond du coarse** — corrige une hypothèse initiale
  (§7.3) qui attribuait à tort ce plafond à l'INT4.
- **Tuile BiLLM (salient outliers) : reportée sur preuve mesurée.** L'exemple
  `salient_outliers` injecte des canaux outliers : le mécanisme est **réel en
  reconstruction** (RMSE INT4 0,08 → 0,52 à ×32 ; salient-`s` reste plat si
  `s ≥` nb d'outliers), mais (a) la **sortie** sous INT4 reste à cos ≥ 0,977
  même à ×32 (le softmax absorbe), donc le gain end-to-end est **modeste**, et
  (b) le budget tuile (2 valeurs FP) peut **sous-performer** quand les outliers
  sont plus nombreux. → ne vaut les 16 o (pris à σ_E / `group_scales`) que si le
  nb de canaux outliers du modèle cible tient dans le budget. Décision
  **mesurée**, pas supposée.

## 4. Honnêteté & limites

- Le **paper v1** contenait des affirmations fausses (tuile « 104 o » alors que
  `align(64)` ⇒ 128 o ; `read_volatile` et `avx2` contradictoires) — corrigées.
- Une de **mes propres** conclusions (§7.3, « goulot = INT4 ») a été **réfutée**
  par la mesure INT8 et corrigée (§7.8). C'est l'intérêt de mesurer.
- **Non mesurable dans ce sandbox** : compteurs de cache `perf` (§6.1,
  `perf_event_paranoid=2`), perplexité d'un vrai modèle (§6.3), entraînement
  conjoint des projections.

## 5. Première validation sur activations RÉELLES (GPT-2) — NO-GO, diagnostic, correctifs

Le harnais Phase 0 a tourné sur de vraies activations (**GPT-2 couche 6**,
d=768 pleine largeur, corpus train/test **disjoints** de 1024 tokens,
projection **tenue à l'écart** — le protocole de `docs/SUCCESS_CRITERIA.md` §3).

| configuration (held-out, HOT) | cos sortie↑ | KL(ppl)↓ |
|---|---|---|
| INT4 uniforme + PCA-clés (départ) | 0,834 | 0,81 |
| + codec **MIXTE 8/4-bit** (`--codec mixed`) | 0,954 | 0,16 |
| + projection **JOINTE** clés+requêtes (`--joint`) | **0,966** | **0,12** |
| *plafond flottant du sous-espace 128 (mesuré)* | *0,971* | — |

**Diagnostic (chaîne causale mesurée).** Le spectre réel des clés est
pathologiquement raide — **40 % de l'énergie totale dans UNE direction**, 87 %
dans quatre, un rapport **56×** à l'intérieur du premier groupe de scaling.
(1) L'INT4 uniforme (16 niveaux) ne couvre pas cette dynamique : le score
coarse passe de 0,958 (flottant) à 0,834 (quantifié) — corrigé par
`LatentCodec::Mixed` (tête 8 dims @8-bit, corps 112 dims @4-bit, même budget
64 o, invariant 128 o intact). (2) La PCA-clés ne garde que **69,6 %** de
l'énergie des vraies requêtes (le score est `⟨Pq, Pk⟩`) — corrigé par
`fit_joint` (second moment poolé clés+requêtes).

**Écartés par la mesure** (chaque piste chiffrée avant d'être abandonnée) :
RHT 0,834 (nul) ; whitening 0,585 (pire) ; NF4 0,769 (pire) ; centrage des
clés 0,841 (marginal) ; couplage du résidu à la reconstruction quantifiée
0,966 (nul — l'erreur INT4 résiduelle est sous la résolution du 1-bit) ; SGD
score-objectif warm-starté du joint : plafond inchangé (0,971).

**Lecture honnête.** (1) Le pipeline opère à **99,4 % de son plafond** : le
1,4 point restant vers le seuil GO (0,98) est la **troncature de rang 768→128
elle-même** sur ce proxy — aucun codec ni résidu ne peut le combler, et le SGD
ne trouve pas de meilleur sous-espace. (2) Ce proxy (têtes concaténées,
d=768) n'est pas le point de déploiement nominal ; la décision Phase 2 se
prend sur le rang effectif de la distribution cible, pas sur ce seul chiffre.
(3) Une conclusion synthétique a été **réfutée par le réel** (« la largeur de
bits n'est pas le goulot », §2) — c'est le §4 en action : mesurer, corriger,
re-mesurer.

Reproduire :
```text
python scripts/dump_activations.py --model gpt2 --layer 6 --out /tmp/train --file train.txt
python scripts/dump_activations.py --model gpt2 --layer 6 --out /tmp/test  --file test.txt
cargo run --release --example train_on_real_activations -- --dump /tmp/train --joint --out p.slhw
cargo run --release --example offline_validation -- --dump /tmp/test --weights p.slhw --codec mixed
```

**Post-scriptum (2026-07-02) — TQ3 sur le même protocole.** Le codec TQ3
(portage TurboQuant, `docs/TURBOQUANT.md`), au niveau des codecs 4 bits sur
synthétique, a été mesuré sur le même protocole held-out (GPT-2 couche 6,
corpus disjoints de 1024 tokens — WikiText-2 cette fois, d'où de légers écarts
absolus : mixte 0,985 / 0,055 ici vs 0,966 / 0,12 ci-dessus). Projection
jointe tenue à l'écart : **TQ3 cos 0,791 / KL 1,10**, contre INT4 groupé
0,884 / 0,58 et mixte **0,985 / 0,055** ; et la jointe, qui relève l'INT4
groupé (0,758 → 0,884), laisse TQ3 quasi inchangé (0,783 → 0,791). Verdict :
la grille uniforme 8 niveaux sans zéro s'effondre sur le spectre raide réel
comme l'INT4 uniforme — **NO-GO comme codec HOT sur activations réelles** ; le
mixte reste le codec des spectres réels, TQ3 garde son intérêt propre (plans
séparables, pagination) sur distributions plates. Tableau complet (9 configs) :
`docs/TURBOQUANT.md` §3bis.

## 6. Prochaines étapes (hors périmètre sandbox)

1. **Entraîner conjointement** `W_down`/`W_up` avec un vrai modèle (le §7.7 en
   établit le principe sur données synthétiques).
2. **Validation matérielle réelle** : `perf stat` (cache misses) et perplexité
   sur un modèle + jeu de données réels.
3. Intégration dans une vraie pile d'inférence pour mesurer le gain **de bout en
   bout** (et non au seul niveau kernel).
4. **Balayer les couches** (le §5 ne mesure que la couche 6 de GPT-2) : une
   commande par couche via `dump_activations.py --layer N`.

---
*Réf. : crate `scirust/` (78 tests dont property/fuzz + doctests + calibration λ + CCOS, criterion, CI), paper `SLHAv2.md` §1–8.*
