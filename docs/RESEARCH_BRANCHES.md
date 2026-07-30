# Branches non fusionnées — inventaire décisionnel

État au 30 juillet 2026 (`master` = `531d957`). But : permettre de trancher
**fusionner / archiver / abandonner** pour chaque branche sans relire les
diffs. Aucune fusion n'est faite ici ; ce document constate.

> **Décisions exécutées (30 juillet 2026, sur instruction du mainteneur).**
> Le rapatriement recommandé est fait : les deux commits de résultats de
> `research/llama-rank-transplant-oracle` (`de14212`, `f0d3930`) sont dans
> l'arbre — l'artefact et sa section README vivent désormais avec le code.
>
> L'archivage/suppression des branches absorbées n'a **pas** pu être exécuté
> par la session (le proxy git refuse les pushs de tags et les suppressions de
> branches) ; il reste à faire côté mainteneur, en sécurité :
>
> - les têtes des six branches absorbées restent joignables via les refs de
>   leurs pull requests (`refs/pull/58..63/head`) même après suppression ;
> - supprimer `research/llama-rank-transplant-oracle` **seulement après** que
>   le rapatriement ci-dessus a atteint la branche principale — avant cela,
>   cette branche reste le seul exemplaire mergé de l'artefact ;
> - `chore/remediation-audit` (absorbée par #66) peut être supprimée de même.
>
> Le tableau ci-dessous est conservé comme constat d'origine.

Méthode : pour chaque branche, `git diff origin/master origin/<branche>`
restreint aux fichiers que la branche ajoute ou modifie (le reste du diff est
du retard sur master, pas du contenu). « Reproductible depuis le dépôt seul »
signifie : re-dériver le résultat sans artefact externe (modèle GGUF, corpus,
transcripts hors dépôt).

## Synthèse

| Branche | Tête | Ce qu'elle établit | Contenu unique vs master | Décision attendue |
|---|---|---|---|---|
| `feat/llama-kv-quality-roundtrip` | `d6d2c2b` (07-27) | Round-trip K-cache + collecte d'activations (jalon 1) | **aucun** (absorbée par une PR antérieure à #58) | archiver/supprimer |
| `claude/compressed-score-quality-gate-qiqbm5` | `ce6be8a` (07-29) | Première mesure stricte du remplacement de score | **aucun** (squash #58) | archiver/supprimer |
| `fix/llama-calibration-nonfinite-rows` | `930bc3d` (07-29) | Rejet fail-closed des lignes de calibration non finies | **aucun** (squash #59) | archiver/supprimer |
| `research/llama-score-gap-layerwise` | `9a8cf2c` (07-29) | Isolation par couche de la dégradation (masques de couche) | **aucun** (squash #60) | archiver/supprimer |
| `research/llama-score-temperature-calibration` | `adda9d1` (07-29) | Infrastructure de calibration de température de score | **aucun** (squash #61) | archiver/supprimer |
| `research/llama-score-temperature-results` | `aa59134` (07-30) | **Réfutation** de l'hypothèse température + correction d'une data race | **aucun** (squash #62) | archiver/supprimer |
| `research/llama-rank-transplant-oracle` | `f0d3930` (07-30) | **65,92 % du déficit = classement des clés ; top-16 = 98,42 % de ce bénéfice ; NO-GO chiffré du critère F** | **`results/rank_transplant_oracle.json` (5 778 lignes) + section README (271 lignes)** — le code des oracles est dans master (#63), les **résultats non** | **fusionner les 2 commits de résultats** (`de14212`, `f0d3930`) — voir ci-dessous |
| `research/llama-topk-preserving-training` | `5e66e62` (07-30) | Chaîne de vérification d'entraînement (parser strict, liaison d'exécution, trainer atomique) | **aucun** (arbre bit-identique à master, squashes #64/#65 ; `5e66e62` n'est qu'un merge de réconciliation) | supprimer (une session active peut la recréer depuis master) |

## Le seul point qui demande une décision réelle

**`research/llama-rank-transplant-oracle` porte le seul résultat scientifique
du dépôt qui ne soit pas sur `master`** : l'artefact
`integration/llama.cpp/results/rank_transplant_oracle.json` et la section
README qui l'interprète. C'est la **source du verdict NO-GO** consigné dans
`docs/SUCCESS_CRITERIA.md` §5 et `FINDINGS.md` §5bis — aujourd'hui, ces
documents de master citent un artefact d'une branche non fusionnée.

Options :

1. **Fusionner** (recommandé) : rapatrier `de14212` + `f0d3930` (artefact +
   README). Le verdict et sa source vivent alors dans le même historique.
   Attention : la section README de la branche décrit des chiffres de débit et
   un protocole que le README de master ne porte pas encore — la fusion est
   textuellement propre (fichiers additifs) mais mérite une relecture.
2. **Archiver en l'état** : garder la branche comme dépôt de preuve. Coût :
   la source du NO-GO reste hors master, et une suppression accidentelle de la
   branche détruirait le seul exemplaire versionné de l'artefact.

## Reproductibilité (honnête, par famille)

- **Code et tests** (toutes branches) : reproductibles — tout est dans le
  dépôt, la CI compile et exécute.
- **Résultats de mesure** (`measurements.json`,
  `layerwise_score_gap.json`, `score_temperature_calibration.json` sur master ;
  `rank_transplant_oracle.json` sur la branche oracle) : **non reproductibles
  depuis le dépôt seul.** Ils exigent le GGUF `Qwen2.5-1.5B-Instruct q8_0`
  (hash épinglé `d7efb072…`), WikiText-2 (hash épinglé), un build llama.cpp
  au tag `b9860`, et ~10⁴ s de calcul. Les scripts (`build_and_roundtrip.sh`,
  `scripts/`) refont le chemin ; les hachages permettent de vérifier qu'on
  mesure bien la même chose. Les transcripts des portes de complétude de
  l'artefact oracle vivaient dans un scratch de conteneur détruit — seuls
  leurs sha256 subsistent (voir `integration/llama.cpp/results/README.md`).
- **Suppression de branche ≠ suppression de preuve** uniquement pour les six
  branches absorbées : leur contenu est dans les squashes #58–#65. Pour la
  branche oracle, supprimer sans fusionner **perd l'artefact**.

## Ce que ce document ne fait pas

Il ne fusionne rien, ne supprime rien, et ne re-note aucun résultat. Les
décisions appartiennent au mainteneur ; chaque ligne ci-dessus donne le
minimum factuel pour les prendre.
