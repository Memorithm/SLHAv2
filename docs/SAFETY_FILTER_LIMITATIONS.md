# Limites du filtre géométrique latent

Statut : composant expérimental, non qualifié comme contrôle de sécurité.

## Ce que le module mesure

`LatentSafetyGuard` applique trois heuristiques à la représentation latente :

- cosinus par rapport à un vecteur de référence ;
- score d’un classifieur linéaire optionnel ;
- moyenne glissante du cosinus sur quatre observations.

Un résultat `Anomalous` signifie seulement qu’un seuil géométrique a été franchi.
Il n’identifie pas la cause, l’intention, une famille d’attaque ou un auteur.

## Non-objectifs et risques connus

Le module n’est pas un détecteur d’injection de prompt validé et ne doit pas être
utilisé comme seule barrière. Les seuils par défaut ne sont pas calibrés sur une
population représentative. Un changement bénin de sujet, langue, format, modèle,
couche ou codec peut produire un faux positif ; une attaque qui reste proche de
la distribution de référence peut produire un faux négatif.

Le signal ne remplace pas l’authentification, l’autorisation, l’isolation,
la validation des outils, les limites d’exécution, le filtrage des sorties ou
la supervision applicative.

## Qualification requise avant toute promotion

Toute affirmation de sécurité doit être liée à un protocole versionné qui fixe :

1. le modèle, le tokenizer, les couches, les codecs et la population d’usage ;
2. une population bénigne séparée de la calibration et du test ;
3. un corpus adversarial documenté, incluant attaques inconnues du calibrage ;
4. les taux de faux positifs et faux négatifs, avec intervalles d’incertitude ;
5. les courbes de calibration et le choix de seuils avant le test final ;
6. les dérives de distribution par langue, domaine, longueur et version ;
7. les contrôles concurrents et l’effet marginal réel du filtre ;
8. l’identité exacte du code, des données, de la configuration et de l’exécution.

Les résultats synthétiques, la vitesse du kernel, le nombre de cycles ou la
détection d’outliers ne suffisent pas à établir une capacité de sécurité.

## Règle opérationnelle actuelle

Le filtre peut servir de télémétrie expérimentale ou de signal additionnel.
Toute action bloquante doit rester décidée par une politique explicite, réversible
et observée, avec une voie de repli sûre. L’étiquette publique reste « filtre
géométrique expérimental » jusqu’à ce que le protocole ci-dessus soit exécuté et
que ses limites soient publiées.
