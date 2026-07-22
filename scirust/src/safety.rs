//! Filtre de sécurité géométrique dans l'espace latent (representation engineering).
//!
//! Ce module implémente un classifieur ultra-léger qui opère **directement sur les
//! vecteurs latents compressés** (`[u8; 64]`, soit 128 dims INT4) sans déquantisation,
//! détectant les anomalies géométriques typiques des attaques par injection de prompts,
//! jailbreaks ou dérives sémantiques **avant la phase de décompression**.
//!
//! ## Principe
//!
//! L'espace latent compressé de SLHAv2 est un condensé topologique des activations du
//! modèle. Les attaques complexes provoquent des anomalies de distribution mesurables :
//!
//! 1. **Déviation angulaire** — le vecteur projeté s'écarte du vecteur directeur de
//!    référence (activation « normale »). Mesuré par produit scalaire normalisé
//!    (cosinus), indépendant de la norme du vecteur analysé.
//! 2. **Isolation orthogonale** — le vecteur devient quasi-orthogonal à un classifieur
//!    linéaire calibré (signature d'une injection de prompt structurée créant un
//!    sous-espace disjoint).
//! 3. **Dérive sémantique** — la moyenne glissante du cosinus s'effondre sur une fenêtre
//!    récente : des vecteurs individuellement plausibles mais collectivement dérivants.
//!
//! ## Performance
//!
//! - Zéro allocation dans la boucle chaude (analyse d'une tuile : ~200 cycles).
//! - Opère sur les nibbles INT4 sans déquantification complète (point zéro signé à 8).
//! - Fonctionne sur toutes architectures (x86_64, aarch64, RISC-V…).
//!
//! ## Intégration
//!
//! Ce module est **additif** — il n'altère ni la tuile de 128 octets, ni les kernels
//! SIMD de base. Il s'insère dans le pipeline d'inférence juste après la projection
//! latente et avant la décompression :
//!
//! ```rust,no_run
//! use scirust::safety::{LatentSafetyGuard, SafetyResult};
//!
//! # fn main() {
//! // `reference` serait calibrée sur un corpus de prompts normaux, puis normalisée.
//! let reference = [1.0f32; 128];
//! let mut guard = LatentSafetyGuard::new(reference, 0.5);
//!
//! let latent_kv: [u8; 64] = [0; 64]; // tuile compressée
//! match guard.analyze(&latent_kv) {
//!     SafetyResult::Safe => { /* décompresser et générer le token */ }
//!     SafetyResult::Anomalous { deviation, reason } => {
//!         eprintln!("anomalie détectée: écart={deviation:.2}, raison={reason}");
//!         // bloquer l'inférence avant décompression
//!     }
//! }
//! # }
//! ```

use crate::attention::slha_v2::{D_C, LATENT_BYTES};

// ── Types publics ────────────────────────────────────────────────────────

/// Résultat d'analyse du guard de sécurité.
#[derive(Clone, Copy, Debug)]
pub enum SafetyResult {
    /// Le vecteur latent est dans les normes attendues.
    Safe,
    /// Anomalie géométrique détectée au-delà du seuil configuré.
    Anomalous {
        /// Écart mesuré (positif, exprimé dans l'unité du signal déclencheur).
        deviation: f32,
        /// Raison de l'anomalie.
        reason: SafetyReason,
    },
}

// La comparaison de `deviation` utilise une tolérance et n'est donc pas
// transitive. `SafetyResult` ne doit volontairement pas implémenter `Eq`.
impl PartialEq for SafetyResult {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Safe, Self::Safe) => true,
            (
                Self::Anomalous {
                    deviation: d1,
                    reason: r1,
                },
                Self::Anomalous {
                    deviation: d2,
                    reason: r2,
                },
            ) => r1 == r2 && (d1 - d2).abs() < 1e-6, // tolérance f32
            _ => false,
        }
    }
}

/// Classification de la nature de l'anomalie détectée.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafetyReason {
    /// Le cosinus avec le vecteur directeur est trop faible — le vecteur s'écarte de la
    /// distribution normale (un vecteur nul est aussi rangé ici : norme indéfinie).
    DotProductDeviation,
    /// Le score du classifieur linéaire est trop faible — signature d'une injection de
    /// prompt structurée créant un sous-espace disjoint.
    OrthogonalIsolation,
    /// La moyenne glissante du cosigne s'effondre sur la fenêtre récente (activation
    /// drift) : vecteurs individuellement plausibles mais collectivement dérivants.
    ActivationDrift,
}

impl core::fmt::Display for SafetyReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DotProductDeviation => write!(f, "dot-product-deviation"),
            Self::OrthogonalIsolation => write!(f, "orthogonal-isolation"),
            Self::ActivationDrift => write!(f, "activation-drift"),
        }
    }
}

// ── Constantes internes ─────────────────────────────────────────────────

/// Seuil par défaut du cosinus (signal 1) : cos(60°) ≈ 0.5. En dessous, le vecteur est
/// jugé hors distribution. Surchageable via [`LatentSafetyGuard::new`].
const DEFAULT_DOT_THRESHOLD: f32 = 0.5;

/// Seuil par défaut du score linéaire (signal 2) : un score normalisé inférieur à cette
/// valeur signale une isolation orthogonale.
const DEFAULT_ORTHOGONAL_THRESHOLD: f32 = 0.15;

/// Seuil par défaut de la dérive (signal 3) : si la moyenne glissante du cosigne sur la
/// fenêtre passe sous ce seuil, on signale une dérive. La bande `[dot_threshold,
/// drift_threshold[` correspond à des vecteurs individuellement plausibles mais
/// collectivement dérivants.
const DEFAULT_DRIFT_THRESHOLD: f32 = 0.8;

/// Taille de la fenêtre glissante pour le signal 3 (en échantillons).
const DRIFT_WINDOW: usize = 4;

// ── Filtre principal ────────────────────────────────────────────────────

/// Filtre de sécurité géométrique dans l'espace latent.
///
/// Analyse les vecteurs latents compressés directement (sans déquantisation) pour
/// détecter des anomalies typiques des attaques par injection de prompts ou des
/// dérives sémantiques, **avant la phase de décompression et de génération du token**.
///
/// ## Conception
///
/// Le guard combine trois signaux indépendants :
/// 1. **Cosinus vs référence** — détecte les vecteurs hors distribution (magnitude
///    invariante grâce à la normalisation par la norme du vecteur analysé).
/// 2. **Classifieur linéaire** — détecte les sous-espaces disjoints (injections). Optionnel ;
///    activé via [`LatentSafetyGuard::with_linear_classifier`].
/// 3. **Dérive glissante** — détecte les glissements progressifs sur une fenêtre de
///    `DRIFT_WINDOW` échantillons. N'est évaluée qu'une fois la fenêtre pleine, pour
///    éviter les faux positifs pendant la phase de remplissage.
///
/// Chaque signal est testé dans l'ordre ; le premier qui déclenche retourne son anomalie.
/// Le coût est d'environ 200 cycles par analyse (produit scalaire sur 128 dims).
pub struct LatentSafetyGuard {
    /// Vecteur directeur de référence (128 dims), normalisé à l'unité lors de la
    /// construction.
    reference: [f32; D_C],

    /// Seuil de cosinus pour le signal 1.
    dot_threshold: f32,

    /// Poids du classifieur linéaire (signal 2), normalisés à l'unité. `None` = signal 2
    /// désactivé.
    weights: Option<[f32; D_C]>,

    /// Bias du classifieur linéaire.
    linear_bias: f32,

    /// Seuil du score linéaire pour le signal 2.
    orthogonal_threshold: f32,

    /// Seuil de la moyenne glissante pour le signal 3.
    drift_threshold: f32,

    /// Historique circulaire des cosinus récents (signal 3).
    drift_window: [f32; DRIFT_WINDOW],
    drift_idx: usize,
    /// Nombre d'échantillons écrits dans la fenêtre (plafonne à [`DRIFT_WINDOW`]).
    /// Permet de ne juger la dérive qu'une fois la fenêtre pleine.
    drift_count: usize,

    /// Dernier cosinus mesuré (1.0 = alignement parfait avec la référence).
    last_cosine: f32,
}

impl LatentSafetyGuard {
    /// Crée un guard avec un vecteur de référence et un seuil de cosinus.
    ///
    /// `reference` est normalisé à l'unité (si sa norme est non nulle). `dot_threshold`
    /// est le cosinus minimum acceptable entre le vecteur analysé et la référence ;
    /// on utilise `DEFAULT_DOT_THRESHOLD` (≈cos 60°) si la valeur passée n'est pas
    /// finie strictement positive.
    ///
    /// ## Exemple
    /// ```rust
    /// use scirust::safety::LatentSafetyGuard;
    /// let guard = LatentSafetyGuard::new([1.0f32; 128], 0.5);
    /// ```
    pub fn new(reference: [f32; D_C], dot_threshold: f32) -> Self {
        let dot_threshold = if dot_threshold.is_finite() && dot_threshold > 0.0 {
            dot_threshold
        } else {
            DEFAULT_DOT_THRESHOLD
        };
        Self {
            reference: Self::normalize(reference),
            dot_threshold,
            weights: None,
            linear_bias: 0.0,
            orthogonal_threshold: DEFAULT_ORTHOGONAL_THRESHOLD,
            drift_threshold: DEFAULT_DRIFT_THRESHOLD,
            drift_window: [0.0; DRIFT_WINDOW],
            drift_idx: 0,
            drift_count: 0,
            last_cosine: 0.0,
        }
    }

    /// Crée un guard avec un classifieur linéaire complet (poids + bias) pour le signal 2.
    ///
    /// `reference` pilote le signal 1 (cosinus), `weights`/`bias` pilote le signal 2
    /// (score linéaire). `orthogonal_threshold` est le score linéaire minimum ; en
    /// dessous, on signale une isolation orthogonale. Le signal 1 utilise
    /// `DEFAULT_DOT_THRESHOLD`.
    ///
    /// Utile pour un entraînement préalable sur un corpus de prompts normaux vs
    /// attaquants.
    pub fn with_linear_classifier(
        reference: [f32; D_C],
        weights: [f32; D_C],
        bias: f32,
        orthogonal_threshold: f32,
    ) -> Self {
        let orthogonal_threshold = if orthogonal_threshold.is_finite() {
            orthogonal_threshold
        } else {
            DEFAULT_ORTHOGONAL_THRESHOLD
        };
        let mut guard = Self::new(reference, DEFAULT_DOT_THRESHOLD);
        guard.weights = Some(Self::normalize(weights));
        guard.linear_bias = bias;
        guard.orthogonal_threshold = orthogonal_threshold;
        guard
    }

    /// Analyse un vecteur latent compressé et retourne `Safe` ou `Anomalous`.
    ///
    /// Zéro allocation. Opère directement sur les nibbles INT4 sans déquantification
    /// complète : chaque byte contient deux valeurs INT4 (nibble haut + nibble bas),
    /// décodées avec un point zéro signé à 8 (le neutre du quantizer SLHAv2).
    ///
    /// ## Complexité
    /// - Temps : O(D_C) = O(128) opérations flottantes
    /// - Mémoire : O(1) — aucune allocation
    /// - Cycles estimés : ~200 sur ARM NEON, ~150 sur x86 AVX2
    pub fn analyze(&mut self, latent_kv: &[u8; LATENT_BYTES]) -> SafetyResult {
        let decoded = Self::decode_nibbles(latent_kv);
        self.analyze_dequantized(&decoded)
    }

    /// Analyse un vecteur déquantisé complet (128 dims f32).
    ///
    /// Teste les trois signaux dans l'ordre : déviation angulaire, isolation orthogonale
    /// (si un classifieur est configuré), puis dérive glissante (une fois la fenêtre
    /// pleine). Retourne `Safe` uniquement si aucun signal ne déclenche.
    pub fn analyze_dequantized(&mut self, v: &[f32; D_C]) -> SafetyResult {
        let norm_v = Self::norm(v);

        // Vecteur nul ou non fini (NaN/Inf dans v → norme NaN/Inf) : norme indéfinie →
        // rangé hors distribution. Évite qu'un NaN ne passe toutes les comparaisons (`NaN
        // < seuil` est faux) et ne court-circuite silencieusement le filtre en Safe.
        if norm_v == 0.0 || !norm_v.is_finite() {
            self.last_cosine = 0.0;
            return SafetyResult::Anomalous {
                deviation: self.dot_threshold,
                reason: SafetyReason::DotProductDeviation,
            };
        }

        // Signal 1 : cosinus vs référence (référence unitaire → dot / norm_v).
        let cosine = Self::dot_product(&self.reference, v) / norm_v;

        // Une référence mal configurée (NaN/Inf) ne doit jamais permettre un
        // contournement du filtre. Toute valeur non finie est rejetée fail-safe.
        if !cosine.is_finite() {
            self.last_cosine = 0.0;
            return SafetyResult::Anomalous {
                deviation: self.dot_threshold.abs().max(f32::EPSILON),
                reason: SafetyReason::DotProductDeviation,
            };
        }

        self.last_cosine = cosine;
        if cosine < self.dot_threshold {
            return SafetyResult::Anomalous {
                deviation: self.dot_threshold - cosine,
                reason: SafetyReason::DotProductDeviation,
            };
        }

        // Signal 2 : classifieur linéaire (poids unitaires → magnitude invariant).
        if let Some(ref weights) = self.weights {
            let linear_score = Self::dot_product(weights, v) / norm_v + self.linear_bias;

            // Des poids ou un biais non finis représentent une configuration
            // invalide : le filtre doit échouer fermé, jamais retourner Safe.
            if !linear_score.is_finite() || linear_score < self.orthogonal_threshold {
                let deviation = if linear_score.is_finite() {
                    self.orthogonal_threshold - linear_score
                } else {
                    self.orthogonal_threshold.abs().max(f32::EPSILON)
                };

                return SafetyResult::Anomalous {
                    deviation,
                    reason: SafetyReason::OrthogonalIsolation,
                };
            }
        }

        // Signal 3 : dérive de la moyenne glissante (fenêtre pleine seulement).
        self.drift_window[self.drift_idx] = cosine;
        self.drift_idx = (self.drift_idx + 1) % DRIFT_WINDOW;
        if self.drift_count < DRIFT_WINDOW {
            self.drift_count += 1;
        }
        if self.drift_count >= DRIFT_WINDOW {
            let mean = Self::window_mean(&self.drift_window);

            if !mean.is_finite() || mean < self.drift_threshold {
                let deviation = if mean.is_finite() {
                    self.drift_threshold - mean
                } else {
                    self.drift_threshold.abs().max(f32::EPSILON)
                };

                return SafetyResult::Anomalous {
                    deviation,
                    reason: SafetyReason::ActivationDrift,
                };
            }
        }

        SafetyResult::Safe
    }

    /// Retourne le dernier cosinus mesuré (1.0 = alignement parfait avec la référence,
    /// 0.0 = orthogonal, < 0.0 = opposé). Utile pour les logs/telemetry.
    pub fn last_cosine(&self) -> f32 {
        self.last_cosine
    }

    // ── Primitives internes ────────────────────────────────────────────

    /// Décode les 64 bytes latents en 128 valeurs f32 (nibble - 8).
    ///
    /// Layout SLHAv2 canonique (cohérent avec `attention::slha_v2::{quantize_latent,
    /// dequant_at}`) : chaque byte contient deux nibbles INT4 — le **bas** (bits 3:0)
    /// encode la dimension **paire** `2i`, le **haut** (bits 7:4) encode la dimension
    /// **impaire** `2i+1`. Le point zéro signé est à 8 (le neutre du quantizer).
    #[inline(always)]
    fn decode_nibbles(latent_kv: &[u8; LATENT_BYTES]) -> [f32; D_C] {
        let mut out = [0.0f32; D_C];
        for i in 0..LATENT_BYTES {
            let low = (latent_kv[i] & 0x0F) as f32 - 8.0; // dim paire 2i
            let high = ((latent_kv[i] >> 4) & 0x0F) as f32 - 8.0; // dim impaire 2i+1
            out[i * 2] = low;
            out[i * 2 + 1] = high;
        }
        out
    }

    /// Produit scalaire de deux vecteurs 128-dims.
    #[inline(always)]
    fn dot_product(a: &[f32; D_C], b: &[f32; D_C]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..D_C {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Norme L2 d'un vecteur 128-dims.
    #[inline(always)]
    fn norm(v: &[f32; D_C]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalise un vecteur à l'unité ; si la norme est nulle, retourne le vecteur tel
    /// quel (laissé à zéro).
    #[inline(always)]
    fn normalize(v: [f32; D_C]) -> [f32; D_C] {
        let n = Self::norm(&v);
        if n > 0.0 {
            core::array::from_fn(|i| v[i] / n)
        } else {
            v
        }
    }

    /// Moyenne des éléments de la fenêtre circulaire.
    #[inline(always)]
    fn window_mean(window: &[f32; DRIFT_WINDOW]) -> f32 {
        let sum: f32 = window.iter().sum();
        sum / DRIFT_WINDOW as f32
    }
}

// ── Tests unitaires ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Direction « tous-un » normalisée : `[1.0; 128] / sqrt(128)`.
    fn ones_unit() -> [f32; D_C] {
        let n = (D_C as f32).sqrt();
        core::array::from_fn(|_| 1.0 / n)
    }

    /// Direction alternée `±1` normalisée, orthogonale à `ones_unit`.
    fn alt_unit() -> [f32; D_C] {
        let n = (D_C as f32).sqrt();
        core::array::from_fn(|i| if i % 2 == 0 { 1.0 / n } else { -1.0 / n })
    }

    /// Vecteur unitaire de cosinus `c` connu avec `ones_unit`, obtenu en mélangeant
    /// `ones_unit` (c) et `alt_unit` (sqrt(1-c²)), qui sont orthonormés.
    fn vector_with_cosine(c: f32) -> [f32; D_C] {
        let s = (1.0 - c * c).sqrt();
        let a = ones_unit();
        let b = alt_unit();
        core::array::from_fn(|i| c * a[i] + s * b[i])
    }

    #[test]
    fn test_decode_nibbles_roundtrip() {
        let mut latent = [0u8; LATENT_BYTES];
        // byte 0 = 0xA5 : nibble haut = 10 (→ +2), nibble bas = 5 (→ -3).
        // Convention canonique SLHAv2 : dim paire ← nibble bas, dim impaire ← nibble haut.
        latent[0] = (10 << 4) | 5;
        let decoded = LatentSafetyGuard::decode_nibbles(&latent);
        assert!(
            (decoded[0] - (-3.0)).abs() < 1e-6,
            "dim paire ← bas : {}",
            decoded[0]
        );
        assert!(
            (decoded[1] - 2.0).abs() < 1e-6,
            "dim impaire ← haut : {}",
            decoded[1]
        );
    }

    #[test]
    fn test_decode_nibbles_matches_canonical_quantizer() {
        // Verrouille l'ordre des nibbles contre le quantizer canonique du crate
        // (`quantize_latent` : dim paire → nibble bas, dim impaire → nibble haut).
        use crate::attention::slha_v2::quantize_latent;
        let v: [f32; D_C] = core::array::from_fn(|d| ((d as f32 % 7.0) - 3.0) * 0.5);
        let (packed, _scale) = quantize_latent(&v);
        let decoded = LatentSafetyGuard::decode_nibbles(&packed);
        for d in 0..D_C {
            let nib = if d & 1 == 0 {
                packed[d >> 1] & 0x0F
            } else {
                (packed[d >> 1] >> 4) & 0x0F
            };
            assert!(
                (decoded[d] - (nib as f32 - 8.0)).abs() < 1e-6,
                "dim {d}: decoded={} attendu={}",
                decoded[d],
                nib as f32 - 8.0
            );
        }
    }

    #[test]
    fn test_safe_for_aligned_vector() {
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        // Identique à la référence → cosinus 1.0 ≥ 0.5 ; fenêtre non pleine → Safe.
        assert_eq!(guard.analyze_dequantized(&ones_unit()), SafetyResult::Safe);
        assert!((guard.last_cosine() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_anomalous_for_orthogonal_vector() {
        // `ones_unit` est orthogonal à `alt_unit` → cosinus 0 < 0.9.
        let mut guard = LatentSafetyGuard::new(alt_unit(), 0.9);
        match guard.analyze_dequantized(&ones_unit()) {
            SafetyResult::Anomalous {
                reason: SafetyReason::DotProductDeviation,
                deviation,
            } => assert!(deviation > 0.0),
            other => panic!("attendu DotProductDeviation, obtenu {other:?}"),
        }
    }

    #[test]
    fn test_zero_vector_is_anomalous() {
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        match guard.analyze_dequantized(&[0.0f32; D_C]) {
            SafetyResult::Anomalous {
                reason: SafetyReason::DotProductDeviation,
                ..
            } => {}
            other => panic!("attendu DotProductDeviation, obtenu {other:?}"),
        }
    }

    #[test]
    fn test_non_finite_reference_fails_closed() {
        let mut reference = ones_unit();
        reference[0] = f32::NAN;

        let mut guard = LatentSafetyGuard::new(reference, 0.5);

        match guard.analyze_dequantized(&ones_unit()) {
            SafetyResult::Anomalous {
                reason: SafetyReason::DotProductDeviation,
                deviation,
            } => assert!(deviation.is_finite() && deviation > 0.0),
            other => panic!("une référence NaN doit échouer fermée, obtenu {other:?}"),
        }

        assert_eq!(guard.last_cosine(), 0.0);
    }

    #[test]
    fn test_non_finite_linear_classifier_fails_closed() {
        let mut weights = ones_unit();
        weights[0] = f32::NAN;

        let mut bad_weights = LatentSafetyGuard::with_linear_classifier(
            ones_unit(),
            weights,
            0.0,
            DEFAULT_ORTHOGONAL_THRESHOLD,
        );

        match bad_weights.analyze_dequantized(&ones_unit()) {
            SafetyResult::Anomalous {
                reason: SafetyReason::OrthogonalIsolation,
                deviation,
            } => assert!(deviation.is_finite() && deviation > 0.0),
            other => panic!("des poids NaN doivent échouer fermés, obtenu {other:?}"),
        }

        let mut bad_bias = LatentSafetyGuard::with_linear_classifier(
            ones_unit(),
            ones_unit(),
            f32::INFINITY,
            DEFAULT_ORTHOGONAL_THRESHOLD,
        );

        match bad_bias.analyze_dequantized(&ones_unit()) {
            SafetyResult::Anomalous {
                reason: SafetyReason::OrthogonalIsolation,
                deviation,
            } => assert!(deviation.is_finite() && deviation > 0.0),
            other => panic!("un biais infini doit échouer fermé, obtenu {other:?}"),
        }
    }

    #[test]
    fn test_non_finite_input_is_anomalous() {
        // Un NaN dans v rend la norme NaN ; sans garde, `NaN < seuil` est faux et le
        // filtre tomberait à Safe (bypass). On exige Anomalous.
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        let mut v = ones_unit();
        v[0] = f32::NAN;
        match guard.analyze_dequantized(&v) {
            SafetyResult::Anomalous {
                reason: SafetyReason::DotProductDeviation,
                ..
            } => {}
            other => panic!("attendu DotProductDeviation pour entrée NaN, obtenu {other:?}"),
        }
        // Idem avec +inf.
        let mut w = ones_unit();
        w[0] = f32::INFINITY;
        match guard.analyze_dequantized(&w) {
            SafetyResult::Anomalous {
                reason: SafetyReason::DotProductDeviation,
                ..
            } => {}
            other => panic!("attendu DotProductDeviation pour entrée Inf, obtenu {other:?}"),
        }
    }

    #[test]
    fn test_drift_detection_when_mean_drops() {
        // cosinus 0.6 : passe le signal 1 (≥ 0.5) mais sous le seuil de dérive 0.8.
        let v = vector_with_cosine(0.6);
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        // Fenêtre non pleine : Safe.
        for k in 0..DRIFT_WINDOW - 1 {
            assert_eq!(
                guard.analyze_dequantized(&v),
                SafetyResult::Safe,
                "étape {k}"
            );
        }
        // Fenêtre pleine, moyenne 0.6 < 0.8 → ActivationDrift.
        match guard.analyze_dequantized(&v) {
            SafetyResult::Anomalous {
                reason: SafetyReason::ActivationDrift,
                ..
            } => {}
            other => panic!("attendu ActivationDrift, obtenu {other:?}"),
        }
    }

    #[test]
    fn test_no_drift_for_steady_normal_stream() {
        // Flux constant d'alignements parfaits → moyenne 1.0 ≥ 0.8, jamais de dérive.
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        for _ in 0..10 {
            assert_eq!(guard.analyze_dequantized(&ones_unit()), SafetyResult::Safe);
        }
    }

    #[test]
    fn test_drift_averaging_and_recovery() {
        // Discrimine la vraie moyenne glissante d'un bug « dernier échantillon » :
        // 3 cosinus 0.6 puis 1 cosinus 1.0 → moyenne 0,70 < 0,8 (dérive) alors que le
        // dernier échantillon (1,0) ≥ 0,8 : un bug dernier-échantillon / max ne
        // déclencherait pas. Vérifie aussi la levée après repeuplement de la fenêtre.
        let v_mid = vector_with_cosine(0.6);
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        for _ in 0..3 {
            assert_eq!(guard.analyze_dequantized(&v_mid), SafetyResult::Safe);
        }
        match guard.analyze_dequantized(&ones_unit()) {
            SafetyResult::Anomalous {
                reason: SafetyReason::ActivationDrift,
                ..
            } => {}
            other => panic!("attendu ActivationDrift, obtenu {other:?}"),
        }
        // Levée : 4 alignements parfaits → la moyenne remonte à 1,0 ≥ 0,8 → Safe.
        for _ in 0..DRIFT_WINDOW {
            assert_eq!(guard.analyze_dequantized(&ones_unit()), SafetyResult::Safe);
        }
    }

    #[test]
    fn test_no_allocation_in_analyze() {
        // Vérification heuristique : analyze sur un buffer empilé ne panique pas.
        let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);
        let latent: [u8; LATENT_BYTES] = core::array::from_fn(|i| (i % 16) as u8);
        let _ = guard.analyze(&latent);
    }

    #[test]
    fn test_display_safety_reason() {
        assert_eq!(
            format!("{}", SafetyReason::DotProductDeviation),
            "dot-product-deviation"
        );
        assert_eq!(
            format!("{}", SafetyReason::OrthogonalIsolation),
            "orthogonal-isolation"
        );
        assert_eq!(
            format!("{}", SafetyReason::ActivationDrift),
            "activation-drift"
        );
    }

    #[test]
    fn test_linear_classifier_isolation() {
        // reference = tous-un ; weights = alterné ±1. Le vecteur tous-un est aligné à
        // la référence (cosinus 1) mais orthogonal aux poids (score 0) → isolation.
        let weights = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
        let mut guard = LatentSafetyGuard::with_linear_classifier(ones_unit(), weights, 0.0, 0.15);
        match guard.analyze_dequantized(&ones_unit()) {
            SafetyResult::Anomalous {
                reason: SafetyReason::OrthogonalIsolation,
                ..
            } => {}
            other => panic!("attendu OrthogonalIsolation, obtenu {other:?}"),
        }
    }

    #[test]
    fn test_linear_classifier_accepts_aligned() {
        // reference = weights = direction alternée. v = même direction → cosinus 1 et
        // score linéaire élevé → Safe.
        let dir = alt_unit();
        let weights = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
        let mut guard = LatentSafetyGuard::with_linear_classifier(dir, weights, 0.0, 0.15);
        assert_eq!(guard.analyze_dequantized(&dir), SafetyResult::Safe);
    }

    #[test]
    fn test_invalid_threshold_falls_back_to_default() {
        // -1.0 n'est pas > 0.0 → repli sur DEFAULT_DOT_THRESHOLD (0.5). Discrimine :
        //   avec repli (seuil 0.5), cosinus 0.4 < 0.5 → Anomalous ;
        //   si le bug gardait -1.0, 0.4 >= -1.0 → Safe.
        let mut guard = LatentSafetyGuard::new(ones_unit(), -1.0);
        match guard.analyze_dequantized(&vector_with_cosine(0.4)) {
            SafetyResult::Anomalous {
                reason: SafetyReason::DotProductDeviation,
                ..
            } => {}
            other => panic!("attendu Anomalous (repli 0.5), obtenu {other:?}"),
        }
    }
}
