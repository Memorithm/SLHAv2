//! Tests d'intégration pour le filtre de sécurité géométrique.
//!
//! Ces tests vérifient le comportement du [`LatentSafetyGuard`] sur des cas
//! réalistes : vecteurs normaux, vecteurs attaquants, isolation orthogonale, dérive
//! progressive, et analyse directe sur les tuiles compressées INT4.

use scirust::safety::{LatentSafetyGuard, SafetyReason, SafetyResult};

const D: usize = 128;

/// Direction « tous-un » normalisée.
fn ones_unit() -> [f32; D] {
    let n = (D as f32).sqrt();
    core::array::from_fn(|_| 1.0 / n)
}

/// Direction alternée `±1` normalisée, orthogonale à `ones_unit`.
fn alt_unit() -> [f32; D] {
    let n = (D as f32).sqrt();
    core::array::from_fn(|i| if i % 2 == 0 { 1.0 / n } else { -1.0 / n })
}

/// Vecteur unitaire de cosinus `c` connu avec `ones_unit` (mélange orthonormé).
fn with_cosine(c: f32) -> [f32; D] {
    let s = (1.0 - c * c).sqrt();
    let a = ones_unit();
    let b = alt_unit();
    core::array::from_fn(|i| c * a[i] + s * b[i])
}

#[test]
fn test_guard_accepts_normal_vectors() {
    let reference = ones_unit();
    let mut guard = LatentSafetyGuard::new(reference, 0.5);

    // Vecteur identique à la référence : cosinus 1.0 → Safe.
    assert_eq!(guard.analyze_dequantized(&reference), SafetyResult::Safe);

    // Vecteur proche (cosinus 0.95) → Safe.
    let close = with_cosine(0.95);
    assert_eq!(guard.analyze_dequantized(&close), SafetyResult::Safe);
}

#[test]
fn test_guard_detects_attacker_vector() {
    let reference = ones_unit();
    let mut guard = LatentSafetyGuard::new(reference, 0.5);

    // Vecteur opposé (cosinus -1.0) → bien au-dessous du seuil.
    let attacker = with_cosine(-1.0);
    match guard.analyze_dequantized(&attacker) {
        SafetyResult::Anomalous { reason, deviation } => {
            assert_eq!(reason, SafetyReason::DotProductDeviation);
            assert!(deviation > 0.0);
        }
        SafetyResult::Safe => panic!("le vecteur attaquant aurait dû être détecté"),
    }
}

#[test]
fn test_guard_detects_orthogonal_isolation() {
    // reference = tous-un (signal 1 cosinus 1 → passe) ; weights = alterné ±1.
    // Le vecteur tous-un est orthogonal aux weights → score 0 < 0.15 → isolation.
    let reference = ones_unit();
    let weights = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
    let mut guard = LatentSafetyGuard::with_linear_classifier(reference, weights, 0.0, 0.15);

    match guard.analyze_dequantized(&ones_unit()) {
        SafetyResult::Anomalous { reason, .. } => {
            assert_eq!(reason, SafetyReason::OrthogonalIsolation);
        }
        SafetyResult::Safe => panic!("le vecteur isolé aurait dû être détecté"),
    }
}

#[test]
fn test_drift_detection_over_multiple_analyses() {
    // cosinus 0.6 : passe le signal 1 (≥ 0.5) mais moyenne < seuil de dérive 0.8.
    let drifted = with_cosine(0.6);
    let mut guard = LatentSafetyGuard::new(ones_unit(), 0.5);

    // Tant que la fenêtre n'est pas pleine, on reste Safe.
    for _ in 0..3 {
        assert_eq!(guard.analyze_dequantized(&drifted), SafetyResult::Safe);
    }
    // Au 4e appel, la fenêtre est pleine et la moyenne 0.6 < 0.8 → ActivationDrift.
    match guard.analyze_dequantized(&drifted) {
        SafetyResult::Anomalous { reason, .. } => assert_eq!(reason, SafetyReason::ActivationDrift),
        SafetyResult::Safe => panic!("la dérive aurait dû être détectée"),
    }
}

#[test]
fn test_analyze_on_compressed_latent() {
    let mut guard = LatentSafetyGuard::new([1.0f32; D], 0.5);

    // Vecteur latent « normal » : tous les nibbles à 12 (valeur +4) → aligné avec la
    // référence tous-un (cosinus 1.0) → Safe.
    let normal_latent: [u8; 64] = core::array::from_fn(|_| 0xCC); // high=12, low=12
    match guard.analyze(&normal_latent) {
        SafetyResult::Safe => {}
        result => panic!("vecteur normal compressé aurait dû être accepté : {result:?}"),
    }

    // Vecteur latent « attaquant » : tous les nibbles à 0 (valeur -8) → opposé à la
    // référence (cosinus -1.0) → DotProductDeviation.
    let attacker_latent: [u8; 64] = core::array::from_fn(|_| 0x00);
    match guard.analyze(&attacker_latent) {
        SafetyResult::Anomalous { reason, .. } => {
            assert_eq!(reason, SafetyReason::DotProductDeviation);
        }
        SafetyResult::Safe => panic!("vecteur attaquant compressé aurait dû être détecté"),
    }
}

#[test]
fn test_threshold_sensitivity() {
    let reference = ones_unit();
    // cosinus 0.4 : « borderline ».
    let borderline = with_cosine(0.4);

    // Seuil bas (0.1) : 0.4 ≥ 0.1 et fenêtre non pleine au 1er appel → Safe.
    let mut guard_low = LatentSafetyGuard::new(reference, 0.1);
    assert_eq!(
        guard_low.analyze_dequantized(&borderline),
        SafetyResult::Safe
    );

    // Seuil haut (0.9) : 0.4 < 0.9 → Anomalous (DotProductDeviation, signal 1).
    let mut guard_high = LatentSafetyGuard::new(reference, 0.9);
    match guard_high.analyze_dequantized(&borderline) {
        SafetyResult::Anomalous {
            reason: SafetyReason::DotProductDeviation,
            ..
        } => {}
        other => panic!("attendu DotProductDeviation, obtenu {other:?}"),
    }
}

#[test]
fn test_perfect_match_cosine() {
    let reference = ones_unit();
    let mut guard = LatentSafetyGuard::new(reference, 0.5);

    assert_eq!(guard.analyze_dequantized(&reference), SafetyResult::Safe);
    // cos(0°) = 1.0 = alignement parfait.
    assert!((guard.last_cosine() - 1.0).abs() < 1e-6);
}

#[test]
fn test_analyze_uses_canonical_nibble_order() {
    // Verrouille end-to-end l'ordre des nibbles : la référence est la déquant
    // canonique d'un latent asymétrique, donc alignée avec decode_nibbles canonique
    // (cosinus 1.0 → Safe). Un décodage inversé pair-permuterait le vecteur et le
    // cosinus chuterait sous le seuil → Anomalous.
    use scirust::attention::slha_v2::quantize_latent;
    let raw: [f32; D] = core::array::from_fn(|d| ((d as f32 % 7.0) - 3.0) * 0.5);
    let (packed, scale) = quantize_latent(&raw);
    // Déquant canonique = (nibble-8)*scale, dim paire ← nibble bas, impaire ← haut.
    let reference: [f32; D] = core::array::from_fn(|d| {
        let nib = if d & 1 == 0 {
            packed[d >> 1] & 0x0F
        } else {
            (packed[d >> 1] >> 4) & 0x0F
        };
        (nib as f32 - 8.0) * scale
    });
    let mut guard = LatentSafetyGuard::new(reference, 0.99);
    match guard.analyze(&packed) {
        SafetyResult::Safe => {}
        other => panic!("le guard devrait reconnaître le latent canonique : {other:?}"),
    }
}

#[test]
fn test_analyze_tile_uses_canonical_decoder_for_every_codec() {
    use scirust::attention::slha_v2::{
        quantize_latent, quantize_latent_grouped, quantize_latent_mix3, quantize_latent_mixed,
        quantize_latent_nf4, quantize_latent_tq3, SciRustSlhaTile, FLAG_HOT, FLAG_MIX3, FLAG_MIXED,
        FLAG_NF4, FLAG_TQ3, FLAG_TQ3_NOCORR, N_GROUPS, RESIDUAL_WORDS,
    };

    fn make_tile(
        latent_kv: [u8; 64],
        scale: f32,
        group_scales: [u8; N_GROUPS],
        flags: u16,
    ) -> SciRustSlhaTile {
        SciRustSlhaTile {
            latent_kv,
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags,
            group_scales,
        }
    }

    let raw: [f32; D] = core::array::from_fn(|index| {
        let base = ((index * 17 + 5) % 37) as f32 - 18.0;
        base / (3.0 + (index % 5) as f32)
    });

    let mut cases: Vec<(&str, SciRustSlhaTile)> = Vec::new();

    let (latent, scale) = quantize_latent(&raw);
    cases.push((
        "int4-single",
        make_tile(latent, scale, [255; N_GROUPS], FLAG_HOT),
    ));

    let (latent, scale, group_scales) = quantize_latent_grouped(&raw);
    cases.push((
        "int4-grouped",
        make_tile(latent, scale, group_scales, FLAG_HOT),
    ));

    let (latent, scale, group_scales) = quantize_latent_nf4(&raw);
    cases.push(("nf4", make_tile(latent, scale, group_scales, FLAG_NF4)));

    let (latent, scale, group_scales) = quantize_latent_mixed(&raw);
    cases.push(("mixed", make_tile(latent, scale, group_scales, FLAG_MIXED)));

    let (latent, scale, group_scales) = quantize_latent_tq3(&raw);
    let tq3 = make_tile(latent, scale, group_scales, FLAG_TQ3);
    cases.push(("tq3", tq3));

    let mut tq3_without_correction = tq3;
    tq3_without_correction.flags |= FLAG_TQ3_NOCORR;
    cases.push(("tq3-no-correction", tq3_without_correction));

    let (latent, scale, group_scales) = quantize_latent_mix3(&raw);
    let mix3 = make_tile(latent, scale, group_scales, FLAG_MIX3);
    cases.push(("mix3", mix3));

    let mut mix3_without_correction = mix3;
    mix3_without_correction.flags |= FLAG_TQ3_NOCORR;
    cases.push(("mix3-no-correction", mix3_without_correction));

    for (codec, tile) in cases {
        let canonical = tile.dequant_latent();

        let mut tile_guard = LatentSafetyGuard::new(canonical, 0.999);
        let tile_result = tile_guard.analyze_tile(&tile);

        let mut vector_guard = LatentSafetyGuard::new(canonical, 0.999);
        let vector_result = vector_guard.analyze_dequantized(&canonical);

        assert_eq!(
            tile_result, vector_result,
            "résultat différent du décodeur canonique pour {codec}"
        );

        assert_eq!(
            tile_result,
            SafetyResult::Safe,
            "la tuile alignée devrait être acceptée pour {codec}"
        );

        assert!(
            (tile_guard.last_cosine() - vector_guard.last_cosine()).abs() < 1e-6,
            "cosinus différent du décodeur canonique pour {codec}"
        );

        assert!(
            (tile_guard.last_cosine() - 1.0).abs() < 1e-5,
            "cosinus non unitaire pour {codec}: {}",
            tile_guard.last_cosine()
        );
    }
}

#[test]
fn test_analyze_tile_fails_closed_on_non_finite_scale() {
    use scirust::attention::slha_v2::{
        quantize_latent_grouped, SciRustSlhaTile, FLAG_HOT, RESIDUAL_WORDS,
    };

    let raw: [f32; D] = core::array::from_fn(|index| index as f32 / D as f32 - 0.5);

    let (latent_kv, _scale, group_scales) = quantize_latent_grouped(&raw);

    let tile = SciRustSlhaTile {
        latent_kv,
        residual_bitmap: [0; RESIDUAL_WORDS],
        scale: f32::NAN,
        dynamic_lambda: 0.0,
        residual_sigma: 0.0,
        token_id: 0,
        position: 0,
        head_id: 0,
        flags: FLAG_HOT,
        group_scales,
    };

    let mut guard = LatentSafetyGuard::new([1.0; D], 0.5);

    match guard.analyze_tile(&tile) {
        SafetyResult::Anomalous {
            reason: SafetyReason::DotProductDeviation,
            deviation,
        } => {
            assert!(deviation.is_finite());
            assert!(deviation > 0.0);
        }
        other => panic!("une échelle non finie doit échouer fermée, obtenu {other:?}"),
    }
}
