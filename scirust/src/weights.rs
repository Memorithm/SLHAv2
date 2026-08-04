//! Portable weights file for a learned projection.
//!
//! A projection trained once can be saved and reloaded without re-fitting.
//! The format is little-endian, deterministic and zero-dependency.
//!
//! ## Version 1 — format historique
//!
//! ```text
//! [u32 magic = 0x534C4857 ("SLHW")]
//! [u32 version = 1]
//! [u32 d]
//! [u64 seed]
//! [u8 rht][u8 pad ×3]
//! [f32 × (D_C·d)] projection
//! ```
//!
//! ## Version 2 — format courant
//!
//! ```text
//! [u32 magic = 0x534C4857 ("SLHW")]
//! [u32 version = 2]
//! [u32 d]
//! [u64 seed]
//! [u8 rht][u8 pad ×3]
//! [u32 checksum]
//! [f32 × (D_C·d)] projection
//! ```
//!
//! The version-2 checksum is FNV-1a over the common 24-byte header followed by
//! the projection payload. It detects accidental corruption but is not a
//! cryptographic authentication mechanism.

use crate::attention::slha_v2::D_C;
use crate::learned::LearnedModel;
use std::io;

const MAGIC: u32 = 0x534C_4857; // "SLHW"
const VERSION_V1: u32 = 1;
const VERSION: u32 = 2;

/// Common header shared by versions 1 and 2:
/// magic, version, d, seed, rht and three padding bytes.
const COMMON_HEADER: usize = 4 + 4 + 4 + 8 + 4; // 24

/// Version-2 header: common header plus checksum.
const HEADER: usize = COMMON_HEADER + 4; // 28

fn projection_len(d: usize) -> Result<usize, String> {
    D_C.checked_mul(d)
        .ok_or_else(|| "weights: projection element count overflow".to_string())
}

fn payload_bytes(d: usize) -> Result<usize, String> {
    projection_len(d)?
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "weights: projection byte count overflow".to_string())
}

fn expected_file_size(header: usize, d: usize) -> Result<usize, String> {
    header
        .checked_add(payload_bytes(d)?)
        .ok_or_else(|| "weights: file-size calculation overflow".to_string())
}

/// Deterministic FNV-1a checksum for a version-2 weights file.
///
/// The checksum field itself is excluded.
fn weights_checksum(common_header: &[u8], payload: &[u8]) -> u32 {
    const OFFSET: u32 = 0x811C_9DC5;
    const PRIME: u32 = 0x0100_0193;

    let mut hash = OFFSET;

    for &byte in common_header.iter().chain(payload) {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }

    hash
}

/// Serialize a fitted model into the current version-2 format.
///
/// Unlike [`to_bytes`], this function returns validation errors rather than
/// panicking when given an invalid in-memory model.
pub fn try_to_bytes(model: &LearnedModel, seed: u64, rht: bool) -> Result<Vec<u8>, String> {
    if model.d <= D_C {
        return Err(format!(
            "weights: invalid dimension {} (must be greater than D_C={D_C})",
            model.d
        ));
    }

    let d_u32 = u32::try_from(model.d)
        .map_err(|_| format!("weights: dimension {} does not fit in u32", model.d))?;

    let projection = model.projection();
    let expected_elements = projection_len(model.d)?;

    if projection.len() != expected_elements {
        return Err(format!(
            "weights: projection length mismatch (got {}, expected {expected_elements})",
            projection.len()
        ));
    }

    if let Some(index) = projection.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "weights: non-finite projection value at index {index}"
        ));
    }

    let capacity = expected_file_size(HEADER, model.d)?;
    let mut out = Vec::with_capacity(capacity);

    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&d_u32.to_le_bytes());
    out.extend_from_slice(&seed.to_le_bytes());
    out.push(u8::from(rht));
    out.extend_from_slice(&[0u8; 3]);

    // Checksum placeholder.
    out.extend_from_slice(&0u32.to_le_bytes());

    for &value in projection {
        out.extend_from_slice(&value.to_le_bytes());
    }

    debug_assert_eq!(out.len(), capacity);

    let checksum = weights_checksum(&out[..COMMON_HEADER], &out[HEADER..]);
    out[COMMON_HEADER..HEADER].copy_from_slice(&checksum.to_le_bytes());

    Ok(out)
}

/// Serialize a fitted model into the current version-2 format.
///
/// # Panics
///
/// Panics if the model dimension, projection shape or projection values are
/// invalid. Call [`try_to_bytes`] when handling an untrusted model.
pub fn to_bytes(model: &LearnedModel, seed: u64, rht: bool) -> Vec<u8> {
    try_to_bytes(model, seed, rht).expect("weights: cannot serialize invalid learned model")
}

/// Reconstruct a [`LearnedModel`] from version 1 or version 2 weights.
///
/// The loader validates:
///
/// - magic and supported version;
/// - `d > D_C`;
/// - all size arithmetic using checked operations;
/// - exact file length;
/// - canonical RHT boolean and zero padding;
/// - version-2 checksum;
/// - finite projection coefficients.
pub fn from_bytes(bytes: &[u8]) -> Result<LearnedModel, String> {
    if bytes.len() < COMMON_HEADER {
        return Err("weights: truncated header".into());
    }

    let u32_at = |offset: usize| {
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("validated header range"),
        )
    };

    if u32_at(0) != MAGIC {
        return Err("weights: bad magic (not an SLHW file)".into());
    }

    let version = u32_at(4);
    let header = match version {
        VERSION_V1 => COMMON_HEADER,
        VERSION => HEADER,
        other => {
            return Err(format!(
                "weights: unsupported version {other} \
                 (supported: {VERSION_V1} and {VERSION})"
            ));
        }
    };

    if bytes.len() < header {
        return Err("weights: truncated header".into());
    }

    let d = usize::try_from(u32_at(8))
        .map_err(|_| "weights: dimension does not fit usize".to_string())?;

    if d <= D_C {
        return Err(format!(
            "weights: invalid dimension {d} (must be greater than D_C={D_C})"
        ));
    }

    let seed = u64::from_le_bytes(
        bytes[12..20]
            .try_into()
            .expect("validated seed header range"),
    );

    let rht = match bytes[20] {
        0 => false,
        1 => true,
        other => {
            return Err(format!(
                "weights: invalid RHT flag {other} (expected 0 or 1)"
            ));
        }
    };

    if bytes[21..24] != [0, 0, 0] {
        return Err("weights: non-zero reserved header bytes".into());
    }

    let expected = expected_file_size(header, d)?;

    if bytes.len() != expected {
        return Err(format!(
            "weights: size mismatch (got {}, expected {expected})",
            bytes.len()
        ));
    }

    if version == VERSION {
        let stored = u32::from_le_bytes(
            bytes[COMMON_HEADER..HEADER]
                .try_into()
                .expect("validated checksum range"),
        );

        let calculated = weights_checksum(&bytes[..COMMON_HEADER], &bytes[HEADER..]);

        if stored != calculated {
            return Err(format!(
                "weights: checksum mismatch \
                 (stored {stored:#010x}, calculated {calculated:#010x})"
            ));
        }
    }

    let elements = projection_len(d)?;
    let mut projection = Vec::with_capacity(elements);

    for index in 0..elements {
        let offset = header + index * core::mem::size_of::<f32>();

        let value = f32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("validated projection range"),
        );

        if !value.is_finite() {
            return Err(format!(
                "weights: non-finite projection value at index {index}"
            ));
        }

        projection.push(value);
    }

    LearnedModel::try_from_projection_with(projection, d, seed, rht)
        .map_err(|e| format!("weights: {e}"))
}

/// Write a fitted projection to `path`.
///
/// Invalid in-memory models are returned as [`io::ErrorKind::InvalidInput`].
pub fn save(path: &str, model: &LearnedModel, seed: u64, rht: bool) -> io::Result<()> {
    let bytes = try_to_bytes(model, seed, rht)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;

    std::fs::write(path, bytes)
}

/// Maximum accepted `.slhw` file size (256 MiB). A d=1M projection is 512 MB,
/// so this cap is above any sane trained model but well below OOM territory —
/// it rejects hostile or corrupt files before they are read into memory.
const MAX_SLHW_BYTES: u64 = 256 * 1024 * 1024;

/// Load and validate a projection from `path`.
///
/// Rejects non-regular files (FIFOs, device nodes — which would otherwise
/// block the read forever) and files larger than the 256 MiB safety cap
/// before reading, so an untrusted path cannot cause an unbounded allocation.
pub fn load(path: &str) -> Result<LearnedModel, String> {
    let meta =
        std::fs::metadata(path).map_err(|error| format!("weights: cannot stat {path}: {error}"))?;
    if !meta.file_type().is_file() {
        return Err(format!(
            "weights: {path} is not a regular file (refusing to read a FIFO/device)"
        ));
    }
    if meta.len() > MAX_SLHW_BYTES {
        return Err(format!(
            "weights: {path} is {} bytes, above the {MAX_SLHW_BYTES}-byte safety cap",
            meta.len()
        ));
    }

    let bytes =
        std::fs::read(path).map_err(|error| format!("weights: cannot read {path}: {error}"))?;

    from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::gen_keys;

    /// Smallest valid model dimension for tests: LearnedModel requires d > D_C.
    const TEST_D: usize = D_C + 1;

    fn fitted_model(d: usize, seed: u64, rht: bool) -> LearnedModel {
        let keys = gen_keys(1, 300, d, d, 0.9, 0.02);
        LearnedModel::fit_with(&keys, d, seed, false, rht)
    }

    fn version_one_bytes(model: &LearnedModel, seed: u64, rht: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(expected_file_size(COMMON_HEADER, model.d).unwrap());

        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION_V1.to_le_bytes());
        out.extend_from_slice(&(model.d as u32).to_le_bytes());
        out.extend_from_slice(&seed.to_le_bytes());
        out.push(u8::from(rht));
        out.extend_from_slice(&[0u8; 3]);

        for &value in model.projection() {
            out.extend_from_slice(&value.to_le_bytes());
        }

        out
    }

    fn from_bytes_error(bytes: &[u8], context: &str) -> String {
        match from_bytes(bytes) {
            Ok(_) => panic!("{context}"),
            Err(error) => error,
        }
    }

    #[test]
    fn roundtrip_preserves_scores() {
        let (d, seed, rht) = (256usize, 0x0000_ABCD_u64, true);
        let model = fitted_model(d, seed, rht);

        let bytes = to_bytes(&model, seed, rht);

        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), VERSION);

        let reloaded = from_bytes(&bytes).expect("version-2 roundtrip");

        assert_eq!(
            model.projection(),
            reloaded.projection(),
            "projection differs"
        );

        let keys = gen_keys(2, 4, d, d, 0.9, 0.02);
        let q = &keys[0];

        assert_eq!(model.query_coarse(q), reloaded.query_coarse(q));
        assert_eq!(model.sign_bits(q), reloaded.sign_bits(q));

        let coarse = model.query_coarse(q);
        let signs = model.sign_bits(q);

        let score_original = model
            .encode(&keys[1], 1, false)
            .compute_score(&coarse, &signs);

        let score_reloaded = reloaded
            .encode(&keys[1], 1, false)
            .compute_score(&coarse, &signs);

        assert_eq!(score_original, score_reloaded);
    }

    #[test]
    fn version_two_detects_payload_corruption() {
        let (d, seed, rht) = (TEST_D, 0xC0FFEE, false);
        let model = fitted_model(d, seed, rht);
        let mut bytes = to_bytes(&model, seed, rht);

        bytes[HEADER + 17] ^= 0x80;

        let error = from_bytes_error(&bytes, "corrupted version-2 payload must be rejected");

        assert!(
            error.contains("checksum mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn version_two_checksum_covers_header() {
        let (d, seed, rht) = (TEST_D, 0xA55A, false);
        let model = fitted_model(d, seed, rht);
        let mut bytes = to_bytes(&model, seed, rht);

        // Mutate one seed byte while preserving structural validity.
        bytes[12] ^= 0x01;

        let error = from_bytes_error(&bytes, "corrupted version-2 header must be rejected");

        assert!(
            error.contains("checksum mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn legacy_version_one_remains_readable() {
        let (d, seed, rht) = (TEST_D, 0x1E6A_C001, true);
        let model = fitted_model(d, seed, rht);

        let legacy = version_one_bytes(&model, seed, rht);
        let reloaded = from_bytes(&legacy).expect("version-1 compatibility");

        assert_eq!(model.projection(), reloaded.projection());
        assert_eq!(model.d, reloaded.d);
    }

    #[test]
    fn rejects_invalid_dimension_without_panicking() {
        let mut bytes = [0u8; COMMON_HEADER];

        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&VERSION_V1.to_le_bytes());
        bytes[8..12].copy_from_slice(&(D_C as u32).to_le_bytes());

        let error = from_bytes_error(&bytes, "d <= D_C must be rejected");

        assert!(
            error.contains("invalid dimension"),
            "unexpected error: {error}"
        );

        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());

        assert!(
            from_bytes(&bytes).is_err(),
            "hostile dimensions must return an error, not panic"
        );
    }

    #[test]
    fn rejects_invalid_rht_flag_and_reserved_bytes() {
        let (d, seed, rht) = (TEST_D, 7, false);
        let model = fitted_model(d, seed, rht);

        let mut invalid_flag = version_one_bytes(&model, seed, rht);
        invalid_flag[20] = 2;

        let error = from_bytes_error(&invalid_flag, "invalid RHT flag must be rejected");

        assert!(error.contains("invalid RHT flag"));

        let mut invalid_padding = version_one_bytes(&model, seed, rht);
        invalid_padding[22] = 1;

        let error = from_bytes_error(&invalid_padding, "non-zero padding must be rejected");

        assert!(error.contains("reserved"));
    }

    #[test]
    fn rejects_non_finite_projection_values() {
        let (d, seed, rht) = (TEST_D, 9, false);
        let model = fitted_model(d, seed, rht);
        let mut legacy = version_one_bytes(&model, seed, rht);

        legacy[COMMON_HEADER..COMMON_HEADER + 4].copy_from_slice(&f32::NAN.to_le_bytes());

        let error = from_bytes_error(&legacy, "NaN projection coefficient must be rejected");

        assert!(
            error.contains("non-finite projection value"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn serializer_rejects_non_finite_in_memory_model() {
        let d = TEST_D;
        let mut projection = vec![0.0f32; D_C * d];
        projection[7] = f32::INFINITY;

        let model = LearnedModel::from_projection_with(projection, d, 123, false);

        let error = try_to_bytes(&model, 123, false)
            .expect_err("invalid in-memory projection must be rejected");

        assert!(
            error.contains("non-finite projection value"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_malformed() {
        assert!(from_bytes(&[0, 1, 2]).is_err());
        assert!(from_bytes(&[0; 32]).is_err());
    }
}
