// These are numerical kernels: range loops and fused signatures are clearer.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub use scirust::attention::slha_v2::{
    FLAG_HOT, FLAG_MIX3, FLAG_MIXED, FLAG_NF4, FLAG_TQ3, FLAG_TQ3_NOCORR, FLAG_WARM, GROUP_DIM,
    MIX3_CODES_OFF, MIX3_CODE_BYTES, MIX3_CORR_BYTES, MIX3_CORR_OFF, MIXED_DIMS, MIXED_HI_DIMS,
    MIXED_LO_DIMS, MIXED_LO_GROUPS, NF4_CODEBOOK, N_GROUPS, TQ3_CODE_BYTES, TQ3_CORRECTION,
    TQ3_CORR_BYTES, TQ3_HALF_RANGE,
};
pub use scirust::{D_C, D_S, LATENT_BYTES, RESIDUAL_WORDS};

pub const TILE_BYTES: usize = 128;

pub const HOT_BYTES: usize = 128;
pub const WARM_BYTES: usize = HOT_BYTES - RESIDUAL_WORDS * 8;

pub const N_GROUP_SCALES: usize = N_GROUPS;

pub const RESIDUAL_OFFSET: usize = LATENT_BYTES;
pub const SCALE_OFFSET: usize = 96;
pub const DYNAMIC_LAMBDA_OFFSET: usize = 100;
pub const RESIDUAL_SIGMA_OFFSET: usize = 104;
pub const TOKEN_ID_OFFSET: usize = 108;
pub const POSITION_OFFSET: usize = 112;
pub const HEAD_ID_OFFSET: usize = 116;
pub const FLAGS_OFFSET: usize = 118;
pub const GROUP_SCALES_OFFSET: usize = 120;

pub fn has_flag(flags: u16, mask: u16) -> bool {
    flags & mask == mask
}

pub fn ext_nibble(byte: u8, hi: bool) -> u8 {
    if hi {
        byte >> 4
    } else {
        byte & 0x0F
    }
}

pub fn decode_int4_nibble(nib: u8) -> f32 {
    (nib as i32 - 8) as f32
}

pub fn effective_scale(scale: f32, group_scale: u8) -> f32 {
    scale * (group_scale as f32) * (1.0 / 255.0)
}

pub fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn u64_slice_to_le_bytes(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn le_bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, &'static str> {
    if !bytes.len().is_multiple_of(4) {
        return Err("byte length not divisible by 4 for f32 conversion");
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

pub fn le_bytes_to_u64_vec(bytes: &[u8]) -> Result<Vec<u64>, &'static str> {
    if !bytes.len().is_multiple_of(8) {
        return Err("byte length not divisible by 8 for u64 conversion");
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| u64::from_le_bytes(*chunk))
        .collect())
}

pub fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32, &'static str> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or("f32 read out of bounds")?;
    <[u8; 4]>::try_from(slice)
        .map(f32::from_le_bytes)
        .map_err(|_| "invalid f32 bytes")
}

pub fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16, &'static str> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or("u16 read out of bounds")?;
    <[u8; 2]>::try_from(slice)
        .map(u16::from_le_bytes)
        .map_err(|_| "invalid u16 bytes")
}

pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Bytes of a physically packed WARM tile: the full tile minus the
/// 32-byte residual plane (which is zeroed and flagged `FLAG_WARM`).
pub const WARM_PACKED_BYTES: usize = TILE_BYTES - RESIDUAL_WORDS * 8; // 96

/// Pack a WARM tile into its physically smaller 96-byte form.
///
/// The residual bytes are semantically absent (the tile must carry
/// `FLAG_WARM`); packing copies everything except the 32 residual bytes.
/// The result is NOT a valid `SerializedTile` byte-for-byte; it is the
/// WARM residency representation.
pub fn pack_warm(tile: &[u8; TILE_BYTES]) -> [u8; WARM_PACKED_BYTES] {
    let mut out = [0u8; WARM_PACKED_BYTES];
    out[..RESIDUAL_OFFSET].copy_from_slice(&tile[..RESIDUAL_OFFSET]);
    out[RESIDUAL_OFFSET..].copy_from_slice(&tile[RESIDUAL_OFFSET + RESIDUAL_WORDS * 8..]);
    out
}

/// Unpack a 96-byte WARM form back into a full 128-byte tile (residual
/// zeroed). The result carries the same flags (including `FLAG_WARM`).
pub fn unpack_warm(packed: &[u8; WARM_PACKED_BYTES]) -> [u8; TILE_BYTES] {
    let mut out = [0u8; TILE_BYTES];
    out[..RESIDUAL_OFFSET].copy_from_slice(&packed[..RESIDUAL_OFFSET]);
    out[RESIDUAL_OFFSET + RESIDUAL_WORDS * 8..].copy_from_slice(&packed[RESIDUAL_OFFSET..]);
    out
}

pub fn score_hot(
    q_coarse: &[f32],
    q_sign: &[u64],
    latent: &[u8],
    residual: &[u64],
    scale: f32,
    dynamic_lambda: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    let coarse = dot_coarse(q_coarse, latent, scale, group_scales, flags);
    let ham = hamming_distance(q_sign, residual);
    coarse + dynamic_lambda * (D_S as f32 - 2.0 * ham as f32)
}

pub fn score_warm(
    q_coarse: &[f32],
    latent: &[u8],
    scale: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    dot_coarse(q_coarse, latent, scale, group_scales, flags)
}

pub fn dot_coarse(
    q_coarse: &[f32],
    latent: &[u8],
    scale: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..D_C {
        let val = dequant_at(latent, d, scale, group_scales, flags);
        sum += q_coarse[d] * val;
    }
    sum
}

/// Error for a tile whose flag combination selects no supported codec.
///
/// The vram codecs implement uniform INT4, NF4, MIXED, TQ3 and MIX3 — the
/// full scirust set. Any other flag combination (e.g. two mutually exclusive
/// codec flags set together) is rejected instead of being silently decoded
/// as INT4 (the old behaviour, which produced wrong scores for unsupported
/// combos).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecError {
    pub flags: u16,
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unsupported tile codec flag combination {:#06x} (supported: INT4/NF4/MIXED/TQ3/MIX3)",
            self.flags
        )
    }
}

impl std::error::Error for CodecError {}

/// Validate a tile's flag combination selects exactly one supported codec.
///
/// Returns `Ok(())` for a valid INT4/NF4/MIXED/TQ3/MIX3 tile (HOT or WARM,
/// with or without `FLAG_TQ3_NOCORR`), else `Err(CodecError)`.
pub fn validate_codec(flags: u16) -> Result<(), CodecError> {
    let codecs = [
        (FLAG_MIXED, "MIXED"),
        (FLAG_TQ3, "TQ3"),
        (FLAG_MIX3, "MIX3"),
        (FLAG_NF4, "NF4"),
    ];
    let selected = codecs
        .iter()
        .filter(|(mask, _)| has_flag(flags, *mask))
        .count();
    if selected > 1 {
        return Err(CodecError { flags });
    }
    // `FLAG_TQ3_NOCORR` is only meaningful with TQ3 or MIX3.
    if has_flag(flags, FLAG_TQ3_NOCORR)
        && !(has_flag(flags, FLAG_TQ3) || has_flag(flags, FLAG_MIX3))
    {
        return Err(CodecError { flags });
    }
    Ok(())
}

pub fn dequant_at(latent: &[u8], d: usize, scale: f32, group_scales: &[u8], flags: u16) -> f32 {
    if has_flag(flags, FLAG_MIXED) {
        return dequant_mixed(latent, d, scale, group_scales);
    }
    if has_flag(flags, FLAG_TQ3) {
        return dequant_tq3(latent, d, scale, group_scales, flags);
    }
    if has_flag(flags, FLAG_MIX3) {
        return dequant_mix3(latent, d, scale, group_scales, flags);
    }
    if has_flag(flags, FLAG_NF4) {
        dequant_nf4(latent, d, scale, group_scales)
    } else {
        dequant_int4(latent, d, scale, group_scales)
    }
}

fn dequant_int4(latent: &[u8], d: usize, scale: f32, group_scales: &[u8]) -> f32 {
    let byte = latent[d >> 1];
    let nib = ext_nibble(byte, (d & 1) != 0);
    let level = decode_int4_nibble(nib);
    let gs = effective_scale(scale, group_scales[d / GROUP_DIM]);
    level * gs
}

fn dequant_nf4(latent: &[u8], d: usize, scale: f32, group_scales: &[u8]) -> f32 {
    let byte = latent[d >> 1];
    let nib = ext_nibble(byte, (d & 1) != 0);
    let level = NF4_CODEBOOK[nib as usize];
    let gs = effective_scale(scale, group_scales[d / GROUP_DIM]);
    level * gs
}

/// Mixed-precision layout: dims `0..MIXED_HI_DIMS` are signed bytes
/// (zero-point 128) scaled by `group_scales[0]`; dims
/// `MIXED_HI_DIMS..MIXED_DIMS` are nibbles in `GROUP_DIM`-wide groups scaled
/// by `group_scales[1..]`; the dropped tail decodes to 0. Mirrors
/// `scirust`'s `dequant_at_mixed`.
fn dequant_mixed(latent: &[u8], d: usize, scale: f32, group_scales: &[u8]) -> f32 {
    if d < MIXED_HI_DIMS {
        let level = latent[d] as i32 - 128;
        level as f32 * (scale * group_scales[0] as f32 / 255.0)
    } else if d < MIXED_DIMS {
        let ld = d - MIXED_HI_DIMS;
        let byte = latent[MIXED_HI_DIMS + (ld >> 1)];
        let nib = ext_nibble(byte, (ld & 1) != 0);
        let g = 1 + ld / GROUP_DIM;
        (nib as i32 - 8) as f32 * (scale * group_scales[g] as f32 / 255.0)
    } else {
        0.0
    }
}

/// TQ3 layout: dim `d`'s 3-bit code is bits `[3d, 3d+3)` of the code plane;
/// its correction sign is bit `d` of the correction plane. Decoded level =
/// `(code − 3.5) ± TQ3_CORRECTION`, times the dim's group scale. With
/// `FLAG_TQ3_NOCORR` the correction plane is ignored. Mirrors `scirust`'s
/// `dequant_at_tq3`.
fn dequant_tq3(latent: &[u8], d: usize, scale: f32, group_scales: &[u8], flags: u16) -> f32 {
    let bit = 3 * d;
    let byte = bit >> 3;
    let shift = bit & 7;
    let lo = u16::from(latent[byte]);
    let hi = if byte + 1 < TQ3_CODE_BYTES {
        u16::from(latent[byte + 1]) << 8
    } else {
        0
    };
    let code = ((lo | hi) >> shift) & 0x7;
    let mut level = code as f32 - TQ3_HALF_RANGE;
    if !has_flag(flags, FLAG_TQ3_NOCORR) {
        let corr = (latent[TQ3_CODE_BYTES + (d >> 3)] >> (d & 7)) & 1;
        let sign = if corr == 1 { 1.0 } else { -1.0 };
        level += sign * TQ3_CORRECTION;
    }
    level * effective_scale(scale, group_scales[d / GROUP_DIM])
}

/// MIX3 layout: dims `0..MIXED_HI_DIMS` decode like the mixed head; dims
/// `MIXED_HI_DIMS..MIXED_DIMS` decode like a TQ3 body at
/// [`MIX3_CODES_OFF`]/[`MIX3_CORR_OFF`]; the tail decodes to 0. Mirrors
/// `scirust`'s `dequant_at_mix3`.
fn dequant_mix3(latent: &[u8], d: usize, scale: f32, group_scales: &[u8], flags: u16) -> f32 {
    if d < MIXED_HI_DIMS {
        let level = latent[d] as i32 - 128;
        return level as f32 * (scale * group_scales[0] as f32 / 255.0);
    }
    if d >= MIXED_DIMS {
        return 0.0;
    }
    let ld = d - MIXED_HI_DIMS;
    let bit = 3 * ld;
    let byte = MIX3_CODES_OFF + (bit >> 3);
    let shift = bit & 7;
    let lo = u16::from(latent[byte]);
    let hi = if byte + 1 < MIX3_CORR_OFF {
        u16::from(latent[byte + 1]) << 8
    } else {
        0
    };
    let code = ((lo | hi) >> shift) & 0x7;
    let mut level = code as f32 - TQ3_HALF_RANGE;
    if !has_flag(flags, FLAG_TQ3_NOCORR) {
        let corr = (latent[MIX3_CORR_OFF + (ld >> 3)] >> (ld & 7)) & 1;
        let sign = if corr == 1 { 1.0 } else { -1.0 };
        level += sign * TQ3_CORRECTION;
    }
    let g = 1 + ld / GROUP_DIM;
    level * (scale * group_scales[g] as f32 / 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_int4() {
        assert_eq!(decode_int4_nibble(0x0), -8.0);
        assert_eq!(decode_int4_nibble(0x8), 0.0);
        assert_eq!(decode_int4_nibble(0xF), 7.0);
    }

    #[test]
    fn test_nf4_codebook_exact_scirust() {
        let expected: [f32; 16] = [
            -1.0, -0.7075, -0.5421, -0.4165, -0.3108, -0.2158, -0.1272, -0.0421, 0.0421, 0.1272,
            0.2158, 0.3108, 0.4165, 0.5421, 0.7075, 1.0,
        ];
        for i in 0..16 {
            assert!(
                (NF4_CODEBOOK[i] - expected[i]).abs() < 1e-6,
                "NF4_CODEBOOK[{i}]: {} != {}",
                NF4_CODEBOOK[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_nf4_codebook_range() {
        assert!((NF4_CODEBOOK[0] - (-1.0)).abs() < 1e-6);
        assert!((NF4_CODEBOOK[7] - (-0.0421)).abs() < 1e-6);
        assert!((NF4_CODEBOOK[8] - 0.0421).abs() < 1e-6);
        assert!((NF4_CODEBOOK[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flags() {
        assert!(has_flag(FLAG_WARM, FLAG_WARM));
        assert!(!has_flag(FLAG_WARM, FLAG_NF4));
        assert!(has_flag(FLAG_WARM | FLAG_NF4, FLAG_WARM));
    }

    #[test]
    fn test_effective_scale() {
        let eff = effective_scale(2.0, 128);
        assert!((eff - 1.0039216).abs() < 1e-4);
    }

    #[test]
    fn test_f32_slice_to_le_bytes_roundtrip() {
        let input = vec![1.0f32, -2.5, std::f32::consts::PI];
        let bytes = f32_slice_to_le_bytes(&input);
        let output = le_bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(input.len(), output.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_u64_slice_to_le_bytes_roundtrip() {
        let input = vec![0xDEAD_BEEF_CAFE_FACEu64, 0x1234_5678_9ABC_DEF0u64];
        let bytes = u64_slice_to_le_bytes(&input);
        let output = le_bytes_to_u64_vec(&bytes).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_le_bytes_f32_rejects_odd_len() {
        assert!(le_bytes_to_f32_vec(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_le_bytes_u64_rejects_odd_len() {
        assert!(le_bytes_to_u64_vec(&[0u8; 7]).is_err());
    }

    #[test]
    fn test_read_f32_le() {
        let orig = std::f32::consts::PI;
        let bytes = orig.to_le_bytes();
        let buf = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let val = read_f32_le(&buf, 0).unwrap();
        assert!((val - orig).abs() < 1e-6);
    }

    #[test]
    fn test_read_f32_le_out_of_bounds() {
        let buf = [0u8; 3];
        assert!(read_f32_le(&buf, 0).is_err());
    }

    #[test]
    fn test_read_u16_le() {
        let bytes = 0xABCDu16.to_le_bytes();
        let val = read_u16_le(&bytes, 0).unwrap();
        assert_eq!(val, 0xABCD);
    }

    #[test]
    fn test_constants_match_scirust() {
        assert_eq!(D_C, scirust::D_C);
        assert_eq!(D_S, scirust::D_S);
        assert_eq!(LATENT_BYTES, scirust::LATENT_BYTES);
        assert_eq!(RESIDUAL_WORDS, scirust::RESIDUAL_WORDS);
        assert_eq!(TILE_BYTES, 128);
        assert_eq!(N_GROUPS, 8);
        assert_eq!(GROUP_DIM, 16);
    }

    #[test]
    fn test_offsets_are_sensible() {
        assert_eq!(RESIDUAL_OFFSET, LATENT_BYTES);
        assert_eq!(SCALE_OFFSET, 96);
        assert_eq!(DYNAMIC_LAMBDA_OFFSET, 100);
        assert_eq!(FLAGS_OFFSET, 118);
        assert_eq!(GROUP_SCALES_OFFSET, 120);
        const _: () = assert!(GROUP_SCALES_OFFSET + 8 <= TILE_BYTES);
    }
}
