// These are numerical kernels: range loops and fused signatures are clearer.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub use scirust::attention::slha_v2::{
    FLAG_HOT, FLAG_NF4, FLAG_WARM, GROUP_DIM, NF4_CODEBOOK, N_GROUPS,
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
    if bytes.len() % 4 != 0 {
        return Err("byte length not divisible by 4 for f32 conversion");
    }
    bytes
        .chunks_exact(4)
        .map(|c| {
            <[u8; 4]>::try_from(c)
                .map(f32::from_le_bytes)
                .map_err(|_| "invalid f32 byte chunk")
        })
        .collect()
}

pub fn le_bytes_to_u64_vec(bytes: &[u8]) -> Result<Vec<u64>, &'static str> {
    if bytes.len() % 8 != 0 {
        return Err("byte length not divisible by 8 for u64 conversion");
    }
    bytes
        .chunks_exact(8)
        .map(|c| {
            <[u8; 8]>::try_from(c)
                .map(u64::from_le_bytes)
                .map_err(|_| "invalid u64 byte chunk")
        })
        .collect()
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

pub fn dequant_at(latent: &[u8], d: usize, scale: f32, group_scales: &[u8], flags: u16) -> f32 {
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
