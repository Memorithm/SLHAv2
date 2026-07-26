//! SLHAv2 tile constants, flags, and decoding helpers matching the scirust
//! source of truth (slha_v2.rs / ccos.rs).

pub const TILE_BYTES: usize = 128;

pub const D_C: usize = 128;

pub const RESIDUAL_WORDS: usize = 4;

pub const LATENT_KV_WORDS: usize = 64;

pub const D_S: usize = 256;

pub const MIXED_HI_DIMS: usize = 32;

pub const TQ3_NUM_BITS: usize = 3;

pub const TQ3_HALF_RANGE: usize = 4;

pub const TQ3_CORR_STRIDE: usize = 8;

pub const HOT_BYTES: usize = 128;

pub const WARM_BYTES: usize = HOT_BYTES - RESIDUAL_WORDS * 8;

pub const RESIDUAL_OFFSET: usize = 64;
pub const SCALE_OFFSET: usize = 96;
pub const DYNAMIC_LAMBDA_OFFSET: usize = 100;
pub const FLAGS_OFFSET: usize = 118;
pub const GROUP_SCALES_OFFSET: usize = 120;

pub const FLAG_WARM: u16 = 1 << 0;
pub const FLAG_NF4: u16 = 1 << 1;
pub const FLAG_MIXED: u16 = 1 << 2;
pub const FLAG_TQ3: u16 = 1 << 3;
pub const FLAG_MIX3: u16 = 1 << 4;
pub const FLAG_TQ3_NOCORR: u16 = 1 << 5;

pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.7075,
    -0.5421,
    -0.4165,
    -0.3108,
    -0.2160,
    -0.0685,
    0.0,
    0.0685,
    0.2160,
    0.3108,
    0.4165,
    0.5421,
    0.7075,
    1.0,
    1.0,
];

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

pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

pub fn score_hot(
    q_coarse: &[f32],
    q_sign: &[u64],
    latent_kv: &[u8],
    residual: &[u64],
    scale: f32,
    dynamic_lambda: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    let coarse = dot_coarse(q_coarse, latent_kv, scale, group_scales, flags);
    let ham = hamming_distance(q_sign, residual);
    coarse + dynamic_lambda * (D_S as f32 - 2.0 * ham as f32)
}

pub fn score_warm(
    q_coarse: &[f32],
    latent_kv: &[u8],
    scale: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    dot_coarse(q_coarse, latent_kv, scale, group_scales, flags)
}

pub fn dot_coarse(
    q_coarse: &[f32],
    latent_kv: &[u8],
    scale: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..D_C {
        let val = dequant_at(latent_kv, d, scale, group_scales, flags);
        sum += q_coarse[d] * val;
    }
    sum
}

pub fn dequant_at(
    latent_kv: &[u8],
    d: usize,
    scale: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    if has_flag(flags, FLAG_NF4) {
        dequant_nf4(latent_kv, d, scale, group_scales)
    } else if has_flag(flags, FLAG_MIX3) {
        dequant_mix3(latent_kv, d, scale, group_scales)
    } else if has_flag(flags, FLAG_TQ3) {
        dequant_tq3(latent_kv, d, scale, group_scales, flags)
    } else if has_flag(flags, FLAG_MIXED) {
        dequant_mixed(latent_kv, d, scale, group_scales)
    } else {
        dequant_int4(latent_kv, d, scale, group_scales)
    }
}

fn dequant_int4(
    latent_kv: &[u8],
    d: usize,
    scale: f32,
    group_scales: &[u8],
) -> f32 {
    let byte = latent_kv[d >> 1];
    let nib = ext_nibble(byte, (d & 1) != 0);
    let level = decode_int4_nibble(nib);
    let gs = effective_scale(scale, group_scales[d / 16]);
    level * gs
}

fn dequant_nf4(
    latent_kv: &[u8],
    d: usize,
    scale: f32,
    group_scales: &[u8],
) -> f32 {
    let byte = latent_kv[d >> 1];
    let nib = ext_nibble(byte, (d & 1) != 0);
    let level = NF4_CODEBOOK[nib as usize];
    let gs = effective_scale(scale, group_scales[d / 16]);
    level * gs
}

fn dequant_mixed(
    latent_kv: &[u8],
    d: usize,
    scale: f32,
    group_scales: &[u8],
) -> f32 {
    if d < MIXED_HI_DIMS {
        let val = latent_kv[d] as i32 - 128;
        let gs = effective_scale(scale, group_scales[d / 16]);
        (val as f32) * gs
    } else {
        let adj = d - MIXED_HI_DIMS;
        let byte = latent_kv[MIXED_HI_DIMS + (adj >> 1)];
        let nib = ext_nibble(byte, (adj & 1) != 0);
        let level = decode_int4_nibble(nib);
        let gs = effective_scale(scale, group_scales[d / 16]);
        level * gs
    }
}

fn dequant_tq3(
    latent_kv: &[u8],
    d: usize,
    scale: f32,
    group_scales: &[u8],
    flags: u16,
) -> f32 {
    let (nib0, bit_offset) = tq3_bit_position(d);
    let code = tq3_read_bits(latent_kv, nib0, bit_offset);

    let correction = if !has_flag(flags, FLAG_TQ3_NOCORR) {
        let cb = latent_kv[LATENT_KV_WORDS + tq3_correction_byte(d)];
        (cb >> (d & 7)) & 1
    } else {
        0
    };

    let level: f32 = if correction != 0 {
        if code >= TQ3_HALF_RANGE {
            (code as i32 + 1 - TQ3_HALF_RANGE as i32) as f32
        } else {
            (code as i32 - 1 + TQ3_HALF_RANGE as i32) as f32
        }
    } else {
        (code as i32 - TQ3_HALF_RANGE as i32) as f32
    };

    let gs = effective_scale(scale, group_scales[d / 16]);
    level * gs
}

fn tq3_bit_position(d: usize) -> (usize, usize) {
    let total_bits = d * 3;
    let nib0 = total_bits / 8;
    let bit_offset = total_bits % 8;
    (nib0, bit_offset)
}

fn tq3_read_bits(latent_kv: &[u8], nib0: usize, bit_offset: usize) -> usize {
    let needed = bit_offset + 3;
    let n_read = (needed + 7) / 8;
    let mut val: usize = 0;
    for i in 0..n_read {
        let idx = nib0 + i;
        let b = if idx < latent_kv.len() {
            latent_kv[idx] as usize
        } else {
            0
        };
        val |= b << (i * 8);
    }
    (val >> bit_offset) & 0x7
}

fn tq3_correction_byte(d: usize) -> usize {
    d / 8
}

fn dequant_mix3(
    latent_kv: &[u8],
    d: usize,
    scale: f32,
    group_scales: &[u8],
) -> f32 {
    if d < MIXED_HI_DIMS {
        let val = latent_kv[d] as i32 - 128;
        let gs = effective_scale(scale, group_scales[d / 16]);
        (val as f32) * gs
    } else {
        let adj = d - MIXED_HI_DIMS;
        dequant_tq3_inner(latent_kv, MIXED_HI_DIMS, adj, scale, group_scales)
    }
}

fn dequant_tq3_inner(
    latent_kv: &[u8],
    base: usize,
    d: usize,
    scale: f32,
    group_scales: &[u8],
) -> f32 {
    let total_bits = d * 3;
    let nib0 = base + total_bits / 8;
    let bit_offset = total_bits % 8;

    let needed = bit_offset + 3;
    let n_read = (needed + 7) / 8;
    let mut val: usize = 0;
    for i in 0..n_read {
        let idx = nib0 + i;
        let b = if idx < latent_kv.len() {
            latent_kv[idx] as usize
        } else {
            0
        };
        val |= b << (i * 8);
    }
    let code = (val >> bit_offset) & 0x7;

    let correction_byte = d / 8;
    let correction = (latent_kv[LATENT_KV_WORDS + correction_byte] >> (d & 7)) & 1;

    let level: f32 = if correction != 0 {
        if code >= TQ3_HALF_RANGE {
            (code as i32 + 1 - TQ3_HALF_RANGE as i32) as f32
        } else {
            (code as i32 - 1 + TQ3_HALF_RANGE as i32) as f32
        }
    } else {
        (code as i32 - TQ3_HALF_RANGE as i32) as f32
    };

    let gs = effective_scale(scale, group_scales[d / 16]);
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
    fn test_nf4_codebook_range() {
        assert!((NF4_CODEBOOK[0] - (-1.0)).abs() < 1e-6);
        assert!((NF4_CODEBOOK[7]).abs() < 1e-6);
        assert!((NF4_CODEBOOK[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let a = [0u64, 0, 0, 0];
        let b = [0u64, 0, 0, 0];
        assert_eq!(hamming_distance(&a, &b), 0);

        let c = [0xFFu64, 0, 0, 0];
        assert_eq!(hamming_distance(&a, &c), 8);
    }

    #[test]
    fn test_flags() {
        assert!(has_flag(FLAG_WARM, FLAG_WARM));
        assert!(!has_flag(FLAG_WARM, FLAG_NF4));
        assert!(has_flag(FLAG_WARM | FLAG_NF4, FLAG_WARM));
    }

    #[test]
    fn test_tq3_bit_position() {
        let (nib, bit) = tq3_bit_position(0);
        assert_eq!(nib, 0);
        assert_eq!(bit, 0);

        let (nib, bit) = tq3_bit_position(1);
        assert_eq!(nib, 0);
        assert_eq!(bit, 3);

        let (nib, bit) = tq3_bit_position(3);
        assert_eq!(nib, 1);
        assert_eq!(bit, 1);
    }

    #[test]
    fn test_effective_scale() {
        let eff = effective_scale(2.0, 128);
        assert!((eff - 1.0039216).abs() < 1e-4);
    }
}
