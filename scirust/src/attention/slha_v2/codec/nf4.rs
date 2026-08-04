//! NF4 (NormalFloat-4) codebook quantisation.

use super::super::constants::*;

/// Index of the nearest NF4 codebook level to `t` (expects `t ∈ [-1, 1]`).
#[inline]
pub fn nf4_nearest(t: f32) -> u8 {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &c) in NF4_CODEBOOK.iter().enumerate() {
        let dd = (t - c).abs();
        if dd < best_d {
            best_d = dd;
            best = i;
        }
    }
    best as u8
}

/// Per-group NF4 quantisation. Each group is scaled by its absmax (the NF4
/// codebook spans `[-1, 1]`); the scale is stored relative to the global one,
/// exactly like [`super::int4::quantize_latent_grouped`]. Returns `(nibbles, global, gs)`.
pub fn quantize_latent_nf4(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32, [u8; N_GROUPS]) {
    let mut group_scale = [0.0f32; N_GROUPS];
    for g in 0..N_GROUPS {
        let mut mx = 0.0f32;
        for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
            mx = mx.max(v[d].abs());
        }
        group_scale[g] = mx; // absmax (codebook max == 1.0)
    }
    let global = group_scale.iter().copied().fold(0.0f32, f32::max);
    let global = if global > 0.0 { global } else { 1.0 };

    let mut gs = [0u8; N_GROUPS];
    for g in 0..N_GROUPS {
        let r = (group_scale[g] / global * 255.0).round();
        gs[g] = r.clamp(1.0, 255.0) as u8;
    }

    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
        let t = if eff > 0.0 {
            (v[d] / eff).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let nib = nf4_nearest(t);
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, global, gs)
}
