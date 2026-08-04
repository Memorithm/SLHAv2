//! Uniform signed INT4 quantisation (single-scale and per-group MX).

use super::super::constants::*;

/// Quantise a latent vector to signed INT4 with a symmetric per-tile scale.
///
/// Returns the packed nibbles and the scale. `value ≈ (nibble - 8) * scale`,
/// with `nibble ∈ [0, 15]` mapping to the signed range `[-8, 7]`.
pub fn quantize_latent(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    // Map the largest magnitude to +7 so it survives the [-8, 7] clamp.
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let q = (v[d] / scale).round() as i32;
        let nib = (q.clamp(-8, 7) + 8) as u8 & 0x0F; // 0..=15
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, scale)
}

/// Per-group ("micro-scaling") signed INT4 quantisation.
///
/// Splits the latent into [`N_GROUPS`] groups of [`GROUP_DIM`] dims; each group
/// gets its own scale stored as a `u8` relative to the global (max) scale:
/// `effective_scale(g) = global · gs[g]/255`. Because PCA orders the latent by
/// descending variance, grouping gives the low-variance tail its own finer
/// scale instead of being crushed by a single global scale. Returns
/// `(nibbles, global_scale, group_bytes)`.
pub fn quantize_latent_grouped(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32, [u8; N_GROUPS]) {
    let mut group_scale = [0.0f32; N_GROUPS];
    for g in 0..N_GROUPS {
        let mut mx = 0.0f32;
        for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
            mx = mx.max(v[d].abs());
        }
        group_scale[g] = mx / 7.0;
    }
    let global = group_scale.iter().copied().fold(0.0f32, f32::max);
    let global = if global > 0.0 { global } else { 1.0 };

    let mut gs = [0u8; N_GROUPS];
    for g in 0..N_GROUPS {
        let r = (group_scale[g] / global * 255.0).round();
        gs[g] = r.clamp(1.0, 255.0) as u8; // never 0, so dequant stays well-defined
    }

    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
        let nib = (((v[d] / eff).round() as i32).clamp(-8, 7) + 8) as u8 & 0x0F;
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, global, gs)
}
