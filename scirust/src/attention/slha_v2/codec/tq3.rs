//! TurboQuant TQ3 quantisation: 3-bit symmetric grid + 1-bit correction.

use super::super::constants::*;

/// TurboQuant TQ3 quantisation (see [`FLAG_TQ3`]): every dim to a 3-bit
/// symmetric grid (codes 0..=7 → levels `code − 3.5`, no zero level) plus a
/// 1-bit residual sign correction of [`TQ3_CORRECTION`] step, per-group
/// scales exactly like [`super::int4::quantize_latent_grouped`]. The 64-byte
/// budget is split 48 B codes + 16 B correction bits. Returns `(bytes, global, gs)`.
pub fn quantize_latent_tq3(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32, [u8; N_GROUPS]) {
    let mut group_scale = [0.0f32; N_GROUPS];
    for g in 0..N_GROUPS {
        let mut mx = 0.0f32;
        for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
            mx = mx.max(v[d].abs());
        }
        // The group's absmax maps to the outermost level ±3.5.
        group_scale[g] = mx / TQ3_HALF_RANGE;
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
        let t = v[d] / eff; // value in grid-step units, within ±3.5
        let code = ((t + TQ3_HALF_RANGE).round() as i32).clamp(0, 7) as u8;
        // Pack the 3-bit code at bits [3d, 3d+3) (little-endian bitstream).
        let bit = 3 * d;
        let byte = bit >> 3;
        let shift = bit & 7;
        out[byte] |= code << shift;
        if shift > 5 {
            out[byte + 1] |= code >> (8 - shift);
        }
        // Correction sign: 1 ⇒ the true value sits above the decoded level.
        let level = f32::from(code) - TQ3_HALF_RANGE;
        if t - level > 0.0 {
            out[TQ3_CODE_BYTES + (d >> 3)] |= 1 << (d & 7);
        }
    }
    (out, global, gs)
}
