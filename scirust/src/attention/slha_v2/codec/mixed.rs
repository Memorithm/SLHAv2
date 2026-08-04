//! Mixed-precision quantisation: 8-bit head + 4-bit body, tail dropped.

use super::super::constants::*;

/// Mixed-precision quantisation (see [`FLAG_MIXED`]): the top
/// [`MIXED_HI_DIMS`] dims as signed bytes (zero-point 128, step
/// `max|·|/127`), the next [`MIXED_LO_DIMS`] as per-group signed INT4, the
/// remaining tail **dropped** — the same 64-byte budget spent where the
/// energy is. Assumes the latent is ordered by decreasing variance (PCA
/// order). Scale bytes: `gs[0]` is the 8-bit block's step relative to
/// `global`, `gs[1..]` the 4-bit groups' steps, exactly like the grouped
/// codec. Returns `(bytes, global, gs)`.
pub fn quantize_latent_mixed(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32, [u8; N_GROUPS]) {
    // Per-section quantisation steps.
    let mut steps = [0.0f32; N_GROUPS];
    let hi_max = v[..MIXED_HI_DIMS]
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    steps[0] = hi_max / 127.0;
    for g in 0..MIXED_LO_GROUPS {
        let base = MIXED_HI_DIMS + g * GROUP_DIM;
        let mut mx = 0.0f32;
        for d in base..base + GROUP_DIM {
            mx = mx.max(v[d].abs());
        }
        steps[1 + g] = mx / 7.0;
    }
    let global = steps.iter().copied().fold(0.0f32, f32::max);
    let global = if global > 0.0 { global } else { 1.0 };

    let mut gs = [0u8; N_GROUPS];
    for g in 0..N_GROUPS {
        gs[g] = (steps[g] / global * 255.0).round().clamp(1.0, 255.0) as u8;
    }

    let mut out = [0u8; LATENT_BYTES];
    // 8-bit head: one signed byte per dim, zero-point 128.
    let eff_hi = global * (gs[0] as f32 / 255.0);
    for d in 0..MIXED_HI_DIMS {
        let q = (v[d] / eff_hi).round() as i32;
        out[d] = (q.clamp(-128, 127) + 128) as u8;
    }
    // 4-bit body: nibbles after the head, grouped like the uniform codec.
    for ld in 0..MIXED_LO_DIMS {
        let d = MIXED_HI_DIMS + ld;
        let eff = global * (gs[1 + ld / GROUP_DIM] as f32 / 255.0);
        let nib = (((v[d] / eff).round() as i32).clamp(-8, 7) + 8) as u8 & 0x0F;
        let byte = MIXED_HI_DIMS + (ld >> 1);
        if ld & 1 == 0 {
            out[byte] = (out[byte] & 0xF0) | nib;
        } else {
            out[byte] = (out[byte] & 0x0F) | (nib << 4);
        }
    }
    // Tail dims MIXED_DIMS..D_C are dropped (decode to 0).
    (out, global, gs)
}
