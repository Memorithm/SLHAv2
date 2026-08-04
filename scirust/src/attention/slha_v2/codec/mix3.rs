//! MIX3 quantisation: mixed 8-bit head + TQ3 body, tail dropped.

use super::super::constants::*;

/// MIX3 quantisation (see [`FLAG_MIX3`]): the top [`MIXED_HI_DIMS`] dims as
/// signed bytes (zero-point 128, step `max|·|/127`, scale byte `gs[0]`) —
/// exactly the mixed codec's head — then [`MIXED_LO_DIMS`] dims on the TQ3
/// grid (3-bit codes at [`MIX3_CODES_OFF`] + separable 1-bit correction
/// plane at [`MIX3_CORR_OFF`], per-group scales `gs[1..]`, step `max|·|/3.5`),
/// remaining tail **dropped**. Returns `(bytes, global, gs)`.
pub fn quantize_latent_mix3(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32, [u8; N_GROUPS]) {
    // Per-section quantisation steps (head like mixed, body like TQ3).
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
        steps[1 + g] = mx / TQ3_HALF_RANGE;
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
    // TQ3 body: 3-bit codes + correction signs, grouped like the mixed body.
    for ld in 0..MIXED_LO_DIMS {
        let d = MIXED_HI_DIMS + ld;
        let eff = global * (gs[1 + ld / GROUP_DIM] as f32 / 255.0);
        let t = v[d] / eff; // value in grid-step units, within ±3.5
        let code = ((t + TQ3_HALF_RANGE).round() as i32).clamp(0, 7) as u8;
        let bit = 3 * ld;
        let byte = MIX3_CODES_OFF + (bit >> 3);
        let shift = bit & 7;
        out[byte] |= code << shift;
        if shift > 5 {
            out[byte + 1] |= code >> (8 - shift);
        }
        // Correction sign: 1 ⇒ the true value sits above the decoded level.
        let level = f32::from(code) - TQ3_HALF_RANGE;
        if t - level > 0.0 {
            out[MIX3_CORR_OFF + (ld >> 3)] |= 1 << (ld & 7);
        }
    }
    // Tail dims MIXED_DIMS..D_C are dropped (decode to 0).
    (out, global, gs)
}
