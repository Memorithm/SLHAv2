//! Layout constants, tile state flags and per-codec byte geometry.
//!
//! Everything here is `const`-evaluable; the `const _: () = assert!(...)`
//! items below are compile-time layout checks that pin the codecs to the
//! 64-byte latent budget.

/// Latent dimensionality stored per tile (INT4).
pub const D_C: usize = 128;
/// Sign-LSH residual width, in bits.
pub const D_S: usize = 256;
/// Bytes used by the INT4 latent block: two 4-bit samples per byte.
pub const LATENT_BYTES: usize = D_C / 2; // 64
/// Number of `u64` words in the residual bitmap.
pub const RESIDUAL_WORDS: usize = D_S / 64; // 4
/// Number of micro-scaling groups for the INT4 latent (one scale byte each).
pub const N_GROUPS: usize = 8;
/// Latent dimensions per micro-scaling group.
pub const GROUP_DIM: usize = D_C / N_GROUPS; // 16

// --- Tile state flags (the CCOS Soft-Paging modes of spec §4) ---------------
/// Full-fidelity tile: latent + residual both live (cache L1/L2).
pub const FLAG_HOT: u16 = 0;
/// Elastic paging: residual bitmap considered freed; score uses the latent
/// base only (`dynamic_lambda` is bypassed). 25% footprint drop (32 o of 128),
/// no I/O. Driven by [`crate::ccos::ElasticKvCache`].
pub const FLAG_WARM: u16 = 1 << 0;
/// Latent uses the NF4 (NormalFloat-4) codebook instead of uniform INT4.
pub const FLAG_NF4: u16 = 1 << 1;
/// Latent uses the mixed-precision layout: the top [`MIXED_HI_DIMS`] dims at
/// 8-bit, the next [`MIXED_LO_DIMS`] at 4-bit, the tail dropped — same 64 bytes.
pub const FLAG_MIXED: u16 = 1 << 2;
/// Latent uses the TurboQuant TQ3 layout: 3-bit codes ([`TQ3_CODE_BYTES`])
/// plus a 1-bit sign-correction plane ([`TQ3_CORR_BYTES`]) — same 64 bytes.
pub const FLAG_TQ3: u16 = 1 << 3;
/// Separable-plane elastic paging (CCOS): the codec's 1-bit correction
/// plane is considered freed; decode falls back to the bare 3-bit grid
/// (worst-case error 0.5 step instead of 0.25 on the covered dims). Only
/// meaningful with [`FLAG_TQ3`] (16-byte plane) or [`FLAG_MIX3`] (14-byte
/// plane). Sticky — the Soft-Paging ladder only degrades. Set by
/// [`crate::ccos::ElasticKvCache::drop_correction`].
pub const FLAG_TQ3_NOCORR: u16 = 1 << 4;
/// Latent uses the MIX3 synthesis layout: mixed-precision head (top
/// [`MIXED_HI_DIMS`] dims at 8-bit) + TQ3 body ([`MIXED_LO_DIMS`] dims at
/// 3-bit with a separable 1-bit correction plane) — same 64 bytes. Built
/// after the measured NO-GO of uniform grids on real activations: the
/// 8-bit head covers the steep spectrum like [`FLAG_MIXED`], while the
/// separable plane keeps the CCOS paging rung TQ3 introduced.
pub const FLAG_MIX3: u16 = 1 << 5;

// --- Mixed-precision latent layout (FLAG_MIXED) ------------------------------
// Real transformer keys concentrate energy in a few directions (GPT-2 layer 6:
// 40% of ALL key energy in ONE direction, 87% in four, a 56× magnitude range
// inside the first 16-dim scaling group). Uniform INT4's 16 levels cannot span
// that range, and the resulting coarse-score error dominates the total loss
// (measured: attention-output cosine 0.958 float → 0.834 uniform INT4).
// Spending the same 64 bytes non-uniformly — 8 bits where the energy is —
// recovers nearly all of it (0.953–0.956 in the same measurement).
/// Dims stored at 8-bit (one signed byte each) by the mixed codec.
pub const MIXED_HI_DIMS: usize = 8;
/// Dims stored at 4-bit after the 8-bit block: the remaining 56 bytes.
pub const MIXED_LO_DIMS: usize = 2 * (LATENT_BYTES - MIXED_HI_DIMS); // 112
/// Latent dims the mixed codec keeps; the `D_C − MIXED_DIMS` lowest-variance
/// dims are dropped (PCA orders dims by decreasing variance, so the tail is
/// the ~0%-energy end of the spectrum).
pub const MIXED_DIMS: usize = MIXED_HI_DIMS + MIXED_LO_DIMS; // 120
/// 4-bit micro-scaling groups (16 dims each, like the uniform grouped codec);
/// `group_scales[0]` is the 8-bit block's scale, `group_scales[1..]` these.
pub const MIXED_LO_GROUPS: usize = N_GROUPS - 1; // 7 × GROUP_DIM = 112

// The mixed layout must spend exactly the 64-byte latent budget, reuse the
// 16-dim group geometry, use every scale byte, and fit in D_C dims.
const _: () = assert!(MIXED_HI_DIMS + MIXED_LO_DIMS / 2 == LATENT_BYTES);
const _: () = assert!(MIXED_LO_DIMS == MIXED_LO_GROUPS * GROUP_DIM);
const _: () = assert!(1 + MIXED_LO_GROUPS == N_GROUPS);
const _: () = assert!(MIXED_DIMS <= D_C);

// --- TurboQuant TQ3 latent layout (FLAG_TQ3) ---------------------------------
// Port of the TurboQuant KV-cache codec (QJL: 3-bit grid + 1-bit residual
// sign correction) into the 64-byte latent budget. All D_C dims are kept:
// 128 × 3 bits = 48 bytes of codes, then 128 × 1 bit = 16 bytes of
// per-dim correction signs — 64 bytes exactly, zero padding.
//
// Grid: 8 symmetric levels {±0.5, ±1.5, ±2.5, ±3.5} (code − 3.5, no zero
// level), per-group scaled like the other codecs. The correction bit moves
// the decoded level ±[`TQ3_CORRECTION`] (a quarter step, the optimal fixed
// magnitude for a uniform residual), so the worst-case error is 0.25 step —
// the same worst-case resolution as INT4 at the same 4 bits/dim total.
// Honest trade-off (measured in the tests below): the grid has no zero
// level, so values near 0 always pay ≥ 0.25 step; on a Gaussian latent the
// MSE is ~1.3–1.6× grouped INT4's. What the split buys instead: the two
// planes are separable — dropping the 16-byte correction plane degrades
// gracefully to a pure 3-bit tile (a future CCOS paging state, finer than
// HOT→WARM), which no nibble codec can offer.
//
// TurboQuant also rotates the vector before quantising (PolarQuant); on the
// SLHA latent this is unnecessary: `learned::LearnedModel` whitens the
// latent (per-dim `1/s_k`), which already equalises dynamic range, and the
// optional RHT (`incoherence`) covers the residual projection. See
// docs/TURBOQUANT.md.
/// Bytes of packed 3-bit codes: dim `d` occupies bits `[3d, 3d+3)` of the
/// little-endian bitstream in `latent_kv[0..TQ3_CODE_BYTES]`.
pub const TQ3_CODE_BYTES: usize = D_C * 3 / 8; // 48
/// Bytes of 1-bit correction signs: dim `d` is bit `d & 7` of
/// `latent_kv[TQ3_CODE_BYTES + d/8]`.
pub const TQ3_CORR_BYTES: usize = D_C / 8; // 16
/// Half-range of the 3-bit grid: codes 0..=7 decode to `code − 3.5`.
pub const TQ3_HALF_RANGE: f32 = 3.5;
/// Magnitude of the 1-bit correction, in grid steps (quarter step: the
/// residual after rounding is uniform in ±0.5 step, so E|r| = 0.25 step).
pub const TQ3_CORRECTION: f32 = 0.25;

// The TQ3 layout must spend exactly the 64-byte latent budget.
const _: () = assert!(TQ3_CODE_BYTES + TQ3_CORR_BYTES == LATENT_BYTES);
const _: () = assert!(D_C.is_multiple_of(8)); // both planes are byte-aligned

// --- MIX3 latent layout (FLAG_MIX3): mixed head × TQ3 body -------------------
// The GPT-2 measurement (docs/TURBOQUANT.md §3bis) showed the bottleneck of
// uniform grids on real activations is the missing 8-bit head, not the
// subspace. MIX3 combines both worlds in the same 64 bytes: the mixed
// codec's 8-bit head where the energy is, and a TQ3 body whose 1-bit
// correction plane stays separable — so the codec keeps near-mixed quality
// on steep spectra AND the CCOS correction-drop paging rung.
/// Byte where the MIX3 3-bit code plane starts (after the 8-bit head).
pub const MIX3_CODES_OFF: usize = MIXED_HI_DIMS; // 8
/// Bytes of packed 3-bit codes for the [`MIXED_LO_DIMS`] body dims.
pub const MIX3_CODE_BYTES: usize = MIXED_LO_DIMS * 3 / 8; // 42
/// Byte where the MIX3 correction plane starts.
pub const MIX3_CORR_OFF: usize = MIX3_CODES_OFF + MIX3_CODE_BYTES; // 50
/// Bytes of the separable 1-bit correction plane (one bit per body dim).
pub const MIX3_CORR_BYTES: usize = MIXED_LO_DIMS / 8; // 14

// The MIX3 layout must spend exactly the 64-byte latent budget.
const _: () = assert!(MIXED_HI_DIMS + MIX3_CODE_BYTES + MIX3_CORR_BYTES == LATENT_BYTES);
const _: () = assert!(MIXED_LO_DIMS.is_multiple_of(8)); // both planes byte-aligned

/// NF4 codebook: 16 levels at the quantiles of `N(0, 1)`, normalised to
/// `[-1, 1]` (denser near 0, where most latent mass lies). Ascending order.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.7075, -0.5421, -0.4165, -0.3108, -0.2158, -0.1272, -0.0421, 0.0421, 0.1272, 0.2158,
    0.3108, 0.4165, 0.5421, 0.7075, 1.0,
];
