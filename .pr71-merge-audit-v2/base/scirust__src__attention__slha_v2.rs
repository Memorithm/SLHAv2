//! SLHA v2 — Sub-Low Rank Hybrid Attention micro-kernel (reference).
//!
//! Layout-aware tile + the fused float/binary score of eq. (2.3):
//!
//! ```text
//! score = <q_coarse, dequant(latent)>  +  lambda * (d_s - 2 * popcount(q_sign ^ B))
//!         \_______ continuous _______/     \_____________ binary (sign-LSH) ______/
//! ```
//!
//! ## What changed vs. the v1 listing (see spec §5.1)
//! - **No `read_volatile`.** The v1 reference read every element through
//!   `core::ptr::read_volatile`, which forbids LLVM from vectorising/reordering
//!   and thus *defeats* the stated performance goals. The hot path is now plain,
//!   auto-vectorisable scalar code over slices.
//! - **Signed INT4 (zero-point).** Dequantisation is `(nibble - 8) * scale`, so
//!   the latent base can represent **negative** values (real keys are signed).
//!   v1 used `nibble * scale`, clamping the base to `[0, 15]·scale >= 0`.
//! - **Safe API, no bogus `target_feature`.** `count_ones()` already lowers to
//!   `POPCNT` when the target supports it, with a portable fallback otherwise;
//!   no `unsafe` and no misleading `avx2` gate (the body has no AVX2 intrinsics).
//! - **Tile is exactly 128 bytes with zero padding** (see [`SciRustSlhaTile`]).

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

/// Which codec a tile's latent uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentCodec {
    /// Uniform INT4, single per-tile scale.
    Int4Single,
    /// Uniform INT4, per-group (MX) scales.
    Int4Grouped,
    /// NF4 (normal-float) codebook, per-group scales.
    Nf4,
    /// Mixed precision: top [`MIXED_HI_DIMS`] dims at 8-bit, next
    /// [`MIXED_LO_DIMS`] at 4-bit, tail dropped. Built for steep real-key
    /// spectra (outlier channels) that uniform INT4 cannot span. Assumes the
    /// latent is ordered by decreasing variance (PCA order).
    Mixed,
    /// TurboQuant TQ3: 3-bit symmetric grid plus a separable 1-bit
    /// sign-correction plane, per-group scales. Same worst-case resolution
    /// as INT4 in the same 64 bytes; the correction plane can be paged out
    /// independently (see [`FLAG_TQ3`]).
    Tq3,
    /// MIX3 synthesis: the mixed codec's 8-bit head (top [`MIXED_HI_DIMS`]
    /// dims) + a TQ3 body ([`MIXED_LO_DIMS`] dims at 3-bit with a separable
    /// correction plane), tail dropped. Near-mixed quality on steep real
    /// spectra with the TQ3 paging rung (see [`FLAG_MIX3`]).
    Mix3,
}

/// A single SLHA v2 context tile.
///
/// The field set makes the type **exactly 128 bytes with no padding**. Alignment
/// defaults to `align(64)` — two 64-byte cache lines, 64-aligned so it never
/// straddles a third line — which is correct and optimal on every 64-byte-line
/// part, i.e. **all our targets**: x86-64, and AArch64/Neoverse-V3AE (measured at
/// 64 B across L1d/L1i/L2 on a Jetson Thor AGX 128 — the "128" there is the 128 GB
/// unified CPU/GPU LPDDR5X memory, not the cache line).
///
/// On a **native** build whose host has genuine 128-byte lines (e.g. Apple
/// Silicon), [`build.rs`](../../build.rs) detects it and sets `cfg(cache_line_128)`,
/// raising the tile to `align(128)` so it occupies a single line. The size is
/// 128 bytes either way (a multiple of both alignments), so zero padding holds.
///
/// The 24 bytes that were *tail padding* in the v1 layout (104 useful bytes
/// rounded up by the alignment) are now spent on useful per-tile metadata.
///
/// Byte map (offsets): latent 0..64 | residual 64..96 | scale 96 | lambda 100 |
/// residual_sigma 104 | token_id 108 | position 112 | head_id 116 |
/// flags 118 | group_scales 120..128.
#[cfg_attr(cache_line_128, repr(C, align(128)))]
#[cfg_attr(not(cache_line_128), repr(C, align(64)))]
#[derive(Clone, Copy)]
pub struct SciRustSlhaTile {
    /// Latent base `h_KV` (128 dims) quantised to signed INT4. 64 bytes.
    pub latent_kv: [u8; LATENT_BYTES],
    /// Johnson–Lindenstrauss sign residual: 256 bits. 32 bytes.
    pub residual_bitmap: [u64; RESIDUAL_WORDS],
    /// INT4 dequantisation **global** scale; per-group refinement in `group_scales`.
    pub scale: f32,
    /// Binary-correction weight λ (eq. 3.2), calibrated per tile.
    pub dynamic_lambda: f32,
    /// Per-tile residual energy estimate σ_E (kept for λ recalibration).
    pub residual_sigma: f32,
    /// Token identifier (causal/event-log bookkeeping).
    pub token_id: u32,
    /// Sequence position.
    pub position: u32,
    /// Attention head id.
    pub head_id: u16,
    /// State flags (`FLAG_HOT` / `FLAG_WARM`).
    pub flags: u16,
    /// Per-group micro-scaling bytes: `effective_scale(g) = scale · gs[g]/255`.
    /// (Was reserved padding; now refines the INT4 latent — keeps the tile 128 B.)
    pub group_scales: [u8; N_GROUPS],
}

impl SciRustSlhaTile {
    /// True if the residual has been paged out (WARM mode).
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.flags & FLAG_WARM != 0
    }

    /// True if the latent uses the NF4 codebook (else uniform INT4).
    #[inline]
    pub fn is_nf4(&self) -> bool {
        self.flags & FLAG_NF4 != 0
    }

    /// True if the latent uses the mixed-precision (8-bit head) layout.
    #[inline]
    pub fn is_mixed(&self) -> bool {
        self.flags & FLAG_MIXED != 0
    }

    /// True if the latent uses the TurboQuant TQ3 (3-bit + 1-bit) layout.
    #[inline]
    pub fn is_tq3(&self) -> bool {
        self.flags & FLAG_TQ3 != 0
    }

    /// True if the TQ3 correction plane has been paged out (see
    /// [`FLAG_TQ3_NOCORR`]): decode uses the bare 3-bit grid.
    #[inline]
    pub fn is_tq3_nocorr(&self) -> bool {
        self.flags & FLAG_TQ3_NOCORR != 0
    }

    /// True if the latent uses the MIX3 (mixed head × TQ3 body) layout.
    #[inline]
    pub fn is_mix3(&self) -> bool {
        self.flags & FLAG_MIX3 != 0
    }

    /// Size in bytes of this tile's separable correction plane (0 when the
    /// codec has none). This is what the CCOS correction-drop rung reclaims.
    #[inline]
    pub fn separable_corr_bytes(&self) -> usize {
        if self.is_tq3() {
            TQ3_CORR_BYTES
        } else if self.is_mix3() {
            MIX3_CORR_BYTES
        } else {
            0
        }
    }

    /// Effective dequant scale for dimension `d`: global scale × the dim's
    /// per-group micro-scale.
    #[inline]
    pub fn group_scale(&self, d: usize) -> f32 {
        self.scale * (self.group_scales[d / GROUP_DIM] as f32 / 255.0)
    }

    /// Dequantise one latent dimension `d` with its per-group scale, decoding
    /// the nibble via uniform INT4 (signed zero-point), the NF4 codebook, or
    /// the mixed-precision layout.
    #[inline]
    pub fn dequant_at(&self, d: usize) -> f32 {
        if self.is_mixed() {
            return self.dequant_at_mixed(d);
        }
        if self.is_tq3() {
            return self.dequant_at_tq3(d);
        }
        if self.is_mix3() {
            return self.dequant_at_mix3(d);
        }
        let byte = self.latent_kv[d >> 1];
        let nib = (if d & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as usize;
        let level = if self.is_nf4() {
            NF4_CODEBOOK[nib]
        } else {
            (nib as i32 - 8) as f32
        };
        level * self.group_scale(d)
    }

    /// Mixed layout: dims `0..MIXED_HI_DIMS` are signed bytes (zero-point 128)
    /// scaled by `group_scales[0]`; dims `MIXED_HI_DIMS..MIXED_DIMS` are
    /// nibbles in `GROUP_DIM`-wide groups scaled by `group_scales[1..]`; the
    /// dropped tail decodes to 0.
    #[inline]
    fn dequant_at_mixed(&self, d: usize) -> f32 {
        if d < MIXED_HI_DIMS {
            let level = self.latent_kv[d] as i32 - 128;
            level as f32 * (self.scale * self.group_scales[0] as f32 / 255.0)
        } else if d < MIXED_DIMS {
            let ld = d - MIXED_HI_DIMS;
            let byte = self.latent_kv[MIXED_HI_DIMS + (ld >> 1)];
            let nib = (if ld & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as i32;
            let g = 1 + ld / GROUP_DIM;
            (nib - 8) as f32 * (self.scale * self.group_scales[g] as f32 / 255.0)
        } else {
            0.0
        }
    }

    /// TQ3 layout: dim `d`'s 3-bit code is bits `[3d, 3d+3)` of the code
    /// plane; its correction sign is bit `d` of the correction plane. Decoded
    /// level = `(code − 3.5) ± TQ3_CORRECTION`, times the dim's group scale.
    /// With [`FLAG_TQ3_NOCORR`] the correction plane is considered freed and
    /// the bare grid level is returned.
    #[inline]
    fn dequant_at_tq3(&self, d: usize) -> f32 {
        let bit = 3 * d;
        let byte = bit >> 3;
        let shift = bit & 7;
        // A 3-bit field spans at most two bytes; the last code (d = 127)
        // ends exactly on the plane boundary, so guard the second read.
        let lo = u16::from(self.latent_kv[byte]);
        let hi = if byte + 1 < TQ3_CODE_BYTES {
            u16::from(self.latent_kv[byte + 1]) << 8
        } else {
            0
        };
        let code = ((lo | hi) >> shift) & 0x7;
        let mut level = code as f32 - TQ3_HALF_RANGE;
        if !self.is_tq3_nocorr() {
            let corr = (self.latent_kv[TQ3_CODE_BYTES + (d >> 3)] >> (d & 7)) & 1;
            let sign = if corr == 1 { 1.0 } else { -1.0 };
            level += sign * TQ3_CORRECTION;
        }
        level * self.group_scale(d)
    }

    /// MIX3 layout: dims `0..MIXED_HI_DIMS` decode exactly like the mixed
    /// head (signed bytes, `group_scales[0]`); dims
    /// `MIXED_HI_DIMS..MIXED_DIMS` decode like a TQ3 body — 3-bit code at
    /// bits `[3·ld, 3·ld+3)` of the plane starting at [`MIX3_CODES_OFF`],
    /// correction bit `ld` of the plane at [`MIX3_CORR_OFF`] (skipped with
    /// [`FLAG_TQ3_NOCORR`]) — scaled by `group_scales[1..]`; the dropped
    /// tail decodes to 0.
    #[inline]
    fn dequant_at_mix3(&self, d: usize) -> f32 {
        if d < MIXED_HI_DIMS {
            let level = self.latent_kv[d] as i32 - 128;
            return level as f32 * (self.scale * self.group_scales[0] as f32 / 255.0);
        }
        if d >= MIXED_DIMS {
            return 0.0;
        }
        let ld = d - MIXED_HI_DIMS;
        let bit = 3 * ld;
        let byte = MIX3_CODES_OFF + (bit >> 3);
        let shift = bit & 7;
        // A 3-bit field spans at most two bytes; the last code (ld = 111)
        // ends exactly on the plane boundary, so guard the second read.
        let lo = u16::from(self.latent_kv[byte]);
        let hi = if byte + 1 < MIX3_CORR_OFF {
            u16::from(self.latent_kv[byte + 1]) << 8
        } else {
            0
        };
        let code = ((lo | hi) >> shift) & 0x7;
        let mut level = code as f32 - TQ3_HALF_RANGE;
        if !self.is_tq3_nocorr() {
            let corr = (self.latent_kv[MIX3_CORR_OFF + (ld >> 3)] >> (ld & 7)) & 1;
            let sign = if corr == 1 { 1.0 } else { -1.0 };
            level += sign * TQ3_CORRECTION;
        }
        let g = 1 + ld / GROUP_DIM;
        level * (self.scale * self.group_scales[g] as f32 / 255.0)
    }

    /// Materialise the full dequantised latent vector (mostly for tests).
    pub fn dequant_latent(&self) -> [f32; D_C] {
        let mut out = [0.0f32; D_C];
        for (d, o) in out.iter_mut().enumerate() {
            *o = self.dequant_at(d);
        }
        out
    }

    /// Fused asymmetric attention score (eq. 2.3).
    ///
    /// `q_coarse` is `Q · W_up` in the latent space (`D_C` dims); `q_sign` is
    /// the packed sign of `Q · Zᵀ`. In WARM mode the binary term is dropped.
    ///
    /// Dispatches to an AVX-512/AVX2 (x86-64) or NEON (aarch64) path at
    /// runtime when available, else the portable scalar path. All paths yield
    /// the same result up to float reassociation.
    #[inline]
    pub fn compute_score(&self, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        // Every codec has a SIMD decode at every ISA level: uniform-INT4
        // nibbles, the NF4 codebook, the mixed 8-bit-head layout, TQ3
        // (3-bit code + 1-bit correction) and MIX3 (mixed head × TQ3 body).
        // The codec checks mirror `dequant_at`'s flag precedence; hosts with
        // neither AVX level fall back to the portable scalar path.
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by runtime feature detection.
                return unsafe {
                    if self.is_mixed() {
                        self.compute_score_avx512_mixed(q_coarse, q_sign)
                    } else if self.is_tq3() {
                        self.compute_score_avx512_tq3(q_coarse, q_sign)
                    } else if self.is_mix3() {
                        self.compute_score_avx512_mix3(q_coarse, q_sign)
                    } else if self.is_nf4() {
                        self.compute_score_avx512_nf4(q_coarse, q_sign)
                    } else {
                        self.compute_score_avx512(q_coarse, q_sign)
                    }
                };
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime feature detection.
                return unsafe {
                    if self.is_mixed() {
                        self.compute_score_avx2_mixed(q_coarse, q_sign)
                    } else if self.is_tq3() {
                        self.compute_score_avx2_tq3(q_coarse, q_sign)
                    } else if self.is_mix3() {
                        self.compute_score_avx2_mix3(q_coarse, q_sign)
                    } else if self.is_nf4() {
                        self.compute_score_avx2_nf4(q_coarse, q_sign)
                    } else {
                        self.compute_score_avx2(q_coarse, q_sign)
                    }
                };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64 — no runtime detection needed.
            // SAFETY: NEON is always available on this target.
            unsafe {
                if self.is_mixed() {
                    self.compute_score_neon_mixed(q_coarse, q_sign)
                } else if self.is_tq3() {
                    self.compute_score_neon_tq3(q_coarse, q_sign)
                } else if self.is_mix3() {
                    self.compute_score_neon_mix3(q_coarse, q_sign)
                } else if self.is_nf4() {
                    self.compute_score_neon_nf4(q_coarse, q_sign)
                } else {
                    self.compute_score_neon(q_coarse, q_sign)
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.compute_score_scalar(q_coarse, q_sign)
    }

    /// Binary 1-bit correction: λ · (d_s − 2·popcount(q_sign ^ B)).
    /// popcount(XOR) is the Hamming distance; d_s − 2·Hamming is the signed dot
    /// product of the two ±1 sign vectors.
    #[inline]
    fn residual_term(&self, q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        let hamming = hamming_distance(q_sign, &self.residual_bitmap);
        self.dynamic_lambda * (D_S as f32 - 2.0 * hamming as f32)
    }

    /// Portable scalar reference path.
    pub fn compute_score_scalar(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        let k = self.dequant_latent();
        let mut coarse = 0.0f32;
        for d in 0..D_C {
            coarse += q_coarse[d] * k[d];
        }
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 path: vectorised INT4 dequant + dot for the coarse term.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm256_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(nibble - 8) * group_scale` for 8 dims, multiply by q, accumulate.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let n = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($bytes));
                let v = _mm256_mul_ps($gs_v, _mm256_sub_ps(n, eight));
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            // Per-byte high nibble: shift 16-bit lanes, then mask each byte.
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            // Interleave so nibbles come out in dimension order.
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 TQ3 path: vectorised 3-bit + correction-bit dequant + dot for the
    /// coarse term of a [`FLAG_TQ3`] tile (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Decode strategy: 8 dims per step. Dim `8b+l` occupies bits
    /// `[3l, 3l+3)` of the 4-byte little-endian window at byte `3b`, so a
    /// broadcast of the window plus per-lane shifts `3l` (`vpsrlvd`) and a
    /// `& 7` mask denibbles the whole block; the matching correction bits are
    /// exactly byte `TQ3_CODE_BYTES + b`, extracted the same way with shifts
    /// `l` and `& 1`.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_tq3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
        let corr_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let seven = _mm256_set1_epi32(7);
        let one = _mm256_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // Fold the correction into the grid offset: level = code − 3.5 ±
        // 0.25 = code − 3.75 + corr·0.5 (all quarter-step values are exact
        // in f32, so this matches the scalar decode exactly per dim).
        let bias = _mm256_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm256_set1_ps(2.0 * TQ3_CORRECTION);
        let mut acc = _mm256_setzero_ps();
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 16 blocks × 8 dims (3 code bytes + 1 correction byte each) = 128
        // dims; one scale per 16-dim group = two consecutive blocks.
        for b in 0..D_C / 8 {
            let gs_v = _mm256_set1_ps(global * (self.group_scales[b / 2] as f32 * inv255));
            // 4-byte little-endian window holding all 8 code fields of this
            // block. The last block (b = 15) reads bytes 45..=48; byte 48 is
            // the first correction byte — in-bounds of `latent_kv`, and its
            // bits sit at window positions >= 24 > 3l + 2 for every lane
            // l <= 7, so the `& 7` mask after the per-lane shift keeps them
            // out of every decoded code.
            let window =
                u32::from_le_bytes([lk[3 * b], lk[3 * b + 1], lk[3 * b + 2], lk[3 * b + 3]]);
            let codes = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(window as i32), code_shifts),
                seven,
            );
            let mut level = _mm256_sub_ps(_mm256_cvtepi32_ps(codes), bias);
            if !nocorr {
                let cbyte = i32::from(lk[TQ3_CODE_BYTES + b]);
                let corr = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(cbyte), corr_shifts),
                    one,
                );
                level = _mm256_add_ps(level, _mm256_mul_ps(_mm256_cvtepi32_ps(corr), corr_step));
            }
            let v = _mm256_mul_ps(level, gs_v);
            let qv = _mm256_loadu_ps(q.add(8 * b));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 NF4 path: vectorised codebook dequant + dot for the coarse term
    /// of a [`FLAG_NF4`] tile.
    ///
    /// Decode strategy: nibbles unpack to i32 lanes exactly like
    /// [`Self::compute_score_avx2`]; the 16-entry [`NF4_CODEBOOK`] lookup is
    /// two `_mm256_permutevar8x32_ps` gathers (one per codebook half, both
    /// indexed by the low 3 nibble bits) blended on nibble bit 3.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_nf4(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let cb_lo = _mm256_loadu_ps(NF4_CODEBOOK.as_ptr());
        let cb_hi = _mm256_loadu_ps(NF4_CODEBOOK.as_ptr().add(8));
        let seven = _mm256_set1_epi32(7);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `NF4_CODEBOOK[nib] * group_scale` for 8 dims, multiply by
        // q, accumulate. `permutevar8x32` indexes with the low 3 bits only,
        // so both halves are gathered and nibble bit 3 picks the right one.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let nib = _mm256_cvtepu8_epi32($bytes);
                let lo_v = _mm256_permutevar8x32_ps(cb_lo, nib);
                let hi_v = _mm256_permutevar8x32_ps(cb_hi, nib);
                let pick_hi = _mm256_castsi256_ps(_mm256_cmpgt_epi32(nib, seven));
                let level = _mm256_blendv_ps(lo_v, hi_v, pick_hi);
                let v = _mm256_mul_ps($gs_v, level);
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 mixed-precision path: vectorised dequant + dot for the coarse
    /// term of a [`FLAG_MIXED`] tile.
    ///
    /// Decode strategy: the 8-bit head (dims `0..MIXED_HI_DIMS`) is a single
    /// 8-lane block — bytes to f32 minus the 128 zero-point, times the head
    /// scale `gs[0]`; the 4-bit body (dims `MIXED_HI_DIMS..MIXED_DIMS`) is
    /// [`Self::compute_score_avx2`]'s denibbling shifted 8 bytes/8 dims with
    /// group scales `gs[1..]`; the dropped tail (dims `MIXED_DIMS..D_C`)
    /// decodes to 0 and contributes nothing.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_mixed(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm256_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS, one signed byte each
        // (zero-point 128), scaled by gs[0].
        let hs_v = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            latent as *const __m128i,
        )));
        let hv = _mm256_mul_ps(hs_v, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm256_mul_ps(hv, _mm256_loadu_ps(q));

        // Dequant `(nibble - 8) * group_scale` for 8 dims, multiply by q, accumulate.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let n = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($bytes));
                let v = _mm256_mul_ps($gs_v, _mm256_sub_ps(n, eight));
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 4-bit body: 7 groups × 8 bytes = 112 dims after the head; one
        // scale byte per 16-dim group (gs[1..]).
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[1 + g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(MIXED_HI_DIMS + g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 MIX3 path: vectorised dequant + dot for the coarse term of a
    /// [`FLAG_MIX3`] tile (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Decode strategy: the 8-bit head is the mixed kernel's 8-lane block;
    /// the TQ3 body reuses `Self::compute_score_avx2_tq3`'s
    /// 8-dims-per-3-bytes window trick with every byte offset shifted to
    /// the MIX3 planes ([`MIX3_CODES_OFF`] / [`MIX3_CORR_OFF`]).
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_mix3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
        let corr_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let seven = _mm256_set1_epi32(7);
        let one = _mm256_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = _mm256_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm256_set1_ps(2.0 * TQ3_CORRECTION);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS, exactly the mixed kernel's block.
        let hs_v = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            lk.as_ptr() as *const __m128i
        )));
        let hv = _mm256_mul_ps(hs_v, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm256_mul_ps(hv, _mm256_loadu_ps(q));

        // TQ3 body: 14 blocks × 8 dims (3 code bytes + 1 correction byte
        // each) = 112 dims after the head; one scale per 16-dim group = two
        // consecutive blocks (gs[1..]).
        for b in 0..MIXED_LO_DIMS / 8 {
            let gs_v = _mm256_set1_ps(global * (self.group_scales[1 + b / 2] as f32 * inv255));
            // 4-byte little-endian window at MIX3_CODES_OFF + 3b holding all
            // 8 code fields of this block. The last block (b = 13) reads
            // bytes 47..=50; byte 50 is the first correction byte
            // (MIX3_CORR_OFF) — in-bounds of `latent_kv`, and its bits sit
            // at window positions >= 24 > 3l + 2 for every lane l <= 7, so
            // the `& 7` mask after the per-lane shift keeps them out of
            // every decoded code.
            let o = MIX3_CODES_OFF + 3 * b;
            let window = u32::from_le_bytes([lk[o], lk[o + 1], lk[o + 2], lk[o + 3]]);
            let codes = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(window as i32), code_shifts),
                seven,
            );
            let mut level = _mm256_sub_ps(_mm256_cvtepi32_ps(codes), bias);
            if !nocorr {
                let cbyte = i32::from(lk[MIX3_CORR_OFF + b]);
                let corr = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(cbyte), corr_shifts),
                    one,
                );
                level = _mm256_add_ps(level, _mm256_mul_ps(_mm256_cvtepi32_ps(corr), corr_step));
            }
            let v = _mm256_mul_ps(level, gs_v);
            let qv = _mm256_loadu_ps(q.add(MIXED_HI_DIMS + 8 * b));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 path: one 16-wide FMA per group (16 latent dims).
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm512_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm512_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // One group (16 dims = 8 bytes) per 16-wide fused multiply-add.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // 16 bytes = dims base..base+15
            let n = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(d16));
            let v = _mm512_mul_ps(gs, _mm512_sub_ps(n, eight));
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 TQ3 path: one 16-wide FMA per group for a [`FLAG_TQ3`] tile
    /// (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Same decode strategy as `Self::compute_score_avx2_tq3`, two 8-dim
    /// blocks per iteration: lanes 0..8 shift the 4-byte window at byte `6g`,
    /// lanes 8..16 the window at byte `6g + 3`, each by `3·(l mod 8)`; the
    /// group's 16 correction bits are the `u16` at `TQ3_CODE_BYTES + 2g`,
    /// shifted per lane by `l`.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_tq3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        #[rustfmt::skip]
        let code_shifts = _mm512_setr_epi32(
            0, 3, 6, 9, 12, 15, 18, 21,
            0, 3, 6, 9, 12, 15, 18, 21,
        );
        #[rustfmt::skip]
        let corr_shifts = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        );
        let seven = _mm512_set1_epi32(7);
        let one = _mm512_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = _mm512_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm512_set1_ps(2.0 * TQ3_CORRECTION);
        let mut acc = _mm512_setzero_ps();
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // One group (16 dims = 6 code bytes + 2 correction bytes) per
        // 16-wide fused multiply-add.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            // Two 4-byte little-endian windows: dims base..base+8 live at
            // bits [3l, 3l+3) of w0 (byte 6g), dims base+8..base+16 at the
            // same offsets of w1 (byte 6g+3). For g = 7, w1 reads bytes
            // 45..=48; byte 48 is the first correction byte — in-bounds of
            // `latent_kv`, and its bits sit at window positions >= 24, which
            // the shift-then-`& 7` can never bring into a decoded code.
            let w0 = u32::from_le_bytes([lk[6 * g], lk[6 * g + 1], lk[6 * g + 2], lk[6 * g + 3]]);
            let w1 =
                u32::from_le_bytes([lk[6 * g + 3], lk[6 * g + 4], lk[6 * g + 5], lk[6 * g + 6]]);
            let windows = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi32(w0 as i32)),
                _mm256_set1_epi32(w1 as i32),
                1,
            );
            let codes = _mm512_and_epi32(_mm512_srlv_epi32(windows, code_shifts), seven);
            let mut level = _mm512_sub_ps(_mm512_cvtepi32_ps(codes), bias);
            if !nocorr {
                let c16 = i32::from(lk[TQ3_CODE_BYTES + 2 * g])
                    | (i32::from(lk[TQ3_CODE_BYTES + 2 * g + 1]) << 8);
                let corr =
                    _mm512_and_epi32(_mm512_srlv_epi32(_mm512_set1_epi32(c16), corr_shifts), one);
                level = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr), corr_step, level);
            }
            let v = _mm512_mul_ps(level, gs);
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 NF4 path: one 16-wide FMA per group for a [`FLAG_NF4`] tile.
    ///
    /// Decode strategy: same denibbling as [`Self::compute_score_avx512`];
    /// the 16-entry [`NF4_CODEBOOK`] fits a single 512-bit register, so the
    /// codebook lookup is one `_mm512_permutexvar_ps` per group.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_nf4(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let codebook = _mm512_loadu_ps(NF4_CODEBOOK.as_ptr());
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm512_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // One group (16 dims = 8 bytes) per 16-wide fused multiply-add.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // 16 bytes = dims base..base+15
            let level = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(d16), codebook);
            let v = _mm512_mul_ps(gs, level);
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 mixed-precision path: one 16-wide FMA per body group for a
    /// [`FLAG_MIXED`] tile.
    ///
    /// Same decode strategy as `Self::compute_score_avx2_mixed`: the
    /// 8-bit head is one 8-lane block (zero-extended into the 16-wide
    /// accumulator), the 4-bit body is [`Self::compute_score_avx512`]'s
    /// denibbling shifted 8 bytes/8 dims with group scales `gs[1..]`, and
    /// the dropped tail contributes nothing.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_mixed(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm512_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS as one 8-lane block (zero-point
        // 128, scale gs[0]), zero-extended into the 16-wide accumulator.
        let hs = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            latent as *const __m128i,
        )));
        let hv = _mm256_mul_ps(hs, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm512_zextps256_ps512(_mm256_mul_ps(hv, _mm256_loadu_ps(q)));

        // 4-bit body: one group (16 dims = 8 bytes after the head) per
        // 16-wide fused multiply-add, scales gs[1..].
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[1 + g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(MIXED_HI_DIMS + g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi);
            let n = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(d16));
            let v = _mm512_mul_ps(gs, _mm512_sub_ps(n, eight));
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 MIX3 path: one 16-wide FMA per body group for a
    /// [`FLAG_MIX3`] tile (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Same decode strategy as [`Self::compute_score_avx512_tq3`] for the
    /// body — two 4-byte windows per group, shifted to the MIX3 planes
    /// ([`MIX3_CODES_OFF`] / [`MIX3_CORR_OFF`]) — with the mixed kernel's
    /// 8-lane head zero-extended into the 16-wide accumulator.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_mix3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        #[rustfmt::skip]
        let code_shifts = _mm512_setr_epi32(
            0, 3, 6, 9, 12, 15, 18, 21,
            0, 3, 6, 9, 12, 15, 18, 21,
        );
        #[rustfmt::skip]
        let corr_shifts = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        );
        let seven = _mm512_set1_epi32(7);
        let one = _mm512_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = _mm512_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm512_set1_ps(2.0 * TQ3_CORRECTION);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 8-bit head: same 8-lane block as the mixed kernel, zero-extended
        // into the 16-wide accumulator.
        let hs = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            lk.as_ptr() as *const __m128i
        )));
        let hv = _mm256_mul_ps(hs, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm512_zextps256_ps512(_mm256_mul_ps(hv, _mm256_loadu_ps(q)));

        // TQ3 body: one group (16 dims = 6 code bytes + 2 correction bytes
        // after the head) per 16-wide fused multiply-add, scales gs[1..].
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[1 + g] as f32 * inv255));
            // Two 4-byte little-endian windows on the MIX3 code plane: body
            // dims base..base+8 live at bits [3l, 3l+3) of w0 (byte
            // MIX3_CODES_OFF + 6g), dims base+8..base+16 at the same offsets
            // of w1 (3 bytes later). For g = 6, w1 reads bytes 47..=50; byte
            // 50 is the first correction byte (MIX3_CORR_OFF) — in-bounds of
            // `latent_kv`, and its bits sit at window positions >= 24, which
            // the shift-then-`& 7` can never bring into a decoded code.
            let o = MIX3_CODES_OFF + 6 * g;
            let w0 = u32::from_le_bytes([lk[o], lk[o + 1], lk[o + 2], lk[o + 3]]);
            let w1 = u32::from_le_bytes([lk[o + 3], lk[o + 4], lk[o + 5], lk[o + 6]]);
            let windows = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi32(w0 as i32)),
                _mm256_set1_epi32(w1 as i32),
                1,
            );
            let codes = _mm512_and_epi32(_mm512_srlv_epi32(windows, code_shifts), seven);
            let mut level = _mm512_sub_ps(_mm512_cvtepi32_ps(codes), bias);
            if !nocorr {
                let c16 = i32::from(lk[MIX3_CORR_OFF + 2 * g])
                    | (i32::from(lk[MIX3_CORR_OFF + 2 * g + 1]) << 8);
                let corr =
                    _mm512_and_epi32(_mm512_srlv_epi32(_mm512_set1_epi32(c16), corr_shifts), one);
                level = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr), corr_step, level);
            }
            let v = _mm512_mul_ps(level, gs);
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON path (aarch64): vectorised INT4 dequant + dot for the coarse term.
    /// NEON is baseline on aarch64, so the dispatcher calls this unconditionally.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = vdupq_n_f32(8.0);
        let mask = vdup_n_u8(0x0F);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(n - 8) * group_scale` for a 4-lane chunk, then FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let v = vmulq_f32(vsubq_f32($n4, eight), $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[g] as f32 * inv255));
            let packed = vld1_u8(latent.add(g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_lo))), base, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_lo))), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_hi))), base + 8, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_hi))), base + 12, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON TQ3 path (aarch64): vectorised 3-bit + correction-bit dequant +
    /// dot for the coarse term of a [`FLAG_TQ3`] tile (honours
    /// [`FLAG_TQ3_NOCORR`]). The dispatcher calls this unconditionally on
    /// aarch64, like [`Self::compute_score_neon`].
    ///
    /// Same decode strategy as `Self::compute_score_avx2_tq3`, split into
    /// two 4-lane quads per 8-dim block: `vshlq_u32` with negative per-lane
    /// counts is the NEON variable right-shift.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_tq3_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_tq3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        // Negative counts = right shifts: code fields at bits 3l, correction
        // bits at bit l, for lanes l of the low/high quad of an 8-dim block.
        const CODE_SH_LO: [i32; 4] = [0, -3, -6, -9];
        const CODE_SH_HI: [i32; 4] = [-12, -15, -18, -21];
        const CORR_SH_LO: [i32; 4] = [0, -1, -2, -3];
        const CORR_SH_HI: [i32; 4] = [-4, -5, -6, -7];

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_sh_lo = vld1q_s32(CODE_SH_LO.as_ptr());
        let code_sh_hi = vld1q_s32(CODE_SH_HI.as_ptr());
        let corr_sh_lo = vld1q_s32(CORR_SH_LO.as_ptr());
        let corr_sh_hi = vld1q_s32(CORR_SH_HI.as_ptr());
        let seven = vdupq_n_u32(7);
        let one = vdupq_n_u32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = vdupq_n_f32(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = vdupq_n_f32(2.0 * TQ3_CORRECTION);
        let mut acc = vdupq_n_f32(0.0);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // Decode 4 dims from the block's window/correction byte, FMA with q.
        macro_rules! quad {
            ($window:expr, $cbyte:expr, $code_sh:expr, $corr_sh:expr, $off:expr, $gs:expr) => {{
                let codes = vandq_u32(vshlq_u32(vdupq_n_u32($window), $code_sh), seven);
                let mut level = vsubq_f32(vcvtq_f32_u32(codes), bias);
                if !nocorr {
                    let corr = vandq_u32(vshlq_u32(vdupq_n_u32($cbyte), $corr_sh), one);
                    level = vfmaq_f32(level, vcvtq_f32_u32(corr), corr_step);
                }
                let v = vmulq_f32(level, $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 16 blocks × 8 dims (3 code bytes + 1 correction byte each) = 128
        // dims; one scale per 16-dim group = two consecutive blocks.
        for b in 0..D_C / 8 {
            let gs = vdupq_n_f32(global * (self.group_scales[b / 2] as f32 * inv255));
            // 4-byte little-endian window holding all 8 code fields of this
            // block. The last block (b = 15) reads bytes 45..=48; byte 48 is
            // the first correction byte — in-bounds of `latent_kv`, and its
            // bits sit at window positions >= 24 > 3l + 2 for every lane
            // l <= 7, so the `& 7` mask after the per-lane shift keeps them
            // out of every decoded code.
            let window =
                u32::from_le_bytes([lk[3 * b], lk[3 * b + 1], lk[3 * b + 2], lk[3 * b + 3]]);
            let cbyte = u32::from(lk[TQ3_CODE_BYTES + b]);
            quad!(window, cbyte, code_sh_lo, corr_sh_lo, 8 * b, gs);
            quad!(window, cbyte, code_sh_hi, corr_sh_hi, 8 * b + 4, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON NF4 path (aarch64): vectorised codebook dequant + dot for the
    /// coarse term of a [`FLAG_NF4`] tile. The dispatcher calls this
    /// unconditionally on aarch64, like [`Self::compute_score_neon`].
    ///
    /// Decode strategy: same denibbling as [`Self::compute_score_neon`];
    /// the 16-entry f32 [`NF4_CODEBOOK`] is viewed as a 64-byte table
    /// (aarch64 is little-endian, so byte `k` of level `n` sits at table
    /// index `4n + k`) and one `vqtbl4q_u8` per quad gathers 4 whole levels.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_nf4_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_nf4(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let mask = vdup_n_u8(0x0F);
        let codebook = vld1q_u8_x4(NF4_CODEBOOK.as_ptr() as *const u8);
        // Little-endian byte offsets 0..=3 of an f32, one per index byte.
        let byte_lanes = vdupq_n_u32(0x0302_0100);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Gather 4 codebook levels from a quad of nibble indices ($n4:
        // uint32x4, values 0..=15): each u32 lane becomes the four table
        // indices `4n + {0,1,2,3}` (all < 64), one `vqtbl4q_u8` reads the 4
        // f32s whole; then scale and FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let idx = vaddq_u32(vmulq_n_u32($n4, 0x0404_0404), byte_lanes);
                let level = vreinterpretq_f32_u8(vqtbl4q_u8(codebook, vreinterpretq_u8_u32(idx)));
                let v = vmulq_f32(level, $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[g] as f32 * inv255));
            let packed = vld1_u8(latent.add(g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vmovl_u16(vget_low_u16(w_lo)), base, gs);
            quad!(vmovl_u16(vget_high_u16(w_lo)), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vmovl_u16(vget_low_u16(w_hi)), base + 8, gs);
            quad!(vmovl_u16(vget_high_u16(w_hi)), base + 12, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON mixed-precision path (aarch64): vectorised dequant + dot for the
    /// coarse term of a [`FLAG_MIXED`] tile. The dispatcher calls this
    /// unconditionally on aarch64, like [`Self::compute_score_neon`].
    ///
    /// Same decode strategy as `Self::compute_score_avx2_mixed`, in
    /// 4-lane quads: the 8-bit head (zero-point 128, scale `gs[0]`) as two
    /// quads, then [`Self::compute_score_neon`]'s denibbling shifted
    /// 8 bytes/8 dims with group scales `gs[1..]`; the dropped tail decodes
    /// to 0 and contributes nothing.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_mixed_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_mixed(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = vdupq_n_f32(8.0);
        let zero_point = vdupq_n_f32(128.0);
        let mask = vdup_n_u8(0x0F);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS, one signed byte each
        // (zero-point 128, scale gs[0]), as two 4-lane FMAs.
        {
            let hs = vdupq_n_f32(global * (self.group_scales[0] as f32 * inv255));
            let w = vmovl_u8(vld1_u8(latent)); // u16×8
            let n_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
            let n_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_lo, zero_point), hs),
                vld1q_f32(q),
            );
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_hi, zero_point), hs),
                vld1q_f32(q.add(4)),
            );
        }

        // Dequant `(n - 8) * group_scale` for a 4-lane chunk, then FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let v = vmulq_f32(vsubq_f32($n4, eight), $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 4-bit body: 7 groups × 8 bytes = 112 dims after the head; one
        // scale byte per 16-dim group (gs[1..]).
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[1 + g] as f32 * inv255));
            let packed = vld1_u8(latent.add(MIXED_HI_DIMS + g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_lo))), base, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_lo))), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_hi))), base + 8, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_hi))), base + 12, gs);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON MIX3 path (aarch64): vectorised dequant + dot for the coarse
    /// term of a [`FLAG_MIX3`] tile (honours [`FLAG_TQ3_NOCORR`]). The
    /// dispatcher calls this unconditionally on aarch64, like
    /// [`Self::compute_score_neon`].
    ///
    /// Same decode strategy as [`Self::compute_score_neon_tq3`] for the body
    /// — two 4-lane quads per 8-dim block, every byte offset shifted to the
    /// MIX3 planes ([`MIX3_CODES_OFF`] / [`MIX3_CORR_OFF`]) — with the mixed
    /// kernel's two head quads in front.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_mix3_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_mix3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        // Negative counts = right shifts: code fields at bits 3l, correction
        // bits at bit l, for lanes l of the low/high quad of an 8-dim block.
        const CODE_SH_LO: [i32; 4] = [0, -3, -6, -9];
        const CODE_SH_HI: [i32; 4] = [-12, -15, -18, -21];
        const CORR_SH_LO: [i32; 4] = [0, -1, -2, -3];
        const CORR_SH_HI: [i32; 4] = [-4, -5, -6, -7];

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_sh_lo = vld1q_s32(CODE_SH_LO.as_ptr());
        let code_sh_hi = vld1q_s32(CODE_SH_HI.as_ptr());
        let corr_sh_lo = vld1q_s32(CORR_SH_LO.as_ptr());
        let corr_sh_hi = vld1q_s32(CORR_SH_HI.as_ptr());
        let seven = vdupq_n_u32(7);
        let one = vdupq_n_u32(1);
        let zero_point = vdupq_n_f32(128.0);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = vdupq_n_f32(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = vdupq_n_f32(2.0 * TQ3_CORRECTION);
        let mut acc = vdupq_n_f32(0.0);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS (zero-point 128, scale gs[0]),
        // exactly the mixed kernel's two quads.
        {
            let hs = vdupq_n_f32(global * (self.group_scales[0] as f32 * inv255));
            let w = vmovl_u8(vld1_u8(lk.as_ptr())); // u16×8
            let n_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
            let n_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_lo, zero_point), hs),
                vld1q_f32(q),
            );
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_hi, zero_point), hs),
                vld1q_f32(q.add(4)),
            );
        }

        // Decode 4 dims from the block's window/correction byte, FMA with q.
        macro_rules! quad {
            ($window:expr, $cbyte:expr, $code_sh:expr, $corr_sh:expr, $off:expr, $gs:expr) => {{
                let codes = vandq_u32(vshlq_u32(vdupq_n_u32($window), $code_sh), seven);
                let mut level = vsubq_f32(vcvtq_f32_u32(codes), bias);
                if !nocorr {
                    let corr = vandq_u32(vshlq_u32(vdupq_n_u32($cbyte), $corr_sh), one);
                    level = vfmaq_f32(level, vcvtq_f32_u32(corr), corr_step);
                }
                let v = vmulq_f32(level, $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // TQ3 body: 14 blocks × 8 dims (3 code bytes + 1 correction byte
        // each) = 112 dims after the head; one scale per 16-dim group = two
        // consecutive blocks (gs[1..]).
        for b in 0..MIXED_LO_DIMS / 8 {
            let gs = vdupq_n_f32(global * (self.group_scales[1 + b / 2] as f32 * inv255));
            // 4-byte little-endian window at MIX3_CODES_OFF + 3b holding all
            // 8 code fields of this block. The last block (b = 13) reads
            // bytes 47..=50; byte 50 is the first correction byte
            // (MIX3_CORR_OFF) — in-bounds of `latent_kv`, and its bits sit
            // at window positions >= 24 > 3l + 2 for every lane l <= 7, so
            // the `& 7` mask after the per-lane shift keeps them out of
            // every decoded code.
            let o = MIX3_CODES_OFF + 3 * b;
            let window = u32::from_le_bytes([lk[o], lk[o + 1], lk[o + 2], lk[o + 3]]);
            let cbyte = u32::from(lk[MIX3_CORR_OFF + b]);
            quad!(
                window,
                cbyte,
                code_sh_lo,
                corr_sh_lo,
                MIXED_HI_DIMS + 8 * b,
                gs
            );
            quad!(
                window,
                cbyte,
                code_sh_hi,
                corr_sh_hi,
                MIXED_HI_DIMS + 8 * b + 4,
                gs
            );
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }
}

/// Hamming distance over the 256-bit residual: `Σ popcount(aᵢ ⊕ bᵢ)`, the hot
/// inner term of eq. (2.3) (`d_s − 2·Hamming` is the signed ±1 dot product).
///
/// **Branchless**: a fixed 4-word reduction with no data-dependent control flow
/// (important for in-order issue / tight pipelines like ARM Neoverse). On
/// x86-64 CPUs advertising AVX-512 `VPOPCNTDQ`+`VL` it folds the whole 256-bit
/// residual into a single vector `vpopcntq`; otherwise it falls back to
/// `u64::count_ones`, which already lowers to `POPCNT` (x86) / `CNT` (AArch64
/// NEON). The one-time feature-detection branch is predicted and cached.
#[inline]
pub fn hamming_distance(a: &[u64; RESIDUAL_WORDS], b: &[u64; RESIDUAL_WORDS]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512vl") {
            // SAFETY: both required target features were just detected at runtime.
            return unsafe { hamming_vpopcntdq(a, b) };
        }
    }
    hamming_scalar(a, b)
}

/// Portable branchless reference: 4× `count_ones` (→ `POPCNT`/`CNT`).
#[inline]
fn hamming_scalar(a: &[u64; RESIDUAL_WORDS], b: &[u64; RESIDUAL_WORDS]) -> u32 {
    let mut h = 0u32;
    for w in 0..RESIDUAL_WORDS {
        h += (a[w] ^ b[w]).count_ones();
    }
    h
}

/// AVX-512 VPOPCNTDQ path: XOR the two 256-bit residuals and popcount all four
/// 64-bit lanes in a single `vpopcntq`, then reduce. Equivalent to
/// [`hamming_scalar`] (asserted by `vpopcntdq_hamming_matches_scalar` on capable
/// CPUs; compile-checked elsewhere).
///
/// # Safety
/// Requires the `avx512vpopcntdq` and `avx512vl` target features; the only
/// caller ([`hamming_distance`]) gates this behind `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
unsafe fn hamming_vpopcntdq(a: &[u64; RESIDUAL_WORDS], b: &[u64; RESIDUAL_WORDS]) -> u32 {
    use core::arch::x86_64::*;
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let pc = _mm256_popcnt_epi64(_mm256_xor_si256(va, vb)); // per-lane popcount
    let mut lanes = [0u64; RESIDUAL_WORDS];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, pc);
    (lanes[0] + lanes[1] + lanes[2] + lanes[3]) as u32
}

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

/// Index of the nearest NF4 codebook level to `t` (expects `t ∈ [-1, 1]`).
#[inline]
fn nf4_nearest(t: f32) -> u8 {
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
/// exactly like [`quantize_latent_grouped`]. Returns `(nibbles, global, gs)`.
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

/// TurboQuant TQ3 quantisation (see [`FLAG_TQ3`]): every dim to a 3-bit
/// symmetric grid (codes 0..=7 → levels `code − 3.5`, no zero level) plus a
/// 1-bit residual sign correction of [`TQ3_CORRECTION`] step, per-group
/// scales exactly like [`quantize_latent_grouped`]. The 64-byte budget is
/// split 48 B codes + 16 B correction bits. Returns `(bytes, global, gs)`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn tile_is_exactly_128_bytes_zero_padding() {
        // Default align(64) (= two 64-byte lines) on every 64-byte-line part,
        // including all our targets (x86-64 and AArch64/Neoverse — the Thor
        // measures 64 B). build.rs bumps to align(128) only on a native
        // 128-byte-line host (cfg cache_line_128). Size is 128 B with no padding
        // either way.
        let expected_align = if cfg!(cache_line_128) { 128 } else { 64 };
        assert_eq!(align_of::<SciRustSlhaTile>(), expected_align);
        assert_eq!(size_of::<SciRustSlhaTile>(), 128);

        // Sum of field sizes == struct size  =>  no padding anywhere.
        let field_bytes = LATENT_BYTES            // latent_kv
            + RESIDUAL_WORDS * 8                  // residual_bitmap
            + 4 + 4 + 4                           // scale, dynamic_lambda, residual_sigma
            + 4 + 4                               // token_id, position
            + 2 + 2                               // head_id, flags
            + 8; // group_scales
        assert_eq!(field_bytes, 128);
        assert_eq!(field_bytes, size_of::<SciRustSlhaTile>());
    }

    fn tile_from(
        latent_kv: [u8; LATENT_BYTES],
        scale: f32,
        group_scales: [u8; N_GROUPS],
    ) -> SciRustSlhaTile {
        SciRustSlhaTile {
            latent_kv,
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: FLAG_HOT,
            group_scales,
        }
    }

    #[test]
    fn int4_dequant_round_trips_signed_values() {
        // A vector with both signs must survive quantise -> dequantise within
        // one quantisation step, and crucially keep negative values negative.
        let mut v = [0.0f32; D_C];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32) - 64.0) / 16.0; // spans negative and positive
        }
        let (packed, scale) = quantize_latent(&v);
        // [255; N_GROUPS] makes every group's effective scale == the global scale,
        // i.e. exactly the single-scale behaviour.
        let tile = tile_from(packed, scale, [255; N_GROUPS]);
        let dq = tile.dequant_latent();
        // At least one strictly-negative reconstructed value (zero-point works).
        assert!(
            dq.iter().any(|&x| x < 0.0),
            "no negative values reconstructed"
        );
        for d in 0..D_C {
            assert!(
                (dq[d] - v[d]).abs() <= scale + 1e-6,
                "dim {d}: |{} - {}| > step {scale}",
                dq[d],
                v[d]
            );
        }
    }

    #[test]
    fn grouped_int4_beats_single_on_spread_variance() {
        // Per-group magnitudes spanning orders of magnitude (like PCA components
        // ordered by eigenvalue): a single global scale crushes the small
        // groups, per-group scaling does not.
        let mut v = [0.0f32; D_C];
        let mut rng = crate::rng::Rng::new(5);
        for g in 0..N_GROUPS {
            let amp = 10f32.powi(-(g as i32)); // 1, 0.1, 0.01, ...
            for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
                v[d] = amp * rng.next_gaussian();
            }
        }
        let sq_err = |t: &SciRustSlhaTile| -> f32 {
            let dq = t.dequant_latent();
            (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
        };
        let (p1, s1) = quantize_latent(&v);
        let e_single = sq_err(&tile_from(p1, s1, [255; N_GROUPS]));
        let (p2, s2, gs2) = quantize_latent_grouped(&v);
        let e_grouped = sq_err(&tile_from(p2, s2, gs2));
        assert!(
            e_grouped < e_single * 0.5,
            "grouped err {e_grouped} not clearly < single err {e_single}"
        );
    }

    #[test]
    fn nf4_beats_uniform_int4_on_gaussian_latent() {
        // NF4's normal-quantile codebook should reconstruct Gaussian latent
        // values more accurately than uniform INT4 at the same 4-bit budget.
        let mut v = [0.0f32; D_C];
        let mut rng = crate::rng::Rng::new(8);
        for x in v.iter_mut() {
            *x = rng.next_gaussian();
        }
        let sq_err = |t: &SciRustSlhaTile| -> f32 {
            let dq = t.dequant_latent();
            (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
        };

        let (p1, s1, g1) = quantize_latent_grouped(&v);
        let e_uniform = sq_err(&tile_from(p1, s1, g1)); // FLAG_HOT -> uniform

        let (p2, s2, g2) = quantize_latent_nf4(&v);
        let mut nf4 = tile_from(p2, s2, g2);
        nf4.flags |= FLAG_NF4;
        let e_nf4 = sq_err(&nf4);

        assert!(
            e_nf4 < e_uniform,
            "NF4 err {e_nf4} not < uniform {e_uniform}"
        );
    }

    /// A steep, GPT-2-like latent spectrum (the measured motivation for the
    /// codec): per-dim std ~ 37·(d+1)^-0.9, i.e. λ0/λ63 ≈ 40× in std.
    fn steep_latent(seed: u64) -> [f32; D_C] {
        let mut rng = crate::rng::Rng::new(seed);
        let mut v = [0.0f32; D_C];
        for (d, x) in v.iter_mut().enumerate() {
            let amp = 37.0 * ((d + 1) as f32).powf(-0.9);
            *x = amp * rng.next_gaussian();
        }
        v
    }

    #[test]
    fn mixed_dequant_roundtrips_and_drops_tail() {
        let v = steep_latent(11);
        let (packed, global, gs) = quantize_latent_mixed(&v);
        let mut tile = tile_from(packed, global, gs);
        tile.flags |= FLAG_MIXED;
        let dq = tile.dequant_latent();
        // 8-bit head: error within one 8-bit step.
        let eff_hi = global * (gs[0] as f32 / 255.0);
        for d in 0..MIXED_HI_DIMS {
            assert!(
                (dq[d] - v[d]).abs() <= eff_hi + 1e-6,
                "hi dim {d}: |{} - {}| > step {eff_hi}",
                dq[d],
                v[d]
            );
        }
        // 4-bit body: error within one step of its group.
        for d in MIXED_HI_DIMS..MIXED_DIMS {
            let g = 1 + (d - MIXED_HI_DIMS) / GROUP_DIM;
            let eff = global * (gs[g] as f32 / 255.0);
            assert!(
                (dq[d] - v[d]).abs() <= eff + 1e-6,
                "lo dim {d}: |{} - {}| > step {eff}",
                dq[d],
                v[d]
            );
        }
        // Dropped tail decodes to exactly 0.
        for d in MIXED_DIMS..D_C {
            assert_eq!(dq[d], 0.0, "tail dim {d} must decode to 0");
        }
    }

    #[test]
    fn mixed_beats_uniform_int4_on_steep_spectrum() {
        // On the steep spectrum the codec was built for, the 8-bit head must
        // cut the reconstruction error decisively — including the price of the
        // dropped tail (which carries ~no energy at this decay).
        let sq_err = |t: &SciRustSlhaTile, v: &[f32; D_C]| -> f32 {
            let dq = t.dequant_latent();
            (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
        };
        let (mut worse, mut total) = (0, 0);
        for seed in 0..8u64 {
            let v = steep_latent(100 + seed);
            let (p1, s1, g1) = quantize_latent_grouped(&v);
            let e_uniform = sq_err(&tile_from(p1, s1, g1), &v);
            let (p2, s2, g2) = quantize_latent_mixed(&v);
            let mut mixed = tile_from(p2, s2, g2);
            mixed.flags |= FLAG_MIXED;
            let e_mixed = sq_err(&mixed, &v);
            total += 1;
            if e_mixed >= e_uniform * 0.5 {
                worse += 1;
            }
        }
        assert_eq!(
            worse, 0,
            "mixed did not halve the uniform error on {worse}/{total} steep latents"
        );
    }

    #[test]
    fn tq3_dequant_roundtrips_within_quarter_step() {
        // The same 16 values tiled across every group ⇒ identical per-group
        // absmax ⇒ every gs byte is 255 and the effective step is exactly the
        // global scale: the strict quarter-step bound of the corrected grid
        // must hold on every dim.
        let mut pattern = [0.0f32; GROUP_DIM];
        let mut rng = crate::rng::Rng::new(17);
        for x in pattern.iter_mut() {
            *x = rng.next_gaussian();
        }
        let mut v = [0.0f32; D_C];
        for (d, x) in v.iter_mut().enumerate() {
            *x = pattern[d % GROUP_DIM];
        }
        let (packed, global, gs) = quantize_latent_tq3(&v);
        assert_eq!(gs, [255u8; N_GROUPS], "equal-spread groups must share gs");
        let mut tile = tile_from(packed, global, gs);
        tile.flags |= FLAG_TQ3;
        let dq = tile.dequant_latent();
        for d in 0..D_C {
            assert!(
                (dq[d] - v[d]).abs() <= TQ3_CORRECTION * global + 1e-5,
                "dim {d}: |{} - {}| > quarter step {}",
                dq[d],
                v[d],
                TQ3_CORRECTION * global
            );
        }
    }

    #[test]
    fn tq3_roundtrips_on_steep_spectrum() {
        // On the realistic steep spectrum, per-group scaling plus the u8
        // scale rounding keeps every dim within one grid step of the truth
        // (same bound convention as the grouped INT4 test).
        let v = steep_latent(19);
        let (packed, global, gs) = quantize_latent_tq3(&v);
        let mut tile = tile_from(packed, global, gs);
        tile.flags |= FLAG_TQ3;
        let dq = tile.dequant_latent();
        for d in 0..D_C {
            let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
            assert!(
                (dq[d] - v[d]).abs() <= eff + 1e-6,
                "dim {d}: |{} - {}| > step {eff}",
                dq[d],
                v[d]
            );
        }
    }

    #[test]
    fn tq3_positive_values_do_not_collapse() {
        // Regression guard for the upstream TurboQuant defect this port
        // fixes: a mis-sized grid (15 levels clamped to 8) decoded every
        // positive input to 0. Both signs must survive the round trip.
        let mut v = [0.0f32; D_C];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32) - 64.0) / 16.0; // spans negative and positive
        }
        let (packed, global, gs) = quantize_latent_tq3(&v);
        let mut tile = tile_from(packed, global, gs);
        tile.flags |= FLAG_TQ3;
        let dq = tile.dequant_latent();
        let mut pos = 0;
        for d in 0..D_C {
            let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
            // Any value at least one step from zero must keep its sign.
            if v[d] >= eff {
                assert!(dq[d] > 0.0, "dim {d}: positive {} decoded {}", v[d], dq[d]);
                pos += 1;
            }
            if v[d] <= -eff {
                assert!(dq[d] < 0.0, "dim {d}: negative {} decoded {}", v[d], dq[d]);
            }
        }
        assert!(pos > 0, "test vector must exercise the positive half");
    }

    #[test]
    fn tq3_error_matches_int4_grouped_resolution() {
        // 3-bit + 1-bit correction spends the same 4 bits/dim as INT4, and
        // both grids share the same worst-case error (0.25 step). On a
        // Gaussian latent TQ3 pays a measured ~1.3–1.6× MSE penalty for its
        // missing zero level (near-zero mass always costs ≥ 0.25 step) — the
        // documented trade for the separable correction plane. Guard that
        // the penalty stays in that class and never explodes.
        let sq_err = |t: &SciRustSlhaTile, v: &[f32; D_C]| -> f32 {
            let dq = t.dequant_latent();
            (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
        };
        for seed in 0..8u64 {
            let mut v = [0.0f32; D_C];
            let mut rng = crate::rng::Rng::new(200 + seed);
            for x in v.iter_mut() {
                *x = rng.next_gaussian();
            }
            let (p1, s1, g1) = quantize_latent_grouped(&v);
            let e_int4 = sq_err(&tile_from(p1, s1, g1), &v);
            let (p2, s2, g2) = quantize_latent_tq3(&v);
            let mut tq3 = tile_from(p2, s2, g2);
            tq3.flags |= FLAG_TQ3;
            let e_tq3 = sq_err(&tq3, &v);
            assert!(
                e_tq3 <= e_int4 * 2.0,
                "seed {seed}: TQ3 err {e_tq3} way above INT4 err {e_int4}"
            );
        }
    }

    #[test]
    fn tq3_nocorr_decodes_the_bare_grid() {
        // With FLAG_TQ3_NOCORR the decoder must ignore the correction plane
        // (even a zeroed one, which would otherwise bias every dim by
        // −TQ3_CORRECTION): the error bound relaxes to half a step, and the
        // corrected tile must reconstruct at least as well on aggregate.
        let v = steep_latent(29);
        let (packed, global, gs) = quantize_latent_tq3(&v);
        let mut with_corr = tile_from(packed, global, gs);
        with_corr.flags |= FLAG_TQ3;
        let mut nocorr = with_corr;
        nocorr.flags |= FLAG_TQ3_NOCORR;
        // CCOS masks the plane when paging it out; emulate that too.
        for b in nocorr.latent_kv[TQ3_CODE_BYTES..].iter_mut() {
            *b = 0;
        }
        let (mut e_corr, mut e_bare) = (0.0f32, 0.0f32);
        for d in 0..D_C {
            let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
            let err = (nocorr.dequant_at(d) - v[d]).abs();
            assert!(
                err <= 0.5 * eff + 1e-5,
                "dim {d}: bare-grid error {err} > half step {}",
                0.5 * eff
            );
            e_corr += (with_corr.dequant_at(d) - v[d]).powi(2);
            e_bare += (nocorr.dequant_at(d) - v[d]).powi(2);
        }
        assert!(
            e_corr < e_bare,
            "correction must help on aggregate: {e_corr} !< {e_bare}"
        );
    }

    /// TQ3 equivalence fixture: a query, its sign bits, and TQ3 tiles built
    /// via `quantize_latent_tq3` from both data shapes (Gaussian scenario
    /// tokens and the steep GPT-2-like spectrum), sweeping all four
    /// HOT/WARM × corr/nocorr flag combinations. Shared by the dispatch and
    /// per-ISA score-equivalence tests.
    fn tq3_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
        use crate::scenario::{generate, Projection};
        let proj = Projection::new(9);
        let (q, toks) = generate(123, 32, 0.4);
        let q_sign = proj.sign_bits(&q);
        let mut tiles = Vec::new();
        let mut push = |v: &[f32; D_C], bitmap: [u64; RESIDUAL_WORDS], i: usize| {
            let (packed, global, gs) = quantize_latent_tq3(v);
            let mut tile = tile_from(packed, global, gs);
            tile.residual_bitmap = bitmap;
            tile.dynamic_lambda = 0.37;
            tile.position = i as u32;
            tile.flags = FLAG_TQ3
                | if i.is_multiple_of(2) { FLAG_WARM } else { 0 }
                | if (i / 2).is_multiple_of(2) {
                    FLAG_TQ3_NOCORR
                } else {
                    0
                };
            tiles.push(tile);
        };
        for (i, t) in toks.iter().enumerate() {
            push(&t.k_coarse, proj.sign_bits(&t.e), i);
        }
        for seed in 0..8u64 {
            push(&steep_latent(300 + seed), [!0, 0, !0, 0], seed as usize);
        }
        (q, q_sign, tiles)
    }

    #[test]
    fn mix3_dequant_roundtrips_and_drops_tail() {
        let v = steep_latent(41);
        let (packed, global, gs) = quantize_latent_mix3(&v);
        let mut tile = tile_from(packed, global, gs);
        tile.flags |= FLAG_MIX3;
        let dq = tile.dequant_latent();
        // 8-bit head: error within one 8-bit step (identical to mixed).
        let eff_hi = global * (gs[0] as f32 / 255.0);
        for d in 0..MIXED_HI_DIMS {
            assert!(
                (dq[d] - v[d]).abs() <= eff_hi + 1e-6,
                "hi dim {d}: |{} - {}| > step {eff_hi}",
                dq[d],
                v[d]
            );
        }
        // TQ3 body: within one grid step of its group (same convention as
        // the grouped/TQ3 tests; the corrected bound is quarter-step when
        // gs rounding is exact).
        for d in MIXED_HI_DIMS..MIXED_DIMS {
            let g = 1 + (d - MIXED_HI_DIMS) / GROUP_DIM;
            let eff = global * (gs[g] as f32 / 255.0);
            assert!(
                (dq[d] - v[d]).abs() <= eff + 1e-6,
                "lo dim {d}: |{} - {}| > step {eff}",
                dq[d],
                v[d]
            );
        }
        // Dropped tail decodes to exactly 0.
        for d in MIXED_DIMS..D_C {
            assert_eq!(dq[d], 0.0, "tail dim {d} must decode to 0");
        }
    }

    #[test]
    fn mix3_stays_in_the_mixed_error_class_and_beats_uniform() {
        // The whole point of the synthesis: on the steep spectrum the codec
        // was built for, MIX3 must (a) clearly beat uniform INT4 (the 8-bit
        // head does the work, like mixed), and (b) stay in the mixed codec's
        // error class — its body spends the same 4 bits/dim (3-bit grid +
        // 1-bit correction), paying only the zero-free-grid penalty.
        let sq_err = |t: &SciRustSlhaTile, v: &[f32; D_C]| -> f32 {
            let dq = t.dequant_latent();
            (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
        };
        for seed in 0..8u64 {
            let v = steep_latent(300 + seed);
            let (p0, s0, g0) = quantize_latent_grouped(&v);
            let e_uniform = sq_err(&tile_from(p0, s0, g0), &v);
            let (p1, s1, g1) = quantize_latent_mixed(&v);
            let mut mixed = tile_from(p1, s1, g1);
            mixed.flags |= FLAG_MIXED;
            let e_mixed = sq_err(&mixed, &v);
            let (p2, s2, g2) = quantize_latent_mix3(&v);
            let mut mix3 = tile_from(p2, s2, g2);
            mix3.flags |= FLAG_MIX3;
            let e_mix3 = sq_err(&mix3, &v);
            assert!(
                e_mix3 < e_uniform,
                "seed {seed}: MIX3 err {e_mix3} not < uniform {e_uniform}"
            );
            assert!(
                e_mix3 <= e_mixed * 2.0,
                "seed {seed}: MIX3 err {e_mix3} way above mixed {e_mixed}"
            );
        }
    }

    #[test]
    fn mix3_nocorr_decodes_the_bare_grid() {
        // With FLAG_TQ3_NOCORR the body decodes without correction (head
        // untouched): body error relaxes to half a step, and the corrected
        // tile must reconstruct at least as well on aggregate.
        let v = steep_latent(43);
        let (packed, global, gs) = quantize_latent_mix3(&v);
        let mut with_corr = tile_from(packed, global, gs);
        with_corr.flags |= FLAG_MIX3;
        let mut nocorr = with_corr;
        nocorr.flags |= FLAG_TQ3_NOCORR;
        for b in nocorr.latent_kv[MIX3_CORR_OFF..].iter_mut() {
            *b = 0;
        }
        let (mut e_corr, mut e_bare) = (0.0f32, 0.0f32);
        for d in MIXED_HI_DIMS..MIXED_DIMS {
            let g = 1 + (d - MIXED_HI_DIMS) / GROUP_DIM;
            let eff = global * (gs[g] as f32 / 255.0);
            let err = (nocorr.dequant_at(d) - v[d]).abs();
            assert!(
                err <= 0.5 * eff + 1e-5,
                "dim {d}: bare-grid error {err} > half step {}",
                0.5 * eff
            );
            e_corr += (with_corr.dequant_at(d) - v[d]).powi(2);
            e_bare += (nocorr.dequant_at(d) - v[d]).powi(2);
        }
        // Head decode identical with and without the flag.
        for d in 0..MIXED_HI_DIMS {
            assert_eq!(
                with_corr.dequant_at(d).to_bits(),
                nocorr.dequant_at(d).to_bits()
            );
        }
        assert!(
            e_corr < e_bare,
            "correction must help on aggregate: {e_corr} !< {e_bare}"
        );
    }

    /// MIX3 equivalence fixture: a Gaussian query/sign pair and MIX3 tiles
    /// built via `quantize_latent_mix3` from the steep GPT-2-like spectrum
    /// the codec was designed for, sweeping all four HOT/WARM × corr/nocorr
    /// flag combinations. Shared by the dispatch and per-ISA
    /// score-equivalence tests.
    fn mix3_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
        let mut rng = crate::rng::Rng::new(53);
        let mut q = [0.0f32; D_C];
        rng.fill_gaussian(&mut q);
        let q_sign = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let mut tiles = Vec::new();
        for seed in 0..8u64 {
            let (packed, global, gs) = quantize_latent_mix3(&steep_latent(600 + seed));
            for warm in [false, true] {
                for nocorr in [false, true] {
                    let mut tile = tile_from(packed, global, gs);
                    tile.dynamic_lambda = 0.37;
                    tile.residual_bitmap = [!0, rng.next_u64(), !0, rng.next_u64()];
                    tile.flags = FLAG_MIX3
                        | if warm { FLAG_WARM } else { 0 }
                        | if nocorr { FLAG_TQ3_NOCORR } else { 0 };
                    tiles.push(tile);
                }
            }
        }
        (q, q_sign, tiles)
    }

    #[test]
    fn mix3_score_paths_agree() {
        // Replaces `mix3_tiles_route_to_the_scalar_path`: MIX3 tiles now take
        // a SIMD path where the CPU offers one (AVX-512/AVX2 on x86-64, NEON
        // on aarch64) and fall back to scalar elsewhere, so whatever path the
        // dispatcher selects is checked for *equivalence* with the scalar
        // reference (up to float reassociation), not bit-exactness — HOT and
        // WARM, with and without the correction plane.
        let (q, q_sign, tiles) = mix3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = tile.compute_score(&q, &q_sign);
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs dispatch {a}",
                tile.flags
            );
        }
    }

    #[test]
    fn avx2_mix3_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("avx2 unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = mix3_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx2 checked just above.
                let a = unsafe { tile.compute_score_avx2_mix3(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#08b}): scalar {s} vs avx2 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    fn avx512_mix3_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = mix3_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx512f checked just above.
                let a = unsafe { tile.compute_score_avx512_mix3(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#08b}): scalar {s} vs avx512 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_mix3_path_matches_scalar() {
        let (q, q_sign, tiles) = mix3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: NEON is always available on aarch64.
            let a = unsafe { tile.compute_score_neon_mix3(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs neon {a}",
                tile.flags
            );
        }
    }

    #[test]
    fn tq3_score_paths_agree() {
        // Replaces `tq3_tiles_route_to_the_scalar_path`: TQ3 tiles now take a
        // SIMD path where the CPU offers one (AVX-512/AVX2 on x86-64, NEON on
        // aarch64) and fall back to scalar elsewhere, so whatever path the
        // dispatcher selects is checked for *equivalence* with the scalar
        // reference (up to float reassociation), not bit-exactness — HOT and
        // WARM, with and without the correction plane.
        let (q, q_sign, tiles) = tq3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = tile.compute_score(&q, &q_sign);
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#07b}): scalar {s} vs dispatch {a}",
                tile.flags
            );
        }
    }

    #[test]
    fn avx2_tq3_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("avx2 unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = tq3_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx2 checked just above.
                let a = unsafe { tile.compute_score_avx2_tq3(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#07b}): scalar {s} vs avx2 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    fn avx512_tq3_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = tq3_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx512f checked just above.
                let a = unsafe { tile.compute_score_avx512_tq3(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#07b}): scalar {s} vs avx512 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_tq3_path_matches_scalar() {
        let (q, q_sign, tiles) = tq3_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: NEON is always available on aarch64.
            let a = unsafe { tile.compute_score_neon_tq3(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#07b}): scalar {s} vs neon {a}",
                tile.flags
            );
        }
    }

    /// NF4 equivalence fixture: a query, its sign bits, and NF4 tiles built
    /// via `quantize_latent_nf4` from both data shapes (Gaussian scenario
    /// tokens and the steep GPT-2-like spectrum), sweeping HOT/WARM. Shared
    /// by the dispatch and per-ISA score-equivalence tests.
    fn nf4_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
        use crate::scenario::{generate, Projection};
        let proj = Projection::new(9);
        let (q, toks) = generate(123, 32, 0.4);
        let q_sign = proj.sign_bits(&q);
        let mut tiles = Vec::new();
        let mut push = |v: &[f32; D_C], bitmap: [u64; RESIDUAL_WORDS], i: usize| {
            let (packed, global, gs) = quantize_latent_nf4(v);
            let mut tile = tile_from(packed, global, gs);
            tile.residual_bitmap = bitmap;
            tile.dynamic_lambda = 0.37;
            tile.position = i as u32;
            tile.flags = FLAG_NF4 | if i.is_multiple_of(2) { FLAG_WARM } else { 0 };
            tiles.push(tile);
        };
        for (i, t) in toks.iter().enumerate() {
            push(&t.k_coarse, proj.sign_bits(&t.e), i);
        }
        for seed in 0..8u64 {
            push(&steep_latent(400 + seed), [!0, 0, !0, 0], seed as usize);
        }
        (q, q_sign, tiles)
    }

    #[test]
    fn nf4_score_paths_agree() {
        // NF4 tiles take a SIMD path where the CPU offers one (AVX-512/AVX2
        // on x86-64, NEON on aarch64) and fall back to scalar elsewhere, so
        // whatever path the dispatcher selects is checked for *equivalence*
        // with the scalar reference (up to float reassociation), not
        // bit-exactness — HOT and WARM.
        let (q, q_sign, tiles) = nf4_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = tile.compute_score(&q, &q_sign);
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs dispatch {a}",
                tile.flags
            );
        }
    }

    #[test]
    fn avx2_nf4_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("avx2 unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = nf4_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx2 checked just above.
                let a = unsafe { tile.compute_score_avx2_nf4(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#08b}): scalar {s} vs avx2 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    fn avx512_nf4_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = nf4_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx512f checked just above.
                let a = unsafe { tile.compute_score_avx512_nf4(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#08b}): scalar {s} vs avx512 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_nf4_path_matches_scalar() {
        let (q, q_sign, tiles) = nf4_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: NEON is always available on aarch64.
            let a = unsafe { tile.compute_score_neon_nf4(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs neon {a}",
                tile.flags
            );
        }
    }

    /// Mixed-precision equivalence fixture: a Gaussian query/sign pair and
    /// mixed tiles built via `quantize_latent_mixed` from the steep
    /// GPT-2-like spectrum the codec was designed for, sweeping HOT/WARM.
    /// Shared by the dispatch and per-ISA score-equivalence tests.
    fn mixed_equivalence_fixture() -> ([f32; D_C], [u64; RESIDUAL_WORDS], Vec<SciRustSlhaTile>) {
        let mut rng = crate::rng::Rng::new(33);
        let mut q = [0.0f32; D_C];
        rng.fill_gaussian(&mut q);
        let q_sign = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let mut tiles = Vec::new();
        for seed in 0..8u64 {
            let (packed, global, gs) = quantize_latent_mixed(&steep_latent(500 + seed));
            for warm in [false, true] {
                let mut tile = tile_from(packed, global, gs);
                tile.dynamic_lambda = 0.37;
                tile.residual_bitmap = [!0, rng.next_u64(), !0, rng.next_u64()];
                tile.flags = FLAG_MIXED | if warm { FLAG_WARM } else { 0 };
                tiles.push(tile);
            }
        }
        (q, q_sign, tiles)
    }

    #[test]
    fn mixed_score_paths_agree() {
        // Replaces `mixed_tiles_route_to_the_scalar_path`: mixed tiles now
        // take a SIMD path where the CPU offers one (AVX-512/AVX2 on x86-64,
        // NEON on aarch64) and fall back to scalar elsewhere, so whatever
        // path the dispatcher selects is checked for *equivalence* with the
        // scalar reference (up to float reassociation), not bit-exactness —
        // HOT and WARM.
        let (q, q_sign, tiles) = mixed_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = tile.compute_score(&q, &q_sign);
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs dispatch {a}",
                tile.flags
            );
        }
    }

    #[test]
    fn avx2_mixed_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("avx2 unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = mixed_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx2 checked just above.
                let a = unsafe { tile.compute_score_avx2_mixed(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#08b}): scalar {s} vs avx2 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    fn avx512_mixed_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable — skipping equivalence check");
                return;
            }
            let (q, q_sign, tiles) = mixed_equivalence_fixture();
            for (i, tile) in tiles.iter().enumerate() {
                let s = tile.compute_score_scalar(&q, &q_sign);
                // SAFETY: avx512f checked just above.
                let a = unsafe { tile.compute_score_avx512_mixed(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i} (flags {:#08b}): scalar {s} vs avx512 {a}",
                    tile.flags
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_mixed_path_matches_scalar() {
        let (q, q_sign, tiles) = mixed_equivalence_fixture();
        for (i, tile) in tiles.iter().enumerate() {
            let s = tile.compute_score_scalar(&q, &q_sign);
            // SAFETY: NEON is always available on aarch64.
            let a = unsafe { tile.compute_score_neon_mixed(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i} (flags {:#08b}): scalar {s} vs neon {a}",
                tile.flags
            );
        }
    }

    #[test]
    fn avx2_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("avx2 unavailable — skipping equivalence check");
                return;
            }
            use crate::scenario::{build_tile, generate, Projection};
            let proj = Projection::new(9);
            let (q, toks) = generate(123, 64, 0.4);
            let q_sign = proj.sign_bits(&q);
            for (i, t) in toks.iter().enumerate() {
                // Alternate HOT / WARM to cover both branches.
                let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
                let s = tile.compute_score_scalar(&q, &q_sign);
                let a = unsafe { tile.compute_score_avx2(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i}: scalar {s} vs avx2 {a}"
                );
            }
        }
    }

    #[test]
    fn avx512_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                eprintln!("avx512f unavailable — skipping equivalence check");
                return;
            }
            use crate::scenario::{build_tile, generate, Projection};
            let proj = Projection::new(9);
            let (q, toks) = generate(123, 64, 0.4);
            let q_sign = proj.sign_bits(&q);
            for (i, t) in toks.iter().enumerate() {
                let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
                let s = tile.compute_score_scalar(&q, &q_sign);
                let a = unsafe { tile.compute_score_avx512(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i}: scalar {s} vs avx512 {a}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_path_matches_scalar() {
        use crate::scenario::{build_tile, generate, Projection};
        let proj = Projection::new(9);
        let (q, toks) = generate(123, 64, 0.4);
        let q_sign = proj.sign_bits(&q);
        for (i, t) in toks.iter().enumerate() {
            // Alternate HOT / WARM to cover both branches.
            let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = unsafe { tile.compute_score_neon(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i}: scalar {s} vs neon {a}"
            );
        }
    }

    /// The public `hamming_distance` dispatcher (whatever SIMD path the running
    /// CPU selects) must equal a brute-force per-bit count, over random inputs.
    #[test]
    fn hamming_distance_matches_bruteforce() {
        let mut rng = crate::rng::Rng::new(0x4D31);
        for _ in 0..4000 {
            let a = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            let b = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            let brute: u32 = (0..D_S)
                .map(|s| (((a[s >> 6] >> (s & 63)) ^ (b[s >> 6] >> (s & 63))) & 1) as u32)
                .sum();
            assert_eq!(hamming_distance(&a, &b), brute);
        }
    }

    /// On CPUs that advertise it, the AVX-512 VPOPCNTDQ path must be bit-exact
    /// with the scalar reduction. Compile-checked on every x86-64 build; only
    /// *executed* where the feature is present (skipped on this bench otherwise).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn vpopcntdq_hamming_matches_scalar() {
        if !(std::is_x86_feature_detected!("avx512vpopcntdq")
            && std::is_x86_feature_detected!("avx512vl"))
        {
            eprintln!("avx512vpopcntdq+vl unavailable — skipping (compile-checked only)");
            return;
        }
        let mut rng = crate::rng::Rng::new(0x5E42);
        for _ in 0..4000 {
            let a = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            let b = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            // SAFETY: features checked just above.
            let simd = unsafe { hamming_vpopcntdq(&a, &b) };
            assert_eq!(simd, hamming_scalar(&a, &b));
        }
    }
}
