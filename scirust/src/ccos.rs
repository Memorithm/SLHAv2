//! CCOS elastic KV-cache manager — drives the §4 "Soft-Paging" policy over a
//! **contiguous arena** of tiles, with three states:
//!
//! - **HOT**  — full tile (latent + residual), 128 B.
//! - **WARM** — residual bitmap masked/freed (`FLAG_WARM`, `λ = 0`); the score
//!   falls back to the coarse term. ~32 B reclaimed (logical footprint 96 B).
//! - **COLD** — evicted from the active set; its slot is recycled on the next
//!   `insert` (no I/O here; a real CCOS would snapshot it to the EventLog).
//!
//! **Separable-plane codecs (TQ3, MIX3) get one extra, finer rung**
//! (orthogonal to the state): their 1-bit correction plane (16 B for TQ3,
//! 14 B for MIX3) can be paged out first ([`FLAG_TQ3_NOCORR`], sticky),
//! degrading the covered dims from quarter-step to half-step error while
//! keeping the residual. TQ3 ladder: HOT 128 → HOT¬corr 112 → WARM 96 →
//! WARM¬corr 80 → COLD 0 (MIX3: 128 → 114 → 96 → 82 → 0). No nibble codec
//! can offer this rung.
//!
//! `enforce_budget()` keeps the **logical** footprint under a byte budget by
//! dropping TQ3 correction planes, then paging HOT→WARM (per
//! [`PageOutPolicy`]) and, if needed, evicting →COLD.
//!
//! Note: tiles physically remain 128 B in the arena `Vec`; `live_bytes()` is the
//! *elastic* accounting (HOT 128 / WARM 96 / COLD 0, minus the codec's
//! separable-plane bytes when paged out) — i.e. what a packed store would
//! occupy. Masking is O(1) (zero a few bytes + flip a flag), no allocation.

use crate::attention::slha_v2::{
    SciRustSlhaTile, D_C, FLAG_TQ3_NOCORR, FLAG_WARM, LATENT_BYTES, RESIDUAL_WORDS,
};

/// Logical footprint of a full (HOT) tile.
pub const HOT_BYTES: usize = 128;
/// Logical footprint of a WARM tile (residual's 32 B reclaimed).
pub const WARM_BYTES: usize = HOT_BYTES - RESIDUAL_WORDS * 8; // 96

/// Soft-Paging state of a slot (spec §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileState {
    Hot,
    Warm,
    Cold,
}

/// Order in which HOT tiles are paged out (HOT→WARM) under memory pressure.
///
/// Note this only governs the **paging** phase. Eviction (WARM/HOT→COLD), which
/// only kicks in once paging the whole working set is not enough, is governed by
/// a separate [`EvictionPolicy`] — dropping a token entirely is a harder loss
/// than freeing its residual, so the two phases use different criteria.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PageOutPolicy {
    /// **Hybrid (recommended, default).** Page out the lowest-`σ_E` tiles first
    /// — their 1-bit residual matters least, so WARM is near-lossless there
    /// (cf. §7.2) — then, if still over budget, evict the oldest by age. Best of
    /// both: free residuals where they hurt least, drop tokens where they matter
    /// least causally.
    #[default]
    LowestImpactFirst,
    /// Pure causal: page out the oldest-inserted tiles first (causal distance,
    /// §4). Eviction order is unchanged (also oldest-first).
    OldestFirst,
}

/// Order in which live tiles are evicted (→COLD) once paging the whole working
/// set to WARM is not enough (plan axis **A5** — informed eviction).
///
/// `σ_E` already governs the *paging* phase via [`PageOutPolicy`] (free the
/// residual where it hurts least); this policy governs the harsher *eviction*
/// phase (drop the token entirely). Pure-causal eviction ignores how much
/// attention a token actually receives and destroys heavy-hitters / attention
/// sinks. The informed alternative preserves them.
///
/// See: H2O (arXiv 2306.14048), StreamingLLM (2309.17453), SnapKV (2404.14469),
/// PyramidKV (2406.02069), FastGen (2310.01801).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// **Causal (default, back-compatible).** Evict the oldest-inserted live
    /// tiles first (causal distance, §4). This is the original SLHA v2 policy.
    #[default]
    Causal,
    /// **Informed eviction (plan axis A5).** Evict the lowest-importance live
    /// tiles first, but **never the attention sinks** — the first
    /// `sink_window` tokens (by `position`, cf. StreamingLLM) are pinned and
    /// dropped only when nothing else remains. Importance is the **cumulative
    /// attention mass** each token has received (H2O), recorded via
    /// [`ElasticKvCache::observe_scores`] across decoding steps. `σ_E` stays a
    /// complementary signal through the paging phase, untouched here.
    Importance { sink_window: usize },
}

/// Elastic KV-cache over a contiguous arena of [`SciRustSlhaTile`].
pub struct ElasticKvCache {
    tiles: Vec<SciRustSlhaTile>,
    state: Vec<TileState>,
    seq: Vec<u64>, // insertion order (paging tie-break + eviction), survives reuse
    /// Cumulative attention mass per slot (H2O importance, plan axis A5).
    /// Reset on (re)insert; only read by [`EvictionPolicy::Importance`].
    importance: Vec<f32>,
    free: Vec<usize>,
    budget_bytes: usize,
    policy: PageOutPolicy,
    eviction: EvictionPolicy,
    next_seq: u64,
    /// Optional persistence sink (plan: COLD → EventLog). `None` ⇒ the cache
    /// behaves exactly as before this field existed — eviction is in-memory
    /// recycling only. Attach one with [`Self::attach_event_log`].
    event_log: Option<crate::eventlog::EventLog>,
    /// Count of failed EventLog appends (see [`Self::evict`] error policy).
    log_errors: u64,
}

impl ElasticKvCache {
    pub fn new(budget_bytes: usize, policy: PageOutPolicy) -> Self {
        Self {
            tiles: Vec::new(),
            state: Vec::new(),
            seq: Vec::new(),
            importance: Vec::new(),
            free: Vec::new(),
            budget_bytes,
            policy,
            eviction: EvictionPolicy::default(),
            next_seq: 0,
            event_log: None,
            log_errors: 0,
        }
    }

    /// Attach an [`EventLog`](crate::eventlog::EventLog) so that evicted
    /// (COLD) tiles are snapshotted to durable storage before their slot is
    /// recycled, enabling later [`Self::rehydrate`]. Without one, eviction is
    /// in-memory only (the historical behaviour). Attaching does not touch
    /// already-live tiles.
    pub fn attach_event_log(&mut self, log: crate::eventlog::EventLog) {
        self.event_log = Some(log);
    }

    /// Number of EventLog appends that failed during eviction. Eviction never
    /// fails the in-memory cache; append errors are counted here instead (see
    /// [`Self::evict`]).
    #[must_use]
    pub fn log_errors(&self) -> u64 {
        self.log_errors
    }

    /// Convenience constructor with the recommended default policy (the hybrid
    /// [`PageOutPolicy::LowestImpactFirst`]: page by `σ_E`, evict by age).
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self::new(budget_bytes, PageOutPolicy::default())
    }

    /// As [`Self::with_budget`] but with an explicit eviction policy (plan axis
    /// A5). Use [`EvictionPolicy::Importance`] to preserve heavy-hitters and
    /// attention sinks under pressure instead of dropping oldest-first.
    pub fn with_eviction(budget_bytes: usize, eviction: EvictionPolicy) -> Self {
        let mut c = Self::with_budget(budget_bytes);
        c.eviction = eviction;
        c
    }

    /// **First-touch NUMA hint (Linux + `numa` feature).** Pin the calling thread
    /// to its current CPU's local NUMA node *before* bulk-inserting / warming the
    /// arena, so the first-touch policy places the arena's pages on the local node
    /// (avoids inter-socket traffic on multi-socket hosts). Best-effort: returns
    /// the pinned CPU on success, or `None` if the `numa` feature is off / the
    /// target is non-Linux / pinning failed — in which case the cache still works
    /// correctly, just without the locality guarantee.
    ///
    /// Call this once, from the inference thread, right before the warm-up loop
    /// (or before the first `insert` storm). It is a no-op allocation-wise and
    /// safe to call multiple times. On a single-NUMA-node host (e.g. Jetson Thor)
    /// it still pins, which avoids spurious thread migration.
    ///
    /// Note: the arena is a plain `Vec` (allocator-global, not page-aligned), so we
    /// rely on first-touch rather than `mbind` (which needs page-aligned regions —
    /// see [`crate::numa::NumaBuffer`] for the page-aligned path).
    pub fn pin_caller_to_local_numa() -> Option<usize> {
        crate::numa::pin_current_thread_local().ok()
    }

    /// Insert a tile, reusing a recycled (COLD) slot when available.
    ///
    /// The logical state is derived from the tile's encoded [`FLAG_WARM`]:
    /// an already-degraded WARM tile must not be advertised as HOT.
    /// The slot's H2O importance is (re)set to 0.
    pub fn insert(&mut self, tile: SciRustSlhaTile) -> usize {
        let state = if tile.is_warm() {
            TileState::Warm
        } else {
            TileState::Hot
        };

        self.insert_with_state(tile, state)
    }

    fn insert_with_state(&mut self, tile: SciRustSlhaTile, state: TileState) -> usize {
        debug_assert_ne!(state, TileState::Cold);

        let slot = if let Some(s) = self.free.pop() {
            self.tiles[s] = tile;
            self.state[s] = state;
            self.seq[s] = self.next_seq;
            self.importance[s] = 0.0;
            s
        } else {
            self.tiles.push(tile);
            self.state.push(state);
            self.seq.push(self.next_seq);
            self.importance.push(0.0);
            self.tiles.len() - 1
        };

        self.next_seq += 1;
        slot
    }

    /// Cumulative attention mass (H2O importance, plan axis A5) accumulated on
    /// `slot` via [`Self::observe_scores`]. Zero for a freshly inserted slot.
    pub fn importance(&self, slot: usize) -> f32 {
        self.importance[slot]
    }

    /// H2O-style importance accumulation (plan axis A5): add the softmax
    /// attention mass of `scores` (raw logits, divided by `temperature`) to each
    /// referenced live slot's cumulative importance. Tokens that consistently
    /// attract attention become the heavy-hitters the
    /// [`EvictionPolicy::Importance`] policy preserves.
    ///
    /// `scores` is typically the output of [`Self::score_all`]; cold slots it
    /// references are skipped (they carry no attention once evicted).
    pub fn observe_scores(&mut self, scores: &[(usize, f32)], temperature: f32) {
        if scores.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
            return;
        }

        // Les slots absents ou COLD ne doivent pas absorber une partie de la
        // masse softmax destinée aux éléments réellement actifs.
        let live: Vec<(usize, f32)> = scores
            .iter()
            .copied()
            .filter(|&(slot, _)| slot < self.state.len() && self.state[slot] != TileState::Cold)
            .collect();

        if live.is_empty() || live.iter().any(|&(_, score)| !score.is_finite()) {
            return;
        }

        let m = live
            .iter()
            .map(|&(_, score)| score)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        let mut weights = vec![0.0f32; live.len()];

        for (i, &(_, score)) in live.iter().enumerate() {
            weights[i] = ((score - m) / temperature).exp();
            sum += weights[i];
        }

        if !sum.is_finite() || sum <= 0.0 {
            return;
        }

        let inv = 1.0 / sum;

        for (i, &(slot, _)) in live.iter().enumerate() {
            self.importance[slot] += weights[i] * inv;
        }
    }

    /// Fused attention score for an existing live slot.
    ///
    /// Returns `None` for an absent or COLD slot. WARM slots return the coarse
    /// term only because the kernel honours [`FLAG_WARM`].
    pub fn try_score(
        &self,
        slot: usize,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> Option<f32> {
        let tile = self.tiles.get(slot)?;

        match self.state.get(slot)? {
            TileState::Cold => None,
            TileState::Hot | TileState::Warm => Some(tile.compute_score(q_coarse, q_sign)),
        }
    }

    /// Fused attention score for a live slot.
    ///
    /// # Panics
    ///
    /// Panics with an explicit diagnostic for an absent or COLD slot. New code
    /// handling untrusted slot identifiers should prefer [`Self::try_score`].
    pub fn score(&self, slot: usize, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        self.try_score(slot, q_coarse, q_sign)
            .expect("cannot score an absent or COLD CCOS slot")
    }

    /// Score the query against every live (non-COLD) tile: `(slot, score)`.
    pub fn score_all(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> Vec<(usize, f32)> {
        (0..self.tiles.len())
            .filter(|&s| self.state[s] != TileState::Cold)
            .map(|s| (s, self.tiles[s].compute_score(q_coarse, q_sign)))
            .collect()
    }

    pub fn state(&self, slot: usize) -> TileState {
        self.state[slot]
    }

    /// Read-only view of a slot's tile (whatever its state — a COLD slot
    /// keeps its last contents until the slot is recycled by `insert`).
    pub fn tile(&self, slot: usize) -> &SciRustSlhaTile {
        &self.tiles[slot]
    }

    /// HOT → WARM: mask/free the 32-byte residual bitmap (zero it, drop λ, set
    /// the flag). No I/O, no allocation.
    pub fn page_out(&mut self, slot: usize) {
        if self.state[slot] == TileState::Hot {
            self.tiles[slot].residual_bitmap = [0u64; RESIDUAL_WORDS];
            self.tiles[slot].dynamic_lambda = 0.0;
            self.tiles[slot].flags |= FLAG_WARM;
            self.state[slot] = TileState::Warm;
        }
    }

    /// Separable-plane finer rung (TQ3 and MIX3): mask/free the codec's
    /// 1-bit correction plane (zero it, set [`FLAG_TQ3_NOCORR`]) — the
    /// decoder falls back to the bare 3-bit grid on the covered dims. Sticky
    /// (the ladder only degrades), orthogonal to HOT/WARM, no-op on codecs
    /// without a separable plane or on COLD slots. Returns `true` if bytes
    /// were reclaimed.
    pub fn drop_correction(&mut self, slot: usize) -> bool {
        let t = &mut self.tiles[slot];
        let n = t.separable_corr_bytes();
        if self.state[slot] == TileState::Cold || n == 0 || t.is_tq3_nocorr() {
            return false;
        }
        // Both layouts put the plane at the end of the 64-byte latent.
        t.latent_kv[LATENT_BYTES - n..].fill(0);
        t.flags |= FLAG_TQ3_NOCORR;
        true
    }

    /// Evict a slot (→ COLD) and recycle it for a future `insert`.
    pub fn evict(&mut self, slot: usize) {
        if self.state[slot] != TileState::Cold {
            // Snapshot to the EventLog (if attached) BEFORE the slot is marked
            // COLD and made recyclable. Error policy: a failed append must NOT
            // corrupt the in-memory cache — the tile is still evicted (memory
            // semantics preserved) and the failure is counted in `log_errors`.
            if let Some(log) = self.event_log.as_mut() {
                if log
                    .append(self.seq[slot], slot as u32, &self.tiles[slot])
                    .is_err()
                {
                    self.log_errors += 1;
                }
            }
            self.state[slot] = TileState::Cold;
            self.free.push(slot);
        }
    }

    /// Restore a previously-evicted tile from the attached
    /// [`EventLog`](crate::eventlog::EventLog) by its eviction `seq`.
    ///
    /// The restored logical state matches the encoded tile: a record carrying
    /// [`FLAG_WARM`] is restored as WARM, otherwise as HOT. Importance is reset
    /// and the rehydrated tile gets a **new** insertion `seq`; this is a fresh
    /// admission, not a rewind of history. Returns `None` if no log is attached
    /// or no record matches `seq`.
    ///
    /// # Errors
    /// Propagates I/O errors from reading the log.
    pub fn rehydrate(&mut self, seq: u64) -> std::io::Result<Option<usize>> {
        let Some(log) = self.event_log.as_mut() else {
            return Ok(None);
        };
        let Some(rec) = log.fetch_by_seq(seq)? else {
            return Ok(None);
        };

        let state = if rec.tile.is_warm() {
            TileState::Warm
        } else {
            TileState::Hot
        };

        Ok(Some(self.insert_with_state(rec.tile, state)))
    }

    /// Elastic logical footprint: Σ over live slots (HOT 128, WARM 96, COLD 0;
    /// minus the codec's separable-plane bytes where it is paged out).
    pub fn live_bytes(&self) -> usize {
        self.state
            .iter()
            .zip(&self.tiles)
            .map(|(s, t)| {
                let base = match s {
                    TileState::Hot => HOT_BYTES,
                    TileState::Warm => WARM_BYTES,
                    TileState::Cold => return 0,
                };
                base - if t.is_tq3_nocorr() {
                    t.separable_corr_bytes()
                } else {
                    0
                }
            })
            .sum()
    }

    /// Number of live slots whose separable correction plane is paged out.
    pub fn nocorr_count(&self) -> usize {
        self.state
            .iter()
            .zip(&self.tiles)
            .filter(|(s, t)| **s != TileState::Cold && t.is_tq3_nocorr())
            .count()
    }

    /// Total bytes reclaimed by paged-out correction planes on live slots.
    pub fn reclaimed_correction_bytes(&self) -> usize {
        self.state
            .iter()
            .zip(&self.tiles)
            .filter(|(s, t)| **s != TileState::Cold && t.is_tq3_nocorr())
            .map(|(_, t)| t.separable_corr_bytes())
            .sum()
    }

    /// `(hot, warm, cold)` slot counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut c = (0usize, 0usize, 0usize);
        for s in &self.state {
            match s {
                TileState::Hot => c.0 += 1,
                TileState::Warm => c.1 += 1,
                TileState::Cold => c.2 += 1,
            }
        }
        c
    }

    /// Slots in `state`, ordered by the paging policy (lowest `σ_E` first for
    /// the hybrid default, else oldest first).
    fn slots_in_paging_order(&self, state: TileState) -> Vec<usize> {
        let mut slots: Vec<usize> = (0..self.tiles.len())
            .filter(|&s| self.state[s] == state)
            .collect();
        match self.policy {
            PageOutPolicy::LowestImpactFirst => slots.sort_by(|&a, &b| {
                self.tiles[a]
                    .residual_sigma
                    .total_cmp(&self.tiles[b].residual_sigma)
            }),
            PageOutPolicy::OldestFirst => slots.sort_by_key(|&s| self.seq[s]),
        }
        slots
    }

    /// Bring the logical footprint under `budget_bytes` in strictly
    /// increasing harshness:
    ///
    /// 1. **Drop HOT TQ3 correction planes** (the finest, cheapest rung:
    ///    −16 B per tile, latent error 0.25 → 0.5 step, residual kept) in
    ///    [`PageOutPolicy`] order. No-op on non-TQ3 tiles.
    /// 2. **Page** HOT→WARM in [`PageOutPolicy`] order (default hybrid: lowest
    ///    `σ_E` first — free the residual where it hurts least).
    /// 3. **Drop the remaining WARM TQ3 correction planes** (last soft rung:
    ///    those tiles keep only the bare 3-bit latent).
    /// 4. If still over budget, **evict** live tiles →COLD per
    ///    [`EvictionPolicy`]: default `Causal` (oldest first), or `Importance`
    ///    (plan axis A5: lowest cumulative attention first, attention sinks
    ///    pinned) — dropping a token is the harder loss.
    pub fn enforce_budget(&mut self) {
        if self.live_bytes() <= self.budget_bytes {
            return;
        }

        // Phase 1 — HOT TQ3 correction planes.
        for s in self.slots_in_paging_order(TileState::Hot) {
            if self.live_bytes() <= self.budget_bytes {
                return;
            }
            self.drop_correction(s);
        }

        // Phase 2 — HOT→WARM.
        for s in self.slots_in_paging_order(TileState::Hot) {
            if self.live_bytes() <= self.budget_bytes {
                return;
            }
            self.page_out(s);
        }

        // Phase 3 — WARM TQ3 correction planes (tiles paged in phase 2 have
        // already lost theirs in phase 1; this catches inserted-WARM tiles).
        for s in self.slots_in_paging_order(TileState::Warm) {
            if self.live_bytes() <= self.budget_bytes {
                return;
            }
            self.drop_correction(s);
        }

        // Still over budget: evict live tiles per the eviction policy.
        let mut live: Vec<usize> = (0..self.tiles.len())
            .filter(|&s| self.state[s] != TileState::Cold)
            .collect();
        match self.eviction {
            EvictionPolicy::Causal => live.sort_by_key(|&s| self.seq[s]),
            EvictionPolicy::Importance { sink_window } => {
                let sw = sink_window as u32;
                // Sinks (position < sink_window) sort LAST (evicted only when
                // nothing else remains); among the rest, lowest H2O importance
                // first; ties broken oldest-first (causal stability).
                live.sort_by(|&a, &b| {
                    let sa = self.tiles[a].position < sw;
                    let sb = self.tiles[b].position < sw;
                    sa.cmp(&sb)
                        .then_with(|| self.importance[a].total_cmp(&self.importance[b]))
                        .then_with(|| self.seq[a].cmp(&self.seq[b]))
                });
            }
        }
        for s in live {
            if self.live_bytes() <= self.budget_bytes {
                return;
            }
            self.evict(s);
        }
    }
}
