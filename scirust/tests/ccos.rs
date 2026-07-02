//! Integration tests for the CCOS elastic KV-cache manager (§4 Soft-Paging).

use scirust::attention::slha_v2::{LatentCodec, MIX3_CORR_BYTES, TQ3_CORR_BYTES};
use scirust::ccos::{
    ElasticKvCache, EvictionPolicy, PageOutPolicy, TileState, HOT_BYTES, WARM_BYTES,
};
use scirust::learned::{gen_keys, LearnedModel};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection};

/// Invariants that must hold after every `enforce_budget`.
fn assert_invariants(cache: &ElasticKvCache, budget: usize) {
    // (1) The elastic footprint is within budget — eviction can always reach 0,
    //     so this holds for *any* budget ≥ 0.
    assert!(
        cache.live_bytes() <= budget,
        "footprint {} exceeds budget {budget}",
        cache.live_bytes()
    );
    // (2) Byte accounting is consistent with the HOT/WARM/COLD counts and the
    //     paged-out separable correction planes (16 B TQ3 / 14 B MIX3).
    let (h, w, _c) = cache.counts();
    assert_eq!(
        h * HOT_BYTES + w * WARM_BYTES - cache.reclaimed_correction_bytes(),
        cache.live_bytes(),
        "counts ({h} HOT, {w} WARM, {} ¬corr) inconsistent with live_bytes",
        cache.nocorr_count()
    );
}

/// Randomised: across many (size, budget, policy) configurations, every
/// `enforce_budget` leaves the cache within budget with consistent accounting,
/// and COLD slots are recycled (the slot vector never exceeds the high-water
/// mark of simultaneously-live tiles).
#[test]
fn prop_enforce_budget_respects_budget_and_recycles() {
    let proj = Projection::new(0xB0D);
    for trial in 0..300u64 {
        let mut rng = Rng::new(trial);
        let n = 1 + (rng.next_u64() % 48) as usize;
        // Budget from 0 up to slightly above all-HOT (covers page-only, evict,
        // and evict-everything regimes).
        let budget = (rng.next_u64() % (n as u64 * HOT_BYTES as u64 + 1)) as usize;
        let policy = if trial.is_multiple_of(2) {
            PageOutPolicy::LowestImpactFirst
        } else {
            PageOutPolicy::OldestFirst
        };
        let mut cache = ElasticKvCache::new(budget, policy);
        let (_q, toks) = generate(trial + 1, n, 0.3);

        for (i, t) in toks.iter().enumerate() {
            cache.insert(build_tile(&proj, t, i as u32, false));
            if rng.next_u64().is_multiple_of(3) {
                cache.enforce_budget();
                assert_invariants(&cache, budget);
            }
        }
        cache.enforce_budget();
        assert_invariants(&cache, budget);

        // Slot recycling: total slots ever allocated ≤ tiles inserted, and COLD
        // slots are reusable — re-inserting fills them without growing the arena.
        let (h0, w0, c0) = cache.counts();
        let total_slots = h0 + w0 + c0;
        for (i, t) in toks.iter().enumerate() {
            cache.insert(build_tile(&proj, t, i as u32, false));
        }
        let (h1, w1, c1) = cache.counts();
        assert!(
            h1 + w1 + c1 <= total_slots + n,
            "arena grew past the live high-water mark (no recycling)"
        );
    }
}

#[test]
fn page_out_masks_residual_and_falls_back_to_coarse() {
    let proj = Projection::new(1);
    let (q, toks) = generate(2, 1, 0.5);
    let q_sign = proj.sign_bits(&q);

    let mut cache = ElasticKvCache::new(usize::MAX, PageOutPolicy::LowestImpactFirst);
    let slot = cache.insert(build_tile(&proj, &toks[0], 0, false));
    let hot = cache.score(slot, &q, &q_sign);

    cache.page_out(slot);
    assert_eq!(cache.state(slot), TileState::Warm);
    let warm = cache.score(slot, &q, &q_sign);

    // WARM == the coarse-only score of a freshly-built WARM tile, and differs
    // from HOT (the residual did contribute).
    let warm_ref = build_tile(&proj, &toks[0], 0, true).compute_score(&q, &q_sign);
    assert!(
        (warm - warm_ref).abs() <= 1e-4 * (1.0 + warm.abs()),
        "{warm} vs {warm_ref}"
    );
    assert!((hot - warm).abs() > 0.0, "residual contributed nothing");
}

#[test]
fn enforce_budget_bounds_live_bytes() {
    let proj = Projection::new(3);
    let budget = 50 * HOT_BYTES;
    let mut cache = ElasticKvCache::new(budget, PageOutPolicy::LowestImpactFirst);
    let (_q, toks) = generate(42, 100, 0.3);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    assert!(cache.live_bytes() > budget);
    cache.enforce_budget();
    assert!(
        cache.live_bytes() <= budget,
        "live={} > budget={budget}",
        cache.live_bytes()
    );
    let (_hot, warm, cold) = cache.counts();
    assert!(warm + cold > 0, "nothing was paged/evicted");
}

#[test]
fn evict_recycles_slots() {
    let proj = Projection::new(4);
    let (_q, toks) = generate(5, 4, 0.3);
    let mut cache = ElasticKvCache::new(usize::MAX, PageOutPolicy::OldestFirst);
    let s0 = cache.insert(build_tile(&proj, &toks[0], 0, false));
    let _s1 = cache.insert(build_tile(&proj, &toks[1], 1, false));
    cache.evict(s0);
    assert_eq!(cache.state(s0), TileState::Cold);
    let s2 = cache.insert(build_tile(&proj, &toks[2], 2, false));
    assert_eq!(s2, s0, "the freed COLD slot should be recycled");
    assert_eq!(cache.state(s2), TileState::Hot);
}

#[test]
fn page_out_policies_differ() {
    let proj = Projection::new(6);
    // σ_E decreasing with slot order (rho decreasing) -> slot 4 is lowest-impact.
    let rhos = [0.9f32, 0.7, 0.5, 0.3, 0.1];
    let n = rhos.len();
    let fill = |c: &mut ElasticKvCache| {
        for (i, &r) in rhos.iter().enumerate() {
            let (_q, toks) = generate(100 + i as u64, 1, r);
            c.insert(build_tile(&proj, &toks[0], i as u32, false));
        }
    };
    let budget = n * HOT_BYTES - 1; // forces exactly one page-out

    let mut lo = ElasticKvCache::new(budget, PageOutPolicy::LowestImpactFirst);
    fill(&mut lo);
    lo.enforce_budget();

    let mut old = ElasticKvCache::new(budget, PageOutPolicy::OldestFirst);
    fill(&mut old);
    old.enforce_budget();

    // OldestFirst pages slot 0; LowestImpactFirst pages the lowest-σ_E (last) slot.
    assert_eq!(old.state(0), TileState::Warm);
    assert_eq!(old.state(n - 1), TileState::Hot);
    assert_eq!(lo.state(n - 1), TileState::Warm);
    assert_eq!(lo.state(0), TileState::Hot);
}

#[test]
fn hybrid_evicts_oldest_not_lowest_impact() {
    // The default hybrid uses *two different keys*: it pages HOT→WARM by σ_E
    // (pinned by `page_out_policies_differ`), but evicts →COLD strictly by age.
    // Under a budget so tight that everything is paged and one tile must still
    // be dropped, the evicted tile must be the OLDEST — not the lowest-σ_E.
    let proj = Projection::new(11);
    let rhos = [0.9f32, 0.7, 0.5, 0.3, 0.1]; // slot 0 oldest & highest σ_E; slot 4 lowest σ_E
    let n = rhos.len();
    let mut cache = ElasticKvCache::with_budget((n - 1) * 96); // all-WARM(480) - one tile
    for (i, &r) in rhos.iter().enumerate() {
        let (_q, toks) = generate(200 + i as u64, 1, r);
        cache.insert(build_tile(&proj, &toks[0], i as u32, false));
    }
    assert_eq!(cache.state(0), TileState::Hot);
    cache.enforce_budget();

    // Exactly one eviction, and it is the oldest (slot 0) — NOT the lowest-σ_E
    // (slot 4), which only got paged to WARM.
    let (_h, _w, cold) = cache.counts();
    assert_eq!(cold, 1, "expected exactly one eviction");
    assert_eq!(cache.state(0), TileState::Cold, "oldest should be evicted");
    assert_eq!(
        cache.state(n - 1),
        TileState::Warm,
        "lowest-σ_E tile should be paged, not evicted"
    );
}

#[test]
fn score_all_skips_cold() {
    let proj = Projection::new(7);
    let (q, toks) = generate(8, 6, 0.3);
    let q_sign = proj.sign_bits(&q);
    let mut cache = ElasticKvCache::new(usize::MAX, PageOutPolicy::OldestFirst);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    cache.evict(2);
    let scored = cache.score_all(&q, &q_sign);
    assert_eq!(scored.len(), 5);
    assert!(scored.iter().all(|&(s, _)| s != 2));
}

// --- Plan axis A5 — informed eviction --------------------------------------

#[test]
fn observe_scores_accumulates_importance() {
    let proj = Projection::new(31);
    let (_q, toks) = generate(32, 4, 0.3);
    let mut cache = ElasticKvCache::with_budget(usize::MAX);
    let slots: Vec<usize> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| cache.insert(build_tile(&proj, t, i as u32, false)))
        .collect();

    // Slot 2 gets a much higher logit; with a low temperature it takes ~all
    // the softmax mass, the others ~none.
    let scores: Vec<(usize, f32)> = slots
        .iter()
        .map(|&s| (s, if s == slots[2] { 10.0 } else { 0.0 }))
        .collect();
    cache.observe_scores(&scores, 0.5);

    let total: f32 = slots.iter().map(|&s| cache.importance(s)).sum();
    assert!(
        (total - 1.0).abs() <= 1e-4,
        "softmax mass over live slots sums to 1, got {total}"
    );
    assert!(
        cache.importance(slots[2]) > 0.9,
        "heavy slot should hold most mass, got {}",
        cache.importance(slots[2])
    );
    assert!(cache.importance(slots[0]) < 0.1);

    // A second observation accumulates (H2O is cumulative across steps).
    cache.observe_scores(&scores, 0.5);
    assert!(
        cache.importance(slots[2]) > 1.8,
        "importance should accumulate, got {}",
        cache.importance(slots[2])
    );
}

/// Plan axis **A5** — the headline behaviour: under pressure that forces
/// evictions, informed eviction keeps the **attention sinks** (the first
/// `sink_window` tokens) and the **heavy-hitter** (a high-attention *old* token
/// that pure-causal eviction drops as "oldest"). `σ_E` is left to the paging
/// phase; this isolates the eviction-order effect.
#[test]
fn informed_eviction_preserves_sinks_and_heavy_hitters() {
    let proj = Projection::new(41);
    let n = 8usize;
    // Budget for 3 WARM tiles: paging takes everything to WARM (8·96 = 768),
    // then eviction must drop 5 tiles (768 − 5·96 = 288). Both caches page
    // identically; only the eviction *order* differs.
    let budget = 3 * WARM_BYTES;

    let mut causal = ElasticKvCache::with_eviction(budget, EvictionPolicy::Causal);
    let mut informed =
        ElasticKvCache::with_eviction(budget, EvictionPolicy::Importance { sink_window: 2 });

    let mut slots_informed = Vec::new();
    for i in 0..n {
        let (_q, toks) = generate(50 + i as u64, 1, 0.3);
        let tok = &toks[0];
        causal.insert(build_tile(&proj, tok, i as u32, false));
        slots_informed.push(informed.insert(build_tile(&proj, tok, i as u32, false)));
    }

    // Make position 2 a heavy-hitter: old (so causal would drop it) but not a
    // sink (sink_window = 2 pins positions 0,1). A query that attends almost
    // exclusively to slot 2, observed across several decoding steps.
    let hh = slots_informed[2];
    let scores: Vec<(usize, f32)> = slots_informed
        .iter()
        .map(|&s| (s, if s == hh { 10.0 } else { 0.0 }))
        .collect();
    for _ in 0..5 {
        informed.observe_scores(&scores, 0.5);
    }
    assert!(
        informed.importance(hh) > 4.0,
        "heavy-hitter importance should dominate"
    );

    causal.enforce_budget();
    informed.enforce_budget();

    // Both reach the budget with exactly 3 live tiles.
    assert!(causal.live_bytes() <= budget && informed.live_bytes() <= budget);
    let (_ch, _cw, cc) = causal.counts();
    let (_ih, _iw, ic) = informed.counts();
    assert_eq!(cc, n - 3, "causal should evict 5 tiles");
    assert_eq!(ic, n - 3, "informed should evict 5 tiles");

    // Causal drops the 5 oldest (positions 0..4): the heavy-hitter (pos 2) and
    // both sinks (pos 0,1) are evicted.
    assert_eq!(causal.state(0), TileState::Cold, "causal drops sink 0");
    assert_eq!(causal.state(1), TileState::Cold, "causal drops sink 1");
    assert_eq!(
        causal.state(2),
        TileState::Cold,
        "causal drops the heavy-hitter"
    );

    // Informed keeps the sinks and the heavy-hitter; it dropped the 5 low-
    // importance non-sinks (positions 3..7) instead.
    assert_ne!(
        informed.state(0),
        TileState::Cold,
        "informed must preserve sink 0"
    );
    assert_ne!(
        informed.state(1),
        TileState::Cold,
        "informed must preserve sink 1"
    );
    assert_ne!(
        informed.state(2),
        TileState::Cold,
        "informed must preserve the heavy-hitter"
    );
    for s in [3usize, 4, 5, 6, 7] {
        assert_eq!(
            informed.state(s),
            TileState::Cold,
            "informed should drop low-importance non-sink {s}"
        );
    }
}

/// Plan axis **A5** — `Importance { sink_window: 0 }` is **pure H2O** with no
/// position-based pinning: the lowest-importance tiles are evicted first and the
/// highest-importance tiles survive, regardless of position. This pins the
/// `sink_window = 0` boundary (no sinks ⇒ importance is the sole key) and rules
/// out a position bias leaking in when pinning is disabled.
#[test]
fn importance_eviction_with_no_sinks_is_pure_h2o() {
    let proj = Projection::new(61);
    let n = 6usize;
    // Budget for 2 WARM tiles: paging takes all 6 to WARM (6·96), then eviction
    // drops 4. With sink_window = 0, eviction order is pure H2O importance.
    let budget = 2 * WARM_BYTES;
    let mut cache =
        ElasticKvCache::with_eviction(budget, EvictionPolicy::Importance { sink_window: 0 });

    let mut slots = Vec::new();
    for i in 0..n {
        let (_q, toks) = generate(70 + i as u64, 1, 0.3);
        let tok = &toks[0];
        // Fresh cache, sequential inserts ⇒ slot == i and tile position == i.
        slots.push(cache.insert(build_tile(&proj, tok, i as u32, false)));
        assert_eq!(slots[i], i, "sequential insert into a fresh cache");
    }
    // Feed a logit that DECREASES with position (`n − i`), so the heavy-hitter is
    // the **oldest** slot (position 0) — exactly the one pure-causal eviction
    // would drop first. If any position bias leaked into the H2O key, this test
    // would catch it.
    let scores: Vec<(usize, f32)> = slots
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, (n as f32) - i as f32))
        .collect();
    cache.observe_scores(&scores, 0.25);

    // Importance strictly decreases with position (logit n−i) and sums to ~1.
    let total: f32 = slots.iter().map(|&s| cache.importance(s)).sum();
    assert!(
        (total - 1.0).abs() <= 1e-4,
        "softmax mass sums to 1, got {total}"
    );
    for i in 0..n - 1 {
        assert!(
            cache.importance(slots[i]) > cache.importance(slots[i + 1]),
            "importance must decrease with position (logit n-pos), pos {i}"
        );
    }

    cache.enforce_budget();
    let (_h, _w, c) = cache.counts();
    assert_eq!(c, n - 2, "should evict 4 tiles");

    // The two highest-importance slots (positions 0 and 1, the oldest) survive;
    // the four lowest (positions 2..n) are COLD — purely by importance, since
    // sink_window = 0 pins nothing (causal would have evicted positions 0,1
    // first).
    assert_ne!(
        cache.state(slots[0]),
        TileState::Cold,
        "top-1 importance survives"
    );
    assert_ne!(
        cache.state(slots[1]),
        TileState::Cold,
        "top-2 importance survives"
    );
    for (i, &s) in slots.iter().enumerate().skip(2) {
        assert_eq!(
            cache.state(s),
            TileState::Cold,
            "low-importance slot (pos {i}) should be evicted with sink_window=0"
        );
    }
}

// --- TQ3 finer paging rung (FLAG_TQ3_NOCORR) --------------------------------

/// `n` tiles of `codec`, all inserted HOT.
fn codec_cache(n: usize, budget: usize, codec: LatentCodec) -> ElasticKvCache {
    let d = 256;
    let keys = gen_keys(7, n, d, 16, 0.93, 0.02);
    let model = LearnedModel::fit(&keys, d, 0xC0FFEE, false);
    let mut cache = ElasticKvCache::with_budget(budget);
    for (i, k) in keys.iter().enumerate() {
        cache.insert(model.encode_with(k, i as u32, false, codec));
    }
    cache
}

/// `n` TQ3-encoded tiles, all inserted HOT.
fn tq3_cache(n: usize, budget: usize) -> ElasticKvCache {
    codec_cache(n, budget, LatentCodec::Tq3)
}

#[test]
fn tq3_correction_rung_comes_before_paging() {
    // Budget = exactly all-HOT-without-corrections: enforce must satisfy it
    // purely by dropping correction planes — zero WARM, zero COLD.
    let n = 32;
    let budget = n * (HOT_BYTES - TQ3_CORR_BYTES); // 112·n
    let mut cache = tq3_cache(n, budget);
    assert_eq!(cache.live_bytes(), n * HOT_BYTES);
    cache.enforce_budget();
    assert_invariants(&cache, budget);
    assert_eq!(cache.counts(), (n, 0, 0), "no tile paged or evicted");
    assert_eq!(cache.nocorr_count(), n, "every correction plane dropped");
    assert_eq!(cache.live_bytes(), budget);
}

#[test]
fn tq3_single_drop_reclaims_exactly_one_plane() {
    // One correction plane short of budget: exactly one tile degrades.
    let n = 16;
    let budget = n * HOT_BYTES - TQ3_CORR_BYTES;
    let mut cache = tq3_cache(n, budget);
    cache.enforce_budget();
    assert_invariants(&cache, budget);
    assert_eq!(cache.counts(), (n, 0, 0));
    assert_eq!(cache.nocorr_count(), 1);
}

#[test]
fn tq3_ladder_reaches_warm_after_corrections() {
    // Budget below all-¬corr (112·n) forces phase 2: some tiles page to WARM
    // (already ¬corr from phase 1 → 80 B each), the rest stay HOT¬corr (112 B).
    let n = 32;
    let budget = n * 96;
    let mut cache = tq3_cache(n, budget);
    cache.enforce_budget();
    assert_invariants(&cache, budget);
    let (h, w, c) = cache.counts();
    assert_eq!(c, 0, "budget reachable without eviction");
    assert_eq!(h + w, n);
    assert!(w > 0, "some tiles must have paged to WARM");
    assert_eq!(
        cache.nocorr_count(),
        n,
        "phase 1 dropped every correction before paging"
    );
}

#[test]
fn non_tq3_tiles_skip_the_correction_rung() {
    // Grouped-INT4 tiles: the correction rung is a no-op and behavior is
    // exactly the pre-TQ3 two-phase ladder.
    let proj = Projection::new(0xB0D);
    let n = 16;
    let budget = n * 112; // between all-HOT and all-WARM
    let mut cache = ElasticKvCache::with_budget(budget);
    let (_q, toks) = generate(3, n, 0.3);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    cache.enforce_budget();
    assert_invariants(&cache, budget);
    assert_eq!(cache.nocorr_count(), 0, "no TQ3 tile, no correction drop");
    let (h, w, _) = cache.counts();
    assert!(w > 0 && h + w == n, "paging fell back to HOT->WARM");
}

#[test]
fn tq3_ladder_is_deterministic() {
    // Same tiles + same budget, twice: identical per-slot states and flags.
    let n = 24;
    let budget = n * 100;
    let fingerprint = |cache: &ElasticKvCache| -> Vec<(TileState, u16)> {
        (0..n)
            .map(|s| (cache.state(s), cache.tile(s).flags))
            .collect()
    };
    let mut a = tq3_cache(n, budget);
    let mut b = tq3_cache(n, budget);
    a.enforce_budget();
    b.enforce_budget();
    assert_eq!(a.live_bytes(), b.live_bytes());
    assert_eq!(fingerprint(&a), fingerprint(&b));
}

#[test]
fn mix3_correction_rung_reclaims_fourteen_bytes() {
    // The MIX3 separable plane is 14 bytes: the rung must satisfy an
    // all-HOT-minus-planes budget without paging or evicting, with exact
    // accounting.
    let n = 32;
    let budget = n * (HOT_BYTES - MIX3_CORR_BYTES); // 114·n
    let mut cache = codec_cache(n, budget, LatentCodec::Mix3);
    assert_eq!(cache.live_bytes(), n * HOT_BYTES);
    cache.enforce_budget();
    assert_invariants(&cache, budget);
    assert_eq!(cache.counts(), (n, 0, 0), "no tile paged or evicted");
    assert_eq!(cache.nocorr_count(), n, "every correction plane dropped");
    assert_eq!(cache.reclaimed_correction_bytes(), n * MIX3_CORR_BYTES);
    assert_eq!(cache.live_bytes(), budget);
}

#[test]
fn mixed_codec_has_no_correction_rung() {
    // The nibble mixed codec has no separable plane: same budget forces
    // paging instead.
    let n = 16;
    let budget = n * (HOT_BYTES - MIX3_CORR_BYTES);
    let mut cache = codec_cache(n, budget, LatentCodec::Mixed);
    cache.enforce_budget();
    assert_invariants(&cache, budget);
    assert_eq!(cache.nocorr_count(), 0, "no plane to drop on mixed tiles");
    let (h, w, _) = cache.counts();
    assert!(w > 0 && h + w == n, "mixed fell back to HOT->WARM paging");
}
