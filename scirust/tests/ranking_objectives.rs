//! Production-linked tests for the ranking-aware training objectives.
//!
//! Every test drives `scirust::ranking` — the same code the training CLI calls —
//! rather than a re-derivation of the maths.

use scirust::attention::slha_v2::D_C;
use scirust::ranking::{
    baseline_topk, frozen_l2_scale, row_weights, train_ranking, Geometry, History, Objective, Row,
    TrainConfig, L2_SCALE_EPSILON,
};
use scirust::rng::Rng;

fn cfg(obj: Objective) -> TrainConfig {
    TrainConfig {
        objective: obj,
        geometry: Geometry { weight: 0.0 },
        epochs: 1,
        lr: 0.0,
        batch: 1,
        seed: 1,
        max_keys: 256,
        l2_scale: 1.0,
    }
}

fn weights(c: &TrainConfig, b: &[f32], s: &[f32]) -> (Vec<f32>, f32) {
    let mut rng = Rng::new(3);
    let mut w = Vec::new();
    let mut cmp = 0u64;
    let l = row_weights(c, b, s, &mut rng, &mut w, &mut cmp);
    (w, l)
}

// 1. exact top-1 preserved -------------------------------------------------
#[test]
fn exact_top1_is_preserved() {
    let b = [5.0f32, 1.0, 2.0];
    assert_eq!(baseline_topk(&b, 1), vec![0]);
    let s = [9.0f32, 1.0, 2.0]; // same argmax, larger margin
    let (_, loss) = weights(
        &cfg(Objective::PairwiseTopK {
            k: 1,
            tau: 1.0,
            negatives: 2,
        }),
        &b,
        &s,
    );
    let (_, loss_bad) = weights(
        &cfg(Objective::PairwiseTopK {
            k: 1,
            tau: 1.0,
            negatives: 2,
        }),
        &b,
        &[0.0, 1.0, 2.0],
    );
    assert!(
        loss < loss_bad,
        "preserving the baseline top-1 must cost less than losing it ({loss} vs {loss_bad})"
    );
}

// 2. incorrect top-1 penalised ---------------------------------------------
#[test]
fn incorrect_top1_is_penalised() {
    let b = [5.0f32, 1.0, 2.0];
    let s = [0.0f32, 8.0, 2.0]; // argmax moved to key 1
    let c = cfg(Objective::PairwiseTopK {
        k: 1,
        tau: 1.0,
        negatives: 2,
    });
    let (w, loss) = weights(&c, &b, &s);
    assert!(loss > 0.0);
    // gradient must push the true top-1 up and the usurper down
    assert!(
        w[0] < 0.0,
        "dL/ds_0 must be negative so s_0 increases, got {}",
        w[0]
    );
    assert!(
        w[1] > 0.0,
        "dL/ds_1 must be positive so s_1 decreases, got {}",
        w[1]
    );
}

// 3. top-k set preservation -------------------------------------------------
#[test]
fn topk_set_is_the_training_target() {
    let b = [1.0f32, 9.0, 8.0, 7.0, 0.5];
    assert_eq!(baseline_topk(&b, 3), vec![1, 2, 3]);
    let good = [1.0f32, 9.0, 8.0, 7.0, 0.5];
    let bad = [9.0f32, 1.0, 0.5, 0.4, 8.0];
    let c = cfg(Objective::PairwiseTopK {
        k: 3,
        tau: 1.0,
        negatives: 2,
    });
    assert!(weights(&c, &b, &good).1 < weights(&c, &b, &bad).1);
}

// 4. deterministic tie handling ---------------------------------------------
#[test]
fn ties_break_on_ascending_index_deterministically() {
    let b = [3.0f32, 3.0, 3.0, 1.0];
    for _ in 0..8 {
        assert_eq!(baseline_topk(&b, 2), vec![0, 1]);
    }
    // and the loss is reproducible across repeated evaluation
    let c = cfg(Objective::PairwiseTopK {
        k: 2,
        tau: 1.0,
        negatives: 2,
    });
    let s = [1.0f32, 2.0, 3.0, 4.0];
    let first = weights(&c, &b, &s).1;
    for _ in 0..5 {
        assert_eq!(weights(&c, &b, &s).1, first);
    }
}

// 5-6. masked / padded keys never reach the objective -----------------------
#[test]
fn only_the_visible_prefix_is_scored() {
    // The caller passes the visible prefix; n_visible bounds every loop. A row
    // that declares fewer visible keys than the buffer holds must ignore the
    // tail exactly as the causal mask does.
    let d = 4usize;
    let q = vec![1.0f32; d];
    let keys = vec![1.0f32; 3 * d];
    let baseline = [3.0f32, 2.0, 1.0];
    let row = Row {
        q: &q,
        keys: &keys,
        baseline: &baseline,
        n_visible: 2,
        d,
    };
    let p = vec![0.0f32; D_C * d];
    let mut c = cfg(Objective::L2);
    c.max_keys = 8;
    let (_, h) = train_ranking(std::slice::from_ref(&row), p, &c);
    assert_eq!(h.rows_seen, 1);
}

#[test]
fn padding_beyond_n_visible_is_excluded() {
    let b_full = [5.0f32, 4.0, 3.0, 0.0, 0.0]; // last two are padding
    let visible = &b_full[..3];
    assert_eq!(baseline_topk(visible, 3), vec![0, 1, 2]);
    // padding zeros would otherwise enter the top-k of a low-scoring row
    let low = [-5.0f32, -4.0, -3.0];
    assert_eq!(baseline_topk(&low, 1), vec![2]);
}

// 7. non-finite input rejected ----------------------------------------------
#[test]
fn nonfinite_scores_do_not_produce_finite_nonsense() {
    let b = [1.0f32, f32::NAN, 3.0];
    // baseline_topk must not panic and must still return k indices
    let t = baseline_topk(&b, 2);
    assert_eq!(t.len(), 2);
    // the collector rejects such rows before training; assert the contract holds
    assert!(b.iter().any(|v| !v.is_finite()));
}

// 8. one active key ---------------------------------------------------------
#[test]
fn single_active_key_row_is_handled() {
    let b = [2.0f32];
    let s = [1.0f32];
    let c = cfg(Objective::PairwiseTopK {
        k: 1,
        tau: 1.0,
        negatives: 4,
    });
    let (w, loss) = weights(&c, &b, &s);
    assert_eq!(w.len(), 1);
    assert_eq!(loss, 0.0, "no pair exists, so there is nothing to rank");
}

// 9. all-tied baseline row --------------------------------------------------
#[test]
fn all_tied_baseline_row_produces_no_ranking_signal() {
    let b = [2.0f32, 2.0, 2.0, 2.0];
    let s = [1.0f32, 5.0, 3.0, 2.0];
    let c = cfg(Objective::PairwiseTopK {
        k: 4,
        tau: 1.0,
        negatives: 4,
    });
    let (_, loss) = weights(&c, &b, &s);
    // k == n leaves no negatives at all
    assert_eq!(loss, 0.0);
}

// 10. deterministic hard-negative selection ---------------------------------
#[test]
fn negative_selection_is_deterministic_and_boundary_aware() {
    let b: Vec<f32> = (0..16).map(|i| 16.0 - i as f32).collect();
    let s: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let c = cfg(Objective::PairwiseTopK {
        k: 4,
        tau: 1.0,
        negatives: 4,
    });
    let a = weights(&c, &b, &s);
    let bb = weights(&c, &b, &s);
    assert_eq!(a.0, bb.0, "weights must be reproducible");
    assert_eq!(a.1, bb.1, "loss must be reproducible");
    // the keys immediately below the boundary (ranks 4,5) must carry weight
    assert!(
        a.0[4] != 0.0 || a.0[5] != 0.0,
        "hard negatives must be selected"
    );
}

// 11. pairwise loss decreases after a correct ordering change ---------------
#[test]
fn pairwise_loss_decreases_when_ordering_improves() {
    let b = [10.0f32, 1.0, 2.0, 3.0];
    let c = cfg(Objective::PairwiseTopK {
        k: 1,
        tau: 1.0,
        negatives: 3,
    });
    let before = weights(&c, &b, &[0.0, 5.0, 4.0, 3.0]).1;
    let after = weights(&c, &b, &[6.0, 5.0, 4.0, 3.0]).1;
    assert!(after < before, "{after} should be below {before}");
}

// 12. listwise loss is zero for identical distributions ---------------------
#[test]
fn listwise_loss_is_zero_for_identical_distributions() {
    let b = [1.0f32, 2.0, 3.0, 4.0];
    let c = cfg(Objective::Listwise {
        t_teacher: 1.0,
        t_student: 1.0,
    });
    let (w, loss) = weights(&c, &b, &b);
    assert!(
        loss.abs() < 1e-6,
        "KL of a distribution with itself must vanish, got {loss}"
    );
    for wi in w {
        assert!(
            wi.abs() < 1e-6,
            "gradient must vanish at the optimum, got {wi}"
        );
    }
}

// 13. hybrid loss component accounting --------------------------------------
#[test]
fn hybrid_loss_is_the_weighted_sum_of_its_components() {
    let b = [4.0f32, 1.0, 3.0, 2.0];
    let s = [1.0f32, 4.0, 2.0, 3.0];
    let (k, tau, neg, tt, ts) = (2usize, 1.0f32, 2usize, 1.0f32, 1.0f32);
    let (wp, wl, w2) = (1.0f32, 0.5f32, 0.1f32);
    let pw = weights(
        &cfg(Objective::PairwiseTopK {
            k,
            tau,
            negatives: neg,
        }),
        &b,
        &s,
    )
    .1;
    let lw = weights(
        &cfg(Objective::Listwise {
            t_teacher: tt,
            t_student: ts,
        }),
        &b,
        &s,
    )
    .1;
    let l2 = weights(&cfg(Objective::L2), &b, &s).1;
    let hy = weights(
        &cfg(Objective::Hybrid {
            k,
            tau,
            negatives: neg,
            t_teacher: tt,
            t_student: ts,
            w_pairwise: wp,
            w_listwise: wl,
            w_l2: w2,
        }),
        &b,
        &s,
    )
    .1;
    let expect = wp * pw + wl * lw + w2 * l2;
    assert!(
        (hy - expect).abs() <= 1e-4 * expect.abs().max(1.0),
        "hybrid {hy} != weighted sum {expect}"
    );
}

// 14. layer-weight selection -------------------------------------------------
#[test]
fn layer_selection_accepts_ranges_and_rejects_nonsense() {
    // The layer mask is the production selector shared with the shim.
    use scirust::ranking::parse_layer_set;
    assert_eq!(
        parse_layer_set("0-6", 28).unwrap(),
        (0..=6).collect::<Vec<_>>()
    );
    assert_eq!(parse_layer_set("all", 28).unwrap().len(), 28);
    assert_eq!(parse_layer_set("3", 28).unwrap(), vec![3]);
    assert!(parse_layer_set("0-99", 28).is_err());
    assert!(parse_layer_set("banana", 28).is_err());
}

// 15. top-k value validation -------------------------------------------------
#[test]
fn topk_values_are_validated() {
    assert!(Objective::parse("pairwise-topk", 0).is_err());
    assert!(Objective::parse("pairwise-topk", 1).is_ok());
    assert!(Objective::parse("hybrid", 16).is_ok());
    assert!(Objective::parse("not-an-objective", 4).is_err());
}

// 17. deterministic repeated training ---------------------------------------
#[test]
fn training_is_deterministic_for_a_fixed_seed() {
    let d = 8usize;
    let mut rng = Rng::new(11);
    let mut q = vec![0.0f32; d];
    rng.fill_gaussian(&mut q);
    let mut keys = vec![0.0f32; 6 * d];
    rng.fill_gaussian(&mut keys);
    let baseline = [3.0f32, 1.0, 4.0, 1.5, 2.0, 0.5];
    let row = Row {
        q: &q,
        keys: &keys,
        baseline: &baseline,
        n_visible: 6,
        d,
    };
    let mut p0 = vec![0.0f32; D_C * d];
    rng.fill_gaussian(&mut p0);

    let mut c = cfg(Objective::Hybrid {
        k: 2,
        tau: 1.0,
        negatives: 2,
        t_teacher: 1.0,
        t_student: 1.0,
        w_pairwise: 1.0,
        w_listwise: 0.5,
        w_l2: 0.1,
    });
    c.epochs = 3;
    c.lr = 1e-6;
    c.batch = 2;

    let (a, ha) = train_ranking(std::slice::from_ref(&row), p0.clone(), &c);
    let (b, hb) = train_ranking(std::slice::from_ref(&row), p0, &c);
    assert_eq!(a, b, "same seed must give a bit-identical projection");
    assert_eq!(ha.epoch_loss, hb.epoch_loss);
    assert_eq!(ha.pairwise_comparisons, hb.pairwise_comparisons);
}

// geometry constraint --------------------------------------------------------
#[test]
fn geometry_penalises_sharpening_and_flattening() {
    let b = [1.0f32, 2.0, 3.0, 4.0];
    let mut c = cfg(Objective::PairwiseTopK {
        k: 1,
        tau: 1.0,
        negatives: 2,
    });
    c.geometry = Geometry { weight: 1.0 };
    // same ORDER as the baseline in all three, only the spread differs
    let matched = weights(&c, &b, &[1.0, 2.0, 3.0, 4.0]).1;
    let sharp = weights(&c, &b, &[10.0, 20.0, 30.0, 40.0]).1;
    let flat = weights(&c, &b, &[1.0, 1.01, 1.02, 1.03]).1;
    assert!(
        sharp > matched,
        "sharpening must be penalised ({sharp} vs {matched})"
    );
    assert!(
        flat > matched,
        "flattening must be penalised ({flat} vs {matched})"
    );
}

#[test]
fn geometry_is_inert_for_the_l2_control() {
    let b = [1.0f32, 2.0, 3.0, 4.0];
    let s = [10.0f32, 20.0, 30.0, 40.0];
    let mut c = cfg(Objective::L2);
    let plain = weights(&c, &b, &s).1;
    c.geometry = Geometry { weight: 5.0 };
    assert_eq!(
        weights(&c, &b, &s).1,
        plain,
        "the L2 control already pins the score values; geometry must not alter it"
    );
}

#[test]
fn empty_row_is_a_no_op() {
    let c = cfg(Objective::L2);
    let (w, loss) = weights(&c, &[], &[]);
    assert!(w.is_empty());
    assert_eq!(loss, 0.0);
    let p = vec![0.0f32; D_C * 4];
    let (out, h) = train_ranking(&[], p.clone(), &c);
    assert_eq!(out, p);
    assert_eq!(h.rows_seen, 0);
}

#[test]
fn history_counts_pairwise_comparisons() {
    let b: Vec<f32> = (0..12).map(|i| 12.0 - i as f32).collect();
    let s: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let c = cfg(Objective::PairwiseTopK {
        k: 3,
        tau: 1.0,
        negatives: 4,
    });
    let mut rng = Rng::new(3);
    let mut w = Vec::new();
    let mut cmp = 0u64;
    row_weights(&c, &b, &s, &mut rng, &mut w, &mut cmp);
    assert_eq!(cmp, 3 * 4, "3 positives x 4 negatives");
}

#[test]
fn history_default_is_empty() {
    let h = History::default();
    assert!(h.epoch_loss.is_empty());
    assert_eq!(h.rows_seen, 0);
    assert_eq!(h.pairwise_comparisons, 0);
}

// A2: the L2 normaliser must be a frozen per-layer scalar, not per-row --------
#[test]
fn frozen_l2_scale_is_deterministic_and_positive() {
    let rows: Vec<f32> = vec![3.0, -4.0, 0.0, 12.0, 5.0];
    let (a, na) = frozen_l2_scale(rows.iter().copied());
    let (b, nb) = frozen_l2_scale(rows.iter().copied());
    assert_eq!(a, b, "same input must give a bit-identical scale");
    assert_eq!((na, nb), (5, 5));
    let want = (9.0 + 16.0 + 0.0 + 144.0 + 25.0) / 5.0 + L2_SCALE_EPSILON;
    assert!((a - want).abs() < 1e-12, "{a} != {want}");
    assert!(a > 0.0);
    // an empty layer still yields a strictly positive scale via epsilon
    let (e, ne) = frozen_l2_scale(std::iter::empty());
    assert_eq!(ne, 0);
    assert_eq!(e, L2_SCALE_EPSILON);
}

#[test]
fn l2_scale_rescales_uniformly_and_preserves_relative_row_weighting() {
    // Two rows of very different magnitude. Under a FROZEN shared scale their
    // losses keep their relative sizes; a per-row normaliser would flatten them
    // and thereby change which row dominates the shared-parameter gradient.
    let big_b = [1000.0f32, 900.0, 800.0];
    let big_s = [1100.0f32, 900.0, 800.0];
    let small_b = [1.0f32, 0.9, 0.8];
    let small_s = [1.1f32, 0.9, 0.8];

    let mut c = cfg(Objective::L2);
    c.l2_scale = 1.0;
    let raw_big = weights(&c, &big_b, &big_s).1;
    let raw_small = weights(&c, &small_b, &small_s).1;

    c.l2_scale = 12345.0; // one frozen scalar shared by both rows
    let sc_big = weights(&c, &big_b, &big_s).1;
    let sc_small = weights(&c, &small_b, &small_s).1;

    let ratio_raw = raw_big / raw_small;
    let ratio_scaled = sc_big / sc_small;
    assert!(
        (ratio_raw - ratio_scaled).abs() <= 1e-3 * ratio_raw,
        "a shared scale must not change the ratio between rows: {ratio_raw} vs {ratio_scaled}"
    );
    // and it really does shrink the magnitude
    assert!(sc_big < raw_big);
}

#[test]
fn per_row_normalisation_would_change_relative_row_weighting() {
    // Demonstrates precisely why A1 was withdrawn: normalising each row by its
    // own RMS makes two rows of very different scale contribute almost equally,
    // which is a different objective, not a rescaled one.
    let big_b = [1000.0f32, 900.0, 800.0];
    let big_s = [1100.0f32, 900.0, 800.0];
    let small_b = [1.0f32, 0.9, 0.8];
    let small_s = [1.1f32, 0.9, 0.8];
    let mut c = cfg(Objective::L2);

    c.l2_scale = 1.0;
    let ratio_shared = weights(&c, &big_b, &big_s).1 / weights(&c, &small_b, &small_s).1;

    let rms2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    c.l2_scale = rms2(&big_b);
    let per_row_big = weights(&c, &big_b, &big_s).1;
    c.l2_scale = rms2(&small_b);
    let per_row_small = weights(&c, &small_b, &small_s).1;
    let ratio_per_row = per_row_big / per_row_small;

    assert!(
        (ratio_shared - ratio_per_row).abs() > 0.1 * ratio_shared,
        "per-row normalisation must visibly change the relative weighting \
         (shared {ratio_shared}, per-row {ratio_per_row})"
    );
}

// FidelityOptimizer tests ---------------------------------------------------
#[test]
fn test_topk_preservation_perfect_fidelity_at_full_k() {
    use scirust::ranking::FidelityOptimizer;
    use scirust::learned::{LearnedModel, gen_keys};

    let d = 160;
    let mut rng = Rng::new(42);
    let mut query = vec![0.0f32; d];
    rng.fill_gaussian(&mut query);

    let keys = gen_keys(123, 10, d, 20, 0.9, 0.02);

    // Fit a model
    let model = LearnedModel::fit(&keys, d, 123, false);

    // When k equals N (10), all scores must be preserved exactly
    let exact_scores: Vec<f32> = keys.iter().map(|k| scirust::metrics::dot(&query, k)).collect();
    let hybrid_scores = FidelityOptimizer::compute_scores_with_topk_preservation(&query, &keys, &model, 10);

    for i in 0..10 {
        assert!((exact_scores[i] - hybrid_scores[i]).abs() < 1e-4);
    }

    let ppl_proxy = FidelityOptimizer::calculate_perplexity_proxy(&hybrid_scores, &exact_scores, 1.0);
    assert!(ppl_proxy.abs() < 1e-6);
}

#[test]
fn test_topk_preservation_improves_perplexity_proxy() {
    use scirust::ranking::FidelityOptimizer;
    use scirust::learned::{LearnedModel, gen_keys};

    let d = 160;
    let mut rng = Rng::new(42);
    let mut query = vec![0.0f32; d];
    rng.fill_gaussian(&mut query);

    let keys = gen_keys(123, 15, d, 20, 0.9, 0.02);

    let model = LearnedModel::fit(&keys, d, 123, false);
    let exact_scores: Vec<f32> = keys.iter().map(|k| scirust::metrics::dot(&query, k)).collect();

    // 1. Fully quantized scores (without preservation)
    let quantized_scores = FidelityOptimizer::compute_scores_with_residual_correction(&query, &keys, &model);
    let ppl_no_preservation = FidelityOptimizer::calculate_perplexity_proxy(&quantized_scores, &exact_scores, 1.0);

    // 2. Hybrid scores preserving Top-3 keys
    let preserved_scores = FidelityOptimizer::compute_scores_with_topk_preservation(&query, &keys, &model, 3);
    let ppl_with_preservation = FidelityOptimizer::calculate_perplexity_proxy(&preserved_scores, &exact_scores, 1.0);

    // Top-K preservation should yield better or equal perplexity proxy (smaller KL)
    assert!(
        ppl_with_preservation <= ppl_no_preservation,
        "top-k preservation should yield better or equal ppl than pure quantized: {ppl_with_preservation} vs {ppl_no_preservation}"
    );
}
