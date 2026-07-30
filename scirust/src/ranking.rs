//! Ranking-aware training of the SLHA projection.
//!
//! The rank-transplant experiment (`integration/llama.cpp/results/rank_transplant_oracle.json`)
//! showed that restoring the baseline ordering of the causally visible keys —
//! while keeping the SLHA score-value multiset — recovers most of the
//! compressed-attention quality gap. That oracle needs exact baseline scores at
//! inference and is therefore not deployable. This module trains the projection
//! so the ordering is preserved *without* baseline access at inference: baseline
//! logits are used only here, offline, as training labels.
//!
//! # Parameterisation
//!
//! The coarse SLHA score is `⟨Pq, Pk⟩` for a projection `P` of shape `D_C × d`
//! (row-major). Writing `a = Pq` and `b_j = Pk_j`, the score of key `j` is
//! `s_j = ⟨a, b_j⟩` and
//!
//! ```text
//! ds_j / dP[r][m] = q[m] * b_j[r] + k_j[m] * a[r]
//! ```
//!
//! Every objective below reduces to a per-key weight `w_j = dL/ds_j`, after
//! which the gradient accumulation is shared. Only the coarse term is trained;
//! the sign-LSH residual correction is a fixed, non-differentiable refinement
//! applied by the codec at encode time.
//!
//! # Objectives
//!
//! * [`Objective::L2`] — the existing score-reconstruction control,
//!   `Σ_j (B_j − s_j)²`. Unchanged behaviour, kept as the comparison baseline.
//! * [`Objective::PairwiseTopK`] — for every baseline top-`k` key `i` and every
//!   sampled non-top-`k` key `j`, a logistic ranking loss on the margin
//!   `s_i − s_j`. Negatives are sampled deterministically and always include the
//!   keys immediately below the top-`k` boundary, which are the ones a ranking
//!   error actually swaps.
//! * [`Objective::Listwise`] — `KL(softmax(B/T_teacher) ‖ softmax(s/T_student))`
//!   over the visible keys, at fixed documented temperatures.
//! * [`Objective::Hybrid`] — an explicit weighted sum of the above plus the
//!   geometry regulariser, with every component weight recorded.
//!
//! # Geometry
//!
//! A pure ranking objective is free to inflate margins without bound, which
//! sharpens or flattens the attention distribution even when the ordering is
//! perfect. [`Geometry`] adds a variance-ratio penalty that keeps the spread of
//! the trained scores close to the baseline's, so a ranking gain cannot be
//! bought by destroying the score geometry.

use crate::attention::slha_v2::D_C;
use crate::rng::Rng;

/// One causally visible training row: a query, the visible keys, and the
/// baseline logits that define the target ordering.
pub struct Row<'a> {
    pub q: &'a [f32],
    /// `n_visible` keys, each of length `d`, flattened row-major.
    pub keys: &'a [f32],
    /// Baseline logits, one per visible key. Training labels only.
    pub baseline: &'a [f32],
    pub n_visible: usize,
    pub d: usize,
}

impl Row<'_> {
    fn key(&self, j: usize) -> &[f32] {
        &self.keys[j * self.d..(j + 1) * self.d]
    }
}

/// Which loss to optimise. Selected by `SLHA_TRAIN_OBJECTIVE`.
#[derive(Clone, Debug, PartialEq)]
pub enum Objective {
    /// Score reconstruction (the existing control).
    L2,
    /// Logistic pairwise ranking over the baseline top-`k`.
    PairwiseTopK {
        k: usize,
        tau: f32,
        negatives: usize,
    },
    /// Distribution matching over the visible keys.
    Listwise { t_teacher: f32, t_student: f32 },
    /// Explicit weighted combination. Every weight is recorded in the manifest.
    Hybrid {
        k: usize,
        tau: f32,
        negatives: usize,
        t_teacher: f32,
        t_student: f32,
        w_pairwise: f32,
        w_listwise: f32,
        w_l2: f32,
    },
}

impl Objective {
    /// Parse the `SLHA_TRAIN_OBJECTIVE` spelling. Strict: an unknown name is an
    /// error rather than a silent fallback to the control.
    pub fn parse(spec: &str, k: usize) -> Result<Self, String> {
        if spec != "l2" && spec != "listwise" && k == 0 {
            return Err(format!(
                "objective {spec:?} needs a top-k size >= 1, got {k}"
            ));
        }
        match spec {
            "l2" => Ok(Objective::L2),
            "pairwise-topk" => Ok(Objective::PairwiseTopK {
                k,
                tau: DEFAULT_TAU,
                negatives: DEFAULT_NEGATIVES,
            }),
            "listwise" => Ok(Objective::Listwise {
                t_teacher: DEFAULT_T_TEACHER,
                t_student: DEFAULT_T_STUDENT,
            }),
            "hybrid" => Ok(Objective::Hybrid {
                k,
                tau: DEFAULT_TAU,
                negatives: DEFAULT_NEGATIVES,
                t_teacher: DEFAULT_T_TEACHER,
                t_student: DEFAULT_T_STUDENT,
                w_pairwise: DEFAULT_W_PAIRWISE,
                w_listwise: DEFAULT_W_LISTWISE,
                w_l2: DEFAULT_W_L2,
            }),
            other => Err(format!(
                "unknown SLHA_TRAIN_OBJECTIVE {other:?}; expected one of \
                 l2, pairwise-topk, listwise, hybrid"
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Objective::L2 => "l2",
            Objective::PairwiseTopK { .. } => "pairwise-topk",
            Objective::Listwise { .. } => "listwise",
            Objective::Hybrid { .. } => "hybrid",
        }
    }
}

/// Parse a layer selection: `all`, a single index, or an inclusive `a-b` range.
///
/// Strict: an out-of-range or unparseable selection is an error, never a silent
/// "all layers". The spelling matches the shim's `SLHA_SCORE_LAYERS` so the
/// training scope and the measurement scope are written the same way.
pub fn parse_layer_set(spec: &str, n_layers: usize) -> Result<Vec<usize>, String> {
    let s = spec.trim();
    if s == "all" {
        return Ok((0..n_layers).collect());
    }
    if s == "none" {
        return Ok(Vec::new());
    }
    let bad = |w: &str| format!("cannot parse layer selection {w:?}");
    if let Some((a, b)) = s.split_once('-') {
        let lo: usize = a.trim().parse().map_err(|_| bad(s))?;
        let hi: usize = b.trim().parse().map_err(|_| bad(s))?;
        if lo > hi || hi >= n_layers {
            return Err(format!(
                "layer range {s:?} is outside 0..{} or is inverted",
                n_layers - 1
            ));
        }
        return Ok((lo..=hi).collect());
    }
    let one: usize = s.parse().map_err(|_| bad(s))?;
    if one >= n_layers {
        return Err(format!("layer {one} is outside 0..{}", n_layers - 1));
    }
    Ok(vec![one])
}

/// Fixed, documented defaults. They are NOT tuned against the final evaluation
/// set; they are chosen once and recorded in every training manifest.
pub const DEFAULT_TAU: f32 = 1.0;
pub const DEFAULT_NEGATIVES: usize = 8;
pub const DEFAULT_T_TEACHER: f32 = 1.0;
pub const DEFAULT_T_STUDENT: f32 = 1.0;
pub const DEFAULT_W_PAIRWISE: f32 = 1.0;
pub const DEFAULT_W_LISTWISE: f32 = 0.5;
pub const DEFAULT_W_L2: f32 = 0.1;
/// Weight of the variance-ratio penalty in every non-L2 objective.
pub const DEFAULT_W_GEOMETRY: f32 = 0.25;

/// Score-geometry constraint. Keeps the trained scores' spread near the
/// baseline's so a ranking gain cannot come from pathological sharpening or
/// flattening of the softmax.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub weight: f32,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            weight: DEFAULT_W_GEOMETRY,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub objective: Objective,
    pub geometry: Geometry,
    pub epochs: usize,
    pub lr: f32,
    pub batch: usize,
    pub seed: u64,
    /// Cap on visible keys considered per row. Keeps the pairwise term bounded
    /// on long rows; the retained prefix always contains the baseline top-k.
    pub max_keys: usize,
    /// FROZEN scale for the score-reconstruction term: `rms_B_L^2 + epsilon`,
    /// computed once per layer over that layer's TRAINING split only.
    ///
    /// This must not vary between rows. A per-row normaliser reweights rows
    /// against each other, and because the projection is shared across every
    /// row of the layer, `sum_r L_r` and `sum_r L_r / c_r` have different
    /// minimisers whenever `c_r` varies. A single fixed positive scalar per
    /// layer rescales the whole layer objective uniformly, so it preserves the
    /// layer's minimiser while keeping the gradient magnitude tractable.
    pub l2_scale: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            objective: Objective::L2,
            geometry: Geometry::default(),
            epochs: 4,
            lr: 1.0e-7,
            batch: 16,
            seed: 7,
            max_keys: 256,
            l2_scale: 1.0,
        }
    }
}

/// Documented epsilon guarding the frozen normaliser against a degenerate layer.
pub const L2_SCALE_EPSILON: f64 = 1.0e-6;

/// Frozen per-layer L2 scale `rms_B^2 + epsilon` from the TRAINING split only.
///
/// Deterministic: the caller supplies the rows in a fixed order and the
/// accumulation is f64 in that order, so the value is reproducible bit for bit.
/// Validation, diagnostic and test rows must never be passed here.
pub fn frozen_l2_scale(training_baselines: impl Iterator<Item = f32>) -> (f64, u64) {
    let mut acc = 0.0f64;
    let mut n = 0u64;
    for v in training_baselines {
        acc += (v as f64) * (v as f64);
        n += 1;
    }
    let ms = if n == 0 { 0.0 } else { acc / n as f64 };
    (ms + L2_SCALE_EPSILON, n)
}

#[derive(Clone, Debug, Default)]
pub struct History {
    pub epoch_loss: Vec<f32>,
    pub rows_seen: u64,
    pub pairwise_comparisons: u64,
}

/// Indices of the `k` largest baseline scores, ties broken by ascending index.
///
/// The deterministic index tiebreak matters: with exactly-equal baseline logits
/// the top-`k` set would otherwise depend on sort stability, and the training
/// target would not be reproducible.
pub fn baseline_topk(baseline: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..baseline.len()).collect();
    idx.sort_by(|&a, &b| {
        baseline[b]
            .partial_cmp(&baseline[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k.min(baseline.len()));
    idx
}

/// Deterministic negatives for one positive: the keys immediately below the
/// top-`k` boundary (the hard ones a ranking error actually swaps) padded with
/// a seeded sample from the rest of the row.
fn negatives_for(order: &[usize], k: usize, want: usize, rng: &mut Rng) -> Vec<usize> {
    let n = order.len();
    if n <= k {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(want);
    // hard negatives: ranks k, k+1, ... just outside the boundary
    for &j in order.iter().skip(k).take(want.div_ceil(2)) {
        out.push(j);
    }
    // remaining negatives sampled deterministically from the tail
    let tail = n - k;
    while out.len() < want && tail > 0 {
        let pick = order[k + (rng.next_u64() as usize) % tail];
        if !out.contains(&pick) {
            out.push(pick);
        } else if out.len() >= tail {
            break;
        }
    }
    out
}

fn softmax_into(src: &[f32], t: f32, dst: &mut Vec<f32>) {
    dst.clear();
    dst.reserve(src.len());
    let m = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &v in src {
        let e = ((v - m) / t).exp();
        dst.push(e);
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in dst.iter_mut() {
        *v *= inv;
    }
}

fn variance(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n
}

/// Per-key loss weights `w_j = dL/ds_j` and the loss value for one row.
///
/// Split out from the gradient accumulation so it can be unit-tested directly
/// against the production objective rather than a re-derivation.
pub fn row_weights(
    cfg: &TrainConfig,
    baseline: &[f32],
    scores: &[f32],
    rng: &mut Rng,
    w: &mut Vec<f32>,
    comparisons: &mut u64,
) -> f32 {
    let n = scores.len();
    w.clear();
    w.resize(n, 0.0);
    if n == 0 {
        return 0.0;
    }
    let mut loss = 0.0f32;
    let mut sb = Vec::new();
    let mut ss = Vec::new();

    let mut add_pairwise = |k: usize,
                            tau: f32,
                            negatives: usize,
                            scale: f32,
                            w: &mut Vec<f32>,
                            comparisons: &mut u64|
     -> f32 {
        let order = baseline_topk(baseline, n);
        let k = k.min(n);
        let mut l = 0.0f32;
        for &i in order.iter().take(k) {
            for j in negatives_for(&order, k, negatives, rng) {
                let m = (scores[i] - scores[j]) / tau;
                // logistic ranking loss, numerically stable
                l += if m > 0.0 {
                    (1.0 + (-m).exp()).ln()
                } else {
                    -m + (1.0 + m.exp()).ln()
                };
                // d/dm log(1+exp(-m)) = -sigmoid(-m)
                let g = -1.0 / (1.0 + m.exp());
                w[i] += scale * g / tau;
                w[j] -= scale * g / tau;
                *comparisons += 1;
            }
        }
        scale * l
    };

    let mut add_listwise = |t_teacher: f32, t_student: f32, scale: f32, w: &mut Vec<f32>| -> f32 {
        softmax_into(baseline, t_teacher, &mut sb);
        softmax_into(scores, t_student, &mut ss);
        let mut l = 0.0f32;
        for j in 0..n {
            if sb[j] > 0.0 {
                l += sb[j] * (sb[j].max(1e-30) / ss[j].max(1e-30)).ln();
            }
            // d KL / d s_j = (softmax(s)_j - softmax(B)_j) / T_student
            w[j] += scale * (ss[j] - sb[j]) / t_student;
        }
        scale * l
    };

    // Score reconstruction against a FROZEN per-layer scale.
    //
    // The raw attention logits reach ~1e4 on this model, so an unnormalised
    // squared error is ~1e8 and its gradient overflows f32 within one step. The
    // fix must not be a per-row normaliser: the projection is shared across
    // every row of the layer, and dividing each row by its own constant changes
    // the relative weight of rows, so `sum_r L_r` and `sum_r L_r / c_r` have
    // different minimisers. `cfg.l2_scale` is instead a single positive scalar
    // computed once per layer over the training split, so it rescales the whole
    // layer objective uniformly and leaves that layer's minimiser intact.
    let add_l2 = |scale: f32, w: &mut Vec<f32>| -> f32 {
        let nf = n as f32;
        let norm = cfg.l2_scale.max(f32::MIN_POSITIVE);
        let mut l = 0.0f32;
        for j in 0..n {
            let r = scores[j] - baseline[j];
            l += r * r / norm;
            w[j] += scale * 2.0 * r / (norm * nf);
        }
        scale * l / nf
    };

    match cfg.objective.clone() {
        Objective::L2 => {
            loss += add_l2(1.0, w);
        }
        Objective::PairwiseTopK { k, tau, negatives } => {
            loss += add_pairwise(k, tau, negatives, 1.0, w, comparisons);
        }
        Objective::Listwise {
            t_teacher,
            t_student,
        } => {
            loss += add_listwise(t_teacher, t_student, 1.0, w);
        }
        Objective::Hybrid {
            k,
            tau,
            negatives,
            t_teacher,
            t_student,
            w_pairwise,
            w_listwise,
            w_l2,
        } => {
            loss += add_pairwise(k, tau, negatives, w_pairwise, w, comparisons);
            loss += add_listwise(t_teacher, t_student, w_listwise, w);
            loss += add_l2(w_l2, w);
        }
    }

    // Geometry: penalise the squared log-ratio of the score variances. Applied
    // to every objective except the pure L2 control, which already pins the
    // scores to the baseline values.
    if cfg.geometry.weight > 0.0 && cfg.objective != Objective::L2 && n >= 2 {
        let vs = variance(scores).max(1e-12);
        let vb = variance(baseline).max(1e-12);
        let r = (vs / vb).ln();
        loss += cfg.geometry.weight * r * r;
        // d/ds_j [ (ln(vs/vb))^2 ] = 2*ln(vs/vb) * (1/vs) * dvs/ds_j
        let nf = n as f32;
        let mean = scores.iter().sum::<f32>() / nf;
        let c = cfg.geometry.weight * 2.0 * r / vs * (2.0 / nf);
        for j in 0..n {
            w[j] += c * (scores[j] - mean);
        }
    }

    loss
}

/// Train a projection with the selected ranking-aware objective, warm-started
/// from `init_p` (normally the PCA solution).
///
/// Returns the trained projection and the training history. The projection
/// shape and the caller's contract are identical to
/// [`crate::learned::train_projection`], so the deployable path is unchanged.
pub fn train_ranking(rows: &[Row<'_>], init_p: Vec<f32>, cfg: &TrainConfig) -> (Vec<f32>, History) {
    assert!(
        init_p.len().is_multiple_of(D_C),
        "projection must be D_C×d (row-major)"
    );
    let d = init_p.len() / D_C;
    let mut p = init_p;
    if rows.is_empty() {
        return (p, History::default());
    }
    let mut rng = Rng::new(cfg.seed);
    let mut hist = History::default();

    let mut a = vec![0.0f32; D_C];
    let mut bcols: Vec<f32> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();
    let mut w: Vec<f32> = Vec::new();
    let mut grad = vec![0.0f32; D_C * d];

    let steps = (rows.len() / cfg.batch).max(1);
    for ep in 0..cfg.epochs {
        let cur_lr = cfg.lr * (1.0 - ep as f32 / cfg.epochs as f32);
        let mut epoch_loss = 0.0f64;
        for _ in 0..steps {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
            let mut batch_loss = 0.0f32;
            for _ in 0..cfg.batch {
                let row = &rows[(rng.next_u64() as usize) % rows.len()];
                debug_assert_eq!(row.d, d);
                let n = row.n_visible.min(cfg.max_keys);
                if n == 0 {
                    continue;
                }
                // a = P q
                for r in 0..D_C {
                    let pr = &p[r * d..(r + 1) * d];
                    let mut s = 0.0f32;
                    for m in 0..d {
                        s += pr[m] * row.q[m];
                    }
                    a[r] = s;
                }
                // b_j = P k_j, and s_j = <a, b_j>
                bcols.clear();
                bcols.resize(n * D_C, 0.0);
                scores.clear();
                scores.resize(n, 0.0);
                for j in 0..n {
                    let kj = row.key(j);
                    let mut sj = 0.0f32;
                    for r in 0..D_C {
                        let pr = &p[r * d..(r + 1) * d];
                        let mut s = 0.0f32;
                        for m in 0..d {
                            s += pr[m] * kj[m];
                        }
                        bcols[j * D_C + r] = s;
                        sj += a[r] * s;
                    }
                    scores[j] = sj;
                }

                batch_loss += row_weights(
                    cfg,
                    &row.baseline[..n],
                    &scores,
                    &mut rng,
                    &mut w,
                    &mut hist.pairwise_comparisons,
                );
                hist.rows_seen += 1;

                // grad[r][m] += w_j * (q[m] * b_j[r] + k_j[m] * a[r])
                for j in 0..n {
                    let wj = w[j];
                    if wj == 0.0 {
                        continue;
                    }
                    let kj = row.key(j);
                    for r in 0..D_C {
                        let br = bcols[j * D_C + r];
                        let ar = a[r];
                        let gr = &mut grad[r * d..(r + 1) * d];
                        for m in 0..d {
                            gr[m] += wj * (row.q[m] * br + kj[m] * ar);
                        }
                    }
                }
            }
            let scale = cur_lr / cfg.batch as f32;
            for (pi, gi) in p.iter_mut().zip(grad.iter()) {
                *pi -= scale * gi;
            }
            epoch_loss += batch_loss as f64 / cfg.batch as f64;
        }
        hist.epoch_loss.push((epoch_loss / steps as f64) as f32);
    }
    (p, hist)
}
