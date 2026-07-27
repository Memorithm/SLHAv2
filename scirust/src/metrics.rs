//! Small statistics helpers for validating approximation quality.
//! No external dependencies.
//!
//! Pairwise metrics require equal-length inputs and panic with an explicit
//! message when their dimensions differ. Empty but dimensionally valid inputs
//! retain the historical finite-zero behavior.
//!
//! Non-finite observations are not silently converted into plausible quality
//! measurements: scalar metrics return `NaN`, ranks return `NaN` entries, and
//! softmax clears its output to an all-zero invalid distribution.

use std::collections::HashSet;

#[inline]
fn assert_same_len(name: &str, left: usize, right: usize) {
    assert_eq!(
        left, right,
        "{name} requires equal-length inputs: left={left}, right={right}"
    );
}

#[inline]
fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Plain dot product — the full-precision ground-truth attention score.
///
/// Accumulation uses `f64` to avoid avoidable intermediate overflow and
/// cancellation. The returned value remains `f32`; a true result outside the
/// representable `f32` range therefore becomes an infinity.
///
/// Returns `NaN` when either input contains a non-finite value.
///
/// # Panics
///
/// Panics when the input lengths differ.
///
/// ```
/// use scirust::metrics::dot;
/// assert!((dot(&[1.0, 2.0, 3.0], &[1.0, 0.0, -1.0]) + 2.0).abs() < 1e-6);
/// ```
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len("dot", a.len(), b.len());

    if !all_finite(a) || !all_finite(b) {
        return f32::NAN;
    }

    a.iter()
        .zip(b)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum::<f64>() as f32
}

/// Pearson correlation coefficient.
///
/// Constant or empty vectors return `0.0`. A non-finite observation returns
/// `NaN`, preventing invalid data from being accepted by a quality threshold.
///
/// # Panics
///
/// Panics when the input lengths differ.
pub fn pearson(x: &[f32], y: &[f32]) -> f32 {
    assert_same_len("pearson", x.len(), y.len());

    if x.is_empty() {
        return 0.0;
    }

    if !all_finite(x) || !all_finite(y) {
        return f32::NAN;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().map(|&value| value as f64).sum::<f64>() / n;
    let mean_y = y.iter().map(|&value| value as f64).sum::<f64>() / n;

    let (mut covariance, mut variance_x, mut variance_y) = (0.0f64, 0.0f64, 0.0f64);

    for (&left, &right) in x.iter().zip(y) {
        let dx = left as f64 - mean_x;
        let dy = right as f64 - mean_y;

        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    if variance_x <= 0.0 || variance_y <= 0.0 {
        return 0.0;
    }

    let denominator = (variance_x * variance_y).sqrt();

    if !denominator.is_finite() || denominator <= 0.0 {
        return f32::NAN;
    }

    (covariance / denominator).clamp(-1.0, 1.0) as f32
}

/// Fractional zero-based ranks.
///
/// Equal values receive their average rank. Signed zeroes are treated as an
/// equality. If any observation is non-finite, every returned rank is `NaN`
/// so that downstream correlations fail closed.
pub fn ranks(values: &[f32]) -> Vec<f32> {
    if !all_finite(values) {
        return vec![f32::NAN; values.len()];
    }

    let mut indices: Vec<usize> = (0..values.len()).collect();

    indices.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });

    let mut result = vec![0.0f32; values.len()];
    let mut start = 0usize;

    while start < indices.len() {
        let mut end = start + 1;

        while end < indices.len() && values[indices[end]] == values[indices[start]] {
            end += 1;
        }

        let average_rank = (start + end - 1) as f32 / 2.0;

        for &index in &indices[start..end] {
            result[index] = average_rank;
        }

        start = end;
    }

    result
}

/// Spearman rank correlation.
///
/// Equal observations use fractional average ranks rather than an arbitrary
/// index-dependent ordering.
///
/// # Panics
///
/// Panics when the input lengths differ.
pub fn spearman(x: &[f32], y: &[f32]) -> f32 {
    assert_same_len("spearman", x.len(), y.len());
    pearson(&ranks(x), &ranks(y))
}

/// Fraction of the top-`k` indices shared between two score vectors.
///
/// Equal scores are ordered deterministically by their original index.
/// `k == 0` returns `0.0`. Non-finite observations return `NaN`.
///
/// # Panics
///
/// Panics when the vector lengths differ or when `k` exceeds their length.
pub fn topk_overlap(truth: &[f32], approx: &[f32], k: usize) -> f32 {
    assert_same_len("topk_overlap", truth.len(), approx.len());

    if k == 0 {
        return 0.0;
    }

    assert!(
        k <= truth.len(),
        "topk_overlap requires k <= vector length: k={k}, len={}",
        truth.len()
    );

    if !all_finite(truth) || !all_finite(approx) {
        return f32::NAN;
    }

    let top = |values: &[f32]| -> HashSet<usize> {
        let mut indices: Vec<usize> = (0..values.len()).collect();

        indices.sort_by(|&left, &right| {
            values[right]
                .total_cmp(&values[left])
                .then_with(|| left.cmp(&right))
        });

        indices.truncate(k);
        indices.into_iter().collect()
    };

    let truth_top = top(truth);
    let approx_top = top(approx);

    truth_top.intersection(&approx_top).count() as f32 / k as f32
}

/// Root-mean-square value of a slice.
///
/// Accumulation uses `f64`, so the RMS of finite `f32` inputs remains finite
/// whenever its mathematically correct result fits in `f32`. Empty input
/// returns `0.0`; non-finite input returns `NaN`.
#[inline]
pub fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    if !all_finite(values) {
        return f32::NAN;
    }

    let mean_square = values
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        / values.len() as f64;

    mean_square.sqrt() as f32
}

/// Cosine similarity between two vectors.
///
/// Empty vectors and zero-norm vectors return `0.0`. Non-finite observations
/// return `NaN`.
///
/// # Panics
///
/// Panics when the input lengths differ.
///
/// ```
/// use scirust::metrics::cosine;
/// assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
/// assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
/// ```
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len("cosine", a.len(), b.len());

    if a.is_empty() {
        return 0.0;
    }

    if !all_finite(a) || !all_finite(b) {
        return f32::NAN;
    }

    let (mut product, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);

    for (&left, &right) in a.iter().zip(b) {
        let left = left as f64;
        let right = right as f64;

        product += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }

    let denominator = (norm_a * norm_b).sqrt();

    if !denominator.is_finite() || denominator <= 0.0 {
        return f32::NAN;
    }

    (product / denominator).clamp(-1.0, 1.0) as f32
}

/// Relative L2 error `||reference - approx|| / ||reference||`.
///
/// Equal zero vectors return `0.0`. A zero reference with a non-zero
/// approximation returns positive infinity, because the relative error is
/// mathematically unbounded. Non-finite observations return `NaN`.
///
/// # Panics
///
/// Panics when the input lengths differ.
pub fn rel_l2(reference: &[f32], approx: &[f32]) -> f32 {
    assert_same_len("rel_l2", reference.len(), approx.len());

    if reference.is_empty() {
        return 0.0;
    }

    if !all_finite(reference) || !all_finite(approx) {
        return f32::NAN;
    }

    let (mut squared_error, mut squared_reference) = (0.0f64, 0.0f64);

    for (&expected, &observed) in reference.iter().zip(approx) {
        let expected = expected as f64;
        let observed = observed as f64;
        let difference = expected - observed;

        squared_error += difference * difference;
        squared_reference += expected * expected;
    }

    if squared_reference == 0.0 {
        return if squared_error == 0.0 {
            0.0
        } else {
            f32::INFINITY
        };
    }

    (squared_error / squared_reference).sqrt() as f32
}

/// Numerically stable softmax of `scores * scale` into `out`.
///
/// `-∞` logits are treated as masked entries. If every logit is masked, if a
/// score is `NaN`, or if `scale` is non-finite, the output is cleared to zero.
/// This deliberately avoids inventing probability mass for invalid tokens.
///
/// Positive-infinite logits share the complete probability mass equally.
///
/// # Panics
///
/// Panics when `scores` and `out` have different lengths.
///
/// ```
/// use scirust::metrics::softmax_into;
/// let mut weights = [0.0f32; 3];
/// softmax_into(&[0.0, 0.0, 0.0], 1.0, &mut weights);
/// assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// assert!((weights[0] - 1.0 / 3.0).abs() < 1e-6);
/// ```
pub fn softmax_into(scores: &[f32], scale: f32, out: &mut [f32]) {
    assert_same_len("softmax_into", scores.len(), out.len());

    if out.is_empty() {
        return;
    }

    if !scale.is_finite() {
        out.fill(0.0);
        return;
    }

    let scale = scale as f64;
    let mut maximum = f64::NEG_INFINITY;
    let mut positive_infinities = 0usize;

    for &score in scores {
        if score.is_nan() {
            out.fill(0.0);
            return;
        }

        let logit = score as f64 * scale;

        if logit.is_nan() {
            out.fill(0.0);
            return;
        }

        if logit.is_infinite() && logit.is_sign_positive() {
            positive_infinities += 1;
        }

        maximum = maximum.max(logit);
    }

    if positive_infinities > 0 {
        let probability = 1.0 / positive_infinities as f32;

        for (output, &score) in out.iter_mut().zip(scores) {
            let logit = score as f64 * scale;

            *output = if logit.is_infinite() && logit.is_sign_positive() {
                probability
            } else {
                0.0
            };
        }

        return;
    }

    if maximum.is_infinite() && maximum.is_sign_negative() {
        out.fill(0.0);
        return;
    }

    let mut sum = 0.0f64;

    for (output, &score) in out.iter_mut().zip(scores) {
        let logit = score as f64 * scale;
        let exponential = (logit - maximum).exp();

        *output = exponential as f32;
        sum += exponential;
    }

    if !sum.is_finite() || sum <= 0.0 {
        out.fill(0.0);
        return;
    }

    for output in out.iter_mut() {
        *output = (*output as f64 / sum) as f32;
    }

    // Compensate the small rounding error introduced by the f64 -> f32 casts.
    let rounded_sum = out.iter().map(|&value| value as f64).sum::<f64>();

    if rounded_sum.is_finite() && rounded_sum > 0.0 {
        for output in out.iter_mut() {
            *output = (*output as f64 / rounded_sum) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn degenerate_inputs_have_explicit_finite_contracts() {
        let empty: &[f32] = &[];

        assert_eq!(rms(empty), 0.0);
        assert_eq!(dot(empty, empty), 0.0);
        assert_eq!(pearson(empty, empty), 0.0);
        assert_eq!(spearman(empty, empty), 0.0);
        assert_eq!(cosine(empty, empty), 0.0);
        assert_eq!(rel_l2(empty, empty), 0.0);
        assert_eq!(topk_overlap(empty, empty, 0), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(rel_l2(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert!(rel_l2(&[0.0, 0.0], &[1.0, 0.0]).is_infinite());

        let mut empty_output: [f32; 0] = [];
        softmax_into(empty, 1.0, &mut empty_output);

        let mut masked = [f32::NAN; 4];
        softmax_into(&[f32::NEG_INFINITY; 4], 1.0, &mut masked);
        assert_eq!(masked, [0.0; 4]);
    }

    #[test]
    fn pairwise_dimension_mismatches_are_rejected_in_release_too() {
        assert!(catch_unwind(|| dot(&[1.0], &[])).is_err());
        assert!(catch_unwind(|| pearson(&[1.0], &[])).is_err());
        assert!(catch_unwind(|| spearman(&[1.0], &[])).is_err());
        assert!(catch_unwind(|| topk_overlap(&[1.0], &[], 0)).is_err());
        assert!(catch_unwind(|| cosine(&[1.0], &[])).is_err());
        assert!(catch_unwind(|| rel_l2(&[1.0], &[])).is_err());

        let mut output = [0.0f32; 1];

        assert!(catch_unwind(AssertUnwindSafe(|| {
            softmax_into(&[1.0, 2.0], 1.0, &mut output);
        }))
        .is_err());
    }

    #[test]
    fn non_finite_observations_fail_closed() {
        let nan = f32::NAN;
        let infinity = f32::INFINITY;

        assert!(dot(&[nan], &[1.0]).is_nan());
        assert!(pearson(&[1.0, nan], &[1.0, 2.0]).is_nan());
        assert!(spearman(&[1.0, nan], &[1.0, 2.0]).is_nan());
        assert!(topk_overlap(&[infinity], &[1.0], 1).is_nan());
        assert!(rms(&[infinity]).is_nan());
        assert!(cosine(&[1.0, nan], &[1.0, 2.0]).is_nan());
        assert!(rel_l2(&[1.0], &[infinity]).is_nan());
        assert!(ranks(&[1.0, nan]).iter().all(|rank| rank.is_nan()));

        let mut output = [1.0f32; 2];

        softmax_into(&[0.0, nan], 1.0, &mut output);
        assert_eq!(output, [0.0, 0.0]);

        softmax_into(&[0.0, 1.0], f32::INFINITY, &mut output);
        assert_eq!(output, [0.0, 0.0]);
    }

    #[test]
    fn fractional_ranks_average_ties_deterministically() {
        assert_eq!(ranks(&[2.0, 1.0, 2.0, 1.0]), [2.5, 0.5, 2.5, 0.5]);
        assert_eq!(ranks(&[0.0, -0.0]), [0.5, 0.5]);

        let left = [10.0, 10.0, 5.0, 5.0];
        let right = [3.0, 3.0, 1.0, 1.0];

        assert!((spearman(&left, &right) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn topk_contract_is_deterministic_and_bounded() {
        let truth = [3.0, 3.0, 2.0, 1.0];
        let same = [3.0, 3.0, 2.0, 1.0];
        let shifted_tie = [3.0, 2.0, 3.0, 1.0];

        assert_eq!(topk_overlap(&truth, &same, 2), 1.0);
        assert_eq!(topk_overlap(&truth, &shifted_tie, 2), 0.5);

        assert!(catch_unwind(|| topk_overlap(&truth, &same, truth.len() + 1)).is_err());
    }

    #[test]
    fn softmax_handles_masks_infinities_and_zero_scale() {
        let mut output = [0.0f32; 4];

        softmax_into(
            &[1.0, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY],
            1.0,
            &mut output,
        );
        assert_eq!(output, [0.0, 0.5, 0.5, 0.0]);

        softmax_into(
            &[0.0, f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            1.0,
            &mut output,
        );
        assert!(output[0] > 0.0);
        assert_eq!(output[1], 0.0);
        assert!(output[2] > output[0]);
        assert_eq!(output[3], 0.0);
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        softmax_into(&[1.0, 2.0, 3.0, 4.0], 0.0, &mut output);
        assert!(output.iter().all(|&value| (value - 0.25).abs() < 1e-6));
    }

    #[test]
    fn large_finite_values_do_not_overflow_normalized_metrics() {
        let maximum = f32::MAX;
        let values = [maximum, maximum];

        assert!(rms(&values).is_finite());
        assert!((rms(&values) / maximum - 1.0).abs() < 1e-6);
        assert!((cosine(&values, &values) - 1.0).abs() < 1e-6);
        assert_eq!(rel_l2(&values, &values), 0.0);

        let centered = [maximum, 0.0, -maximum];
        assert!((pearson(&centered, &centered) - 1.0).abs() < 1e-6);
    }
}
