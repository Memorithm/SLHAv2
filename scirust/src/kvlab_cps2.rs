//! Destination-owned verification of the frozen KVLab CPS-2 development CSV.
//!
//! This is deliberately not a general CSV reader or a runtime selection API.
//! It checks the two N=5, D=2 fixtures from KVLab #175 with an independent scalar
//! oracle. A matching source-revision string is a provenance assertion, not an
//! authenticated attestation. Successful verification never promotes a policy.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

pub const KVLAB_REVISION: &str = "265a2a65120b8f6f06c3cbb6604cf821ceda94d0";
pub const FLAT_REVISION: &str = "ad1634fc922f6223dd3a83ac84154a82b1a35562";
pub const CPS2_SCHEMA: &str = "kvlab.cps2-compact-quality/v1";
pub const MAX_CSV_BYTES: usize = 65_536;
pub const CSV_HEADER: &str = concat!(
    "schema,flat_source_revision,case,arm,row,eligible_keys,selected_ids,",
    "reference_top_ids,top_k_hits,top_k_recall,retained_softmax_mass,",
    "omitted_softmax_mass,selected_density,selector_score_components,",
    "numerical_pairs_executed,output_max_abs_error,lse_abs_error,",
    "timing_measured,physical_traffic_measured,model_quality_measured,",
    "promotion_authorized"
);

// The producer prints probabilities with 12 decimals and errors with 9 decimals.
// Error diagnostics are differences of f32 FLAT outputs; the independent oracle
// below uses f64. These tolerances apply only to the frozen bounded fixtures.
const MASS_TOLERANCE: f64 = 1.0e-12;
const OUTPUT_TOLERANCE: f64 = 4.0e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DevelopmentCase {
    AlignedCoordinate,
    OmittedDominantCoordinate,
}

impl DevelopmentCase {
    pub const fn label(self) -> &'static str {
        match self {
            Self::AlignedCoordinate => "aligned_coordinate",
            Self::OmittedDominantCoordinate => "omitted_dominant_coordinate",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cps2Arm {
    AllAccept,
    CompactProjected,
    FullScoreTopK,
    RecentTail,
    MatchedRandom,
}

impl Cps2Arm {
    pub const fn label(self) -> &'static str {
        match self {
            Self::AllAccept => "all_accept",
            Self::CompactProjected => "compact_projected",
            Self::FullScoreTopK => "full_score_topk",
            Self::RecentTail => "recent_tail",
            Self::MatchedRandom => "matched_random",
        }
    }
}

/// Read-only diagnostics. These values carry no runtime admission authority.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentRow {
    pub case: DevelopmentCase,
    pub arm: Cps2Arm,
    pub query_row: usize,
    pub selected_keys: Vec<usize>,
    pub reference_top_keys: Vec<usize>,
    pub top_k_hits: usize,
    pub top_k_recall: f64,
    pub retained_softmax_mass: f64,
    pub omitted_softmax_mass: f64,
    pub selected_density: f64,
    pub selector_score_components: usize,
    pub numerical_pairs_executed: usize,
    pub output_max_abs_error: f64,
    pub lse_abs_error: f64,
}

/// Complete, internally verified synthetic panel, not authenticated evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentEvidence {
    rows: Vec<DevelopmentRow>,
}

impl DevelopmentEvidence {
    pub fn rows(&self) -> &[DevelopmentRow] {
        &self.rows
    }

    pub const fn asserted_kvlab_revision(&self) -> &'static str {
        KVLAB_REVISION
    }

    pub const fn permits_runtime_promotion(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cps2EvidenceError {
    pub line: usize,
    pub field: &'static str,
}

impl fmt::Display for Cps2EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid CPS-2 {} at line {}", self.field, self.line)
    }
}

impl std::error::Error for Cps2EvidenceError {}

fn require(ok: bool, line: usize, field: &'static str) -> Result<(), Cps2EvidenceError> {
    if ok {
        Ok(())
    } else {
        Err(Cps2EvidenceError { line, field })
    }
}

fn integer(text: &str, line: usize, field: &'static str) -> Result<usize, Cps2EvidenceError> {
    require(
        !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()),
        line,
        field,
    )?;
    text.parse().map_err(|_| Cps2EvidenceError { line, field })
}

fn number(text: &str, line: usize, field: &'static str) -> Result<f64, Cps2EvidenceError> {
    let value: f64 = text.parse().map_err(|_| Cps2EvidenceError { line, field })?;
    require(value.is_finite() && value >= 0.0, line, field)?;
    Ok(value)
}

fn key_ids(text: &str, line: usize, field: &'static str) -> Result<Vec<usize>, Cps2EvidenceError> {
    let keys = text
        .split('|')
        .map(|token| integer(token, line, field))
        .collect::<Result<Vec<_>, _>>()?;
    require(
        keys.len() <= 5 && keys.iter().all(|&key| key < 5),
        line,
        field,
    )?;
    require(keys.windows(2).all(|pair| pair[0] < pair[1]), line, field)?;
    Ok(keys)
}

fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn reference_keys(case: DevelopmentCase) -> Vec<usize> {
    match case {
        DevelopmentCase::AlignedCoordinate => vec![3, 4],
        DevelopmentCase::OmittedDominantCoordinate => vec![0, 4],
    }
}

fn expected_selection(case: DevelopmentCase, arm: Cps2Arm, row: usize) -> Vec<usize> {
    match arm {
        Cps2Arm::AllAccept => (0..5).collect(),
        Cps2Arm::FullScoreTopK => reference_keys(case),
        Cps2Arm::CompactProjected | Cps2Arm::RecentTail => vec![3, 4],
        Cps2Arm::MatchedRandom => {
            let seed = splitmix64(0x4350_5332 ^ row as u64);
            let mut ranked = (0..5_usize)
                .map(|key| (splitmix64(seed ^ key as u64), key))
                .collect::<Vec<_>>();
            ranked.sort_unstable();
            let mut selected = ranked[..2].iter().map(|&(_, key)| key).collect::<Vec<_>>();
            selected.sort_unstable();
            selected
        }
    }
}

// Independent closed fixture oracle, not a copy of FLAT's attention kernel.
// V_j = [j, 4-j], so both output-coordinate errors have the same magnitude.
fn reference_statistics(case: DevelopmentCase, selected: &[usize]) -> (f64, f64, f64) {
    let scores = std::array::from_fn::<_, 5, _>(|key| {
        if case == DevelopmentCase::OmittedDominantCoordinate && key == 0 {
            20.0
        } else {
            key as f64
        }
    });
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores.map(|score| (score - maximum).exp());
    let total = weights.iter().sum::<f64>();
    let kept = selected.iter().map(|&key| weights[key]).sum::<f64>();
    let dense_mean = weights
        .iter()
        .enumerate()
        .map(|(key, weight)| key as f64 * weight)
        .sum::<f64>()
        / total;
    let selected_mean = selected
        .iter()
        .map(|&key| key as f64 * weights[key])
        .sum::<f64>()
        / kept;
    let mass = kept / total;
    (mass, (dense_mean - selected_mean).abs(), -mass.ln())
}

fn parse_row(text: &str, line: usize) -> Result<DevelopmentRow, Cps2EvidenceError> {
    let fields = text.split(',').collect::<Vec<_>>();
    require(fields.len() == 21, line, "column count")?;
    require(fields[0] == CPS2_SCHEMA, line, "schema")?;
    require(fields[1] == FLAT_REVISION, line, "FLAT revision")?;
    let case = match fields[2] {
        "aligned_coordinate" => DevelopmentCase::AlignedCoordinate,
        "omitted_dominant_coordinate" => DevelopmentCase::OmittedDominantCoordinate,
        _ => return Err(Cps2EvidenceError { line, field: "case" }),
    };
    let arm = match fields[3] {
        "all_accept" => Cps2Arm::AllAccept,
        "compact_projected" => Cps2Arm::CompactProjected,
        "full_score_topk" => Cps2Arm::FullScoreTopK,
        "recent_tail" => Cps2Arm::RecentTail,
        "matched_random" => Cps2Arm::MatchedRandom,
        _ => return Err(Cps2EvidenceError { line, field: "arm" }),
    };
    let query_row = integer(fields[4], line, "query row")?;
    require(query_row < 5, line, "query row")?;
    require(integer(fields[5], line, "eligible keys")? == 5, line, "eligible keys")?;
    let selected = key_ids(fields[6], line, "selected IDs")?;
    let reference = key_ids(fields[7], line, "reference IDs")?;
    require(selected == expected_selection(case, arm, query_row), line, "selection")?;
    require(reference == reference_keys(case), line, "reference IDs")?;
    let hits = integer(fields[8], line, "top-k hits")?;
    let expected_hits = reference.iter().filter(|key| selected.contains(key)).count();
    require(hits == expected_hits, line, "top-k hits")?;
    let recall = number(fields[9], line, "top-k recall")?;
    require(recall == hits as f64 / 2.0, line, "top-k recall")?;
    let mass = number(fields[10], line, "retained mass")?;
    let omitted = number(fields[11], line, "omitted mass")?;
    let density = number(fields[12], line, "density")?;
    require(mass <= 1.0 && omitted <= 1.0 && density <= 1.0, line, "probability range")?;
    let (expected_mass, expected_error, expected_lse) = reference_statistics(case, &selected);
    require(close(mass, expected_mass, MASS_TOLERANCE), line, "retained mass")?;
    require(close(omitted, 1.0 - expected_mass, MASS_TOLERANCE), line, "omitted mass")?;
    require(close(density, selected.len() as f64 / 5.0, MASS_TOLERANCE), line, "density")?;
    let components = integer(fields[13], line, "selector components")?;
    let expected_components = match arm {
        Cps2Arm::CompactProjected => 5,
        Cps2Arm::FullScoreTopK => 10,
        _ => 0,
    };
    require(components == expected_components, line, "selector components")?;
    let pairs = integer(fields[14], line, "numerical pairs")?;
    require(pairs == selected.len(), line, "numerical pairs")?;
    let error = number(fields[15], line, "output error")?;
    let lse = number(fields[16], line, "LSE error")?;
    if arm == Cps2Arm::AllAccept {
        require(error == 0.0 && lse == 0.0, line, "all-accept parity")?;
    } else {
        require(close(error, expected_error, OUTPUT_TOLERANCE), line, "output error")?;
        require(close(lse, expected_lse, OUTPUT_TOLERANCE), line, "LSE error")?;
    }
    require(fields[17..].iter().all(|value| *value == "false"), line, "claim flags")?;
    Ok(DevelopmentRow {
        case,
        arm,
        query_row,
        selected_keys: selected,
        reference_top_keys: reference,
        top_k_hits: hits,
        top_k_recall: recall,
        retained_softmax_mass: mass,
        omitted_softmax_mass: omitted,
        selected_density: density,
        selector_score_components: components,
        numerical_pairs_executed: pairs,
        output_max_abs_error: error,
        lse_abs_error: lse,
    })
}

/// Verify exactly the frozen two-fixture, five-arm, five-row development panel.
///
/// The revision argument is asserted provenance, not proof of execution origin.
/// The caller must retain the original CSV and its execution provenance. This
/// function neither evaluates a real model nor authorizes runtime activation.
pub fn verify_development_csv(
    text: &str,
    asserted_kvlab_revision: &str,
) -> Result<DevelopmentEvidence, Cps2EvidenceError> {
    require(asserted_kvlab_revision == KVLAB_REVISION, 0, "KVLab revision")?;
    require(text.len() <= MAX_CSV_BYTES, 0, "input size")?;
    let mut lines = text.lines();
    require(lines.next() == Some(CSV_HEADER), 1, "header")?;
    let mut rows = Vec::with_capacity(50);
    let mut seen = BTreeSet::new();
    for (index, text) in lines.enumerate() {
        let line = index + 2;
        require(rows.len() < 50, line, "panel size")?;
        let row = parse_row(text, line)?;
        require(seen.insert((row.case, row.arm, row.query_row)), line, "duplicate row")?;
        rows.push(row);
    }
    // The key universe has exactly 2 * 5 * 5 entries. Uniqueness, bounds and
    // this cardinality together establish complete coverage of every arm/row.
    require(rows.len() == 50, 0, "incomplete panel")?;
    Ok(DevelopmentEvidence { rows })
}
