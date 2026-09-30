use scirust::kvlab_cps2::{
    verify_development_csv, Cps2Arm, DevelopmentCase, CSV_HEADER, KVLAB_REVISION, MAX_CSV_BYTES,
};

// Captured by the bounded RemoteOps run from the pinned, unmodified KVLab
// producer. Provenance and the byte-identical second run are recorded in docs.
const CSV: &str = include_str!("fixtures/kvlab_cps2_v1.csv");

fn replace_field(case: &str, arm: &str, query: &str, field: &str, replacement: &str) -> String {
    let column = CSV_HEADER
        .split(',')
        .position(|name| name == field)
        .unwrap();
    let mut found = false;
    let lines = CSV
        .lines()
        .map(|line| {
            let mut fields = line.split(',').map(str::to_owned).collect::<Vec<_>>();
            if fields[2] == case && fields[3] == arm && fields[4] == query {
                fields[column] = replacement.to_owned();
                found = true;
            }
            fields.join(",")
        })
        .collect::<Vec<_>>();
    assert!(found, "test target must exist");
    format!("{}\n", lines.join("\n"))
}

fn mutate(field: &str, value: &str) -> String {
    replace_field(
        "omitted_dominant_coordinate",
        "compact_projected",
        "4",
        field,
        value,
    )
}

#[test]
fn accepts_complete_pinned_panel_without_runtime_authority() {
    let evidence = verify_development_csv(CSV, KVLAB_REVISION).unwrap();
    assert_eq!(evidence.rows().len(), 50);
    assert_eq!(evidence.asserted_kvlab_revision(), KVLAB_REVISION);
    assert!(!evidence.permits_runtime_promotion());
}

#[test]
fn all_accept_preserves_dense_identity() {
    let evidence = verify_development_csv(CSV, KVLAB_REVISION).unwrap();
    let rows = evidence
        .rows()
        .iter()
        .filter(|row| row.arm == Cps2Arm::AllAccept);
    for row in rows {
        assert_eq!(row.retained_softmax_mass, 1.0);
        assert_eq!(row.output_max_abs_error, 0.0);
        assert_eq!(row.lse_abs_error, 0.0);
        assert_eq!(row.numerical_pairs_executed, 5);
    }
}

#[test]
fn negative_control_is_accepted_as_negative_not_as_quality_success() {
    let evidence = verify_development_csv(CSV, KVLAB_REVISION).unwrap();
    let rows = evidence.rows().iter().filter(|row| {
        row.case == DevelopmentCase::OmittedDominantCoordinate
            && row.arm == Cps2Arm::CompactProjected
    });
    let mut count = 0;
    for row in rows {
        assert_eq!(row.selected_keys, vec![3, 4]);
        assert_eq!(row.reference_top_keys, vec![0, 4]);
        assert_eq!(row.top_k_recall, 0.5);
        assert!(row.retained_softmax_mass < 2.0e-7);
        assert!(row.output_max_abs_error > 3.7);
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn rejects_source_and_schema_drift() {
    assert!(verify_development_csv(CSV, "main").is_err());
    assert!(verify_development_csv(
        &mutate("schema", "kvlab.cps2-compact-quality/v2"),
        KVLAB_REVISION
    )
    .is_err());
    assert!(
        verify_development_csv(&mutate("flat_source_revision", "main"), KVLAB_REVISION).is_err()
    );
}

#[test]
fn rejects_all_unsupported_claim_flags() {
    for field in [
        "timing_measured",
        "physical_traffic_measured",
        "model_quality_measured",
        "promotion_authorized",
    ] {
        for value in ["true", "False", "", "0"] {
            assert!(verify_development_csv(&mutate(field, value), KVLAB_REVISION).is_err());
        }
    }
}

#[test]
fn rejects_missing_duplicate_and_extra_rows() {
    let mut lines = CSV.lines().collect::<Vec<_>>();
    lines.pop();
    assert!(verify_development_csv(&lines.join("\n"), KVLAB_REVISION).is_err());
    lines.push(lines[1]);
    assert!(verify_development_csv(&lines.join("\n"), KVLAB_REVISION).is_err());
    let extra = format!("{}{}\n", CSV, CSV.lines().nth(1).unwrap());
    assert!(verify_development_csv(&extra, KVLAB_REVISION).is_err());
}

#[test]
fn rejects_case_arm_and_geometry_drift() {
    for (field, value) in [
        ("case", "real_model"),
        ("arm", "new_policy"),
        ("row", "5"),
        ("row", "+4"),
        ("eligible_keys", "6"),
    ] {
        assert!(verify_development_csv(&mutate(field, value), KVLAB_REVISION).is_err());
    }
}

#[test]
fn rejects_corrupt_or_reordered_original_ids() {
    for field in ["selected_ids", "reference_top_ids"] {
        for value in ["4|3", "3|3", "3|5", "", "3", "3|4|4", "-1|4", "3|+4"] {
            assert!(verify_development_csv(&mutate(field, value), KVLAB_REVISION).is_err());
        }
    }
}

#[test]
fn rejects_forged_metrics_and_accounting() {
    for (field, value) in [
        ("top_k_hits", "2"),
        ("top_k_recall", "1.0"),
        ("retained_softmax_mass", "1.0"),
        ("omitted_softmax_mass", "0.0"),
        ("selected_density", "1.0"),
        ("selector_score_components", "0"),
        ("numerical_pairs_executed", "1"),
        ("output_max_abs_error", "0.0"),
        ("lse_abs_error", "0.0"),
    ] {
        assert!(verify_development_csv(&mutate(field, value), KVLAB_REVISION).is_err());
    }
}

#[test]
fn rejects_nonfinite_negative_and_out_of_range_diagnostics() {
    for field in [
        "top_k_recall",
        "retained_softmax_mass",
        "omitted_softmax_mass",
        "selected_density",
        "output_max_abs_error",
        "lse_abs_error",
    ] {
        for value in ["NaN", "inf", "-inf", "-0.1", "1e999"] {
            assert!(verify_development_csv(&mutate(field, value), KVLAB_REVISION).is_err());
        }
    }
    assert!(
        verify_development_csv(&mutate("retained_softmax_mass", "1.01"), KVLAB_REVISION).is_err()
    );
}

#[test]
fn checks_row_scoped_random_control_not_just_density() {
    let changed = replace_field(
        "aligned_coordinate",
        "matched_random",
        "0",
        "selected_ids",
        "3|4",
    );
    assert!(verify_development_csv(&changed, KVLAB_REVISION).is_err());
    let evidence = verify_development_csv(CSV, KVLAB_REVISION).unwrap();
    let expected = [vec![0, 2], vec![1, 2], vec![0, 3], vec![0, 3], vec![0, 4]];
    for row in evidence
        .rows()
        .iter()
        .filter(|row| row.arm == Cps2Arm::MatchedRandom)
    {
        assert_eq!(row.selected_keys, expected[row.query_row]);
    }
}

#[test]
fn checks_exact_all_accept_identity_even_within_numeric_tolerance() {
    let changed = replace_field(
        "aligned_coordinate",
        "all_accept",
        "0",
        "output_max_abs_error",
        "0.000000001",
    );
    assert!(verify_development_csv(&changed, KVLAB_REVISION).is_err());
}

#[test]
fn bounds_input_and_rejects_unknown_columns_and_blank_rows() {
    assert!(verify_development_csv(&"x".repeat(MAX_CSV_BYTES + 1), KVLAB_REVISION).is_err());
    assert!(
        verify_development_csv(&CSV.replace(CSV_HEADER, "wrong_header"), KVLAB_REVISION).is_err()
    );
    let extra_column = mutate("promotion_authorized", "false,extra");
    assert!(verify_development_csv(&extra_column, KVLAB_REVISION).is_err());
    assert!(verify_development_csv(&CSV.replacen('\n', "\n\n", 1), KVLAB_REVISION).is_err());
}

#[test]
fn accepts_crlf_transport_without_changing_row_diagnostics() {
    let unix = verify_development_csv(CSV, KVLAB_REVISION).unwrap();
    let windows = verify_development_csv(&CSV.replace('\n', "\r\n"), KVLAB_REVISION).unwrap();
    assert_eq!(unix, windows);
}
