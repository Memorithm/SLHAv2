//! Inspect the frozen CPS-2 CSV without changing SLHA runtime state.

use std::fs::File;
use std::io::Read;

use scirust::kvlab_cps2::{verify_development_csv, Cps2Arm, DevelopmentCase, MAX_CSV_BYTES};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("usage: cps2_inspect <csv> <KVLab revision>")?;
    let revision = args.next().ok_or("missing exact KVLab producer revision")?;
    let revision = revision.to_str().ok_or("producer revision is not UTF-8")?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let mut csv = String::new();
    File::open(path)?
        .take((MAX_CSV_BYTES + 1) as u64)
        .read_to_string(&mut csv)?;
    let evidence = verify_development_csv(&csv, revision)?;
    println!("schema=slha.cps2-development-import/v1");
    println!("asserted_kvlab_revision={}", evidence.asserted_kvlab_revision());
    println!("rows={}", evidence.rows().len());
    println!("scope=unsigned_synthetic_diagnostic");
    println!("runtime_promotion={}", evidence.permits_runtime_promotion());
    println!("case,arm,rows,mean_top_k_recall,mean_retained_mass,max_output_error");
    for case in [
        DevelopmentCase::AlignedCoordinate,
        DevelopmentCase::OmittedDominantCoordinate,
    ] {
        for arm in [
            Cps2Arm::AllAccept,
            Cps2Arm::CompactProjected,
            Cps2Arm::FullScoreTopK,
            Cps2Arm::RecentTail,
            Cps2Arm::MatchedRandom,
        ] {
            let rows = evidence
                .rows()
                .iter()
                .filter(|row| row.case == case && row.arm == arm)
                .collect::<Vec<_>>();
            let count = rows.len() as f64;
            let recall = rows.iter().map(|row| row.top_k_recall).sum::<f64>() / count;
            let mass = rows.iter().map(|row| row.retained_softmax_mass).sum::<f64>() / count;
            let error = rows.iter().map(|row| row.output_max_abs_error).fold(0.0_f64, f64::max);
            println!(
                "{},{},{},{:.12},{:.12},{:.9}",
                case.label(),
                arm.label(),
                rows.len(),
                recall,
                mass,
                error
            );
        }
    }
    Ok(())
}
