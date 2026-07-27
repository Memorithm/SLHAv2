//! `slha-audit` — run the SLHA v2 self-audit and emit a report.
//!
//! ```text
//! slha-audit              # human-readable Markdown to stdout
//! slha-audit --json       # compact JSON (machine-readable / CI / agents)
//! slha-audit --pretty     # indented JSON
//! slha-audit --out FILE   # also write the rendered report to FILE
//! slha-audit --diff PRIOR # run now, diff against a prior JSON report
//! slha-audit --perf       # add a `perf` section with real cache-miss counters
//!                         #   (needs `perf`; falls back to `perf: null` otherwise)
//! ```
//!
//! Exit code: `0` if the audit verdict is ok (and, with `--diff`, no changes);
//! `1` if a check failed or a regression was found; `2` on I/O / parse errors.

use scirust::attention::slha_v2::SciRustSlhaTile;
use scirust::audit;
use scirust::json::{obj, Json};
use scirust::scenario::{build_tile, generate, Projection};
use std::fmt::Write as _;
use std::path::Path;
use std::process::{exit, Command};

const HELP: &str = "\
slha-audit — SLHA v2 self-audit + report

USAGE:
    slha-audit [--json | --pretty] [--out FILE] [--diff PRIOR.json] [--perf]

OPTIONS:
    --json         Emit compact JSON instead of Markdown
    --pretty       Emit indented JSON
    --out FILE     Also write the rendered report to FILE
    --diff PRIOR   Run now and diff against a prior JSON report (regression check)
    --perf         Add a `perf` section: cache-miss counters from `perf stat` over
                   a representative kernel micro-benchmark (or `null` if perf is
                   unavailable / `perf_event_paranoid` too restrictive)
    -h, --help     Show this help

EXIT CODE:
    0  audit ok (no regressions);  1  a check failed / report changed;  2  I/O error
";

// `perf stat` event sets (full, then a minimal portable fallback) + bench size.
const PERF_EVENTS: &str = "cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses";
const PERF_EVENTS_MIN: &str = "cache-references,cache-misses";
const BENCH_TILES: usize = 8192;
const BENCH_ITERS: usize = 2000;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Internal mode: run the representative micro-benchmark and exit. Invoked as
    // a child by the --perf path so `perf stat` measures just the hot kernel.
    if args.iter().any(|a| a == "--bench-internal") {
        run_bench();
        return;
    }

    let has = |f: &str| args.iter().any(|a| a == f);
    let val = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    if has("-h") || has("--help") {
        print!("{HELP}");
        return;
    }

    let mut report = audit::run();
    let ok = report
        .get("verdict")
        .and_then(|v| v.get("ok"))
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    // --perf: append an OPTIONAL `perf` section (additive — existing reports stay
    // backward-compatible). `perf: null` when perf is unavailable / restricted.
    if has("--perf") {
        let perf = perf_section();
        if let Json::Obj(ref mut m) = report {
            m.push(("perf".to_string(), perf));
        }
    }

    let markdown = !has("--json") && !has("--pretty");
    let mut rendered = if has("--json") {
        report.to_compact()
    } else if has("--pretty") {
        report.to_pretty()
    } else {
        audit::to_markdown(&report)
    };
    // to_markdown (in the lib) only knows the core sections; render perf here so
    // the library stays untouched.
    if markdown && has("--perf") {
        rendered.push_str(&perf_markdown(report.get("perf")));
    }

    if let Some(path) = val("--out") {
        if let Err(e) = std::fs::write(&path, &rendered) {
            eprintln!("slha-audit: cannot write {path}: {e}");
            exit(2);
        }
        eprintln!("slha-audit: wrote {path}");
    }

    // --diff: compare against a prior JSON report and report changes.
    if let Some(prior_path) = val("--diff") {
        let txt = match std::fs::read_to_string(&prior_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("slha-audit: cannot read {prior_path}: {e}");
                exit(2);
            }
        };
        let prior = match Json::parse(&txt) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("slha-audit: cannot parse {prior_path}: {e}");
                exit(2);
            }
        };
        let changes = audit::diff(&prior, &report);
        if changes.is_empty() {
            println!("slha-audit: no changes vs {prior_path}");
        } else {
            println!("slha-audit: {} change(s) vs {prior_path}:", changes.len());
            for line in &changes {
                println!("  {line}");
            }
        }
        exit(if ok && changes.is_empty() { 0 } else { 1 });
    }

    print!("{rendered}");
    if !rendered.ends_with('\n') {
        println!();
    }
    exit(if ok { 0 } else { 1 });
}

/// Representative hot-path micro-benchmark: score a ~1 MB working set of
/// `BENCH_TILES` HOT tiles `BENCH_ITERS` times (exercises L1/L2/LLC). Run as a
/// child under `perf stat` by [`perf_section`].
fn run_bench() {
    let proj = Projection::new(0xB00B5);
    let (q, toks) = generate(0xB00B5, BENCH_TILES, 0.3);
    let qs = proj.sign_bits(&q);
    let tiles: Vec<SciRustSlhaTile> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();
    let mut acc = 0.0f32;
    for _ in 0..BENCH_ITERS {
        for t in &tiles {
            acc += t.compute_score(&q, &qs);
        }
    }
    std::hint::black_box(acc);
}

/// Run the micro-benchmark under `perf stat` and return a `perf` JSON section,
/// or [`Json::Null`] if perf is unavailable / restricted on this host.
fn perf_section() -> Json {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return Json::Null,
    };
    // Try the full event set, then a minimal portable one (events vary by uarch).
    let events = match run_perf(PERF_EVENTS, &exe).or_else(|| run_perf(PERF_EVENTS_MIN, &exe)) {
        Some(e) if !e.is_empty() => e,
        _ => return Json::Null,
    };

    let counters = obj(events
        .iter()
        .map(|(k, v)| (k.as_str(), Json::Num(*v)))
        .collect());
    let refs = events
        .iter()
        .find(|(k, _)| k == "cache-references")
        .map(|(_, v)| *v);
    let miss = events
        .iter()
        .find(|(k, _)| k == "cache-misses")
        .map(|(_, v)| *v);

    let mut top = vec![
        ("available", Json::Bool(true)),
        ("tiles", Json::Num(BENCH_TILES as f64)),
        ("iterations", Json::Num(BENCH_ITERS as f64)),
        ("counters", counters),
    ];
    if let (Some(r), Some(m)) = (refs, miss) {
        if r > 0.0 {
            top.push(("cache_miss_rate", Json::Num(m / r)));
        }
    }
    obj(top)
}

/// Spawn `perf stat -x , -e <events> <self> --bench-internal` and parse counters.
/// `None` if `perf` is missing or exits non-zero (e.g. paranoid restrictions).
fn run_perf(events: &str, exe: &Path) -> Option<Vec<(String, f64)>> {
    let output = Command::new("perf")
        .args(["stat", "-x", ",", "-e", events])
        .arg(exe)
        .arg("--bench-internal")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_perf_csv(&String::from_utf8_lossy(&output.stderr)))
}

/// Parse `perf stat -x ,` CSV (on stderr). Each line is `value,unit,event,...`;
/// `<not supported>` / `<not counted>` rows yield no numeric value and are dropped.
fn parse_perf_csv(s: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for line in s.lines() {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 3 {
            continue;
        }
        let event = f[2].trim();
        if event.is_empty() {
            continue;
        }
        if let Ok(v) = f[0].trim().parse::<f64>() {
            out.push((event.to_string(), v));
        }
    }
    out
}

/// Render the `perf` section as a small Markdown block (the lib renderer doesn't
/// know about it).
fn perf_markdown(perf: Option<&Json>) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\n## perf\n");
    match perf {
        Some(p @ Json::Obj(_)) => {
            let _ = writeln!(s, "| Compteur | Valeur |\n|---|---|");
            if let Some(Json::Obj(c)) = p.get("counters") {
                for (k, v) in c {
                    let _ = writeln!(s, "| `{}` | {} |", k, v.to_compact());
                }
            }
            if let Some(r) = p.get("cache_miss_rate") {
                let _ = writeln!(s, "| **cache_miss_rate** | {} |", r.to_compact());
            }
        }
        _ => {
            let _ = writeln!(
                s,
                "_perf indisponible (binaire `perf` absent ou `perf_event_paranoid` trop restrictif)._"
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_perf_csv_skipping_unsupported() {
        let sample = "\
1234567,,cache-references,1000000,100.00,,
89012,,cache-misses,1000000,100.00,,
<not supported>,,L1-dcache-load-misses,0,0.00,,
4567,,LLC-load-misses,1000000,100.00,,";
        let p = parse_perf_csv(sample);
        assert_eq!(p.len(), 3); // the <not supported> row is dropped
        let get = |name: &str| p.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
        assert_eq!(get("cache-references"), Some(1234567.0));
        assert_eq!(get("cache-misses"), Some(89012.0));
        assert_eq!(get("LLC-load-misses"), Some(4567.0));
        assert_eq!(get("L1-dcache-load-misses"), None);
    }

    #[test]
    fn perf_field_keeps_report_valid_and_backward_compatible() {
        // Adding an optional `perf` field must not break the existing report.
        let mut report = audit::run();
        if let Json::Obj(ref mut m) = report {
            m.push(("perf".to_string(), Json::Null));
        }
        let parsed = Json::parse(&report.to_compact()).expect("report still valid JSON");
        assert_eq!(
            parsed.get("tool").and_then(|t| t.as_str()),
            Some("slha-audit")
        );
        assert!(parsed.get("perf").is_some());
        // Core sections untouched.
        assert!(parsed.get("verdict").is_some());
    }

    #[test]
    fn perf_markdown_handles_unavailable() {
        let md = perf_markdown(Some(&Json::Null));
        assert!(md.contains("## perf"));
        assert!(md.contains("indisponible"));
    }
}
