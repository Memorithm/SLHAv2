//! Self-audit: runs the system's internal invariants and produces a structured,
//! machine-readable report ([`Json`]) plus a human Markdown rendering and a
//! report-vs-report [`diff`]. Reused by the `slha-audit` binary and the
//! `slha-mcp` server, so an LLM/agent and a human read the *same* facts.
//!
//! Every number is computed at runtime from the real kernel — tile layout, a
//! live SIMD-vs-scalar equivalence check, platform features/caches, output
//! fidelity vs full attention, the CCOS budget invariant, and determinism.

use crate::attention::slha_v2::{
    SciRustSlhaTile, D_C, D_S, FLAG_WARM, LATENT_BYTES, RESIDUAL_WORDS,
};
use crate::ccos::ElasticKvCache;
use crate::json::{obj, Json};
use crate::metrics::{cosine, dot, softmax_into};
use crate::rng::Rng;
use crate::scenario::{build_tile, generate, Projection, D_K};
use std::fmt::Write as _;

/// Run the full self-audit and return the structured report.
pub fn run() -> Json {
    let mut checks: Vec<(String, bool)> = Vec::new();
    let tile = tile_section(&mut checks);
    let simd = simd_section(&mut checks);
    let platform = platform_section();
    let fidelity = fidelity_section(&mut checks);
    let ccos = ccos_section(&mut checks);
    let determinism = determinism_section(&mut checks);

    let passed = checks.iter().filter(|(_, p)| *p).count();
    let failed = checks.len() - passed;
    let checks_json = Json::Arr(
        checks
            .iter()
            .map(|(name, p)| {
                obj(vec![
                    ("check", Json::str(name.clone())),
                    ("pass", Json::Bool(*p)),
                ])
            })
            .collect(),
    );

    obj(vec![
        ("tool", Json::str("slha-audit")),
        ("crate_version", Json::str(env!("CARGO_PKG_VERSION"))),
        (
            "verdict",
            obj(vec![
                ("checks", Json::Num(checks.len() as f64)),
                ("passed", Json::Num(passed as f64)),
                ("failed", Json::Num(failed as f64)),
                ("ok", Json::Bool(failed == 0)),
            ]),
        ),
        ("checks", checks_json),
        ("tile", tile),
        ("simd", simd),
        ("platform", platform),
        ("fidelity", fidelity),
        ("ccos", ccos),
        ("determinism", determinism),
    ])
}

// ── sections ─────────────────────────────────────────────────────────────────

fn tile_section(checks: &mut Vec<(String, bool)>) -> Json {
    let size = std::mem::size_of::<SciRustSlhaTile>();
    let align = std::mem::align_of::<SciRustSlhaTile>();
    // Field byte budget: latent 64 + residual 32 + 5×f32/u32 (20) + 2×u16 (4) + 8.
    let field_bytes = LATENT_BYTES + RESIDUAL_WORDS * 8 + 20 + 4 + 8;
    let zero_pad = size == field_bytes;
    let cache_line_128 = cfg!(cache_line_128);

    checks.push(("tile.size_is_128".into(), size == 128));
    checks.push(("tile.zero_padding".into(), zero_pad));
    checks.push((
        "tile.align_is_cache_line".into(),
        align == 64 || align == 128,
    ));

    obj(vec![
        ("size_of", Json::Num(size as f64)),
        ("align_of", Json::Num(align as f64)),
        ("field_bytes", Json::Num(field_bytes as f64)),
        ("zero_padding", Json::Bool(zero_pad)),
        ("cache_line_128_cfg", Json::Bool(cache_line_128)),
        ("d_c", Json::Num(D_C as f64)),
        ("d_s", Json::Num(D_S as f64)),
    ])
}

fn simd_section(checks: &mut Vec<(String, bool)>) -> Json {
    let proj = Projection::new(0x000A_0D17);
    let (q, toks) = generate(0x000A_0D17, 256, 0.3);
    let qs = proj.sign_bits(&q);
    let mut max_rel = 0.0f32;
    for (i, tok) in toks.iter().enumerate() {
        let tile = build_tile(&proj, tok, i as u32, false);
        let disp = tile.compute_score(&q, &qs); // runtime-dispatched (SIMD where available)
        let scal = tile.compute_score_scalar(&q, &qs);
        max_rel = max_rel.max((disp - scal).abs() / (1.0 + scal.abs()));
    }
    let pass = max_rel <= 1.0e-3;
    checks.push(("simd.scalar_equivalence".into(), pass));

    obj(vec![
        ("arch", Json::str(std::env::consts::ARCH)),
        ("dispatched_path", Json::str(dispatched_path(false))),
        (
            "features",
            obj(cpu_features()
                .into_iter()
                .map(|(k, v)| (k, Json::Bool(v)))
                .collect()),
        ),
        ("scalar_equivalence_max_rel_err", Json::Num(max_rel as f64)),
        ("scalar_equivalence_pass", Json::Bool(pass)),
        ("tiles_checked", Json::Num(256.0)),
    ])
}

fn platform_section() -> Json {
    let levels = cache_levels();
    let l1d_line = levels
        .iter()
        .find(|(lvl, typ, line, _)| lvl == "1" && typ == "Data" && *line > 0)
        .map(|(_, _, l, _)| *l);
    let size = std::mem::size_of::<SciRustSlhaTile>();
    let ratio = match l1d_line {
        Some(l) if l > 0 => Json::Num(size.div_ceil(l) as f64),
        _ => Json::Null,
    };

    obj(vec![
        ("arch", Json::str(std::env::consts::ARCH)),
        ("os", Json::str(std::env::consts::OS)),
        (
            "features",
            obj(cpu_features()
                .into_iter()
                .map(|(k, v)| (k, Json::Bool(v)))
                .collect()),
        ),
        (
            "cache_levels",
            Json::Arr(
                levels
                    .iter()
                    .map(|(lvl, typ, line, sz)| {
                        obj(vec![
                            ("level", Json::str(lvl.clone())),
                            ("type", Json::str(typ.clone())),
                            ("line_bytes", Json::Num(*line as f64)),
                            ("size", Json::str(sz.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("tile_l1d_line_span", ratio),
    ])
}

fn fidelity_section(checks: &mut Vec<(String, bool)>) -> Json {
    let proj = Projection::new(0x000F_1DE1);
    let n = 256usize;
    let dv = 64usize;
    let scale = 1.0 / (D_K as f32).sqrt();
    let (q, toks) = generate(0x000F_1DE1, n, 0.3);
    let qs = proj.sign_bits(&q);
    let tiles: Vec<SciRustSlhaTile> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();

    let mut rng = Rng::new(7);
    let values: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut v = vec![0.0f32; dv];
            rng.fill_gaussian(&mut v);
            v
        })
        .collect();

    let s_true: Vec<f32> = toks.iter().map(|t| dot(&q, &t.k_real)).collect();
    let s_hot: Vec<f32> = tiles.iter().map(|t| t.compute_score(&q, &qs)).collect();
    let s_warm: Vec<f32> = tiles
        .iter()
        .map(|t| {
            let mut w = *t;
            w.flags |= FLAG_WARM;
            w.compute_score(&q, &qs)
        })
        .collect();

    let out_true = attn_out(&s_true, &values, scale, dv);
    let hot_cos = cosine(&out_true, &attn_out(&s_hot, &values, scale, dv));
    let warm_cos = cosine(&out_true, &attn_out(&s_warm, &values, scale, dv));
    let hot_ge_warm = hot_cos >= warm_cos - 1.0e-3;
    checks.push(("fidelity.hot_ge_warm".into(), hot_ge_warm));

    obj(vec![
        ("context_tokens", Json::Num(n as f64)),
        ("residual_rho", Json::Num(0.3)),
        ("hot_output_cosine", Json::Num(hot_cos as f64)),
        ("warm_output_cosine", Json::Num(warm_cos as f64)),
        ("hot_ge_warm", Json::Bool(hot_ge_warm)),
    ])
}

fn ccos_section(checks: &mut Vec<(String, bool)>) -> Json {
    let proj = Projection::new(0x000C_0501);
    let n = 1024usize;
    let (q, toks) = generate(0x000C_0501, n, 0.3);
    let qs = proj.sign_bits(&q);
    let tiles: Vec<SciRustSlhaTile> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();

    let naive = n * 128;
    let budget = n * 112; // between WARM(96) and HOT(128) totals → pages, never evicts
    let mut cache = ElasticKvCache::with_budget(budget);
    for (i, t) in tiles.iter().enumerate() {
        cache.insert(*t);
        if i % 256 == 0 {
            cache.enforce_budget();
        }
    }
    cache.enforce_budget();
    let (hot, warm, cold) = cache.counts();
    let live = cache.live_bytes();
    let within = live <= budget;
    checks.push(("ccos.live_bytes_within_budget".into(), within));

    let s_ref: Vec<f32> = tiles.iter().map(|t| t.compute_score(&q, &qs)).collect();
    let s_cache: Vec<f32> = (0..n).map(|i| cache.score(i, &q, &qs)).collect();
    let cache_cos = cosine(&s_ref, &s_cache);

    obj(vec![
        ("context_tiles", Json::Num(n as f64)),
        ("naive_bytes", Json::Num(naive as f64)),
        ("budget_bytes", Json::Num(budget as f64)),
        ("live_bytes", Json::Num(live as f64)),
        ("within_budget", Json::Bool(within)),
        ("hot", Json::Num(hot as f64)),
        ("warm", Json::Num(warm as f64)),
        ("cold", Json::Num(cold as f64)),
        (
            "footprint_pct_of_naive",
            Json::Num(100.0 * live as f64 / naive as f64),
        ),
        ("score_cosine_vs_all_hot", Json::Num(cache_cos as f64)),
    ])
}

fn determinism_section(checks: &mut Vec<(String, bool)>) -> Json {
    let sample = || -> Vec<f32> {
        let proj = Projection::new(0x0000_0DE7);
        let (q, toks) = generate(0x0000_0DE7, 128, 0.3);
        let qs = proj.sign_bits(&q);
        toks.iter()
            .enumerate()
            .map(|(i, t)| build_tile(&proj, t, i as u32, false).compute_score(&q, &qs))
            .collect()
    };
    let a = sample();
    let b = sample();
    let repeatable = a == b;
    checks.push(("determinism.repeatable".into(), repeatable));

    obj(vec![
        ("samples", Json::Num(a.len() as f64)),
        ("bitwise_repeatable", Json::Bool(repeatable)),
    ])
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn attn_out(scores: &[f32], values: &[Vec<f32>], scale: f32, dv: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; scores.len()];
    softmax_into(scores, scale, &mut w);
    let mut o = vec![0.0f32; dv];
    for (wi, vi) in w.iter().zip(values) {
        for j in 0..dv {
            o[j] += wi * vi[j];
        }
    }
    o
}

fn cpu_features() -> Vec<(&'static str, bool)> {
    #[cfg(target_arch = "x86_64")]
    {
        vec![
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
            ("avx512vl", std::is_x86_feature_detected!("avx512vl")),
            (
                "avx512vpopcntdq",
                std::is_x86_feature_detected!("avx512vpopcntdq"),
            ),
        ]
    }
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
            ("sve", std::arch::is_aarch64_feature_detected!("sve")),
            ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
        ]
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Vec::new()
    }
}

fn dispatched_path(is_nf4: bool) -> &'static str {
    if is_nf4 {
        return "scalar (NF4)";
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            "avx512"
        } else if std::is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "scalar"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}

/// CPU0 cache levels via Linux sysfs: `(level, type, line_bytes, size)`. Empty
/// on platforms without that sysfs tree (the report simply omits cache data).
fn cache_levels() -> Vec<(String, String, usize, String)> {
    let mut out = Vec::new();
    for idx in 0..8 {
        let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let level = match std::fs::read_to_string(format!("{dir}/level")) {
            Ok(s) => s.trim().to_string(),
            Err(_) => break,
        };
        let typ = std::fs::read_to_string(format!("{dir}/type"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let line = std::fs::read_to_string(format!("{dir}/coherency_line_size"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let size = std::fs::read_to_string(format!("{dir}/size"))
            .unwrap_or_default()
            .trim()
            .to_string();
        out.push((level, typ, line, size));
    }
    out
}

// ── rendering & diff ─────────────────────────────────────────────────────────

/// Render a report as human-readable Markdown.
pub fn to_markdown(r: &Json) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# SLHA v2 — Audit Report\n");
    let version = r
        .get("crate_version")
        .and_then(|x| x.as_str())
        .unwrap_or("?");
    let _ = writeln!(s, "- **Tool:** `slha-audit` (scirust v{version})");
    if let Some(v) = r.get("verdict") {
        let ok = v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false);
        let _ = writeln!(
            s,
            "- **Verdict:** {} — {} checks, {} passed, {} failed",
            if ok { "✅ OK" } else { "❌ FAIL" },
            field(v, "checks"),
            field(v, "passed"),
            field(v, "failed"),
        );
    }
    if let Some(Json::Arr(cs)) = r.get("checks") {
        let _ = writeln!(s, "\n## Checks\n\n| Check | Result |\n|---|:--:|");
        for c in cs {
            let name = c.get("check").and_then(|x| x.as_str()).unwrap_or("?");
            let pass = c.get("pass").and_then(|x| x.as_bool()).unwrap_or(false);
            let _ = writeln!(s, "| `{}` | {} |", name, if pass { "✅" } else { "❌" });
        }
    }
    for sec in [
        "tile",
        "simd",
        "fidelity",
        "ccos",
        "determinism",
        "platform",
    ] {
        if let Some(section) = r.get(sec) {
            let _ = writeln!(s, "\n## {sec}\n");
            render_obj_table(&mut s, section);
        }
    }
    s
}

fn field(j: &Json, k: &str) -> String {
    j.get(k)
        .map(|v| v.to_compact())
        .unwrap_or_else(|| "?".into())
}

fn render_obj_table(s: &mut String, j: &Json) {
    if let Json::Obj(m) = j {
        let _ = writeln!(s, "| Field | Value |\n|---|---|");
        for (k, v) in m {
            let val = match v {
                Json::Obj(_) | Json::Arr(_) => format!("`{}`", v.to_compact()),
                Json::Str(x) => x.clone(),
                _ => v.to_compact(),
            };
            let _ = writeln!(s, "| {k} | {val} |");
        }
    }
}

/// Compare two reports field-by-field, returning one human line per change.
/// Stable across runs (every number is seeded), so a non-empty diff is a real
/// regression — the basis for `slha-audit --diff PRIOR.json`.
pub fn diff(prior: &Json, current: &Json) -> Vec<String> {
    let mut out = Vec::new();
    diff_walk("", prior, current, &mut out);
    out
}

fn diff_walk(path: &str, a: &Json, b: &Json, out: &mut Vec<String>) {
    let join = |k: &str| {
        if path.is_empty() {
            k.to_string()
        } else {
            format!("{path}.{k}")
        }
    };
    match (a, b) {
        (Json::Obj(am), Json::Obj(bm)) => {
            for (k, av) in am {
                match bm.iter().find(|(bk, _)| bk == k) {
                    Some((_, bv)) => diff_walk(&join(k), av, bv, out),
                    None => out.push(format!("- {}: removed ({})", join(k), av.to_compact())),
                }
            }
            for (k, bv) in bm {
                if !am.iter().any(|(ak, _)| ak == k) {
                    out.push(format!("+ {}: added ({})", join(k), bv.to_compact()));
                }
            }
        }
        (Json::Arr(_), Json::Arr(_)) => {
            if a != b {
                out.push(format!("~ {path}: array changed"));
            }
        }
        _ => {
            if a != b {
                out.push(format!(
                    "~ {path}: {} -> {}",
                    a.to_compact(),
                    b.to_compact()
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_runs_and_all_checks_pass() {
        let r = run();
        let ok = r
            .get("verdict")
            .and_then(|v| v.get("ok"))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        assert!(ok, "audit verdict not ok:\n{}", r.to_pretty());
    }

    #[test]
    fn report_roundtrips_through_json() {
        let r = run();
        let parsed = Json::parse(&r.to_compact()).expect("report parses");
        assert_eq!(
            parsed.get("tool").and_then(|x| x.as_str()),
            Some("slha-audit")
        );
        // The structured tree survives a serialize→parse round-trip.
        assert_eq!(parsed, r);
    }

    #[test]
    fn markdown_mentions_verdict() {
        let md = to_markdown(&run());
        assert!(md.contains("Audit Report"));
        assert!(md.contains("Verdict"));
    }

    #[test]
    fn self_diff_is_empty_mutated_diff_is_not() {
        let r = run();
        assert!(diff(&r, &r).is_empty(), "self-diff must be empty");
        let mutated =
            Json::parse(&r.to_compact().replacen("\"ok\":true", "\"ok\":false", 1)).unwrap();
        let d = diff(&r, &mutated);
        assert!(
            d.iter().any(|l| l.contains("ok")),
            "diff should flag ok: {d:?}"
        );
    }
}
