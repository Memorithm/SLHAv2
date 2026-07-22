//! `slha-mcp` — a Model Context Protocol (MCP) server over stdio for SLHA v2.
//!
//! Zero external dependencies: JSON-RPC framing and (de)serialization reuse
//! [`scirust::json`]. It speaks **newline-delimited JSON-RPC 2.0** on
//! stdin/stdout (the MCP stdio transport) and exposes the SLHA v2 self-audit and
//! kernel as agent-callable tools.
//!
//! Wire it into an MCP client — e.g. Claude Code:
//! ```text
//! claude mcp add slha -- cargo run -q -p slha-mcp
//! ```
//! or build it (`cargo build --release -p slha-mcp`) and point the client at the
//! binary `target/release/slha-mcp`.

use scirust::attention::slha_v2::{
    quantize_latent, quantize_latent_grouped, quantize_latent_mixed, quantize_latent_nf4,
    quantize_latent_tq3, FLAG_HOT, FLAG_MIXED, FLAG_NF4, FLAG_TQ3, N_GROUPS,
};
use scirust::json::{obj, Json};
use scirust::metrics::dot;
use scirust::scenario::{build_tile, generate, ContextToken, Projection};
use std::io::{self, BufRead, Write};

/// Latest stable MCP revision supported by this server.
const PROTOCOL_VERSION: &str = "2025-11-25";

/// Revisions whose stdio/tool subset is supported by this implementation.
///
/// When the requested version appears here, it is echoed during negotiation.
/// Otherwise the server proposes [`PROTOCOL_VERSION`], allowing the client to
/// disconnect if it cannot support that revision.
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 4] =
    ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const SERVER_NAME: &str = "slha-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const DIMS: usize = 128; // D_C

/// Maximum JSON-RPC frame size, excluding the newline delimiter.
///
/// SLHA tools need only a few kilobytes for two 128-dimensional vectors, so
/// 256 KiB leaves substantial headroom without permitting unbounded allocation.
const MAX_FRAME_BYTES: usize = 256 * 1024;

/// Largest numeric request identifier that is exactly representable by f64.
const MAX_SAFE_JSON_INTEGER: f64 = 9_007_199_254_740_991.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Lifecycle {
    /// No successful `initialize` request has been processed.
    #[default]
    Uninitialized,

    /// The initialize response was sent; the server is waiting for
    /// `notifications/initialized`.
    AwaitingInitialized,

    /// Normal MCP operations are permitted.
    Ready,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameRead {
    Eof,
    Ready,
    TooLarge,
}

/// Read one newline-delimited stdio frame without allowing unbounded growth.
///
/// Oversized frames are completely discarded through their newline delimiter,
/// so the next call starts at the next JSON-RPC message.
fn read_frame<R: BufRead>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    max_bytes: usize,
) -> io::Result<FrameRead> {
    buffer.clear();

    let mut saw_input = false;
    let mut too_large = false;

    loop {
        let available = reader.fill_buf()?;

        if available.is_empty() {
            if !saw_input {
                return Ok(FrameRead::Eof);
            }

            break;
        }

        saw_input = true;

        let newline = available.iter().position(|&byte| byte == b'\n');
        let payload_len = newline.unwrap_or(available.len());
        let consumed = payload_len + usize::from(newline.is_some());

        if !too_large {
            if payload_len > max_bytes.saturating_sub(buffer.len()) {
                too_large = true;
                buffer.clear();
            } else {
                buffer.extend_from_slice(&available[..payload_len]);
            }
        }

        reader.consume(consumed);

        if newline.is_some() {
            break;
        }
    }

    if too_large {
        Ok(FrameRead::TooLarge)
    } else {
        Ok(FrameRead::Ready)
    }
}

fn write_response<W: Write>(out: &mut W, response: &Json) -> io::Result<()> {
    writeln!(out, "{}", response.to_compact())?;
    out.flush()
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    let mut input = stdin.lock();
    let mut out = stdout.lock();
    let mut frame = Vec::with_capacity(4096);
    let mut lifecycle = Lifecycle::Uninitialized;

    loop {
        let response = match read_frame(&mut input, &mut frame, MAX_FRAME_BYTES) {
            Ok(FrameRead::Eof) => break,
            Ok(FrameRead::TooLarge) => Some(err_response(
                Json::Null,
                -32700,
                &format!("parse error: JSON-RPC frame exceeds {MAX_FRAME_BYTES} bytes"),
            )),
            Ok(FrameRead::Ready) => {
                let text = match std::str::from_utf8(&frame) {
                    Ok(text) => text.trim(),
                    Err(error) => {
                        let response = err_response(
                            Json::Null,
                            -32700,
                            &format!("parse error: invalid UTF-8: {error}"),
                        );

                        if write_response(&mut out, &response).is_err() {
                            break;
                        }

                        continue;
                    }
                };

                if text.is_empty() {
                    continue;
                }

                match Json::parse_with_limit(text, MAX_FRAME_BYTES) {
                    Ok(request) => handle(&request, &mut lifecycle),
                    Err(error) => Some(err_response(
                        Json::Null,
                        -32700,
                        &format!("parse error: {error}"),
                    )),
                }
            }
            Err(error) => {
                eprintln!("slha-mcp: stdin read error: {error}");
                break;
            }
        };

        if let Some(response) = response {
            if write_response(&mut out, &response).is_err() {
                break;
            }
        }
    }
}

fn valid_request_id(id: &Json) -> bool {
    match id {
        Json::Str(_) => true,
        Json::Num(number) => {
            number.is_finite() && number.fract() == 0.0 && number.abs() <= MAX_SAFE_JSON_INTEGER
        }
        _ => false,
    }
}

fn negotiate_protocol_version(requested: &str) -> &'static str {
    SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .copied()
        .find(|supported| *supported == requested)
        .unwrap_or(PROTOCOL_VERSION)
}

fn handle_initialize(id: Json, params: Option<&Json>, lifecycle: &mut Lifecycle) -> Json {
    if *lifecycle != Lifecycle::Uninitialized {
        return err_response(id, -32600, "server is already initialized");
    }

    let params = match params {
        Some(Json::Obj(_)) => params.expect("matched Some"),
        _ => {
            return err_response(id, -32602, "initialize params must be an object");
        }
    };

    let requested = match params.get("protocolVersion").and_then(Json::as_str) {
        Some(version) if !version.is_empty() => version,
        _ => {
            return err_response(
                id,
                -32602,
                "initialize requires a non-empty protocolVersion",
            );
        }
    };

    if !matches!(params.get("capabilities"), Some(Json::Obj(_))) {
        return err_response(id, -32602, "initialize requires a capabilities object");
    }

    let client_info = match params.get("clientInfo") {
        Some(Json::Obj(_)) => params.get("clientInfo").expect("matched Some"),
        _ => {
            return err_response(id, -32602, "initialize requires a clientInfo object");
        }
    };

    if client_info
        .get("name")
        .and_then(Json::as_str)
        .is_none_or(str::is_empty)
    {
        return err_response(id, -32602, "clientInfo.name must be a non-empty string");
    }

    if client_info
        .get("version")
        .and_then(Json::as_str)
        .is_none_or(str::is_empty)
    {
        return err_response(id, -32602, "clientInfo.version must be a non-empty string");
    }

    let negotiated = negotiate_protocol_version(requested);
    *lifecycle = Lifecycle::AwaitingInitialized;

    ok_response(
        id,
        obj(vec![
            ("protocolVersion", Json::str(negotiated)),
            (
                "capabilities",
                obj(vec![(
                    "tools",
                    obj(vec![("listChanged", Json::Bool(false))]),
                )]),
            ),
            (
                "serverInfo",
                obj(vec![
                    ("name", Json::str(SERVER_NAME)),
                    ("title", Json::str("SLHA v2 MCP Server")),
                    ("version", Json::str(SERVER_VERSION)),
                    (
                        "description",
                        Json::str("Deterministic SLHA v2 compression, scoring and audit tools"),
                    ),
                ]),
            ),
            (
                "instructions",
                Json::str(
                    "Use slha.audit for invariants, slha.compress for latent \
                     codecs, slha.score for score comparison and slha.benchmark \
                     for local throughput.",
                ),
            ),
        ]),
    )
}

/// Handle one JSON-RPC message.
///
/// Valid notifications return `None`. Malformed objects are not treated as
/// notifications merely because they lack an `id`.
fn handle(req: &Json, lifecycle: &mut Lifecycle) -> Option<Json> {
    if !matches!(req, Json::Obj(_)) {
        return Some(err_response(
            Json::Null,
            -32600,
            "invalid request: top-level value must be an object",
        ));
    }

    if req.get("jsonrpc").and_then(Json::as_str) != Some("2.0") {
        return Some(err_response(
            Json::Null,
            -32600,
            "invalid request: jsonrpc must equal \"2.0\"",
        ));
    }

    let method = match req.get("method").and_then(Json::as_str) {
        Some(method) if !method.is_empty() => method,
        _ => {
            return Some(err_response(
                Json::Null,
                -32600,
                "invalid request: method must be a non-empty string",
            ));
        }
    };

    let id_opt = req.get("id").cloned();

    if let Some(id) = id_opt.as_ref() {
        if !valid_request_id(id) {
            return Some(err_response(
                Json::Null,
                -32600,
                "invalid request: id must be a string or safe integer",
            ));
        }
    }

    let is_notification = id_opt.is_none();
    let id = id_opt.unwrap_or(Json::Null);

    if let Some(params) = req.get("params") {
        if !matches!(params, Json::Obj(_)) {
            if is_notification {
                return None;
            }

            return Some(err_response(id, -32602, "params must be an object"));
        }
    }

    match method {
        "initialize" => {
            if is_notification {
                // MCP requires initialize to be a request, but JSON-RPC
                // notifications must never receive a response.
                None
            } else {
                Some(handle_initialize(id, req.get("params"), lifecycle))
            }
        }

        "notifications/initialized" => {
            if !is_notification {
                return Some(err_response(
                    id,
                    -32600,
                    "notifications/initialized must not contain an id",
                ));
            }

            if *lifecycle == Lifecycle::AwaitingInitialized {
                *lifecycle = Lifecycle::Ready;
            }

            None
        }

        "ping" => {
            if is_notification {
                None
            } else {
                Some(ok_response(id, obj(vec![])))
            }
        }

        _ if *lifecycle != Lifecycle::Ready => {
            if is_notification {
                None
            } else {
                Some(err_response(id, -32002, "server is not initialized"))
            }
        }

        "tools/list" => {
            if is_notification {
                None
            } else {
                Some(ok_response(id, obj(vec![("tools", tool_definitions())])))
            }
        }

        "tools/call" => {
            if is_notification {
                None
            } else {
                Some(handle_tool_call(id, req.get("params")))
            }
        }

        _ if is_notification => None,

        other => Some(err_response(
            id,
            -32601,
            &format!("method not found: {other}"),
        )),
    }
}

// ── JSON-RPC helpers ─────────────────────────────────────────────────────────

fn ok_response(id: Json, result: Json) -> Json {
    obj(vec![
        ("jsonrpc", Json::str("2.0")),
        ("id", id),
        ("result", result),
    ])
}

fn err_response(id: Json, code: i64, message: &str) -> Json {
    obj(vec![
        ("jsonrpc", Json::str("2.0")),
        ("id", id),
        (
            "error",
            obj(vec![
                ("code", Json::Num(code as f64)),
                ("message", Json::str(message)),
            ]),
        ),
    ])
}

/// A successful `tools/call` result wrapping a single text content block.
/// (MCP reports *tool* failures via `isError: true`, not a JSON-RPC error.)
fn tool_result(text: String, is_error: bool) -> Json {
    obj(vec![
        (
            "content",
            Json::Arr(vec![obj(vec![
                ("type", Json::str("text")),
                ("text", Json::str(text)),
            ])]),
        ),
        ("isError", Json::Bool(is_error)),
    ])
}

// ── tool registry ────────────────────────────────────────────────────────────

fn tool_definitions() -> Json {
    let vec_schema = |desc: &str| {
        obj(vec![
            ("type", Json::str("array")),
            ("items", obj(vec![("type", Json::str("number"))])),
            ("description", Json::str(desc)),
        ])
    };
    let tool = |name: &str, desc: &str, props: Vec<(&str, Json)>, required: Vec<&str>| {
        obj(vec![
            ("name", Json::str(name)),
            ("description", Json::str(desc)),
            (
                "inputSchema",
                obj(vec![
                    ("type", Json::str("object")),
                    ("properties", obj(props)),
                    (
                        "required",
                        Json::Arr(required.into_iter().map(Json::str).collect()),
                    ),
                ]),
            ),
        ])
    };

    Json::Arr(vec![
        tool(
            "slha.audit",
            "Run the SLHA v2 self-audit (tile layout, live SIMD-vs-scalar equivalence, CPU features/caches, output fidelity vs full attention, CCOS budget invariant, determinism). Returns the full JSON report.",
            vec![],
            vec![],
        ),
        tool(
            "slha.explain",
            "Explain what SLHA v2 is, the 128-byte tile, and how the hybrid attention score works. Returns prose for the agent to read or relay.",
            vec![],
            vec![],
        ),
        tool(
            "slha.compress",
            "Quantize a 128-dim key vector into the 64-byte latent of a 128-byte tile and report the compression vs FP32. Optional `codec` selects the latent quantizer (default int4 = single-scale INT4).",
            vec![
                ("key", vec_schema("128 numbers (the key vector)")),
                (
                    "codec",
                    obj(vec![
                        ("type", Json::str("string")),
                        (
                            "enum",
                            Json::Arr(
                                ["int4", "grouped", "nf4", "mixed", "tq3"]
                                    .into_iter()
                                    .map(Json::str)
                                    .collect(),
                            ),
                        ),
                        (
                            "description",
                            Json::str("latent codec (default int4: uniform INT4, single scale). grouped = per-group MX scales; nf4 = normal-float codebook; mixed = 8 dims @8-bit + 112 @4-bit; tq3 = TurboQuant port, 3-bit grid + 1-bit sign-correction plane."),
                        ),
                    ]),
                ),
            ],
            vec!["key"],
        ),
        tool(
            "slha.score",
            "Build a tile from `key` and compute the SLHA coarse score for `query`, vs the exact dot product — shows the INT4 reconstruction error.",
            vec![
                ("key", vec_schema("128 numbers (the context key)")),
                ("query", vec_schema("128 numbers (the query)")),
            ],
            vec!["key", "query"],
        ),
        tool(
            "slha.benchmark",
            "Measure SLHA score throughput on this host (scores/sec, ns/score, dispatched SIMD path). Optional `n` = iteration count.",
            vec![("n", obj(vec![("type", Json::str("number")), ("description", Json::str("iterations (default 200000)"))]))],
            vec![],
        ),
    ])
}

fn handle_tool_call(id: Json, params: Option<&Json>) -> Json {
    let params = match params {
        Some(p) => p,
        None => return err_response(id, -32602, "missing params"),
    };
    let name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => return err_response(id, -32602, "missing tool name"),
    };
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or(Json::Obj(vec![]));

    let result = match name {
        "slha.audit" => Ok(scirust::audit::run().to_pretty()),
        "slha.explain" => Ok(explain_text()),
        "slha.compress" => tool_compress(&args),
        "slha.score" => tool_score(&args),
        "slha.benchmark" => tool_benchmark(&args),
        other => Err(format!("unknown tool: {other}")),
    };
    match result {
        Ok(text) => ok_response(id, tool_result(text, false)),
        Err(e) => ok_response(id, tool_result(e, true)),
    }
}

// ── tools ────────────────────────────────────────────────────────────────────

fn explain_text() -> String {
    "SLHA v2 (Sub-Low-rank Hybrid Attention) compresses each transformer KV entry \
into a 128-byte, cache-line-aware tile so long-context inference fits in CPU cache \
instead of GPU VRAM.\n\n\
Each tile (exactly 128 bytes, zero padding) stores: a 64-byte low-rank latent \
(128 dims), a 32-byte 1-bit sign-LSH residual (256 bits) that corrects what the \
low-rank base misses, plus metadata (scale, lambda, sigma_E, ids, flags, MX group \
scales). The latent codec is selectable via flags, same 64-byte budget: uniform INT4 \
(single or per-group scales), an NF4 codebook, a mixed 8/4-bit layout, or TQ3 — a \
port of the TurboQuant KV codec storing all 128 dims as 3-bit codes (48 bytes) plus \
a separable per-dim 1-bit sign-correction plane (16 bytes). The attention score \
fuses a continuous dot product over the dequantized latent with a branchless \
popcount term over the residual: score = <q, dequant(latent)> \
+ lambda * (d_s - 2 * popcount(q_sign XOR B)). SIMD paths (AVX2/AVX-512/NEON) are \
runtime-dispatched and proven bit-equivalent to a scalar reference; NF4/mixed/TQ3 \
tiles decode on the scalar path.\n\n\
An elastic KV cache (CCOS Soft-Paging) bounds memory by paging HOT->WARM (drop the \
32-byte residual) and evicting ->COLD by age. Use `slha.audit` for live invariants, \
`slha.compress`/`slha.score` to exercise the kernel, and `slha.benchmark` for host \
throughput."
        .to_string()
}

fn tool_compress(args: &Json) -> Result<String, String> {
    let key = f32_dims(args, "key")?;
    let codec = match args.get("codec") {
        None => "int4",
        Some(v) => v
            .as_str()
            .ok_or_else(|| "'codec' must be a string".to_string())?,
    };
    // Same codec -> (quantizer, FLAG_*) mapping as `LearnedModel::encode_with`
    // and the `offline_validation --codec` example.
    let (latent, scale, group_scales, flags) = match codec {
        "int4" => {
            let (l, s) = quantize_latent(&key);
            (l, s, [255u8; N_GROUPS], FLAG_HOT)
        }
        "grouped" => {
            let (l, s, gs) = quantize_latent_grouped(&key);
            (l, s, gs, FLAG_HOT)
        }
        "nf4" => {
            let (l, s, gs) = quantize_latent_nf4(&key);
            (l, s, gs, FLAG_NF4)
        }
        "mixed" => {
            let (l, s, gs) = quantize_latent_mixed(&key);
            (l, s, gs, FLAG_MIXED)
        }
        "tq3" => {
            let (l, s, gs) = quantize_latent_tq3(&key);
            (l, s, gs, FLAG_TQ3)
        }
        other => {
            return Err(format!(
                "unknown codec '{other}': expected int4 | grouped | nf4 | mixed | tq3"
            ))
        }
    };
    let preview: Vec<Json> = latent
        .iter()
        .take(8)
        .map(|&b| Json::Num(b as f64))
        .collect();
    Ok(obj(vec![
        ("input_dims", Json::Num(DIMS as f64)),
        ("input_bytes_f32", Json::Num((DIMS * 4) as f64)),
        ("tile_bytes", Json::Num(128.0)),
        ("latent_bytes", Json::Num(latent.len() as f64)),
        ("codec", Json::str(codec)),
        ("flags", Json::Num(flags as f64)),
        (
            "compression_ratio_vs_fp32_key",
            Json::Num((DIMS * 4) as f64 / 128.0),
        ),
        ("scale", Json::Num(scale as f64)),
        (
            "group_scales",
            Json::Arr(group_scales.iter().map(|&g| Json::Num(g as f64)).collect()),
        ),
        ("latent_first_8_bytes", Json::Arr(preview)),
    ])
    .to_pretty())
}

fn tool_score(args: &Json) -> Result<String, String> {
    let key = f32_dims(args, "key")?;
    let query = f32_dims(args, "query")?;
    let proj = Projection::new(0x5C04E);
    // Treat the key as captured exactly (residual e = 0): isolates INT4 latent error.
    let tok = ContextToken {
        k_coarse: key,
        e: [0.0f32; DIMS],
        k_real: key,
    };
    let tile = build_tile(&proj, &tok, 0, false);
    let qs = proj.sign_bits(&query);
    let slha = tile.compute_score(&query, &qs);
    let truth = dot(&query, &key);
    Ok(obj(vec![
        ("slha_score", Json::Num(slha as f64)),
        ("true_dot", Json::Num(truth as f64)),
        ("abs_err", Json::Num((slha - truth).abs() as f64)),
        (
            "rel_err",
            Json::Num(((slha - truth).abs() / (1.0 + truth.abs())) as f64),
        ),
        (
            "note",
            Json::str("residual e=0 here, so this isolates the INT4 latent reconstruction error"),
        ),
    ])
    .to_pretty())
}

fn tool_benchmark(args: &Json) -> Result<String, String> {
    let n = args
        .get("n")
        .and_then(|v| v.as_f64())
        .map(|x| x as usize)
        .unwrap_or(200_000)
        .clamp(1_000, 5_000_000);
    let proj = Projection::new(0xB0001);
    let (q, toks) = generate(0xB0001, 64, 0.3);
    let qs = proj.sign_bits(&q);
    let tiles: Vec<_> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();

    let mut acc = 0.0f32;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        acc += tiles[i % tiles.len()].compute_score(&q, &qs);
    }
    let secs = t0.elapsed().as_secs_f64();
    std::hint::black_box(acc);

    Ok(obj(vec![
        ("scores", Json::Num(n as f64)),
        ("seconds", Json::Num(secs)),
        ("scores_per_sec", Json::Num(n as f64 / secs.max(1e-12))),
        ("ns_per_score", Json::Num(secs * 1e9 / n as f64)),
        ("dispatched_path", Json::str(dispatched_path())),
        ("arch", Json::str(std::env::consts::ARCH)),
    ])
    .to_pretty())
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn f32_dims(args: &Json, key: &str) -> Result<[f32; DIMS], String> {
    let arr = args
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing number array '{key}' (expected {DIMS} values)"))?;
    if arr.len() != DIMS {
        return Err(format!(
            "'{key}' must have {DIMS} numbers, got {}",
            arr.len()
        ));
    }
    let mut out = [0.0f32; DIMS];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v
            .as_f64()
            .ok_or_else(|| format!("'{key}[{i}]' is not a number"))? as f32;
    }
    Ok(out)
}

fn dispatched_path() -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_id(method: &str, params: Json, id: Json) -> Json {
        obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("id", id),
            ("method", Json::str(method)),
            ("params", params),
        ])
    }

    fn request(method: &str, params: Json, id: i64) -> Json {
        request_with_id(method, params, Json::Num(id as f64))
    }

    fn initialize_params(protocol_version: &str) -> Json {
        obj(vec![
            ("protocolVersion", Json::str(protocol_version)),
            ("capabilities", Json::Obj(vec![])),
            (
                "clientInfo",
                obj(vec![
                    ("name", Json::str("slha-mcp-tests")),
                    ("version", Json::str("1.0.0")),
                ]),
            ),
        ])
    }

    fn call(method: &str, params: Json, id: i64) -> Json {
        let request = request(method, params, id);
        let mut lifecycle = Lifecycle::Ready;

        handle(&request, &mut lifecycle).expect("expected a response")
    }

    fn error_code(response: &Json) -> Option<f64> {
        response
            .get("error")
            .and_then(|error| error.get("code"))
            .and_then(Json::as_f64)
    }

    #[test]
    fn initialize_advertises_server() {
        let request = request("initialize", initialize_params(PROTOCOL_VERSION), 1);
        let mut lifecycle = Lifecycle::Uninitialized;

        let response = handle(&request, &mut lifecycle).expect("initialize must return a response");

        assert_eq!(lifecycle, Lifecycle::AwaitingInitialized);

        let result = response.get("result").unwrap();
        let info = result.get("serverInfo").unwrap();

        assert_eq!(info.get("name").unwrap().as_str(), Some("slha-mcp"));
        assert_eq!(
            result.get("protocolVersion").unwrap().as_str(),
            Some(PROTOCOL_VERSION)
        );
        assert!(result.get("capabilities").is_some());
    }

    #[test]
    fn supported_legacy_protocol_version_is_echoed() {
        let request = request("initialize", initialize_params("2024-11-05"), 1);
        let mut lifecycle = Lifecycle::Uninitialized;

        let response = handle(&request, &mut lifecycle).unwrap();

        assert_eq!(
            response
                .get("result")
                .unwrap()
                .get("protocolVersion")
                .unwrap()
                .as_str(),
            Some("2024-11-05")
        );
    }

    #[test]
    fn unsupported_protocol_version_falls_back_to_current() {
        let request = request("initialize", initialize_params("1900-01-01"), 1);
        let mut lifecycle = Lifecycle::Uninitialized;

        let response = handle(&request, &mut lifecycle).unwrap();

        assert_eq!(
            response
                .get("result")
                .unwrap()
                .get("protocolVersion")
                .unwrap()
                .as_str(),
            Some(PROTOCOL_VERSION)
        );
    }

    #[test]
    fn initialize_notification_gets_no_reply_and_preserves_lifecycle() {
        let notification = obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("method", Json::str("initialize")),
            ("params", initialize_params(PROTOCOL_VERSION)),
        ]);
        let mut lifecycle = Lifecycle::Uninitialized;

        assert!(handle(&notification, &mut lifecycle).is_none());
        assert_eq!(lifecycle, Lifecycle::Uninitialized);
    }

    #[test]
    fn notifications_get_no_reply() {
        let notification = obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("method", Json::str("notifications/initialized")),
        ]);
        let mut lifecycle = Lifecycle::AwaitingInitialized;

        assert!(handle(&notification, &mut lifecycle).is_none());
        assert_eq!(lifecycle, Lifecycle::Ready);
    }

    #[test]
    fn tools_require_completed_initialization() {
        let list_request = request("tools/list", Json::Obj(vec![]), 2);
        let mut lifecycle = Lifecycle::Uninitialized;

        let before_initialize = handle(&list_request, &mut lifecycle).unwrap();
        assert_eq!(error_code(&before_initialize), Some(-32002.0));

        let initialize = request("initialize", initialize_params(PROTOCOL_VERSION), 1);
        assert!(handle(&initialize, &mut lifecycle).is_some());
        assert_eq!(lifecycle, Lifecycle::AwaitingInitialized);

        let before_notification = handle(&list_request, &mut lifecycle).unwrap();
        assert_eq!(error_code(&before_notification), Some(-32002.0));

        let notification = obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("method", Json::str("notifications/initialized")),
        ]);
        assert!(handle(&notification, &mut lifecycle).is_none());
        assert_eq!(lifecycle, Lifecycle::Ready);

        let ready = handle(&list_request, &mut lifecycle).unwrap();
        assert!(ready.get("result").is_some());
    }

    #[test]
    fn invalid_json_rpc_envelopes_are_rejected() {
        let mut lifecycle = Lifecycle::Ready;

        let array = Json::Arr(vec![]);
        assert_eq!(
            error_code(&handle(&array, &mut lifecycle).unwrap()),
            Some(-32600.0)
        );

        let missing_version = obj(vec![("id", Json::Num(1.0)), ("method", Json::str("ping"))]);
        assert_eq!(
            error_code(&handle(&missing_version, &mut lifecycle).unwrap()),
            Some(-32600.0)
        );

        let fractional_id = obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("id", Json::Num(1.5)),
            ("method", Json::str("ping")),
        ]);
        assert_eq!(
            error_code(&handle(&fractional_id, &mut lifecycle).unwrap()),
            Some(-32600.0)
        );

        let null_id = obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("id", Json::Null),
            ("method", Json::str("ping")),
        ]);
        assert_eq!(
            error_code(&handle(&null_id, &mut lifecycle).unwrap()),
            Some(-32600.0)
        );

        let array_params = obj(vec![
            ("jsonrpc", Json::str("2.0")),
            ("id", Json::Num(1.0)),
            ("method", Json::str("ping")),
            ("params", Json::Arr(vec![])),
        ]);
        assert_eq!(
            error_code(&handle(&array_params, &mut lifecycle).unwrap()),
            Some(-32602.0)
        );
    }

    #[test]
    fn string_request_id_round_trips() {
        let request = request_with_id("ping", Json::Obj(vec![]), Json::str("request-42"));
        let mut lifecycle = Lifecycle::Ready;

        let response = handle(&request, &mut lifecycle).unwrap();

        assert_eq!(
            response.get("id").and_then(Json::as_str),
            Some("request-42")
        );
    }

    #[test]
    fn bounded_frame_reader_discards_oversized_frame_and_recovers() {
        let valid = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let input = format!("{}\n{valid}\n", "x".repeat(MAX_FRAME_BYTES + 1));

        let mut reader = std::io::Cursor::new(input.into_bytes());
        let mut buffer = Vec::new();

        assert_eq!(
            read_frame(&mut reader, &mut buffer, MAX_FRAME_BYTES).unwrap(),
            FrameRead::TooLarge
        );

        assert_eq!(
            read_frame(&mut reader, &mut buffer, MAX_FRAME_BYTES).unwrap(),
            FrameRead::Ready
        );

        assert_eq!(std::str::from_utf8(&buffer).unwrap(), valid);

        assert_eq!(
            read_frame(&mut reader, &mut buffer, MAX_FRAME_BYTES).unwrap(),
            FrameRead::Eof
        );
    }

    #[test]
    fn tools_list_has_all_five() {
        let r = call("tools/list", Json::Obj(vec![]), 2);
        let tools = r
            .get("result")
            .unwrap()
            .get("tools")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tools.len(), 5);
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();
        for want in [
            "slha.audit",
            "slha.explain",
            "slha.compress",
            "slha.score",
            "slha.benchmark",
        ] {
            assert!(names.contains(&want), "missing tool {want}");
        }
    }

    #[test]
    fn audit_tool_runs_and_reports_ok() {
        let params = obj(vec![
            ("name", Json::str("slha.audit")),
            ("arguments", Json::Obj(vec![])),
        ]);
        let r = call("tools/call", params, 3);
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(false));
        let text = res.get("content").unwrap().as_array().unwrap()[0]
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();
        let report = Json::parse(text).expect("audit text is JSON");
        assert_eq!(
            report.get("verdict").unwrap().get("ok").unwrap().as_bool(),
            Some(true)
        );
    }

    #[test]
    fn compress_reports_4x() {
        let key = Json::Arr(
            (0..128)
                .map(|i| Json::Num((i as f64 / 128.0) - 0.5))
                .collect(),
        );
        let params = obj(vec![
            ("name", Json::str("slha.compress")),
            ("arguments", obj(vec![("key", key)])),
        ]);
        let r = call("tools/call", params, 4);
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(false));
        let text = res.get("content").unwrap().as_array().unwrap()[0]
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();
        let j = Json::parse(text).unwrap();
        assert_eq!(j.get("tile_bytes").unwrap().as_f64(), Some(128.0));
        assert_eq!(
            j.get("compression_ratio_vs_fp32_key").unwrap().as_f64(),
            Some(4.0)
        );
    }

    /// Call `slha.compress` on a fixed 128-dim key, with an optional codec.
    fn compress_with(codec: Option<&str>) -> Json {
        let key = Json::Arr(
            (0..128)
                .map(|i| Json::Num((i as f64 / 128.0) - 0.5))
                .collect(),
        );
        let mut args = vec![("key", key)];
        if let Some(c) = codec {
            args.push(("codec", Json::str(c)));
        }
        let params = obj(vec![
            ("name", Json::str("slha.compress")),
            ("arguments", obj(args)),
        ]);
        call("tools/call", params, 7)
    }

    /// Parse the JSON text payload of a successful compress result.
    fn compress_json(r: &Json) -> Json {
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(false));
        let text = res.get("content").unwrap().as_array().unwrap()[0]
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();
        Json::parse(text).unwrap()
    }

    #[test]
    fn compress_default_codec_is_single_scale_int4() {
        let j = compress_json(&compress_with(None));
        assert_eq!(j.get("codec").unwrap().as_str(), Some("int4"));
        assert_eq!(j.get("flags").unwrap().as_f64(), Some(FLAG_HOT as f64));
        // Single-scale INT4: every group micro-scale is the identity (255).
        let gs = j.get("group_scales").unwrap().as_array().unwrap();
        assert_eq!(gs.len(), N_GROUPS);
        assert!(gs.iter().all(|g| g.as_f64() == Some(255.0)));
    }

    #[test]
    fn compress_codec_sets_matching_flags() {
        for (codec, flag) in [
            ("int4", FLAG_HOT),
            ("grouped", FLAG_HOT),
            ("nf4", FLAG_NF4),
            ("mixed", FLAG_MIXED),
            ("tq3", FLAG_TQ3),
        ] {
            let j = compress_json(&compress_with(Some(codec)));
            assert_eq!(j.get("codec").unwrap().as_str(), Some(codec));
            assert_eq!(
                j.get("flags").unwrap().as_f64(),
                Some(flag as f64),
                "wrong flags for codec {codec}"
            );
            // Every codec spends the same budgets: 64-byte latent, 128-byte tile.
            assert_eq!(j.get("latent_bytes").unwrap().as_f64(), Some(64.0));
            assert_eq!(j.get("tile_bytes").unwrap().as_f64(), Some(128.0));
            assert_eq!(
                j.get("compression_ratio_vs_fp32_key").unwrap().as_f64(),
                Some(4.0)
            );
        }
    }

    #[test]
    fn compress_rejects_unknown_codec() {
        let r = compress_with(Some("fp8"));
        assert_eq!(
            r.get("result").unwrap().get("isError").unwrap().as_bool(),
            Some(true)
        );
    }

    #[test]
    fn bad_args_surface_as_tool_error() {
        let params = obj(vec![
            ("name", Json::str("slha.compress")),
            (
                "arguments",
                obj(vec![("key", Json::Arr(vec![Json::Num(1.0)]))]),
            ), // wrong length
        ]);
        let r = call("tools/call", params, 5);
        assert_eq!(
            r.get("result").unwrap().get("isError").unwrap().as_bool(),
            Some(true)
        );
    }

    #[test]
    fn unknown_method_is_json_rpc_error() {
        let r = call("does/not/exist", Json::Obj(vec![]), 6);
        assert!(r.get("error").is_some());
        assert_eq!(
            r.get("error").unwrap().get("code").unwrap().as_f64(),
            Some(-32601.0)
        );
    }
}
