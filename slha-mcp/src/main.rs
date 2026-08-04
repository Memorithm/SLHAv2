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
    quantize_latent, quantize_latent_grouped, quantize_latent_mix3, quantize_latent_mixed,
    quantize_latent_nf4, quantize_latent_tq3, FLAG_HOT, FLAG_MIX3, FLAG_MIXED, FLAG_NF4, FLAG_TQ3,
    N_GROUPS,
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

const BENCHMARK_DEFAULT_ITERATIONS: usize = 200_000;
const BENCHMARK_MIN_ITERATIONS: usize = 1_000;
const BENCHMARK_MAX_ITERATIONS: usize = 5_000_000;

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
        // Bind in the pattern: no panic path, however unreachable.
        Some(p @ Json::Obj(_)) => p,
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
        // Bind in the pattern instead of re-fetching and expecting.
        Some(ci @ Json::Obj(_)) => ci,
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
    let vector_schema = |description: &str| {
        obj(vec![
            ("type", Json::str("array")),
            (
                "items",
                obj(vec![
                    ("type", Json::str("number")),
                    ("minimum", Json::Num(-(f32::MAX as f64))),
                    ("maximum", Json::Num(f32::MAX as f64)),
                ]),
            ),
            ("minItems", Json::Num(DIMS as f64)),
            ("maxItems", Json::Num(DIMS as f64)),
            ("description", Json::str(description)),
        ])
    };

    let tool =
        |name: &str, description: &str, properties: Vec<(&str, Json)>, required: Vec<&str>| {
            obj(vec![
                ("name", Json::str(name)),
                ("description", Json::str(description)),
                (
                    "inputSchema",
                    obj(vec![
                        (
                            "$schema",
                            Json::str("https://json-schema.org/draft/2020-12/schema"),
                        ),
                        ("type", Json::str("object")),
                        ("properties", obj(properties)),
                        (
                            "required",
                            Json::Arr(required.into_iter().map(Json::str).collect()),
                        ),
                        ("additionalProperties", Json::Bool(false)),
                    ]),
                ),
            ])
        };

    Json::Arr(vec![
        tool(
            "slha.audit",
            "Run the SLHA v2 self-audit and return the complete JSON report.",
            vec![],
            vec![],
        ),
        tool(
            "slha.explain",
            "Explain the SLHA v2 tile, codecs, hybrid attention score and elastic paging.",
            vec![],
            vec![],
        ),
        tool(
            "slha.compress",
            "Quantize one 128-dimensional key into the 64-byte latent of an SLHA v2 tile.",
            vec![
                (
                    "key",
                    vector_schema(
                        "Exactly 128 finite numbers representable as IEEE-754 f32.",
                    ),
                ),
                (
                    "codec",
                    obj(vec![
                        ("type", Json::str("string")),
                        (
                            "enum",
                            Json::Arr(
                                [
                                    "int4",
                                    "grouped",
                                    "nf4",
                                    "mixed",
                                    "tq3",
                                    "mix3",
                                ]
                                .into_iter()
                                .map(Json::str)
                                .collect(),
                            ),
                        ),
                        ("default", Json::str("int4")),
                        (
                            "description",
                            Json::str(
                                "int4: single-scale uniform INT4; grouped: \
                                 per-group INT4; nf4: NormalFloat-4; mixed: \
                                 8-bit head plus 4-bit body; tq3: 3-bit grid \
                                 plus correction plane; mix3: 8-bit head plus \
                                 TQ3 body and separable correction plane.",
                            ),
                        ),
                    ]),
                ),
            ],
            vec!["key"],
        ),
        tool(
            "slha.score",
            "Compare the SLHA coarse score with the exact dot product for two 128-dimensional vectors.",
            vec![
                (
                    "key",
                    vector_schema(
                        "Exactly 128 finite context-key values representable as f32.",
                    ),
                ),
                (
                    "query",
                    vector_schema(
                        "Exactly 128 finite query values representable as f32.",
                    ),
                ),
            ],
            vec!["key", "query"],
        ),
        tool(
            "slha.benchmark",
            "Measure local SLHA score throughput with a bounded iteration count.",
            vec![(
                "n",
                obj(vec![
                    ("type", Json::str("integer")),
                    (
                        "minimum",
                        Json::Num(BENCHMARK_MIN_ITERATIONS as f64),
                    ),
                    (
                        "maximum",
                        Json::Num(BENCHMARK_MAX_ITERATIONS as f64),
                    ),
                    (
                        "default",
                        Json::Num(BENCHMARK_DEFAULT_ITERATIONS as f64),
                    ),
                    (
                        "description",
                        Json::str("Number of score iterations."),
                    ),
                ]),
            )],
            vec![],
        ),
    ])
}

fn validate_object_fields(value: &Json, allowed: &[&str], context: &str) -> Result<(), String> {
    let fields = match value {
        Json::Obj(fields) => fields,
        _ => return Err(format!("{context} must be an object")),
    };

    for (key, _) in fields {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{context} contains unknown property '{key}'"));
        }
    }

    Ok(())
}

fn handle_tool_call(id: Json, params: Option<&Json>) -> Json {
    let params = match params {
        // Bind in the pattern: no panic path, however unreachable.
        Some(p @ Json::Obj(_)) => p,
        Some(_) => {
            return err_response(id, -32602, "tools/call params must be an object");
        }
        None => return err_response(id, -32602, "missing params"),
    };

    if let Err(error) = validate_object_fields(params, &["name", "arguments"], "tools/call params")
    {
        return err_response(id, -32602, &error);
    }

    let name = match params.get("name").and_then(Json::as_str) {
        Some(name) if !name.is_empty() => name,
        _ => return err_response(id, -32602, "missing tool name"),
    };

    let arguments = match params.get("arguments") {
        None => Json::Obj(vec![]),
        Some(a @ Json::Obj(_)) => a.clone(),
        Some(_) => {
            return err_response(id, -32602, "tool arguments must be an object");
        }
    };

    let result = match name {
        "slha.audit" => validate_object_fields(&arguments, &[], "slha.audit arguments")
            .map(|()| scirust::audit::run().to_pretty()),

        "slha.explain" => validate_object_fields(&arguments, &[], "slha.explain arguments")
            .map(|()| explain_text()),

        "slha.compress" => tool_compress(&arguments),
        "slha.score" => tool_score(&arguments),
        "slha.benchmark" => tool_benchmark(&arguments),

        other => Err(format!("unknown tool: {other}")),
    };

    match result {
        Ok(output) => ok_response(id, tool_result(output, false)),
        Err(error) => ok_response(id, tool_result(error, true)),
    }
}

// ── tools ────────────────────────────────────────────────────────────────────

fn explain_text() -> String {
    "SLHA v2 (Sub-Low-rank Hybrid Attention) compresses each transformer KV \
entry into a deterministic 128-byte, cache-line-aware tile.\n\n\
Each tile stores a 64-byte low-rank latent, a 32-byte one-bit sign-LSH \
residual and compact metadata. The latent codec can be uniform INT4, grouped \
INT4, NF4, mixed 8/4-bit, TQ3, or MIX3. TQ3 and MIX3 keep a separable \
correction plane that CCOS can page out independently.\n\n\
The attention score combines the dequantized continuous term with a \
branchless residual-popcount correction. Scalar, AVX2, AVX-512 and NEON \
implementations are selected at runtime where supported and are regression \
tested against the scalar reference.\n\n\
CCOS Soft-Paging moves tiles through HOT, WARM and COLD states while enforcing \
a memory budget. Use `slha.audit` for invariants, `slha.compress` for codec \
inspection, `slha.score` for score comparison and `slha.benchmark` for local \
throughput."
        .to_string()
}

fn tool_compress(args: &Json) -> Result<String, String> {
    validate_object_fields(args, &["key", "codec"], "slha.compress arguments")?;

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
        "mix3" => {
            let (l, s, gs) = quantize_latent_mix3(&key);
            (l, s, gs, FLAG_MIX3)
        }
        other => {
            return Err(format!(
                "unknown codec '{other}': expected int4 | grouped | nf4 | mixed | tq3 | mix3"
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
    validate_object_fields(args, &["key", "query"], "slha.score arguments")?;

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
    let abs_err = (slha - truth).abs();
    let rel_err = abs_err / (1.0 + truth.abs());

    if !slha.is_finite() || !truth.is_finite() || !abs_err.is_finite() || !rel_err.is_finite() {
        return Err(
            "score calculation produced a non-finite value; normalize the input vectors"
                .to_string(),
        );
    }

    Ok(obj(vec![
        ("slha_score", Json::Num(slha as f64)),
        ("true_dot", Json::Num(truth as f64)),
        ("abs_err", Json::Num(abs_err as f64)),
        ("rel_err", Json::Num(rel_err as f64)),
        (
            "note",
            Json::str("residual e=0 here, so this isolates the INT4 latent reconstruction error"),
        ),
    ])
    .to_pretty())
}

fn bounded_integer_argument(
    args: &Json,
    key: &str,
    default: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize, String> {
    let Some(value) = args.get(key) else {
        return Ok(default);
    };

    let number = value
        .as_f64()
        .ok_or_else(|| format!("'{key}' must be an integer"))?;

    if !number.is_finite() || number.fract() != 0.0 {
        return Err(format!("'{key}' must be a finite integer"));
    }

    if number < minimum as f64 || number > maximum as f64 {
        return Err(format!("'{key}' must be between {minimum} and {maximum}"));
    }

    Ok(number as usize)
}

fn tool_benchmark(args: &Json) -> Result<String, String> {
    validate_object_fields(args, &["n"], "slha.benchmark arguments")?;

    let n = bounded_integer_argument(
        args,
        "n",
        BENCHMARK_DEFAULT_ITERATIONS,
        BENCHMARK_MIN_ITERATIONS,
        BENCHMARK_MAX_ITERATIONS,
    )?;

    let projection = Projection::new(0xB0001);
    let (query, tokens) = generate(0xB0001, 64, 0.3);
    let query_signs = projection.sign_bits(&query);

    let tiles: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| build_tile(&projection, token, index as u32, false))
        .collect();

    let mut accumulator = 0.0f32;
    let started = std::time::Instant::now();

    for index in 0..n {
        accumulator += tiles[index % tiles.len()].compute_score(&query, &query_signs);
    }

    let seconds = started.elapsed().as_secs_f64();
    std::hint::black_box(accumulator);

    Ok(obj(vec![
        ("scores", Json::Num(n as f64)),
        ("seconds", Json::Num(seconds)),
        ("scores_per_sec", Json::Num(n as f64 / seconds.max(1e-12))),
        ("ns_per_score", Json::Num(seconds * 1e9 / n as f64)),
        ("dispatched_path", Json::str(dispatched_path())),
        ("arch", Json::str(std::env::consts::ARCH)),
    ])
    .to_pretty())
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn f32_dims(args: &Json, key: &str) -> Result<[f32; DIMS], String> {
    let array = args
        .get(key)
        .and_then(Json::as_array)
        .ok_or_else(|| format!("missing number array '{key}' (expected {DIMS} values)"))?;

    if array.len() != DIMS {
        return Err(format!(
            "'{key}' must have {DIMS} numbers, got {}",
            array.len()
        ));
    }

    let mut output = [0.0f32; DIMS];

    for (index, value) in array.iter().enumerate() {
        let number = value
            .as_f64()
            .ok_or_else(|| format!("'{key}[{index}]' is not a number"))?;

        if !number.is_finite() {
            return Err(format!("'{key}[{index}]' must be finite"));
        }

        let converted = number as f32;

        if !converted.is_finite() {
            return Err(format!("'{key}[{index}]' is outside the finite f32 range"));
        }

        output[index] = converted;
    }

    Ok(output)
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
            ("mix3", FLAG_MIX3),
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
    fn tool_schemas_are_strict_and_bounded() {
        let tools = tool_definitions();
        let tools = tools.as_array().unwrap();

        for tool in tools {
            let schema = tool.get("inputSchema").unwrap();

            assert_eq!(
                schema.get("additionalProperties").and_then(Json::as_bool),
                Some(false),
                "tool {} does not forbid additional properties",
                tool.get("name").and_then(Json::as_str).unwrap()
            );
        }

        let compress = tools
            .iter()
            .find(|tool| tool.get("name").and_then(Json::as_str) == Some("slha.compress"))
            .unwrap();

        let properties = compress
            .get("inputSchema")
            .unwrap()
            .get("properties")
            .unwrap();

        let key = properties.get("key").unwrap();

        assert_eq!(
            key.get("minItems").and_then(Json::as_f64),
            Some(DIMS as f64)
        );
        assert_eq!(
            key.get("maxItems").and_then(Json::as_f64),
            Some(DIMS as f64)
        );

        let item = key.get("items").unwrap();

        assert_eq!(
            item.get("minimum").and_then(Json::as_f64),
            Some(-(f32::MAX as f64))
        );
        assert_eq!(
            item.get("maximum").and_then(Json::as_f64),
            Some(f32::MAX as f64)
        );

        let codec_values: Vec<&str> = properties
            .get("codec")
            .unwrap()
            .get("enum")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Json::as_str)
            .collect();

        assert!(codec_values.contains(&"mix3"));

        let benchmark = tools
            .iter()
            .find(|tool| tool.get("name").and_then(Json::as_str) == Some("slha.benchmark"))
            .unwrap();

        let n = benchmark
            .get("inputSchema")
            .unwrap()
            .get("properties")
            .unwrap()
            .get("n")
            .unwrap();

        assert_eq!(n.get("type").and_then(Json::as_str), Some("integer"));
        assert_eq!(
            n.get("minimum").and_then(Json::as_f64),
            Some(BENCHMARK_MIN_ITERATIONS as f64)
        );
        assert_eq!(
            n.get("maximum").and_then(Json::as_f64),
            Some(BENCHMARK_MAX_ITERATIONS as f64)
        );
    }

    #[test]
    fn tool_call_rejects_non_object_arguments() {
        let params = obj(vec![
            ("name", Json::str("slha.audit")),
            ("arguments", Json::Arr(vec![])),
        ]);

        let response = call("tools/call", params, 20);

        assert_eq!(error_code(&response), Some(-32602.0));
    }

    #[test]
    fn tools_reject_unknown_argument_properties() {
        let key = Json::Arr((0..DIMS).map(|_| Json::Num(0.0)).collect());

        let params = obj(vec![
            ("name", Json::str("slha.compress")),
            (
                "arguments",
                obj(vec![("key", key), ("unexpected", Json::Bool(true))]),
            ),
        ]);

        let response = call("tools/call", params, 21);

        assert_eq!(
            response
                .get("result")
                .unwrap()
                .get("isError")
                .and_then(Json::as_bool),
            Some(true)
        );
    }

    #[test]
    fn vectors_reject_values_outside_f32_range() {
        let mut values = vec![Json::Num(0.0); DIMS];
        values[17] = Json::Num(f64::MAX);

        let params = obj(vec![
            ("name", Json::str("slha.compress")),
            ("arguments", obj(vec![("key", Json::Arr(values))])),
        ]);

        let response = call("tools/call", params, 22);

        assert_eq!(
            response
                .get("result")
                .unwrap()
                .get("isError")
                .and_then(Json::as_bool),
            Some(true)
        );
    }

    #[test]
    fn score_rejects_non_finite_results() {
        let values = Json::Arr((0..DIMS).map(|_| Json::Num(f32::MAX as f64)).collect());

        let params = obj(vec![
            ("name", Json::str("slha.score")),
            (
                "arguments",
                obj(vec![("key", values.clone()), ("query", values)]),
            ),
        ]);

        let response = call("tools/call", params, 23);

        assert_eq!(
            response
                .get("result")
                .unwrap()
                .get("isError")
                .and_then(Json::as_bool),
            Some(true)
        );
    }

    #[test]
    fn benchmark_requires_a_bounded_integer() {
        for invalid in [
            Json::Num(1_000.5),
            Json::Num((BENCHMARK_MIN_ITERATIONS - 1) as f64),
            Json::Num((BENCHMARK_MAX_ITERATIONS + 1) as f64),
            Json::str("200000"),
        ] {
            let params = obj(vec![
                ("name", Json::str("slha.benchmark")),
                ("arguments", obj(vec![("n", invalid)])),
            ]);

            let response = call("tools/call", params, 24);

            assert_eq!(
                response
                    .get("result")
                    .unwrap()
                    .get("isError")
                    .and_then(Json::as_bool),
                Some(true)
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
