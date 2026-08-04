//! Minimal, zero-dependency JSON — value model, serializer, and parser.
//!
//! Just enough for the [`crate::audit`] reports and the `slha-mcp` server: no
//! `serde`, no external crates, one file. The serializer emits **compact
//! single-line** output (required for MCP's newline-delimited stdio framing) and
//! a **pretty** variant for human-readable report files. The parser is a small
//! recursive-descent reader for arbitrary JSON (objects, arrays, strings with
//! `\uXXXX`/surrogate escapes, numbers, booleans, null).

use std::fmt::Write as _;

/// A JSON value. Objects keep insertion order (a `Vec` of pairs) so reports and
/// MCP messages render deterministically.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// Build an object from `(&str, Json)` pairs (ergonomic constructor).
pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

impl Json {
    /// String value from anything `Into<String>`.
    pub fn str(v: impl Into<String>) -> Json {
        Json::Str(v.into())
    }

    // ---- accessors (ergonomic for MCP dispatch / report diffing) ----

    /// Object field by key, if this is an object containing it.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    // ---- serialize ----

    /// Compact single-line JSON (used for MCP stdio messages).
    pub fn to_compact(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, None, 0);
        s
    }

    /// Pretty, 2-space-indented JSON (used for human-readable report files).
    pub fn to_pretty(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, Some(2), 0);
        s
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if !n.is_finite() {
                    out.push_str("null"); // JSON has no NaN/Inf
                } else if *n == n.trunc() && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64); // integers without ".0"
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Json::Str(s) => write_escaped(out, s),
            Json::Arr(a) => {
                // Serializer depth cap: mirror the parser's limit so a deeply
                // nested value built from untrusted data cannot recurse
                // unbounded on the way out either.
                if level >= MAX_DEPTH {
                    out.push_str("[...]");
                    return;
                }
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline_indent(out, indent, level + 1);
                    v.write(out, indent, level + 1);
                }
                newline_indent(out, indent, level);
                out.push(']');
            }
            Json::Obj(m) => {
                if level >= MAX_DEPTH {
                    out.push_str("{...}");
                    return;
                }
                if m.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline_indent(out, indent, level + 1);
                    write_escaped(out, k);
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    v.write(out, indent, level + 1);
                }
                newline_indent(out, indent, level);
                out.push('}');
            }
        }
    }

    // ---- parse ----

    /// Parse a JSON document with the default input-size limit.
    ///
    /// Duplicate object keys, non-standard numbers and unescaped control
    /// characters are rejected.
    pub fn parse(input: &str) -> Result<Json, String> {
        Self::parse_with_limit(input, MAX_INPUT_BYTES)
    }

    /// Parse a JSON document with an explicit maximum input size.
    ///
    /// This is useful for protocol boundaries where the caller wants a limit
    /// lower than [`MAX_INPUT_BYTES`].
    pub fn parse_with_limit(input: &str, max_bytes: usize) -> Result<Json, String> {
        if input.len() > max_bytes {
            return Err(format!(
                "JSON input too large: {} bytes exceeds limit {max_bytes}",
                input.len()
            ));
        }

        let mut p = Parser {
            b: input.as_bytes(),
            i: 0,
            depth: 0,
        };

        p.ws();
        let value = p.value()?;
        p.ws();

        if p.i != p.b.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }

        Ok(value)
    }
}

fn newline_indent(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(w) = indent {
        out.push('\n');
        for _ in 0..w * level {
            out.push(' ');
        }
    }
}

fn write_escaped(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

/// Default maximum JSON document size accepted by [`Json::parse`].
pub const MAX_INPUT_BYTES: usize = 1024 * 1024;

/// Maximum number of nested arrays and objects.
const MAX_DEPTH: usize = 64;

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();

        let next = self.peek();

        if matches!(next, Some(b'{') | Some(b'[')) && self.depth >= MAX_DEPTH {
            return Err("JSON nesting too deep".into());
        }

        match next {
            Some(b'{') => {
                self.depth += 1;
                let r = self.object();
                self.depth -= 1;
                r
            }
            Some(b'[') => {
                self.depth += 1;
                let r = self.array();
                self.depth -= 1;
                r
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected byte '{}' at {}", c as char, self.i)),
            None => Err("unexpected end of input".into()),
        }
    }
    fn literal(&mut self, lit: &str, val: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Ok(val)
        } else {
            Err(format!("invalid literal at {}", self.i))
        }
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;

        if self.peek() == Some(b'-') {
            self.i += 1;
        }

        // Integer part: either exactly zero, or a non-zero digit followed by
        // digits. JSON forbids leading zeroes such as 01 and -01.
        match self.peek() {
            Some(b'0') => {
                self.i += 1;

                if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return Err(format!("leading zero in number at byte {start}"));
                }
            }
            Some(b'1'..=b'9') => {
                self.i += 1;

                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return Err(format!("invalid number at byte {start}")),
        }

        // Fractional part requires at least one digit after the decimal point.
        if self.peek() == Some(b'.') {
            self.i += 1;

            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(format!(
                    "missing fractional digit in number at byte {}",
                    self.i
                ));
            }

            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
        }

        // Exponent requires at least one digit after the optional sign.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.i += 1;

            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }

            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(format!(
                    "missing exponent digit in number at byte {}",
                    self.i
                ));
            }

            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
        }

        let source =
            std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad utf8 in number")?;

        let number = source
            .parse::<f64>()
            .map_err(|_| format!("bad number '{source}'"))?;

        if !number.is_finite() {
            return Err(format!("number out of finite range '{source}'"));
        }

        Ok(Json::Num(number))
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(s);
                }
                Some(b'\\') => {
                    self.i += 1; // backslash
                    let e = self.peek().ok_or("eof in escape")?;
                    self.i += 1; // selector
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{0008}'),
                        b'f' => s.push('\u{000C}'),
                        b'u' => s.push(self.unicode_escape()?),
                        _ => return Err(format!("bad escape '\\{}' at {}", e as char, self.i)),
                    }
                }
                Some(control) if control < 0x20 => {
                    return Err(format!("unescaped control character at byte {}", self.i));
                }
                Some(_) => {
                    let len = utf8_len(self.b[self.i]);
                    let chunk = self
                        .b
                        .get(self.i..self.i + len)
                        .ok_or("truncated utf8 in string")?;
                    s.push_str(std::str::from_utf8(chunk).map_err(|_| "bad utf8 in string")?);
                    self.i += len;
                }
            }
        }
    }
    /// Reads the 4 hex digits after `\u` (already consumed), resolving surrogate
    /// pairs into a single `char`.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let hi = self.hex4()?;
        if (0xD800..=0xDBFF).contains(&hi) {
            if self.peek() == Some(b'\\') && self.b.get(self.i + 1) == Some(&b'u') {
                self.i += 2; // consume "\u"
                let lo = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&lo) {
                    return Err("invalid low surrogate".into());
                }
                let c = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                return char::from_u32(c).ok_or_else(|| "invalid surrogate pair".into());
            }
            return Err("lone high surrogate".into());
        }
        char::from_u32(hi).ok_or_else(|| "invalid code point".into())
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek().ok_or("short \\u escape")?;
            let d = match c {
                b'0'..=b'9' => (c - b'0') as u32,
                b'a'..=b'f' => (c - b'a' + 10) as u32,
                b'A'..=b'F' => (c - b'A' + 10) as u32,
                _ => return Err("bad hex digit".into()),
            };
            v = v * 16 + d;
            self.i += 1;
        }
        Ok(v)
    }
    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // [
        let mut a = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(a));
        }
        loop {
            a.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.i)),
            }
        }
    }
    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // {
        let mut m = Vec::new();
        // O(1) duplicate-key detection (the old `m.iter().any(...)` was O(n²)
        // and a hostile object with many single-char keys could burn unbounded
        // CPU inside the 256 KiB frame cap).
        let mut seen = std::collections::HashSet::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected key string at {}", self.i));
            }
            let key_offset = self.i;
            let k = self.string()?;

            if !seen.insert(k.clone()) {
                return Err(format!("duplicate object key '{}' at byte {key_offset}", k));
            }

            self.ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.i));
            }
            self.i += 1;
            let v = self.value()?;
            m.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
            }
        }
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_compact() {
        let v = obj(vec![
            ("ok", Json::Bool(true)),
            ("n", Json::Num(42.0)),
            ("f", Json::Num(0.5)),
            ("s", Json::str("a\"b\\c\n")),
            ("arr", Json::Arr(vec![Json::Num(1.0), Json::Null])),
            ("empty_obj", Json::Obj(vec![])),
        ]);
        let parsed = Json::parse(&v.to_compact()).expect("parse compact");
        assert_eq!(parsed, v);
        let parsed_pretty = Json::parse(&v.to_pretty()).expect("parse pretty");
        assert_eq!(parsed_pretty, v);
    }

    #[test]
    fn integers_have_no_dot_zero() {
        assert_eq!(Json::Num(128.0).to_compact(), "128");
        assert_eq!(Json::Num(-3.0).to_compact(), "-3");
        assert_eq!(Json::Num(0.25).to_compact(), "0.25");
    }

    #[test]
    fn parses_nested_and_escapes() {
        let j = Json::parse(r#"{ "a": [1, 2.5e1, {"b": "x\tyé"}], "z": false }"#).unwrap();
        assert_eq!(j.get("a").unwrap().as_array().unwrap().len(), 3);
        assert_eq!(
            j.get("a").unwrap().as_array().unwrap()[1].as_f64(),
            Some(25.0)
        );
        let inner = &j.get("a").unwrap().as_array().unwrap()[2];
        assert_eq!(inner.get("b").unwrap().as_str(), Some("x\ty\u{e9}"));
        assert_eq!(j.get("z").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn non_finite_becomes_null() {
        assert_eq!(Json::Num(f64::NAN).to_compact(), "null");
        assert_eq!(Json::Num(f64::INFINITY).to_compact(), "null");
    }

    #[test]
    fn rejects_duplicate_object_keys() {
        let error = Json::parse(r#"{"method":"first","method":"second"}"#)
            .expect_err("duplicate object keys must be rejected");

        assert!(
            error.contains("duplicate object key"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_unescaped_control_characters() {
        for input in [
            "{\"value\":\"line\nbreak\"}",
            "{\"value\":\"tab\tcharacter\"}",
            "{\"value\":\"carriage\rreturn\"}",
            "{\"value\":\"\u{0001}\"}",
        ] {
            let error = Json::parse(input).expect_err("raw control characters must be rejected");

            assert!(
                error.contains("unescaped control character"),
                "unexpected error for {input:?}: {error}"
            );
        }

        let escaped = Json::parse(r#"{"value":"line\nbreak\tand\u0001"}"#)
            .expect("escaped control characters are valid JSON");

        assert_eq!(
            escaped.get("value").and_then(Json::as_str),
            Some("line\nbreak\tand\u{0001}")
        );
    }

    #[test]
    fn enforces_strict_json_number_grammar() {
        for valid in [
            "0", "-0", "1", "-1", "0.5", "-12.75", "1e3", "1E+3", "-2.5e-2",
        ] {
            assert!(
                Json::parse(valid).is_ok(),
                "valid JSON number rejected: {valid}"
            );
        }

        for invalid in ["01", "-01", "1.", ".1", "+1", "1e", "1e+", "--1", "1e9999"] {
            assert!(
                Json::parse(invalid).is_err(),
                "invalid JSON number accepted: {invalid}"
            );
        }
    }

    #[test]
    fn enforces_input_size_limit() {
        let oversized = " ".repeat(MAX_INPUT_BYTES + 1);

        let error = Json::parse(&oversized).expect_err("oversized JSON input must be rejected");

        assert!(
            error.contains("JSON input too large"),
            "unexpected error: {error}"
        );

        assert!(
            Json::parse_with_limit("{}", 2).is_ok(),
            "explicit limit should accept a document at the limit"
        );
        assert!(
            Json::parse_with_limit("{}", 1).is_err(),
            "explicit limit should reject an oversized document"
        );
    }

    #[test]
    fn nesting_boundary_is_exact() {
        let accepted = "[".repeat(MAX_DEPTH) + "0" + &"]".repeat(MAX_DEPTH);
        Json::parse(&accepted).expect("MAX_DEPTH nesting should be accepted");

        let rejected = "[".repeat(MAX_DEPTH + 1) + "0" + &"]".repeat(MAX_DEPTH + 1);

        assert_eq!(Json::parse(&rejected).unwrap_err(), "JSON nesting too deep");
    }

    #[test]
    fn rejects_malformed() {
        assert!(Json::parse("{").is_err());
        assert!(Json::parse("[1,]").is_err());
        assert!(Json::parse("nul").is_err());
        assert!(Json::parse("\"unterminated").is_err());
        assert!(Json::parse("1 2").is_err());
    }

    #[test]
    fn rejects_deep_nesting() {
        let deep = "[".repeat(100) + &"]".repeat(100);
        let r = Json::parse(&deep);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), "JSON nesting too deep");
    }
}
