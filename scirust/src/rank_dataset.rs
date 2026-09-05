//! Strict reader for the versioned offline ranking dataset emitted by the
//! llama.cpp diagnostic collector.
//!
//! This is research/evaluation plumbing. Baseline logits carried by the dataset
//! are labels for offline training or validation only and must never be exposed
//! to the deployable external-K inference path.

use std::fs;

const MAGIC: u32 = 0x534C_4841;
const VERSION: u32 = 2;
const HEADER: usize = 40;
const MAX_DATASET_FILE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug)]
pub struct LayerRows {
    pub layer: u32,
    pub q_dim: usize,
    pub key_dim: usize,
    pub rows: usize,
    pub heads: Vec<i32>,
    pub gqa_groups: Vec<i32>,
    pub tokens: Vec<i64>,
    pub n_visible: Vec<usize>,
    pub chunks: Vec<usize>,
    pub q: Vec<f32>,
    pub baseline: Vec<f32>,
    pub control_scores: Vec<f32>,
    pub keys: Vec<Vec<f32>>,
    starts: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct RowRef<'a> {
    pub head: i32,
    pub gqa_group: i32,
    pub token: i64,
    pub chunk: usize,
    pub q: &'a [f32],
    pub baseline: &'a [f32],
    pub control_scores: &'a [f32],
    pub keys: &'a [f32],
    pub n_visible: usize,
    pub key_dim: usize,
}

impl LayerRows {
    pub fn row(&self, index: usize) -> Result<RowRef<'_>, String> {
        if index >= self.rows {
            return Err(format!(
                "rank dataset: row index {index} outside 0..{}",
                self.rows
            ));
        }
        let chunk = self.chunks[index];
        let n_visible = self.n_visible[index];
        let score_start = self.starts[index];
        let score_end = self.starts[index + 1];
        let key_end = n_visible
            .checked_mul(self.key_dim)
            .ok_or_else(|| "rank dataset: key slice size overflow".to_string())?;
        let q_start = index
            .checked_mul(self.q_dim)
            .ok_or_else(|| "rank dataset: query slice offset overflow".to_string())?;
        let q_end = q_start
            .checked_add(self.q_dim)
            .ok_or_else(|| "rank dataset: query slice end overflow".to_string())?;
        Ok(RowRef {
            head: self.heads[index],
            gqa_group: self.gqa_groups[index],
            token: self.tokens[index],
            chunk,
            q: &self.q[q_start..q_end],
            baseline: &self.baseline[score_start..score_end],
            control_scores: &self.control_scores[score_start..score_end],
            keys: &self.keys[chunk][..key_end],
            n_visible,
            key_dim: self.key_dim,
        })
    }

    pub fn indices_for_chunks<'a>(
        &'a self,
        selected: &'a [usize],
    ) -> impl Iterator<Item = usize> + 'a {
        self.chunks
            .iter()
            .enumerate()
            .filter(move |(_, chunk)| selected.contains(chunk))
            .map(|(index, _)| index)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(n)
            .ok_or_else(|| format!("rank dataset: {what} offset overflow"))?;
        if end > self.bytes.len() {
            return Err(format!(
                "rank dataset: truncated {what} at byte {} (need {n} bytes, file has {})",
                self.offset,
                self.bytes.len()
            ));
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn u64(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8, what)?.try_into().expect("exact u64 width"),
        ))
    }

    fn i32(&mut self, what: &str) -> Result<i32, String> {
        Ok(i32::from_le_bytes(
            self.take(4, what)?.try_into().expect("exact i32 width"),
        ))
    }

    fn i64(&mut self, what: &str) -> Result<i64, String> {
        Ok(i64::from_le_bytes(
            self.take(8, what)?.try_into().expect("exact i64 width"),
        ))
    }

    fn finite_f32_vec(&mut self, count: usize, what: &str) -> Result<Vec<f32>, String> {
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| format!("rank dataset: {what} byte count overflow"))?;
        let raw = self.take(bytes, what)?;
        let (chunks, remainder) = raw.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        let mut out = Vec::with_capacity(count);
        for (index, chunk) in chunks.iter().enumerate() {
            let value = f32::from_le_bytes(*chunk);
            if !value.is_finite() {
                return Err(format!(
                    "rank dataset: non-finite {what} value at element {index}"
                ));
            }
            out.push(value);
        }
        Ok(out)
    }
}

fn usize_from_u64(value: u64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("rank dataset: {what} does not fit usize"))
}

pub fn read_layer(path: &str) -> Result<LayerRows, String> {
    let meta = fs::metadata(path).map_err(|e| format!("rank dataset: cannot stat {path}: {e}"))?;
    if !meta.file_type().is_file() {
        return Err(format!("rank dataset: {path} is not a regular file"));
    }
    if meta.len() > MAX_DATASET_FILE_BYTES {
        return Err(format!(
            "rank dataset: {path} is {} bytes, above the {MAX_DATASET_FILE_BYTES}-byte safety cap",
            meta.len()
        ));
    }
    let raw = fs::read(path).map_err(|e| format!("rank dataset: cannot read {path}: {e}"))?;
    parse_layer(&raw, path)
}

pub fn parse_layer(raw: &[u8], source: &str) -> Result<LayerRows, String> {
    if raw.len() < HEADER {
        return Err(format!("rank dataset: {source}: truncated header"));
    }
    let u32_at = |offset: usize| {
        u32::from_le_bytes(
            raw[offset..offset + 4]
                .try_into()
                .expect("validated header range"),
        )
    };
    let u64_at = |offset: usize| {
        u64::from_le_bytes(
            raw[offset..offset + 8]
                .try_into()
                .expect("validated header range"),
        )
    };

    if u32_at(0) != MAGIC {
        return Err(format!("rank dataset: {source}: bad magic"));
    }
    if u32_at(4) != VERSION {
        return Err(format!(
            "rank dataset: {source}: unsupported version {} (expected {VERSION})",
            u32_at(4)
        ));
    }

    let layer = u32_at(8);
    let q_dim = usize::try_from(u32_at(12))
        .map_err(|_| format!("rank dataset: {source}: q_dim does not fit usize"))?;
    let rows = usize_from_u64(u64_at(16), "row count")?;
    let n_chunks = usize_from_u64(u64_at(24), "chunk count")?;
    let key_dim = usize::try_from(u32_at(32))
        .map_err(|_| format!("rank dataset: {source}: key_dim does not fit usize"))?;

    if q_dim == 0 || key_dim == 0 {
        return Err(format!(
            "rank dataset: {source}: q_dim and key_dim must be positive"
        ));
    }
    if n_chunks == 0 {
        return Err(format!("rank dataset: {source}: no chunks"));
    }

    let mut c = Cursor::new(raw, HEADER);
    let mut key_rows = Vec::with_capacity(n_chunks);
    for index in 0..n_chunks {
        key_rows.push(usize_from_u64(
            c.u64(&format!("key row count for chunk {index}"))?,
            "key row count",
        )?);
    }

    let mut heads = Vec::with_capacity(rows);
    for _ in 0..rows {
        heads.push(c.i32("head ids")?);
    }
    let mut gqa_groups = Vec::with_capacity(rows);
    for _ in 0..rows {
        gqa_groups.push(c.i32("gqa group ids")?);
    }
    let mut tokens = Vec::with_capacity(rows);
    for _ in 0..rows {
        tokens.push(c.i64("token positions")?);
    }

    let mut n_visible = Vec::with_capacity(rows);
    for index in 0..rows {
        let value = c.i32("visible counts")?;
        if value <= 0 {
            return Err(format!(
                "rank dataset: {source}: row {index} has non-positive visible count {value}"
            ));
        }
        n_visible.push(value as usize);
    }

    let mut chunks = Vec::with_capacity(rows);
    for index in 0..rows {
        let value = c.i32("chunk ids")?;
        if value < 0 {
            return Err(format!(
                "rank dataset: {source}: row {index} has negative chunk {value}"
            ));
        }
        let chunk = value as usize;
        if chunk >= n_chunks {
            return Err(format!(
                "rank dataset: {source}: row {index} chunk {chunk} outside 0..{n_chunks}"
            ));
        }
        if n_visible[index] > key_rows[chunk] {
            return Err(format!(
                "rank dataset: {source}: row {index} needs {} visible keys but chunk {chunk} contains {}",
                n_visible[index], key_rows[chunk]
            ));
        }
        chunks.push(chunk);
    }

    let q_count = rows
        .checked_mul(q_dim)
        .ok_or_else(|| format!("rank dataset: {source}: query element count overflow"))?;
    let q = c.finite_f32_vec(q_count, "queries")?;

    let mut starts = Vec::with_capacity(rows + 1);
    let mut total_scores = 0usize;
    for &count in &n_visible {
        starts.push(total_scores);
        total_scores = total_scores
            .checked_add(count)
            .ok_or_else(|| format!("rank dataset: {source}: score count overflow"))?;
    }
    starts.push(total_scores);

    let baseline = c.finite_f32_vec(total_scores, "baseline logits")?;
    let control_scores = c.finite_f32_vec(total_scores, "recorded control scores")?;

    let mut keys = Vec::with_capacity(n_chunks);
    for (chunk, &count) in key_rows.iter().enumerate() {
        let elements = count
            .checked_mul(key_dim)
            .ok_or_else(|| format!("rank dataset: {source}: key element count overflow"))?;
        keys.push(c.finite_f32_vec(elements, &format!("keys for chunk {chunk}"))?);
    }

    if c.offset != raw.len() {
        return Err(format!(
            "rank dataset: {source}: trailing bytes (parsed {}, file {})",
            c.offset,
            raw.len()
        ));
    }

    Ok(LayerRows {
        layer,
        q_dim,
        key_dim,
        rows,
        heads,
        gqa_groups,
        tokens,
        n_visible,
        chunks,
        q,
        baseline,
        control_scores,
        keys,
        starts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let mut out = vec![0u8; HEADER];
        out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        out[4..8].copy_from_slice(&VERSION.to_le_bytes());
        out[8..12].copy_from_slice(&3u32.to_le_bytes());
        out[12..16].copy_from_slice(&3u32.to_le_bytes());
        out[16..24].copy_from_slice(&1u64.to_le_bytes());
        out[24..32].copy_from_slice(&1u64.to_le_bytes());
        out[32..36].copy_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes()); // two keys in chunk 0
        out.extend_from_slice(&1i32.to_le_bytes()); // head
        out.extend_from_slice(&0i32.to_le_bytes()); // gqa
        out.extend_from_slice(&7i64.to_le_bytes()); // token
        out.extend_from_slice(&2i32.to_le_bytes()); // visible
        out.extend_from_slice(&0i32.to_le_bytes()); // chunk
        for value in [1.0f32, 2.0, 3.0] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in [4.0f32, 1.0] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in [3.5f32, 1.5] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_version_two_fixture() {
        let parsed = parse_layer(&fixture(), "fixture").expect("valid fixture");
        assert_eq!(parsed.layer, 3);
        assert_eq!(parsed.rows, 1);
        let row = parsed.row(0).expect("row zero");
        assert_eq!(row.token, 7);
        assert_eq!(row.n_visible, 2);
        assert_eq!(row.baseline, [4.0, 1.0]);
        assert_eq!(row.control_scores, [3.5, 1.5]);
        assert_eq!(row.keys.len(), 6);
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = fixture();
        bytes.push(0);
        let error = parse_layer(&bytes, "fixture").unwrap_err();
        assert!(error.contains("trailing bytes"));
    }

    #[test]
    fn rejects_nonfinite_labels() {
        let mut bytes = fixture();
        // Header + key rows + head + gqa + token + nvis + chunk + q = 40+8+4+4+8+4+4+12.
        let baseline_offset = 84;
        bytes[baseline_offset..baseline_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let error = parse_layer(&bytes, "fixture").unwrap_err();
        assert!(error.contains("non-finite baseline logits"));
    }
}
