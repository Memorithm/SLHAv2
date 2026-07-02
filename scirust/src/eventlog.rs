//! CCOS EventLog — append-only, deterministic on-disk snapshot of evicted tiles.
//!
//! The [`crate::ccos::ElasticKvCache`] doc long noted that COLD eviction is
//! in-memory recycling only, and that "a real CCOS would snapshot it to the
//! EventLog". This module is that EventLog: a std-only (zero external
//! dependency), append-only binary log that persists every evicted 128-byte
//! tile so it can be [rehydrated](crate::ccos::ElasticKvCache::rehydrate) later.
//!
//! ## Format (little-endian throughout)
//! ```text
//! Header (16 bytes):
//!   [u32 magic = 0x534C4C47 ("SLLG")]
//!   [u32 version = 1]
//!   [u32 tile_size = 128]   // sanity: must match the compiled tile
//!   [u32 reserved = 0]
//! Record (144 bytes), repeated:
//!   [u64 seq]               // the slot's insertion sequence at eviction
//!   [u32 slot]              // the arena slot it was evicted from
//!   [u32 reserved = 0]
//!   [128 bytes tile]        // see `tile_to_bytes`
//! ```
//!
//! ## Determinism
//! The format contains **no timestamps and no randomness**: the same sequence
//! of `append` calls always produces a **byte-identical** file. This is a
//! deliberate invariant (fixed-seed reproducibility, like the rest of the
//! crate) and is asserted by the tests.
//!
//! ## Tile serialization
//! [`tile_to_bytes`] / [`tile_from_bytes`] write each field explicitly in
//! little-endian order — no `transmute`, no reliance on struct padding — so the
//! log is portable across targets and the round trip is exact.

use crate::attention::slha_v2::{SciRustSlhaTile, LATENT_BYTES, N_GROUPS, RESIDUAL_WORDS};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic bytes at the start of an EventLog file: "SLLG".
const MAGIC: u32 = 0x534C_4C47;
/// On-disk format version.
const VERSION: u32 = 1;
/// Serialized size of one tile, in bytes (the 128-byte tile invariant).
pub const TILE_BYTES: usize = 128;
/// Size of the file header, in bytes.
pub const HEADER_BYTES: usize = 16;
/// Size of one log record (seq + slot + reserved + tile), in bytes.
pub const RECORD_BYTES: usize = 8 + 4 + 4 + TILE_BYTES; // 144

// The serialized tile must be exactly the 128-byte tile invariant. This
// mirrors the field byte-map of `SciRustSlhaTile`.
const _: () =
    assert!(LATENT_BYTES + RESIDUAL_WORDS * 8 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + N_GROUPS == TILE_BYTES);

/// One decoded log entry: the eviction sequence, the source slot, and the tile.
/// (`SciRustSlhaTile` is not `Debug`, so neither is this.)
#[derive(Clone, Copy)]
pub struct LogRecord {
    /// The slot's insertion sequence number at the moment of eviction.
    pub seq: u64,
    /// The arena slot the tile was evicted from.
    pub slot: u32,
    /// The evicted tile itself.
    pub tile: SciRustSlhaTile,
}

/// Serialize a tile to its 128 canonical little-endian bytes.
///
/// Field order matches the struct's byte-map: `latent_kv` (64) ‖
/// `residual_bitmap` (32) ‖ `scale` ‖ `dynamic_lambda` ‖ `residual_sigma` ‖
/// `token_id` ‖ `position` ‖ `head_id` ‖ `flags` ‖ `group_scales` (8).
#[must_use]
pub fn tile_to_bytes(tile: &SciRustSlhaTile) -> [u8; TILE_BYTES] {
    let mut out = [0u8; TILE_BYTES];
    let mut o = 0;
    let mut put = |bytes: &[u8], o: &mut usize| {
        out[*o..*o + bytes.len()].copy_from_slice(bytes);
        *o += bytes.len();
    };
    put(&tile.latent_kv, &mut o);
    for w in &tile.residual_bitmap {
        put(&w.to_le_bytes(), &mut o);
    }
    put(&tile.scale.to_le_bytes(), &mut o);
    put(&tile.dynamic_lambda.to_le_bytes(), &mut o);
    put(&tile.residual_sigma.to_le_bytes(), &mut o);
    put(&tile.token_id.to_le_bytes(), &mut o);
    put(&tile.position.to_le_bytes(), &mut o);
    put(&tile.head_id.to_le_bytes(), &mut o);
    put(&tile.flags.to_le_bytes(), &mut o);
    put(&tile.group_scales, &mut o);
    debug_assert_eq!(o, TILE_BYTES);
    out
}

/// Deserialize a tile from its 128 canonical little-endian bytes (inverse of
/// [`tile_to_bytes`]).
#[must_use]
pub fn tile_from_bytes(b: &[u8; TILE_BYTES]) -> SciRustSlhaTile {
    let mut o = 0;
    let take = |n: usize, o: &mut usize| {
        let s = &b[*o..*o + n];
        *o += n;
        s
    };
    let mut latent_kv = [0u8; LATENT_BYTES];
    latent_kv.copy_from_slice(take(LATENT_BYTES, &mut o));
    let mut residual_bitmap = [0u64; RESIDUAL_WORDS];
    for w in &mut residual_bitmap {
        *w = u64::from_le_bytes(take(8, &mut o).try_into().unwrap());
    }
    let f32_at = |o: &mut usize| f32::from_le_bytes(take(4, o).try_into().unwrap());
    let scale = f32_at(&mut o);
    let dynamic_lambda = f32_at(&mut o);
    let residual_sigma = f32_at(&mut o);
    let token_id = u32::from_le_bytes(take(4, &mut o).try_into().unwrap());
    let position = u32::from_le_bytes(take(4, &mut o).try_into().unwrap());
    let head_id = u16::from_le_bytes(take(2, &mut o).try_into().unwrap());
    let flags = u16::from_le_bytes(take(2, &mut o).try_into().unwrap());
    let mut group_scales = [0u8; N_GROUPS];
    group_scales.copy_from_slice(take(N_GROUPS, &mut o));
    SciRustSlhaTile {
        latent_kv,
        residual_bitmap,
        scale,
        dynamic_lambda,
        residual_sigma,
        token_id,
        position,
        head_id,
        flags,
        group_scales,
    }
}

/// Append-only log of evicted tiles.
///
/// Holds an open file handle positioned at the end of the log. Reads scan the
/// file from the first record; at CCOS scale (thousands of tiles) the linear
/// scans in [`Self::fetch_by_seq`] / [`Self::fetch_last_for_slot`] are
/// negligible, and are documented as O(records).
pub struct EventLog {
    file: File,
    records: u64,
}

impl EventLog {
    /// Create a fresh log at `path` (truncating any existing file) and write
    /// the header.
    ///
    /// # Errors
    /// Returns any I/O error from opening or writing the file.
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut header = [0u8; HEADER_BYTES];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&(TILE_BYTES as u32).to_le_bytes());
        // header[12..16] reserved = 0.
        file.write_all(&header)?;
        Ok(Self { file, records: 0 })
    }

    /// Open an existing log at `path`, validating the header.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if the magic, version, or tile
    /// size do not match, or if the body is not a whole number of records;
    /// otherwise any underlying I/O error.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.seek(SeekFrom::End(0))?;
        if len < HEADER_BYTES as u64 {
            return Err(invalid("EventLog: file shorter than header"));
        }
        file.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        let at = |o: usize| u32::from_le_bytes(header[o..o + 4].try_into().unwrap());
        if at(0) != MAGIC {
            return Err(invalid("EventLog: bad magic"));
        }
        if at(4) != VERSION {
            return Err(invalid("EventLog: unsupported version"));
        }
        if at(8) != TILE_BYTES as u32 {
            return Err(invalid("EventLog: tile-size mismatch"));
        }
        let body = len - HEADER_BYTES as u64;
        if !body.is_multiple_of(RECORD_BYTES as u64) {
            return Err(invalid("EventLog: truncated record"));
        }
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            records: body / RECORD_BYTES as u64,
        })
    }

    /// Append one `(seq, slot, tile)` record at the end of the log.
    ///
    /// # Errors
    /// Returns any I/O error from writing; on failure the record count is not
    /// incremented (the log stays consistent).
    pub fn append(&mut self, seq: u64, slot: u32, tile: &SciRustSlhaTile) -> io::Result<()> {
        self.file.seek(SeekFrom::End(0))?;
        let mut rec = [0u8; RECORD_BYTES];
        rec[0..8].copy_from_slice(&seq.to_le_bytes());
        rec[8..12].copy_from_slice(&slot.to_le_bytes());
        // rec[12..16] reserved = 0.
        rec[16..16 + TILE_BYTES].copy_from_slice(&tile_to_bytes(tile));
        self.file.write_all(&rec)?;
        self.records += 1;
        Ok(())
    }

    /// Number of records currently in the log.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.records
    }

    /// True if the log holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records == 0
    }

    /// Flush and fsync the log to durable storage. Explicit — the log does not
    /// fsync on every append.
    ///
    /// # Errors
    /// Returns any I/O error from flushing or syncing.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_all()
    }

    /// Read every record, in append order.
    ///
    /// # Errors
    /// Returns any I/O error, or [`io::ErrorKind::InvalidData`] if the file was
    /// truncated since it was opened.
    pub fn read_all(&mut self) -> io::Result<Vec<LogRecord>> {
        self.file.seek(SeekFrom::Start(HEADER_BYTES as u64))?;
        let mut out = Vec::with_capacity(usize::try_from(self.records).unwrap_or(0));
        let mut rec = [0u8; RECORD_BYTES];
        for _ in 0..self.records {
            self.file.read_exact(&mut rec)?;
            let seq = u64::from_le_bytes(rec[0..8].try_into().unwrap());
            let slot = u32::from_le_bytes(rec[8..12].try_into().unwrap());
            let mut tile_bytes = [0u8; TILE_BYTES];
            tile_bytes.copy_from_slice(&rec[16..16 + TILE_BYTES]);
            out.push(LogRecord {
                seq,
                slot,
                tile: tile_from_bytes(&tile_bytes),
            });
        }
        self.file.seek(SeekFrom::End(0))?;
        Ok(out)
    }

    /// The most recently appended record for `seq`, if any. O(records).
    ///
    /// # Errors
    /// Propagates I/O errors from [`Self::read_all`].
    pub fn fetch_by_seq(&mut self, seq: u64) -> io::Result<Option<LogRecord>> {
        Ok(self.read_all()?.into_iter().rev().find(|r| r.seq == seq))
    }

    /// The most recently appended record for arena `slot`, if any. O(records).
    ///
    /// # Errors
    /// Propagates I/O errors from [`Self::read_all`].
    pub fn fetch_last_for_slot(&mut self, slot: u32) -> io::Result<Option<LogRecord>> {
        Ok(self.read_all()?.into_iter().rev().find(|r| r.slot == slot))
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::slha_v2::FLAG_TQ3;

    fn sample_tile(seed: u8) -> SciRustSlhaTile {
        let mut latent_kv = [0u8; LATENT_BYTES];
        for (i, b) in latent_kv.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(seed).wrapping_add(3);
        }
        SciRustSlhaTile {
            latent_kv,
            residual_bitmap: [0x0123_4567_89AB_CDEF ^ u64::from(seed), 1, 2, 3],
            scale: 0.5 * f32::from(seed) + 0.125,
            dynamic_lambda: 0.37,
            residual_sigma: 1.5,
            token_id: 42 + u32::from(seed),
            position: 7 + u32::from(seed),
            head_id: 2,
            flags: FLAG_TQ3,
            group_scales: [seed, 254, 3, 4, 5, 6, 7, 8],
        }
    }

    fn tiles_eq(a: &SciRustSlhaTile, b: &SciRustSlhaTile) -> bool {
        tile_to_bytes(a) == tile_to_bytes(b)
    }

    #[test]
    fn tile_bytes_round_trip_is_exact() {
        for s in 0..16u8 {
            let t = sample_tile(s);
            let bytes = tile_to_bytes(&t);
            let back = tile_from_bytes(&bytes);
            assert!(tiles_eq(&t, &back), "seed {s}: tile round trip differs");
            // Bit-exact on the float fields specifically.
            assert_eq!(t.scale.to_bits(), back.scale.to_bits());
            assert_eq!(back.flags, FLAG_TQ3);
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scirust_eventlog_{}_{name}", std::process::id()));
        p
    }

    #[test]
    fn append_read_reopen_round_trip() {
        let path = tmp_path("rw");
        let _ = std::fs::remove_file(&path);
        {
            let mut log = EventLog::create(&path).unwrap();
            for s in 0..5u8 {
                log.append(u64::from(s) * 10, u32::from(s), &sample_tile(s))
                    .unwrap();
            }
            assert_eq!(log.len(), 5);
            log.sync().unwrap();
        }
        // Reopen and verify every record.
        let mut log = EventLog::open(&path).unwrap();
        assert_eq!(log.len(), 5);
        let recs = log.read_all().unwrap();
        assert_eq!(recs.len(), 5);
        for (s, r) in recs.iter().enumerate() {
            assert_eq!(r.seq, s as u64 * 10);
            assert_eq!(r.slot, s as u32);
            assert!(tiles_eq(&r.tile, &sample_tile(s as u8)));
        }
        // Appends continue after reopen.
        log.append(999, 9, &sample_tile(9)).unwrap();
        assert_eq!(log.len(), 6);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn identical_runs_are_byte_identical() {
        let p1 = tmp_path("det1");
        let p2 = tmp_path("det2");
        for p in [&p1, &p2] {
            let _ = std::fs::remove_file(p);
            let mut log = EventLog::create(p).unwrap();
            for s in 0..7u8 {
                log.append(u64::from(s), u32::from(s) * 2, &sample_tile(s))
                    .unwrap();
            }
            log.sync().unwrap();
        }
        let a = std::fs::read(&p1).unwrap();
        let b = std::fs::read(&p2).unwrap();
        assert_eq!(
            a, b,
            "identical append sequences must produce identical files"
        );
        assert_eq!(a.len(), HEADER_BYTES + 7 * RECORD_BYTES);
        std::fs::remove_file(&p1).unwrap();
        std::fs::remove_file(&p2).unwrap();
    }

    #[test]
    fn fetch_helpers_find_latest() {
        let path = tmp_path("fetch");
        let _ = std::fs::remove_file(&path);
        let mut log = EventLog::create(&path).unwrap();
        log.append(100, 0, &sample_tile(1)).unwrap();
        log.append(100, 0, &sample_tile(2)).unwrap(); // same seq/slot, newer
        log.append(200, 1, &sample_tile(3)).unwrap();
        let by_seq = log.fetch_by_seq(100).unwrap().unwrap();
        assert!(
            tiles_eq(&by_seq.tile, &sample_tile(2)),
            "must return the latest for seq"
        );
        let by_slot = log.fetch_last_for_slot(0).unwrap().unwrap();
        assert!(tiles_eq(&by_slot.tile, &sample_tile(2)));
        assert!(log.fetch_by_seq(999).unwrap().is_none());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_bad_header_and_truncation() {
        // Bad magic.
        let path = tmp_path("badmagic");
        std::fs::write(&path, [0u8; HEADER_BYTES]).unwrap();
        assert!(EventLog::open(&path).is_err());
        // Valid header but a truncated (partial) record body.
        let path2 = tmp_path("trunc");
        {
            let mut log = EventLog::create(&path2).unwrap();
            log.append(1, 0, &sample_tile(1)).unwrap();
        }
        // Chop 10 bytes off the single record.
        let mut bytes = std::fs::read(&path2).unwrap();
        bytes.truncate(bytes.len() - 10);
        std::fs::write(&path2, &bytes).unwrap();
        assert!(
            EventLog::open(&path2).is_err(),
            "partial record must be rejected"
        );
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&path2).unwrap();
    }
}
