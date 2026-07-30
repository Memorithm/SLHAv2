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
//!   [u32 version = 2]
//!   [u32 tile_size = 128]   // sanity: must match the compiled tile
//!   [u32 reserved = 0]
//! Record (144 bytes), repeated:
//!   [u64 seq]               // the slot's insertion sequence at eviction
//!   [u32 slot]              // the arena slot it was evicted from
//!   [u32 checksum]          // FNV-1a(seq || slot || tile), version 2
//!   [128 bytes tile]        // see `tile_to_bytes`
//!
//! Version 1 used zero in place of `checksum`. It remains readable and
//! appendable for backward compatibility.
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
/// Legacy format without per-record checksums.
const VERSION_V1: u32 = 1;
/// Current format with a checksum in every record.
const VERSION: u32 = 2;
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
    // Infallible conversions: `take(n, ..)` returns exactly n bytes by
    // construction, so `try_into()` to [u8; n] cannot fail. This holds for
    // every unwrap in this decoder.
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

/// Deterministic zero-dependency checksum for one version-2 record.
///
/// FNV-1a is not cryptographic. Its role is to detect accidental corruption,
/// torn writes and unexpected byte modifications.
fn record_checksum(seq: u64, slot: u32, tile: &[u8; TILE_BYTES]) -> u32 {
    const OFFSET: u32 = 0x811C_9DC5;
    const PRIME: u32 = 0x0100_0193;

    let mut hash = OFFSET;

    for byte in seq.to_le_bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }

    for byte in slot.to_le_bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }

    for &byte in tile {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }

    hash
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
    version: u32,
    writable: bool,
}

impl EventLog {
    /// Create a fresh log at `path`, truncating an existing file.
    ///
    /// Prefer [`Self::create_new`] when overwriting an existing log would be a
    /// data-loss bug.
    ///
    /// # Errors
    /// Returns any I/O error from opening or writing the file.
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::create_impl(path, false)
    }

    /// Create a log only if `path` does not already exist.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::AlreadyExists`] rather than truncating an
    /// existing journal.
    pub fn create_new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::create_impl(path, true)
    }

    fn create_impl<P: AsRef<Path>>(path: P, exclusive: bool) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);

        if exclusive {
            options.create_new(true);
        } else {
            options.create(true).truncate(true);
        }

        let mut file = options.open(path)?;
        let mut header = [0u8; HEADER_BYTES];

        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&(TILE_BYTES as u32).to_le_bytes());
        // header[12..16] reserved = 0.

        file.write_all(&header)?;

        Ok(Self {
            file,
            records: 0,
            version: VERSION,
            writable: true,
        })
    }

    /// Open an existing log strictly, validating its header, complete-record
    /// framing and all version-2 checksums.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if the file is truncated,
    /// corrupted or uses an unsupported format.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::open_impl(path, false, true)
    }

    /// Open an existing log and recover from one partial record at the tail.
    ///
    /// Only bytes after the final complete record are removed. Header errors,
    /// checksum failures and corruption of complete records remain fatal.
    pub fn open_recover<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::open_impl(path, true, true)
    }

    /// Open and validate an existing log in read-only mode.
    ///
    /// Reading and lookup remain available, but [`Self::append`] and
    /// [`Self::append_durable`] return [`io::ErrorKind::PermissionDenied`].
    pub fn open_read_only<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::open_impl(path, false, false)
    }

    fn open_impl<P: AsRef<Path>>(path: P, recover_tail: bool, writable: bool) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(writable);

        let mut file = options.open(path)?;
        let mut len = file.seek(SeekFrom::End(0))?;

        if len < HEADER_BYTES as u64 {
            return Err(invalid("EventLog: file shorter than header"));
        }

        file.seek(SeekFrom::Start(0))?;

        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header)?;

        // Infallible: `header[o..o + 4]` is exactly 4 bytes, so the conversion
        // to [u8; 4] cannot fail.
        let at = |o: usize| u32::from_le_bytes(header[o..o + 4].try_into().unwrap());

        if at(0) != MAGIC {
            return Err(invalid("EventLog: bad magic"));
        }

        let version = at(4);

        if version != VERSION_V1 && version != VERSION {
            return Err(invalid("EventLog: unsupported version"));
        }

        if at(8) != TILE_BYTES as u32 {
            return Err(invalid("EventLog: tile-size mismatch"));
        }

        let body = len - HEADER_BYTES as u64;
        let remainder = body % RECORD_BYTES as u64;

        if remainder != 0 {
            if !recover_tail {
                return Err(invalid("EventLog: truncated record"));
            }

            len -= remainder;
            file.set_len(len)?;
            file.sync_all()?;
        }

        let records = (len - HEADER_BYTES as u64) / RECORD_BYTES as u64;

        let mut log = Self {
            file,
            records,
            version,
            writable,
        };

        log.validate_records()?;
        log.file.seek(SeekFrom::End(0))?;

        Ok(log)
    }

    /// Append one `(seq, slot, tile)` record at the end of the log.
    ///
    /// Version-2 records carry a deterministic checksum. Version-1 logs remain
    /// appendable and continue writing zero in the legacy reserved field.
    ///
    /// # Errors
    /// On a normal write failure, the implementation attempts to restore the
    /// previous file length before returning the I/O error.
    pub fn append(&mut self, seq: u64, slot: u32, tile: &SciRustSlhaTile) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "EventLog: log opened read-only",
            ));
        }

        let start = self.file.seek(SeekFrom::End(0))?;
        let tile_bytes = tile_to_bytes(tile);

        let mut rec = [0u8; RECORD_BYTES];
        rec[0..8].copy_from_slice(&seq.to_le_bytes());
        rec[8..12].copy_from_slice(&slot.to_le_bytes());

        if self.version == VERSION {
            let checksum = record_checksum(seq, slot, &tile_bytes);
            rec[12..16].copy_from_slice(&checksum.to_le_bytes());
        }

        rec[16..16 + TILE_BYTES].copy_from_slice(&tile_bytes);

        if let Err(error) = self.file.write_all(&rec) {
            let _ = self.file.set_len(start);
            let _ = self.file.seek(SeekFrom::End(0));
            return Err(error);
        }

        self.records += 1;
        Ok(())
    }

    /// Append a record and synchronize it to durable storage before returning.
    ///
    /// If synchronization fails, the implementation attempts to remove the
    /// newly appended record and restore the previous record count.
    pub fn append_durable(
        &mut self,
        seq: u64,
        slot: u32,
        tile: &SciRustSlhaTile,
    ) -> io::Result<()> {
        let start = self.file.seek(SeekFrom::End(0))?;
        let previous_records = self.records;

        self.append(seq, slot, tile)?;

        if let Err(sync_error) = self.sync() {
            if self.file.set_len(start).is_ok() {
                self.records = previous_records;
                let _ = self.file.seek(SeekFrom::End(0));
                let _ = self.file.sync_all();
            }

            return Err(sync_error);
        }

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

    fn decode_record(&self, rec: &[u8; RECORD_BYTES]) -> io::Result<LogRecord> {
        // Infallible: fixed-width subslices of a fixed-size record buffer —
        // 8 and 4 bytes exactly — so `try_into()` cannot fail.
        let seq = u64::from_le_bytes(rec[0..8].try_into().unwrap());
        let slot = u32::from_le_bytes(rec[8..12].try_into().unwrap());

        let mut tile_bytes = [0u8; TILE_BYTES];
        tile_bytes.copy_from_slice(&rec[16..16 + TILE_BYTES]);

        if self.version == VERSION {
            // Infallible: rec[12..16] is exactly 4 bytes.
            let stored = u32::from_le_bytes(rec[12..16].try_into().unwrap());
            let expected = record_checksum(seq, slot, &tile_bytes);

            if stored != expected {
                return Err(invalid("EventLog: record checksum mismatch"));
            }
        }

        Ok(LogRecord {
            seq,
            slot,
            tile: tile_from_bytes(&tile_bytes),
        })
    }

    fn validate_records(&mut self) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(HEADER_BYTES as u64))?;
        let mut rec = [0u8; RECORD_BYTES];

        for _ in 0..self.records {
            self.file.read_exact(&mut rec)?;
            self.decode_record(&rec)?;
        }

        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }

    /// Read every record, in append order.
    ///
    /// # Errors
    /// Returns any I/O error or [`io::ErrorKind::InvalidData`] for truncation
    /// or a version-2 checksum mismatch.
    pub fn read_all(&mut self) -> io::Result<Vec<LogRecord>> {
        self.file.seek(SeekFrom::Start(HEADER_BYTES as u64))?;

        let mut out = Vec::with_capacity(usize::try_from(self.records).unwrap_or(0));
        let mut rec = [0u8; RECORD_BYTES];

        for _ in 0..self.records {
            self.file.read_exact(&mut rec)?;
            out.push(self.decode_record(&rec)?);
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
    fn append_durable_round_trips_after_reopen() {
        let path = tmp_path("durable");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = EventLog::create(&path).unwrap();

            log.append_durable(77, 4, &sample_tile(7)).unwrap();

            assert_eq!(log.len(), 1);
        }

        let mut reopened = EventLog::open(&path).unwrap();
        let records = reopened.read_all().unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 77);
        assert_eq!(records[0].slot, 4);
        assert!(tiles_eq(&records[0].tile, &sample_tile(7)));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn read_only_log_allows_reads_but_rejects_appends() {
        let path = tmp_path("readonly");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = EventLog::create(&path).unwrap();
            log.append_durable(1, 2, &sample_tile(3)).unwrap();
        }

        let mut read_only = EventLog::open_read_only(&path).unwrap();

        assert_eq!(read_only.read_all().unwrap().len(), 1);

        let error = read_only
            .append(2, 3, &sample_tile(4))
            .expect_err("read-only EventLog must reject append");

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(read_only.len(), 1);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn create_new_refuses_to_overwrite_existing_log() {
        let path = tmp_path("create_new");
        let _ = std::fs::remove_file(&path);

        let log = EventLog::create_new(&path).unwrap();
        drop(log);

        let error = match EventLog::create_new(&path) {
            Ok(_) => panic!("create_new must not overwrite an existing log"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn version_two_detects_record_corruption() {
        let path = tmp_path("checksum");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = EventLog::create(&path).unwrap();
            log.append(7, 3, &sample_tile(9)).unwrap();
            log.sync().unwrap();
        }

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_BYTES + 16 + 5] ^= 0x80;
        std::fs::write(&path, bytes).unwrap();

        let error = match EventLog::open(&path) {
            Ok(_) => panic!("corrupted record must not open successfully"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            error.to_string().contains("checksum"),
            "unexpected error: {error}"
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn open_recover_discards_only_the_partial_tail() {
        let path = tmp_path("recover");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = EventLog::create(&path).unwrap();
            log.append(10, 1, &sample_tile(1)).unwrap();
            log.append(20, 2, &sample_tile(2)).unwrap();
            log.sync().unwrap();
        }

        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 17);
        std::fs::write(&path, bytes).unwrap();

        assert!(
            EventLog::open(&path).is_err(),
            "strict open must reject a partial tail"
        );

        let mut recovered = EventLog::open_recover(&path).unwrap();

        assert_eq!(recovered.len(), 1);
        let records = recovered.read_all().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 10);
        assert!(tiles_eq(&records[0].tile, &sample_tile(1)));

        recovered.append(30, 3, &sample_tile(3)).unwrap();
        recovered.sync().unwrap();
        drop(recovered);

        let mut reopened = EventLog::open(&path).unwrap();
        assert_eq!(reopened.len(), 2);

        let records = reopened.read_all().unwrap();
        assert_eq!(records[0].seq, 10);
        assert_eq!(records[1].seq, 30);

        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            (HEADER_BYTES + 2 * RECORD_BYTES) as u64
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn legacy_version_one_remains_readable_and_appendable() {
        let path = tmp_path("legacy_v1");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = EventLog::create(&path).unwrap();
            log.append(1, 4, &sample_tile(4)).unwrap();
            log.sync().unwrap();
        }

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&VERSION_V1.to_le_bytes());
        bytes[HEADER_BYTES + 12..HEADER_BYTES + 16].fill(0);
        std::fs::write(&path, bytes).unwrap();

        let mut legacy = EventLog::open(&path).unwrap();
        assert_eq!(legacy.len(), 1);

        let records = legacy.read_all().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 1);
        assert!(tiles_eq(&records[0].tile, &sample_tile(4)));

        legacy.append(2, 5, &sample_tile(5)).unwrap();
        legacy.sync().unwrap();
        drop(legacy);

        let mut reopened = EventLog::open(&path).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.read_all().unwrap().len(), 2);

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
