// SLHA calibration data validation — production integrity gate.
//
// This module is the single source of truth for validating the K-activation
// calibration dumps produced by the collect path before they are handed to the
// projection trainer. It is dependency-light (C++17 standard library only, no
// ggml/llama headers) so the exact same implementation is linked by:
//   * the trainer gate CLI (slha_calibrate_cli.cpp), invoked before any fit;
//   * the production-linked unit tests (tests/calibration_tests.cpp).
//
// Policy (default: reject):
//   * reject   — any non-finite (NaN / +Inf / -Inf) scalar makes the whole
//                calibration invalid; the run must fail. No file is modified.
//   * drop-row — research/recovery mode only: whole rows containing any
//                non-finite scalar are removed (finite scalars are never
//                clamped or imputed); the output is marked sanitized and both
//                raw and clean hashes are recorded.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slha_calib {

// On-disk header magic for a `layer-<id>-k.bin` dump: "SLHA" little-endian.
constexpr uint32_t kCalibMagic = 0x534C4841u;

enum class NonFinitePolicy {
    Reject = 0,
    DropRow = 1,
};

// Parse "reject" / "drop-row"; returns false on an unknown string (out=Reject).
bool parse_policy(const std::string & s, NonFinitePolicy & out);
const char * policy_name(NonFinitePolicy p);

// Structural status of a single dump file.
enum class FileStatus {
    Ok = 0,
    NotFound,
    TruncatedHeader,   // fewer than 12 header bytes
    BadMagic,
    ZeroCols,          // cols == 0
    EmptyData,         // rows == 0
    SizeMismatch,      // byte size != 12 + rows*cols*4 (truncated / trailing)
    DimMismatch,       // cols != expected dimension
};
const char * file_status_name(FileStatus s);

// Per-row non-finite scan of a loaded row-major [rows x cols] matrix.
struct RowScan {
    uint64_t rows = 0;
    uint32_t cols = 0;
    uint64_t finite_rows = 0;
    uint64_t nan_rows = 0;      // rows containing at least one NaN
    uint64_t posinf_rows = 0;   // rows containing at least one +Inf
    uint64_t neginf_rows = 0;   // rows containing at least one -Inf
    uint64_t nonfinite_rows = 0;
    uint64_t nonfinite_scalars = 0;
    std::vector<uint64_t> nonfinite_row_indices;  // sorted, ascending
};

// Report for one layer file after reading + policy application.
struct FileReport {
    int32_t layer_id = -1;
    std::string path;
    FileStatus status = FileStatus::Ok;
    uint32_t magic = 0;
    RowScan scan;
    uint64_t rows_observed = 0;
    uint64_t rows_accepted = 0;
    uint64_t rows_rejected = 0;
    std::string raw_sha256;
    std::string clean_sha256;  // equals raw_sha256 unless the file was rewritten
    bool rewritten = false;    // true only when drop-row removed >=1 row
};

// SHA-256 (lowercase hex). Standalone, std-only.
std::string sha256_hex(const uint8_t * data, size_t len);
bool sha256_file(const std::string & path, std::string & out_hex);

// Read + structurally validate a dump file. On Ok, `data` holds rows*cols
// floats (row-major) and rows/cols are set. On any non-Ok status `data` is
// cleared and `err` carries a human-readable reason. `expected_cols == 0`
// disables the dimension check.
FileStatus read_calib_file(const std::string & path,
                           uint32_t expected_cols,
                           uint32_t & magic,
                           uint32_t & rows,
                           uint32_t & cols,
                           std::vector<float> & data,
                           std::string & err);

// Scan a loaded matrix for non-finite values.
RowScan scan_matrix(const std::vector<float> & data, uint64_t rows, uint32_t cols);

// Produce a copy with every row that contains a non-finite scalar removed.
// Finite scalar values are copied byte-for-byte (never clamped/imputed).
void drop_nonfinite_rows(const std::vector<float> & data,
                         uint64_t rows,
                         uint32_t cols,
                         const RowScan & scan,
                         std::vector<float> & out);

// Atomically write a dump file (write `<path>.tmp`, fsync-less flush, rename).
bool write_calib_file(const std::string & path,
                      uint32_t magic,
                      uint32_t cols,
                      const std::vector<float> & data,
                      std::string & err);

struct LayerFile {
    int32_t layer_id;
    std::string path;
};

// Discover `layer-<id>-k.bin` files in `dir`, sorted ascending by id.
// Any id seen more than once is appended to `duplicates_out`.
std::vector<LayerFile> discover_layer_files(const std::string & dir,
                                            std::vector<int32_t> & duplicates_out);

// Provenance fields recorded verbatim in the manifest (caller-supplied).
struct ManifestProvenance {
    int format_version = 1;
    std::string implementation_commit;
    std::string llama_cpp_commit;
    std::string model_identifier;
    std::string model_sha256;
    std::string dataset_sha256;
    std::string collection_command;
    std::string timestamp_utc;
    std::string codec;
    uint64_t min_rows = 0;
};

// Aggregate result over a calibration directory.
struct DirReport {
    std::string dir;
    std::string policy;  // "reject" | "drop-row"
    std::vector<FileReport> files;
    uint32_t expected_dim = 0;
    uint32_t observed_dim = 0;
    int expected_layers = -1;  // -1 = not enforced
    std::vector<int32_t> missing_layers;
    std::vector<int32_t> duplicate_layers;
    uint64_t total_rows_observed = 0;
    uint64_t total_rows_accepted = 0;
    uint64_t total_rows_rejected = 0;
    uint64_t total_nan_rows = 0;
    uint64_t total_posinf_rows = 0;
    uint64_t total_neginf_rows = 0;
    uint64_t total_nonfinite_scalars = 0;
    bool dim_consistent = true;
    bool sanitized = false;  // true only if drop-row actually removed rows
    bool valid = false;      // final production gate
    std::string error;       // first hard structural error, if any
};

// Main production entry point. Discovers, structurally validates, non-finite
// scans, and applies the policy across a directory.
//   reject   — no file modified; valid iff no structural error and zero
//              non-finite rows and (if enforced) all expected layers present.
//   drop-row — rewrites files removing non-finite rows (atomic), sets
//              sanitized=true when any row removed; valid iff no structural
//              error, all expected layers present, and every remaining file
//              has >= min_rows.
// Does not write the manifest; caller serializes via build_manifest_json.
DirReport validate_dir(const std::string & dir,
                       NonFinitePolicy policy,
                       int expected_layers,
                       uint32_t expected_dim,
                       uint64_t min_rows);

// Serialize a manifest (pure; no I/O).
std::string build_manifest_json(const DirReport & rep, const ManifestProvenance & prov);

// Write a string to a path atomically (`<path>.tmp` -> rename).
bool write_file_atomic(const std::string & path, const std::string & content, std::string & err);

}  // namespace slha_calib
