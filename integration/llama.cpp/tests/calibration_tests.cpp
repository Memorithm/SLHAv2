// Production-linked tests for the SLHA calibration validator.
//
// These exercise the real implementation in ../shim/slha_calibration.cpp
// (linked, not copied). Temporary fixture files are written under a local
// scratch directory and removed on completion; no binaries are committed.
#include "slha_calibration.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace slha_calib;

static int g_failures = 0;
static int g_total_failures = 0;

#define TEST(name) do { std::printf("  test: %s ... ", name); g_failures = 0; } while (0)
#define CHECK(cond) do { \
    if (!(cond)) { std::printf("FAIL (line %d) ", __LINE__); ++g_failures; ++g_total_failures; } \
} while (0)
#define DONE() do { std::printf(g_failures == 0 ? "ok\n" : "FAILED\n"); } while (0)

static const std::string SCRATCH = "./calib_test_tmp";

static void ensure_scratch() { ::mkdir(SCRATCH.c_str(), 0755); }

// Write a raw dump file with an explicit header (allows crafting bad files).
static void write_raw(const std::string & path, uint32_t magic, uint32_t rows, uint32_t cols,
                      const std::vector<float> & payload) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(&magic), 4);
    out.write(reinterpret_cast<const char *>(&rows), 4);
    out.write(reinterpret_cast<const char *>(&cols), 4);
    out.write(reinterpret_cast<const char *>(payload.data()),
              std::streamsize(payload.size() * sizeof(float)));
}

static std::string layer_path(const std::string & dir, int id) {
    return dir + "/layer-" + std::to_string(id) + "-k.bin";
}

static void rm(const std::string & p) { std::remove(p.c_str()); }

static void clean_dir(const std::string & dir) {
    // Remove any layer files this suite may create.
    for (int id = 0; id < 40; ++id) { rm(layer_path(dir, id)); rm(layer_path(dir, id) + ".tmp"); }
    rm(dir + "/layer-01-k.bin");
    rm(dir + "/manifest.json");
}

// -------------------------------------------------------------------------
static void t_sha256_known_answer() {
    TEST("sha256 known-answer (abc)");
    std::string h = sha256_hex(reinterpret_cast<const uint8_t *>("abc"), 3);
    CHECK(h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::string h0 = sha256_hex(reinterpret_cast<const uint8_t *>(""), 0);
    CHECK(h0 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    DONE();
}

static void t_finite_accepted() {
    TEST("fully finite calibration accepted");
    std::string dir = SCRATCH + "/t1"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    write_raw(layer_path(dir, 0), kCalibMagic, 3, 4, {1,2,3,4, 5,6,7,8, 9,10,11,12});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 4, 0);
    CHECK(r.valid == true);
    CHECK(r.total_rows_rejected == 0);
    CHECK(r.total_rows_observed == 3);
    CHECK(r.total_rows_accepted == 3);
    clean_dir(dir);
    DONE();
}

static void t_nan_rejected() {
    TEST("one NaN scalar rejected");
    std::string dir = SCRATCH + "/t2"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    write_raw(layer_path(dir, 0), kCalibMagic, 3, 4, {1,2,3,4, 5,nan,7,8, 9,10,11,12});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 4, 0);
    CHECK(r.valid == false);
    CHECK(r.total_nan_rows == 1);
    CHECK(r.total_rows_rejected == 1);
    CHECK(r.total_nonfinite_scalars == 1);
    clean_dir(dir);
    DONE();
}

static void t_posinf_rejected() {
    TEST("positive infinity rejected");
    std::string dir = SCRATCH + "/t3"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float pinf = HUGE_VALF;
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, pinf,4});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 2, 0);
    CHECK(r.valid == false);
    CHECK(r.total_posinf_rows == 1);
    CHECK(r.total_neginf_rows == 0);
    clean_dir(dir);
    DONE();
}

static void t_neginf_rejected() {
    TEST("negative infinity rejected");
    std::string dir = SCRATCH + "/t4"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float ninf = -HUGE_VALF;
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, 3,ninf});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 2, 0);
    CHECK(r.valid == false);
    CHECK(r.total_neginf_rows == 1);
    CHECK(r.total_posinf_rows == 0);
    clean_dir(dir);
    DONE();
}

static void t_multiple_nonfinite_exact() {
    TEST("multiple non-finite rows reported exactly");
    std::string dir = SCRATCH + "/t5"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf(""), pinf = HUGE_VALF;
    // rows 1 and 3 non-finite (0-indexed)
    write_raw(layer_path(dir, 0), kCalibMagic, 4, 2, {1,2, nan,4, 5,6, 7,pinf});
    std::vector<float> data; uint32_t m,rw,cl; std::string err;
    FileStatus st = read_calib_file(layer_path(dir, 0), 2, m, rw, cl, data, err);
    CHECK(st == FileStatus::Ok);
    RowScan s = scan_matrix(data, rw, cl);
    CHECK(s.nonfinite_rows == 2);
    CHECK(s.finite_rows == 2);
    CHECK(s.nan_rows == 1);
    CHECK(s.posinf_rows == 1);
    CHECK(s.nonfinite_scalars == 2);
    clean_dir(dir);
    DONE();
}

static void t_nonfinite_index_correct() {
    TEST("non-finite row index recorded correctly");
    std::string dir = SCRATCH + "/t6"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    write_raw(layer_path(dir, 0), kCalibMagic, 5, 2, {1,1, 2,2, nan,3, 4,4, 5,5});
    std::vector<float> data; uint32_t m,rw,cl; std::string err;
    read_calib_file(layer_path(dir, 0), 2, m, rw, cl, data, err);
    RowScan s = scan_matrix(data, rw, cl);
    CHECK(s.nonfinite_row_indices.size() == 1);
    CHECK(s.nonfinite_row_indices[0] == 2);
    clean_dir(dir);
    DONE();
}

static void t_truncated_rejected() {
    TEST("truncated row rejected");
    std::string dir = SCRATCH + "/t7"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    // Header claims 3 rows x 4 cols = 12 floats but only 8 provided.
    std::string p = layer_path(dir, 0);
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    uint32_t magic = kCalibMagic, rows = 3, cols = 4;
    out.write(reinterpret_cast<const char *>(&magic), 4);
    out.write(reinterpret_cast<const char *>(&rows), 4);
    out.write(reinterpret_cast<const char *>(&cols), 4);
    std::vector<float> partial(8, 1.0f);
    out.write(reinterpret_cast<const char *>(partial.data()), std::streamsize(8 * 4));
    out.close();
    uint32_t m,rw,cl; std::vector<float> data; std::string err;
    FileStatus st = read_calib_file(p, 4, m, rw, cl, data, err);
    CHECK(st == FileStatus::SizeMismatch);
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 4, 0);
    CHECK(r.valid == false);
    clean_dir(dir);
    DONE();
}

static void t_dim_mismatch_rejected() {
    TEST("inconsistent dimension rejected");
    std::string dir = SCRATCH + "/t8"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 4, {1,2,3,4, 5,6,7,8});
    write_raw(layer_path(dir, 1), kCalibMagic, 2, 2, {1,2, 3,4});
    // expected_dim 0 (do not enforce a specific value) so the inconsistency,
    // not a fixed expectation, is what fails.
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 2, 0, 0);
    CHECK(r.dim_consistent == false);
    CHECK(r.valid == false);
    // And an explicit expected dim rejects the wrong layer as DimMismatch.
    DirReport r2 = validate_dir(dir, NonFinitePolicy::Reject, 2, 4, 0);
    CHECK(r2.valid == false);
    clean_dir(dir);
    DONE();
}

static void t_empty_rejected() {
    TEST("empty file rejected");
    std::string dir = SCRATCH + "/t9"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    write_raw(layer_path(dir, 0), kCalibMagic, 0, 4, {});
    uint32_t m,rw,cl; std::vector<float> data; std::string err;
    FileStatus st = read_calib_file(layer_path(dir, 0), 4, m, rw, cl, data, err);
    CHECK(st == FileStatus::EmptyData);
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 4, 0);
    CHECK(r.valid == false);
    clean_dir(dir);
    DONE();
}

static void t_missing_layer_rejected() {
    TEST("missing layer rejected");
    std::string dir = SCRATCH + "/t10"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, 3,4});
    write_raw(layer_path(dir, 2), kCalibMagic, 2, 2, {1,2, 3,4});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 3, 2, 0);
    CHECK(r.missing_layers.size() == 1);
    CHECK(r.missing_layers[0] == 1);
    CHECK(r.valid == false);
    clean_dir(dir);
    DONE();
}

static void t_duplicate_layer_rejected() {
    TEST("duplicate layer rejected");
    std::string dir = SCRATCH + "/t11"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    // layer-1 and layer-01 both resolve to id 1.
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, 3,4});
    write_raw(layer_path(dir, 1), kCalibMagic, 2, 2, {1,2, 3,4});
    write_raw(dir + "/layer-01-k.bin", kCalibMagic, 2, 2, {1,2, 3,4});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 2, 2, 0);
    CHECK(r.duplicate_layers.size() == 1);
    CHECK(r.duplicate_layers[0] == 1);
    CHECK(r.valid == false);
    clean_dir(dir);
    DONE();
}

static void t_droprow_removes_only_affected() {
    TEST("drop-row removes only complete affected rows");
    std::string dir = SCRATCH + "/t14"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    // 4 rows x 2 cols, row 1 has a NaN.
    write_raw(layer_path(dir, 0), kCalibMagic, 4, 2, {10,11, nan,99, 12,13, 14,15});
    DirReport r = validate_dir(dir, NonFinitePolicy::DropRow, 1, 2, 1);
    CHECK(r.valid == true);
    CHECK(r.sanitized == true);
    CHECK(r.total_rows_rejected == 1);
    // Reload cleaned file: 3 rows, and the surviving rows are exactly the
    // finite ones, byte-identical.
    uint32_t m,rw,cl; std::vector<float> data; std::string err;
    FileStatus st = read_calib_file(layer_path(dir, 0), 2, m, rw, cl, data, err);
    CHECK(st == FileStatus::Ok);
    CHECK(rw == 3);
    std::vector<float> expect = {10,11, 12,13, 14,15};
    CHECK(data.size() == expect.size());
    bool identical = (std::memcmp(data.data(), expect.data(), expect.size() * sizeof(float)) == 0);
    CHECK(identical);
    clean_dir(dir);
    DONE();
}

static void t_finite_bit_identical_after_sanitize() {
    TEST("finite values byte-identical after sanitization");
    std::string dir = SCRATCH + "/t15"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    // Use values whose bit patterns must survive exactly (fractions, negatives).
    std::vector<float> payload = {0.125f, -3.5f, nan, 42.0f, 1e-7f, -0.0f, 7.25f, 9.5f};
    write_raw(layer_path(dir, 0), kCalibMagic, 4, 2, payload);  // row 1 = {nan,42} dropped
    validate_dir(dir, NonFinitePolicy::DropRow, 1, 2, 1);
    uint32_t m,rw,cl; std::vector<float> data; std::string err;
    read_calib_file(layer_path(dir, 0), 2, m, rw, cl, data, err);
    std::vector<float> expect = {0.125f, -3.5f, 1e-7f, -0.0f, 7.25f, 9.5f};
    CHECK(data.size() == expect.size());
    CHECK(std::memcmp(data.data(), expect.data(), expect.size() * sizeof(float)) == 0);
    clean_dir(dir);
    DONE();
}

static void t_raw_and_clean_hashes_recorded() {
    TEST("raw and clean hashes recorded (differ iff rewritten)");
    std::string dir = SCRATCH + "/t16"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    write_raw(layer_path(dir, 0), kCalibMagic, 3, 2, {1,2, nan,4, 5,6});
    DirReport r = validate_dir(dir, NonFinitePolicy::DropRow, 1, 2, 1);
    CHECK(r.files.size() == 1);
    CHECK(!r.files[0].raw_sha256.empty());
    CHECK(!r.files[0].clean_sha256.empty());
    CHECK(r.files[0].rewritten == true);
    CHECK(r.files[0].raw_sha256 != r.files[0].clean_sha256);
    clean_dir(dir);
    // In reject mode with no rewrite, raw == clean.
    std::string dir2 = SCRATCH + "/t16b"; ::mkdir(dir2.c_str(), 0755); clean_dir(dir2);
    write_raw(layer_path(dir2, 0), kCalibMagic, 2, 2, {1,2, 3,4});
    DirReport r2 = validate_dir(dir2, NonFinitePolicy::Reject, 1, 2, 0);
    CHECK(r2.files[0].raw_sha256 == r2.files[0].clean_sha256);
    clean_dir(dir2);
    DONE();
}

static void t_deterministic_repeat() {
    TEST("deterministic repeated execution");
    std::string dir = SCRATCH + "/t17"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    write_raw(layer_path(dir, 0), kCalibMagic, 3, 2, {1,2, nan,4, 5,6});
    write_raw(layer_path(dir, 1), kCalibMagic, 3, 2, {7,8, 9,10, 11,12});
    ManifestProvenance prov;
    prov.timestamp_utc = "fixed";  // hold non-data provenance constant
    DirReport a = validate_dir(dir, NonFinitePolicy::Reject, 2, 2, 0);
    std::string ja = build_manifest_json(a, prov);
    // Re-run over the same (unmodified, reject mode) inputs.
    DirReport b = validate_dir(dir, NonFinitePolicy::Reject, 2, 2, 0);
    std::string jb = build_manifest_json(b, prov);
    CHECK(ja == jb);
    clean_dir(dir);
    DONE();
}

static void t_manifest_valid_false_on_reject() {
    TEST("manifest valid=false when any row rejected (reject mode)");
    std::string dir = SCRATCH + "/t18"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, nan,4});
    DirReport r = validate_dir(dir, NonFinitePolicy::Reject, 1, 2, 0);
    ManifestProvenance prov;
    std::string j = build_manifest_json(r, prov);
    CHECK(j.find("\"valid\": false") != std::string::npos);
    CHECK(j.find("\"sanitized\": false") != std::string::npos);
    CHECK(j.find("\"policy\": \"reject\"") != std::string::npos);
    clean_dir(dir);
    DONE();
}

static void t_manifest_sanitized_only_droprow() {
    TEST("manifest sanitized=true only in drop-row mode");
    std::string dir = SCRATCH + "/t19"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    write_raw(layer_path(dir, 0), kCalibMagic, 3, 2, {1,2, nan,4, 5,6});
    // reject mode: never sanitized, even with a non-finite present.
    DirReport rr = validate_dir(dir, NonFinitePolicy::Reject, 1, 2, 0);
    ManifestProvenance prov;
    CHECK(build_manifest_json(rr, prov).find("\"sanitized\": false") != std::string::npos);
    clean_dir(dir);
    // drop-row mode with a removal: sanitized true.
    write_raw(layer_path(dir, 0), kCalibMagic, 3, 2, {1,2, nan,4, 5,6});
    DirReport rd = validate_dir(dir, NonFinitePolicy::DropRow, 1, 2, 1);
    CHECK(build_manifest_json(rd, prov).find("\"sanitized\": true") != std::string::npos);
    clean_dir(dir);
    // drop-row mode with nothing to drop: sanitized false.
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, 3,4});
    DirReport rd2 = validate_dir(dir, NonFinitePolicy::DropRow, 1, 2, 1);
    CHECK(build_manifest_json(rd2, prov).find("\"sanitized\": false") != std::string::npos);
    CHECK(rd2.valid == true);
    clean_dir(dir);
    DONE();
}

static void t_droprow_min_rows_gate() {
    TEST("drop-row honours min-rows floor");
    std::string dir = SCRATCH + "/t19b"; ::mkdir(dir.c_str(), 0755); clean_dir(dir);
    float nan = std::nanf("");
    // 2 rows, one non-finite -> 1 remains; require min-rows=2 -> invalid.
    write_raw(layer_path(dir, 0), kCalibMagic, 2, 2, {1,2, nan,4});
    DirReport r = validate_dir(dir, NonFinitePolicy::DropRow, 1, 2, 2);
    CHECK(r.valid == false);
    clean_dir(dir);
    DONE();
}

int main() {
    ensure_scratch();
    std::printf("=== SLHA calibration validator tests ===\n");
    t_sha256_known_answer();
    t_finite_accepted();
    t_nan_rejected();
    t_posinf_rejected();
    t_neginf_rejected();
    t_multiple_nonfinite_exact();
    t_nonfinite_index_correct();
    t_truncated_rejected();
    t_dim_mismatch_rejected();
    t_empty_rejected();
    t_missing_layer_rejected();
    t_duplicate_layer_rejected();
    t_droprow_removes_only_affected();
    t_finite_bit_identical_after_sanitize();
    t_raw_and_clean_hashes_recorded();
    t_deterministic_repeat();
    t_manifest_valid_false_on_reject();
    t_manifest_sanitized_only_droprow();
    t_droprow_min_rows_gate();
    std::printf("=== calibration tests complete: %s ===\n",
                g_total_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_total_failures == 0 ? 0 : 1;
}
