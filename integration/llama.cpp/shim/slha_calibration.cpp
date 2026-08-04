// SLHA calibration data validation — implementation. See slha_calibration.hpp.
#include "slha_calibration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace slha_calib {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Self-contained, std-only.
// ---------------------------------------------------------------------------
namespace {

struct Sha256 {
    uint32_t h[8];
    uint64_t len = 0;      // total message length in bytes
    uint8_t buf[64];
    size_t buf_len = 0;

    Sha256() {
        h[0] = 0x6a09e667; h[1] = 0xbb67ae85; h[2] = 0x3c6ef372; h[3] = 0xa54ff53a;
        h[4] = 0x510e527f; h[5] = 0x9b05688c; h[6] = 0x1f83d9ab; h[7] = 0x5be0cd19;
    }

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
                   (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const uint8_t * data, size_t n) {
        len += n;
        while (n > 0) {
            size_t take = std::min(n, size_t(64) - buf_len);
            std::memcpy(buf + buf_len, data, take);
            buf_len += take;
            data += take;
            n -= take;
            if (buf_len == 64) { block(buf); buf_len = 0; }
        }
    }

    std::string hex() {
        uint64_t bit_len = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buf_len != 56) { update(&zero, 1); }
        uint8_t lenbuf[8];
        for (int i = 0; i < 8; ++i) lenbuf[i] = uint8_t(bit_len >> (56 - i * 8));
        update(lenbuf, 8);
        static const char * hexd = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int i = 0; i < 8; ++i) {
            for (int s = 28; s >= 0; s -= 4) out.push_back(hexd[(h[i] >> s) & 0xF]);
        }
        return out;
    }
};

}  // namespace

std::string sha256_hex(const uint8_t * data, size_t len) {
    Sha256 s;
    s.update(data, len);
    return s.hex();
}

bool sha256_file(const std::string & path, std::string & out_hex) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    Sha256 s;
    std::array<char, 1 << 16> buf;
    while (in) {
        in.read(buf.data(), buf.size());
        std::streamsize got = in.gcount();
        if (got > 0) s.update(reinterpret_cast<const uint8_t *>(buf.data()), size_t(got));
    }
    out_hex = s.hex();
    return true;
}

// ---------------------------------------------------------------------------
// Policy parsing.
// ---------------------------------------------------------------------------
bool parse_policy(const std::string & s, NonFinitePolicy & out) {
    if (s == "reject") { out = NonFinitePolicy::Reject; return true; }
    if (s == "drop-row") { out = NonFinitePolicy::DropRow; return true; }
    out = NonFinitePolicy::Reject;
    return false;
}

const char * policy_name(NonFinitePolicy p) {
    return p == NonFinitePolicy::DropRow ? "drop-row" : "reject";
}

const char * file_status_name(FileStatus s) {
    switch (s) {
        case FileStatus::Ok: return "ok";
        case FileStatus::NotFound: return "not-found";
        case FileStatus::TruncatedHeader: return "truncated-header";
        case FileStatus::BadMagic: return "bad-magic";
        case FileStatus::ZeroCols: return "zero-cols";
        case FileStatus::EmptyData: return "empty-data";
        case FileStatus::SizeMismatch: return "size-mismatch";
        case FileStatus::DimMismatch: return "dim-mismatch";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// File I/O.
// ---------------------------------------------------------------------------
FileStatus read_calib_file(const std::string & path,
                           uint32_t expected_cols,
                           uint32_t & magic,
                           uint32_t & rows,
                           uint32_t & cols,
                           std::vector<float> & data,
                           std::string & err) {
    magic = 0; rows = 0; cols = 0; data.clear(); err.clear();

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { err = "cannot open file"; return FileStatus::NotFound; }
    std::streamoff size = in.tellg();
    in.seekg(0);
    if (size < 12) { err = "fewer than 12 header bytes"; return FileStatus::TruncatedHeader; }

    uint32_t hdr[3];
    in.read(reinterpret_cast<char *>(hdr), 12);
    magic = hdr[0];
    rows = hdr[1];
    cols = hdr[2];

    if (magic != kCalibMagic) { err = "bad magic"; return FileStatus::BadMagic; }
    if (cols == 0) { err = "zero columns"; return FileStatus::ZeroCols; }

    // Exact size relation: header (12) + rows*cols*4. Detects truncation and
    // trailing garbage before any allocation.
    const uint64_t expect = uint64_t(12) + uint64_t(rows) * uint64_t(cols) * 4ull;
    if (uint64_t(size) != expect) {
        std::ostringstream os;
        os << "size mismatch: file=" << uint64_t(size) << " expected=" << expect
           << " (rows=" << rows << " cols=" << cols << ")";
        err = os.str();
        return FileStatus::SizeMismatch;
    }
    if (rows == 0) { err = "zero rows"; return FileStatus::EmptyData; }
    if (expected_cols != 0 && cols != expected_cols) {
        std::ostringstream os;
        os << "dimension mismatch: cols=" << cols << " expected=" << expected_cols;
        err = os.str();
        return FileStatus::DimMismatch;
    }

    const uint64_t n = uint64_t(rows) * uint64_t(cols);
    data.resize(n);
    in.read(reinterpret_cast<char *>(data.data()), std::streamsize(n * 4));
    if (!in) { err = "short read of payload"; data.clear(); return FileStatus::SizeMismatch; }
    return FileStatus::Ok;
}

RowScan scan_matrix(const std::vector<float> & data, uint64_t rows, uint32_t cols) {
    RowScan s;
    s.rows = rows;
    s.cols = cols;
    for (uint64_t r = 0; r < rows; ++r) {
        const float * row = data.data() + r * cols;
        bool has_nan = false, has_pinf = false, has_ninf = false, has_nonfinite = false;
        for (uint32_t c = 0; c < cols; ++c) {
            float v = row[c];
            if (std::isnan(v)) { has_nan = true; has_nonfinite = true; ++s.nonfinite_scalars; }
            else if (std::isinf(v)) {
                has_nonfinite = true; ++s.nonfinite_scalars;
                if (v > 0) has_pinf = true; else has_ninf = true;
            }
        }
        if (has_nonfinite) {
            ++s.nonfinite_rows;
            s.nonfinite_row_indices.push_back(r);
            if (has_nan) ++s.nan_rows;
            if (has_pinf) ++s.posinf_rows;
            if (has_ninf) ++s.neginf_rows;
        } else {
            ++s.finite_rows;
        }
    }
    return s;
}

void drop_nonfinite_rows(const std::vector<float> & data,
                         uint64_t rows,
                         uint32_t cols,
                         const RowScan & scan,
                         std::vector<float> & out) {
    out.clear();
    if (scan.nonfinite_rows == 0) { out = data; return; }
    out.reserve((rows - scan.nonfinite_rows) * cols);
    // nonfinite_row_indices is ascending; walk it as we copy.
    size_t next = 0;
    for (uint64_t r = 0; r < rows; ++r) {
        if (next < scan.nonfinite_row_indices.size() && scan.nonfinite_row_indices[next] == r) {
            ++next;
            continue;  // skip whole row; finite scalars in kept rows are untouched
        }
        const float * row = data.data() + r * cols;
        out.insert(out.end(), row, row + cols);
    }
}

bool write_calib_file(const std::string & path,
                      uint32_t magic,
                      uint32_t cols,
                      const std::vector<float> & data,
                      std::string & err) {
    err.clear();
    if (cols == 0 || data.size() % cols != 0) { err = "payload not a multiple of cols"; return false; }
    const uint32_t rows = uint32_t(data.size() / cols);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { err = "cannot open tmp for write"; return false; }
        out.write(reinterpret_cast<const char *>(&magic), 4);
        out.write(reinterpret_cast<const char *>(&rows), 4);
        out.write(reinterpret_cast<const char *>(&cols), 4);
        out.write(reinterpret_cast<const char *>(data.data()),
                  std::streamsize(data.size() * sizeof(float)));
        if (!out) { err = "write failed"; out.close(); std::remove(tmp.c_str()); return false; }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "rename failed";
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool write_file_atomic(const std::string & path, const std::string & content, std::string & err) {
    err.clear();
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { err = "cannot open tmp"; return false; }
        out.write(content.data(), std::streamsize(content.size()));
        if (!out) { err = "write failed"; out.close(); std::remove(tmp.c_str()); return false; }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "rename failed"; std::remove(tmp.c_str()); return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Directory discovery.
// ---------------------------------------------------------------------------
std::vector<LayerFile> discover_layer_files(const std::string & dir,
                                            std::vector<int32_t> & duplicates_out) {
    std::vector<LayerFile> found;
    std::vector<int32_t> seen;
    DIR * d = opendir(dir.c_str());
    if (!d) return found;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        // Match layer-<id>-k.bin exactly (id = non-negative integer).
        const std::string prefix = "layer-";
        const std::string suffix = "-k.bin";
        if (name.size() <= prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        std::string mid = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        if (mid.empty()) continue;
        bool all_digit = true;
        for (char c : mid) if (c < '0' || c > '9') { all_digit = false; break; }
        if (!all_digit) continue;
        int32_t id = 0;
        try { id = std::stoi(mid); } catch (...) { continue; }
        if (std::find(seen.begin(), seen.end(), id) != seen.end()) {
            duplicates_out.push_back(id);  // filesystem can't hold two, but guards logic
        }
        seen.push_back(id);
        found.push_back({id, dir + "/" + name});
    }
    closedir(d);
    std::sort(found.begin(), found.end(),
              [](const LayerFile & a, const LayerFile & b) { return a.layer_id < b.layer_id; });
    std::sort(duplicates_out.begin(), duplicates_out.end());
    duplicates_out.erase(std::unique(duplicates_out.begin(), duplicates_out.end()),
                         duplicates_out.end());
    return found;
}

// ---------------------------------------------------------------------------
// Directory validation.
// ---------------------------------------------------------------------------
DirReport validate_dir(const std::string & dir,
                       NonFinitePolicy policy,
                       int expected_layers,
                       uint32_t expected_dim,
                       uint64_t min_rows) {
    DirReport rep;
    rep.dir = dir;
    rep.policy = policy_name(policy);
    rep.expected_dim = expected_dim;
    rep.expected_layers = expected_layers;

    std::vector<int32_t> dups;
    std::vector<LayerFile> files = discover_layer_files(dir, dups);
    rep.duplicate_layers = dups;

    if (files.empty()) {
        rep.error = "no layer-<id>-k.bin files found";
        rep.valid = false;
        return rep;
    }

    // Expected-layer / missing-layer check (contiguous 0..N-1 when enforced).
    if (expected_layers >= 0) {
        std::vector<bool> present(size_t(expected_layers), false);
        for (const auto & f : files) {
            if (f.layer_id >= 0 && f.layer_id < expected_layers) present[size_t(f.layer_id)] = true;
        }
        for (int i = 0; i < expected_layers; ++i) if (!present[size_t(i)]) rep.missing_layers.push_back(i);
    }

    bool structural_ok = true;
    for (const auto & lf : files) {
        FileReport fr;
        fr.layer_id = lf.layer_id;
        fr.path = lf.path;
        sha256_file(lf.path, fr.raw_sha256);
        fr.clean_sha256 = fr.raw_sha256;

        uint32_t magic = 0, rows = 0, cols = 0;
        std::vector<float> data;
        std::string err;
        fr.status = read_calib_file(lf.path, expected_dim, magic, rows, cols, data, err);
        fr.magic = magic;
        if (fr.status != FileStatus::Ok) {
            structural_ok = false;
            if (rep.error.empty()) {
                rep.error = "layer " + std::to_string(lf.layer_id) + ": " +
                            file_status_name(fr.status) + " (" + err + ")";
            }
            rep.files.push_back(std::move(fr));
            continue;
        }

        if (rep.observed_dim == 0) rep.observed_dim = cols;
        else if (rep.observed_dim != cols) rep.dim_consistent = false;

        fr.scan = scan_matrix(data, rows, cols);
        fr.rows_observed = rows;
        rep.total_rows_observed += rows;
        rep.total_nan_rows += fr.scan.nan_rows;
        rep.total_posinf_rows += fr.scan.posinf_rows;
        rep.total_neginf_rows += fr.scan.neginf_rows;
        rep.total_nonfinite_scalars += fr.scan.nonfinite_scalars;

        if (policy == NonFinitePolicy::DropRow && fr.scan.nonfinite_rows > 0) {
            std::vector<float> clean;
            drop_nonfinite_rows(data, rows, cols, fr.scan, clean);
            std::string werr;
            if (!write_calib_file(lf.path, magic, cols, clean, werr)) {
                structural_ok = false;
                if (rep.error.empty())
                    rep.error = "layer " + std::to_string(lf.layer_id) + ": rewrite failed (" + werr + ")";
                rep.files.push_back(std::move(fr));
                continue;
            }
            fr.rewritten = true;
            rep.sanitized = true;
            sha256_file(lf.path, fr.clean_sha256);
            fr.rows_accepted = rows - fr.scan.nonfinite_rows;
            fr.rows_rejected = fr.scan.nonfinite_rows;
        } else {
            // reject mode (or drop-row with nothing to drop)
            fr.rows_accepted = fr.scan.finite_rows;
            fr.rows_rejected = fr.scan.nonfinite_rows;
        }
        rep.total_rows_accepted += fr.rows_accepted;
        rep.total_rows_rejected += fr.rows_rejected;
        rep.files.push_back(std::move(fr));
    }

    if (!rep.dim_consistent && rep.error.empty()) rep.error = "inconsistent dimensions across layers";

    // Final validity gate.
    bool ok = structural_ok && rep.dim_consistent && rep.missing_layers.empty() &&
              rep.duplicate_layers.empty();
    if (policy == NonFinitePolicy::Reject) {
        ok = ok && (rep.total_rows_rejected == 0);
    } else {  // drop-row
        for (const auto & fr : rep.files) {
            if (fr.status == FileStatus::Ok) {
                uint64_t remaining = fr.rows_observed - fr.rows_rejected;
                if (remaining < min_rows) { ok = false; break; }
            }
        }
    }
    rep.valid = ok;
    return rep;
}

// ---------------------------------------------------------------------------
// Manifest serialization.
// ---------------------------------------------------------------------------
namespace {
std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    o += "\"";
    return o;
}
std::string jbool(bool b) { return b ? "true" : "false"; }
template <typename T> std::string jnum(T v) { return std::to_string(v); }
std::string jarr_u64(const std::vector<uint64_t> & v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) o += ","; o += std::to_string(v[i]); }
    o += "]";
    return o;
}
std::string jarr_i32(const std::vector<int32_t> & v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) o += ","; o += std::to_string(v[i]); }
    o += "]";
    return o;
}
}  // namespace

std::string build_manifest_json(const DirReport & rep, const ManifestProvenance & prov) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"format_version\": " << prov.format_version << ",\n";
    o << "  \"kind\": \"slha-calibration-manifest\",\n";
    o << "  \"implementation_commit\": " << jstr(prov.implementation_commit) << ",\n";
    o << "  \"llama_cpp_commit\": " << jstr(prov.llama_cpp_commit) << ",\n";
    o << "  \"model_identifier\": " << jstr(prov.model_identifier) << ",\n";
    o << "  \"model_sha256\": " << jstr(prov.model_sha256) << ",\n";
    o << "  \"dataset_sha256\": " << jstr(prov.dataset_sha256) << ",\n";
    o << "  \"codec\": " << jstr(prov.codec) << ",\n";
    o << "  \"collection_command\": " << jstr(prov.collection_command) << ",\n";
    o << "  \"timestamp_utc\": " << jstr(prov.timestamp_utc) << ",\n";
    o << "  \"policy\": " << jstr(rep.policy) << ",\n";
    o << "  \"min_rows\": " << jnum(prov.min_rows) << ",\n";
    o << "  \"num_layers\": " << rep.files.size() << ",\n";
    o << "  \"expected_layers\": " << rep.expected_layers << ",\n";
    o << "  \"expected_dim\": " << rep.expected_dim << ",\n";
    o << "  \"observed_dim\": " << rep.observed_dim << ",\n";
    o << "  \"dim_consistent\": " << jbool(rep.dim_consistent) << ",\n";
    o << "  \"missing_layers\": " << jarr_i32(rep.missing_layers) << ",\n";
    o << "  \"duplicate_layers\": " << jarr_i32(rep.duplicate_layers) << ",\n";
    o << "  \"total_rows_observed\": " << rep.total_rows_observed << ",\n";
    o << "  \"total_rows_accepted\": " << rep.total_rows_accepted << ",\n";
    o << "  \"total_rows_rejected\": " << rep.total_rows_rejected << ",\n";
    o << "  \"nan_row_count\": " << rep.total_nan_rows << ",\n";
    o << "  \"posinf_row_count\": " << rep.total_posinf_rows << ",\n";
    o << "  \"neginf_row_count\": " << rep.total_neginf_rows << ",\n";
    o << "  \"nonfinite_scalar_count\": " << rep.total_nonfinite_scalars << ",\n";
    o << "  \"sanitized\": " << jbool(rep.sanitized) << ",\n";
    o << "  \"error\": " << jstr(rep.error) << ",\n";
    o << "  \"valid\": " << jbool(rep.valid) << ",\n";
    o << "  \"layers\": [\n";
    for (size_t i = 0; i < rep.files.size(); ++i) {
        const FileReport & f = rep.files[i];
        o << "    {\n";
        o << "      \"layer\": " << f.layer_id << ",\n";
        o << "      \"status\": " << jstr(file_status_name(f.status)) << ",\n";
        o << "      \"rows_observed\": " << f.rows_observed << ",\n";
        o << "      \"cols\": " << f.scan.cols << ",\n";
        o << "      \"finite_rows\": " << f.scan.finite_rows << ",\n";
        o << "      \"nan_rows\": " << f.scan.nan_rows << ",\n";
        o << "      \"posinf_rows\": " << f.scan.posinf_rows << ",\n";
        o << "      \"neginf_rows\": " << f.scan.neginf_rows << ",\n";
        o << "      \"nonfinite_rows\": " << f.scan.nonfinite_rows << ",\n";
        o << "      \"nonfinite_scalars\": " << f.scan.nonfinite_scalars << ",\n";
        o << "      \"nonfinite_row_indices\": " << jarr_u64(f.scan.nonfinite_row_indices) << ",\n";
        o << "      \"rows_accepted\": " << f.rows_accepted << ",\n";
        o << "      \"rows_rejected\": " << f.rows_rejected << ",\n";
        o << "      \"rewritten\": " << jbool(f.rewritten) << ",\n";
        o << "      \"raw_sha256\": " << jstr(f.raw_sha256) << ",\n";
        o << "      \"clean_sha256\": " << jstr(f.clean_sha256) << "\n";
        o << "    }" << (i + 1 < rep.files.size() ? "," : "") << "\n";
    }
    o << "  ]\n";
    o << "}\n";
    return o.str();
}

}  // namespace slha_calib
