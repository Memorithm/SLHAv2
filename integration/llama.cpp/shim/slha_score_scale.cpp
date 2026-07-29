// SLHA experimental score scaling — implementation. See slha_score_scale.hpp.
#include "slha_score_scale.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace slha_scale {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), self-contained. Same construction as the calibration
// validator; duplicated here to keep this module dependency-free.
// ---------------------------------------------------------------------------
namespace {
struct Sha256 {
    uint32_t h[8]; uint64_t len = 0; uint8_t buf[64]; size_t bl = 0;
    Sha256() { const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
               std::memcpy(h, iv, sizeof(iv)); }
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
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[i*4])<<24)|(uint32_t(p[i*4+1])<<16)|(uint32_t(p[i*4+2])<<8)|uint32_t(p[i*4+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^(~e&g), t1=hh+S1+ch+K[i]+w[i];
            uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22), mj=(a&b)^(a&c)^(b&c), t2=S0+mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void update(const uint8_t * data, size_t n) {
        len += n;
        while (n > 0) { size_t t = std::min(n, size_t(64)-bl); std::memcpy(buf+bl,data,t);
                        bl+=t; data+=t; n-=t; if (bl==64){block(buf); bl=0;} }
    }
    std::string hex() {
        uint64_t bl_bits = len*8; uint8_t pad=0x80; update(&pad,1); uint8_t z=0;
        while (bl != 56) update(&z,1);
        uint8_t lb[8]; for (int i=0;i<8;++i) lb[i]=uint8_t(bl_bits>>(56-i*8)); update(lb,8);
        static const char* hd="0123456789abcdef"; std::string o; o.reserve(64);
        for (int i=0;i<8;++i) for (int s=28;s>=0;s-=4) o.push_back(hd[(h[i]>>s)&0xF]);
        return o;
    }
};
}  // namespace

std::string sha256_hex(const std::string & s) {
    Sha256 h; h.update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); return h.hex();
}

// ---------------------------------------------------------------------------
// Numeric parsing: a finite, strictly-positive decimal. Rejects NaN/Inf/<=0
// and any trailing garbage.
// ---------------------------------------------------------------------------
namespace {
bool parse_pos_double(const std::string & s, double & out) {
    if (s.empty()) return false;
    // Reject explicit nan/inf spellings up front (strtod would accept them).
    std::string low; low.reserve(s.size());
    for (char c : s) low.push_back(char(std::tolower((unsigned char)c)));
    if (low.find("nan") != std::string::npos || low.find("inf") != std::string::npos) return false;
    const char * b = s.c_str(); char * end = nullptr;
    errno = 0;
    double v = std::strtod(b, &end);
    if (end == b || *end != '\0') return false;      // trailing garbage / empty
    if (!std::isfinite(v)) return false;
    if (v <= 0.0) return false;
    out = v; return true;
}

bool parse_int(const std::string & s, long & out) {
    if (s.empty()) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    if (s.size() > 9) return false;
    out = std::stol(s); return true;
}
}  // namespace

double ScaleMap::get(int32_t layer_id) const {
    if (mode == Mode::Global) return global;
    auto it = per_layer.find(layer_id);
    return it != per_layer.end() ? it->second : 1.0;
}
bool ScaleMap::has_override(int32_t layer_id) const {
    return mode == Mode::PerLayer && per_layer.count(layer_id) > 0;
}
std::string ScaleMap::canonical() const {
    std::ostringstream o;
    o.setf(std::ios::fixed); o.precision(9);
    if (mode == Mode::Global) {
        o << "global=" << global;
    } else {
        o << "perlayer";
        for (const auto & kv : per_layer) o << ";" << kv.first << "=" << kv.second;
    }
    return o.str();
}

// ---------------------------------------------------------------------------
// Minimal JSON extraction for the scale file: {"global":X,"layers":{"i":s,...}}.
// Dependency-free; tolerant of whitespace, strict on values.
// ---------------------------------------------------------------------------
namespace {
bool parse_scale_file(const std::string & json, int num_layers, ScaleMap & out) {
    out = ScaleMap{}; out.source = "file"; out.spec = json;
    // global
    double g = 1.0;
    size_t gp = json.find("\"global\"");
    if (gp != std::string::npos) {
        size_t colon = json.find(':', gp);
        size_t s = json.find_first_not_of(" \t\r\n", colon + 1);
        size_t e = json.find_first_of(",}\r\n \t", s);
        std::string tok = json.substr(s, e - s);
        if (!parse_pos_double(tok, g)) { out.error = "invalid global scale '" + tok + "'"; return false; }
    }
    size_t lp = json.find("\"layers\"");
    if (lp == std::string::npos) {
        // global-only file
        out.mode = Mode::Global; out.global = g; out.valid = true;
        out.manifest_sha256 = sha256_hex(out.canonical()); return true;
    }
    size_t brace = json.find('{', lp);
    size_t close = json.find('}', brace);
    if (brace == std::string::npos || close == std::string::npos) { out.error = "malformed layers object"; return false; }
    std::string body = json.substr(brace + 1, close - brace - 1);
    out.mode = Mode::PerLayer; out.global = g;
    // parse "id": value pairs
    size_t i = 0;
    while (i < body.size()) {
        size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
        size_t q2 = body.find('"', q1 + 1); if (q2 == std::string::npos) { out.error = "unterminated key"; return false; }
        std::string key = body.substr(q1 + 1, q2 - q1 - 1);
        size_t colon = body.find(':', q2); if (colon == std::string::npos) { out.error = "missing ':'"; return false; }
        size_t vs = body.find_first_not_of(" \t\r\n", colon + 1);
        size_t ve = body.find_first_of(",}\r\n \t", vs);
        std::string val = body.substr(vs, (ve == std::string::npos ? body.size() : ve) - vs);
        long lid; double sc;
        if (!parse_int(key, lid)) { out.error = "invalid layer id '" + key + "'"; return false; }
        if (num_layers > 0 && lid >= num_layers) { out.error = "layer " + key + " out of range"; return false; }
        if (!parse_pos_double(val, sc)) { out.error = "invalid scale for layer " + key + ": '" + val + "'"; return false; }
        if (out.per_layer.count(int32_t(lid))) { out.error = "duplicate layer " + key; return false; }
        out.per_layer[int32_t(lid)] = sc;
        i = (ve == std::string::npos) ? body.size() : ve + 1;
    }
    if (out.per_layer.empty()) { out.error = "scale file has empty layers map"; return false; }
    out.valid = true; out.manifest_sha256 = sha256_hex(out.canonical()); return true;
}
}  // namespace

bool parse_score_scale(const std::string & env_spec, const std::string & file_json,
                       int num_layers, ScaleMap & out) {
    out = ScaleMap{};
    if (!file_json.empty()) return parse_scale_file(file_json, num_layers, out);

    if (env_spec.empty()) {  // default: global 1.0
        out.mode = Mode::Global; out.global = 1.0; out.source = "default"; out.valid = true;
        out.manifest_sha256 = sha256_hex(out.canonical()); return true;
    }
    out.spec = env_spec; out.source = "env";

    const std::string prefix = "layer:";
    if (env_spec.rfind(prefix, 0) == 0) {
        out.mode = Mode::PerLayer; out.global = 1.0;
        std::string body = env_spec.substr(prefix.size());
        if (body.empty()) { out.error = "empty per-layer scale spec"; return false; }
        size_t i = 0;
        while (i <= body.size()) {
            size_t comma = body.find(',', i);
            std::string tok = body.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
            i = (comma == std::string::npos) ? body.size() + 1 : comma + 1;
            if (tok.empty()) { out.error = "empty token in scale spec"; return false; }
            size_t eq = tok.find('=');
            if (eq == std::string::npos) { out.error = "missing '=' in '" + tok + "'"; return false; }
            std::string ks = tok.substr(0, eq), vs = tok.substr(eq + 1);
            long lid; double sc;
            if (!parse_int(ks, lid)) { out.error = "invalid layer id '" + ks + "'"; return false; }
            if (num_layers > 0 && lid >= num_layers) { out.error = "layer " + ks + " out of range"; return false; }
            if (!parse_pos_double(vs, sc)) { out.error = "invalid scale '" + vs + "' for layer " + ks; return false; }
            if (out.per_layer.count(int32_t(lid))) { out.error = "duplicate layer " + ks; return false; }
            out.per_layer[int32_t(lid)] = sc;
        }
        if (out.per_layer.empty()) { out.error = "no scales parsed"; return false; }
        out.valid = true; out.manifest_sha256 = sha256_hex(out.canonical()); return true;
    }

    // global scalar
    double g;
    if (!parse_pos_double(env_spec, g)) { out.error = "invalid global scale '" + env_spec + "'"; return false; }
    out.mode = Mode::Global; out.global = g; out.valid = true;
    out.manifest_sha256 = sha256_hex(out.canonical()); return true;
}

bool resolve_against_selected(ScaleMap & map, const std::vector<int32_t> & selected) {
    if (!map.valid) return false;
    if (map.mode == Mode::Global) return true;  // global covers every layer
    for (int32_t l : selected) {
        if (map.per_layer.find(l) == map.per_layer.end()) {
            map.valid = false;
            map.error = "no scale for selected layer " + std::to_string(l);
            return false;
        }
    }
    return true;
}

}  // namespace slha_scale
