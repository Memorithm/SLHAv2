// Implementation of the SLHA diagnostic score oracles.
// See slha_score_oracle.hpp for the design rationale.
#include "slha_score_oracle.hpp"
#include "slha_score_scale.hpp"   // reuse the audited sha256_hex

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace slha_oracle {

void Workspace::ensure(size_t n) {
    if (perm_b.size() < n) {
        perm_b.resize(n);
        perm_s.resize(n);
        selected.resize(n);
        tail.resize(n);
    }
}

const char * mode_name(Mode m) {
    switch (m) {
        case Mode::Off:                    return "off";
        case Mode::BaselineIdentity:       return "baseline-identity";
        case Mode::SlhaIdentity:           return "slha-identity";
        case Mode::BaselineRankSlhaValues: return "baseline-rank-slha-values";
        case Mode::SlhaRankBaselineValues: return "slha-rank-baseline-values";
        case Mode::BaselineTopKRank:       return "baseline-topk";
    }
    return "unknown";
}

// --------------------------------------------------------------------------
// Ranking: strict total order (score descending, then key index ascending).
// The comparator never reports two distinct keys as equivalent, so the
// permutation is unique regardless of std::sort's instability.
// --------------------------------------------------------------------------
void rank_permutation(const float * v, size_t n, std::vector<int32_t> & perm,
                      uint64_t * ties) {
    if (perm.size() < n) {
        perm.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
        perm[i] = static_cast<int32_t>(i);
    }
    std::sort(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(n),
              [v](int32_t a, int32_t b) {
                  const float va = v[a];
                  const float vb = v[b];
                  if (va != vb) {
                      return va > vb;   // score descending
                  }
                  return a < b;         // stable, deterministic tiebreak
              });
    if (ties) {
        uint64_t t = 0;
        for (size_t r = 1; r < n; ++r) {
            if (v[perm[r]] == v[perm[r - 1]]) {
                ++t;
            }
        }
        *ties += t;
    }
}

static bool all_finite(const float * v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

ApplyStatus apply(Mode mode,
                  int topk,
                  const float * b,
                  const float * s,
                  size_t n,
                  float * out,
                  uint64_t * ties,
                  Workspace & ws) {
    if (mode == Mode::Off) {
        return ApplyStatus::Ok;   // caller keeps its own value; nothing to do
    }
    if (n == 0) {
        return ApplyStatus::EmptyRow;
    }
    if (!b || !s || !out) {
        return ApplyStatus::InvalidPermutation;
    }
    if (!all_finite(b, n) || !all_finite(s, n)) {
        return ApplyStatus::NonFiniteInput;
    }

    ws.ensure(n);

    switch (mode) {
        case Mode::BaselineIdentity:
            std::memcpy(out, b, n * sizeof(float));
            return ApplyStatus::Ok;

        case Mode::SlhaIdentity:
            std::memcpy(out, s, n * sizeof(float));
            return ApplyStatus::Ok;

        case Mode::BaselineRankSlhaValues: {
            // Keys ordered like B; the r-th ranked key receives the r-th
            // largest S value. Values are gathered through S's own ranking
            // permutation, so the S multiset is reproduced exactly, bit for bit
            // (including -0.0), with each element used exactly once.
            rank_permutation(b, n, ws.perm_b, ties);
            rank_permutation(s, n, ws.perm_s, ties);
            for (size_t r = 0; r < n; ++r) {
                out[ws.perm_b[r]] = s[ws.perm_s[r]];
            }
            return ApplyStatus::Ok;
        }

        case Mode::SlhaRankBaselineValues: {
            // Keys ordered like S; the r-th ranked key receives the r-th
            // largest B value.
            rank_permutation(b, n, ws.perm_b, ties);
            rank_permutation(s, n, ws.perm_s, ties);
            for (size_t r = 0; r < n; ++r) {
                out[ws.perm_s[r]] = b[ws.perm_b[r]];
            }
            return ApplyStatus::Ok;
        }

        case Mode::BaselineTopKRank: {
            if (topk < 1) {
                return ApplyStatus::InvalidPermutation;
            }
            rank_permutation(b, n, ws.perm_b, ties);
            rank_permutation(s, n, ws.perm_s, ties);
            const size_t k = std::min(static_cast<size_t>(topk), n);

            // The baseline's top-k keys are promoted into the first k ranks,
            // in baseline order. Every other key keeps its RELATIVE SLHA order.
            // The value multiset is S's, assigned in descending rank order.
            std::fill(ws.selected.begin(), ws.selected.begin() + static_cast<std::ptrdiff_t>(n), 0);
            for (size_t i = 0; i < k; ++i) {
                ws.selected[ws.perm_b[i]] = 1;
            }
            size_t tail_n = 0;
            for (size_t r = 0; r < n; ++r) {
                const int32_t key = ws.perm_s[r];
                if (!ws.selected[key]) {
                    ws.tail[tail_n++] = key;
                }
            }
            if (tail_n + k != n) {
                return ApplyStatus::InvalidPermutation;   // defensive
            }
            for (size_t i = 0; i < k; ++i) {
                out[ws.perm_b[i]] = s[ws.perm_s[i]];
            }
            for (size_t j = 0; j < tail_n; ++j) {
                out[ws.tail[j]] = s[ws.perm_s[k + j]];
            }
            return ApplyStatus::Ok;
        }

        case Mode::Off:
        default:
            return ApplyStatus::Ok;
    }
}

bool same_ranking(const float * x, const float * y, size_t n, Workspace & ws) {
    ws.ensure(n);
    rank_permutation(x, n, ws.perm_b, nullptr);
    rank_permutation(y, n, ws.perm_s, nullptr);
    for (size_t i = 0; i < n; ++i) {
        if (ws.perm_b[i] != ws.perm_s[i]) {
            return false;
        }
    }
    return true;
}

bool same_value_multiset(const float * x, const float * y, size_t n,
                         std::vector<float> & sx, std::vector<float> & sy) {
    sx.assign(x, x + n);
    sy.assign(y, y + n);
    // Compare bitwise on a bit-ordered sort so -0.0 and +0.0 stay distinct.
    auto bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    auto cmp = [&bits](float a, float b) { return bits(a) < bits(b); };
    std::sort(sx.begin(), sx.end(), cmp);
    std::sort(sy.begin(), sy.end(), cmp);
    for (size_t i = 0; i < n; ++i) {
        if (bits(sx[i]) != bits(sy[i])) {
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------
// Specification parsing (strict; unknown modes rejected)
// --------------------------------------------------------------------------
static bool parse_positive_int(const std::string & t, int & out) {
    if (t.empty()) {
        return false;
    }
    for (char c : t) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(t.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0' || v < 1 || v > 1000000) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_oracle(const std::string & spec, Config & out) {
    out = Config{};
    out.spec = spec;

    if (spec.empty() || spec == "off") {
        out.active = false;
        out.valid = true;
        out.mode = Mode::Off;
        out.canonical = "off";
        out.config_sha256 = slha_scale::sha256_hex(out.canonical);
        return true;
    }

    out.active = true;

    auto finish = [&out](Mode m, const std::string & canon, int k) {
        out.mode = m;
        out.topk = k;
        out.valid = true;
        out.canonical = canon;
        out.config_sha256 = slha_scale::sha256_hex(canon);
        return true;
    };
    auto reject = [&out](const char * why) {
        out.valid = false;
        out.mode = Mode::Off;
        out.error = why;
        out.canonical = "invalid";
        out.config_sha256 = slha_scale::sha256_hex("invalid");
        return false;
    };

    if (spec == "baseline-identity")        return finish(Mode::BaselineIdentity, spec, 0);
    if (spec == "slha-identity")            return finish(Mode::SlhaIdentity, spec, 0);
    if (spec == "baseline-rank-slha-values") return finish(Mode::BaselineRankSlhaValues, spec, 0);
    if (spec == "slha-rank-baseline-values") return finish(Mode::SlhaRankBaselineValues, spec, 0);

    const std::string prefix = "baseline-topk:";
    if (spec.compare(0, prefix.size(), prefix) == 0) {
        int k = 0;
        if (!parse_positive_int(spec.substr(prefix.size()), k)) {
            return reject("baseline-topk requires a positive integer k");
        }
        return finish(Mode::BaselineTopKRank, prefix + std::to_string(k), k);
    }

    return reject("unknown oracle mode");
}

}  // namespace slha_oracle
