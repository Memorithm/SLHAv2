// SLHA experimental score oracles (diagnostic only).
//
// PR #62 showed that no rescaling which is CONSTANT WITHIN A SOFTMAX ROW can
// explain the compressed-score quality gap. That leaves two candidate
// mechanisms unseparated:
//
//   A. key-ranking errors      — SLHA puts the wrong keys at the top;
//   B. order-preserving errors — the ranking is right but the score gaps /
//                                shape are wrong.
//
// These oracles partition the end-to-end gap between the two by transplanting
// one property of the baseline row into the SLHA row while holding the other
// fixed. Each oracle rewrites one attention row (one (layer, head, query)
// softmax group) from the paired baseline vector B and SLHA vector S.
//
// THESE ARE NOT DEPLOYABLE. Every oracle needs the baseline Q*K row, which is
// exactly what compression is meant to avoid computing. They are measurement
// instruments, not corrections.
//
// Determinism: ranking uses a STRICT TOTAL ORDER — score descending, then
// original key index ascending. Because the comparator never reports two
// distinct keys as equivalent, the permutation is unique and does not depend on
// sort stability, thread scheduling, or pointer order.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace slha_oracle {

enum class Mode {
    Off = 0,
    BaselineIdentity,          // out = B          (must reproduce pass-through)
    SlhaIdentity,              // out = S          (must reproduce strict replacement)
    BaselineRankSlhaValues,    // Oracle A: baseline ranking, SLHA value multiset
    SlhaRankBaselineValues,    // Oracle B: SLHA ranking, baseline value multiset
    BaselineTopKRank,          // top-k baseline keys promoted, SLHA values + tail order
};

// Outcome of applying an oracle to one row. Anything other than Ok is a
// fail-closed condition: the caller must not write a partial destination row.
enum class ApplyStatus {
    Ok = 0,
    NonFiniteInput,      // B or S contains NaN / +/-Inf
    EmptyRow,            // no active positions
    InvalidPermutation,  // internal consistency check failed (defensive)
};

// Per-row statistics accumulated by the caller.
struct Stats {
    uint64_t vectors = 0;
    uint64_t logits = 0;
    uint64_t permutations = 0;
    uint64_t ties = 0;                 // adjacent equal-score pairs in a ranking
    uint64_t nonfinite_input = 0;
    uint64_t invalid_permutation = 0;
    uint64_t partial_write = 0;        // must stay 0 by construction
};

// Reusable scratch so the hot path performs no allocation per row.
struct Workspace {
    std::vector<int32_t> perm_b;
    std::vector<int32_t> perm_s;
    std::vector<uint8_t> selected;
    std::vector<int32_t> tail;
    void ensure(size_t n);
};

struct Config {
    bool active = false;         // an explicit oracle spec was provided
    bool valid = true;           // spec parsed cleanly
    Mode mode = Mode::Off;
    int topk = 0;                // only for BaselineTopKRank
    std::string spec;            // raw specification
    std::string canonical;       // canonical form used for the hash
    std::string error;
    std::string config_sha256;   // sha256 of the canonical form
};

// Parse the SLHA_SCORE_ORACLE specification. Accepted values:
//   "off"
//   "baseline-identity"
//   "slha-identity"
//   "baseline-rank-slha-values"
//   "slha-rank-baseline-values"
//   "baseline-topk:<k>"          k >= 1
// Anything else is rejected (out.valid = false, out.error set). Strict: no
// silent fallback to a default mode.
bool parse_oracle(const std::string & spec, Config & out);

// Human-readable mode name (also the canonical spec text).
const char * mode_name(Mode m);

// Apply an oracle to one row.
//
//   b, s : paired finite score vectors of length n (n active positions)
//   out  : destination, length n; written only on ApplyStatus::Ok
//   ties : receives the number of adjacent equal-score pairs observed in
//          whichever ranking(s) the mode computes
//
// `out` may not alias `b` or `s`. The value multiset and the key ordering
// guarantees of each mode are documented on the Mode enum and enforced by the
// production-linked tests.
ApplyStatus apply(Mode mode,
                  int topk,
                  const float * b,
                  const float * s,
                  size_t n,
                  float * out,
                  uint64_t * ties,
                  Workspace & ws);

// --- verification helpers (used by tests and by runtime invariant sampling) ---

// Rank permutation under the strict total order (score desc, index asc).
// `perm[r]` is the key index holding the r-th largest score.
void rank_permutation(const float * v, size_t n, std::vector<int32_t> & perm,
                      uint64_t * ties);

// True if the two vectors induce the same ranking permutation. NOTE: this is
// only the right predicate when the assigned values are distinct. When a
// transplant assigns tied values, the relative order of the equal-valued keys
// is not observable from the output, so use respects_ranking() instead.
bool same_ranking(const float * x, const float * y, size_t n, Workspace & ws);

// Tie-aware ordering invariant: true when `out` is non-increasing along the
// ranking permutation of `ref`. This is the correct statement of "keys ordered
// like ref" for a value transplant — wherever the assigned values differ the
// order must match ref exactly, and wherever they tie the order is unobservable
// and therefore unconstrained.
bool respects_ranking(const float * out, const float * ref, size_t n, Workspace & ws);

// True if the two vectors hold the same value multiset (compared bitwise on
// the sorted order, so -0.0 and +0.0 are distinguished).
bool same_value_multiset(const float * x, const float * y, size_t n,
                         std::vector<float> & scratch_x,
                         std::vector<float> & scratch_y);

}  // namespace slha_oracle
