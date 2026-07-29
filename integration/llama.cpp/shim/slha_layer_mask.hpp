// SLHA score-replacement layer mask.
//
// Parses the SLHA_SCORE_LAYERS specification into a validated, deterministic
// set of transformer layer ids whose attention logits are replaced by direct
// SLHA scores. Selected layers use direct compressed-score replacement;
// unselected layers pass baseline Q*K through unchanged. This module is
// dependency-light (C++17 std-only) so the identical parser is linked by the
// shim and by the production-linked unit tests.
//
// Grammar (comma-separated, no spaces):
//   all               every layer
//   none              no layer (the ONLY valid way to select nothing)
//   N                 a single layer id
//   LO-HI             an inclusive range (LO <= HI)
//   combinations      e.g. "0,3,7", "0-6", "0-3,7,12-14"
//
// Rules: ids must be non-negative and, when num_layers > 0, strictly less than
// num_layers; ranges must satisfy LO <= HI; duplicates are removed; the result
// is sorted ascending (deterministic). An empty specification is invalid — use
// "none" to select no layers. Any malformed token fails closed (valid=false).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slha_mask {

enum class MaskKind { All, None, Explicit };

struct ParsedMask {
    bool valid = false;
    MaskKind kind = MaskKind::None;
    std::vector<int32_t> ids;  // sorted, unique; populated for Explicit
    std::string spec;          // the raw specification, verbatim
    std::string error;         // human-readable reason when !valid

    bool contains(int32_t layer_id) const;
    // Concrete resolved id list for display: All -> 0..num_layers-1 (needs a
    // count), None -> {}, Explicit -> ids.
    std::vector<int32_t> resolved_ids(int num_layers) const;
    std::string ids_to_string(int num_layers) const;
};

// Parse `spec` against `num_layers`. When num_layers <= 0 the per-id upper-bound
// check is deferred (used for an early syntax-only pass before the model's layer
// count is known); negative ids and malformed tokens are still rejected.
bool parse_layer_mask(const std::string & spec, int num_layers, ParsedMask & out);

}  // namespace slha_mask
