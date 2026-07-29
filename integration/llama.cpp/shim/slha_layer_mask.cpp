// SLHA score-replacement layer mask — implementation. See slha_layer_mask.hpp.
#include "slha_layer_mask.hpp"

#include <algorithm>
#include <string>

namespace slha_mask {

namespace {

// Parse a non-negative decimal integer with no leading sign, no whitespace, and
// no extraneous characters. Returns false on any deviation.
bool parse_uint(const std::string & s, long & out) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    // Reject absurdly long tokens before conversion.
    if (s.size() > 9) return false;
    out = std::stol(s);
    return true;
}

}  // namespace

bool ParsedMask::contains(int32_t layer_id) const {
    switch (kind) {
        case MaskKind::All: return true;
        case MaskKind::None: return false;
        case MaskKind::Explicit:
            return std::binary_search(ids.begin(), ids.end(), layer_id);
    }
    return false;
}

std::vector<int32_t> ParsedMask::resolved_ids(int num_layers) const {
    if (kind == MaskKind::All) {
        std::vector<int32_t> v;
        for (int i = 0; i < num_layers; ++i) v.push_back(i);
        return v;
    }
    if (kind == MaskKind::None) return {};
    return ids;
}

std::string ParsedMask::ids_to_string(int num_layers) const {
    std::vector<int32_t> v = resolved_ids(num_layers);
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ",";
        o += std::to_string(v[i]);
    }
    return o;
}

bool parse_layer_mask(const std::string & spec, int num_layers, ParsedMask & out) {
    out = ParsedMask{};
    out.spec = spec;

    if (spec == "all") { out.kind = MaskKind::All; out.valid = true; return true; }
    if (spec == "none") { out.kind = MaskKind::None; out.valid = true; return true; }
    if (spec.empty()) {
        out.error = "empty layer mask; use 'none' to select no layers";
        return false;
    }

    std::vector<int32_t> ids;
    size_t i = 0;
    while (i <= spec.size()) {
        // Extract the next comma-delimited token.
        size_t comma = spec.find(',', i);
        std::string tok = spec.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        i = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;

        if (tok.empty()) {
            out.error = "empty token in layer mask (stray comma)";
            return false;
        }

        size_t dash = tok.find('-');
        if (dash == std::string::npos) {
            long v;
            if (!parse_uint(tok, v)) {
                out.error = "invalid layer id token '" + tok + "'";
                return false;
            }
            if (num_layers > 0 && v >= num_layers) {
                out.error = "layer id " + std::to_string(v) + " out of range [0," +
                            std::to_string(num_layers - 1) + "]";
                return false;
            }
            ids.push_back(static_cast<int32_t>(v));
        } else {
            std::string los = tok.substr(0, dash);
            std::string his = tok.substr(dash + 1);
            long lo, hi;
            if (!parse_uint(los, lo) || !parse_uint(his, hi)) {
                out.error = "malformed range '" + tok + "'";
                return false;
            }
            if (lo > hi) {
                out.error = "inverted range '" + tok + "' (lo > hi)";
                return false;
            }
            if (num_layers > 0 && hi >= num_layers) {
                out.error = "range '" + tok + "' exceeds layer count " + std::to_string(num_layers);
                return false;
            }
            for (long v = lo; v <= hi; ++v) ids.push_back(static_cast<int32_t>(v));
        }
    }

    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.empty()) {
        out.error = "layer mask resolved to no layers; use 'none' to select nothing";
        return false;
    }

    out.kind = MaskKind::Explicit;
    out.ids = std::move(ids);
    out.valid = true;
    return true;
}

}  // namespace slha_mask
