// SLHA experimental score scaling (temperature calibration).
//
// Multiplies each direct SLHA score by a per-layer positive scalar `a_layer`
// BEFORE the scores are written at the raw-Q*K replacement seam. Because
// llama.cpp applies the fixed 1/sqrt(head_dim) softmax scale afterwards, this
// is exactly a per-layer softmax-temperature correction (effective inverse
// temperature s*a_layer). This is an EXPERIMENTAL research knob; the default is
// 1.0 (unchanged production behaviour). It does not change the score
// mathematics, only tests the score-magnitude/temperature hypothesis from #60.
//
// Sources:
//   SLHA_SCORE_SCALE       "1.0" | "0.75"                 (global scalar)
//                          "layer:0=0.91,5=0.72"          (per-layer overrides)
//   SLHA_SCORE_SCALE_FILE  JSON: {"global":1.0,"layers":{"0":0.91,"5":0.72}}
//
// Strict, fail-closed: only finite strictly-positive values; zero, negative,
// NaN, +/-Inf, malformed numbers, duplicate layer ids and out-of-range layer
// ids are all rejected. Deterministic ordering. A resolved manifest SHA-256 is
// computed so runs are traceable.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace slha_scale {

enum class Mode { Global, PerLayer };

struct ScaleMap {
    bool valid = false;
    Mode mode = Mode::Global;
    double global = 1.0;                 // Global mode: the scalar
    std::map<int32_t, double> per_layer; // PerLayer mode: layer -> scale
    std::string spec;                    // raw specification (env or file)
    std::string source;                  // "env" | "file" | "default"
    std::string error;
    std::string manifest_sha256;         // sha256 of the canonical resolved form

    // Scale for a layer: Global -> global; PerLayer -> override if present, else 1.0.
    double get(int32_t layer_id) const;
    bool has_override(int32_t layer_id) const;
    // Canonical string used for the manifest hash (deterministic).
    std::string canonical() const;
};

// Parse from env spec and/or file JSON. Exactly one source is used: if
// `file_json` is non-empty it takes precedence; else `env_spec`; else default
// (global 1.0). `num_layers > 0` enables the layer-id range check.
bool parse_score_scale(const std::string & env_spec,
                       const std::string & file_json,
                       int num_layers,
                       ScaleMap & out);

// After the active/selected layer set is known (the SLHA_SCORE_LAYERS mask),
// verify that a PerLayer map provides a scale for every selected layer — a
// missing selected-layer scale is rejected (no silent fallback to 1.0). Global
// mode always resolves. On failure sets out.valid=false and out.error.
bool resolve_against_selected(ScaleMap & map, const std::vector<int32_t> & selected_layers);

// SHA-256 (lowercase hex) — self-contained, std-only. Exposed for testing.
std::string sha256_hex(const std::string & s);

}  // namespace slha_scale
