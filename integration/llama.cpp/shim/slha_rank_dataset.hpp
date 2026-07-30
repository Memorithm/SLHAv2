// SLHA ranking-training dataset collection (offline only).
//
// Collects the paired rows needed to train a projection that PRESERVES the
// baseline top-ranked keys. For a sampled (layer, token, head) the exact
// baseline logits B and the current SLHA logits S are recorded together with
// the extended query vector, restricted to the CAUSALLY VISIBLE key set:
// written tile, current stream, visible to the query, finite, not padding, not
// masked. That is the same domain the rank-transplant oracles used.
//
// The baseline score is exactly reconstructible offline: the extended query
// zero-pads every GQA slot except the head's own, so
//
//     B_j = <q_extended, k_j>
//
// where k_j is the full n_embd_gqa key row. Storing q_extended and the key
// matrix therefore lets a candidate projection be scored offline without
// re-running the model, and without any inference-time baseline access.
//
// THIS IS A TRAINING-DATA PATH. It is enabled only by SLHA_RANK_DATASET_DIR and
// is never part of the deployable inference path. The deployable path is
// asserted separately to never read the baseline score tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace slha_rank_dataset {

// Deterministic sampling of query rows. Recorded in the manifest so the
// collected set is reproducible and its coverage is auditable.
struct Sampling {
    int token_stride = 8;    // record a row when (t % token_stride) == 0
    int max_heads = 32;      // record heads [0, max_heads)
};

bool enabled();
void enable(const char * dir);
const Sampling & sampling();

// True when this (token, head) is sampled. Cheap and side-effect free.
bool wanted(int64_t t, int64_t h);

// The KV cache is cleared between evaluation chunks and positions restart at 0,
// so the key matrix is only meaningful within one chunk. begin_chunk() closes
// the current chunk (persisting its key matrix) and opens the next one; rows are
// tagged with the chunk they belong to, which also gives the dataset a natural
// contiguous-token-range split axis.
void begin_chunk();
int current_chunk();

// Record one causally visible row pair.
//   layer, head, gqa_group, token position t
//   q_extended : n_embd_gqa floats (the padded per-head query)
//   b, s       : n_visible floats each, the baseline and SLHA logits
void add_row(int32_t layer, int32_t head, int32_t gqa_group, int64_t t,
             const float * q_extended, size_t q_dim,
             const float * b, const float * s, size_t n_visible);

// Record the full key matrix for a layer, in KV position order. Called once per
// layer at flush time from the tile store's shadow copy of the written rows.
void add_keys(int32_t layer, const float * rows, size_t n_rows, size_t dim);

// Write every layer's dataset plus a manifest. Returns false on any I/O error;
// a partial dataset is never marked valid.
bool flush(std::string * error);

}  // namespace slha_rank_dataset
