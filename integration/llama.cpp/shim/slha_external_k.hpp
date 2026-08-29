#ifndef SLHA_EXTERNAL_K_HPP
#define SLHA_EXTERNAL_K_HPP

#include <cstddef>
#include <string>

/**
 * Opt-in physical-K experiment.
 *
 * This flag is intentionally orthogonal to SLHA_KV_MODE so the existing
 * tilestore/scorediag experiments retain their historical semantics. Enable
 * with SLHA_EXTERNAL_K=1 together with SLHA_KV_MODE=tilestore and
 * SLHA_SCORE_MODE=replace.
 */
bool slha_external_k_enabled();

/**
 * Validate that the environment is compatible with a baseline-free SLHA score
 * path. Returns false and fills error on any combination that would require
 * the ordinary llama.cpp K tensor or paired baseline Q*K logits.
 */
bool slha_external_k_validate_environment(std::string * error);

/**
 * Size the existing external tile store from llama.cpp's real KV capacity.
 * Must run before the first tile write. Returns false without changing live
 * tile data if the store cannot be prepared safely.
 */
bool slha_external_k_prepare_store(size_t runtime_capacity);

#endif
