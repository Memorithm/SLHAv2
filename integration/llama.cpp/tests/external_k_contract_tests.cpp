#include "slha_external_k.hpp"
#include "slha_llama.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

slha_tile_store g_slha_tile_store;

int slha_get_num_layers() {
    return 3;
}

extern "C" size_t slha_tile_size(void) {
    return 128u;
}

bool slha_tile_store::init(size_t n_layers_, size_t capacity_, size_t tile_bytes_) {
    n_layers = n_layers_;
    capacity = capacity_;
    tile_bytes = tile_bytes_;
    return n_layers > 0 && capacity > 0 && tile_bytes == 128u;
}

namespace {

void unset_contract_env() {
    unsetenv("SLHA_EXTERNAL_K");
    unsetenv("SLHA_KV_MODE");
    unsetenv("SLHA_SCORE_MODE");
    unsetenv("SLHA_SCORE_LAYERS");
    unsetenv("SLHA_SCORE_ORACLE");
    unsetenv("SLHA_ORACLE_METRICS_JSON");
    unsetenv("SLHA_SCALE_FIT_JSON");
    unsetenv("SLHA_RANK_DATASET_DIR");
}

void set_valid_external_env() {
    setenv("SLHA_EXTERNAL_K", "1", 1);
    setenv("SLHA_KV_MODE", "tilestore", 1);
    setenv("SLHA_SCORE_MODE", "replace", 1);
    setenv("SLHA_SCORE_LAYERS", "all", 1);
}

void expect_invalid(const char * needle) {
    std::string error;
    assert(slha_external_k_enabled());
    assert(!slha_external_k_validate_environment(&error));
    assert(error.find(needle) != std::string::npos);
}

} // namespace

int main() {
    unset_contract_env();
    assert(!slha_external_k_enabled());
    std::string error;
    assert(slha_external_k_validate_environment(&error));

    set_valid_external_env();
    error.clear();
    assert(slha_external_k_enabled());
    assert(slha_external_k_validate_environment(&error));
    assert(error.empty());
    assert(slha_external_k_prepare_store(4096));
    assert(g_slha_tile_store.n_layers == 3);
    assert(g_slha_tile_store.capacity == 4096);
    assert(g_slha_tile_store.tile_bytes == 128u);

    set_valid_external_env();
    setenv("SLHA_SCORE_MODE", "shadow", 1);
    expect_invalid("replace");

    set_valid_external_env();
    setenv("SLHA_SCORE_LAYERS", "0-3", 1);
    expect_invalid("all");

    set_valid_external_env();
    setenv("SLHA_SCORE_ORACLE", "rank-transplant", 1);
    expect_invalid("ORACLE");

    set_valid_external_env();
    setenv("SLHA_ORACLE_METRICS_JSON", "/tmp/metrics.json", 1);
    expect_invalid("ORACLE_METRICS");

    set_valid_external_env();
    setenv("SLHA_SCALE_FIT_JSON", "/tmp/fit.json", 1);
    expect_invalid("SCALE_FIT");

    set_valid_external_env();
    setenv("SLHA_RANK_DATASET_DIR", "/tmp/rank", 1);
    expect_invalid("RANK_DATASET");

    unset_contract_env();
    setenv("SLHA_EXTERNAL_K", "banana", 1);
    expect_invalid("1,true,on");

    unset_contract_env();
    std::cout << "external_k_contract_tests: ok\n";
    return 0;
}
