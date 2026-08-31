#include "slha_external_k.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

bool slha_external_k_enabled() {
    const char * value = std::getenv("SLHA_EXTERNAL_K");
    return value && value[0] != '\0' &&
        std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "off") != 0;
}

int main() {
    unsetenv("SLHA_EXTERNAL_K");

    // The policy must not constrain ordinary llama.cpp contexts.
    std::string error = "stale";
    assert(slha_external_k_validate_runtime(4, 8, &error));
    assert(error == "stale");
    assert(slha_external_k_state_serialization_supported());

    setenv("SLHA_EXTERNAL_K", "1", 1);

    error = "stale";
    assert(slha_external_k_validate_runtime(1, 1, &error));
    assert(error.empty());
    assert(!slha_external_k_state_serialization_supported());

    error.clear();
    assert(!slha_external_k_validate_runtime(2, 1, &error));
    assert(error.find("one KV stream") != std::string::npos);

    // A unified cache can expose multiple logical sequences through one
    // physical stream, so this is a distinct invariant from n_stream == 1.
    error.clear();
    assert(!slha_external_k_validate_runtime(1, 2, &error));
    assert(error.find("one logical sequence") != std::string::npos);

    error.clear();
    assert(!slha_external_k_validate_runtime(2, 2, &error));
    assert(error.find("one KV stream") != std::string::npos);

    unsetenv("SLHA_EXTERNAL_K");
    std::cout << "lifecycle_policy_tests: ok\n";
    return 0;
}
