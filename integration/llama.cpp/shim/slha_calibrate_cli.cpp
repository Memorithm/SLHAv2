// slha_calibrate — production calibration validation gate.
//
// Validates a directory of `layer-<id>-k.bin` calibration dumps against the
// non-finite policy, optionally rewrites them (drop-row), writes a provenance
// manifest, and exits non-zero unless the calibration is valid under the
// policy. Invoked by both the collection driver and the trainer, and shares
// its entire implementation with the unit tests (slha_calibration.cpp).
//
// Usage:
//   slha_calibrate <calib_dir> [options]
//     --policy reject|drop-row     default: reject
//     --expect-dim N               expected K dimension (0 = do not enforce)
//     --expect-layers N            expected layer count 0..N-1 (-1 = skip)
//     --min-rows N                 drop-row: min rows each layer must retain
//     --manifest PATH              write manifest JSON to PATH
//     --impl-commit / --llama-commit / --model-id / --model-sha /
//     --dataset-sha / --codec / --command / --timestamp   provenance strings
//
// Exit status: 0 iff the resulting calibration is valid under the policy.
#include "slha_calibration.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace slha_calib;

static const char * arg_after(int argc, char ** argv, int & i) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "[slha_calibrate] missing value for %s\n", argv[i]);
        std::exit(2);
    }
    return argv[++i];
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: slha_calibrate <calib_dir> [--policy reject|drop-row] "
                     "[--expect-dim N] [--expect-layers N] [--min-rows N] "
                     "[--manifest PATH] [provenance flags]\n");
        return 2;
    }

    std::string dir = argv[1];
    std::string policy_str = "reject";
    uint32_t expect_dim = 0;
    int expect_layers = -1;
    uint64_t min_rows = 0;
    std::string manifest_path;
    ManifestProvenance prov;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--policy") policy_str = arg_after(argc, argv, i);
        else if (a == "--expect-dim") expect_dim = uint32_t(std::stoul(arg_after(argc, argv, i)));
        else if (a == "--expect-layers") expect_layers = std::stoi(arg_after(argc, argv, i));
        else if (a == "--min-rows") min_rows = std::stoull(arg_after(argc, argv, i));
        else if (a == "--manifest") manifest_path = arg_after(argc, argv, i);
        else if (a == "--impl-commit") prov.implementation_commit = arg_after(argc, argv, i);
        else if (a == "--llama-commit") prov.llama_cpp_commit = arg_after(argc, argv, i);
        else if (a == "--model-id") prov.model_identifier = arg_after(argc, argv, i);
        else if (a == "--model-sha") prov.model_sha256 = arg_after(argc, argv, i);
        else if (a == "--dataset-sha") prov.dataset_sha256 = arg_after(argc, argv, i);
        else if (a == "--codec") prov.codec = arg_after(argc, argv, i);
        else if (a == "--command") prov.collection_command = arg_after(argc, argv, i);
        else if (a == "--timestamp") prov.timestamp_utc = arg_after(argc, argv, i);
        else {
            std::fprintf(stderr, "[slha_calibrate] unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    NonFinitePolicy policy;
    if (!parse_policy(policy_str, policy)) {
        std::fprintf(stderr, "[slha_calibrate] unknown policy '%s' (use reject|drop-row)\n",
                     policy_str.c_str());
        return 2;
    }
    prov.min_rows = min_rows;

    DirReport rep = validate_dir(dir, policy, expect_layers, expect_dim, min_rows);

    if (!manifest_path.empty()) {
        std::string json = build_manifest_json(rep, prov);
        std::string err;
        if (!write_file_atomic(manifest_path, json, err)) {
            std::fprintf(stderr, "[slha_calibrate] failed to write manifest %s: %s\n",
                         manifest_path.c_str(), err.c_str());
            return 3;
        }
    }

    std::printf("SLHA_CALIBRATION_SUMMARY\n");
    std::printf("dir=%s\n", rep.dir.c_str());
    std::printf("policy=%s\n", rep.policy.c_str());
    std::printf("num_layers=%zu\n", rep.files.size());
    std::printf("expected_layers=%d\n", rep.expected_layers);
    std::printf("expected_dim=%u observed_dim=%u dim_consistent=%s\n",
                rep.expected_dim, rep.observed_dim, rep.dim_consistent ? "true" : "false");
    std::printf("total_rows_observed=%llu total_rows_accepted=%llu total_rows_rejected=%llu\n",
                (unsigned long long)rep.total_rows_observed,
                (unsigned long long)rep.total_rows_accepted,
                (unsigned long long)rep.total_rows_rejected);
    std::printf("nan_rows=%llu posinf_rows=%llu neginf_rows=%llu nonfinite_scalars=%llu\n",
                (unsigned long long)rep.total_nan_rows,
                (unsigned long long)rep.total_posinf_rows,
                (unsigned long long)rep.total_neginf_rows,
                (unsigned long long)rep.total_nonfinite_scalars);
    if (!rep.missing_layers.empty()) {
        std::printf("missing_layers=");
        for (size_t i = 0; i < rep.missing_layers.size(); ++i)
            std::printf("%s%d", i ? "," : "", rep.missing_layers[i]);
        std::printf("\n");
    }
    if (!rep.duplicate_layers.empty()) {
        std::printf("duplicate_layers=");
        for (size_t i = 0; i < rep.duplicate_layers.size(); ++i)
            std::printf("%s%d", i ? "," : "", rep.duplicate_layers[i]);
        std::printf("\n");
    }
    std::printf("sanitized=%s\n", rep.sanitized ? "true" : "false");
    if (!rep.error.empty()) std::printf("error=%s\n", rep.error.c_str());
    std::printf("valid=%s\n", rep.valid ? "true" : "false");
    std::printf("END_SLHA_CALIBRATION_SUMMARY\n");

    return rep.valid ? 0 : 1;
}
