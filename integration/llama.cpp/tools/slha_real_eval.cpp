#include "llama.h"
#include "slha_external_k.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string prompt;
    std::string output_json;
    std::string logits_bin;
    int32_t max_tokens = 64;
    uint32_t context_size = 2048;
    int32_t threads = 4;
    int32_t gpu_layers = 0;
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
    int32_t ccos_cold_cycle_step = -1;
};

[[noreturn]] void usage(const char * argv0, const std::string & error = {}) {
    if (!error.empty()) {
        std::cerr << "ERROR: " << error << "\n";
    }
    std::cerr
        << "usage: " << argv0 << " --model MODEL.gguf --prompt TEXT --output-json REPORT.json "
        << "--logits-bin LOGITS.f32 [--max-tokens N] [--context-size N] [--threads N] "
        << "[--gpu-layers N] [--cache-type-k f16|f32|bf16] [--cache-type-v f16|f32|bf16] "
        << "[--ccos-cold-cycle-step N]\n";
    std::exit(2);
}

ggml_type parse_cache_type(const std::string & value) {
    if (value == "f16") return GGML_TYPE_F16;
    if (value == "f32") return GGML_TYPE_F32;
    if (value == "bf16") return GGML_TYPE_BF16;
    throw std::runtime_error("unsupported cache type '" + value + "' (supported: f16,f32,bf16)");
}

const char * cache_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_BF16: return "bf16";
        default: return "unsupported";
    }
}

Options parse_args(int argc, char ** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], "missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--model") opts.model = next();
        else if (arg == "--prompt") opts.prompt = next();
        else if (arg == "--output-json") opts.output_json = next();
        else if (arg == "--logits-bin") opts.logits_bin = next();
        else if (arg == "--max-tokens") opts.max_tokens = std::stoi(next());
        else if (arg == "--context-size") opts.context_size = static_cast<uint32_t>(std::stoul(next()));
        else if (arg == "--threads") opts.threads = std::stoi(next());
        else if (arg == "--gpu-layers") opts.gpu_layers = std::stoi(next());
        else if (arg == "--cache-type-k") opts.cache_type_k = parse_cache_type(next());
        else if (arg == "--cache-type-v") opts.cache_type_v = parse_cache_type(next());
        else if (arg == "--ccos-cold-cycle-step") opts.ccos_cold_cycle_step = std::stoi(next());
        else usage(argv[0], "unknown option " + arg);
    }

    if (opts.model.empty()) usage(argv[0], "--model is required");
    if (opts.prompt.empty()) usage(argv[0], "--prompt is required");
    if (opts.output_json.empty()) usage(argv[0], "--output-json is required");
    if (opts.logits_bin.empty()) usage(argv[0], "--logits-bin is required");
    if (opts.max_tokens <= 0) usage(argv[0], "--max-tokens must be positive");
    if (opts.context_size == 0) usage(argv[0], "--context-size must be positive");
    if (opts.threads <= 0) usage(argv[0], "--threads must be positive");
    if (opts.ccos_cold_cycle_step < -1 || opts.ccos_cold_cycle_step >= opts.max_tokens - 1) {
        usage(argv[0], "--ccos-cold-cycle-step must be -1 or leave at least one decode step after the cycle");
    }
    return opts;
}

std::string json_escape(const std::string & input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (unsigned char ch : input) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(ch));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(ch));
                }
        }
    }
    return out;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(256);
    int n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (n < 0) {
        buf.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    }
    if (n < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }
    return std::string(buf.data(), static_cast<size_t>(n));
}

llama_token argmax_token(const float * logits, int32_t n_vocab) {
    if (!logits || n_vocab <= 0) {
        throw std::runtime_error("invalid logits buffer");
    }
    llama_token best = 0;
    float best_value = logits[0];
    if (!std::isfinite(best_value)) {
        throw std::runtime_error("non-finite logit at token 0");
    }
    for (int32_t i = 1; i < n_vocab; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite logit at token " + std::to_string(i));
        }
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return best;
}

template <typename T>
void write_integer_array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_double_array(std::ostream & out, const std::vector<double> & values) {
    out << '[' << std::setprecision(17);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}


void write_store_snapshot(std::ostream & out, const slha_external_k_store_stats & stats) {
    out << "{"
        << "\"resident_bytes\":" << stats.resident_bytes << ','
        << "\"offloaded_bytes\":" << stats.offloaded_bytes << ','
        << "\"hot_slots\":" << stats.hot_slots << ','
        << "\"warm_slots\":" << stats.warm_slots << ','
        << "\"cold_slots\":" << stats.cold_slots << ','
        << "\"pinned_slots\":" << stats.pinned_slots << ','
        << "\"evictions\":" << stats.evictions
        << "}";
}


} // namespace

int main(int argc, char ** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        ggml_backend_load_all();

        const auto model_load_start = Clock::now();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("failed to load model");
        }
        const auto model_load_end = Clock::now();

        if (llama_model_has_encoder(model)) {
            llama_model_free(model);
            throw std::runtime_error("slha_real_eval currently supports decoder-only models only");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            llama_model_free(model);
            throw std::runtime_error("model vocabulary is empty");
        }

        int32_t n_prompt = -llama_tokenize(
            vocab, opts.prompt.c_str(), opts.prompt.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) {
            llama_model_free(model);
            throw std::runtime_error("failed to determine prompt token count");
        }
        std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
        const int32_t tokenized = llama_tokenize(
            vocab,
            opts.prompt.c_str(),
            opts.prompt.size(),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            true);
        if (tokenized != n_prompt) {
            llama_model_free(model);
            throw std::runtime_error("prompt tokenization failed or changed size");
        }
        if (static_cast<uint64_t>(n_prompt) + static_cast<uint64_t>(opts.max_tokens) > opts.context_size) {
            llama_model_free(model);
            throw std::runtime_error("prompt + max_tokens exceeds requested context size");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = opts.context_size;
        ctx_params.n_batch = std::max<uint32_t>(1u, std::min<uint32_t>(opts.context_size, static_cast<uint32_t>(n_prompt)));
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.type_k = opts.cache_type_k;
        ctx_params.type_v = opts.cache_type_v;
        ctx_params.no_perf = false;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("failed to create llama context");
        }
        llama_set_n_threads(ctx, opts.threads, opts.threads);

        std::ofstream logits_out(opts.logits_bin, std::ios::binary | std::ios::trunc);
        if (!logits_out) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("cannot create logits output file");
        }

        std::vector<llama_token> generated_tokens;
        std::vector<double> decode_step_ms;
        generated_tokens.reserve(static_cast<size_t>(opts.max_tokens));
        decode_step_ms.reserve(static_cast<size_t>(opts.max_tokens));
        std::string generated_text;
        bool eos_reached = false;
        double prefill_ms = 0.0;
        double ttft_ms = 0.0;
        bool ccos_lifecycle_executed = false;
        slha_external_k_store_stats ccos_before{};
        slha_external_k_store_stats ccos_cold{};
        slha_external_k_store_stats ccos_restored{};

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
        const auto inference_start = Clock::now();

        for (int32_t step = 0; step < opts.max_tokens; ++step) {
            const auto decode_start = Clock::now();
            const int rc = llama_decode(ctx, batch);
            const auto decode_end = Clock::now();
            if (rc != 0) {
                throw std::runtime_error("llama_decode failed at generation step " + std::to_string(step) +
                                         " with code " + std::to_string(rc));
            }
            const double this_decode_ms = elapsed_ms(decode_start, decode_end);
            if (step == 0) {
                prefill_ms = this_decode_ms;
            } else {
                decode_step_ms.push_back(this_decode_ms);
            }

            float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                throw std::runtime_error("llama_get_logits_ith returned null");
            }
            logits_out.write(
                reinterpret_cast<const char *>(logits),
                static_cast<std::streamsize>(static_cast<size_t>(n_vocab) * sizeof(float)));
            if (!logits_out) {
                throw std::runtime_error("failed while writing logits output");
            }

            const llama_token token = argmax_token(logits, n_vocab);
            generated_tokens.push_back(token);
            if (step == 0) {
                ttft_ms = elapsed_ms(inference_start, Clock::now());
            }

            if (llama_vocab_is_eog(vocab, token)) {
                eos_reached = true;
                break;
            }

            generated_text += token_piece(vocab, token);
            batch = llama_batch_get_one(&generated_tokens.back(), 1);

            if (opts.ccos_cold_cycle_step == step) {
                if (!slha_external_k_store_stats_snapshot(&ccos_before)) {
                    throw std::runtime_error("cannot snapshot CCOS state before quiescent COLD cycle");
                }
                const size_t active_before = ccos_before.hot_slots + ccos_before.warm_slots +
                    ccos_before.cold_slots + ccos_before.pinned_slots;
                if (active_before == 0 || ccos_before.cold_slots != 0) {
                    throw std::runtime_error("invalid active CCOS state before quiescent COLD cycle");
                }
                if (!slha_external_k_ccos_offload_quiescent() ||
                    !slha_external_k_store_stats_snapshot(&ccos_cold)) {
                    throw std::runtime_error("CCOS quiescent offload failed");
                }
                if (ccos_cold.resident_bytes != 0 || ccos_cold.cold_slots != active_before) {
                    throw std::runtime_error("CCOS quiescent offload did not produce complete COLD state");
                }
                if (!slha_external_k_ccos_restore_quiescent() ||
                    !slha_external_k_store_stats_snapshot(&ccos_restored)) {
                    throw std::runtime_error("CCOS quiescent restore failed");
                }
                if (ccos_restored.cold_slots != 0 ||
                    ccos_restored.resident_bytes != ccos_before.resident_bytes ||
                    ccos_restored.hot_slots != ccos_before.hot_slots ||
                    ccos_restored.warm_slots != ccos_before.warm_slots) {
                    throw std::runtime_error("CCOS restore did not recover pre-offload residency exactly");
                }
                ccos_lifecycle_executed = true;
            }
        }

        logits_out.close();
        const auto inference_end = Clock::now();
        const double total_inference_ms = elapsed_ms(inference_start, inference_end);
        double decode_total_ms = 0.0;
        for (double value : decode_step_ms) decode_total_ms += value;
        const double decode_tps = decode_total_ms > 0.0
            ? static_cast<double>(decode_step_ms.size()) / (decode_total_ms / 1000.0)
            : 0.0;

        std::ofstream report(opts.output_json, std::ios::trunc);
        if (!report) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("cannot create JSON report");
        }

        report << "{\n";
        report << "  \"schema_version\": 1,\n";
        report << "  \"engine\": \"llama.cpp\",\n";
        report << "  \"decoder_only\": true,\n";
        report << "  \"n_vocab\": " << n_vocab << ",\n";
        report << "  \"context_size\": " << opts.context_size << ",\n";
        report << "  \"threads\": " << opts.threads << ",\n";
        report << "  \"gpu_layers\": " << opts.gpu_layers << ",\n";
        report << "  \"cache_type_k\": \"" << cache_type_name(opts.cache_type_k) << "\",\n";
        report << "  \"cache_type_v\": \"" << cache_type_name(opts.cache_type_v) << "\",\n";
        report << "  \"prompt_token_count\": " << prompt_tokens.size() << ",\n";
        report << "  \"prompt_tokens\": ";
        write_integer_array(report, prompt_tokens);
        report << ",\n";
        report << "  \"generated_token_count\": " << generated_tokens.size() << ",\n";
        report << "  \"generated_tokens\": ";
        write_integer_array(report, generated_tokens);
        report << ",\n";
        report << "  \"generated_text\": \"" << json_escape(generated_text) << "\",\n";
        report << "  \"eos_reached\": " << (eos_reached ? "true" : "false") << ",\n";
        report << "  \"logits\": {\n";
        report << "    \"format\": \"raw-f32-native-endian\",\n";
        report << "    \"rows\": " << generated_tokens.size() << ",\n";
        report << "    \"columns\": " << n_vocab << ",\n";
        report << "    \"path\": \"" << json_escape(opts.logits_bin) << "\"\n";
        report << "  },\n";
        report << "  \"timing\": {\n";
        report << std::setprecision(17);
        report << "    \"model_load_ms\": " << elapsed_ms(model_load_start, model_load_end) << ",\n";
        report << "    \"prefill_ms\": " << prefill_ms << ",\n";
        report << "    \"ttft_ms\": " << ttft_ms << ",\n";
        report << "    \"total_inference_ms\": " << total_inference_ms << ",\n";
        report << "    \"decode_tokens_per_second\": " << decode_tps << ",\n";
        report << "    \"decode_step_ms\": ";
        write_double_array(report, decode_step_ms);
        report << "\n  },\n";
        report << "  \"ccos_lifecycle\": {\n";
        report << "    \"requested\": " << (opts.ccos_cold_cycle_step >= 0 ? "true" : "false") << ",\n";
        report << "    \"executed\": " << (ccos_lifecycle_executed ? "true" : "false") << ",\n";
        report << "    \"step\": " << opts.ccos_cold_cycle_step << ",\n";
        report << "    \"before\": ";
        if (ccos_lifecycle_executed) write_store_snapshot(report, ccos_before); else report << "null";
        report << ",\n    \"cold\": ";
        if (ccos_lifecycle_executed) write_store_snapshot(report, ccos_cold); else report << "null";
        report << ",\n    \"restored\": ";
        if (ccos_lifecycle_executed) write_store_snapshot(report, ccos_restored); else report << "null";
        report << "\n  }\n";
        report << "}\n";
        report.close();

        llama_perf_context_print(ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "slha_real_eval: " << error.what() << "\n";
        return 1;
    }
}
