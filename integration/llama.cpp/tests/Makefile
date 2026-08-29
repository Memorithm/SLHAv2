CXX      ?= g++
CXXFLAGS ?= -O2 -g

override CXXFLAGS += -std=c++17 -Wall -Wextra -Werror -I../shim

.PHONY: all clean test

all: replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests external_k_contract_tests tile_store_tests

replace_strict_tests: replace_strict_tests.cpp ../shim/slha_replace_counters.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

calibration_tests: calibration_tests.cpp ../shim/slha_calibration.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

layer_mask_tests: layer_mask_tests.cpp ../shim/slha_layer_mask.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

score_scale_tests: score_scale_tests.cpp ../shim/slha_score_scale.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

scale_fit_tests: scale_fit_tests.cpp ../shim/slha_scale_fit.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

score_oracle_tests: score_oracle_tests.cpp ../shim/slha_score_oracle.cpp ../shim/slha_score_scale.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

external_k_contract_tests: external_k_contract_tests.cpp ../shim/slha_external_k.cpp
	$(CXX) $(CXXFLAGS) -I../../../slha-c/include -o $@ $^

tile_store_tests: tile_store_tests.cpp ../shim/slha_llama.cpp ../shim/slha_external_k.cpp ../shim/slha_replace_counters.cpp ../shim/slha_layer_mask.cpp ../shim/slha_score_scale.cpp ../shim/slha_scale_fit.cpp ../shim/slha_score_oracle.cpp ../shim/slha_oracle_metrics.cpp ../shim/slha_rank_dataset.cpp
	@test -n "$(LLAMA_CPP_DIR)" || { echo "ERROR: LLAMA_CPP_DIR must point to a built llama.cpp checkout"; exit 2; }
	@test -f "$(LLAMA_CPP_DIR)/ggml/include/ggml.h" || { echo "ERROR: missing $(LLAMA_CPP_DIR)/ggml/include/ggml.h"; exit 2; }
	@test -f "$(LLAMA_CPP_DIR)/build/bin/libllama.so" || { echo "ERROR: llama.cpp is not built: missing build/bin/libllama.so"; exit 2; }
	@test -n "$(SLHA_LIB)" || { echo "ERROR: SLHA_LIB must point to libslha.a"; exit 2; }
	@test -f "$(SLHA_LIB)" || { echo "ERROR: missing SLHA static library: $(SLHA_LIB)"; exit 2; }
	$(CXX) $(CXXFLAGS) -pthread -Wno-unused-function -I../../../slha-c/include \
	  -I$(LLAMA_CPP_DIR)/ggml/include -o $@ $^ $(SLHA_LIB) \
	  -L$(LLAMA_CPP_DIR)/build/bin -lllama -lggml -lggml-base -lggml-cpu \
	  -Wl,-rpath,$(LLAMA_CPP_DIR)/build/bin

test: replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests external_k_contract_tests tile_store_tests
	./replace_strict_tests
	./calibration_tests
	./layer_mask_tests
	./score_scale_tests
	./scale_fit_tests
	./score_oracle_tests
	./external_k_contract_tests
	./tile_store_tests
	bash ./trainer_atomicity_test.sh

# Strict compile gate for the shim sources the test binaries do NOT link.
# Three are self-contained; slha_llama.cpp is the only shim file needing
# external headers (slha.h from ../../..//slha-c, ggml.h from a llama.cpp
# checkout) and gets a -fsyntax-only pass against the REAL pinned headers —
# never against stubs, which could drift from upstream. Point LLAMA_CPP_DIR at
# a checkout of the tag pinned in ../build_and_roundtrip.sh (b9860).
LLAMA_CPP_DIR ?=

compile-check:
	$(CXX) $(CXXFLAGS) -c ../shim/slha_calibrate_cli.cpp -o slha_calibrate_cli.check.o
	$(CXX) $(CXXFLAGS) -c ../shim/slha_oracle_metrics.cpp -o slha_oracle_metrics.check.o
	$(CXX) $(CXXFLAGS) -c ../shim/slha_rank_dataset.cpp -o slha_rank_dataset.check.o
ifneq ($(LLAMA_CPP_DIR),)
	$(CXX) $(CXXFLAGS) -fsyntax-only -I../../../slha-c/include \
	  -I$(LLAMA_CPP_DIR)/ggml/include ../shim/slha_llama.cpp ../shim/slha_external_k.cpp
else
	@echo "LLAMA_CPP_DIR not set: slha_llama.cpp syntax check SKIPPED (needs ggml.h)"
endif

clean:
	rm -f replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests external_k_contract_tests tile_store_tests
	rm -f *.check.o
	rm -rf calib_test_tmp
