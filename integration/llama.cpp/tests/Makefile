CXX      ?= g++
CXXFLAGS ?= -O2 -g

override CXXFLAGS += -std=c++17 -Wall -Wextra -Werror -I../shim

.PHONY: all clean test

all: replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests

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

test: replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests
	./replace_strict_tests
	./calibration_tests
	./layer_mask_tests
	./score_scale_tests
	./scale_fit_tests
	./score_oracle_tests
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
	  -I$(LLAMA_CPP_DIR)/ggml/include ../shim/slha_llama.cpp
else
	@echo "LLAMA_CPP_DIR not set: slha_llama.cpp syntax check SKIPPED (needs ggml.h)"
endif

clean:
	rm -f replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests
	rm -f *.check.o
	rm -rf calib_test_tmp
