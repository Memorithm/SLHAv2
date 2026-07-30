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

clean:
	rm -f replace_strict_tests calibration_tests layer_mask_tests score_scale_tests scale_fit_tests score_oracle_tests
	rm -rf calib_test_tmp
