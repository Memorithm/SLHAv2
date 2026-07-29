CXX      ?= g++
CXXFLAGS ?= -O2 -g

override CXXFLAGS += -std=c++17 -Wall -Wextra -Werror -I../shim

.PHONY: all clean test

all: replace_strict_tests calibration_tests

replace_strict_tests: replace_strict_tests.cpp ../shim/slha_replace_counters.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

calibration_tests: calibration_tests.cpp ../shim/slha_calibration.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test: replace_strict_tests calibration_tests
	./replace_strict_tests
	./calibration_tests
	bash ./trainer_atomicity_test.sh

clean:
	rm -f replace_strict_tests calibration_tests
	rm -rf calib_test_tmp
