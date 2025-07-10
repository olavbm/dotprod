# Makefile for benchmarking dot product implementations
# Usage: make benchmark, make benchmark-small, make benchmark-large, etc.

PYTHON = python3
STANDARDIZED_SCRIPT = benchmarks/src/python/benchmark_standardized.py
RUNS = 10
SIZE = 100
DTYPE = float64

# Compiler settings
CXX = g++
CXXFLAGS = -O3 -std=c++17
RUSTC = rustc
RUSTFLAGS = -C opt-level=3 -C target-feature=+avx2,+fma -C target-cpu=native

# Paths
BENCHMARK_DIR = benchmarks
SRC_DIR = $(BENCHMARK_DIR)/src
BIN_DIR = $(BENCHMARK_DIR)/bin
RESULTS_DIR = $(BENCHMARK_DIR)/results

.PHONY: help test clean
.PHONY: standardized standardized-all compile-cpp compile-rust run-cpp run-rust compare-languages

help:
	@echo "Available targets:"
	@echo ""
	@echo "STANDARDIZED BENCHMARKS:"
	@echo "  standardized    - Run standardized Python benchmark"
	@echo "  standardized-all- Run standardized benchmark for multiple sizes"
	@echo "  compile-cpp     - Compile C++ benchmark"
	@echo "  compile-rust    - Compile Rust benchmark"
	@echo "  run-cpp         - Run C++ benchmark"
	@echo "  run-rust        - Run Rust benchmark"
	@echo "  compare-languages - Run benchmarks for Python, C++, and Rust"
	@echo ""
	@echo "UTILITIES:"
	@echo "  test            - Run basic correctness test"
	@echo "  clean           - Clean up generated files"
	@echo "  check-deps      - Check if dependencies are installed"
	@echo ""
	@echo "PARAMETERS:"
	@echo "  SIZE=<int>      - Matrix size (default: 100)"
	@echo "  RUNS=<int>      - Number of benchmark runs (default: 10)"
	@echo "  DTYPE=<str>     - Data type (default: float64)"
	@echo ""
	@echo "Examples:"
	@echo "  make standardized SIZE=500 RUNS=20 DTYPE=float32"
	@echo "  make compare-languages SIZE=100 DTYPE=int32"
	@echo "  make run-rust SIZE=200 RUNS=3 DTYPE=float64"

test:
	@echo "Running correctness test..."
	$(PYTHON) $(STANDARDIZED_SCRIPT) --size 10 --runs 1 --dtype $(DTYPE)

# Clean up any generated files
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -f profile_output.txt
	rm -f $(BIN_DIR)/benchmark_*
	rm -f $(RESULTS_DIR)/python/*.json
	rm -f $(RESULTS_DIR)/cpp/*.json
	rm -f $(RESULTS_DIR)/rust/*.json

# Check if required dependencies are installed
check-deps:
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
	@$(PYTHON) -c "import statistics; print('Statistics module: OK')"
	@echo "All dependencies are installed."

# STANDARDIZED BENCHMARKS
standardized:
	@echo "Running standardized Python benchmark..."
	$(PYTHON) $(STANDARDIZED_SCRIPT) --size $(SIZE) --runs $(RUNS) --dtype $(DTYPE)

standardized-all:
	@echo "Running standardized benchmark for multiple sizes..."
	$(PYTHON) $(STANDARDIZED_SCRIPT) --sizes 100 500 1000 --runs $(RUNS) --dtype $(DTYPE)

# Compile C++ benchmark
compile-cpp:
	@echo "Compiling C++ benchmark..."
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/benchmark_cpp $(SRC_DIR)/cpp/benchmark_example.cpp

# Compile C++ benchmark with Eigen (if available)
compile-cpp-eigen:
	@echo "Compiling C++ benchmark with Eigen..."
	$(CXX) $(CXXFLAGS) -DUSE_EIGEN -o $(BIN_DIR)/benchmark_cpp_eigen $(SRC_DIR)/cpp/benchmark_example.cpp

# Compile Rust benchmark
compile-rust:
	@echo "Compiling Rust benchmark..."
	$(RUSTC) $(RUSTFLAGS) -o $(BIN_DIR)/benchmark_rust $(SRC_DIR)/rust/benchmark_example.rs

# Run C++ benchmark
run-cpp: compile-cpp
	@echo "Running C++ benchmark..."
	$(BIN_DIR)/benchmark_cpp $(SIZE) $(RUNS) $(DTYPE)

# Run Rust benchmark
run-rust: compile-rust
	@echo "Running Rust benchmark..."
	$(BIN_DIR)/benchmark_rust $(SIZE) $(RUNS) $(DTYPE)

# Compare Python, C++, and Rust
compare-languages: compile-cpp compile-rust
	@echo "========================================="
	@echo "CROSS-LANGUAGE BENCHMARK COMPARISON"
	@echo "Matrix Size: $(SIZE)x$(SIZE)"
	@echo "Data Type: $(DTYPE)"
	@echo "Runs: $(RUNS)"
	@echo "========================================="
	@echo ""
	@echo "PYTHON RESULTS:"
	@echo "---------------"
	@$(PYTHON) $(STANDARDIZED_SCRIPT) --size $(SIZE) --runs $(RUNS) --dtype $(DTYPE)
	@echo ""
	@echo "C++ RESULTS:"
	@echo "------------"
	@$(BIN_DIR)/benchmark_cpp $(SIZE) $(RUNS) $(DTYPE)
	@echo ""
	@echo "RUST RESULTS:"
	@echo "-------------"
	@$(BIN_DIR)/benchmark_rust $(SIZE) $(RUNS) $(DTYPE)

# Generate standardized benchmark report
report:
	@echo "Generating benchmark report..."
	@echo "Date: $$(date)" > benchmark_report.txt
	@echo "System: $$(uname -a)" >> benchmark_report.txt
	@echo "Python: $$($(PYTHON) --version)" >> benchmark_report.txt
	@echo "NumPy: $$($(PYTHON) -c 'import numpy; print(numpy.__version__)')" >> benchmark_report.txt
	@echo "Data Type: $(DTYPE)" >> benchmark_report.txt
	@echo "" >> benchmark_report.txt
	@echo "Benchmark Results:" >> benchmark_report.txt
	@echo "==================" >> benchmark_report.txt
	$(PYTHON) $(STANDARDIZED_SCRIPT) --sizes 50 100 200 500 --runs 10 --dtype $(DTYPE) >> benchmark_report.txt
	@echo "Report generated: benchmark_report.txt"

# Make the benchmark script executable
make-executable:
	chmod +x $(BENCHMARK_SCRIPT)
	chmod +x $(STANDARDIZED_SCRIPT)