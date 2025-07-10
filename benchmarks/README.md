# Matrix Multiplication Benchmarks

This directory contains cross-language matrix multiplication benchmarks comparing Python, C++, and Rust implementations.

## Quick Start

```bash
# Run all benchmarks
make compare-languages

# Run specific language
make run-rust SIZE=100 RUNS=10 DTYPE=float32
make run-cpp SIZE=200 RUNS=5 DTYPE=float64
make standardized SIZE=500 RUNS=20 DTYPE=int32

# Test different data types
make run-rust DTYPE=float32
make run-rust DTYPE=float64
make run-rust DTYPE=int32
make run-rust DTYPE=int8
```

## Directory Structure

```
benchmarks/
├── src/                    # Source code
│   ├── python/            # Python implementations
│   ├── cpp/               # C++ implementations
│   └── rust/              # Rust implementations
├── bin/                   # Compiled binaries (git-ignored)
├── results/               # Benchmark results
│   ├── python/            # Python results (.json)
│   ├── cpp/               # C++ results (.json)
│   ├── rust/              # Rust results (.json)
│   └── analysis/          # Analysis reports
└── BENCHMARK_SPEC.md      # Detailed benchmark specification
```

## Supported Data Types

- **float64/f64**: Double precision floating point
- **float32/f32**: Single precision floating point  
- **int32/i32**: 32-bit signed integer
- **int8/i8**: 8-bit signed integer

## Performance Results Summary

Based on 100x100 matrix benchmarks:

| Data Type | Fastest Implementation | Performance |
|-----------|----------------------|-------------|
| float32   | NumPy @ operator     | 36.1 GFLOPS |
| float64   | NumPy @ operator     | 27.6 GFLOPS |
| int32     | Rust Flat Array      | 6.91 GFLOPS |
| int8      | Rust Flat Array      | 10.13 GFLOPS |

**Key Findings:**
- float32 is fastest overall due to better SIMD utilization
- NumPy dominates floating-point operations
- Rust shows consistent performance across data types
- Manual implementations are ~1000x slower than optimized libraries

## Build Requirements

- **Python**: NumPy 1.26.4+
- **C++**: g++ with C++17 support
- **Rust**: rustc with AVX2 support
- **Make**: GNU Make

## Usage Examples

```bash
# Quick performance test
make test

# Compare all languages with float32
make compare-languages DTYPE=float32 SIZE=200

# Generate comprehensive report
make report

# Clean up binaries and results
make clean
```

See [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) for detailed benchmark methodology.