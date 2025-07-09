# Cross-Language Matrix Multiplication Benchmark Specification

## Overview
This specification defines a standardized benchmark for comparing matrix multiplication implementations across different programming languages.

## Benchmark Requirements

### 1. Matrix Specifications
- **Data Type**: 64-bit floating point numbers (double precision)
- **Matrix Sizes**: 100×100, 500×500, 1000×1000 (minimum required)
- **Initialization**: 
  - Matrix A: Random values between 0.0 and 1.0 with seed 42
  - Matrix B: Random values between 0.0 and 1.0 with seed 43
  - Use language-specific random number generators for reproducibility

### 2. Implementation Categories

#### Category 1: Native Optimized Libraries
- Use the most optimized linear algebra library available
- Examples: NumPy (Python), Eigen (C++), BLAS/LAPACK (Fortran), etc.
- **Goal**: Show peak performance for each language

#### Category 2: Standard Library
- Use built-in matrix operations if available
- No external optimization libraries
- **Goal**: Show baseline language performance

#### Category 3: Manual Implementation
- Triple-nested loop implementation
- No vectorization or optimization
- **Goal**: Show raw language performance

### 3. Timing Methodology

#### Measurement Protocol
1. **Warm-up**: 3 runs (not measured) to warm up JIT/caches
2. **Measurement**: 10 timed runs minimum
3. **Timing Scope**: Only matrix multiplication operation
4. **Exclude**: Memory allocation, initialization, result verification

#### Timing Implementation
```
// Pseudo-code for timing
for warmup in range(3):
    result = matmul(A, B)  // Not timed

times = []
for run in range(10):
    start_time = high_precision_timer()
    result = matmul(A, B)
    end_time = high_precision_timer()
    times.append(end_time - start_time)

// Report: mean, std_dev, min, max
```

### 4. Output Format (JSON)
```json
{
  "language": "Python",
  "version": "3.11.2",
  "implementation": "NumPy",
  "library_version": "1.26.4",
  "matrix_size": 100,
  "data_type": "float64",
  "num_runs": 10,
  "results": {
    "mean_time_ms": 0.068,
    "std_dev_ms": 0.052,
    "min_time_ms": 0.045,
    "max_time_ms": 0.121,
    "all_times_ms": [0.068, 0.045, 0.121, ...]
  },
  "system_info": {
    "os": "Linux",
    "cpu": "Intel Core i7-9700K",
    "cores": 8,
    "memory_gb": 32,
    "compiler": "GCC 12.2.0"
  }
}
```

### 5. System Requirements

#### Environment Standardization
- Same hardware for all language tests
- Same OS and kernel version
- Minimal background processes
- CPU frequency scaling disabled (if possible)

#### Compiler/Runtime Versions
- Use recent stable versions
- Document optimization flags
- Report runtime/compiler versions

### 6. Implementation Guidelines

#### Memory Management
- Pre-allocate result matrix
- Don't include allocation time in measurements
- Use stack allocation where possible

#### Optimization Levels
- **Category 1**: Maximum optimization (-O3, release builds)
- **Category 2**: Standard optimization (-O2, default)
- **Category 3**: Minimal optimization (-O1, debug friendly)

#### Threading
- Single-threaded implementations preferred for fair comparison
- Multi-threaded results should be clearly marked
- Report thread count used

### 7. Verification Protocol

#### Correctness Check
- Verify result against reference implementation
- Use relative tolerance: 1e-10 for double precision
- Check a few specific elements, not entire matrix

#### Reference Values (for verification)
Using seed 42 for A and seed 43 for B:
- 100×100: A[0,0] = 0.3745, B[0,0] = 0.3749, Result[0,0] ≈ 24.9999
- Result matrix sum should be within tolerance of expected value

### 8. Reporting Format

#### Individual Results
```
Language: Python 3.11.2
Implementation: NumPy 1.26.4
Matrix Size: 100x100
Mean Time: 0.068 ms ± 0.052 ms
Performance: 2.94 GFLOPS
```

#### Comparative Results
```
Matrix Size: 100x100
Rank | Language      | Implementation | Time (ms) | Speedup | GFLOPS
1    | C++           | Eigen         | 0.042     | 1.00x   | 4.76
2    | Python        | NumPy         | 0.068     | 0.62x   | 2.94
```

### 9. GFLOPS Calculation
```
GFLOPS = (2 * n^3 - n^2) / (time_in_seconds * 1e9)
```
Where n is the matrix dimension. For 100×100 matrices: ~2M operations.

### 10. File Organization
```
benchmarks/
├── python/
│   ├── benchmark_numpy.py
│   ├── benchmark_manual.py
│   └── results_python.json
├── cpp/
│   ├── benchmark_eigen.cpp
│   ├── benchmark_manual.cpp
│   └── results_cpp.json
└── results/
    ├── combined_results.json
    └── benchmark_report.md
```

This specification ensures fair, reproducible comparisons across languages while accounting for each language's strengths and characteristics.