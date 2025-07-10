#!/usr/bin/env python3
"""
NumPy-only benchmark for large matrices
"""

import time
import numpy as np
import sys

def benchmark_numpy(matrix_size, num_runs=1):
    print(f"NumPy Matrix Multiplication Benchmark")
    print(f"Matrix Size: {matrix_size}×{matrix_size}")
    print(f"Runs: {num_runs}")
    print("="*50)
    
    # Generate matrices with same seeds as Rust
    np.random.seed(42)
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    np.random.seed(43)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    
    def benchmark_impl(name, func):
        print(f"Benchmarking {name}...")
        
        # Warmup
        _ = func(A, B)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = func(A, B)
            end = time.perf_counter()
            times.append(end - start)
        
        mean_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate GFLOPS
        flops = 2 * matrix_size**3 - matrix_size**2
        gflops = flops / (mean_time * 1e9)
        
        print(f"{name}:")
        print(f"  Mean time: {mean_time:.3f} s ± {std_time:.3f} s")
        print(f"  Min time:  {min_time:.3f} s")
        print(f"  Max time:  {max_time:.3f} s")
        print(f"  GFLOPS:    {gflops:.3f}")
        print()
        
        return result
    
    # NumPy implementations
    result1 = benchmark_impl("NumPy @ operator", lambda A, B: A @ B)
    result2 = benchmark_impl("NumPy np.dot", lambda A, B: np.dot(A, B))
    result3 = benchmark_impl("NumPy np.matmul", lambda A, B: np.matmul(A, B))
    
    # Reference values
    print("Reference values:")
    print(f"A[0,0] = {A[0,0]:.6f}")
    print(f"B[0,0] = {B[0,0]:.6f}")
    print(f"Result[0,0] = {result1[0,0]:.6f}")

if __name__ == "__main__":
    matrix_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    benchmark_numpy(matrix_size, num_runs)