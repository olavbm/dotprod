#!/usr/bin/env python3
"""
Standardized cross-language matrix multiplication benchmark.
Follows the benchmark specification for fair language comparisons.
"""

import time
import json
import numpy as np
import platform
import sys
import argparse
from typing import Dict, List, Tuple, Any


class StandardizedBenchmark:
    """Standardized benchmark following cross-language specification."""
    
    def __init__(self, matrix_size: int = 100, num_runs: int = 10, warmup_runs: int = 3, dtype: str = "float64"):
        self.matrix_size = matrix_size
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.dtype_str = dtype
        self.dtype = self._get_numpy_dtype(dtype)
        self.results = {}
        
    def _get_numpy_dtype(self, dtype_str: str):
        """Convert string dtype to numpy dtype."""
        dtype_map = {
            "float64": np.float64,
            "float32": np.float32,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
            "f64": np.float64,
            "f32": np.float32,
            "i8": np.int8,
            "i16": np.int16,
            "i32": np.int32,
            "i64": np.int64,
            "u8": np.uint8,
            "u16": np.uint16,
            "u32": np.uint32,
            "u64": np.uint64,
        }
        
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_str}. Supported types: {list(dtype_map.keys())}")
        
        return dtype_map[dtype_str]
        
    def generate_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate reproducible test matrices according to specification."""
        # Matrix A: seed 42
        np.random.seed(42)
        if self.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            # For integer types, generate values in appropriate range
            if self.dtype == np.int8:
                A = np.random.randint(-10, 10, (self.matrix_size, self.matrix_size), dtype=self.dtype)
            elif self.dtype == np.uint8:
                A = np.random.randint(0, 20, (self.matrix_size, self.matrix_size), dtype=self.dtype)
            elif self.dtype in [np.int16, np.uint16]:
                A = np.random.randint(-100, 100, (self.matrix_size, self.matrix_size), dtype=self.dtype)
            else:  # int32, int64, uint32, uint64
                A = np.random.randint(-1000, 1000, (self.matrix_size, self.matrix_size), dtype=self.dtype)
        else:
            # For floating point types
            A = np.random.rand(self.matrix_size, self.matrix_size).astype(self.dtype)
        
        # Matrix B: seed 43
        np.random.seed(43)
        if self.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            # For integer types, generate values in appropriate range
            if self.dtype == np.int8:
                B = np.random.randint(-10, 10, (self.matrix_size, self.matrix_size), dtype=self.dtype)
            elif self.dtype == np.uint8:
                B = np.random.randint(0, 20, (self.matrix_size, self.matrix_size), dtype=self.dtype)
            elif self.dtype in [np.int16, np.uint16]:
                B = np.random.randint(-100, 100, (self.matrix_size, self.matrix_size), dtype=self.dtype)
            else:  # int32, int64, uint32, uint64
                B = np.random.randint(-1000, 1000, (self.matrix_size, self.matrix_size), dtype=self.dtype)
        else:
            # For floating point types
            B = np.random.rand(self.matrix_size, self.matrix_size).astype(self.dtype)
        
        return A, B
    
    def verify_result(self, result: np.ndarray, A: np.ndarray, B: np.ndarray) -> bool:
        """Verify result correctness using reference implementation."""
        reference = np.dot(A, B)
        
        # Set tolerance based on data type
        if self.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            # For integer types, use exact comparison
            tolerance = 0
            return np.array_equal(result, reference)
        elif self.dtype == np.float32:
            tolerance = 1e-6
        else:  # float64 and other float types
            tolerance = 1e-10
        
        # Check specific elements and overall tolerance
        if not np.allclose(result[0, 0], reference[0, 0], rtol=tolerance):
            return False
        
        # Check overall matrix
        if not np.allclose(result, reference, rtol=tolerance):
            return False
            
        return True
    
    def calculate_gflops(self, time_seconds: float) -> float:
        """Calculate GFLOPS according to specification."""
        n = self.matrix_size
        flops = 2 * n**3 - n**2  # Matrix multiplication FLOP count
        return flops / (time_seconds * 1e9)
    
    def benchmark_implementation(self, impl_func, impl_name: str, 
                                library_version: str = "") -> Dict[str, Any]:
        """Benchmark a specific implementation."""
        print(f"Benchmarking {impl_name}...")
        
        # Generate matrices
        A, B = self.generate_matrices()
        
        # Pre-allocate result matrix
        result = np.zeros((self.matrix_size, self.matrix_size), dtype=self.dtype)
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            _ = impl_func(A, B)
        
        # Timed runs
        times = []
        for _ in range(self.num_runs):
            start_time = time.perf_counter()
            result = impl_func(A, B)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Verify correctness
        if not self.verify_result(result, A, B):
            raise ValueError(f"Implementation {impl_name} failed correctness check")
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]  # Convert to milliseconds
        mean_time_ms = np.mean(times_ms)
        std_dev_ms = np.std(times_ms)
        min_time_ms = np.min(times_ms)
        max_time_ms = np.max(times_ms)
        
        # Calculate GFLOPS
        gflops = self.calculate_gflops(mean_time_ms / 1000)
        
        return {
            "implementation": impl_name,
            "library_version": library_version,
            "matrix_size": self.matrix_size,
            "data_type": self.dtype_str,
            "num_runs": self.num_runs,
            "results": {
                "mean_time_ms": round(mean_time_ms, 6),
                "std_dev_ms": round(std_dev_ms, 6),
                "min_time_ms": round(min_time_ms, 6),
                "max_time_ms": round(max_time_ms, 6),
                "all_times_ms": [round(t, 6) for t in times_ms],
                "gflops": round(gflops, 3)
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "cpu": platform.processor() or "Unknown",
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform()
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark implementations."""
        results = {
            "language": "Python",
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.get_system_info(),
            "benchmarks": []
        }
        
        # Implementation 1: NumPy @ operator (optimized)
        try:
            result = self.benchmark_implementation(
                lambda A, B: A @ B,
                "NumPy @ operator",
                np.__version__
            )
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"Error benchmarking NumPy @ operator: {e}")
        
        # Implementation 2: NumPy np.dot (optimized)
        try:
            result = self.benchmark_implementation(
                lambda A, B: np.dot(A, B),
                "NumPy np.dot",
                np.__version__
            )
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"Error benchmarking NumPy np.dot: {e}")
        
        # Implementation 3: Manual implementation (baseline)
        try:
            result = self.benchmark_implementation(
                self.manual_matmul,
                "Manual Python",
                "native"
            )
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"Error benchmarking manual implementation: {e}")
        
        return results
    
    def manual_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Manual matrix multiplication implementation."""
        rows_a, cols_a = A.shape
        rows_b, cols_b = B.shape
        
        if cols_a != rows_b:
            raise ValueError("Matrix dimensions don't match")
        
        result = np.zeros((rows_a, cols_b), dtype=self.dtype)
        
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i, j] += A[i, k] * B[k, j]
        
        return result
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"results_python_{self.matrix_size}x{self.matrix_size}_{self.dtype_str}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results."""
        print(f"\n{'='*80}")
        print(f"STANDARDIZED BENCHMARK RESULTS")
        print(f"{'='*80}")
        print(f"Language: {results['language']} {results['version']}")
        print(f"Matrix Size: {self.matrix_size}×{self.matrix_size}")
        print(f"Data Type: {self.dtype_str}")
        print(f"Runs: {self.num_runs}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"NumPy Version: {results['system_info']['numpy_version']}")
        print(f"{'='*80}")
        
        print(f"\n{'Implementation':<20} {'Time (ms)':<12} {'±Std (ms)':<12} {'GFLOPS':<10} {'Category'}")
        print(f"{'-'*80}")
        
        for bench in results['benchmarks']:
            impl_name = bench['implementation']
            mean_time = bench['results']['mean_time_ms']
            std_dev = bench['results']['std_dev_ms']
            gflops = bench['results']['gflops']
            
            # Categorize implementation
            if 'NumPy' in impl_name:
                category = "Optimized"
            else:
                category = "Manual"
            
            print(f"{impl_name:<20} {mean_time:<12.3f} {std_dev:<12.3f} {gflops:<10.1f} {category}")
        
        print(f"\n{'='*80}")
        
        # Print reference values for verification
        A, B = self.generate_matrices()
        print(f"Reference values (for verification):")
        print(f"A[0,0] = {A[0,0]:.6f}, B[0,0] = {B[0,0]:.6f}")
        result = np.dot(A, B)
        print(f"Result[0,0] = {result[0,0]:.6f}")
        print(f"Result sum = {np.sum(result):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Standardized matrix multiplication benchmark')
    parser.add_argument('--size', type=int, default=100, 
                       help='Matrix size (default: 100)')
    parser.add_argument('--runs', type=int, default=10, 
                       help='Number of benchmark runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=3, 
                       help='Number of warmup runs (default: 3)')
    parser.add_argument('--output', type=str, 
                       help='Output JSON file (default: auto-generated)')
    parser.add_argument('--sizes', nargs='+', type=int, 
                       help='Multiple matrix sizes to test')
    parser.add_argument('--dtype', type=str, default='float64',
                       help='Data type for matrices (default: float64). Supported: float64, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64, f64, f32, i8, i16, i32, i64, u8, u16, u32, u64')
    
    args = parser.parse_args()
    
    # Test multiple sizes if specified
    sizes = args.sizes if args.sizes else [args.size]
    
    for size in sizes:
        print(f"\nRunning benchmark for {size}×{size} matrices...")
        
        benchmark = StandardizedBenchmark(
            matrix_size=size,
            num_runs=args.runs,
            warmup_runs=args.warmup,
            dtype=args.dtype
        )
        
        results = benchmark.run_all_benchmarks()
        benchmark.print_results(results)
        
        # Save results
        output_file = args.output
        if not output_file:
            output_file = f"results_python_{size}x{size}_{args.dtype}.json"
        
        benchmark.save_results(results, output_file)


if __name__ == "__main__":
    main()