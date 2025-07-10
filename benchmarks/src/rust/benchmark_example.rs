/*
 * Standardized Rust Matrix Multiplication Benchmark
 * Follows the cross-language benchmark specification
 */

use std::env;
use std::time::Instant;
use std::fmt::Display;

// SIMD imports for the optimized implementations (preserved for later)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Trait to define matrix operations
trait MatrixOps: Copy + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + std::ops::AddAssign + Display {
    fn from_seed(seed: u64) -> Self;
    
    // SIMD support - default to false for most types
    fn has_simd_support() -> bool {
        false
    }
}

impl MatrixOps for f64 {
    fn from_seed(seed: u64) -> Self {
        (seed as f64) / (0x7fffffff as f64)
    }
    
    fn has_simd_support() -> bool {
        true
    }
}

impl MatrixOps for f32 {
    fn from_seed(seed: u64) -> Self {
        (seed as f32) / (0x7fffffff as f32)
    }
    
    fn has_simd_support() -> bool {
        true
    }
}

impl MatrixOps for i32 {
    fn from_seed(seed: u64) -> Self {
        ((seed % 2001) as i32) - 1000 // Range -1000 to 1000
    }
}

impl MatrixOps for i8 {
    fn from_seed(seed: u64) -> Self {
        ((seed % 21) as i8) - 10 // Range -10 to 10
    }
}

struct StandardizedBenchmark<T: MatrixOps> {
    matrix_size: usize,
    num_runs: usize,
    warmup_runs: usize,
    dtype_str: String,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: MatrixOps> StandardizedBenchmark<T> {
    fn new(matrix_size: usize, num_runs: usize, warmup_runs: usize, dtype_str: String) -> Self {
        Self {
            matrix_size,
            num_runs,
            warmup_runs,
            dtype_str,
            _phantom: std::marker::PhantomData,
        }
    }

    fn generate_matrices(&self) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
        let generate_matrix = |seed: u64| {
            let mut matrix = vec![vec![T::default(); self.matrix_size]; self.matrix_size];
            let mut rng_state = seed;
            
            for row in matrix.iter_mut() {
                for cell in row.iter_mut() {
                    rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                    *cell = T::from_seed(rng_state);
                }
            }
            matrix
        };

        (generate_matrix(42), generate_matrix(43))
    }

    fn manual_matmul(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut result = vec![vec![T::default(); self.matrix_size]; self.matrix_size];
        
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                for k in 0..self.matrix_size {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        result
    }

    fn optimized_matmul(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut result = vec![vec![T::default(); self.matrix_size]; self.matrix_size];
        
        // Cache-friendly loop ordering (ikj)
        for i in 0..self.matrix_size {
            for k in 0..self.matrix_size {
                let aik = a[i][k];
                for j in 0..self.matrix_size {
                    result[i][j] += aik * b[k][j];
                }
            }
        }
        
        result
    }

    fn highly_optimized_matmul(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut result = vec![vec![T::default(); self.matrix_size]; self.matrix_size];
        
        // Block size for cache optimization
        const BLOCK_SIZE: usize = 64;
        
        // Blocked matrix multiplication for better cache utilization
        for ii in (0..self.matrix_size).step_by(BLOCK_SIZE) {
            for jj in (0..self.matrix_size).step_by(BLOCK_SIZE) {
                for kk in (0..self.matrix_size).step_by(BLOCK_SIZE) {
                    let i_max = (ii + BLOCK_SIZE).min(self.matrix_size);
                    let j_max = (jj + BLOCK_SIZE).min(self.matrix_size);
                    let k_max = (kk + BLOCK_SIZE).min(self.matrix_size);
                    
                    for i in ii..i_max {
                        for k in kk..k_max {
                            let aik = a[i][k];
                            for j in jj..j_max {
                                result[i][j] += aik * b[k][j];
                            }
                        }
                    }
                }
            }
        }
        
        result
    }

    fn flat_array_matmul(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        // Use flat arrays for better memory layout
        let mut a_flat = vec![T::default(); self.matrix_size * self.matrix_size];
        let mut b_flat = vec![T::default(); self.matrix_size * self.matrix_size];
        let mut result_flat = vec![T::default(); self.matrix_size * self.matrix_size];
        
        // Convert to flat arrays
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                a_flat[i * self.matrix_size + j] = a[i][j];
                b_flat[i * self.matrix_size + j] = b[i][j];
            }
        }
        
        // Optimized flat multiplication
        for i in 0..self.matrix_size {
            for k in 0..self.matrix_size {
                let aik = a_flat[i * self.matrix_size + k];
                let base_idx = i * self.matrix_size;
                let b_base = k * self.matrix_size;
                
                for j in 0..self.matrix_size {
                    result_flat[base_idx + j] += aik * b_flat[b_base + j];
                }
            }
        }
        
        // Convert back to 2D
        let mut result = vec![vec![T::default(); self.matrix_size]; self.matrix_size];
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                result[i][j] = result_flat[i * self.matrix_size + j];
            }
        }
        
        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_matmul_f64(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        
        // SIMD-optimized matrix multiplication using AVX2
        for i in 0..self.matrix_size {
            for k in 0..self.matrix_size {
                let aik = _mm256_set1_pd(a[i][k]);
                let mut j = 0;
                
                // Process 4 elements at a time with AVX2
                while j + 4 <= self.matrix_size {
                    let b_vec = _mm256_loadu_pd(&b[k][j] as *const f64);
                    let result_vec = _mm256_loadu_pd(&result[i][j] as *const f64);
                    let mul_result = _mm256_fmadd_pd(aik, b_vec, result_vec);
                    _mm256_storeu_pd(&mut result[i][j] as *mut f64, mul_result);
                    j += 4;
                }
                
                // Handle remaining elements
                while j < self.matrix_size {
                    result[i][j] += a[i][k] * b[k][j];
                    j += 1;
                }
            }
        }
        
        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_matmul_f32(&self, a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        
        // SIMD-optimized matrix multiplication using AVX2 for f32
        for i in 0..self.matrix_size {
            for k in 0..self.matrix_size {
                let aik = _mm256_set1_ps(a[i][k]);
                let mut j = 0;
                
                // Process 8 elements at a time with AVX2 for f32
                while j + 8 <= self.matrix_size {
                    let b_vec = _mm256_loadu_ps(&b[k][j] as *const f32);
                    let result_vec = _mm256_loadu_ps(&result[i][j] as *const f32);
                    let mul_result = _mm256_fmadd_ps(aik, b_vec, result_vec);
                    _mm256_storeu_ps(&mut result[i][j] as *mut f32, mul_result);
                    j += 8;
                }
                
                // Handle remaining elements
                while j < self.matrix_size {
                    result[i][j] += a[i][k] * b[k][j];
                    j += 1;
                }
            }
        }
        
        result
    }

    fn unsafe_optimized_matmul(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut result = vec![vec![T::default(); self.matrix_size]; self.matrix_size];
        
        // Use unsafe for bounds check elimination
        unsafe {
            for i in 0..self.matrix_size {
                for k in 0..self.matrix_size {
                    let aik = *a.get_unchecked(i).get_unchecked(k);
                    let result_row = result.get_unchecked_mut(i);
                    let b_row = b.get_unchecked(k);
                    
                    for j in 0..self.matrix_size {
                        *result_row.get_unchecked_mut(j) += aik * *b_row.get_unchecked(j);
                    }
                }
            }
        }
        
        result
    }

    // Generic SIMD dispatcher that works with supported types
    fn simd_matmul_generic(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        // Since we can't do runtime type checking directly with generics,
        // we'll use a different approach - fall back to the best non-SIMD implementation
        // This is a safe fallback that works for all types
        self.unsafe_optimized_matmul(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn flat_simd_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        // Convert to flat arrays for better memory layout
        let mut a_flat = vec![0.0; self.matrix_size * self.matrix_size];
        let mut b_flat = vec![0.0; self.matrix_size * self.matrix_size];
        let mut result_flat = vec![0.0; self.matrix_size * self.matrix_size];
        
        // Convert to flat arrays (row-major order)
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                a_flat[i * self.matrix_size + j] = a[i][j];
                b_flat[i * self.matrix_size + j] = b[i][j];
            }
        }
        
        // SIMD-optimized flat matrix multiplication
        for i in 0..self.matrix_size {
            for k in 0..self.matrix_size {
                let aik = _mm256_set1_pd(a_flat[i * self.matrix_size + k]);
                let result_base = i * self.matrix_size;
                let b_base = k * self.matrix_size;
                
                let mut j = 0;
                
                // Process 4 elements at a time with AVX2
                while j + 4 <= self.matrix_size {
                    let b_vec = _mm256_loadu_pd(b_flat.as_ptr().add(b_base + j));
                    let result_vec = _mm256_loadu_pd(result_flat.as_ptr().add(result_base + j));
                    let mul_result = _mm256_fmadd_pd(aik, b_vec, result_vec);
                    _mm256_storeu_pd(result_flat.as_mut_ptr().add(result_base + j), mul_result);
                    j += 4;
                }
                
                // Handle remaining elements
                while j < self.matrix_size {
                    result_flat[result_base + j] += a_flat[i * self.matrix_size + k] * b_flat[b_base + j];
                    j += 1;
                }
            }
        }
        
        // Convert back to 2D format
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                result[i][j] = result_flat[i * self.matrix_size + j];
            }
        }
        
        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn ultimate_optimized_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        // Ultimate optimization: Flat arrays + SIMD + Cache blocking + Unsafe
        let mut a_flat = vec![0.0; self.matrix_size * self.matrix_size];
        let mut b_flat = vec![0.0; self.matrix_size * self.matrix_size];
        let mut result_flat = vec![0.0; self.matrix_size * self.matrix_size];
        
        // Convert to flat arrays
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                a_flat[i * self.matrix_size + j] = a[i][j];
                b_flat[i * self.matrix_size + j] = b[i][j];
            }
        }
        
        // Block size optimized for cache
        const BLOCK_SIZE: usize = 64;
        
        // Blocked SIMD matrix multiplication
        for ii in (0..self.matrix_size).step_by(BLOCK_SIZE) {
            for kk in (0..self.matrix_size).step_by(BLOCK_SIZE) {
                for jj in (0..self.matrix_size).step_by(BLOCK_SIZE) {
                    let i_max = (ii + BLOCK_SIZE).min(self.matrix_size);
                    let k_max = (kk + BLOCK_SIZE).min(self.matrix_size);
                    let j_max = (jj + BLOCK_SIZE).min(self.matrix_size);
                    
                    for i in ii..i_max {
                        for k in kk..k_max {
                            let aik = _mm256_set1_pd(a_flat[i * self.matrix_size + k]);
                            let result_base = i * self.matrix_size;
                            let b_base = k * self.matrix_size;
                            
                            let mut j = jj;
                            
                            // SIMD processing within block
                            while j + 4 <= j_max {
                                let b_vec = _mm256_loadu_pd(b_flat.as_ptr().add(b_base + j));
                                let result_vec = _mm256_loadu_pd(result_flat.as_ptr().add(result_base + j));
                                let mul_result = _mm256_fmadd_pd(aik, b_vec, result_vec);
                                _mm256_storeu_pd(result_flat.as_mut_ptr().add(result_base + j), mul_result);
                                j += 4;
                            }
                            
                            // Handle remaining elements in block
                            while j < j_max {
                                result_flat[result_base + j] += a_flat[i * self.matrix_size + k] * b_flat[b_base + j];
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
        
        // Convert back to 2D format
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                result[i][j] = result_flat[i * self.matrix_size + j];
            }
        }
        
        result
    }

    fn calculate_gflops(&self, time_seconds: f64) -> f64 {
        let n = self.matrix_size as u64;
        let flops = 2 * n * n * n - n * n;
        flops as f64 / (time_seconds * 1e9)
    }

    fn benchmark_implementation<F>(&self, name: &str, func: F, a: &Vec<Vec<T>>, b: &Vec<Vec<T>>)
    where
        F: Fn(&Vec<Vec<T>>, &Vec<Vec<T>>) -> Vec<Vec<T>>,
    {
        println!("Benchmarking {}...", name);
        
        // Warmup runs
        for _ in 0..self.warmup_runs {
            let _result = func(a, b);
        }
        
        // Timed runs
        let mut times = Vec::new();
        for _ in 0..self.num_runs {
            let start = Instant::now();
            let _result = func(a, b);
            let duration = start.elapsed();
            times.push(duration.as_secs_f64());
        }
        
        // Calculate statistics
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter()
            .map(|&t| (t - mean_time).powi(2))
            .sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gflops = self.calculate_gflops(mean_time);
        
        // Print results
        println!("{}:", name);
        println!("  Mean time: {:.3} ms ± {:.3} ms", mean_time * 1000.0, std_dev * 1000.0);
        println!("  Min time:  {:.3} ms", min_time * 1000.0);
        println!("  Max time:  {:.3} ms", max_time * 1000.0);
        println!("  GFLOPS:    {:.3}", gflops);
        println!();
    }

    fn run_naive_benchmark(&self) {
        println!("================================");
        println!("Rust Matrix Multiplication Benchmark (Naive Implementation)");
        println!("Matrix Size: {}×{}", self.matrix_size, self.matrix_size);
        println!("Data Type: {}", self.dtype_str);
        println!("Runs: {}", self.num_runs);
        println!("Rust Version: {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "Unknown".to_string()));
        println!("================================");
        
        let (a, b) = self.generate_matrices();
        
        // Manual implementation only
        self.benchmark_implementation("Manual Rust", |a, b| self.manual_matmul(a, b), &a, &b);
        
        // Print reference values
        println!("Reference values:");
        println!("A[0,0] = {}", a[0][0]);
        println!("B[0,0] = {}", b[0][0]);
        
        let result = self.manual_matmul(&a, &b);
        println!("Result[0,0] = {}", result[0][0]);
    }

    fn run_benchmarks(&self) {
        print_benchmark_header(self.matrix_size, self.num_runs, &self.dtype_str, false);
        
        let (a, b) = self.generate_matrices();
        run_generic_benchmarks(self, &a, &b);
        print_reference_values(self, &a, &b);
    }
}

fn run_benchmark<T: MatrixOps>(matrix_size: usize, num_runs: usize, dtype: String) {
    let benchmark = StandardizedBenchmark::<T>::new(matrix_size, num_runs, 3, dtype);
    benchmark.run_naive_benchmark();
}

// Helper function to print benchmark header
fn print_benchmark_header(matrix_size: usize, num_runs: usize, dtype: &str, simd_enabled: bool) {
    println!("================================");
    if simd_enabled {
        println!("Rust Matrix Multiplication Benchmark (with SIMD)");
    } else {
        println!("Rust Matrix Multiplication Benchmark");
    }
    println!("Matrix Size: {}×{}", matrix_size, matrix_size);
    println!("Data Type: {}", dtype);
    println!("Runs: {}", num_runs);
    println!("Rust Version: {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "Unknown".to_string()));
    println!("================================");
}

// Helper function to print reference values
fn print_reference_values<T: MatrixOps>(benchmark: &StandardizedBenchmark<T>, a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) {
    println!("Reference values:");
    println!("A[0,0] = {}", a[0][0]);
    println!("B[0,0] = {}", b[0][0]);
    
    let result = benchmark.manual_matmul(a, b);
    println!("Result[0,0] = {}", result[0][0]);
}

// Helper function to run all generic benchmarks
fn run_generic_benchmarks<T: MatrixOps>(benchmark: &StandardizedBenchmark<T>, a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) {
    benchmark.benchmark_implementation("Manual Rust", |a, b| benchmark.manual_matmul(a, b), a, b);
    benchmark.benchmark_implementation("Optimized Rust", |a, b| benchmark.optimized_matmul(a, b), a, b);
    benchmark.benchmark_implementation("Highly Optimized Rust", |a, b| benchmark.highly_optimized_matmul(a, b), a, b);
    benchmark.benchmark_implementation("Flat Array Rust", |a, b| benchmark.flat_array_matmul(a, b), a, b);
    benchmark.benchmark_implementation("Unsafe Optimized Rust", |a, b| benchmark.unsafe_optimized_matmul(a, b), a, b);
    benchmark.benchmark_implementation("SIMD Generic Rust", |a, b| benchmark.simd_matmul_generic(a, b), a, b);
}

// Unified function to run all benchmarks with optional SIMD
fn run_all_benchmarks<T: MatrixOps>(matrix_size: usize, num_runs: usize, dtype: String) {
    let benchmark = StandardizedBenchmark::<T>::new(matrix_size, num_runs, 3, dtype.clone());
    
    print_benchmark_header(matrix_size, num_runs, &dtype, false);
    
    let (a, b) = benchmark.generate_matrices();
    run_generic_benchmarks(&benchmark, &a, &b);
    print_reference_values(&benchmark, &a, &b);
}

// Specialized function for f64 with SIMD
fn run_all_benchmarks_f64(matrix_size: usize, num_runs: usize, dtype: String) {
    let benchmark = StandardizedBenchmark::<f64>::new(matrix_size, num_runs, 3, dtype.clone());
    
    print_benchmark_header(matrix_size, num_runs, &dtype, true);
    
    let (a, b) = benchmark.generate_matrices();
    run_generic_benchmarks(&benchmark, &a, &b);
    
    // F64-specific SIMD implementation
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        benchmark.benchmark_implementation("SIMD F64 Rust", |a, b| unsafe { benchmark.simd_matmul_f64(a, b) }, &a, &b);
    } else {
        println!("SIMD F64 Rust: AVX2 not supported on this CPU");
    }
    
    print_reference_values(&benchmark, &a, &b);
}

// Specialized function for f32 with SIMD
fn run_all_benchmarks_f32(matrix_size: usize, num_runs: usize, dtype: String) {
    let benchmark = StandardizedBenchmark::<f32>::new(matrix_size, num_runs, 3, dtype.clone());
    
    print_benchmark_header(matrix_size, num_runs, &dtype, true);
    
    let (a, b) = benchmark.generate_matrices();
    run_generic_benchmarks(&benchmark, &a, &b);
    
    // F32-specific SIMD implementation
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        benchmark.benchmark_implementation("SIMD F32 Rust", |a, b| unsafe { benchmark.simd_matmul_f32(a, b) }, &a, &b);
    } else {
        println!("SIMD F32 Rust: AVX2 not supported on this CPU");
    }
    
    print_reference_values(&benchmark, &a, &b);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let matrix_size = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let num_runs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let dtype = args.get(3).cloned().unwrap_or_else(|| "float64".to_string());
    let mode = args.get(4).cloned().unwrap_or_else(|| "naive".to_string());
    
    let run_all = mode == "all";
    
    match dtype.as_str() {
        "float64" | "f64" => {
            if run_all {
                run_all_benchmarks_f64(matrix_size, num_runs, dtype);
            } else {
                run_benchmark::<f64>(matrix_size, num_runs, dtype);
            }
        }
        "float32" | "f32" => {
            if run_all {
                run_all_benchmarks_f32(matrix_size, num_runs, dtype);
            } else {
                run_benchmark::<f32>(matrix_size, num_runs, dtype);
            }
        }
        "int32" | "i32" => {
            if run_all {
                run_all_benchmarks::<i32>(matrix_size, num_runs, dtype);
            } else {
                run_benchmark::<i32>(matrix_size, num_runs, dtype);
            }
        }
        "int8" | "i8" => {
            if run_all {
                run_all_benchmarks::<i8>(matrix_size, num_runs, dtype);
            } else {
                run_benchmark::<i8>(matrix_size, num_runs, dtype);
            }
        }
        _ => {
            eprintln!("Unsupported dtype: {}. Supported types: float64, float32, int32, int8", dtype);
            std::process::exit(1);
        }
    }
}