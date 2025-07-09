/*
 * Standardized Rust Matrix Multiplication Benchmark
 * Follows the cross-language benchmark specification
 */

use std::env;
use std::time::Instant;
use std::arch::x86_64::*;

struct StandardizedBenchmark {
    matrix_size: usize,
    num_runs: usize,
    warmup_runs: usize,
}

impl StandardizedBenchmark {
    fn new(matrix_size: usize, num_runs: usize, warmup_runs: usize) -> Self {
        Self {
            matrix_size,
            num_runs,
            warmup_runs,
        }
    }

    fn generate_matrices(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Matrix A: seed 42 (using LCG for reproducibility)
        let mut a = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        let mut seed_a = 42u64;
        
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                seed_a = (seed_a.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                a[i][j] = (seed_a as f64) / (0x7fffffff as f64);
            }
        }

        // Matrix B: seed 43
        let mut b = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        let mut seed_b = 43u64;
        
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                seed_b = (seed_b.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                b[i][j] = (seed_b as f64) / (0x7fffffff as f64);
            }
        }

        (a, b)
    }

    fn manual_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                for k in 0..self.matrix_size {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        result
    }

    fn optimized_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        
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

    fn highly_optimized_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        
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

    fn flat_array_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        // Use flat arrays for better memory layout
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
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                result[i][j] = result_flat[i * self.matrix_size + j];
            }
        }
        
        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn simd_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    fn unsafe_optimized_matmul(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.matrix_size]; self.matrix_size];
        
        // Use unsafe for bounds check elimination
        unsafe {
            for i in 0..self.matrix_size {
                for k in 0..self.matrix_size {
                    let aik = *a.get_unchecked(i).get_unchecked(k);
                    let result_row = result.get_unchecked_mut(i);
                    let b_row = b.get_unchecked(k);
                    
                    for j in 0..self.matrix_size {
                        *result_row.get_unchecked_mut(j) += aik * b_row.get_unchecked(j);
                    }
                }
            }
        }
        
        result
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

    fn benchmark_implementation<F>(&self, name: &str, func: F, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>)
    where
        F: Fn(&Vec<Vec<f64>>, &Vec<Vec<f64>>) -> Vec<Vec<f64>>,
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

    fn run_benchmarks(&self) {
        println!("================================");
        println!("Rust Matrix Multiplication Benchmark");
        println!("Matrix Size: {}×{}", self.matrix_size, self.matrix_size);
        println!("Runs: {}", self.num_runs);
        println!("Rust Version: {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "Unknown".to_string()));
        println!("================================");
        
        let (a, b) = self.generate_matrices();
        
        // Manual implementation
        self.benchmark_implementation("Manual Rust", |a, b| self.manual_matmul(a, b), &a, &b);
        
        // Optimized implementation
        self.benchmark_implementation("Optimized Rust", |a, b| self.optimized_matmul(a, b), &a, &b);
        
        // Highly optimized (blocked) implementation
        self.benchmark_implementation("Blocked Rust", |a, b| self.highly_optimized_matmul(a, b), &a, &b);
        
        // Flat array implementation
        self.benchmark_implementation("Flat Array Rust", |a, b| self.flat_array_matmul(a, b), &a, &b);
        
        // Unsafe optimized implementation
        self.benchmark_implementation("Unsafe Rust", |a, b| self.unsafe_optimized_matmul(a, b), &a, &b);
        
        // SIMD implementation (if AVX2 is available)
        if is_x86_feature_detected!("avx2") {
            self.benchmark_implementation("SIMD Rust", |a, b| unsafe { self.simd_matmul(a, b) }, &a, &b);
            
            // Flat Array + SIMD hybrid
            self.benchmark_implementation("Flat+SIMD Rust", |a, b| unsafe { self.flat_simd_matmul(a, b) }, &a, &b);
            
            // Ultimate optimization combining all techniques
            self.benchmark_implementation("Ultimate Rust", |a, b| unsafe { self.ultimate_optimized_matmul(a, b) }, &a, &b);
        }
        
        // Print reference values
        println!("Reference values:");
        println!("A[0,0] = {:.6}", a[0][0]);
        println!("B[0,0] = {:.6}", b[0][0]);
        
        let result = self.manual_matmul(&a, &b);
        println!("Result[0,0] = {:.6}", result[0][0]);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let matrix_size = if args.len() > 1 {
        args[1].parse().unwrap_or(100)
    } else {
        100
    };
    
    let num_runs = if args.len() > 2 {
        args[2].parse().unwrap_or(10)
    } else {
        10
    };
    
    let benchmark = StandardizedBenchmark::new(matrix_size, num_runs, 3);
    benchmark.run_benchmarks();
}