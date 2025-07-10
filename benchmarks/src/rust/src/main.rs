/*
 * Optimized Rust Matrix Multiplication Benchmark
 * Using professional libraries: ndarray, nalgebra, faer, matrixmultiply
 * Comparable to NumPy's performance
 */

use std::env;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ndarray imports
use ndarray::{Array2, ArrayBase, OwnedRepr, Dim};
type NdArray<T> = ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>;

// nalgebra imports
use nalgebra::DMatrix;

// faer imports - skip for now
// use faer::Mat;

// matrixmultiply imports
use matrixmultiply::{sgemm, dgemm};

struct OptimizedBenchmark {
    matrix_size: usize,
    num_runs: usize,
    warmup_runs: usize,
    dtype_str: String,
}

impl OptimizedBenchmark {
    fn new(matrix_size: usize, num_runs: usize, warmup_runs: usize, dtype_str: String) -> Self {
        Self {
            matrix_size,
            num_runs,
            warmup_runs,
            dtype_str,
        }
    }

    fn print_header(&self) {
        println!("================================");
        println!("Rust Optimized Matrix Multiplication Benchmark");
        println!("Matrix Size: {}×{}", self.matrix_size, self.matrix_size);
        println!("Data Type: {}", self.dtype_str);
        println!("Runs: {}", self.num_runs);
        println!("Libraries: ndarray, nalgebra, matrixmultiply");
        println!("================================");
    }

    fn calculate_gflops(&self, time_seconds: f64) -> f64 {
        let n = self.matrix_size as u64;
        let flops = 2 * n * n * n - n * n;
        flops as f64 / (time_seconds * 1e9)
    }

    fn benchmark_implementation<F, R>(&self, name: &str, func: F, setup: impl Fn() -> R) 
    where
        F: Fn(&R) -> (),
    {
        println!("Benchmarking {}...", name);
        
        // Setup data
        let data = setup();
        
        // Warmup runs
        for _ in 0..self.warmup_runs {
            func(&data);
        }
        
        // Timed runs
        let mut times = Vec::new();
        for _ in 0..self.num_runs {
            let start = Instant::now();
            func(&data);
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

    fn run_f64_benchmarks(&self) {
        self.print_header();
        
        // ndarray benchmark
        self.benchmark_implementation(
            "ndarray",
            |data: &(NdArray<f64>, NdArray<f64>)| {
                let _result = data.0.dot(&data.1);
            },
            || self.generate_ndarray_f64()
        );

        // nalgebra benchmark
        self.benchmark_implementation(
            "nalgebra DMatrix",
            |data: &(DMatrix<f64>, DMatrix<f64>)| {
                let _result = &data.0 * &data.1;
            },
            || self.generate_nalgebra_f64()
        );

        // Skip faer for now due to dependencies

        // matrixmultiply benchmark
        self.benchmark_implementation(
            "matrixmultiply dgemm",
            |data: &(Vec<f64>, Vec<f64>, Vec<f64>)| {
                let mut c = data.2.clone();
                unsafe {
                    dgemm(
                        self.matrix_size, self.matrix_size, self.matrix_size,
                        1.0,
                        data.0.as_ptr(), self.matrix_size as isize, 1,
                        data.1.as_ptr(), self.matrix_size as isize, 1,
                        1.0,
                        c.as_mut_ptr(), self.matrix_size as isize, 1,
                    );
                }
            },
            || self.generate_matrixmultiply_f64()
        );

        self.print_reference_values_f64();
    }

    fn run_f32_benchmarks(&self) {
        self.print_header();
        
        // ndarray benchmark
        self.benchmark_implementation(
            "ndarray",
            |data: &(NdArray<f32>, NdArray<f32>)| {
                let _result = data.0.dot(&data.1);
            },
            || self.generate_ndarray_f32()
        );

        // nalgebra benchmark
        self.benchmark_implementation(
            "nalgebra DMatrix",
            |data: &(DMatrix<f32>, DMatrix<f32>)| {
                let _result = &data.0 * &data.1;
            },
            || self.generate_nalgebra_f32()
        );

        // Skip faer for now due to dependencies

        // matrixmultiply benchmark
        self.benchmark_implementation(
            "matrixmultiply sgemm",
            |data: &(Vec<f32>, Vec<f32>, Vec<f32>)| {
                let mut c = data.2.clone();
                unsafe {
                    sgemm(
                        self.matrix_size, self.matrix_size, self.matrix_size,
                        1.0,
                        data.0.as_ptr(), self.matrix_size as isize, 1,
                        data.1.as_ptr(), self.matrix_size as isize, 1,
                        1.0,
                        c.as_mut_ptr(), self.matrix_size as isize, 1,
                    );
                }
            },
            || self.generate_matrixmultiply_f32()
        );

        self.print_reference_values_f32();
    }

    // Data generation functions
    fn generate_ndarray_f64(&self) -> (NdArray<f64>, NdArray<f64>) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(43);
        
        let a = Array2::from_shape_fn((self.matrix_size, self.matrix_size), |_| rng_a.gen::<f64>());
        let b = Array2::from_shape_fn((self.matrix_size, self.matrix_size), |_| rng_b.gen::<f64>());
        
        (a, b)
    }

    fn generate_ndarray_f32(&self) -> (NdArray<f32>, NdArray<f32>) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(43);
        
        let a = Array2::from_shape_fn((self.matrix_size, self.matrix_size), |_| rng_a.gen::<f32>());
        let b = Array2::from_shape_fn((self.matrix_size, self.matrix_size), |_| rng_b.gen::<f32>());
        
        (a, b)
    }

    fn generate_nalgebra_f64(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(43);
        
        let a = DMatrix::from_fn(self.matrix_size, self.matrix_size, |_, _| rng_a.gen::<f64>());
        let b = DMatrix::from_fn(self.matrix_size, self.matrix_size, |_, _| rng_b.gen::<f64>());
        
        (a, b)
    }

    fn generate_nalgebra_f32(&self) -> (DMatrix<f32>, DMatrix<f32>) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(43);
        
        let a = DMatrix::from_fn(self.matrix_size, self.matrix_size, |_, _| rng_a.gen::<f32>());
        let b = DMatrix::from_fn(self.matrix_size, self.matrix_size, |_, _| rng_b.gen::<f32>());
        
        (a, b)
    }

    // faer functions removed for now

    fn generate_matrixmultiply_f64(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(43);
        
        let size = self.matrix_size * self.matrix_size;
        let a: Vec<f64> = (0..size).map(|_| rng_a.gen::<f64>()).collect();
        let b: Vec<f64> = (0..size).map(|_| rng_b.gen::<f64>()).collect();
        let c: Vec<f64> = vec![0.0; size];
        
        (a, b, c)
    }

    fn generate_matrixmultiply_f32(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(43);
        
        let size = self.matrix_size * self.matrix_size;
        let a: Vec<f32> = (0..size).map(|_| rng_a.gen::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng_b.gen::<f32>()).collect();
        let c: Vec<f32> = vec![0.0; size];
        
        (a, b, c)
    }

    fn print_reference_values_f64(&self) {
        let (a, b) = self.generate_ndarray_f64();
        let result = a.dot(&b);
        
        println!("Reference values:");
        println!("A[0,0] = {:.6}", a[(0, 0)]);
        println!("B[0,0] = {:.6}", b[(0, 0)]);
        println!("Result[0,0] = {:.6}", result[(0, 0)]);
    }

    fn print_reference_values_f32(&self) {
        let (a, b) = self.generate_ndarray_f32();
        let result = a.dot(&b);
        
        println!("Reference values:");
        println!("A[0,0] = {:.6}", a[(0, 0)]);
        println!("B[0,0] = {:.6}", b[(0, 0)]);
        println!("Result[0,0] = {:.6}", result[(0, 0)]);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let matrix_size = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let num_runs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let dtype = args.get(3).cloned().unwrap_or_else(|| "float64".to_string());
    
    let benchmark = OptimizedBenchmark::new(matrix_size, num_runs, 3, dtype.clone());
    
    match dtype.as_str() {
        "float64" | "f64" => benchmark.run_f64_benchmarks(),
        "float32" | "f32" => benchmark.run_f32_benchmarks(),
        _ => {
            eprintln!("Unsupported dtype: {}. Supported: float64, float32", dtype);
            std::process::exit(1);
        }
    }
}
