/*
 * Standardized C++ Matrix Multiplication Benchmark
 * Follows the cross-language benchmark specification
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <type_traits>

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

template<typename T>
class StandardizedBenchmark {
private:
    int matrix_size;
    int num_runs;
    int warmup_runs;
    std::string dtype_str;
    
    using Matrix = std::vector<std::vector<T>>;
    
public:
    StandardizedBenchmark(int size = 100, int runs = 10, int warmup = 3, const std::string& dtype = "float64") 
        : matrix_size(size), num_runs(runs), warmup_runs(warmup), dtype_str(dtype) {}
    
    std::pair<Matrix, Matrix> generate_matrices() {
        // Matrix A: seed 42
        std::mt19937 gen_a(42);
        Matrix A(matrix_size, std::vector<T>(matrix_size));
        
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(0.0, 1.0);
            for (int i = 0; i < matrix_size; ++i) {
                for (int j = 0; j < matrix_size; ++j) {
                    A[i][j] = dis(gen_a);
                }
            }
        } else {
            // For integer types, use appropriate ranges
            if constexpr (std::is_same_v<T, int8_t>) {
                std::uniform_int_distribution<int> dis(-10, 10);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        A[i][j] = static_cast<T>(dis(gen_a));
                    }
                }
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                std::uniform_int_distribution<int> dis(0, 20);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        A[i][j] = static_cast<T>(dis(gen_a));
                    }
                }
            } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
                std::uniform_int_distribution<int> dis(-100, 100);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        A[i][j] = static_cast<T>(dis(gen_a));
                    }
                }
            } else {
                // int32_t, int64_t, uint32_t, uint64_t
                std::uniform_int_distribution<T> dis(-1000, 1000);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        A[i][j] = dis(gen_a);
                    }
                }
            }
        }
        
        // Matrix B: seed 43
        std::mt19937 gen_b(43);
        Matrix B(matrix_size, std::vector<T>(matrix_size));
        
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dis(0.0, 1.0);
            for (int i = 0; i < matrix_size; ++i) {
                for (int j = 0; j < matrix_size; ++j) {
                    B[i][j] = dis(gen_b);
                }
            }
        } else {
            // For integer types, use appropriate ranges
            if constexpr (std::is_same_v<T, int8_t>) {
                std::uniform_int_distribution<int> dis(-10, 10);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        B[i][j] = static_cast<T>(dis(gen_b));
                    }
                }
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                std::uniform_int_distribution<int> dis(0, 20);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        B[i][j] = static_cast<T>(dis(gen_b));
                    }
                }
            } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
                std::uniform_int_distribution<int> dis(-100, 100);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        B[i][j] = static_cast<T>(dis(gen_b));
                    }
                }
            } else {
                // int32_t, int64_t, uint32_t, uint64_t
                std::uniform_int_distribution<T> dis(-1000, 1000);
                for (int i = 0; i < matrix_size; ++i) {
                    for (int j = 0; j < matrix_size; ++j) {
                        B[i][j] = dis(gen_b);
                    }
                }
            }
        }
        
        return {A, B};
    }
    
    Matrix manual_matmul(const Matrix& A, const Matrix& B) {
        Matrix result(matrix_size, std::vector<T>(matrix_size, T(0)));
        
        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                for (int k = 0; k < matrix_size; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    Matrix optimized_matmul(const Matrix& A, const Matrix& B) {
        Matrix result(matrix_size, std::vector<T>(matrix_size, T(0)));
        
        // Cache-friendly loop ordering (ikj)
        for (int i = 0; i < matrix_size; ++i) {
            for (int k = 0; k < matrix_size; ++k) {
                for (int j = 0; j < matrix_size; ++j) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
#ifdef USE_EIGEN
    Eigen::MatrixXd eigen_matmul(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        return A * B;
    }
    
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> generate_eigen_matrices() {
        std::mt19937 gen_a(42);
        std::mt19937 gen_b(43);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        Eigen::MatrixXd A(matrix_size, matrix_size);
        Eigen::MatrixXd B(matrix_size, matrix_size);
        
        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                A(i, j) = dis(gen_a);
                B(i, j) = dis(gen_b);
            }
        }
        
        return {A, B};
    }
#endif
    
    double calculate_gflops(double time_seconds) {
        long long flops = 2LL * matrix_size * matrix_size * matrix_size - 
                         matrix_size * matrix_size;
        return flops / (time_seconds * 1e9);
    }
    
    template<typename Func, typename... Args>
    void benchmark_implementation(const std::string& name, Func func, Args&&... args) {
        std::cout << "Benchmarking " << name << "..." << std::endl;
        
        // Warmup runs
        for (int i = 0; i < warmup_runs; ++i) {
            auto result = func(args...);
        }
        
        // Timed runs
        std::vector<double> times;
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = func(args...);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1e9);  // Convert to seconds
        }
        
        // Calculate statistics
        double mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double variance = 0.0;
        for (double t : times) {
            variance += (t - mean_time) * (t - mean_time);
        }
        variance /= times.size();
        double std_dev = std::sqrt(variance);
        
        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());
        double gflops = calculate_gflops(mean_time);
        
        // Print results
        std::cout << std::fixed << std::setprecision(3);
        std::cout << name << ":" << std::endl;
        std::cout << "  Mean time: " << mean_time * 1000 << " ms ± " << std_dev * 1000 << " ms" << std::endl;
        std::cout << "  Min time:  " << min_time * 1000 << " ms" << std::endl;
        std::cout << "  Max time:  " << max_time * 1000 << " ms" << std::endl;
        std::cout << "  GFLOPS:    " << gflops << std::endl;
        std::cout << std::endl;
    }
    
    void run_benchmarks() {
        std::cout << "================================" << std::endl;
        std::cout << "C++ Matrix Multiplication Benchmark" << std::endl;
        std::cout << "Matrix Size: " << matrix_size << "×" << matrix_size << std::endl;
        std::cout << "Data Type: " << dtype_str << std::endl;
        std::cout << "Runs: " << num_runs << std::endl;
        std::cout << "================================" << std::endl;
        
        auto [A, B] = generate_matrices();
        
        // Manual implementation
        benchmark_implementation("Manual C++", 
            [this](const Matrix& A, const Matrix& B) { return manual_matmul(A, B); },
            A, B);
        
        // Optimized implementation
        benchmark_implementation("Optimized C++", 
            [this](const Matrix& A, const Matrix& B) { return optimized_matmul(A, B); },
            A, B);
        
#ifdef USE_EIGEN
        auto [A_eigen, B_eigen] = generate_eigen_matrices();
        benchmark_implementation("Eigen", 
            [this](const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) { return eigen_matmul(A, B); },
            A_eigen, B_eigen);
#endif
        
        // Print reference values
        std::cout << "Reference values:" << std::endl;
        std::cout << "A[0,0] = " << std::fixed << std::setprecision(6) << A[0][0] << std::endl;
        std::cout << "B[0,0] = " << std::fixed << std::setprecision(6) << B[0][0] << std::endl;
        
        auto result = manual_matmul(A, B);
        std::cout << "Result[0,0] = " << std::fixed << std::setprecision(6) << result[0][0] << std::endl;
    }
};

template<typename T>
void run_benchmark_for_type(int matrix_size, int num_runs, const std::string& dtype_str) {
    StandardizedBenchmark<T> benchmark(matrix_size, num_runs, 3, dtype_str);
    benchmark.run_benchmarks();
}

int main(int argc, char* argv[]) {
    int matrix_size = 100;
    int num_runs = 10;
    std::string dtype = "float64";
    
    if (argc > 1) {
        matrix_size = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_runs = std::atoi(argv[2]);
    }
    if (argc > 3) {
        dtype = argv[3];
    }
    
    // Instantiate the appropriate template based on dtype
    if (dtype == "float64" || dtype == "f64") {
        run_benchmark_for_type<double>(matrix_size, num_runs, dtype);
    } else if (dtype == "float32" || dtype == "f32") {
        run_benchmark_for_type<float>(matrix_size, num_runs, dtype);
    } else if (dtype == "int8" || dtype == "i8") {
        run_benchmark_for_type<int8_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "int16" || dtype == "i16") {
        run_benchmark_for_type<int16_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "int32" || dtype == "i32") {
        run_benchmark_for_type<int32_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "int64" || dtype == "i64") {
        run_benchmark_for_type<int64_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "uint8" || dtype == "u8") {
        run_benchmark_for_type<uint8_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "uint16" || dtype == "u16") {
        run_benchmark_for_type<uint16_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "uint32" || dtype == "u32") {
        run_benchmark_for_type<uint32_t>(matrix_size, num_runs, dtype);
    } else if (dtype == "uint64" || dtype == "u64") {
        run_benchmark_for_type<uint64_t>(matrix_size, num_runs, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        std::cerr << "Supported types: float64, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64" << std::endl;
        std::cerr << "                 f64, f32, i8, i16, i32, i64, u8, u16, u32, u64" << std::endl;
        return 1;
    }
    
    return 0;
}