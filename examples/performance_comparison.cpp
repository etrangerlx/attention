#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <map>
#include <random>
#include <fstream>
#include <cmath>
#include <numeric>

// Include AttentionHPC API headers
#include "api/attention_interface.hpp"
#include "api/backend_factory.hpp"
#include "utils/timer.hpp"
#include "memory/memory_pool.hpp"

using namespace attention_hpc;

// Performance metrics structure
struct PerformanceMetrics {
    std::string backend_name;
    size_t batch_size;
    size_t sequence_length;
    size_t num_heads;
    size_t head_dimension;
    
    double forward_time_ms = 0.0;
    double backward_time_ms = 0.0;
    double total_time_ms = 0.0;
    double memory_usage_mb = 0.0;
    double peak_memory_mb = 0.0;
    double throughput_tokens_per_sec = 0.0;
    double gflops = 0.0;
    
    bool success = false;
    std::string error_message;
    
    // Calculate derived metrics
    double get_theoretical_flops() const {
        // Approximate FLOPs for attention: 2 * batch * heads * seq_len^2 * head_dim
        return 2.0 * batch_size * num_heads * sequence_length * sequence_length * head_dimension;
    }
    
    size_t get_total_elements() const {
        return batch_size * sequence_length * num_heads * head_dimension;
    }
};

// Test configuration
struct TestConfig {
    std::vector<BackendType> backends_to_test;
    std::vector<size_t> batch_sizes = {1, 2, 4, 8};
    std::vector<size_t> sequence_lengths = {64, 128, 256, 512, 1024};
    std::vector<size_t> num_heads = {4, 8, 12, 16};
    std::vector<size_t> head_dimensions = {32, 64, 128};
    int num_warmup_iterations = 3;
    int num_benchmark_iterations = 10;
    bool enable_backward_pass = true;
    bool enable_memory_profiling = true;
    size_t max_memory_gb = 8; // Skip tests that would exceed this memory limit
};

// Global variables
std::unique_ptr<MemoryPool> g_memory_pool;
std::vector<PerformanceMetrics> g_performance_results;

// Helper function to generate random data
std::vector<float> generate_random_data(size_t size, float scale = 1.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// Estimate memory usage for a configuration
size_t estimate_memory_usage(size_t batch_size, size_t sequence_length, 
                           size_t num_heads, size_t head_dimension) {
    size_t total_elements = batch_size * sequence_length * num_heads * head_dimension;
    
    // Estimate: Q, K, V inputs + Output + Attention weights + Workspace
    size_t qkv_memory = 3 * total_elements * sizeof(float);
    size_t output_memory = total_elements * sizeof(float);
    size_t attention_weights = batch_size * num_heads * sequence_length * sequence_length * sizeof(float);
    size_t workspace = total_elements * sizeof(float) * 2; // Conservative estimate
    
    return qkv_memory + output_memory + attention_weights + workspace;
}

// Benchmark a specific configuration
PerformanceMetrics benchmark_configuration(BackendType backend, size_t batch_size, 
                                         size_t sequence_length, size_t num_heads, 
                                         size_t head_dimension, const TestConfig& config) {
    PerformanceMetrics metrics;
    metrics.backend_name = BackendFactory::backendTypeToString(backend);
    metrics.batch_size = batch_size;
    metrics.sequence_length = sequence_length;
    metrics.num_heads = num_heads;
    metrics.head_dimension = head_dimension;
    
    try {
        // Check memory requirements
        size_t estimated_memory = estimate_memory_usage(batch_size, sequence_length, num_heads, head_dimension);
        if (estimated_memory > config.max_memory_gb * 1024 * 1024 * 1024) {
            metrics.error_message = "Configuration would exceed memory limit";
            return metrics;
        }
        
        // Create attention interface
        auto attention = BackendFactory::createFlashAttention(backend);
        
        // Configure attention parameters
        AttentionParams params;
        params.batch_size = batch_size;
        params.sequence_length = sequence_length;
        params.num_heads = num_heads;
        params.head_dimension = head_dimension;
        params.scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dimension));
        
        PerformanceConfig perf_config;
        perf_config.enable_profiling = true;
        perf_config.enable_debug_checks = false; // Disable for performance
        
        attention->configure(params, perf_config);
        
        // Prepare test data
        size_t total_elements = batch_size * sequence_length * num_heads * head_dimension;
        auto query_data = generate_random_data(total_elements, 0.1f);
        auto key_data = generate_random_data(total_elements, 0.1f);
        auto value_data = generate_random_data(total_elements, 0.1f);
        std::vector<float> output_data(total_elements);
        
        // Gradient data for backward pass
        std::vector<float> grad_output_data, grad_query_data, grad_key_data, grad_value_data;
        if (config.enable_backward_pass) {
            grad_output_data = generate_random_data(total_elements, 0.1f);
            grad_query_data.resize(total_elements);
            grad_key_data.resize(total_elements);
            grad_value_data.resize(total_elements);
        }
        
        TensorShape tensor_shape({batch_size, sequence_length, num_heads, head_dimension}, 
                                DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
        
        // Setup workspace
        size_t workspace_size = attention->get_workspace_size(params);
        std::vector<uint8_t> workspace;
        if (workspace_size > 0) {
            workspace.resize(workspace_size);
            attention->set_workspace(workspace.data(), workspace_size);
        }
        
        // Warmup iterations
        for (int i = 0; i < config.num_warmup_iterations; ++i) {
            attention->forward(tensor_shape, query_data.data(),
                             tensor_shape, key_data.data(),
                             tensor_shape, value_data.data(),
                             tensor_shape, output_data.data());
        }
        
        // Benchmark forward pass
        Timer forward_timer;
        forward_timer.start();
        
        for (int i = 0; i < config.num_benchmark_iterations; ++i) {
            attention->forward(tensor_shape, query_data.data(),
                             tensor_shape, key_data.data(),
                             tensor_shape, value_data.data(),
                             tensor_shape, output_data.data());
        }
        
        forward_timer.stop();
        metrics.forward_time_ms = forward_timer.elapsed_milliseconds() / config.num_benchmark_iterations;
        
        // Benchmark backward pass if enabled
        if (config.enable_backward_pass) {
            Timer backward_timer;
            backward_timer.start();
            
            for (int i = 0; i < config.num_benchmark_iterations; ++i) {
                attention->backward(tensor_shape, grad_output_data.data(),
                                  tensor_shape, query_data.data(),
                                  tensor_shape, key_data.data(),
                                  tensor_shape, value_data.data(),
                                  tensor_shape, grad_query_data.data(),
                                  tensor_shape, grad_key_data.data(),
                                  tensor_shape, grad_value_data.data());
            }
            
            backward_timer.stop();
            metrics.backward_time_ms = backward_timer.elapsed_milliseconds() / config.num_benchmark_iterations;
        }
        
        metrics.total_time_ms = metrics.forward_time_ms + metrics.backward_time_ms;
        
        // Calculate throughput and performance metrics
        size_t total_tokens = batch_size * sequence_length;
        metrics.throughput_tokens_per_sec = (total_tokens * 1000.0) / metrics.forward_time_ms;
        
        double theoretical_flops = metrics.get_theoretical_flops();
        metrics.gflops = (theoretical_flops / 1e9) / (metrics.forward_time_ms / 1000.0);
        
        // Memory usage estimation
        metrics.memory_usage_mb = static_cast<double>(estimated_memory) / (1024.0 * 1024.0);
        metrics.peak_memory_mb = metrics.memory_usage_mb * 1.2; // Conservative estimate
        
        // Get profiling results if available
        auto profiling_results = attention->get_profiling_results();
        if (!profiling_results.empty()) {
            std::cout << "  Profiling data available: " << profiling_results.size() << " entries" << std::endl;
        }
        
        metrics.success = true;
        
    } catch (const AttentionException& e) {
        metrics.error_message = "AttentionException: " + std::string(e.what());
    } catch (const std::exception& e) {
        metrics.error_message = "Exception: " + std::string(e.what());
    } catch (...) {
        metrics.error_message = "Unknown exception";
    }
    
    return metrics;
}

// Run comprehensive performance comparison
void run_performance_comparison(const TestConfig& config) {
    std::cout << "=== AttentionHPC Performance Comparison ===" << std::endl;
    std::cout << "Testing " << config.backends_to_test.size() << " backends" << std::endl;
    std::cout << "Warmup iterations: " << config.num_warmup_iterations << std::endl;
    std::cout << "Benchmark iterations: " << config.num_benchmark_iterations << std::endl;
    std::cout << "Backward pass: " << (config.enable_backward_pass ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;
    
    // Initialize memory pool
    g_memory_pool = std::make_unique<MemoryPool>();
    
    size_t total_tests = config.backends_to_test.size() * config.batch_sizes.size() * 
                        config.sequence_lengths.size() * config.num_heads.size() * 
                        config.head_dimensions.size();
    
    size_t completed_tests = 0;
    
    for (BackendType backend : config.backends_to_test) {
        if (!BackendFactory::isBackendAvailable(backend)) {
            std::cout << "Backend " << BackendFactory::backendTypeToString(backend) 
                      << " is not available, skipping..." << std::endl;
            continue;
        }
        
        std::cout << "\n--- Testing Backend: " << BackendFactory::backendTypeToString(backend) << " ---" << std::endl;
        
        for (size_t batch_size : config.batch_sizes) {
            for (size_t seq_len : config.sequence_lengths) {
                for (size_t num_heads : config.num_heads) {
                    for (size_t head_dim : config.head_dimensions) {
                        completed_tests++;
                        
                        std::cout << "Test " << completed_tests << "/" << total_tests << ": "
                                  << "B=" << batch_size << ", S=" << seq_len 
                                  << ", H=" << num_heads << ", D=" << head_dim 
                                  << " ... " << std::flush;
                        
                        auto metrics = benchmark_configuration(backend, batch_size, seq_len, 
                                                             num_heads, head_dim, config);
                        
                        if (metrics.success) {
                            std::cout << "OK (" << std::fixed << std::setprecision(2) 
                                      << metrics.forward_time_ms << "ms, " 
                                      << metrics.gflops << " GFLOPS)" << std::endl;
                        } else {
                            std::cout << "FAILED (" << metrics.error_message << ")" << std::endl;
                        }
                        
                        g_performance_results.push_back(metrics);
                    }
                }
            }
        }
    }
    
    std::cout << "\nCompleted " << completed_tests << " tests." << std::endl;
}

// Generate performance report
void generate_performance_report() {
    std::cout << "\n=== Performance Report ===" << std::endl;
    
    // Filter successful results
    std::vector<PerformanceMetrics> successful_results;
    std::copy_if(g_performance_results.begin(), g_performance_results.end(),
                 std::back_inserter(successful_results),
                 [](const PerformanceMetrics& m) { return m.success; });
    
    if (successful_results.empty()) {
        std::cout << "No successful benchmark results to report." << std::endl;
        return;
    }
    
    // Summary table
    std::cout << "\nSummary Table (Top 20 Results by GFLOPS):" << std::endl;
    std::cout << std::left << std::setw(10) << "Backend"
              << std::setw(20) << "Configuration"
              << std::setw(12) << "Forward(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Tokens/sec"
              << std::setw(12) << "Memory(MB)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Sort by GFLOPS descending
    std::sort(successful_results.begin(), successful_results.end(),
              [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
                  return a.gflops > b.gflops;
              });
    
    for (size_t i = 0; i < std::min(successful_results.size(), size_t(20)); ++i) {
        const auto& result = successful_results[i];
        std::string config = std::to_string(result.batch_size) + "x" +
                           std::to_string(result.sequence_length) + "x" +
                           std::to_string(result.num_heads) + "x" +
                           std::to_string(result.head_dimension);
        
        std::cout << std::left << std::setw(10) << result.backend_name
                  << std::setw(20) << config
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.forward_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.throughput_tokens_per_sec
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.memory_usage_mb
                  << std::endl;
    }
    
    // Backend comparison
    std::cout << "\n=== Backend Comparison ===" << std::endl;
    std::map<std::string, std::vector<PerformanceMetrics>> backend_results;
    for (const auto& result : successful_results) {
        backend_results[result.backend_name].push_back(result);
    }
    
    for (const auto& [backend_name, results] : backend_results) {
        if (results.empty()) continue;
        
        // Calculate statistics
        std::vector<double> gflops_values;
        std::vector<double> throughput_values;
        std::vector<double> memory_values;
        
        for (const auto& result : results) {
            gflops_values.push_back(result.gflops);
            throughput_values.push_back(result.throughput_tokens_per_sec);
            memory_values.push_back(result.memory_usage_mb);
        }
        
        auto calc_stats = [](const std::vector<double>& values) {
            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            double mean = sum / values.size();
            double max_val = *std::max_element(values.begin(), values.end());
            double min_val = *std::min_element(values.begin(), values.end());
            
            // Calculate standard deviation
            double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / values.size() - mean * mean);
            
            return std::make_tuple(mean, max_val, min_val, stdev);
        };
        
        auto [gflops_mean, gflops_max, gflops_min, gflops_stdev] = calc_stats(gflops_values);
        auto [throughput_mean, throughput_max, throughput_min, throughput_stdev] = calc_stats(throughput_values);
        auto [memory_mean, memory_max, memory_min, memory_stdev] = calc_stats(memory_values);
        
        std::cout << "\n" << backend_name << " Statistics:" << std::endl;
        std::cout << "  Tests completed: " << results.size() << std::endl;
        std::cout << "  GFLOPS - Mean: " << std::fixed << std::setprecision(2) << gflops_mean
                  << ", Max: " << gflops_max << ", Min: " << gflops_min 
                  << ", StdDev: " << gflops_stdev << std::endl;
        std::cout << "  Throughput - Mean: " << std::fixed << std::setprecision(0) << throughput_mean
                  << ", Max: " << throughput_max << " tokens/sec" << std::endl;
        std::cout << "  Memory - Mean: " << std::fixed << std::setprecision(1) << memory_mean
                  << ", Max: " << memory_max << " MB" << std::endl;
    }
    
    // Best performing configurations
    std::cout << "\n=== Best Performing Configurations ===" << std::endl;
    
    // Best overall GFLOPS
    auto best_gflops = std::max_element(successful_results.begin(), successful_results.end(),
        [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
            return a.gflops < b.gflops;
        });
    
    if (best_gflops != successful_results.end()) {
        std::cout << "Best GFLOPS: " << best_gflops->backend_name 
                  << " with " << std::fixed << std::setprecision(2) << best_gflops->gflops
                  << " GFLOPS (Config: " << best_gflops->batch_size << "x" 
                  << best_gflops->sequence_length << "x" << best_gflops->num_heads 
                  << "x" << best_gflops->head_dimension << ")" << std::endl;
    }
    
    // Best throughput
    auto best_throughput = std::max_element(successful_results.begin(), successful_results.end(),
        [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
            return a.throughput_tokens_per_sec < b.throughput_tokens_per_sec;
        });
    
    if (best_throughput != successful_results.end()) {
        std::cout << "Best Throughput: " << best_throughput->backend_name 
                  << " with " << std::fixed << std::setprecision(0) << best_throughput->throughput_tokens_per_sec
                  << " tokens/sec" << std::endl;
    }
    
    // Most memory efficient
    auto most_efficient = std::min_element(successful_results.begin(), successful_results.end(),
        [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
            return a.memory_usage_mb < b.memory_usage_mb;
        });
    
    if (most_efficient != successful_results.end()) {
        std::cout << "Most Memory Efficient: " << most_efficient->backend_name 
                  << " with " << std::fixed << std::setprecision(1) << most_efficient->memory_usage_mb
                  << " MB" << std::endl;
    }
}

// Save results to CSV for further analysis
void save_results_to_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // CSV header
    file << "Backend,BatchSize,SequenceLength,NumHeads,HeadDimension,";
    file << "ForwardTimeMs,BackwardTimeMs,TotalTimeMs,MemoryUsageMB,PeakMemoryMB,";
    file << "ThroughputTokensPerSec,GFLOPS,Success,ErrorMessage" << std::endl;
    
    // Data rows
    for (const auto& result : g_performance_results) {
        file << result.backend_name << ","
             << result.batch_size << ","
             << result.sequence_length << ","
             << result.num_heads << ","
             << result.head_dimension << ","
             << std::fixed << std::setprecision(6) << result.forward_time_ms << ","
             << std::fixed << std::setprecision(6) << result.backward_time_ms << ","
             << std::fixed << std::setprecision(6) << result.total_time_ms << ","
             << std::fixed << std::setprecision(2) << result.memory_usage_mb << ","
             << std::fixed << std::setprecision(2) << result.peak_memory_mb << ","
             << std::fixed << std::setprecision(0) << result.throughput_tokens_per_sec << ","
             << std::fixed << std::setprecision(4) << result.gflops << ","
             << (result.success ? "true" : "false") << ","
             << "\"" << result.error_message << "\"" << std::endl;
    }
    
    file.close();
    std::cout << "\nResults saved to: " << filename << std::endl;
}

// Generate visualization data for plotting
void generate_visualization_data() {
    std::cout << "\n=== Generating Visualization Data ===" << std::endl;
    
    // Group results for visualization
    std::map<std::string, std::vector<PerformanceMetrics>> backend_groups;
    for (const auto& result : g_performance_results) {
        if (result.success) {
            backend_groups[result.backend_name].push_back(result);
        }
    }
    
    // Generate data for sequence length vs performance plot
    std::ofstream seq_plot_file("sequence_length_performance.csv");
    if (seq_plot_file.is_open()) {
        seq_plot_file << "Backend,SequenceLength,GFLOPS,ThroughputTokensPerSec" << std::endl;
        
        for (const auto& [backend_name, results] : backend_groups) {
            std::map<size_t, std::vector<double>> seq_gflops;
            std::map<size_t, std::vector<double>> seq_throughput;
            
            for (const auto& result : results) {
                seq_gflops[result.sequence_length].push_back(result.gflops);
                seq_throughput[result.sequence_length].push_back(result.throughput_tokens_per_sec);
            }
            
            for (const auto& [seq_len, gflops_vals] : seq_gflops) {
                double avg_gflops = std::accumulate(gflops_vals.begin(), gflops_vals.end(), 0.0) / gflops_vals.size();
                double avg_throughput = std::accumulate(seq_throughput[seq_len].begin(), 
                                                       seq_throughput[seq_len].end(), 0.0) / seq_throughput[seq_len].size();
                
                seq_plot_file << backend_name << "," << seq_len << "," 
                             << std::fixed << std::setprecision(4) << avg_gflops << ","
                             << std::fixed << std::setprecision(0) << avg_throughput << std::endl;
            }
        }
        seq_plot_file.close();
        std::cout << "Sequence length performance data saved to: sequence_length_performance.csv" << std::endl;
    }
    
    // Generate data for batch size vs performance plot
    std::ofstream batch_plot_file("batch_size_performance.csv");
    if (batch_plot_file.is_open()) {
        batch_plot_file << "Backend,BatchSize,GFLOPS,ThroughputTokensPerSec" << std::endl;
        
        for (const auto& [backend_name, results] : backend_groups) {
            std::map<size_t, std::vector<double>> batch_gflops;
            std::map<size_t, std::vector<double>> batch_throughput;
            
            for (const auto& result : results) {
                batch_gflops[result.batch_size].push_back(result.gflops);
                batch_throughput[result.batch_size].push_back(result.throughput_tokens_per_sec);
            }
            
            for (const auto& [batch_size, gflops_vals] : batch_gflops) {
                double avg_gflops = std::accumulate(gflops_vals.begin(), gflops_vals.end(), 0.0) / gflops_vals.size();
                double avg_throughput = std::accumulate(batch_throughput[batch_size].begin(), 
                                                       batch_throughput[batch_size].end(), 0.0) / batch_throughput[batch_size].size();
                
                batch_plot_file << backend_name << "," << batch_size << "," 
                               << std::fixed << std::setprecision(4) << avg_gflops << ","
                               << std::fixed << std::setprecision(0) << avg_throughput << std::endl;
            }
        }
        batch_plot_file.close();
        std::cout << "Batch size performance data saved to: batch_size_performance.csv" << std::endl;
    }
    
    std::cout << "Visualization data generation completed." << std::endl;
}

int main() {
    try {
        // Configure test parameters
        TestConfig config;
        
        // Get all available backends
        config.backends_to_test = BackendFactory::getAvailableBackends();
        
        if (config.backends_to_test.empty()) {
            std::cerr << "No backends available for testing!" << std::endl;
            return 1;
        }
        
        std::cout << "Available backends for testing: ";
        for (BackendType backend : config.backends_to_test) {
            std::cout << BackendFactory::backendTypeToString(backend) << " ";
        }
        std::cout << std::endl;
        
        // Adjust test parameters for reasonable execution time
        config.batch_sizes = {1, 2, 4};
        config.sequence_lengths = {64, 128, 256, 512};
        config.num_heads = {4, 8, 12};
        config.head_dimensions = {64, 128};
        config.num_warmup_iterations = 2;
        config.num_benchmark_iterations = 5;
        config.max_memory_gb = 4; // Limit memory usage
        
        // Run the performance comparison
        auto start_time = std::chrono::high_resolution_clock::now();
        run_performance_comparison(config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "\nTotal benchmark time: " << duration.count() << " seconds" << std::endl;
        
        // Generate reports and save data
        generate_performance_report();
        save_results_to_csv("performance_results.csv");
        generate_visualization_data();
        
        std::cout << "\nPerformance comparison completed successfully!" << std::endl;
        std::cout << "Results saved to CSV files for further analysis." << std::endl;
        
    } catch (const AttentionException& e) {
        std::cerr << "AttentionHPC Exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred!" << std::endl;
        return 1;
    }
    
    return 0;
}
