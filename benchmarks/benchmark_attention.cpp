#include <benchmark/benchmark.h>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <map>
#include <cmath>

#include "api/attention_interface.hpp"
#include "api/backend_factory.hpp"
#include "api/tensor.hpp"
#include "algorithms/flash_attention.hpp"
#include "algorithms/paged_attention.hpp"
#include "memory/memory_pool.hpp"
#include "utils/timer.hpp"

namespace attention_hpc {
namespace benchmarks {

// Benchmark configuration structure
struct BenchmarkConfig {
    std::vector<size_t> batch_sizes = {1, 2, 4, 8};
    std::vector<size_t> sequence_lengths = {64, 128, 256, 512, 1024, 2048};
    std::vector<size_t> num_heads = {1, 4, 8, 12, 16};
    std::vector<size_t> head_dimensions = {64, 128};
    std::vector<BackendType> backends;
    bool enable_causal_mask = false;
    bool measure_memory = true;
    bool measure_flops = true;
    int warmup_iterations = 5;
    int benchmark_iterations = 10;
};

// Performance metrics structure
struct PerformanceMetrics {
    double forward_time_ms = 0.0;
    double backward_time_ms = 0.0;
    double total_time_ms = 0.0;
    size_t memory_usage_bytes = 0;
    size_t peak_memory_bytes = 0;
    double gflops = 0.0;
    double throughput_tokens_per_sec = 0.0;
    bool success = false;
    std::string error_message;
};

// Benchmark result structure
struct BenchmarkResult {
    BackendType backend;
    std::string backend_name;
    size_t batch_size;
    size_t sequence_length;
    size_t num_heads;
    size_t head_dimension;
    PerformanceMetrics metrics;
    
    // Calculate total parameters
    size_t get_total_elements() const {
        return batch_size * sequence_length * num_heads * head_dimension;
    }
    
    // Calculate theoretical FLOPS for attention
    double get_theoretical_flops() const {
        // Simplified FLOPS calculation for attention mechanism
        // QK^T: batch_size * num_heads * seq_len * seq_len * head_dim
        // Softmax: batch_size * num_heads * seq_len * seq_len * 4 (exp, sum, div)
        // Attention * V: batch_size * num_heads * seq_len * seq_len * head_dim
        double qk_flops = static_cast<double>(batch_size * num_heads * sequence_length * sequence_length * head_dimension);
        double softmax_flops = static_cast<double>(batch_size * num_heads * sequence_length * sequence_length * 4);
        double av_flops = static_cast<double>(batch_size * num_heads * sequence_length * sequence_length * head_dimension);
        return qk_flops + softmax_flops + av_flops;
    }
};

// Global benchmark configuration
static BenchmarkConfig g_benchmark_config;
static std::vector<BenchmarkResult> g_benchmark_results;
static std::unique_ptr<MemoryPool> g_memory_pool;

// Utility function to generate random data
std::vector<float> generate_random_data(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    return data;
}

// Memory usage tracking utility
class MemoryTracker {
public:
    MemoryTracker() : peak_memory_(0), current_memory_(0) {}
    
    void record_allocation(size_t size) {
        current_memory_ += size;
        peak_memory_ = std::max(peak_memory_, current_memory_);
    }
    
    void record_deallocation(size_t size) {
        current_memory_ = (current_memory_ >= size) ? current_memory_ - size : 0;
    }
    
    size_t get_peak_memory() const { return peak_memory_; }
    size_t get_current_memory() const { return current_memory_; }
    
    void reset() {
        peak_memory_ = 0;
        current_memory_ = 0;
    }
    
private:
    size_t peak_memory_;
    size_t current_memory_;
};

static MemoryTracker g_memory_tracker;

// Benchmark a specific attention configuration
PerformanceMetrics benchmark_attention_config(
    BackendType backend,
    size_t batch_size,
    size_t sequence_length,
    size_t num_heads,
    size_t head_dimension,
    const BenchmarkConfig& config) {
    
    PerformanceMetrics metrics;
    
    try {
        // Create attention interface
        auto attention = BackendFactory::createFlashAttention(backend);
        if (!attention) {
            metrics.error_message = "Failed to create backend";
            return metrics;
        }
        
        // Configure attention parameters
        AttentionParams params;
        params.batch_size = batch_size;
        params.sequence_length = sequence_length;
        params.num_heads = num_heads;
        params.head_dimension = head_dimension;
        params.causal_mask = config.enable_causal_mask;
        params.scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dimension));
        
        attention->configure(params);
        
        // Calculate tensor size
        size_t total_elements = batch_size * sequence_length * num_heads * head_dimension;
        
        // Generate test data
        std::vector<float> query_data = generate_random_data(total_elements);
        std::vector<float> key_data = generate_random_data(total_elements);
        std::vector<float> value_data = generate_random_data(total_elements);
        std::vector<float> output_data(total_elements, 0.0f);
        std::vector<float> grad_output_data = generate_random_data(total_elements, 0.01f, 0.1f);
        std::vector<float> grad_query_data(total_elements, 0.0f);
        std::vector<float> grad_key_data(total_elements, 0.0f);
        std::vector<float> grad_value_data(total_elements, 0.0f);
        
        // Create tensor shapes
        TensorShape tensor_shape({batch_size, sequence_length, num_heads, head_dimension}, 
                                DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
        
        // Setup workspace if needed
        size_t workspace_size = attention->get_workspace_size(params);
        std::vector<uint8_t> workspace;
        if (workspace_size > 0) {
            workspace.resize(workspace_size);
            attention->set_workspace(workspace.data(), workspace_size);
        }
        
        // Reset memory tracker
        g_memory_tracker.reset();
        g_memory_tracker.record_allocation(total_elements * sizeof(float) * 4); // Q, K, V, O
        if (workspace_size > 0) {
            g_memory_tracker.record_allocation(workspace_size);
        }
        
        // Warmup iterations
        for (int i = 0; i < config.warmup_iterations; ++i) {
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output_data.data()
            );
        }
        
        // Benchmark forward pass
        Timer forward_timer;
        forward_timer.start();
        
        for (int i = 0; i < config.benchmark_iterations; ++i) {
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output_data.data()
            );
        }
        
        forward_timer.stop();
        metrics.forward_time_ms = forward_timer.to_milliseconds() / config.benchmark_iterations;
        
        // Benchmark backward pass
        Timer backward_timer;
        backward_timer.start();
        
        for (int i = 0; i < config.benchmark_iterations; ++i) {
            attention->backward(
                tensor_shape, grad_output_data.data(),
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, grad_query_data.data(),
                tensor_shape, grad_key_data.data(),
                tensor_shape, grad_value_data.data()
            );
        }
        
        backward_timer.stop();
        metrics.backward_time_ms = backward_timer.to_milliseconds() / config.benchmark_iterations;
        
        // Calculate derived metrics
        metrics.total_time_ms = metrics.forward_time_ms + metrics.backward_time_ms;
        metrics.memory_usage_bytes = g_memory_tracker.get_current_memory();
        metrics.peak_memory_bytes = g_memory_tracker.get_peak_memory();
        
        // Calculate GFLOPS and throughput
        double theoretical_flops = static_cast<double>(batch_size * num_heads * sequence_length * sequence_length * head_dimension * 2); // Forward + backward
        metrics.gflops = (theoretical_flops / 1e9) / (metrics.total_time_ms / 1000.0);
        
        size_t total_tokens = batch_size * sequence_length;
        metrics.throughput_tokens_per_sec = static_cast<double>(total_tokens) / (metrics.forward_time_ms / 1000.0);
        
        metrics.success = true;
        
    } catch (const std::exception& e) {
        metrics.error_message = e.what();
        metrics.success = false;
    }
    
    return metrics;
}

// Benchmark function for Google Benchmark framework
static void BM_FlashAttention(benchmark::State& state) {
    // Extract parameters from benchmark state
    BackendType backend = static_cast<BackendType>(state.range(0));
    size_t batch_size = static_cast<size_t>(state.range(1));
    size_t sequence_length = static_cast<size_t>(state.range(2));
    size_t num_heads = static_cast<size_t>(state.range(3));
    size_t head_dimension = static_cast<size_t>(state.range(4));
    
    // Skip if backend is not available
    if (!BackendFactory::isBackendAvailable(backend)) {
        state.SkipWithError("Backend not available");
        return;
    }
    
    try {
        // Create attention interface
        auto attention = BackendFactory::createFlashAttention(backend);
        
        // Configure attention parameters
        AttentionParams params;
        params.batch_size = batch_size;
        params.sequence_length = sequence_length;
        params.num_heads = num_heads;
        params.head_dimension = head_dimension;
        params.causal_mask = g_benchmark_config.enable_causal_mask;
        params.scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dimension));
        
        attention->configure(params);
        
        // Generate test data
        size_t total_elements = batch_size * sequence_length * num_heads * head_dimension;
        std::vector<float> query_data = generate_random_data(total_elements);
        std::vector<float> key_data = generate_random_data(total_elements);
        std::vector<float> value_data = generate_random_data(total_elements);
        std::vector<float> output_data(total_elements, 0.0f);
        
        TensorShape tensor_shape({batch_size, sequence_length, num_heads, head_dimension}, 
                                DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
        
        // Setup workspace
        size_t workspace_size = attention->get_workspace_size(params);
        std::vector<uint8_t> workspace;
        if (workspace_size > 0) {
            workspace.resize(workspace_size);
            attention->set_workspace(workspace.data(), workspace_size);
        }
        
        // Benchmark loop
        for (auto _ : state) {
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output_data.data()
            );
        }
        
        // Set benchmark counters
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 
                               static_cast<int64_t>(total_elements * sizeof(float) * 4)); // Q, K, V, O
        
        double theoretical_flops = static_cast<double>(batch_size * num_heads * sequence_length * sequence_length * head_dimension);
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(theoretical_flops));
        
        // Add custom counters
        state.counters["BatchSize"] = static_cast<double>(batch_size);
        state.counters["SeqLength"] = static_cast<double>(sequence_length);
        state.counters["NumHeads"] = static_cast<double>(num_heads);
        state.counters["HeadDim"] = static_cast<double>(head_dimension);
        state.counters["TotalElements"] = static_cast<double>(total_elements);
        state.counters["GFLOPS"] = benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
        
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Benchmark function for PagedAttention
static void BM_PagedAttention(benchmark::State& state) {
    // Extract parameters from benchmark state
    BackendType backend = static_cast<BackendType>(state.range(0));
    size_t batch_size = static_cast<size_t>(state.range(1));
    size_t sequence_length = static_cast<size_t>(state.range(2));
    size_t num_heads = static_cast<size_t>(state.range(3));
    size_t head_dimension = static_cast<size_t>(state.range(4));
    
    // Skip if backend is not available
    if (!BackendFactory::isBackendAvailable(backend)) {
        state.SkipWithError("Backend not available");
        return;
    }
    
    try {
        // Create attention interface
        auto attention = BackendFactory::createPagedAttention(backend);
        
        // Configure attention parameters
        AttentionParams params;
        params.batch_size = batch_size;
        params.sequence_length = sequence_length;
        params.num_heads = num_heads;
        params.head_dimension = head_dimension;
        params.max_sequence_length = sequence_length * 2; // Allow for growth
        params.page_size = 16;
        
        attention->configure(params);
        
        // Generate test data
        size_t total_elements = batch_size * sequence_length * num_heads * head_dimension;
        std::vector<float> query_data = generate_random_data(total_elements);
        std::vector<float> key_data = generate_random_data(total_elements);
        std::vector<float> value_data = generate_random_data(total_elements);
        std::vector<float> output_data(total_elements, 0.0f);
        
        TensorShape tensor_shape({batch_size, sequence_length, num_heads, head_dimension}, 
                                DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
        
        // Benchmark loop
        for (auto _ : state) {
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output_data.data()
            );
        }
        
        // Set benchmark counters
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 
                               static_cast<int64_t>(total_elements * sizeof(float) * 4));
        
        // Add custom counters
        state.counters["BatchSize"] = static_cast<double>(batch_size);
        state.counters["SeqLength"] = static_cast<double>(sequence_length);
        state.counters["NumHeads"] = static_cast<double>(num_heads);
        state.counters["HeadDim"] = static_cast<double>(head_dimension);
        state.counters["PageSize"] = static_cast<double>(params.page_size);
        
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Register benchmarks for different configurations
void RegisterBenchmarks() {
    // Initialize benchmark configuration
    g_benchmark_config.backends = BackendFactory::getAvailableBackends();
    
    // Register FlashAttention benchmarks
    for (BackendType backend : g_benchmark_config.backends) {
        for (size_t batch_size : g_benchmark_config.batch_sizes) {
            for (size_t seq_len : g_benchmark_config.sequence_lengths) {
                for (size_t num_heads : g_benchmark_config.num_heads) {
                    for (size_t head_dim : g_benchmark_config.head_dimensions) {
                        // Skip very large configurations to avoid memory issues
                        size_t total_elements = batch_size * seq_len * num_heads * head_dim;
                        if (total_elements > 100000000) { // 100M elements
                            continue;
                        }
                        
                        std::string name = "BM_FlashAttention/" + 
                                         BackendFactory::backendTypeToString(backend) + "/" +
                                         std::to_string(batch_size) + "/" +
                                         std::to_string(seq_len) + "/" +
                                         std::to_string(num_heads) + "/" +
                                         std::to_string(head_dim);
                        
                        benchmark::RegisterBenchmark(name.c_str(), BM_FlashAttention)
                            ->Args({static_cast<int64_t>(backend), static_cast<int64_t>(batch_size), 
                                   static_cast<int64_t>(seq_len), static_cast<int64_t>(num_heads), 
                                   static_cast<int64_t>(head_dim)})
                            ->Unit(benchmark::kMicrosecond)
                            ->UseRealTime();
                    }
                }
            }
        }
    }
    
    // Register PagedAttention benchmarks (subset of configurations)
    for (BackendType backend : g_benchmark_config.backends) {
        for (size_t seq_len : {256, 512, 1024}) {
            for (size_t num_heads : {4, 8}) {
                for (size_t head_dim : {64, 128}) {
                    std::string name = "BM_PagedAttention/" + 
                                     BackendFactory::backendTypeToString(backend) + "/" +
                                     "2/" + // Fixed batch size
                                     std::to_string(seq_len) + "/" +
                                     std::to_string(num_heads) + "/" +
                                     std::to_string(head_dim);
                    
                    benchmark::RegisterBenchmark(name.c_str(), BM_PagedAttention)
                        ->Args({static_cast<int64_t>(backend), 2, 
                               static_cast<int64_t>(seq_len), static_cast<int64_t>(num_heads), 
                               static_cast<int64_t>(head_dim)})
                        ->Unit(benchmark::kMicrosecond)
                        ->UseRealTime();
                }
            }
        }
    }
}

// Comprehensive benchmark function
void RunComprehensiveBenchmarks() {
    std::cout << "Running comprehensive attention benchmarks..." << std::endl;
    
    // Initialize memory pool
    g_memory_pool = std::make_unique<MemoryPool>();
    
    // Clear previous results
    g_benchmark_results.clear();
    
    // Get available backends
    auto available_backends = BackendFactory::getAvailableBackends();
    
    std::cout << "Available backends: ";
    for (BackendType backend : available_backends) {
        std::cout << BackendFactory::backendTypeToString(backend) << " ";
    }
    std::cout << std::endl;
    
    // Run benchmarks for each configuration
    for (BackendType backend : available_backends) {
        std::cout << "\nBenchmarking " << BackendFactory::backendTypeToString(backend) << " backend:" << std::endl;
        
        for (size_t batch_size : g_benchmark_config.batch_sizes) {
            for (size_t seq_len : g_benchmark_config.sequence_lengths) {
                for (size_t num_heads : g_benchmark_config.num_heads) {
                    for (size_t head_dim : g_benchmark_config.head_dimensions) {
                        // Skip very large configurations
                        size_t total_elements = batch_size * seq_len * num_heads * head_dim;
                        if (total_elements > 50000000) { // 50M elements
                            continue;
                        }
                        
                        std::cout << "  Testing config: batch=" << batch_size 
                                  << ", seq_len=" << seq_len 
                                  << ", heads=" << num_heads 
                                  << ", head_dim=" << head_dim << std::flush;
                        
                        BenchmarkResult result;
                        result.backend = backend;
                        result.backend_name = BackendFactory::backendTypeToString(backend);
                        result.batch_size = batch_size;
                        result.sequence_length = seq_len;
                        result.num_heads = num_heads;
                        result.head_dimension = head_dim;
                        
                        result.metrics = benchmark_attention_config(
                            backend, batch_size, seq_len, num_heads, head_dim, g_benchmark_config);
                        
                        g_benchmark_results.push_back(result);
                        
                        if (result.metrics.success) {
                            std::cout << " - OK (" << std::fixed << std::setprecision(2) 
                                      << result.metrics.forward_time_ms << "ms)" << std::endl;
                        } else {
                            std::cout << " - FAILED (" << result.metrics.error_message << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // Generate performance report
    GeneratePerformanceReport();
}

// Generate performance report
void GeneratePerformanceReport() {
    std::cout << "\n=== Performance Report ===" << std::endl;
    
    // Group results by backend
    std::map<BackendType, std::vector<BenchmarkResult>> backend_results;
    for (const auto& result : g_benchmark_results) {
        if (result.metrics.success) {
            backend_results[result.backend].push_back(result);
        }
    }
    
    // Print summary table
    std::cout << std::left << std::setw(12) << "Backend"
              << std::setw(12) << "Config"
              << std::setw(15) << "Forward(ms)"
              << std::setw(15) << "Backward(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Tokens/sec"
              << std::setw(12) << "Memory(MB)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& [backend, results] : backend_results) {
        std::string backend_name = BackendFactory::backendTypeToString(backend);
        
        for (const auto& result : results) {
            std::string config = std::to_string(result.batch_size) + "x" +
                               std::to_string(result.sequence_length) + "x" +
                               std::to_string(result.num_heads) + "x" +
                               std::to_string(result.head_dimension);
            
            std::cout << std::left << std::setw(12) << backend_name
                      << std::setw(12) << config
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.metrics.forward_time_ms
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.metrics.backward_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.metrics.gflops
                      << std::setw(15) << std::fixed << std::setprecision(0) << result.metrics.throughput_tokens_per_sec
                      << std::setw(12) << std::fixed << std::setprecision(1) << (result.metrics.peak_memory_bytes / 1024.0 / 1024.0)
                      << std::endl;
        }
        
        if (!results.empty()) {
            std::cout << std::endl;
        }
    }
    
    // Find best performing configurations
    std::cout << "\n=== Best Performing Configurations ===" << std::endl;
    
    if (!g_benchmark_results.empty()) {
        // Best throughput
        auto best_throughput = std::max_element(g_benchmark_results.begin(), g_benchmark_results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.metrics.success && b.metrics.success ? 
                       a.metrics.throughput_tokens_per_sec < b.metrics.throughput_tokens_per_sec :
                       !a.metrics.success;
            });
        
        if (best_throughput != g_benchmark_results.end() && best_throughput->metrics.success) {
            std::cout << "Best Throughput: " << best_throughput->backend_name 
                      << " with " << std::fixed << std::setprecision(0) 
                      << best_throughput->metrics.throughput_tokens_per_sec << " tokens/sec" << std::endl;
        }
        
        // Best GFLOPS
        auto best_gflops = std::max_element(g_benchmark_results.begin(), g_benchmark_results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.metrics.success && b.metrics.success ? 
                       a.metrics.gflops < b.metrics.gflops :
                       !a.metrics.success;
            });
        
        if (best_gflops != g_benchmark_results.end() && best_gflops->metrics.success) {
            std::cout << "Best GFLOPS: " << best_gflops->backend_name 
                      << " with " << std::fixed << std::setprecision(2) 
                      << best_gflops->metrics.gflops << " GFLOPS" << std::endl;
        }
        
        // Most memory efficient
        auto most_efficient = std::min_element(g_benchmark_results.begin(), g_benchmark_results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.metrics.success && b.metrics.success ? 
                       a.metrics.peak_memory_bytes < b.metrics.peak_memory_bytes :
                       a.metrics.success;
            });
        
        if (most_efficient != g_benchmark_results.end() && most_efficient->metrics.success) {
            std::cout << "Most Memory Efficient: " << most_efficient->backend_name 
                      << " with " << std::fixed << std::setprecision(1) 
                      << (most_efficient->metrics.peak_memory_bytes / 1024.0 / 1024.0) << " MB peak memory" << std::endl;
        }
    }
    
    // Save results to JSON file
    SaveResultsToJSON("benchmark_results.json");
}

// Save results to JSON file
void SaveResultsToJSON(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"benchmark_results\": [\n";
    
    for (size_t i = 0; i < g_benchmark_results.size(); ++i) {
        const auto& result = g_benchmark_results[i];
        
        file << "    {\n";
        file << "      \"backend\": \"" << result.backend_name << "\",\n";
        file << "      \"batch_size\": " << result.batch_size << ",\n";
        file << "      \"sequence_length\": " << result.sequence_length << ",\n";
        file << "      \"num_heads\": " << result.num_heads << ",\n";
        file << "      \"head_dimension\": " << result.head_dimension << ",\n";
        file << "      \"total_elements\": " << result.get_total_elements() << ",\n";
        file << "      \"success\": " << (result.metrics.success ? "true" : "false") << ",\n";
        
        if (result.metrics.success) {
            file << "      \"forward_time_ms\": " << result.metrics.forward_time_ms << ",\n";
            file << "      \"backward_time_ms\": " << result.metrics.backward_time_ms << ",\n";
            file << "      \"total_time_ms\": " << result.metrics.total_time_ms << ",\n";
            file << "      \"memory_usage_bytes\": " << result.metrics.memory_usage_bytes << ",\n";
            file << "      \"peak_memory_bytes\": " << result.metrics.peak_memory_bytes << ",\n";
            file << "      \"gflops\": " << result.metrics.gflops << ",\n";
            file << "      \"throughput_tokens_per_sec\": " << result.metrics.throughput_tokens_per_sec << ",\n";
            file << "      \"theoretical_flops\": " << result.get_theoretical_flops() << "\n";
        } else {
            file << "      \"error_message\": \"" << result.metrics.error_message << "\"\n";
        }
        
        file << "    }";
        if (i < g_benchmark_results.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ],\n";
    file << "  
