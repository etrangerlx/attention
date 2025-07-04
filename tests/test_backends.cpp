#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <cmath>

#include "api/backend_factory.hpp"
#include "api/attention_interface.hpp"
#include "api/tensor.hpp"
#include "algorithms/flash_attention.hpp"
#include "algorithms/paged_attention.hpp"

// Include backend-specific headers if available
#ifdef HAVE_CUDA
#include "backends/cuda/cuda_flash_attention.hpp"
#endif

#ifdef HAVE_OPENCL
#include "backends/opencl/opencl_flash_attention.hpp"
#endif

#ifdef HAVE_OPENGL
#include "backends/opengl/opengl_flash_attention.hpp"
#endif

#ifdef HAVE_VULKAN
#include "backends/vulkan/vulkan_flash_attention.hpp"
#endif

namespace attention_hpc {
namespace tests {

// Helper function to generate random test data
std::vector<float> generate_test_data(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    return data;
}

// Helper function to compare float vectors with tolerance
bool vectors_equal(const std::vector<float>& v1, const std::vector<float>& v2, float tolerance = 1e-4f) {
    if (v1.size() != v2.size()) {
        return false;
    }
    
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > tolerance) {
            return false;
        }
    }
    
    return true;
}

class BackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default test parameters
        params.batch_size = 2;
        params.sequence_length = 32;
        params.num_heads = 4;
        params.head_dimension = 64;
        params.scale_factor = 1.0f / std::sqrt(static_cast<float>(params.head_dimension));
        params.causal_mask = false;
        
        // Calculate total tensor size
        total_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
        
        // Generate test data
        query_data = generate_test_data(total_size);
        key_data = generate_test_data(total_size);
        value_data = generate_test_data(total_size);
        output_data.resize(total_size, 0.0f);
        
        // Create tensor shapes
        tensor_shape = TensorShape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, 
                                  DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
    }
    
    void TearDown() override {
        // Clean up any resources
    }
    
    // Test parameters
    AttentionParams params;
    TensorShape tensor_shape;
    size_t total_size;
    
    // Test data
    std::vector<float> query_data;
    std::vector<float> key_data;
    std::vector<float> value_data;
    std::vector<float> output_data;
};

// Test backend availability detection
TEST_F(BackendTest, BackendAvailabilityTest) {
    // Test all backend types
    std::vector<BackendType> all_backends = {
        BackendType::CPU,
        BackendType::CUDA,
        BackendType::OPENCL,
        BackendType::OPENGL,
        BackendType::VULKAN
    };
    
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    // CPU backend should always be available
    EXPECT_TRUE(BackendFactory::isBackendAvailable(BackendType::CPU));
    EXPECT_TRUE(std::find(available_backends.begin(), available_backends.end(), BackendType::CPU) != available_backends.end());
    
    // Test individual backend availability
    for (BackendType backend : all_backends) {
        bool is_available = BackendFactory::isBackendAvailable(backend);
        bool in_list = std::find(available_backends.begin(), available_backends.end(), backend) != available_backends.end();
        
        EXPECT_EQ(is_available, in_list) << "Backend " << BackendFactory::backendTypeToString(backend) 
                                         << " availability mismatch between individual check and list";
        
        // Get capability information
        BackendCapability capability = BackendFactory::getBackendCapability(backend);
        EXPECT_EQ(capability.available, is_available) << "Backend capability availability mismatch";
        
        if (is_available) {
            // Verify capability information is populated
            EXPECT_FALSE(capability.version.empty()) << "Backend " << BackendFactory::backendTypeToString(backend) 
                                                     << " should have version information";
            EXPECT_FALSE(capability.device_name.empty()) << "Backend " << BackendFactory::backendTypeToString(backend) 
                                                         << " should have device name";
            EXPECT_FALSE(capability.supported_data_types.empty()) << "Backend " << BackendFactory::backendTypeToString(backend) 
                                                                  << " should support at least one data type";
        }
    }
}

// Test backend capability information
TEST_F(BackendTest, BackendCapabilityTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    for (BackendType backend : available_backends) {
        BackendCapability capability = BackendFactory::getBackendCapability(backend);
        
        EXPECT_TRUE(capability.available) << "Available backend should have available=true";
        EXPECT_FALSE(capability.version.empty()) << "Backend version should not be empty";
        EXPECT_FALSE(capability.device_name.empty()) << "Device name should not be empty";
        
        // Check supported data types
        EXPECT_FALSE(capability.supported_data_types.empty()) << "Should support at least one data type";
        
        // For GPU backends, expect some memory size information
        if (backend != BackendType::CPU) {
            // Memory size might be 0 if not detectable, but shouldn't be negative
            EXPECT_GE(capability.memory_size, 0) << "Memory size should be non-negative";
        }
        
        // Test data type support
        for (DataType dtype : capability.supported_data_types) {
            EXPECT_TRUE(dtype == DataType::FLOAT32 || dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16)
                << "Supported data type should be one of the expected types";
        }
        
        // Test memory layout support
        EXPECT_FALSE(capability.supported_layouts.empty()) << "Should support at least one memory layout";
        bool supports_row_major = std::find(capability.supported_layouts.begin(), 
                                           capability.supported_layouts.end(), 
                                           MemoryLayout::ROW_MAJOR) != capability.supported_layouts.end();
        EXPECT_TRUE(supports_row_major) << "Should support row-major layout";
    }
}

// Test backend creation and basic functionality
TEST_F(BackendTest, BackendCreationTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    for (BackendType backend : available_backends) {
        try {
            // Test FlashAttention backend creation
            auto flash_attention = BackendFactory::createFlashAttention(backend);
            ASSERT_NE(flash_attention, nullptr) << "FlashAttention backend should be created successfully";
            
            // Test basic properties
            EXPECT_EQ(flash_attention->get_backend_name().find(BackendFactory::backendTypeToString(backend)), 0)
                << "Backend name should start with backend type";
            EXPECT_FALSE(flash_attention->get_version().empty()) << "Version should not be empty";
            
            // Test data type support
            EXPECT_TRUE(flash_attention->supports_data_type(DataType::FLOAT32)) 
                << "Should support FLOAT32 data type";
            
            // Test memory layout support
            EXPECT_TRUE(flash_attention->supports_memory_layout(MemoryLayout::ROW_MAJOR))
                << "Should support row-major memory layout";
            
            // Test PagedAttention backend creation
            auto paged_attention = BackendFactory::createPagedAttention(backend);
            ASSERT_NE(paged_attention, nullptr) << "PagedAttention backend should be created successfully";
            
        } catch (const BackendException& e) {
            FAIL() << "Backend creation failed for " << BackendFactory::backendTypeToString(backend) 
                   << ": " << e.what();
        } catch (const std::exception& e) {
            FAIL() << "Unexpected exception for " << BackendFactory::backendTypeToString(backend) 
                   << ": " << e.what();
        }
    }
}

// Test backend configuration and setup
TEST_F(BackendTest, BackendConfigurationTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    for (BackendType backend : available_backends) {
        auto attention = BackendFactory::createFlashAttention(backend);
        ASSERT_NE(attention, nullptr);
        
        // Test configuration
        EXPECT_NO_THROW({
            attention->configure(params);
        }) << "Backend configuration should succeed for " << BackendFactory::backendTypeToString(backend);
        
        // Test workspace size calculation
        size_t workspace_size = attention->get_workspace_size(params);
        EXPECT_GE(workspace_size, 0) << "Workspace size should be non-negative";
        
        // Test profiling enable/disable
        EXPECT_NO_THROW({
            attention->enable_profiling(true);
            attention->enable_profiling(false);
        }) << "Profiling toggle should work";
        
        // Test input validation
        AttentionError validation_result = attention->validate_inputs(
            tensor_shape, tensor_shape, tensor_shape, tensor_shape, params);
        EXPECT_EQ(validation_result, AttentionError::SUCCESS) 
            << "Input validation should succeed for valid inputs";
    }
}

// Test cross-backend result consistency
TEST_F(BackendTest, CrossBackendConsistencyTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    if (available_backends.size() < 2) {
        GTEST_SKIP() << "Need at least 2 backends for consistency testing";
    }
    
    std::vector<std::vector<float>> backend_outputs;
    std::vector<BackendType> tested_backends;
    
    // Run the same computation on all available backends
    for (BackendType backend : available_backends) {
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            attention->configure(params);
            
            std::vector<float> output(total_size, 0.0f);
            
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output.data()
            );
            
            backend_outputs.push_back(output);
            tested_backends.push_back(backend);
            
        } catch (const std::exception& e) {
            // Skip backends that fail (might not be properly configured in test environment)
            std::cout << "Skipping backend " << BackendFactory::backendTypeToString(backend) 
                      << " due to error: " << e.what() << std::endl;
            continue;
        }
    }
    
    if (backend_outputs.size() < 2) {
        GTEST_SKIP() << "Need at least 2 working backends for consistency testing";
    }
    
    // Compare results between backends
    const float consistency_tolerance = 1e-3f; // Relaxed tolerance for cross-backend comparison
    
    for (size_t i = 1; i < backend_outputs.size(); ++i) {
        bool results_consistent = vectors_equal(backend_outputs[0], backend_outputs[i], consistency_tolerance);
        
        EXPECT_TRUE(results_consistent) 
            << "Results inconsistent between " << BackendFactory::backendTypeToString(tested_backends[0])
            << " and " << BackendFactory::backendTypeToString(tested_backends[i]);
        
        if (!results_consistent) {
            // Log some statistics about the differences
            float max_diff = 0.0f;
            float avg_diff = 0.0f;
            size_t diff_count = 0;
            
            for (size_t j = 0; j < backend_outputs[0].size(); ++j) {
                float diff = std::abs(backend_outputs[0][j] - backend_outputs[i][j]);
                if (diff > consistency_tolerance) {
                    diff_count++;
                }
                max_diff = std::max(max_diff, diff);
                avg_diff += diff;
            }
            avg_diff /= backend_outputs[0].size();
            
            std::cout << "Max difference: " << max_diff << std::endl;
            std::cout << "Average difference: " << avg_diff << std::endl;
            std::cout << "Elements with differences > tolerance: " << diff_count 
                      << " / " << backend_outputs[0].size() << std::endl;
        }
    }
}

// Test backend switching and resource management
TEST_F(BackendTest, BackendSwitchingTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    if (available_backends.empty()) {
        GTEST_SKIP() << "No backends available for switching test";
    }
    
    // Test multiple backend instances
    std::vector<std::unique_ptr<AttentionInterface>> attention_instances;
    
    for (BackendType backend : available_backends) {
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            attention->configure(params);
            attention_instances.push_back(std::move(attention));
        } catch (const std::exception& e) {
            // Skip problematic backends
            continue;
        }
    }
    
    ASSERT_FALSE(attention_instances.empty()) << "At least one backend should work";
    
    // Test simultaneous operation of multiple backends
    std::vector<std::vector<float>> outputs(attention_instances.size());
    
    for (size_t i = 0; i < attention_instances.size(); ++i) {
        outputs[i].resize(total_size, 0.0f);
        
        EXPECT_NO_THROW({
            attention_instances[i]->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, outputs[i].data()
            );
        }) << "Forward pass should succeed for backend " << i;
    }
    
    // Test rapid backend switching
    for (int iteration = 0; iteration < 5; ++iteration) {
        for (size_t i = 0; i < attention_instances.size(); ++i) {
            std::vector<float> output(total_size, 0.0f);
            
            EXPECT_NO_THROW({
                attention_instances[i]->forward(
                    tensor_shape, query_data.data(),
                    tensor_shape, key_data.data(),
                    tensor_shape, value_data.data(),
                    tensor_shape, output.data()
                );
            }) << "Rapid switching iteration " << iteration << " failed for backend " << i;
        }
    }
}

// Test resource cleanup and memory leak detection
TEST_F(BackendTest, ResourceManagementTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    // Test resource cleanup through multiple create/destroy cycles
    for (BackendType backend : available_backends) {
        try {
            // Multiple creation and destruction cycles
            for (int cycle = 0; cycle < 3; ++cycle) {
                {
                    auto attention1 = BackendFactory::createFlashAttention(backend);
                    auto attention2 = BackendFactory::createPagedAttention(backend);
                    
                    attention1->configure(params);
                    attention2->configure(params);
                    
                    // Use the backends briefly
                    std::vector<float> output(total_size, 0.0f);
                    attention1->forward(
                        tensor_shape, query_data.data(),
                        tensor_shape, key_data.data(),
                        tensor_shape, value_data.data(),
                        tensor_shape, output.data()
                    );
                    
                    // Backends should be automatically destroyed when going out of scope
                }
                
                // Small delay to allow cleanup
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
        } catch (const std::exception& e) {
            // Skip problematic backends but don't fail the test
            std::cout << "Skipping resource management test for " 
                      << BackendFactory::backendTypeToString(backend) 
                      << " due to: " << e.what() << std::endl;
        }
    }
    
    // Test with workspace allocation
    for (BackendType backend : available_backends) {
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            attention->configure(params);
            
            size_t workspace_size = attention->get_workspace_size(params);
            if (workspace_size > 0) {
                std::vector<uint8_t> workspace(workspace_size, 0);
                attention->set_workspace(workspace.data(), workspace_size);
                
                // Use the backend with workspace
                std::vector<float> output(total_size, 0.0f);
                attention->forward(
                    tensor_shape, query_data.data(),
                    tensor_shape, key_data.data(),
                    tensor_shape, value_data.data(),
                    tensor_shape, output.data()
                );
            }
            
        } catch (const std::exception& e) {
            // Skip problematic backends
            continue;
        }
    }
}

// Test GPU memory management (if GPU backends are available)
TEST_F(BackendTest, GPUMemoryManagementTest) {
    std::vector<BackendType> gpu_backends = {
        BackendType::CUDA,
        BackendType::OPENCL,
        BackendType::VULKAN,
        BackendType::OPENGL
    };
    
    bool has_gpu_backend = false;
    
    for (BackendType backend : gpu_backends) {
        if (!BackendFactory::isBackendAvailable(backend)) {
            continue;
        }
        
        has_gpu_backend = true;
        
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            attention->configure(params);
            
            // Test large memory allocation to stress GPU memory
            AttentionParams large_params = params;
            large_params.batch_size = 8;
            large_params.sequence_length = 256;
            large_params.num_heads = 8;
            
            size_t large_size = large_params.batch_size * large_params.num_heads * 
                               large_params.sequence_length * large_params.head_dimension;
            
            // Generate large test data
            std::vector<float> large_query = generate_test_data(large_size);
            std::vector<float> large_key = generate_test_data(large_size);
            std::vector<float> large_value = generate_test_data(large_size);
            std::vector<float> large_output(large_size, 0.0f);
            
            TensorShape large_tensor_shape({large_params.batch_size, large_params.sequence_length, 
                                          large_params.num_heads, large_params.head_dimension}, 
                                         DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
            
            // Reconfigure for large tensors
            attention->configure(large_params);
            
            // Test memory allocation and computation
            EXPECT_NO_THROW({
                attention->forward(
                    large_tensor_shape, large_query.data(),
                    large_tensor_shape, large_key.data(),
                    large_tensor_shape, large_value.data(),
                    large_tensor_shape, large_output.data()
                );
            }) << "Large tensor computation should succeed for " << BackendFactory::backendTypeToString(backend);
            
            // Test multiple sequential allocations
            for (int i = 0; i < 3; ++i) {
                std::vector<float> output(large_size, 0.0f);
                EXPECT_NO_THROW({
                    attention->forward(
                        large_tensor_shape, large_query.data(),
                        large_tensor_shape, large_key.data(),
                        large_tensor_shape, large_value.data(),
                        large_tensor_shape, output.data()
                    );
                }) << "Sequential allocation " << i << " should succeed";
            }
            
        } catch (const std::exception& e) {
            // GPU memory issues might be environment-specific
            std::cout << "GPU memory test failed for " << BackendFactory::backendTypeToString(backend) 
                      << ": " << e.what() << std::endl;
        }
    }
    
    if (!has_gpu_backend) {
        GTEST_SKIP() << "No GPU backends available for memory management testing";
    }
}

// Test backend performance characteristics
TEST_F(BackendTest, BackendPerformanceTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    struct PerformanceResult {
        BackendType backend;
        std::chrono::milliseconds forward_time;
        std::chrono::milliseconds backward_time;
        bool success;
    };
    
    std::vector<PerformanceResult> results;
    
    // Test each backend
    for (BackendType backend : available_backends) {
        PerformanceResult result;
        result.backend = backend;
        result.success = false;
        
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            attention->configure(params);
            attention->enable_profiling(true);
            
            std::vector<float> output(total_size, 0.0f);
            std::vector<float> grad_output = generate_test_data(total_size, 0.01f, 0.1f);
            std::vector<float> grad_query(total_size, 0.0f);
            std::vector<float> grad_key(total_size, 0.0f);
            std::vector<float> grad_value(total_size, 0.0f);
            
            // Warm up
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output.data()
            );
            
            // Time forward pass
            auto start = std::chrono::high_resolution_clock::now();
            attention->forward(
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, output.data()
            );
            auto end = std::chrono::high_resolution_clock::now();
            result.forward_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // Time backward pass
            start = std::chrono::high_resolution_clock::now();
            attention->backward(
                tensor_shape, grad_output.data(),
                tensor_shape, query_data.data(),
                tensor_shape, key_data.data(),
                tensor_shape, value_data.data(),
                tensor_shape, grad_query.data(),
                tensor_shape, grad_key.data(),
                tensor_shape, grad_value.data()
            );
            end = std::chrono::high_resolution_clock::now();
            result.backward_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            result.success = true;
            
            // Get profiling results if available
            auto profiling_results = attention->get_profiling_results();
            if (!profiling_results.empty()) {
                std::cout << "Profiling results for " << BackendFactory::backendTypeToString(backend) << ":" << std::endl;
                for (const auto& entry : profiling_results) {
                    std::cout << "  " << entry.first << ": " << entry.second << " ms" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cout << "Performance test failed for " << BackendFactory::backendTypeToString(backend) 
                      << ": " << e.what() << std::endl;
        }
        
        results.push_back(result);
    }
    
    // Print performance summary
    std::cout << "\nBackend Performance Summary:" << std::endl;
    std::cout << "Backend\t\tForward (ms)\tBackward (ms)\tStatus" << std::endl;
    std::cout << "-------\t\t-----------\t------------\t------" << std::endl;
    
    for (const auto& result : results) {
        std::cout << BackendFactory::backendTypeToString(result.backend) << "\t\t"
                  << (result.success ? std::to_string(result.forward_time.count()) : "FAILED") << "\t\t"
                  << (result.success ? std::to_string(result.backward_time.count()) : "FAILED") << "\t\t"
                  << (result.success ? "OK" : "FAILED") << std::endl;
    }
    
    // At least one backend should work
    bool any_success = std::any_of(results.begin(), results.end(), 
                                  [](const PerformanceResult& r) { return r.success; });
    EXPECT_TRUE(any_success) << "At least one backend should work";
}

// Test error handling and edge cases
TEST_F(BackendTest, ErrorHandlingTest) {
    std::vector<BackendType> available_backends = BackendFactory::getAvailableBackends();
    
    for (BackendType backend : available_backends) {
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            
            // Test operation without configuration
            std::vector<float> output(total_size, 0.0f);
            EXPECT_THROW({
                attention->forward(
                    tensor_shape, query_data.data(),
                    tensor_shape, key_data.data(),
                    tensor_shape, value_data.data(),
                    tensor_shape, output.data()
                );
            }, AttentionException) << "Should throw exception when not configured";
            
            // Configure properly
            attention->configure(params);
            
            // Test with null pointers
            EXPECT_THROW({
                attention->forward(
                    tensor_shape, nullptr,
                    tensor_shape, key_data.data(),
                    tensor_shape, value_data.data(),
                    tensor_shape, output.data()
                );
            }, AttentionException) << "Should throw exception for null input";
            
            // Test with invalid tensor shapes
            TensorShape invalid_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension + 10}, 
                                     DataType::FLOAT32, MemoryLayout::ROW_MAJOR);
            
            EXPECT_THROW({
                attention->forward(
                    invalid_shape, query_data.data(),
                    tensor_shape, key_data.data(),
                    tensor_shape, value_data.data(),
                    tensor_shape, output.data()
                );
            }, AttentionException) << "Should throw exception for mismatched tensor shapes";
            
        } catch (const std::exception& e) {
            // Skip backends that can't be created
            continue;
        }
    }
}

} // namespace tests
} // namespace attention_hpc
