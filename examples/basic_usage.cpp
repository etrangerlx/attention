#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>

// Include AttentionHPC API headers
#include "api/attention_interface.hpp"
#include "api/backend_factory.hpp"
#include "algorithms/paged_attention.hpp" // For downcasting

// Helper function to generate random data
std::vector<float> generate_random_data(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// Helper function to print a small part of a tensor
void print_tensor_sample(const std::string& name, const std::vector<float>& data, size_t num_elements = 8) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(data.size(), num_elements); ++i) {
        std::cout << data[i] << (i == std::min(data.size(), num_elements) - 1 ? "" : ", ");
    }
    if (data.size() > num_elements) {
        std::cout << "...";
    }
    std::cout << "]" << std::endl;
}

void demonstrate_flash_attention() {
    std::cout << "\n--- Demonstrating FlashAttention ---" << std::endl;

    // 1. Define attention parameters
    attention_hpc::AttentionParams params;
    params.batch_size = 2;
    params.sequence_length = 64;
    params.num_heads = 4;
    params.head_dimension = 32;

    // 2. Prepare data
    size_t total_elements = params.batch_size * params.sequence_length * params.num_heads * params.head_dimension;
    auto query_data = generate_random_data(total_elements);
    auto key_data = generate_random_data(total_elements);
    auto value_data = generate_random_data(total_elements);
    std::vector<float> output_data(total_elements, 0.0f);

    print_tensor_sample("Input Query (sample)", query_data);

    // 3. Define tensor shapes
    attention_hpc::TensorShape qkv_shape(
        {params.batch_size, params.sequence_length, params.num_heads, params.head_dimension},
        attention_hpc::DataType::FLOAT32,
        attention_hpc::MemoryLayout::ROW_MAJOR
    );
    attention_hpc::TensorShape output_shape = qkv_shape;

    // 4. Select a backend and create an instance
    auto available_backends = attention_hpc::BackendFactory::getAvailableBackends();
    if (available_backends.empty()) {
        std::cerr << "No backends available." << std::endl;
        return;
    }
    
    // Prioritize GPU backends if available, otherwise use CPU
    attention_hpc::BackendType backend_to_use = attention_hpc::BackendType::CPU;
    std::vector<attention_hpc::BackendType> preferred_order = {
        attention_hpc::BackendType::CUDA,
        attention_hpc::BackendType::VULKAN,
        attention_hpc::BackendType::OPENCL,
        attention_hpc::BackendType::OPENGL,
        attention_hpc::BackendType::CPU
    };

    for(const auto& type : preferred_order) {
        if (attention_hpc::BackendFactory::isBackendAvailable(type)) {
            backend_to_use = type;
            break;
        }
    }

    std::cout << "Using backend: " << attention_hpc::BackendFactory::backendTypeToString(backend_to_use) << std::endl;
    std::unique_ptr<attention_hpc::AttentionInterface> flash_attention;
    try {
        flash_attention = attention_hpc::BackendFactory::createFlashAttention(backend_to_use);
    } catch (const attention_hpc::AttentionException& e) {
        std::cerr << "Failed to create FlashAttention backend: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU." << std::endl;
        backend_to_use = attention_hpc::BackendType::CPU;
        flash_attention = attention_hpc::BackendFactory::createFlashAttention(backend_to_use);
    }

    // 5. Configure and run the attention computation
    flash_attention->configure(params);
    flash_attention->forward(
        qkv_shape, query_data.data(),
        qkv_shape, key_data.data(),
        qkv_shape, value_data.data(),
        output_shape, output_data.data()
    );

    std::cout << "FlashAttention forward pass completed." << std::endl;
    print_tensor_sample("Output (sample)", output_data);
    
    // Resource cleanup is handled by std::unique_ptr
}

void demonstrate_paged_attention() {
    std::cout << "\n--- Demonstrating PagedAttention ---" << std::endl;
    
    // PagedAttention typically processes one token at a time for multiple sequences
    // Here we'll simulate a simple case: one sequence, generating one token.

    // 1. Define attention parameters
    attention_hpc::AttentionParams params;
    params.batch_size = 1; // Number of sequences to process
    params.sequence_length = 1; // Generating one new token
    params.num_heads = 4;
    params.head_dimension = 32;
    params.max_sequence_length = 128; // Max length for KV cache

    // 2. Select backend
    auto available_backends = attention_hpc::BackendFactory::getAvailableBackends();
    if (available_backends.empty()) {
        std::cerr << "No backends available." << std::endl;
        return;
    }
    
    attention_hpc::BackendType backend_to_use = attention_hpc::BackendType::CPU;
     if(attention_hpc::BackendFactory::isBackendAvailable(attention_hpc::BackendType::CUDA)) {
        backend_to_use = attention_hpc::BackendType::CUDA;
    }

    std::cout << "Using backend: " << attention_hpc::BackendFactory::backendTypeToString(backend_to_use) << std::endl;
    auto paged_attention_base = attention_hpc::BackendFactory::createPagedAttention(backend_to_use);

    // We need to cast to access PagedAttention specific methods
    auto* paged_attention = dynamic_cast<attention_hpc::PagedAttention*>(paged_attention_base.get());
    if (!paged_attention) {
        std::cerr << "Failed to cast to PagedAttention interface." << std::endl;
        return;
    }

    // 3. Configure
    paged_attention->configure(params);

    // 4. Manage sequence and KV cache
    size_t sequence_id = paged_attention->create_sequence(params.max_sequence_length);
    std::cout << "Created sequence with ID: " << sequence_id << std::endl;

    // Simulate pre-existing KV cache data (e.g., from a prompt)
    size_t prompt_length = 32;
    size_t kv_elements = prompt_length * params.num_heads * params.head_dimension;
    auto key_cache_data = generate_random_data(kv_elements);
    auto value_cache_data = generate_random_data(kv_elements);
    
    paged_attention->append_kv_cache(sequence_id, key_cache_data.data(), value_cache_data.data(), 
                                     prompt_length, attention_hpc::DataType::FLOAT32);
    std::cout << "Appended " << prompt_length << " tokens to KV cache for sequence " << sequence_id << std::endl;

    // 5. Prepare input for the new token
    size_t query_elements = params.batch_size * params.sequence_length * params.num_heads * params.head_dimension;
    auto query_data = generate_random_data(query_elements);
    std::vector<float> output_data(query_elements, 0.0f);
    
    print_tensor_sample("Input Query (new token)", query_data);

    attention_hpc::TensorShape query_shape(
        {params.batch_size, params.sequence_length, params.num_heads, params.head_dimension},
        attention_hpc::DataType::FLOAT32
    );
    attention_hpc::TensorShape output_shape = query_shape;

    // 6. Run PagedAttention forward pass for the new token using forward_batch
    std::vector<size_t> sequence_ids = {sequence_id};
    std::vector<attention_hpc::TensorShape> query_shapes = {query_shape};
    std::vector<const void*> query_data_ptrs = {query_data.data()};
    std::vector<attention_hpc::TensorShape> output_shapes = {output_shape};
    std::vector<void*> output_data_ptrs = {output_data.data()};

    paged_attention->forward_batch(
        sequence_ids, 
        query_shapes, query_data_ptrs,
        output_shapes, output_data_ptrs
    );

    std::cout << "PagedAttention forward pass completed for new token." << std::endl;
    print_tensor_sample("Output (new token)", output_data);

    // 7. Cleanup
    paged_attention->destroy_sequence(sequence_id);
    std::cout << "Destroyed sequence with ID: " << sequence_id << std::endl;
}

int main() {
    try {
        demonstrate_flash_attention();
        demonstrate_paged_attention();
    } catch (const attention_hpc::AttentionException& e) {
        std::cerr << "\nCaught an AttentionHPC exception:\n"
                  << "  Error Code: " << static_cast<int>(e.get_error_code()) << "\n"
                  << "  Message: " << e.what() << "\n"
                  << "  Details: " << e.get_detailed_message() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nCaught a standard exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nCaught an unknown exception." << std::endl;
        return 1;
    }

    std::cout << "\nBasic usage example completed successfully." << std::endl;
    return 0;
}
```
