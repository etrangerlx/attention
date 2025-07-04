#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

#include "algorithms/flash_attention.hpp"
#include "api/tensor.hpp"
#include "api/backend_factory.hpp"

namespace attention_hpc {
namespace tests {

// Helper function to generate random float data
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

// Reference implementation of attention computation for validation
void compute_reference_attention(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::vector<float>& output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale,
    bool causal_mask) {
    
    // Compute attention for each batch and head
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            // Calculate base offsets
            size_t qkv_offset = (b * num_heads + h) * seq_len * head_dim;
            
            // For each query position
            for (size_t q_idx = 0; q_idx < seq_len; ++q_idx) {
                // Compute attention scores for this query against all keys
                std::vector<float> scores(seq_len);
                std::vector<float> attention_weights(seq_len);
                
                for (size_t k_idx = 0; k_idx < seq_len; ++k_idx) {
                    // Apply causal mask if needed
                    if (causal_mask && q_idx < k_idx) {
                        scores[k_idx] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    
                    // Compute dot product of query and key
                    float dot_product = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot_product += query[qkv_offset + q_idx * head_dim + d] * 
                                       key[qkv_offset + k_idx * head_dim + d];
                    }
                    
                    // Apply scale factor
                    scores[k_idx] = dot_product * scale;
                }
                
                // Compute softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                
                for (size_t k_idx = 0; k_idx < seq_len; ++k_idx) {
                    attention_weights[k_idx] = std::exp(scores[k_idx] - max_score);
                    sum_exp += attention_weights[k_idx];
                }
                
                // Normalize
                for (size_t k_idx = 0; k_idx < seq_len; ++k_idx) {
                    attention_weights[k_idx] /= sum_exp;
                }
                
                // Compute weighted sum of values
                for (size_t d = 0; d < head_dim; ++d) {
                    float weighted_sum = 0.0f;
                    for (size_t k_idx = 0; k_idx < seq_len; ++k_idx) {
                        weighted_sum += attention_weights[k_idx] * value[qkv_offset + k_idx * head_dim + d];
                    }
                    output[qkv_offset + q_idx * head_dim + d] = weighted_sum;
                }
            }
        }
    }
}

class FlashAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default parameters for testing
        params.batch_size = 2;
        params.sequence_length = 16;
        params.num_heads = 4;
        params.head_dimension = 64;
        params.causal_mask = false;
        params.scale_factor = 1.0f / std::sqrt(static_cast<float>(params.head_dimension));
        
        // Create FlashAttention instance
        flash_attention = std::make_unique<FlashAttention>();
        
        // Configure FlashAttention
        flash_attention->configure(params);
    }
    
    void TearDown() override {
        // Clean up resources
    }
    
    // Helper method to check if two float vectors are approximately equal
    bool are_vectors_equal(const std::vector<float>& v1, const std::vector<float>& v2, float tolerance = 1e-5f) {
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
    
    AttentionParams params;
    std::unique_ptr<FlashAttention> flash_attention;
};

// Test FlashAttention forward pass with small inputs
TEST_F(FlashAttentionTest, ForwardPassSmallInput) {
    // Create small tensors for testing
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    std::vector<float> reference_output(total_qkv_size, 0.0f);
    
    // Create tensor shapes
    TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    // Compute reference output
    compute_reference_attention(
        query_data, key_data, value_data, reference_output,
        params.batch_size, params.num_heads, params.sequence_length, params.head_dimension,
        params.scale_factor, params.causal_mask
    );
    
    // Call FlashAttention forward
    flash_attention->forward(
        query_shape, query_data.data(),
        key_shape, key_data.data(),
        value_shape, value_data.data(),
        output_shape, output_data.data()
    );
    
    // Verify results
    EXPECT_TRUE(are_vectors_equal(output_data, reference_output, 1e-4f));
}

// Test FlashAttention with causal mask
TEST_F(FlashAttentionTest, CausalMaskTest) {
    // Enable causal masking
    params.causal_mask = true;
    flash_attention->configure(params);
    
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    std::vector<float> reference_output(total_qkv_size, 0.0f);
    
    // Create tensor shapes
    TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    // Compute reference output with causal masking
    compute_reference_attention(
        query_data, key_data, value_data, reference_output,
        params.batch_size, params.num_heads, params.sequence_length, params.head_dimension,
        params.scale_factor, true
    );
    
    // Call FlashAttention forward
    flash_attention->forward(
        query_shape, query_data.data(),
        key_shape, key_data.data(),
        value_shape, value_data.data(),
        output_shape, output_data.data()
    );
    
    // Verify results
    EXPECT_TRUE(are_vectors_equal(output_data, reference_output, 1e-4f));
}

// Test different sequence lengths
TEST_F(FlashAttentionTest, DifferentSequenceLengths) {
    std::vector<size_t> seq_lengths = {1, 4, 32, 128};
    
    for (size_t seq_len : seq_lengths) {
        // Update parameters
        params.sequence_length = seq_len;
        flash_attention->configure(params);
        
        size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
        
        // Generate random data
        std::vector<float> query_data = generate_random_data(total_qkv_size);
        std::vector<float> key_data = generate_random_data(total_qkv_size);
        std::vector<float> value_data = generate_random_data(total_qkv_size);
        std::vector<float> output_data(total_qkv_size, 0.0f);
        std::vector<float> reference_output(total_qkv_size, 0.0f);
        
        // Create tensor shapes
        TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        
        // Compute reference output
        compute_reference_attention(
            query_data, key_data, value_data, reference_output,
            params.batch_size, params.num_heads, params.sequence_length, params.head_dimension,
            params.scale_factor, params.causal_mask
        );
        
        // Call FlashAttention forward
        flash_attention->forward(
            query_shape, query_data.data(),
            key_shape, key_data.data(),
            value_shape, value_data.data(),
            output_shape, output_data.data()
        );
        
        // Verify results
        EXPECT_TRUE(are_vectors_equal(output_data, reference_output, 1e-4f)) 
            << "Failed with sequence length: " << seq_len;
    }
}

// Test different head dimensions
TEST_F(FlashAttentionTest, DifferentHeadDimensions) {
    std::vector<size_t> head_dims = {16, 32, 64, 128};
    
    for (size_t dim : head_dims) {
        // Update parameters
        params.head_dimension = dim;
        params.scale_factor = 1.0f / std::sqrt(static_cast<float>(params.head_dimension));
        flash_attention->configure(params);
        
        size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
        
        // Generate random data
        std::vector<float> query_data = generate_random_data(total_qkv_size);
        std::vector<float> key_data = generate_random_data(total_qkv_size);
        std::vector<float> value_data = generate_random_data(total_qkv_size);
        std::vector<float> output_data(total_qkv_size, 0.0f);
        std::vector<float> reference_output(total_qkv_size, 0.0f);
        
        // Create tensor shapes
        TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
        
        // Compute reference output
        compute_reference_attention(
            query_data, key_data, value_data, reference_output,
            params.batch_size, params.num_heads, params.sequence_length, params.head_dimension,
            params.scale_factor, params.causal_mask
        );
        
        // Call FlashAttention forward
        flash_attention->forward(
            query_shape, query_data.data(),
            key_shape, key_data.data(),
            value_shape, value_data.data(),
            output_shape, output_data.data()
        );
        
        // Verify results with tolerance proportional to head dimension
        float tolerance = std::max(1e-4f, 1e-5f * std::sqrt(static_cast<float>(dim)));
        EXPECT_TRUE(are_vectors_equal(output_data, reference_output, tolerance))
            << "Failed with head dimension: " << dim;
    }
}

// Test error handling for invalid inputs
TEST_F(FlashAttentionTest, InvalidInputs) {
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    
    // Test with incorrect query shape (wrong head dimension)
    TensorShape invalid_query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension + 10}, DataType::FLOAT32);
    TensorShape valid_key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape valid_value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape valid_output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    EXPECT_THROW({
        flash_attention->forward(
            invalid_query_shape, query_data.data(),
            valid_key_shape, key_data.data(),
            valid_value_shape, value_data.data(),
            valid_output_shape, output_data.data()
        );
    }, AttentionException);
    
    // Test with mismatched batch size
    TensorShape mismatch_batch_shape({params.batch_size + 1, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    EXPECT_THROW({
        flash_attention->forward(
            mismatch_batch_shape, query_data.data(),
            valid_key_shape, key_data.data(),
            valid_value_shape, value_data.data(),
            valid_output_shape, output_data.data()
        );
    }, AttentionException);
    
    // Test with null pointers
    TensorShape valid_query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    EXPECT_THROW({
        flash_attention->forward(
            valid_query_shape, nullptr,
            valid_key_shape, key_data.data(),
            valid_value_shape, value_data.data(),
            valid_output_shape, output_data.data()
        );
    }, AttentionException);
}

// Test different block sizes
TEST_F(FlashAttentionTest, DifferentBlockSizes) {
    // Create FlashAttention configuration with custom block sizes
    FlashAttentionConfig block_config;
    block_config.block_size_q = 32;
    block_config.block_size_k = 32;
    
    // Configure FlashAttention with custom block sizes
    flash_attention->configure(params);
    flash_attention->configure_flash_attention(block_config);
    
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    std::vector<float> reference_output(total_qkv_size, 0.0f);
    
    // Create tensor shapes
    TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    // Compute reference output
    compute_reference_attention(
        query_data, key_data, value_data, reference_output,
        params.batch_size, params.num_heads, params.sequence_length, params.head_dimension,
        params.scale_factor, params.causal_mask
    );
    
    // Call FlashAttention forward
    flash_attention->forward(
        query_shape, query_data.data(),
        key_shape, key_data.data(),
        value_shape, value_data.data(),
        output_shape, output_data.data()
    );
    
    // Verify results
    EXPECT_TRUE(are_vectors_equal(output_data, reference_output, 1e-4f));
}

// Test backward pass
TEST_F(FlashAttentionTest, BackwardPass) {
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    std::vector<float> grad_output_data = generate_random_data(total_qkv_size, 0.01f, 0.1f);
    std::vector<float> grad_query_data(total_qkv_size, 0.0f);
    std::vector<float> grad_key_data(total_qkv_size, 0.0f);
    std::vector<float> grad_value_data(total_qkv_size, 0.0f);
    
    // Create tensor shapes
    TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape grad_output_shape = output_shape;
    TensorShape grad_query_shape = query_shape;
    TensorShape grad_key_shape = key_shape;
    TensorShape grad_value_shape = value_shape;
    
    // First run forward pass to generate attention weights
    flash_attention->forward(
        query_shape, query_data.data(),
        key_shape, key_data.data(),
        value_shape, value_data.data(),
        output_shape, output_data.data()
    );
    
    // Run backward pass
    flash_attention->backward(
        grad_output_shape, grad_output_data.data(),
        query_shape, query_data.data(),
        key_shape, key_data.data(),
        value_shape, value_data.data(),
        grad_query_shape, grad_query_data.data(),
        grad_key_shape, grad_key_data.data(),
        grad_value_shape, grad_value_data.data()
    );
    
    // We can't easily validate the exact gradient values without a reference implementation,
    // but we can check that gradients are non-zero and have reasonable values
    bool has_query_gradients = false;
    bool has_key_gradients = false;
    bool has_value_gradients = false;
    
    // Check grad_query has non-zero values
    for (float val : grad_query_data) {
        if (std::abs(val) > 1e-6) {
            has_query_gradients = true;
            break;
        }
    }
    
    // Check grad_key has non-zero values
    for (float val : grad_key_data) {
        if (std::abs(val) > 1e-6) {
            has_key_gradients = true;
            break;
        }
    }
    
    // Check grad_value has non-zero values
    for (float val : grad_value_data) {
        if (std::abs(val) > 1e-6) {
            has_value_gradients = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_query_gradients) << "Query gradients are all zero";
    EXPECT_TRUE(has_key_gradients) << "Key gradients are all zero";
    EXPECT_TRUE(has_value_gradients) << "Value gradients are all zero";
}

// Test with an attention mask
TEST_F(FlashAttentionTest, AttentionMaskTest) {
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    size_t mask_size = params.batch_size * params.sequence_length * params.sequence_length;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    
    // Create a simple mask where the first half of positions for each sequence are attended to
    std::vector<float> attention_mask(mask_size, 0.0f);
    for (size_t b = 0; b < params.batch_size; ++b) {
        for (size_t q = 0; q < params.sequence_length; ++q) {
            for (size_t k = 0; k < params.sequence_length / 2; ++k) {
                size_t mask_idx = (b * params.sequence_length + q) * params.sequence_length + k;
                attention_mask[mask_idx] = 1.0f;  // 1 means attend to this position
            }
        }
    }
    
    // Create tensor shapes
    TensorShape query_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape key_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape value_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    TensorShape output_shape({params.batch_size, params.sequence_length, params.num_heads, params.head_dimension}, DataType::FLOAT32);
    
    // Call FlashAttention forward with mask
    flash_attention->forward(
        query_shape, query_data.data(),
        key_shape, key_data.data(),
        value_shape, value_data.data(),
        output_shape, output_data.data(),
        attention_mask.data()
    );
    
    // Create a reference implementation that applies the same mask
    std::vector<float> reference_output(total_qkv_size, 0.0f);
    
    // Compute reference attention with custom mask
    for (size_t b = 0; b < params.batch_size; ++b) {
        for (size_t h = 0; h < params.num_heads; ++h) {
            // Calculate base offsets
            size_t qkv_offset = (b * params.num_heads + h) * params.sequence_length * params.head_dimension;
            
            // For each query position
            for (size_t q_idx = 0; q_idx < params.sequence_length; ++q_idx) {
                // Compute attention scores for this query against all keys
                std::vector<float> scores(params.sequence_length);
                std::vector<float> attention_weights(params.sequence_length);
                
                for (size_t k_idx = 0; k_idx < params.sequence_length; ++k_idx) {
                    // Apply mask
                    size_t mask_idx = (b * params.sequence_length + q_idx) * params.sequence_length + k_idx;
                    if (attention_mask[mask_idx] == 0.0f) {
                        scores[k_idx] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    
                    // Compute dot product of query and key
                    float dot_product = 0.0f;
                    for (size_t d = 0; d < params.head_dimension; ++d) {
                        dot_product += query_data[qkv_offset + q_idx * params.head_dimension + d] * 
                                       key_data[qkv_offset + k_idx * params.head_dimension + d];
                    }
                    
                    // Apply scale factor
                    scores[k_idx] = dot_product * params.scale_factor;
                }
                
                // Compute softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                
                for (size_t k_idx = 0; k_idx < params.sequence_length; ++k_idx) {
                    attention_weights[k_idx] = std::exp(scores[k_idx] - max_score);
                    sum_exp += attention_weights[k_idx];
                }
                
                // Normalize
                for (size_t k_idx = 0; k_idx < params.sequence_length; ++k_idx) {
                    attention_weights[k_idx] /= sum_exp;
                }
                
                // Compute weighted sum of values
                for (size_t d = 0; d < params.head_dimension; ++d) {
                    float weighted_sum = 0.0f;
                    for (size_t k_idx = 0; k_idx < params.sequence_length; ++k_idx) {
                        weighted_sum += attention_weights[k_idx] * 
                                        value_data[qkv_offset + k_idx * params.head_dimension + d];
                    }
                    reference_output[qkv_offset + q_idx * params.head_dimension + d] = weighted_sum;
                }
            }
        }
    }
    
    // Verify results
    EXPECT_TRUE(are_vectors_equal(output_data, reference_output, 1e-4f));
}

// Test edge case with sequence length of 1
TEST_F(FlashAttentionTest, EdgeCaseSequenceLength1) {
    // Update parameters for sequence length 1
    params.sequence_length = 1;
    flash_attention->configure(params);
    
    size_t total_qkv_size = params.batch_size * params.num_heads * params.sequence_length * params.head_dimension;
    
    // Generate random data
    std::vector<float> query_data = generate_random_data(total_qkv_size);
    std::vector<float> key_data = generate_random_data(total_qkv_size);
    std::vector<float> value_data = generate_random_data(total_qkv_size);
    std::vector<float> output_data(total_qkv_size, 0.0f);
    std::vector<float> reference_output(total_qkv_size, 0.0f);
    
    // Create
