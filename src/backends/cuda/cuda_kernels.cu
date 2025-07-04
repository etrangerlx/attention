#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <mma.h>
#include <cooperative_groups.h>

namespace cooperative_groups = cg;
using namespace nvcuda;

// CUDA kernel constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int SHARED_MEMORY_BANK_SIZE = 32;

// Shared memory bank conflict avoidance
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// CUDA error checking macro
#define CUDA_KERNEL_CHECK() do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error)); \
    } \
} while(0)

// Utility functions for data type conversion
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// Warp-level reduction operations
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction operations
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// Safe softmax computation to avoid overflow
__device__ __forceinline__ void safe_softmax(float* scores, int length) {
    // Find maximum value
    float max_val = -INFINITY;
    for (int i = 0; i < length; ++i) {
        max_val = fmaxf(max_val, scores[i]);
    }
    
    // Compute exponentials and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < length; ++i) {
        scores[i] = expf(scores[i] - max_val);
        sum_exp += scores[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < length; ++i) {
        scores[i] *= inv_sum;
    }
}

// FlashAttention CUDA kernel for block-wise computation
__global__ void flash_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    float* __restrict__ attention_scores,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal_mask,
    int block_size_q,
    int block_size_k) {
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    const int q_start = q_block_idx * block_size_q;
    const int q_end = min(q_start + block_size_q, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    // Shared memory for storing query, key, value blocks
    extern __shared__ float shmem[];
    float* s_query = shmem;
    float* s_key = s_query + block_size_q * head_dim;
    float* s_value = s_key + block_size_k * head_dim;
    float* s_scores = s_value + block_size_k * head_dim;
    float* s_output = s_scores + block_size_q * block_size_k;
    
    // Base pointers for current batch and head
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* q_base = query + head_offset;
    const float* k_base = key + head_offset;
    const float* v_base = value + head_offset;
    float* o_base = output + head_offset;
    
    // Load query block into shared memory
    for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
        int q_idx = i / head_dim;
        int d_idx = i % head_dim;
        if (q_start + q_idx < seq_len) {
            s_query[q_idx * head_dim + d_idx] = q_base[(q_start + q_idx) * head_dim + d_idx];
        }
    }
    __syncthreads();
    
    // Initialize output accumulator
    for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
        s_output[i] = 0.0f;
    }
    
    // Online softmax state for each query
    float max_vals[32]; // Assuming max block_size_q <= 32 * num_warps
    float sum_exp[32];
    
    if (tid < q_size) {
        max_vals[tid] = -INFINITY;
        sum_exp[tid] = 0.0f;
    }
    
    // Process key-value blocks
    for (int k_block_start = 0; k_block_start < seq_len; k_block_start += block_size_k) {
        const int k_end = min(k_block_start + block_size_k, seq_len);
        const int k_size = k_end - k_block_start;
        
        if (k_size <= 0) break;
        
        // Load key and value blocks into shared memory
        for (int i = tid; i < k_size * head_dim; i += blockDim.x) {
            int k_idx = i / head_dim;
            int d_idx = i % head_dim;
            if (k_block_start + k_idx < seq_len) {
                s_key[k_idx * head_dim + d_idx] = k_base[(k_block_start + k_idx) * head_dim + d_idx];
                s_value[k_idx * head_dim + d_idx] = v_base[(k_block_start + k_idx) * head_dim + d_idx];
            }
        }
        __syncthreads();
        
        // Compute attention scores: Q * K^T
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            for (int k_idx = tid; k_idx < k_size; k_idx += blockDim.x) {
                float score = 0.0f;
                
                // Vectorized dot product
                for (int d = 0; d < head_dim; d += 4) {
                    if (d + 3 < head_dim) {
                        float4 q_vec = FLOAT4(s_query[q_idx * head_dim + d]);
                        float4 k_vec = FLOAT4(s_key[k_idx * head_dim + d]);
                        score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + 
                                q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                    } else {
                        for (int dd = d; dd < head_dim; ++dd) {
                            score += s_query[q_idx * head_dim + dd] * s_key[k_idx * head_dim + dd];
                        }
                    }
                }
                
                score *= scale;
                
                // Apply causal mask
                if (causal_mask && (q_start + q_idx) < (k_block_start + k_idx)) {
                    score = -INFINITY;
                }
                
                s_scores[q_idx * block_size_k + k_idx] = score;
            }
        }
        __syncthreads();
        
        // Online softmax update
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            if (tid == 0) {
                // Find maximum in current block
                float block_max = -INFINITY;
                for (int k_idx = 0; k_idx < k_size; ++k_idx) {
                    block_max = fmaxf(block_max, s_scores[q_idx * block_size_k + k_idx]);
                }
                
                // Update global maximum
                float new_max = fmaxf(max_vals[q_idx], block_max);
                float old_scale = expf(max_vals[q_idx] - new_max);
                float new_scale = expf(block_max - new_max);
                
                // Update sum of exponentials
                float block_sum = 0.0f;
                for (int k_idx = 0; k_idx < k_size; ++k_idx) {
                    float exp_score = expf(s_scores[q_idx * block_size_k + k_idx] - new_max);
                    s_scores[q_idx * block_size_k + k_idx] = exp_score;
                    block_sum += exp_score;
                }
                
                sum_exp[q_idx] = sum_exp[q_idx] * old_scale + block_sum * new_scale;
                max_vals[q_idx] = new_max;
            }
        }
        __syncthreads();
        
        // Compute weighted values and accumulate
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float weighted_sum = 0.0f;
                for (int k_idx = 0; k_idx < k_size; ++k_idx) {
                    weighted_sum += s_scores[q_idx * block_size_k + k_idx] * s_value[k_idx * head_dim + d];
                }
                atomicAdd(&s_output[q_idx * head_dim + d], weighted_sum);
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    for (int q_idx = 0; q_idx < q_size; ++q_idx) {
        float normalizer = 1.0f / sum_exp[q_idx];
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float final_output = s_output[q_idx * head_dim + d] * normalizer;
            o_base[(q_start + q_idx) * head_dim + d] = final_output;
        }
    }
}

// FlashAttention backward pass kernel
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    float* __restrict__ grad_value,
    const float* __restrict__ attention_weights,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal_mask) {
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const int token_offset = head_offset + seq_idx * head_dim;
    
    // Shared memory for gradients computation
    extern __shared__ float shmem[];
    float* s_grad_q = shmem;
    float* s_grad_k = s_grad_q + head_dim;
    float* s_grad_v = s_grad_k + head_dim;
    
    // Initialize shared memory
    if (tid < head_dim) {
        s_grad_q[tid] = 0.0f;
        s_grad_k[tid] = 0.0f;
        s_grad_v[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute gradients for query
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        if (causal_mask && seq_idx < k_idx) continue;
        
        float attention_weight = attention_weights[(batch_idx * num_heads + head_idx) * seq_len * seq_len + seq_idx * seq_len + k_idx];
        
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float grad_contrib = grad_output[token_offset + d] * attention_weight * 
                               key[head_offset + k_idx * head_dim + d] * scale;
            atomicAdd(&s_grad_q[d], grad_contrib);
        }
    }
    
    // Compute gradients for key
    for (int q_idx = 0; q_idx < seq_len; ++q_idx) {
        if (causal_mask && q_idx < seq_idx) continue;
        
        float attention_weight = attention_weights[(batch_idx * num_heads + head_idx) * seq_len * seq_len + q_idx * seq_len + seq_idx];
        
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float grad_contrib = grad_output[head_offset + q_idx * head_dim + d] * attention_weight * 
                               query[head_offset + q_idx * head_dim + d] * scale;
            atomicAdd(&s_grad_k[d], grad_contrib);
        }
    }
    
    // Compute gradients for value
    for (int q_idx = 0; q_idx < seq_len; ++q_idx) {
        if (causal_mask && q_idx < seq_idx) continue;
        
        float attention_weight = attention_weights[(batch_idx * num_heads + head_idx) * seq_len * seq_len + q_idx * seq_len + seq_idx];
        
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float grad_contrib = grad_output[head_offset + q_idx * head_dim + d] * attention_weight;
            atomicAdd(&s_grad_v[d], grad_contrib);
        }
    }
    
    __syncthreads();
    
    // Write gradients to global memory
    if (tid < head_dim) {
        grad_query[token_offset + tid] = s_grad_q[tid];
        grad_key[token_offset + tid] = s_grad_k[tid];
        grad_value[token_offset + tid] = s_grad_v[tid];
    }
}

// PagedAttention kernel for efficient long sequence processing
__global__ void paged_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    const int* __restrict__ page_table,
    const int* __restrict__ sequence_lengths,
    int batch_size,
    int num_heads,
    int head_dim,
    int page_size,
    int max_pages_per_sequence,
    float scale) {
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    const int seq_len = sequence_lengths[batch_idx];
    if (query_idx >= seq_len) return;
    
    // Shared memory for attention computation
    extern __shared__ float shmem[];
    float* s_attention_scores = shmem;
    float* s_max_vals = s_attention_scores + page_size;
    float* s_sum_exp = s_max_vals + 1;
    float* s_output = s_sum_exp + 1;
    
    const int query_offset = (batch_idx * num_heads + head_idx) * head_dim;
    const float* q_ptr = query + query_offset;
    
    // Initialize output accumulator in shared memory
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_output[d] = 0.0f;
    }
    
    // Initialize online softmax state
    if (tid == 0) {
        s_max_vals[0] = -INFINITY;
        s_sum_exp[0] = 0.0f;
    }
    __syncthreads();
    
    // Process each page
    int num_pages = (seq_len + page_size - 1) / page_size;
    for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
        int page_id = page_table[batch_idx * max_pages_per_sequence + page_idx];
        if (page_id < 0) break;
        
        int page_start = page_idx * page_size;
        int page_end = min(page_start + page_size, seq_len);
        int page_length = page_end - page_start;
        
        // Skip if query position is beyond current page for causal attention
        if (query_idx < page_start) continue;
        
        // Compute attention scores for this page
        for (int k_idx = tid; k_idx < page_length; k_idx += blockDim.x) {
            int global_k_idx = page_start + k_idx;
            
            // Skip future positions for causal attention
            if (global_k_idx > query_idx) {
                s_attention_scores[k_idx] = -INFINITY;
                continue;
            }
            
            const float* k_ptr = key_cache + (page_id * page_size * num_heads + 
                                            k_idx * num_heads + head_idx) * head_dim;
            
            float score = 0.0f;
            // Vectorized dot product
            for (int d = 0; d < head_dim; d += 4) {
                if (d + 3 < head_dim) {
                    float4 q_vec = FLOAT4(q_ptr[d]);
                    float4 k_vec = FLOAT4(k_ptr[d]);
                    score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + 
                            q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                } else {
                    for (int dd = d; dd < head_dim; ++dd) {
                        score += q_ptr[dd] * k_ptr[dd];
                    }
                }
            }
            
            s_attention_scores[k_idx] = score * scale;
        }
        __syncthreads();
        
        // Online softmax update
        if (tid == 0) {
            // Find maximum in current page
            float page_max = -INFINITY;
            for (int k_idx = 0; k_idx < page_length; ++k_idx) {
                page_max = fmaxf(page_max, s_attention_scores[k_idx]);
            }
            
            // Update global maximum and scaling factors
            float new_max = fmaxf(s_max_vals[0], page_max);
            float old_scale = expf(s_max_vals[0] - new_max);
            float new_scale = expf(page_max - new_max);
            
            // Compute page sum with new scaling
            float page_sum = 0.0f;
            for (int k_idx = 0; k_idx < page_length; ++k_idx) {
                float exp_score = expf(s_attention_scores[k_idx] - new_max);
                s_attention_scores[k_idx] = exp_score;
                page_sum += exp_score;
            }
            
            // Update online softmax state
            s_sum_exp[0] = s_sum_exp[0] * old_scale + page_sum * new_scale;
            s_max_vals[0] = new_max;
            
            // Scale previous output accumulator
            for (int d = 0; d < head_dim; ++d) {
                s_output[d] *= old_scale;
            }
        }
        __syncthreads();
        
        // Accumulate weighted values
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float weighted_sum = 0.0f;
            for (int k_idx = 0; k_idx < page_length; ++k_idx) {
                int global_k_idx = page_start + k_idx;
                if (global_k_idx <= query_idx) {
                    const float* v_ptr = value_cache + (page_id * page_size * num_heads + 
                                                      k_idx * num_heads + head_idx) * head_dim;
                    weighted_sum += s_attention_scores[k_idx] * v_ptr[d];
                }
            }
            s_output[d] += weighted_sum;
        }
        __syncthreads();
    }
    
    // Final normalization and output
    const int output_offset = (batch_idx * num_heads + head_idx) * head_dim;
    float normalizer = 1.0f / s_sum_exp[0];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[output_offset + d] = s_output[d] * normalizer;
    }
}

// PagedAttention key-value cache update kernel
__global__ void update_kv_cache_kernel(
    const float* __restrict__ key_input,
    const float* __restrict__ value_input,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const int* __restrict__ page_table,
    const int* __restrict__ sequence_lengths,
    const int* __restrict__ slot_mapping,
    int batch_size,
    int num_heads,
    int head_dim,
    int page_size,
    int max_pages_per_sequence) {
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    const int seq_len = sequence_lengths[batch_idx];
    if (token_idx >= seq_len) return;
    
    // Get slot mapping for this token
    int slot_idx = slot_mapping[batch_idx * seq_len + token_idx];
    if (slot_idx < 0) return;
    
    // Calculate page and position within page
    int page_idx = slot_idx / page_size;
    int pos_in_page = slot_idx % page_size;
    
    // Get physical page ID
    int page_id = page_table[batch_idx * max_pages_per_sequence + page_idx];
    if (page_id < 0) return;
    
    // Calculate cache offsets
    const int input_offset = (batch_idx * seq_len + token_idx) * num_heads * head_dim + head_idx * head_dim;
    const int cache_offset = (page_id * page_size + pos_in_page) * num_heads * head_dim + head_idx * head_dim;
    
    // Copy key and value to cache
    for (int d = tid; d < head_dim; d += blockDim.x) {
        key_cache[cache_offset + d] = key_input[input_offset + d];
        value_cache[cache_offset + d] = value_input[input_offset + d];
    }
}

// Online softmax kernel for attention computation
__global__ void online_softmax_kernel(
    const float* __restrict__ attention_scores,
    float* __restrict__ attention_weights,
    float* __restrict__ max_vals,
    float* __restrict__ sum_exp,
    int batch_size,
    int num_heads,
    int seq_len) {
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len;
    const int score_offset = head_offset * seq_len + seq_idx * seq_len;
    
    // Shared memory for warp-level reductions
    __shared__ float s_max_vals[WARP_SIZE];
    __shared__ float s_sum_exp[WARP_SIZE];
    
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Find maximum value across all positions for numerical stability
    float thread_max = -INFINITY;
    for (int pos = tid; pos <= seq_idx; pos += blockDim.x) {
        thread_max = fmaxf(thread_max, attention_scores[score_offset + pos]);
    }
    
    // Warp-level maximum reduction
    thread_max = warp_reduce_max(thread_max);
    if (lane_id == 0) {
        s_max_vals[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Block-level maximum
    if (tid < blockDim.x / WARP_SIZE) {
        thread_max = s_max_vals[tid];
    } else {
        thread_max = -INFINITY;
    }
    if (warp_id == 0) {
        thread_max = warp_reduce_max(thread_max);
    }
    
    // Broadcast maximum to all threads
    if (tid == 0) {
        s_max_vals[0] = thread_max;
    }
    __syncthreads();
    float global_max = s_max_vals[0];
    
    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int pos = tid; pos <= seq_idx; pos += blockDim.x) {
        float exp_val = expf(attention_scores[score_offset + pos] - global_max);
        attention_weights[score_offset + pos] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-level sum reduction
    thread_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        s_sum_exp[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Block-level sum
    if (tid < blockDim.x / WARP_SIZE) {
        thread_sum = s_sum_exp[tid];
    } else {
        thread_sum = 0.0f;
    }
    if (warp_id == 0) {
        thread_sum = warp_reduce_sum(thread_sum);
    }
    
    // Broadcast sum to all threads
    if (tid == 0) {
        s_sum_exp[0] = thread_sum;
    }
    __syncthreads();
    float global_sum = s_sum_exp[0];
    
    // Normalize attention weights
    float inv_sum = 1.0f / global_sum;
    for (int pos = tid; pos <= seq_idx; pos += blockDim.x) {
        attention_weights[score_offset + pos] *= inv_sum;
    }
    
    // Store statistics for online computation
    if (tid == 0) {
        max_vals[head_offset + seq_idx] = global_max;
        sum_exp[head_offset + seq_idx] = global_sum;
    }
}

// Transpose kernel for memory layout optimization
__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool forward) {
    
    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= num_heads) return;
    
    if (forward) {
        // [B, S, H, D] -> [B, H, S, D]
        const int input_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
        const int output_offset = ((batch_idx * num_heads + head_idx) * seq_len +
