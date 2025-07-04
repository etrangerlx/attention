// FlashAttention OpenCL Kernel Implementation
// Optimized for cross-platform compatibility and performance

// Constants and macros
#define WARP_SIZE 32
#define MAX_LOCAL_SIZE 1024
#define FLOAT4_ALIGNMENT 16
#define BANK_CONFLICT_OFFSET 1

// Utility macros for memory access
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4_LOAD(ptr) vload4(0, (ptr))
#define FLOAT4_STORE(data, ptr) vstore4((data), 0, (ptr))

// Work-group level barrier synchronization
#define WORKGROUP_BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define GLOBAL_BARRIER() barrier(CLK_GLOBAL_MEM_FENCE)

// Safe softmax computation to avoid overflow
void safe_softmax_local(__local float* scores, int length, int tid, int local_size) {
    // Find maximum value using work-group reduction
    __local float max_vals[MAX_LOCAL_SIZE / WARP_SIZE];
    __local float sum_vals[MAX_LOCAL_SIZE / WARP_SIZE];
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (local_size + WARP_SIZE - 1) / WARP_SIZE;
    
    // Find local maximum
    float thread_max = -INFINITY;
    for (int i = tid; i < length; i += local_size) {
        thread_max = fmax(thread_max, scores[i]);
    }
    
    // Warp-level reduction for maximum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = work_group_broadcast(thread_max, tid + offset);
        if (tid + offset < local_size) {
            thread_max = fmax(thread_max, other_max);
        }
    }
    
    // Store warp maximums
    if (lane_id == 0 && warp_id < num_warps) {
        max_vals[warp_id] = thread_max;
    }
    WORKGROUP_BARRIER();
    
    // Find global maximum across warps
    if (tid < num_warps) {
        thread_max = max_vals[tid];
    } else {
        thread_max = -INFINITY;
    }
    
    for (int offset = num_warps / 2; offset > 0; offset /= 2) {
        if (tid < offset && tid + offset < num_warps) {
            thread_max = fmax(thread_max, max_vals[tid + offset]);
        }
    }
    
    if (tid == 0) {
        max_vals[0] = thread_max;
    }
    WORKGROUP_BARRIER();
    
    float global_max = max_vals[0];
    
    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < length; i += local_size) {
        float exp_val = exp(scores[i] - global_max);
        scores[i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_sum = work_group_broadcast(thread_sum, tid + offset);
        if (tid + offset < local_size) {
            thread_sum += other_sum;
        }
    }
    
    // Store warp sums
    if (lane_id == 0 && warp_id < num_warps) {
        sum_vals[warp_id] = thread_sum;
    }
    WORKGROUP_BARRIER();
    
    // Find global sum across warps
    if (tid < num_warps) {
        thread_sum = sum_vals[tid];
    } else {
        thread_sum = 0.0f;
    }
    
    for (int offset = num_warps / 2; offset > 0; offset /= 2) {
        if (tid < offset && tid + offset < num_warps) {
            thread_sum += sum_vals[tid + offset];
        }
    }
    
    if (tid == 0) {
        sum_vals[0] = thread_sum;
    }
    WORKGROUP_BARRIER();
    
    float global_sum = sum_vals[0];
    
    // Normalize
    float inv_sum = 1.0f / global_sum;
    for (int i = tid; i < length; i += local_size) {
        scores[i] *= inv_sum;
    }
    WORKGROUP_BARRIER();
}

// Optimized matrix multiplication using local memory
void gemm_local_optimized(
    __local float* A, __local float* B, __local float* C,
    int M, int N, int K, int tid, int local_size,
    bool transpose_A, bool transpose_B) {
    
    // Tiled matrix multiplication with local memory optimization
    const int TILE_SIZE = 16;
    
    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        int k_end = min(tile_k + TILE_SIZE, K);
        
        for (int i = tid; i < M * (k_end - tile_k); i += local_size) {
            int row = i / (k_end - tile_k);
            int col = i % (k_end - tile_k);
            int k_idx = tile_k + col;
            
            if (row < M && k_idx < K) {
                // Load A tile with proper indexing
                int a_idx = transpose_A ? OFFSET(k_idx, row, M) : OFFSET(row, k_idx, K);
                // Use local memory with bank conflict avoidance
                A[row * TILE_SIZE + col + BANK_CONFLICT_OFFSET] = A[a_idx];
            }
        }
        
        for (int i = tid; i < (k_end - tile_k) * N; i += local_size) {
            int row = i / N;
            int col = i % N;
            int k_idx = tile_k + row;
            
            if (k_idx < K && col < N) {
                // Load B tile with proper indexing
                int b_idx = transpose_B ? OFFSET(col, k_idx, K) : OFFSET(k_idx, col, N);
                // Use local memory with bank conflict avoidance
                B[row * N + col + BANK_CONFLICT_OFFSET] = B[b_idx];
            }
        }
        
        WORKGROUP_BARRIER();
        
        // Compute partial products
        for (int i = tid; i < M * N; i += local_size) {
            int row = i / N;
            int col = i % N;
            
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < k_end - tile_k; ++k) {
                    sum += A[row * TILE_SIZE + k + BANK_CONFLICT_OFFSET] * 
                           B[k * N + col + BANK_CONFLICT_OFFSET];
                }
                
                if (tile_k == 0) {
                    C[OFFSET(row, col, N)] = sum;
                } else {
                    C[OFFSET(row, col, N)] += sum;
                }
            }
        }
        
        WORKGROUP_BARRIER();
    }
}

// Main FlashAttention kernel with block-wise computation
__kernel void flash_attention_kernel(
    __global const float* restrict query,     // [batch_size, num_heads, seq_len, head_dim]
    __global const float* restrict key,       // [batch_size, num_heads, seq_len, head_dim]
    __global const float* restrict value,     // [batch_size, num_heads, seq_len, head_dim]
    __global float* restrict output,          // [batch_size, num_heads, seq_len, head_dim]
    __global float* restrict attention_scores, // Workspace for attention scores
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int causal_mask,
    const int block_size_q,
    const int block_size_k,
    __local float* local_memory) {
    
    const int batch_idx = get_group_id(2);
    const int head_idx = get_group_id(1);
    const int q_block_idx = get_group_id(0);
    
    const int tid = get_local_id(0);
    const int local_size = get_local_size(0);
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    const int q_start = q_block_idx * block_size_q;
    const int q_end = min(q_start + block_size_q, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    // Local memory allocation with proper alignment
    __local float* l_query = local_memory;
    __local float* l_key = l_query + block_size_q * head_dim + BANK_CONFLICT_OFFSET;
    __local float* l_value = l_key + block_size_k * head_dim + BANK_CONFLICT_OFFSET;
    __local float* l_scores = l_value + block_size_k * head_dim + BANK_CONFLICT_OFFSET;
    __local float* l_output = l_scores + block_size_q * block_size_k + BANK_CONFLICT_OFFSET;
    __local float* l_workspace = l_output + block_size_q * head_dim + BANK_CONFLICT_OFFSET;
    
    // Base pointers for current batch and head
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    __global const float* q_base = query + head_offset;
    __global const float* k_base = key + head_offset;
    __global const float* v_base = value + head_offset;
    __global float* o_base = output + head_offset;
    
    // Load query block into local memory with coalesced access
    for (int i = tid; i < q_size * head_dim; i += local_size) {
        int q_idx = i / head_dim;
        int d_idx = i % head_dim;
        if (q_start + q_idx < seq_len && d_idx < head_dim) {
            l_query[q_idx * head_dim + d_idx] = q_base[(q_start + q_idx) * head_dim + d_idx];
        } else {
            l_query[q_idx * head_dim + d_idx] = 0.0f;
        }
    }
    WORKGROUP_BARRIER();
    
    // Initialize output accumulator
    for (int i = tid; i < q_size * head_dim; i += local_size) {
        l_output[i] = 0.0f;
    }
    WORKGROUP_BARRIER();
    
    // Online softmax state for each query token
    __local float max_vals[256]; // Assuming max block_size_q <= 256
    __local float sum_exp[256];
    
    for (int i = tid; i < q_size; i += local_size) {
        max_vals[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }
    WORKGROUP_BARRIER();
    
    // Process key-value blocks
    for (int k_block_start = 0; k_block_start < seq_len; k_block_start += block_size_k) {
        const int k_end = min(k_block_start + block_size_k, seq_len);
        const int k_size = k_end - k_block_start;
        
        if (k_size <= 0) break;
        
        // Load key and value blocks into local memory
        for (int i = tid; i < k_size * head_dim; i += local_size) {
            int k_idx = i / head_dim;
            int d_idx = i % head_dim;
            if (k_block_start + k_idx < seq_len && d_idx < head_dim) {
                l_key[k_idx * head_dim + d_idx] = k_base[(k_block_start + k_idx) * head_dim + d_idx];
                l_value[k_idx * head_dim + d_idx] = v_base[(k_block_start + k_idx) * head_dim + d_idx];
            } else {
                l_key[k_idx * head_dim + d_idx] = 0.0f;
                l_value[k_idx * head_dim + d_idx] = 0.0f;
            }
        }
        WORKGROUP_BARRIER();
        
        // Compute attention scores: Q * K^T with vectorized operations
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            for (int k_idx = tid; k_idx < k_size; k_idx += local_size) {
                float score = 0.0f;
                
                // Vectorized dot product using float4 when possible
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float4 q_vec = (float4)(
                        l_query[q_idx * head_dim + d],
                        l_query[q_idx * head_dim + d + 1],
                        l_query[q_idx * head_dim + d + 2],
                        l_query[q_idx * head_dim + d + 3]
                    );
                    float4 k_vec = (float4)(
                        l_key[k_idx * head_dim + d],
                        l_key[k_idx * head_dim + d + 1],
                        l_key[k_idx * head_dim + d + 2],
                        l_key[k_idx * head_dim + d + 3]
                    );
                    score += dot(q_vec, k_vec);
                }
                
                // Handle remaining elements
                for (; d < head_dim; ++d) {
                    score += l_query[q_idx * head_dim + d] * l_key[k_idx * head_dim + d];
                }
                
                score *= scale;
                
                // Apply causal mask
                if (causal_mask && (q_start + q_idx) < (k_block_start + k_idx)) {
                    score = -INFINITY;
                }
                
                l_scores[q_idx * block_size_k + k_idx] = score;
            }
        }
        WORKGROUP_BARRIER();
        
        // Online softmax update for each query token
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            // Single thread per query token to avoid race conditions
            if (tid == q_idx % local_size) {
                // Find maximum in current block
                float block_max = -INFINITY;
                for (int k_idx = 0; k_idx < k_size; ++k_idx) {
                    block_max = fmax(block_max, l_scores[q_idx * block_size_k + k_idx]);
                }
                
                // Update global maximum
                float old_max = max_vals[q_idx];
                float new_max = fmax(old_max, block_max);
                float old_scale = exp(old_max - new_max);
                
                // Compute block sum with new scaling
                float block_sum = 0.0f;
                for (int k_idx = 0; k_idx < k_size; ++k_idx) {
                    float exp_score = exp(l_scores[q_idx * block_size_k + k_idx] - new_max);
                    l_scores[q_idx * block_size_k + k_idx] = exp_score;
                    block_sum += exp_score;
                }
                
                // Update online softmax state
                sum_exp[q_idx] = sum_exp[q_idx] * old_scale + block_sum;
                max_vals[q_idx] = new_max;
                
                // Scale previous output accumulator
                for (int d = 0; d < head_dim; ++d) {
                    l_output[q_idx * head_dim + d] *= old_scale;
                }
            }
        }
        WORKGROUP_BARRIER();
        
        // Compute weighted values and accumulate
        for (int q_idx = 0; q_idx < q_size; ++q_idx) {
            for (int d = tid; d < head_dim; d += local_size) {
                float weighted_sum = 0.0f;
                
                // Vectorized computation when possible
                int k = 0;
                for (; k + 3 < k_size; k += 4) {
                    float4 weights = (float4)(
                        l_scores[q_idx * block_size_k + k],
                        l_scores[q_idx * block_size_k + k + 1],
                        l_scores[q_idx * block_size_k + k + 2],
                        l_scores[q_idx * block_size_k + k + 3]
                    );
                    float4 values = (float4)(
                        l_value[k * head_dim + d],
                        l_value[(k + 1) * head_dim + d],
                        l_value[(k + 2) * head_dim + d],
                        l_value[(k + 3) * head_dim + d]
                    );
                    weighted_sum += dot(weights, values);
                }
                
                // Handle remaining elements
                for (; k < k_size; ++k) {
                    weighted_sum += l_scores[q_idx * block_size_k + k] * l_value[k * head_dim + d];
                }
                
                l_output[q_idx * head_dim + d] += weighted_sum;
            }
        }
        WORKGROUP_BARRIER();
    }
    
    // Final normalization and write output
    for (int q_idx = 0; q_idx < q_size; ++q_idx) {
        float normalizer = 1.0f / sum_exp[q_idx];
        for (int d = tid; d < head_dim; d += local_size) {
            if (q_start + q_idx < seq_len && d < head_dim) {
                float final_output = l_output[q_idx * head_dim + d] * normalizer;
                o_base[(q_start + q_idx) * head_dim + d] = final_output;
            }
        }
    }
    WORKGROUP_BARRIER();
}

// FlashAttention backward pass kernel
__kernel void flash_attention_backward_kernel(
    __global const float* restrict grad_output,    // [batch_size, num_heads, seq_len, head_dim]
    __global const float* restrict query,          // [batch_size, num_heads, seq_len, head_dim]
    __global const float* restrict key,            // [batch_size, num_heads, seq_len, head_dim]
    __global const float* restrict value,          // [batch_size, num_heads, seq_len, head_dim]
    __global float* restrict grad_query,           // [batch_size, num_heads, seq_len, head_dim]
    __global float* restrict grad_key,             // [batch_size, num_heads, seq_len, head_dim]
    __global float* restrict grad_value,           // [batch_size, num_heads, seq_len, head_dim]
    __global const float* restrict attention_weights, // [batch_size, num_heads, seq_len, seq_len]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    const int causal_mask,
    __local float* local_memory) {
    
    const int batch_idx = get_group_id(2);
    const int head_idx = get_group_id(1);
    const int seq_idx = get_group_id(0);
    const int tid = get_local_id(0);
    const int local_size = get_local_size(0);
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const int token_offset = head_offset + seq_idx * head_dim;
    const int attention_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len;
    
    // Local memory allocation
    __local float* l_grad_q = local_memory;
    __local float* l_grad_k = l_grad_q + head_dim + BANK_CONFLICT_OFFSET;
    __local float* l_grad_v = l_grad_k + head_dim + BANK_CONFLICT_OFFSET;
    __local float* l_workspace = l_grad_v + head_dim + BANK_CONFLICT_OFFSET;
    
    // Initialize local gradients
    for (int d = tid; d < head_dim; d += local_size) {
        l_grad_q[d] = 0.0f;
        l_grad_k[d] = 0.0f;
        l_grad_v[d] = 0.0f;
    }
    WORKGROUP_BARRIER();
    
    // Compute gradients for query
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        if (causal_mask && seq_idx < k_idx) continue;
        
        float attention_weight = attention_weights[attention_offset + seq_idx * seq_len + k_idx];
        
        for (int d = tid; d < head_dim; d += local_size) {
            float grad_contrib = grad_output[token_offset + d] * attention_weight * 
                               key[head_offset + k_idx * head_dim + d] * scale;
            
            // Use atomic operations to avoid race conditions
            atomic_add_global(&l_grad_q[d], grad_contrib);
        }
    }
    WORKGROUP_BARRIER();
    
    // Compute gradients for key
    for (int q_idx = 0; q_idx < seq_len; ++q_idx) {
        if (causal_mask && q_idx < seq_idx) continue;
        
        float attention_weight = attention_weights[attention_offset + q_idx * seq_len + seq_idx];
        
        for (int d = tid; d < head_dim; d += local_size) {
            float grad_contrib = grad_output[head_offset + q_idx * head_dim + d] * attention_weight * 
                               query[head_offset + q_idx * head_dim + d] * scale;
            
            atomic_add_global(&l_grad_k[d], grad_contrib);
        }
    }
    WORKGROUP_BARRIER();
    
    // Compute gradients for value
    for (int q_idx = 0; q_idx < seq_len; ++q_idx) {
        if (causal_mask && q_idx < seq_idx) continue;
        
        float attention_weight = attention_weights[attention_offset + q_idx * seq_len + seq_idx];
        
        for (int d = tid; d < head_dim; d += local_size) {
            float grad_contrib = grad_output[head_offset + q_idx * head_dim + d] * attention_weight;
            
            atomic_add_global(&l_grad_v[d], grad_contrib);
        }
    }
    WORKGROUP_BARRIER();
    
    // Write gradients to global memory
    for (int d = tid; d < head_dim; d += local_size) {
        grad_query[token_offset + d] = l_grad_q[d];
        grad_key[token_offset + d] = l_grad_k[d];
        grad_value[token_offset + d] = l_grad_v[d];
    }
}

// Online softmax kernel for streaming attention computation
__kernel void online_softmax_kernel(
    __global const float* restrict attention_scores, // [batch_size, num_heads, seq_len, seq_len]
    __global float* restrict attention_weights,       // [batch_size, num_heads, seq_len, seq_len]
    __global float* restrict max_vals,                // [batch_size, num_heads, seq_len]
    __global float* restrict sum_exp,                 // [batch_size, num_heads, seq_len]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    __local float* local_memory) {
    
    const int batch_idx = get_group_id(2);
    const int head_idx = get_group_id(1);
    const int seq_idx = get_group_id(0);
    const int tid = get_local_id(0);
    const int local_size = get_local_size(0);
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    const int head_offset = (batch_idx * num_heads + head_idx) * seq_len;
    const int score_offset = head_offset * seq_len + seq_idx * seq_len;
    const int state_offset = head_offset + seq_idx;
    
    // Local memory for reductions
    __local float* l_max_vals = local_memory;
    __local float* l_sum_vals = l_max_vals + local_size + BANK_CONFLICT_OFFSET;
    
    // Find maximum value for numerical stability
    float thread_max = -INFINITY;
    for (int pos = tid; pos <= seq_idx; pos += local_size) {
        thread_max = fmax(thread_max, attention_scores[score_offset + pos]);
    }
    
    l_max_vals[tid] = thread_max;
    WORKGROUP_BARRIER();
    
    // Work-group level reduction for maximum
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < local_size) {
            l_max_vals[tid] = fmax(l_max_vals[tid], l_max_vals[tid + stride]);
        }
        WORKGROUP_BARRIER();
    }
    
    float global_max = l_max_vals[0];
    
    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int pos = tid; pos <= seq_idx; pos += local_size) {
        float exp_val = exp(attention_scores[score_offset + pos] - global_max);
        attention_weights[score_offset + pos] = exp_val;
        thread_sum += exp_val;
    }
    
    l_sum_vals[tid] = thread_sum;
    WORKGROUP_BARRIER();
    
    // Work-group level reduction for sum
    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < local_size) {
            l_sum_vals[tid] += l_sum_vals[tid + stride];
        }
        WORKGROUP_BARRIER();
    }
    
    float global_sum = l_sum_vals[0];
    
    // Normalize attention weights
    float inv_sum = 1.0f / global_sum;
    for (int pos = tid; pos <= seq_idx; pos += local_size) {
        attention_weights[score_offset + pos] *= inv_sum;
    }
    
    // Store statistics for online computation
    if (tid == 0) {
        max_vals[state_offset] = global_max;
        sum_exp[state_offset] = global_sum;
    }
}

// Transpose kernel for memory layout optimization
__kernel void transpose_kernel(
    __global const float* restrict input,   // Input tensor
    __global float* restrict output,        // Output tensor
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int forward,                      // 1 for forward, 0 for inverse
    __local float* local_memory) {
    
    const int batch_idx = get_group_id(2);
    const int seq_idx = get_group_id(1);
    const int head_idx = get_group_id(0);
    const int tid = get_local_id(0);
    const int local_size = get_local_size(0);
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= num_heads) return;
    
    // Local memory tile for cache-friendly transpose
    const int TILE_SIZE = 16;
    __local float tile[TILE_SIZE * TILE_SIZE];
    
    if (forward) {
        // [B, S, H, D] -> [B, H, S, D]
        const int input_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
        const int output_offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim;
        
        // Copy data in tiles for better cache performance
        for (int d_start = 0; d_start < head_dim; d_start += TILE_SIZE) {
            int d_end = min(d_start + TILE_SIZE, head_dim);
            
            // Load tile into local memory
            for (int d = d_start + tid; d < d_end; d += local_size) {
                if (d < head_dim) {
                    int local_idx = d - d_start;
                    tile[local_idx] = input[input_offset + d];
                }
            }
            WORKGROUP_BARRIER();
            
            // Store tile to output
            for (int d = d_start + tid; d < d_end; d += local_size) {
                if (d < head_dim) {
                    int local_idx = d - d_start;
                    output[output_offset + d] = tile[local_idx];
                }
            }
            WORKGROUP_BARRIER();
        }
    } else {
        // [B, H, S, D] -> [B, S, H, D]
        const int input_offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim;
        const int output_offset = ((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim;
        
        // Copy data in tiles for better cache performance
        for (int d_start = 0; d_start < head_dim; d_start += TILE_SIZE) {
            int d_end = min(d_start + TILE_SIZE, head_dim);
            
            // Load tile into local memory
            for (int d = d_start + tid; d < d_end; d += local_size) {
                if (d < head_dim) {
                    int local_idx = d - d_start;
                    tile[local_idx] = input[input_offset + d];
                }
            }
            WORKGROUP_BARRIER
