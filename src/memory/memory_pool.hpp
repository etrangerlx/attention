#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace attention_hpc {

// Memory types supported by the pool
enum class MemoryType {
    CPU_MEMORY,
    GPU_MEMORY,
    UNIFIED_MEMORY,
    PINNED_MEMORY
};

// Memory alignment requirements
constexpr size_t DEFAULT_ALIGNMENT = 64;
constexpr size_t GPU_MEMORY_ALIGNMENT = 256;
constexpr size_t CACHE_LINE_SIZE = 64;

// Memory allocation statistics
struct MemoryStats {
    size_t total_allocated = 0;
    size_t total_freed = 0;
    size_t current_usage = 0;
    size_t peak_usage = 0;
    size_t num_allocations = 0;
    size_t num_deallocations = 0;
    size_t num_allocation_failures = 0;
    size_t fragmentation_bytes = 0;
    
    void reset() {
        total_allocated = 0;
        total_freed = 0;
        current_usage = 0;
        peak_usage = 0;
        num_allocations = 0;
        num_deallocations = 0;
        num_allocation_failures = 0;
        fragmentation_bytes = 0;
    }
    
    double get_fragmentation_ratio() const {
        return current_usage > 0 ? static_cast<double>(fragmentation_bytes) / current_usage : 0.0;
    }
};

// Memory block descriptor
struct MemoryBlock {
    void* ptr;
    size_t size;
    size_t alignment;
    MemoryType type;
    bool is_free;
    size_t allocation_id;
    
    MemoryBlock() : ptr(nullptr), size(0), alignment(0), type(MemoryType::CPU_MEMORY), 
                    is_free(true), allocation_id(0) {}
    
    MemoryBlock(void* p, size_t s, size_t a, MemoryType t) 
        : ptr(p), size(s), alignment(a), type(t), is_free(false), allocation_id(0) {}
    
    // Calculate aligned size
    size_t aligned_size() const {
        return (size + alignment - 1) & ~(alignment - 1);
    }
    
    // Check if block can satisfy allocation request
    bool can_satisfy(size_t requested_size, size_t requested_alignment) const {
        if (!is_free || size < requested_size) return false;
        
        // Check alignment
        uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(ptr) + requested_alignment - 1) & 
                                ~(requested_alignment - 1);
        size_t alignment_offset = aligned_addr - reinterpret_cast<uintptr_t>(ptr);
        
        return (size - alignment_offset) >= requested_size;
    }
};

// Memory allocator interface
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;
    
    virtual void* allocate(size_t size, size_t alignment, MemoryType type) = 0;
    virtual void deallocate(void* ptr, MemoryType type) = 0;
    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;
    virtual bool supports_type(MemoryType type) const = 0;
};

// Default system memory allocator
class SystemMemoryAllocator : public MemoryAllocator {
public:
    void* allocate(size_t size, size_t alignment, MemoryType type) override;
    void deallocate(void* ptr, MemoryType type) override;
    size_t get_total_memory() const override;
    size_t get_free_memory() const override;
    bool supports_type(MemoryType type) const override;
    
private:
    void* allocate_aligned(size_t size, size_t alignment);
    void deallocate_aligned(void* ptr);
};

// Memory pool configuration
struct MemoryPoolConfig {
    size_t initial_pool_size = 1024 * 1024 * 1024; // 1GB
    size_t max_pool_size = 8ULL * 1024 * 1024 * 1024; // 8GB
    size_t block_size_increment = 1024 * 1024; // 1MB
    size_t min_block_size = 64;
    size_t max_block_size = 1024 * 1024 * 1024; // 1GB
    
    bool enable_coalescing = true;
    bool enable_splitting = true;
    bool enable_defragmentation = true;
    bool enable_preallocation = false;
    
    double growth_factor = 1.5;
    double fragmentation_threshold = 0.3;
    size_t defragmentation_trigger_count = 100;
    
    // Pool-specific settings
    std::vector<MemoryType> supported_types = {MemoryType::CPU_MEMORY};
    std::unordered_map<MemoryType, size_t> type_specific_alignment;
    std::unordered_map<MemoryType, size_t> type_specific_pool_size;
    
    MemoryPoolConfig() {
        type_specific_alignment[MemoryType::CPU_MEMORY] = DEFAULT_ALIGNMENT;
        type_specific_alignment[MemoryType::GPU_MEMORY] = GPU_MEMORY_ALIGNMENT;
        type_specific_alignment[MemoryType::UNIFIED_MEMORY] = GPU_MEMORY_ALIGNMENT;
        type_specific_alignment[MemoryType::PINNED_MEMORY] = DEFAULT_ALIGNMENT;
        
        type_specific_pool_size[MemoryType::CPU_MEMORY] = initial_pool_size;
        type_specific_pool_size[MemoryType::GPU_MEMORY] = initial_pool_size / 2;
        type_specific_pool_size[MemoryType::UNIFIED_MEMORY] = initial_pool_size / 4;
        type_specific_pool_size[MemoryType::PINNED_MEMORY] = initial_pool_size / 8;
    }
};

// Memory pool exception
class MemoryPoolException : public std::runtime_error {
public:
    explicit MemoryPoolException(const std::string& message) 
        : std::runtime_error("MemoryPool: " + message) {}
};

// Main memory pool class
class MemoryPool {
public:
    explicit MemoryPool(const MemoryPoolConfig& config = MemoryPoolConfig{});
    explicit MemoryPool(std::unique_ptr<MemoryAllocator> allocator, 
                       const MemoryPoolConfig& config = MemoryPoolConfig{});
    
    // Disable copy constructor and assignment
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Enable move constructor and assignment
    MemoryPool(MemoryPool&& other) noexcept;
    MemoryPool& operator=(MemoryPool&& other) noexcept;
    
    ~MemoryPool();
    
    // Memory allocation and deallocation
    void* allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT, 
                   MemoryType type = MemoryType::CPU_MEMORY);
    void deallocate(void* ptr);
    
    // Typed allocation with RAII wrapper
    template<typename T>
    class ManagedPtr {
    public:
        ManagedPtr() : ptr_(nullptr), pool_(nullptr), size_(0) {}
        ManagedPtr(T* ptr, MemoryPool* pool, size_t size) 
            : ptr_(ptr), pool_(pool), size_(size) {}
        
        ~ManagedPtr() {
            if (ptr_ && pool_) {
                pool_->deallocate(ptr_);
            }
        }
        
        // Move constructor and assignment
        ManagedPtr(ManagedPtr&& other) noexcept 
            : ptr_(other.ptr_), pool_(other.pool_), size_(other.size_) {
            other.ptr_ = nullptr;
            other.pool_ = nullptr;
            other.size_ = 0;
        }
        
        ManagedPtr& operator=(ManagedPtr&& other) noexcept {
            if (this != &other) {
                if (ptr_ && pool_) {
                    pool_->deallocate(ptr_);
                }
                ptr_ = other.ptr_;
                pool_ = other.pool_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.pool_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        
        // Disable copy
        ManagedPtr(const ManagedPtr&) = delete;
        ManagedPtr& operator=(const ManagedPtr&) = delete;
        
        T* get() const { return ptr_; }
        T* operator->() const { return ptr_; }
        T& operator*() const { return *ptr_; }
        size_t size() const { return size_; }
        bool valid() const { return ptr_ != nullptr; }
        
        T* release() {
            T* result = ptr_;
            ptr_ = nullptr;
            pool_ = nullptr;
            size_ = 0;
            return result;
        }
        
    private:
        T* ptr_;
        MemoryPool* pool_;
        size_t size_;
    };
    
    template<typename T>
    ManagedPtr<T> allocate_managed(size_t count = 1, size_t alignment = alignof(T), 
                                  MemoryType type = MemoryType::CPU_MEMORY) {
        size_t size = count * sizeof(T);
        T* ptr = static_cast<T*>(allocate(size, alignment, type));
        if (!ptr) {
            throw MemoryPoolException("Failed to allocate managed memory");
        }
        return ManagedPtr<T>(ptr, this, size);
    }
    
    // Memory pool management
    void expand_pool(size_t additional_size, MemoryType type = MemoryType::CPU_MEMORY);
    void shrink_pool(MemoryType type = MemoryType::CPU_MEMORY);
    void defragment(MemoryType type = MemoryType::CPU_MEMORY);
    void clear_pool(MemoryType type = MemoryType::CPU_MEMORY);
    void reset_all_pools();
    
    // Memory statistics and monitoring
    MemoryStats get_stats(MemoryType type) const;
    MemoryStats get_total_stats() const;
    size_t get_pool_size(MemoryType type) const;
    size_t get_used_memory(MemoryType type) const;
    size_t get_free_memory(MemoryType type) const;
    double get_fragmentation_ratio(MemoryType type) const;
    size_t get_largest_free_block(MemoryType type) const;
    
    // Pool configuration and state
    void set_config(const MemoryPoolConfig& config);
    MemoryPoolConfig get_config() const;
    bool supports_memory_type(MemoryType type) const;
    std::vector<MemoryType> get_supported_types() const;
    
    // Advanced memory operations
    void* reallocate(void* ptr, size_t new_size, size_t alignment = DEFAULT_ALIGNMENT);
    bool owns_pointer(void* ptr) const;
    size_t get_allocation_size(void* ptr) const;
    MemoryType get_allocation_type(void* ptr) const;
    
    // Memory debugging and validation
    bool validate_heap() const;
    void dump_memory_layout(MemoryType type) const;
    std::vector<MemoryBlock> get_free_blocks(MemoryType type) const;
    std::vector<MemoryBlock> get_allocated_blocks(MemoryType type) const;
    
    // Thread safety
    void enable_thread_safety(bool enable = true);
    bool is_thread_safe() const;
    
    // Memory prefilling and pattern detection
    void set_debug_pattern(uint8_t pattern = 0xCD);
    void enable_memory_pattern_checking(bool enable = true);
    bool verify_memory_pattern(void* ptr, size_t size) const;
    
    // Callback system for monitoring
    using AllocationCallback = std::function<void(void*, size_t, MemoryType)>;
    using DeallocationCallback = std::function<void(void*, size_t, MemoryType)>;
    
    void set_allocation_callback(AllocationCallback callback);
    void set_deallocation_callback(DeallocationCallback callback);
    void clear_callbacks();

private:
    // Internal data structures
    struct PoolData {
        std::vector<MemoryBlock> blocks;
        std::vector<void*> pool_chunks;
        MemoryStats stats;
        size_t current_pool_size;
        size_t next_allocation_id;
        
        PoolData() : current_pool_size(0), next_allocation_id(1) {}
    };
    
    // Configuration and allocator
    MemoryPoolConfig config_;
    std::unique_ptr<MemoryAllocator> allocator_;
    
    // Per-type pool data
    mutable std::mutex pools_mutex_;
    std::unordered_map<MemoryType, PoolData> pools_;
    
    // Thread safety
    bool thread_safe_enabled_;
    mutable std::mutex allocation_mutex_;
    
    // Debug and monitoring
    uint8_t debug_pattern_;
    bool pattern_checking_enabled_;
    std::atomic<size_t> total_allocations_;
    std::atomic<size_t> total_deallocations_;
    
    // Callbacks
    AllocationCallback allocation_callback_;
    DeallocationCallback deallocation_callback_;
    mutable std::mutex callback_mutex_;
    
    // Internal helper methods
    void initialize_pool(MemoryType type);
    void* allocate_from_pool(size_t size, size_t alignment, MemoryType type);
    bool deallocate_from_pool(void* ptr, MemoryType type);
    
    MemoryBlock* find_free_block(size_t size, size_t alignment, MemoryType type);
    MemoryBlock* find_allocated_block(void* ptr, MemoryType type);
    
    void split_block(MemoryBlock& block, size_t size, size_t alignment);
    void coalesce_free_blocks(MemoryType type);
    void grow_pool(MemoryType type, size_t min_size);
    
    size_t align_size(size_t size, size_t alignment) const;
    void* align_pointer(void* ptr, size_t alignment) const;
    bool is_aligned(void* ptr, size_t alignment) const;
    
    void update_stats(MemoryType type, size_t size, bool is_allocation);
    void fill_with_pattern(void* ptr, size_t size, uint8_t pattern);
    bool check_pattern(void* ptr, size_t size, uint8_t pattern) const;
    
    void validate_pool_integrity(MemoryType type) const;
    void cleanup_pool(MemoryType type);
    
    // Thread safety helpers
    template<typename Func>
    auto with_lock(Func&& func) const -> decltype(func()) {
        if (thread_safe_enabled_) {
            std::lock_guard<std::mutex> lock(allocation_mutex_);
            return func();
        } else {
            return func();
        }
    }
    
    template<typename Func>
    auto with_pools_lock(Func&& func) const -> decltype(func()) {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        return func();
    }
    
    // Memory type validation
    void validate_memory_type(MemoryType type) const;
    size_t get_default_alignment(MemoryType type) const;
    
    // Pool growth strategy
    size_t calculate_growth_size(MemoryType type, size_t requested_size) const;
    bool should_grow_pool(MemoryType type, size_t requested_size) const;
    bool should_defragment(MemoryType type) const;
};

// Memory pool factory functions
std::unique_ptr<MemoryPool> create_cpu_memory_pool(size_t initial_size = 1024 * 1024 * 1024);
std::unique_ptr<MemoryPool> create_gpu_memory_pool(size_t initial_size = 512 * 1024 * 1024);
std::unique_ptr<MemoryPool> create_unified_memory_pool(size_t initial_size = 256 * 1024 * 1024);

// Global memory pool instance
MemoryPool& get_global_memory_pool();
void set_global_memory_pool(std::unique_ptr<MemoryPool> pool);

// Utility functions
size_t align_up(size_t size, size_t alignment);
size_t align_down(size_t size, size_t alignment);
bool is_power_of_two(size_t value);
size_t next_power_of_two(size_t value);

// Memory type conversion utilities
std::string memory_type_to_string(MemoryType type);
MemoryType string_to_memory_type(const std::string& type_str);

// Memory pool statistics utilities
namespace memory_pool_utils {
    void print_memory_stats(const MemoryStats& stats);
    void print_pool_summary(const MemoryPool& pool);
    double calculate_memory_efficiency(const MemoryStats& stats);
    size_t estimate_overhead(size_t allocation_size, size_t alignment);
    
    // Memory usage recommendations
    struct MemoryRecommendation {
        size_t recommended_pool_size;
        size_t recommended_block_size;
        bool enable_defragmentation;
        bool enable_preallocation;
        std::string reason;
    };
    
    MemoryRecommendation analyze_usage_pattern(const MemoryStats& stats);
}

} // namespace attention_hpc
