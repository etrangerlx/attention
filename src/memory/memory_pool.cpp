#include "memory_pool.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#endif

namespace attention_hpc {

// SystemMemoryAllocator implementation
void* SystemMemoryAllocator::allocate(size_t size, size_t alignment, MemoryType type) {
    if (!supports_type(type)) {
        throw MemoryPoolException("Unsupported memory type");
    }
    
    if (type == MemoryType::CPU_MEMORY || type == MemoryType::PINNED_MEMORY) {
        return allocate_aligned(size, alignment);
    }
    
    // For GPU and unified memory, we would need specific GPU APIs
    // For now, fall back to CPU allocation
    return allocate_aligned(size, alignment);
}

void SystemMemoryAllocator::deallocate(void* ptr, MemoryType type) {
    if (ptr) {
        deallocate_aligned(ptr);
    }
}

size_t SystemMemoryAllocator::get_total_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullTotalPhys);
    }
    return 0;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<size_t>(info.totalram * info.mem_unit);
    }
    return 0;
#endif
}

size_t SystemMemoryAllocator::get_free_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys);
    }
    return 0;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<size_t>(info.freeram * info.mem_unit);
    }
    return 0;
#endif
}

bool SystemMemoryAllocator::supports_type(MemoryType type) const {
    return type == MemoryType::CPU_MEMORY || type == MemoryType::PINNED_MEMORY;
}

void* SystemMemoryAllocator::allocate_aligned(size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        alignment = DEFAULT_ALIGNMENT;
    }
    
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    int result = posix_memalign(&ptr, alignment, size);
    return (result == 0) ? ptr : nullptr;
#endif
}

void SystemMemoryAllocator::deallocate_aligned(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// MemoryPool implementation
MemoryPool::MemoryPool(const MemoryPoolConfig& config)
    : config_(config), allocator_(std::make_unique<SystemMemoryAllocator>()),
      thread_safe_enabled_(true), debug_pattern_(0xCD), pattern_checking_enabled_(false),
      total_allocations_(0), total_deallocations_(0) {
    
    // Initialize pools for supported memory types
    for (MemoryType type : config_.supported_types) {
        initialize_pool(type);
    }
}

MemoryPool::MemoryPool(std::unique_ptr<MemoryAllocator> allocator, const MemoryPoolConfig& config)
    : config_(config), allocator_(std::move(allocator)),
      thread_safe_enabled_(true), debug_pattern_(0xCD), pattern_checking_enabled_(false),
      total_allocations_(0), total_deallocations_(0) {
    
    // Initialize pools for supported memory types
    for (MemoryType type : config_.supported_types) {
        initialize_pool(type);
    }
}

MemoryPool::MemoryPool(MemoryPool&& other) noexcept
    : config_(std::move(other.config_)), allocator_(std::move(other.allocator_)),
      pools_(std::move(other.pools_)), thread_safe_enabled_(other.thread_safe_enabled_),
      debug_pattern_(other.debug_pattern_), pattern_checking_enabled_(other.pattern_checking_enabled_),
      total_allocations_(other.total_allocations_.load()), total_deallocations_(other.total_deallocations_.load()),
      allocation_callback_(std::move(other.allocation_callback_)),
      deallocation_callback_(std::move(other.deallocation_callback_)) {
    
    other.thread_safe_enabled_ = false;
    other.pattern_checking_enabled_ = false;
    other.total_allocations_ = 0;
    other.total_deallocations_ = 0;
}

MemoryPool& MemoryPool::operator=(MemoryPool&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        reset_all_pools();
        
        // Move data from other
        config_ = std::move(other.config_);
        allocator_ = std::move(other.allocator_);
        pools_ = std::move(other.pools_);
        thread_safe_enabled_ = other.thread_safe_enabled_;
        debug_pattern_ = other.debug_pattern_;
        pattern_checking_enabled_ = other.pattern_checking_enabled_;
        total_allocations_ = other.total_allocations_.load();
        total_deallocations_ = other.total_deallocations_.load();
        allocation_callback_ = std::move(other.allocation_callback_);
        deallocation_callback_ = std::move(other.deallocation_callback_);
        
        // Reset other
        other.thread_safe_enabled_ = false;
        other.pattern_checking_enabled_ = false;
        other.total_allocations_ = 0;
        other.total_deallocations_ = 0;
    }
    return *this;
}

MemoryPool::~MemoryPool() {
    reset_all_pools();
}

void* MemoryPool::allocate(size_t size, size_t alignment, MemoryType type) {
    if (size == 0) {
        return nullptr;
    }
    
    validate_memory_type(type);
    
    if (alignment == 0) {
        alignment = get_default_alignment(type);
    }
    
    return with_lock([&]() -> void* {
        void* ptr = allocate_from_pool(size, alignment, type);
        
        if (ptr) {
            update_stats(type, size, true);
            total_allocations_++;
            
            if (pattern_checking_enabled_) {
                fill_with_pattern(ptr, size, debug_pattern_);
            }
            
            // Call allocation callback if set
            if (allocation_callback_) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                allocation_callback_(ptr, size, type);
            }
        }
        
        return ptr;
    });
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }
    
    with_lock([&]() {
        // Find which pool owns this pointer
        for (auto& [type, pool_data] : pools_) {
            if (deallocate_from_pool(ptr, type)) {
                total_deallocations_++;
                
                // Call deallocation callback if set
                if (deallocation_callback_) {
                    std::lock_guard<std::mutex> lock(callback_mutex_);
                    deallocation_callback_(ptr, 0, type); // Size not tracked for deallocation
                }
                return;
            }
        }
        
        // If we reach here, the pointer wasn't found in any pool
        throw MemoryPoolException("Attempting to deallocate pointer not owned by this pool");
    });
}

void MemoryPool::expand_pool(size_t additional_size, MemoryType type) {
    validate_memory_type(type);
    
    with_lock([&]() {
        grow_pool(type, additional_size);
    });
}

void MemoryPool::shrink_pool(MemoryType type) {
    validate_memory_type(type);
    
    with_pools_lock([&]() {
        auto& pool_data = pools_[type];
        
        // Coalesce free blocks first
        coalesce_free_blocks(type);
        
        // Find large free blocks at the end that can be released
        std::vector<void*> chunks_to_release;
        
        for (auto it = pool_data.pool_chunks.rbegin(); it != pool_data.pool_chunks.rend(); ++it) {
            void* chunk = *it;
            
            // Check if this entire chunk is free
            bool chunk_is_free = true;
            for (const auto& block : pool_data.blocks) {
                if (block.ptr >= chunk && 
                    static_cast<char*>(block.ptr) < static_cast<char*>(chunk) + config_.block_size_increment) {
                    if (!block.is_free) {
                        chunk_is_free = false;
                        break;
                    }
                }
            }
            
            if (chunk_is_free) {
                chunks_to_release.push_back(chunk);
            } else {
                break; // Stop at first non-free chunk
            }
        }
        
        // Release chunks
        for (void* chunk : chunks_to_release) {
            allocator_->deallocate(chunk, type);
            
            // Remove from pool chunks
            auto chunk_it = std::find(pool_data.pool_chunks.begin(), pool_data.pool_chunks.end(), chunk);
            if (chunk_it != pool_data.pool_chunks.end()) {
                pool_data.pool_chunks.erase(chunk_it);
            }
            
            // Remove corresponding blocks
            pool_data.blocks.erase(
                std::remove_if(pool_data.blocks.begin(), pool_data.blocks.end(),
                    [chunk](const MemoryBlock& block) {
                        return block.ptr >= chunk && 
                               static_cast<char*>(block.ptr) < static_cast<char*>(chunk) + config_.block_size_increment;
                    }), 
                pool_data.blocks.end());
            
            pool_data.current_pool_size -= config_.block_size_increment;
        }
    });
}

void MemoryPool::defragment(MemoryType type) {
    validate_memory_type(type);
    
    with_lock([&]() {
        coalesce_free_blocks(type);
    });
}

void MemoryPool::clear_pool(MemoryType type) {
    validate_memory_type(type);
    
    with_pools_lock([&]() {
        cleanup_pool(type);
        initialize_pool(type);
    });
}

void MemoryPool::reset_all_pools() {
    with_pools_lock([&]() {
        for (auto& [type, pool_data] : pools_) {
            cleanup_pool(type);
        }
        pools_.clear();
    });
}

MemoryStats MemoryPool::get_stats(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> MemoryStats {
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            return it->second.stats;
        }
        return MemoryStats{};
    });
}

MemoryStats MemoryPool::get_total_stats() const {
    return with_pools_lock([&]() -> MemoryStats {
        MemoryStats total_stats;
        
        for (const auto& [type, pool_data] : pools_) {
            const auto& stats = pool_data.stats;
            total_stats.total_allocated += stats.total_allocated;
            total_stats.total_freed += stats.total_freed;
            total_stats.current_usage += stats.current_usage;
            total_stats.peak_usage = std::max(total_stats.peak_usage, stats.peak_usage);
            total_stats.num_allocations += stats.num_allocations;
            total_stats.num_deallocations += stats.num_deallocations;
            total_stats.num_allocation_failures += stats.num_allocation_failures;
            total_stats.fragmentation_bytes += stats.fragmentation_bytes;
        }
        
        return total_stats;
    });
}

size_t MemoryPool::get_pool_size(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> size_t {
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            return it->second.current_pool_size;
        }
        return 0;
    });
}

size_t MemoryPool::get_used_memory(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> size_t {
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            return it->second.stats.current_usage;
        }
        return 0;
    });
}

size_t MemoryPool::get_free_memory(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> size_t {
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            const auto& pool_data = it->second;
            return pool_data.current_pool_size - pool_data.stats.current_usage;
        }
        return 0;
    });
}

double MemoryPool::get_fragmentation_ratio(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> double {
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            return it->second.stats.get_fragmentation_ratio();
        }
        return 0.0;
    });
}

size_t MemoryPool::get_largest_free_block(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> size_t {
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            size_t largest = 0;
            for (const auto& block : it->second.blocks) {
                if (block.is_free && block.size > largest) {
                    largest = block.size;
                }
            }
            return largest;
        }
        return 0;
    });
}

void MemoryPool::set_config(const MemoryPoolConfig& config) {
    with_pools_lock([&]() {
        config_ = config;
        
        // Reinitialize pools with new configuration
        reset_all_pools();
        for (MemoryType type : config_.supported_types) {
            initialize_pool(type);
        }
    });
}

MemoryPoolConfig MemoryPool::get_config() const {
    return config_;
}

bool MemoryPool::supports_memory_type(MemoryType type) const {
    return std::find(config_.supported_types.begin(), config_.supported_types.end(), type) 
           != config_.supported_types.end();
}

std::vector<MemoryType> MemoryPool::get_supported_types() const {
    return config_.supported_types;
}

void* MemoryPool::reallocate(void* ptr, size_t new_size, size_t alignment) {
    if (!ptr) {
        return allocate(new_size, alignment);
    }
    
    if (new_size == 0) {
        deallocate(ptr);
        return nullptr;
    }
    
    return with_lock([&]() -> void* {
        // Find the original allocation
        size_t old_size = 0;
        MemoryType type = MemoryType::CPU_MEMORY;
        
        for (auto& [pool_type, pool_data] : pools_) {
            MemoryBlock* block = find_allocated_block(ptr, pool_type);
            if (block) {
                old_size = block->size;
                type = pool_type;
                break;
            }
        }
        
        if (old_size == 0) {
            throw MemoryPoolException("Reallocating pointer not owned by this pool");
        }
        
        // Try to expand in place if possible
        if (new_size <= old_size) {
            return ptr; // Shrinking or same size, no need to move
        }
        
        // Allocate new block
        void* new_ptr = allocate(new_size, alignment, type);
        if (!new_ptr) {
            return nullptr;
        }
        
        // Copy old data
        std::memcpy(new_ptr, ptr, std::min(old_size, new_size));
        
        // Free old block
        deallocate(ptr);
        
        return new_ptr;
    });
}

bool MemoryPool::owns_pointer(void* ptr) const {
    if (!ptr) {
        return false;
    }
    
    return with_pools_lock([&]() -> bool {
        for (const auto& [type, pool_data] : pools_) {
            for (const auto& chunk : pool_data.pool_chunks) {
                if (ptr >= chunk && 
                    static_cast<char*>(ptr) < static_cast<char*>(chunk) + config_.block_size_increment) {
                    return true;
                }
            }
        }
        return false;
    });
}

size_t MemoryPool::get_allocation_size(void* ptr) const {
    if (!ptr) {
        return 0;
    }
    
    return with_pools_lock([&]() -> size_t {
        for (const auto& [type, pool_data] : pools_) {
            for (const auto& block : pool_data.blocks) {
                if (block.ptr == ptr && !block.is_free) {
                    return block.size;
                }
            }
        }
        return 0;
    });
}

MemoryType MemoryPool::get_allocation_type(void* ptr) const {
    if (!ptr) {
        return MemoryType::CPU_MEMORY;
    }
    
    return with_pools_lock([&]() -> MemoryType {
        for (const auto& [type, pool_data] : pools_) {
            for (const auto& block : pool_data.blocks) {
                if (block.ptr == ptr && !block.is_free) {
                    return type;
                }
            }
        }
        throw MemoryPoolException("Pointer not found in any pool");
    });
}

bool MemoryPool::validate_heap() const {
    return with_pools_lock([&]() -> bool {
        for (const auto& [type, pool_data] : pools_) {
            validate_pool_integrity(type);
        }
        return true;
    });
}

void MemoryPool::dump_memory_layout(MemoryType type) const {
    validate_memory_type(type);
    
    with_pools_lock([&]() {
        auto it = pools_.find(type);
        if (it == pools_.end()) {
            std::cout << "Memory type " << static_cast<int>(type) << " not found\n";
            return;
        }
        
        const auto& pool_data = it->second;
        std::cout << "Memory layout for type " << static_cast<int>(type) << ":\n";
        std::cout << "Pool size: " << pool_data.current_pool_size << " bytes\n";
        std::cout << "Number of blocks: " << pool_data.blocks.size() << "\n";
        
        for (size_t i = 0; i < pool_data.blocks.size(); ++i) {
            const auto& block = pool_data.blocks[i];
            std::cout << "Block " << i << ": ptr=" << block.ptr 
                     << ", size=" << block.size
                     << ", " << (block.is_free ? "FREE" : "ALLOCATED")
                     << ", alignment=" << block.alignment << "\n";
        }
    });
}

std::vector<MemoryBlock> MemoryPool::get_free_blocks(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> std::vector<MemoryBlock> {
        std::vector<MemoryBlock> free_blocks;
        
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            for (const auto& block : it->second.blocks) {
                if (block.is_free) {
                    free_blocks.push_back(block);
                }
            }
        }
        
        return free_blocks;
    });
}

std::vector<MemoryBlock> MemoryPool::get_allocated_blocks(MemoryType type) const {
    validate_memory_type(type);
    
    return with_pools_lock([&]() -> std::vector<MemoryBlock> {
        std::vector<MemoryBlock> allocated_blocks;
        
        auto it = pools_.find(type);
        if (it != pools_.end()) {
            for (const auto& block : it->second.blocks) {
                if (!block.is_free) {
                    allocated_blocks.push_back(block);
                }
            }
        }
        
        return allocated_blocks;
    });
}

void MemoryPool::enable_thread_safety(bool enable) {
    thread_safe_enabled_ = enable;
}

bool MemoryPool::is_thread_safe() const {
    return thread_safe_enabled_;
}

void MemoryPool::set_debug_pattern(uint8_t pattern) {
    debug_pattern_ = pattern;
}

void MemoryPool::enable_memory_pattern_checking(bool enable) {
    pattern_checking_enabled_ = enable;
}

bool MemoryPool::verify_memory_pattern(void* ptr, size_t size) const {
    return check_pattern(ptr, size, debug_pattern_);
}

void MemoryPool::set_allocation_callback(AllocationCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    allocation_callback_ = callback;
}

void MemoryPool::set_deallocation_callback(DeallocationCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    deallocation_callback_ = callback;
}

void MemoryPool::clear_callbacks() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    allocation_callback_ = nullptr;
    deallocation_callback_ = nullptr;
}

// Private helper methods
void MemoryPool::initialize_pool(MemoryType type) {
    auto it = config_.type_specific_pool_size.find(type);
    size_t initial_size = (it != config_.type_specific_pool_size.end()) ? 
                         it->second : config_.initial_pool_size;
    
    PoolData& pool_data = pools_[type];
    pool_data.current_pool_size = 0;
    pool_data.next_allocation_id = 1;
    pool_data.stats.reset();
    
    if (initial_size > 0) {
        grow_pool(type, initial_size);
    }
}

void* MemoryPool::allocate_from_pool(size_t size, size_t alignment, MemoryType type) {
    size_t aligned_size = align_size(size, alignment);
    
    auto it = pools_.find(type);
    if (it == pools_.end()) {
        throw MemoryPoolException("Memory type not supported");
    }
    
    PoolData& pool_data = it->second;
    
    // Try to find a suitable free block
    MemoryBlock* block = find_free_block(aligned_size, alignment, type);
    
    if (!block) {
        // No suitable block found, try to grow the pool
        if (should_grow_pool(type, aligned_size)) {
            size_t growth_size = calculate_growth_size(type, aligned_size);
            grow_pool(type, growth_size);
            block = find_free_block(aligned_size, alignment, type);
        }
        
        if (!block) {
            pool_data.stats.num_allocation_failures++;
            return nullptr;
        }
    }
    
    // Split the block if necessary
    if (config_.enable_splitting && block->size > aligned_size + config_.min_block_size) {
        split_block(*block, aligned_size, alignment);
    }
    
    // Mark block as allocated
    block->is_free = false;
    block->allocation_id = pool_data.next_allocation_id++;
    
    return align_pointer(block->ptr, alignment);
}

bool MemoryPool::deallocate_from_pool(void* ptr, MemoryType type) {
    auto it = pools_.find(type);
    if (it == pools_.end()) {
        return false;
    }
    
    PoolData& pool_data = it->second;
    MemoryBlock* block = find_allocated_block(ptr, type);
    
    if (!block) {
        return false;
    }
    
    // Mark block as free
    block->is_free = true;
    block->allocation_id = 0;
    
    // Fill with debug pattern if enabled
    if (pattern_checking_enabled_) {
        fill_with_pattern(block->ptr, block->size, debug_pattern_);
    }
    
    // Update statistics
    update_stats(type, block->size, false);
    
    // Coalesce with adjacent free blocks if enabled
    if (config_.enable_coalescing) {
        coalesce_free_blocks(type);
    }
    
    return true;
}

MemoryBlock* MemoryPool::find_free_block(size_t size, size_t alignment, MemoryType type) {
    auto it = pools_.find(type);
    if (it == pools_.end()) {
        return nullptr;
    }
    
    PoolData& pool_data = it->second;
    
    // Find the best-fit free block
    MemoryBlock* best_block = nullptr;
    size_t best_size = SIZE_MAX;
    
    for (auto& block : pool_data.blocks) {
        if (block.is_free && block.can_satisfy(size, alignment)) {
            if (block.size < best_size) {
                best_block = &block;
                best_size = block.size;
            }
        }
    }
    
    return best_block;
}

MemoryBlock* MemoryPool::find_allocated_block(void* ptr, MemoryType type) {
    auto it = pools_.find(type);
    if (it == pools_.end()) {
        return nullptr;
    }
    
    PoolData& pool_data = it->second;
    
    for (auto& block : pool_data.blocks) {
        if (!block.is_free && block.ptr == ptr) {
            return &block;
        }
        
        // Also check if ptr is within the aligned region of the block
        if (!block.is_free) {
            void* aligned_ptr = align_pointer(block.ptr, block.alignment);
            if (ptr == aligned_ptr) {
                return &block;
            }
        }
    }
    
    return nullptr;
}

void MemoryPool::split_block(MemoryBlock& block, size_t size, size_t alignment) {
    if (block.size <= size + config_.min_block_size) {
        return; // Block too small to split
    }
    
    // Calculate the split point
    void* aligned_ptr = align_pointer(block.ptr, alignment);
    size_t alignment_offset = static_cast<char*>(aligned_ptr) - static_cast<char*>(block.ptr);
    size_t total_used_size = alignment_offset + size;
    
    if (block.size <= total_used_size + config_.min_block_size) {
        return; // Not enough space for a new block
    }
    
    // Create new block for the remaining space
    MemoryBlock new_block;
    new_block.ptr = static_cast<char*>(block.ptr) + total_used_size;
    new_block.size = block.size - total_used_size;
    new_block.alignment = config_.min_block_size; // Default alignment for split blocks
    new_block.type = block.type;
    new_block.is_free = true;
    new_block.allocation_id = 0;
    
    // Update original block
    block.size = total_used_size;
    
    // Find the correct position to insert the new block
    auto it = pools_.find(block.type);
    if (it != pools_.end()) {
        auto& blocks = it->second.blocks;
        auto block_it = std::find_if(blocks.begin(), blocks.end(),
            [&block](const MemoryBlock& b) { return &b == &block; });
        
        if (block_it != blocks.end()) {
            blocks.insert(block_it + 1, new_block);
        }
    }
}

void MemoryPool::coalesce_free_blocks(MemoryType type) {
    auto it = pools_.find(type);
    if (it == pools_.end()) {
        return;
    }
    
    PoolData& pool_data = it->second;
    auto& blocks = pool_data.blocks;
    
    if (blocks.size() < 2) {
        return;
    }
    
    // Sort blocks by memory address
    std::sort(blocks.begin(), blocks.end(),
        [](const MemoryBlock& a, const MemoryBlock& b) {
            return a.ptr < b.ptr;
        });
    
    // Coalesce adjacent free blocks
    bool changed = true;
    while (changed) {
        changed = false;
        
        for (size_t i = 0; i < blocks.size() - 1; ++i) {
            MemoryBlock& current = blocks[i];
            MemoryBlock& next = blocks[i + 1];
            
            if (current.is_free && next.is_free) {
                // Check if blocks are adjacent
                char* current_end = static_cast<char*>(current.ptr) + current.size;
                if (current_end == next.ptr) {
                    // Merge the blocks
                    current.size += next.size;
                    blocks.erase(blocks.begin() + i + 1);
                    changed = true;
                    
                    // Update fragmentation statistics
                    pool_data.stats.fragmentation_bytes -= std::min(current.size, next.size);
                    break;
                }
            }
        }
    }
}

void MemoryPool::grow_pool(MemoryType type, size_t min_size) {
    auto it = pools_.find(type);
    if (it == pools_.end()) {
        return;
    }
    
    PoolData& pool_data = it->second;
    
