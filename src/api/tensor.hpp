#pragma once

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <initializer_list>

namespace attention_hpc {

// Forward declarations
enum class DataType;
enum class MemoryLayout;

// Memory location enumeration
enum class MemoryLocation {
    CPU,
    GPU,
    UNIFIED  // Unified memory (CUDA managed memory)
};

// Memory alignment utilities
constexpr size_t DEFAULT_ALIGNMENT = 64; // 64-byte alignment for SIMD

template<typename T>
constexpr bool is_supported_type_v = std::is_same_v<T, float> || 
                                    std::is_same_v<T, double> ||
                                    std::is_same_v<T, int32_t> ||
                                    std::is_same_v<T, int64_t> ||
                                    std::is_same_v<T, uint32_t> ||
                                    std::is_same_v<T, uint64_t> ||
                                    std::is_same_v<T, int16_t> ||
                                    std::is_same_v<T, uint16_t>;

// Half precision type (16-bit float)
struct half {
    uint16_t data;
    
    half() : data(0) {}
    half(float f);
    operator float() const;
    
    half& operator=(float f) { *this = half(f); return *this; }
    half& operator+=(const half& other) { *this = half(float(*this) + float(other)); return *this; }
    half& operator-=(const half& other) { *this = half(float(*this) - float(other)); return *this; }
    half& operator*=(const half& other) { *this = half(float(*this) * float(other)); return *this; }
    half& operator/=(const half& other) { *this = half(float(*this) / float(other)); return *this; }
};

// BFloat16 type
struct bfloat16 {
    uint16_t data;
    
    bfloat16() : data(0) {}
    bfloat16(float f);
    operator float() const;
    
    bfloat16& operator=(float f) { *this = bfloat16(f); return *this; }
    bfloat16& operator+=(const bfloat16& other) { *this = bfloat16(float(*this) + float(other)); return *this; }
    bfloat16& operator-=(const bfloat16& other) { *this = bfloat16(float(*this) - float(other)); return *this; }
    bfloat16& operator*=(const bfloat16& other) { *this = bfloat16(float(*this) * float(other)); return *this; }
    bfloat16& operator/=(const bfloat16& other) { *this = bfloat16(float(*this) / float(other)); return *this; }
};

// Type trait to get DataType enum from C++ type
template<typename T>
struct type_to_data_type;

template<> struct type_to_data_type<float> { static constexpr DataType value = DataType::FLOAT32; };
template<> struct type_to_data_type<half> { static constexpr DataType value = DataType::FLOAT16; };
template<> struct type_to_data_type<bfloat16> { static constexpr DataType value = DataType::BFLOAT16; };

template<typename T>
constexpr DataType type_to_data_type_v = type_to_data_type<T>::value;

// Memory management interface
class MemoryManager {
public:
    virtual ~MemoryManager() = default;
    
    virtual void* allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT, 
                          MemoryLocation location = MemoryLocation::CPU) = 0;
    virtual void deallocate(void* ptr, MemoryLocation location = MemoryLocation::CPU) = 0;
    virtual void copy(void* dst, const void* src, size_t size, 
                     MemoryLocation dst_loc, MemoryLocation src_loc) = 0;
    virtual void set_zero(void* ptr, size_t size, MemoryLocation location) = 0;
    
    // Memory pool management
    virtual void* allocate_from_pool(size_t size, size_t alignment = DEFAULT_ALIGNMENT) = 0;
    virtual void return_to_pool(void* ptr) = 0;
    virtual size_t get_peak_memory_usage() const = 0;
    virtual size_t get_current_memory_usage() const = 0;
};

// Default memory manager
std::shared_ptr<MemoryManager> get_default_memory_manager();
void set_default_memory_manager(std::shared_ptr<MemoryManager> manager);

// Tensor shape class
class TensorShape {
public:
    TensorShape() = default;
    TensorShape(std::initializer_list<size_t> dims) : dimensions_(dims) {}
    TensorShape(const std::vector<size_t>& dims) : dimensions_(dims) {}
    
    // Accessors
    size_t ndim() const { return dimensions_.size(); }
    size_t size(size_t dim) const { 
        if (dim >= dimensions_.size()) throw std::out_of_range("Dimension index out of range");
        return dimensions_[dim]; 
    }
    const std::vector<size_t>& dimensions() const { return dimensions_; }
    
    // Total number of elements
    size_t total_size() const {
        size_t total = 1;
        for (size_t dim : dimensions_) {
            total *= dim;
        }
        return total;
    }
    
    // Reshape operations
    TensorShape reshape(const std::vector<size_t>& new_dims) const {
        TensorShape result(new_dims);
        if (result.total_size() != total_size()) {
            throw std::invalid_argument("New shape must have same total size");
        }
        return result;
    }
    
    TensorShape squeeze(int dim = -1) const {
        std::vector<size_t> new_dims;
        for (size_t i = 0; i < dimensions_.size(); ++i) {
            if (dim >= 0) {
                if (static_cast<int>(i) != dim || dimensions_[i] != 1) {
                    new_dims.push_back(dimensions_[i]);
                }
            } else {
                if (dimensions_[i] != 1) {
                    new_dims.push_back(dimensions_[i]);
                }
            }
        }
        return TensorShape(new_dims);
    }
    
    TensorShape unsqueeze(size_t dim) const {
        std::vector<size_t> new_dims = dimensions_;
        new_dims.insert(new_dims.begin() + dim, 1);
        return TensorShape(new_dims);
    }
    
    // Comparison operators
    bool operator==(const TensorShape& other) const {
        return dimensions_ == other.dimensions_;
    }
    
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }
    
    // String representation
    std::string to_string() const {
        std::string result = "[";
        for (size_t i = 0; i < dimensions_.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(dimensions_[i]);
        }
        result += "]";
        return result;
    }

private:
    std::vector<size_t> dimensions_;
};

// Tensor stride calculation utilities
class TensorStrides {
public:
    TensorStrides(const TensorShape& shape, MemoryLayout layout = MemoryLayout::ROW_MAJOR);
    
    size_t stride(size_t dim) const { 
        if (dim >= strides_.size()) throw std::out_of_range("Dimension index out of range");
        return strides_[dim]; 
    }
    const std::vector<size_t>& strides() const { return strides_; }
    
    size_t offset(const std::vector<size_t>& indices) const {
        if (indices.size() != strides_.size()) {
            throw std::invalid_argument("Number of indices must match tensor dimensions");
        }
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

private:
    std::vector<size_t> strides_;
};

// Main Tensor template class
template<typename T>
class Tensor {
    static_assert(is_supported_type_v<T> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16>,
                  "Unsupported tensor data type");

public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    
    // Constructors
    Tensor() : data_(nullptr), shape_(), strides_(shape_), memory_location_(MemoryLocation::CPU),
               owns_data_(false), memory_manager_(get_default_memory_manager()) {}
    
    explicit Tensor(const TensorShape& shape, 
                   MemoryLocation location = MemoryLocation::CPU,
                   MemoryLayout layout = MemoryLayout::ROW_MAJOR)
        : shape_(shape), strides_(shape, layout), memory_location_(location),
          owns_data_(true), memory_manager_(get_default_memory_manager()) {
        allocate_memory();
    }
    
    Tensor(const TensorShape& shape, T* data_ptr,
           MemoryLocation location = MemoryLocation::CPU,
           MemoryLayout layout = MemoryLayout::ROW_MAJOR,
           bool take_ownership = false)
        : data_(data_ptr), shape_(shape), strides_(shape, layout), 
          memory_location_(location), owns_data_(take_ownership),
          memory_manager_(get_default_memory_manager()) {}
    
    Tensor(std::initializer_list<size_t> shape_dims,
           MemoryLocation location = MemoryLocation::CPU,
           MemoryLayout layout = MemoryLayout::ROW_MAJOR)
        : Tensor(TensorShape(shape_dims), location, layout) {}
    
    // Copy constructor
    Tensor(const Tensor& other) 
        : shape_(other.shape_), strides_(other.strides_), 
          memory_location_(other.memory_location_), owns_data_(true),
          memory_manager_(other.memory_manager_) {
        allocate_memory();
        copy_from(other);
    }
    
    // Move constructor
    Tensor(Tensor&& other) noexcept
        : data_(other.data_), shape_(std::move(other.shape_)), 
          strides_(std::move(other.strides_)), memory_location_(other.memory_location_),
          owns_data_(other.owns_data_), memory_manager_(std::move(other.memory_manager_)) {
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    
    // Destructor
    ~Tensor() {
        deallocate_memory();
    }
    
    // Assignment operators
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            deallocate_memory();
            shape_ = other.shape_;
            strides_ = other.strides_;
            memory_location_ = other.memory_location_;
            memory_manager_ = other.memory_manager_;
            owns_data_ = true;
            allocate_memory();
            copy_from(other);
        }
        return *this;
    }
    
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            deallocate_memory();
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            memory_location_ = other.memory_location_;
            owns_data_ = other.owns_data_;
            memory_manager_ = std::move(other.memory_manager_);
            other.data_ = nullptr;
            other.owns_data_ = false;
        }
        return *this;
    }
    
    // Shape and properties
    const TensorShape& shape() const { return shape_; }
    const TensorStrides& strides() const { return strides_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t size(size_t dim) const { return shape_.size(dim); }
    size_t total_size() const { return shape_.total_size(); }
    size_t size_bytes() const { return total_size() * sizeof(T); }
    MemoryLocation memory_location() const { return memory_location_; }
    bool empty() const { return total_size() == 0; }
    
    // Data access
    T* data() { return data_; }
    const T* data() const { return data_; }
    void* raw_data() { return static_cast<void*>(data_); }
    const void* raw_data() const { return static_cast<const void*>(data_); }
    
    // Element access
    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(indices) > 0, "At least one index required");
        return data_[strides_.offset({static_cast<size_t>(indices)...})];
    }
    
    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(indices) > 0, "At least one index required");
        return data_[strides_.offset({static_cast<size_t>(indices)...})];
    }
    
    T& at(const std::vector<size_t>& indices) {
        return data_[strides_.offset(indices)];
    }
    
    const T& at(const std::vector<size_t>& indices) const {
        return data_[strides_.offset(indices)];
    }
    
    // Flat access for 1D operations
    T& operator[](size_t index) {
        if (index >= total_size()) throw std::out_of_range("Index out of range");
        return data_[index];
    }
    
    const T& operator[](size_t index) const {
        if (index >= total_size()) throw std::out_of_range("Index out of range");
        return data_[index];
    }
    
    // Memory operations
    void to(MemoryLocation new_location) {
        if (new_location == memory_location_) return;
        
        T* new_data = static_cast<T*>(memory_manager_->allocate(size_bytes(), DEFAULT_ALIGNMENT, new_location));
        memory_manager_->copy(new_data, data_, size_bytes(), new_location, memory_location_);
        
        if (owns_data_) {
            memory_manager_->deallocate(data_, memory_location_);
        }
        
        data_ = new_data;
        memory_location_ = new_location;
        owns_data_ = true;
    }
    
    Tensor<T> cpu() const {
        if (memory_location_ == MemoryLocation::CPU) return *this;
        Tensor<T> result(shape_, MemoryLocation::CPU, get_memory_layout());
        memory_manager_->copy(result.data_, data_, size_bytes(), 
                             MemoryLocation::CPU, memory_location_);
        return result;
    }
    
    Tensor<T> gpu() const {
        if (memory_location_ == MemoryLocation::GPU) return *this;
        Tensor<T> result(shape_, MemoryLocation::GPU, get_memory_layout());
        memory_manager_->copy(result.data_, data_, size_bytes(), 
                             MemoryLocation::GPU, memory_location_);
        return result;
    }
    
    // Reshape operations
    Tensor<T> reshape(const TensorShape& new_shape) const {
        TensorShape reshaped = shape_.reshape(new_shape.dimensions());
        return Tensor<T>(reshaped, data_, memory_location_, get_memory_layout(), false);
    }
    
    Tensor<T> view(const TensorShape& new_shape) const {
        return reshape(new_shape);
    }
    
    Tensor<T> squeeze(int dim = -1) const {
        TensorShape squeezed = shape_.squeeze(dim);
        return Tensor<T>(squeezed, data_, memory_location_, get_memory_layout(), false);
    }
    
    Tensor<T> unsqueeze(size_t dim) const {
        TensorShape unsqueezed = shape_.unsqueeze(dim);
        return Tensor<T>(unsqueezed, data_, memory_location_, get_memory_layout(), false);
    }
    
    // Fill operations
    void fill(const T& value) {
        if (memory_location_ == MemoryLocation::CPU) {
            std::fill(data_, data_ + total_size(), value);
        } else {
            // For GPU memory, we need to use appropriate fill methods
            // This would be implemented by the memory manager
            throw std::runtime_error("GPU fill operations not implemented in base tensor");
        }
    }
    
    void zero() {
        memory_manager_->set_zero(data_, size_bytes(), memory_location_);
    }
    
    // Type conversion
    template<typename U>
    Tensor<U> cast() const {
        Tensor<U> result(shape_, memory_location_, get_memory_layout());
        if (memory_location_ == MemoryLocation::CPU) {
            for (size_t i = 0; i < total_size(); ++i) {
                result[i] = static_cast<U>(data_[i]);
            }
        } else {
            throw std::runtime_error("GPU type casting not implemented in base tensor");
        }
        return result;
    }
    
    // Slicing (basic implementation)
    Tensor<T> slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
        // Basic slicing implementation - creates a copy
        // More sophisticated implementations would create views
        std::vector<size_t> new_dims;
        for (const auto& range : ranges) {
            new_dims.push_back(range.second - range.first);
        }
        
        TensorShape new_shape(new_dims);
        Tensor<T> result(new_shape, memory_location_, get_memory_layout());
        
        // Copy sliced data (simplified implementation)
        // This would need proper multi-dimensional slicing logic
        throw std::runtime_error("Tensor slicing not fully implemented");
        
        return result;
    }
    
    // Comparison operations
    bool operator==(const Tensor<T>& other) const {
        if (shape_ != other.shape_) return false;
        
        // Ensure both tensors are on CPU for comparison
        auto this_cpu = cpu();
        auto other_cpu = other.cpu();
        
        for (size_t i = 0; i < total_size(); ++i) {
            if (this_cpu[i] != other_cpu[i]) return false;
        }
        return true;
    }
    
    bool operator!=(const Tensor<T>& other) const {
        return !(*this == other);
    }
    
    // Cloning
    Tensor<T> clone() const {
        Tensor<T> result(shape_, memory_location_, get_memory_layout());
        copy_to(result);
        return result;
    }
    
    // Copy operations
    void copy_from(const Tensor<T>& other) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for copy operation");
        }
        memory_manager_->copy(data_, other.data_, size_bytes(), 
                             memory_location_, other.memory_location_);
    }
    
    void copy_to(Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for copy operation");
        }
        memory_manager_->copy(other.data_, data_, size_bytes(), 
                             other.memory_location_, memory_location_);
    }
    
    // Memory management
    void set_memory_manager(std::shared_ptr<MemoryManager> manager) {
        memory_manager_ = manager;
    }
    
    std::shared_ptr<MemoryManager> get_memory_manager() const {
        return memory_manager_;
    }
    
    // Utility methods
    std::string to_string() const {
        return "Tensor" + shape_.to_string() + " on " + memory_location_to_string(memory_location_);
    }
    
    DataType get_data_type() const {
        return type_to_data_type_v<T>;
    }
    
    MemoryLayout get_memory_layout() const {
        // Determine layout from strides (simplified)
        return MemoryLayout::ROW_MAJOR; // Default assumption
    }

private:
    T* data_;
    TensorShape shape_;
    TensorStrides strides_;
    MemoryLocation memory_location_;
    bool owns_data_;
    std::shared_ptr<MemoryManager> memory_manager_;
    
    void allocate_memory() {
        if (total_size() > 0) {
            data_ = static_cast<T*>(memory_manager_->allocate(size_bytes(), DEFAULT_ALIGNMENT, memory_location_));
        } else {
            data_ = nullptr;
        }
    }
    
    void deallocate_memory() {
        if (data_ && owns_data_) {
            memory_manager_->deallocate(data_, memory_location_);
        }
        data_ = nullptr;
    }
    
    std::string memory_location_to_string(MemoryLocation loc) const {
        switch (loc) {
            case MemoryLocation::CPU: return "CPU";
            case MemoryLocation::GPU: return "GPU";
            case MemoryLocation::UNIFIED: return "Unified";
            default: return "Unknown";
        }
    }
};

// Type aliases for common tensor types
using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<half>;
using BFloat16Tensor = Tensor<bfloat16>;
using DoubleTensor = Tensor<double>;
using IntTensor = Tensor<int32_t>;
using LongTensor = Tensor<int64_t>;

// Factory functions
template<typename T>
Tensor<T> zeros(const TensorShape& shape, MemoryLocation location = MemoryLocation::CPU) {
    Tensor<T> tensor(shape, location);
    tensor.zero();
    return tensor;
}

template<typename T>
Tensor<T> ones(const TensorShape& shape, MemoryLocation location = MemoryLocation::CPU) {
    Tensor<T> tensor(shape, location);
    tensor.fill(T(1));
    return tensor;
}

template<typename T>
Tensor<T> full(const TensorShape& shape, const T& value, MemoryLocation location = MemoryLocation::CPU) {
    Tensor<T> tensor(shape, location);
    tensor.fill(value);
    return tensor;
}

// Tensor creation from data
template<typename T>
Tensor<T> from_data(const std::vector<T>& data, const TensorShape& shape, 
                   MemoryLocation location = MemoryLocation::CPU) {
    if (data.size() != shape.total_size()) {
        throw std::invalid_argument("Data size must match tensor shape");
    }
    
    Tensor<T> tensor(shape, location);
    if (location == MemoryLocation::CPU) {
        std::copy(data.begin(), data.end(), tensor.data());
    } else {
        auto cpu_tensor = Tensor<T>(shape, MemoryLocation::CPU);
        std::copy(data.begin(), data.end(), cpu_tensor.data());
        tensor.copy_from(cpu_tensor);
    }
    
    return tensor;
}

} // namespace attention_hpc
