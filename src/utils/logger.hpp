#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <unordered_map>

namespace attention_hpc {

// Log levels enumeration
enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5,
    OFF = 6
};

// Log message structure
struct LogMessage {
    LogLevel level;
    std::string message;
    std::string function_name;
    std::string file_name;
    int line_number;
    std::thread::id thread_id;
    std::chrono::system_clock::time_point timestamp;
    std::string category;
    
    LogMessage() : level(LogLevel::INFO), line_number(0), thread_id(std::this_thread::get_id()) {
        timestamp = std::chrono::system_clock::now();
    }
    
    LogMessage(LogLevel lvl, const std::string& msg, const std::string& func = "", 
               const std::string& file = "", int line = 0, const std::string& cat = "")
        : level(lvl), message(msg), function_name(func), file_name(file), 
          line_number(line), thread_id(std::this_thread::get_id()), category(cat) {
        timestamp = std::chrono::system_clock::now();
    }
};

// Forward declaration
class LogSink;

// Logger configuration
struct LoggerConfig {
    LogLevel min_level = LogLevel::INFO;
    bool enable_console_output = true;
    bool enable_file_output = false;
    bool enable_async_logging = true;
    bool enable_thread_id = true;
    bool enable_function_name = true;
    bool enable_file_line = true;
    bool enable_timestamp = true;
    bool enable_performance_logging = false;
    
    std::string log_file_path = "attention_hpc.log";
    std::string timestamp_format = "%Y-%m-%d %H:%M:%S";
    std::string message_format = "[{timestamp}] [{level}] [{thread}] {message}";
    
    size_t max_queue_size = 10000;
    size_t max_file_size = 100 * 1024 * 1024; // 100MB
    size_t max_backup_files = 5;
    
    bool auto_flush = false;
    std::chrono::milliseconds flush_interval{1000};
    
    // Performance logging specific
    bool profile_memory_usage = true;
    bool profile_execution_time = true;
    std::chrono::milliseconds performance_log_interval{5000};
};

// Abstract base class for log sinks
class LogSink {
public:
    virtual ~LogSink() = default;
    virtual void write(const LogMessage& message) = 0;
    virtual void flush() = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;
    virtual std::string get_name() const = 0;
};

// Console log sink
class ConsoleSink : public LogSink {
public:
    ConsoleSink(bool use_colors = true);
    ~ConsoleSink() override = default;
    
    void write(const LogMessage& message) override;
    void flush() override;
    void close() override;
    bool is_open() const override;
    std::string get_name() const override;
    
    void set_color_enabled(bool enabled);
    bool is_color_enabled() const;

private:
    bool use_colors_;
    mutable std::mutex console_mutex_;
    
    std::string get_color_code(LogLevel level) const;
    std::string get_reset_color_code() const;
};

// File log sink with rotation support
class FileSink : public LogSink {
public:
    explicit FileSink(const std::string& file_path, size_t max_size = 100 * 1024 * 1024, 
                     size_t max_backups = 5);
    ~FileSink() override;
    
    void write(const LogMessage& message) override;
    void flush() override;
    void close() override;
    bool is_open() const override;
    std::string get_name() const override;
    
    void set_max_file_size(size_t max_size);
    void set_max_backup_files(size_t max_backups);
    size_t get_current_file_size() const;

private:
    std::string file_path_;
    size_t max_file_size_;
    size_t max_backup_files_;
    std::unique_ptr<std::ofstream> file_stream_;
    mutable std::mutex file_mutex_;
    
    void rotate_file();
    void open_file();
    std::string get_backup_filename(size_t index) const;
};

// Memory log sink (for testing and debugging)
class MemorySink : public LogSink {
public:
    explicit MemorySink(size_t max_messages = 1000);
    ~MemorySink() override = default;
    
    void write(const LogMessage& message) override;
    void flush() override;
    void close() override;
    bool is_open() const override;
    std::string get_name() const override;
    
    std::vector<LogMessage> get_messages() const;
    void clear_messages();
    size_t get_message_count() const;

private:
    mutable std::mutex memory_mutex_;
    std::vector<LogMessage> messages_;
    size_t max_messages_;
};

// Performance profiler for logging
class PerformanceProfiler {
public:
    PerformanceProfiler();
    ~PerformanceProfiler() = default;
    
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    void record_memory_usage(const std::string& name, size_t bytes);
    void record_custom_metric(const std::string& name, double value, const std::string& unit = "");
    
    std::string get_performance_summary() const;
    void reset_metrics();
    void enable_auto_logging(bool enable);

private:
    struct TimerInfo {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        std::chrono::nanoseconds total_duration{0};
        size_t call_count = 0;
        bool is_running = false;
    };
    
    struct MemoryInfo {
        size_t current_bytes = 0;
        size_t peak_bytes = 0;
        size_t total_allocated = 0;
        size_t allocation_count = 0;
    };
    
    struct CustomMetric {
        double value = 0.0;
        std::string unit;
        size_t sample_count = 0;
        double min_value = std::numeric_limits<double>::max();
        double max_value = std::numeric_limits<double>::lowest();
        double sum_value = 0.0;
    };
    
    mutable std::mutex profiler_mutex_;
    std::unordered_map<std::string, TimerInfo> timers_;
    std::unordered_map<std::string, MemoryInfo> memory_usage_;
    std::unordered_map<std::string, CustomMetric> custom_metrics_;
    bool auto_logging_enabled_;
};

// Main Logger class
class Logger {
public:
    explicit Logger(const LoggerConfig& config = LoggerConfig{});
    ~Logger();
    
    // Disable copy constructor and assignment
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // Enable move constructor and assignment
    Logger(Logger&& other) noexcept;
    Logger& operator=(Logger&& other) noexcept;
    
    // Basic logging methods
    void log(LogLevel level, const std::string& message, const std::string& category = "");
    void trace(const std::string& message, const std::string& category = "");
    void debug(const std::string& message, const std::string& category = "");
    void info(const std::string& message, const std::string& category = "");
    void warning(const std::string& message, const std::string& category = "");
    void error(const std::string& message, const std::string& category = "");
    void critical(const std::string& message, const std::string& category = "");
    
    // Formatted logging methods
    template<typename... Args>
    void logf(LogLevel level, const std::string& format, Args&&... args) {
        log(level, format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void tracef(const std::string& format, Args&&... args) {
        trace(format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void debugf(const std::string& format, Args&&... args) {
        debug(format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void infof(const std::string& format, Args&&... args) {
        info(format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void warningf(const std::string& format, Args&&... args) {
        warning(format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void errorf(const std::string& format, Args&&... args) {
        error(format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void criticalf(const std::string& format, Args&&... args) {
        critical(format_string(format, std::forward<Args>(args)...));
    }
    
    // Advanced logging with source location
    void log_with_location(LogLevel level, const std::string& message, 
                          const std::string& function, const std::string& file, int line,
                          const std::string& category = "");
    
    // Sink management
    void add_sink(std::shared_ptr<LogSink> sink);
    void remove_sink(const std::string& sink_name);
    void clear_sinks();
    std::vector<std::string> get_sink_names() const;
    
    // Configuration
    void set_config(const LoggerConfig& config);
    LoggerConfig get_config() const;
    void set_log_level(LogLevel level);
    LogLevel get_log_level() const;
    void set_message_format(const std::string& format);
    std::string get_message_format() const;
    
    // Control methods
    void flush();
    void close();
    bool is_async_enabled() const;
    void enable_async(bool enable);
    
    // Performance logging
    void enable_performance_logging(bool enable);
    bool is_performance_logging_enabled() const;
    PerformanceProfiler& get_profiler();
    const PerformanceProfiler& get_profiler() const;
    
    // Category filtering
    void enable_category(const std::string& category);
    void disable_category(const std::string& category);
    void clear_category_filters();
    bool is_category_enabled(const std::string& category) const;
    
    // Statistics
    struct LogStats {
        size_t total_messages = 0;
        size_t messages_by_level[static_cast<int>(LogLevel::OFF)] = {0};
        size_t dropped_messages = 0;
        size_t queue_size = 0;
        size_t max_queue_size = 0;
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point last_message_time;
    };
    
    LogStats get_statistics() const;
    void reset_statistics();
    
    // Thread-safe singleton access
    static Logger& get_instance();
    static void set_global_config(const LoggerConfig& config);

private:
    LoggerConfig config_;
    std::vector<std::shared_ptr<LogSink>> sinks_;
    mutable std::mutex sinks_mutex_;
    
    // Async logging
    std::unique_ptr<std::thread> logging_thread_;
    std::queue<LogMessage> message_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::atomic<bool> shutdown_requested_;
    
    // Performance profiling
    std::unique_ptr<PerformanceProfiler> profiler_;
    std::unique_ptr<std::thread> performance_thread_;
    
    // Category filtering
    std::unordered_map<std::string, bool> category_filters_;
    mutable std::mutex category_mutex_;
    
    // Statistics
    mutable LogStats statistics_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void process_log_message(const LogMessage& message);
    void async_logging_worker();
    void performance_logging_worker();
    bool should_log(LogLevel level, const std::string& category) const;
    std::string format_message(const LogMessage& message) const;
    std::string format_timestamp(const std::chrono::system_clock::time_point& time) const;
    std::string level_to_string(LogLevel level) const;
    
    template<typename... Args>
    std::string format_string(const std::string& format, Args&&... args) {
        std::stringstream ss;
        format_string_impl(ss, format, std::forward<Args>(args)...);
        return ss.str();
    }
    
    template<typename T, typename... Args>
    void format_string_impl(std::stringstream& ss, const std::string& format, T&& value, Args&&... args) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            ss << format.substr(0, pos) << std::forward<T>(value);
            format_string_impl(ss, format.substr(pos + 2), std::forward<Args>(args)...);
        } else {
            ss << format;
        }
    }
    
    void format_string_impl(std::stringstream& ss, const std::string& format) {
        ss << format;
    }
    
    void start_async_logging();
    void stop_async_logging();
    void start_performance_logging();
    void stop_performance_logging();
    void update_statistics(const LogMessage& message);
};

// Convenience macros for logging with source location
#define LOG_TRACE(logger, message) \
    (logger).log_with_location(attention_hpc::LogLevel::TRACE, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_DEBUG(logger, message) \
    (logger).log_with_location(attention_hpc::LogLevel::DEBUG, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_INFO(logger, message) \
    (logger).log_with_location(attention_hpc::LogLevel::INFO, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_WARNING(logger, message) \
    (logger).log_with_location(attention_hpc::LogLevel::WARNING, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_ERROR(logger, message) \
    (logger).log_with_location(attention_hpc::LogLevel::ERROR, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_CRITICAL(logger, message) \
    (logger).log_with_location(attention_hpc::LogLevel::CRITICAL, message, __FUNCTION__, __FILE__, __LINE__)

// Formatted logging macros
#define LOG_TRACEF(logger, format, ...) \
    (logger).logf(attention_hpc::LogLevel::TRACE, format, __VA_ARGS__)

#define LOG_DEBUGF(logger, format, ...) \
    (logger).logf(attention_hpc::LogLevel::DEBUG, format, __VA_ARGS__)

#define LOG_INFOF(logger, format, ...) \
    (logger).logf(attention_hpc::LogLevel::INFO, format, __VA_ARGS__)

#define LOG_WARNINGF(logger, format, ...) \
    (logger).logf(attention_hpc::LogLevel::WARNING, format, __VA_ARGS__)

#define LOG_ERRORF(logger, format, ...) \
    (logger).logf(attention_hpc::LogLevel::ERROR, format, __VA_ARGS__)

#define LOG_CRITICALF(logger, format, ...) \
    (logger).logf(attention_hpc::LogLevel::CRITICAL, format, __VA_ARGS__)

// Global logger convenience macros
#define GLOBAL_LOG_TRACE(message) LOG_TRACE(attention_hpc::Logger::get_instance(), message)
#define GLOBAL_LOG_DEBUG(message) LOG_DEBUG(attention_hpc::Logger::get_instance(), message)
#define GLOBAL_LOG_INFO(message) LOG_INFO(attention_hpc::Logger::get_instance(), message)
#define GLOBAL_LOG_WARNING(message) LOG_WARNING(attention_hpc::Logger::get_instance(), message)
#define GLOBAL_LOG_ERROR(message) LOG_ERROR(attention_hpc::Logger::get_instance(), message)
#define GLOBAL_LOG_CRITICAL(message) LOG_CRITICAL(attention_hpc::Logger::get_instance(), message)

// Performance profiling convenience macros
#define PROFILE_FUNCTION(logger) \
    attention_hpc::ScopedTimer __timer((logger).get_profiler(), __FUNCTION__)

#define PROFILE_SCOPE(logger, name) \
    attention_hpc::ScopedTimer __timer((logger).get_profiler(), name)

// Scoped timer for automatic performance measurement
class ScopedTimer {
public:
    ScopedTimer(PerformanceProfiler& profiler, const std::string& name)
        : profiler_(profiler), name_(name) {
        profiler_.start_timer(name_);
    }
    
    ~ScopedTimer() {
        profiler_.end_timer(name_);
    }

private:
    PerformanceProfiler& profiler_;
    std::string name_;
};

// Utility functions
namespace logger_utils {
    std::string level_to_string(LogLevel level);
    LogLevel string_to_level(const std::string& level_str);
    std::string get_current_timestamp(const std::string& format = "%Y-%m-%d %H:%M:%S");
    std::string get_thread_id_string();
    std::string escape_string(const std::string& str);
    
    // Configuration helpers
    LoggerConfig load_config_from_file(const std::string& config_file);
    void save_config_to_file(const LoggerConfig& config, const std::string& config_file);
    LoggerConfig get_default_config();
    LoggerConfig get_performance_config();
    LoggerConfig get_debug_config();
    LoggerConfig get_production_config();
    
    // Log analysis utilities
    struct LogAnalysis {
        size_t total_lines = 0;
        std::unordered_map<LogLevel, size_t> level_counts;
        std::vector<std::string> error_messages;
        std::vector<std::string> warning_messages;
        std::chrono::system_clock::time_point first_timestamp;
        std::chrono::system_clock::time_point last_timestamp;
    };
    
    LogAnalysis analyze_log_file(const std::string& log_file);
    std::vector<LogMessage> parse_log_file(const std::string& log_file);
    void filter_log_messages(std::vector<LogMessage>& messages, LogLevel min_level);
    void filter_log_messages_by_category(std::vector<LogMessage>& messages, const std::string& category);
}

} // namespace attention_hpc
