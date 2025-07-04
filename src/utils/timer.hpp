#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <functional>
#include <thread>
#include <condition_variable>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace attention_hpc {

// High-resolution time point type
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;
using Clock = std::chrono::high_resolution_clock;

// Timer state enumeration
enum class TimerState {
    STOPPED,
    RUNNING,
    PAUSED
};

// Timing statistics structure
struct TimingStats {
    Duration total_time{0};
    Duration min_time{Duration::max()};
    Duration max_time{Duration::min()};
    Duration average_time{0};
    Duration median_time{0};
    Duration std_deviation{0};
    size_t sample_count = 0;
    double samples_per_second = 0.0;
    
    // Percentile data
    Duration p95_time{0};
    Duration p99_time{0};
    Duration p999_time{0};
    
    TimingStats() = default;
    
    void reset() {
        total_time = Duration{0};
        min_time = Duration::max();
        max_time = Duration::min();
        average_time = Duration{0};
        median_time = Duration{0};
        std_deviation = Duration{0};
        sample_count = 0;
        samples_per_second = 0.0;
        p95_time = Duration{0};
        p99_time = Duration{0};
        p999_time = Duration{0};
    }
    
    void update_average() {
        if (sample_count > 0) {
            average_time = Duration{total_time.count() / sample_count};
        }
    }
    
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Timing Statistics:\n"
            << "  Samples: " << sample_count << "\n"
            << "  Total: " << total_time.count() << " ns\n"
            << "  Average: " << average_time.count() << " ns\n"
            << "  Min: " << min_time.count() << " ns\n"
            << "  Max: " << max_time.count() << " ns\n"
            << "  Median: " << median_time.count() << " ns\n"
            << "  Std Dev: " << std_deviation.count() << " ns\n"
            << "  95th percentile: " << p95_time.count() << " ns\n"
            << "  99th percentile: " << p99_time.count() << " ns\n"
            << "  99.9th percentile: " << p999_time.count() << " ns\n"
            << "  Samples/sec: " << samples_per_second;
        return oss.str();
    }
};

// Timer configuration
struct TimerConfig {
    bool enable_statistics = true;
    bool enable_history = false;
    size_t max_history_size = 1000;
    bool auto_reset_on_overflow = true;
    bool thread_safe = true;
    bool high_precision_mode = true;
    
    // Warm-up settings
    bool enable_warmup = false;
    size_t warmup_iterations = 10;
    
    // Auto-reporting settings
    bool enable_auto_report = false;
    Duration report_interval{std::chrono::seconds(60)};
    std::function<void(const std::string&, const TimingStats&)> report_callback;
    
    TimerConfig() = default;
};

// Forward declaration
class TimerManager;

// High-precision timer class
class Timer {
public:
    // Constructors
    Timer();
    explicit Timer(const std::string& name);
    explicit Timer(const TimerConfig& config);
    Timer(const std::string& name, const TimerConfig& config);
    
    // Destructor
    ~Timer();
    
    // Disable copy constructor and assignment
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    
    // Enable move constructor and assignment
    Timer(Timer&& other) noexcept;
    Timer& operator=(Timer&& other) noexcept;
    
    // Basic timer operations
    void start();
    void stop();
    void pause();
    void resume();
    void reset();
    void restart();
    
    // Timing queries
    Duration elapsed() const;
    Duration lap();
    Duration total_elapsed() const;
    TimerState get_state() const;
    bool is_running() const;
    bool is_paused() const;
    bool is_stopped() const;
    
    // Statistics and history
    const TimingStats& get_stats() const;
    void update_stats();
    void clear_stats();
    void add_sample(Duration duration);
    
    // History management
    const std::vector<Duration>& get_history() const;
    void clear_history();
    void enable_history(bool enable, size_t max_size = 1000);
    
    // Configuration
    void set_config(const TimerConfig& config);
    const TimerConfig& get_config() const;
    void set_name(const std::string& name);
    const std::string& get_name() const;
    
    // Thread safety
    void enable_thread_safety(bool enable = true);
    bool is_thread_safe() const;
    
    // Utility methods
    double to_seconds() const;
    double to_milliseconds() const;
    double to_microseconds() const;
    long long to_nanoseconds() const;
    
    std::string to_string() const;
    std::string format_duration(Duration duration) const;
    
    // Static utility methods
    static TimePoint now();
    static Duration since(TimePoint start_time);
    static std::string format_duration_static(Duration duration);
    static Timer create_named_timer(const std::string& name);
    
    // Percentile calculations
    Duration get_percentile(double percentile) const;
    std::vector<Duration> get_percentiles(const std::vector<double>& percentiles) const;

private:
    std::string name_;
    TimerConfig config_;
    mutable std::mutex timer_mutex_;
    
    // Timer state
    std::atomic<TimerState> state_;
    TimePoint start_time_;
    TimePoint pause_time_;
    Duration accumulated_time_;
    Duration total_time_;
    
    // Statistics
    TimingStats stats_;
    std::vector<Duration> sample_history_;
    mutable std::mutex stats_mutex_;
    
    // Auto-reporting
    std::unique_ptr<std::thread> report_thread_;
    std::atomic<bool> stop_reporting_;
    std::condition_variable report_condition_;
    std::mutex report_mutex_;
    
    // Internal methods
    void initialize();
    void cleanup();
    void start_internal();
    void stop_internal();
    void update_stats_internal();
    void calculate_percentiles();
    void start_auto_reporting();
    void stop_auto_reporting();
    void auto_report_worker();
    
    // Thread safety helpers
    template<typename Func>
    auto with_lock(Func&& func) const -> decltype(func()) {
        if (config_.thread_safe) {
            std::lock_guard<std::mutex> lock(timer_mutex_);
            return func();
        } else {
            return func();
        }
    }
    
    template<typename Func>
    auto with_stats_lock(Func&& func) const -> decltype(func()) {
        if (config_.thread_safe) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            return func();
        } else {
            return func();
        }
    }
};

// RAII timer for automatic timing
class ScopedTimer {
public:
    explicit ScopedTimer(Timer& timer);
    ScopedTimer(Timer& timer, const std::string& scope_name);
    ~ScopedTimer();
    
    // Disable copy and move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;
    
    // Manual control
    void finish();
    Duration elapsed() const;
    bool is_finished() const;

private:
    Timer& timer_;
    std::string scope_name_;
    bool finished_;
    TimePoint start_time_;
};

// Timer pool for managing multiple timers
class TimerPool {
public:
    TimerPool();
    explicit TimerPool(const TimerConfig& default_config);
    ~TimerPool();
    
    // Timer management
    Timer* create_timer(const std::string& name);
    Timer* create_timer(const std::string& name, const TimerConfig& config);
    Timer* get_timer(const std::string& name);
    bool remove_timer(const std::string& name);
    void clear_timers();
    
    // Batch operations
    void start_all();
    void stop_all();
    void reset_all();
    void clear_all_stats();
    
    // Queries
    std::vector<std::string> get_timer_names() const;
    size_t get_timer_count() const;
    bool has_timer(const std::string& name) const;
    
    // Statistics
    std::unordered_map<std::string, TimingStats> get_all_stats() const;
    TimingStats get_aggregated_stats() const;
    void print_all_stats() const;
    
    // Configuration
    void set_default_config(const TimerConfig& config);
    const TimerConfig& get_default_config() const;

private:
    TimerConfig default_config_;
    std::unordered_map<std::string, std::unique_ptr<Timer>> timers_;
    mutable std::mutex pool_mutex_;
};

// Global timer manager
class TimerManager {
public:
    static TimerManager& get_instance();
    
    // Timer operations
    void start_timer(const std::string& name);
    void stop_timer(const std::string& name);
    void reset_timer(const std::string& name);
    Duration get_elapsed(const std::string& name);
    
    // Timer creation
    Timer* create_timer(const std::string& name);
    Timer* create_timer(const std::string& name, const TimerConfig& config);
    Timer* get_timer(const std::string& name);
    
    // Statistics
    TimingStats get_stats(const std::string& name);
    std::unordered_map<std::string, TimingStats> get_all_stats();
    void print_all_stats();
    void clear_all_stats();
    
    // Configuration
    void set_default_config(const TimerConfig& config);
    void enable_global_thread_safety(bool enable = true);
    
    // Cleanup
    void shutdown();

private:
    TimerManager() = default;
    ~TimerManager() = default;
    
    TimerManager(const TimerManager&) = delete;
    TimerManager& operator=(const TimerManager&) = delete;
    
    std::unique_ptr<TimerPool> timer_pool_;
    TimerConfig default_config_;
    mutable std::mutex manager_mutex_;
    bool initialized_ = false;
    
    void initialize();
};

// Benchmark timer for performance testing
class BenchmarkTimer {
public:
    explicit BenchmarkTimer(const std::string& name);
    BenchmarkTimer(const std::string& name, size_t iterations);
    ~BenchmarkTimer() = default;
    
    // Benchmark execution
    template<typename Func>
    TimingStats benchmark(Func&& func, size_t iterations = 0) {
        if (iterations == 0) {
            iterations = default_iterations_;
        }
        
        Timer timer(name_ + "_benchmark");
        timer.enable_history(true, iterations);
        
        // Warm-up phase
        if (warmup_iterations_ > 0) {
            for (size_t i = 0; i < warmup_iterations_; ++i) {
                func();
            }
        }
        
        // Actual benchmarking
        for (size_t i = 0; i < iterations; ++i) {
            timer.restart();
            func();
            timer.stop();
        }
        
        timer.update_stats();
        return timer.get_stats();
    }
    
    template<typename Func>
    TimingStats benchmark_duration(Func&& func, Duration target_duration) {
        Timer timer(name_ + "_duration_benchmark");
        timer.enable_history(true);
        
        auto start_time = Clock::now();
        size_t iterations = 0;
        
        while (Clock::now() - start_time < target_duration) {
            timer.restart();
            func();
            timer.stop();
            iterations++;
        }
        
        timer.update_stats();
        return timer.get_stats();
    }
    
    // Configuration
    void set_warmup_iterations(size_t iterations);
    void set_default_iterations(size_t iterations);
    void set_name(const std::string& name);
    
    // Comparative benchmarking
    template<typename Func1, typename Func2>
    std::pair<TimingStats, TimingStats> compare(Func1&& func1, Func2&& func2, size_t iterations = 0) {
        if (iterations == 0) {
            iterations = default_iterations_;
        }
        
        auto stats1 = benchmark(std::forward<Func1>(func1), iterations);
        auto stats2 = benchmark(std::forward<Func2>(func2), iterations);
        
        return std::make_pair(stats1, stats2);
    }

private:
    std::string name_;
    size_t default_iterations_ = 1000;
    size_t warmup_iterations_ = 10;
};

// Profiling timer with automatic scope detection
class ProfileTimer {
public:
    ProfileTimer();
    ~ProfileTimer() = default;
    
    // Profiling control
    void start_profiling();
    void stop_profiling();
    void reset_profiling();
    bool is_profiling() const;
    
    // Scope timing
    void enter_scope(const std::string& scope_name);
    void exit_scope(const std::string& scope_name);
    Duration get_scope_time(const std::string& scope_name) const;
    
    // Hierarchical timing
    void begin_frame(const std::string& frame_name = "frame");
    void end_frame();
    std::unordered_map<std::string, Duration> get_frame_times() const;
    
    // Statistics
    std::unordered_map<std::string, TimingStats> get_all_scope_stats() const;
    void print_profile_report() const;
    void save_profile_report(const std::string& filename) const;
    
    // Thread-local profiling
    void set_thread_local_profiling(bool enable);
    std::unordered_map<std::thread::id, std::unordered_map<std::string, TimingStats>> get_thread_stats() const;

private:
    struct ScopeInfo {
        TimePoint start_time;
        Duration accumulated_time{0};
        size_t call_count = 0;
        std::vector<Duration> samples;
    };
    
    mutable std::mutex profile_mutex_;
    std::atomic<bool> profiling_active_;
    std::unordered_map<std::string, ScopeInfo> scopes_;
    std::stack<std::string> scope_stack_;
    
    // Thread-local data
    thread_local static std::unordered_map<std::string, ScopeInfo> thread_local_scopes_;
    bool thread_local_enabled_ = false;
    
    void update_scope_stats(const std::string& scope_name, Duration duration);
};

// Utility macros for convenient timing
#define TIMER_START(name) \
    attention_hpc::TimerManager::get_instance().start_timer(name)

#define TIMER_STOP(name) \
    attention_hpc::TimerManager::get_instance().stop_timer(name)

#define TIMER_ELAPSED(name) \
    attention_hpc::TimerManager::get_instance().get_elapsed(name)

#define TIMER_SCOPE(name) \
    attention_hpc::Timer __scope_timer(name); \
    attention_hpc::ScopedTimer __scoped_timer(__scope_timer, name)

#define TIMER_FUNCTION() \
    TIMER_SCOPE(__FUNCTION__)

#define TIMER_BENCHMARK(name, func, iterations) \
    attention_hpc::BenchmarkTimer __benchmark(name); \
    auto __stats = __benchmark.benchmark(func, iterations)

// Utility functions
namespace timer_utils {
    // Duration conversion utilities
    double to_seconds(Duration duration);
    double to_milliseconds(Duration duration);
    double to_microseconds(Duration duration);
    long long to_nanoseconds(Duration duration);
    
    // Formatting utilities
    std::string format_duration(Duration duration, int precision = 3);
    std::string format_duration_auto(Duration duration);
    std::string format_stats(const TimingStats& stats);
    
    // Statistical utilities
    Duration calculate_median(std::vector<Duration> samples);
    Duration calculate_percentile(const std::vector<Duration>& samples, double percentile);
    Duration calculate_standard_deviation(const std::vector<Duration>& samples, Duration mean);
    double calculate_samples_per_second(size_t sample_count, Duration total_time);
    
    // Timing utilities
    void sleep_for_duration(Duration duration);
    void busy_wait_for_duration(Duration duration);
    TimePoint get_monotonic_time();
    
    // Timer creation helpers
    std::unique_ptr<Timer> create_high_precision_timer(const std::string& name);
    std::unique_ptr<Timer> create_statistics_timer(const std::string& name, size_t history_size = 1000);
    std::unique_ptr<BenchmarkTimer> create_benchmark_timer(const std::string& name, size_t iterations = 1000);
    
    // Performance analysis
    struct PerformanceReport {
        std::string timer_name;
        TimingStats stats;
        std::vector<std::string> recommendations;
        double efficiency_score;
    };
    
    PerformanceReport analyze_timer_performance(const Timer& timer);
    std::vector<PerformanceReport> analyze_all_timers();
    void print_performance_report(const PerformanceReport& report);
    
    // Timer synchronization utilities
    void synchronize_timers(const std::vector<Timer*>& timers);
    Duration get_max_elapsed(const std::vector<Timer*>& timers);
    Duration get_min_elapsed(const std::vector<Timer*>& timers);
    Duration get_average_elapsed(const std::vector<Timer*>& timers);
}

// Exception class for timer-related errors
class TimerException : public std::runtime_error {
public:
    explicit TimerException(const std::string& message)
        : std::runtime_error("Timer error: " + message) {}
};

} // namespace attention_hpc
