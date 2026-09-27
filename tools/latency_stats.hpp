#ifndef TOOLS__LATENCY_STATS_HPP
#define TOOLS__LATENCY_STATS_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace tools
{
// Fixed memory cost. Quantiles are calculated only when requested, never per frame.
class LatencyStats
{
public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::size_t capacity = 512;

  struct Summary
  {
    std::size_t samples = 0;
    std::uint64_t total_samples = 0;
    double p50_ms = 0;
    double p95_ms = 0;
    double max_ms = 0;
  };

  explicit LatencyStats(bool enabled = false) : enabled_(enabled) {}

  bool enabled() const { return enabled_; }

  void record(double milliseconds)
  {
    if (!enabled_ || !std::isfinite(milliseconds) || milliseconds < 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    append(milliseconds);
  }

  // Call AFTER a successful transport write. Re-sending the same source does not
  // inflate the sample count; out-of-order and untagged commands are ignored.
  bool record_first_send(Clock::time_point source, Clock::time_point sent = Clock::now())
  {
    if (!enabled_ || source == Clock::time_point{} || sent < source) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (source <= last_source_) return false;
    last_source_ = source;
    append(std::chrono::duration<double, std::milli>(sent - source).count());
    return true;
  }

  Summary summary() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Summary result;
    result.samples = count_;
    result.total_samples = total_;
    if (!count_) return result;
    std::vector<double> sorted(values_.begin(), values_.begin() + count_);
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
      const auto index = static_cast<std::size_t>(std::ceil(p * sorted.size())) - 1;
      return sorted[index];
    };
    result.p50_ms = percentile(0.50);
    result.p95_ms = percentile(0.95);
    result.max_ms = sorted.back();
    return result;
  }

private:
  const bool enabled_;
  mutable std::mutex mutex_;
  std::array<double, capacity> values_{};
  std::size_t next_ = 0;
  std::size_t count_ = 0;
  std::uint64_t total_ = 0;
  Clock::time_point last_source_{};

  void append(double milliseconds)
  {
    values_[next_] = milliseconds;
    next_ = (next_ + 1) % capacity;
    count_ = std::min(count_ + 1, capacity);
    ++total_;
  }
};
}  // namespace tools

#endif  // TOOLS__LATENCY_STATS_HPP
