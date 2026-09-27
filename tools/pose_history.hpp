#ifndef TOOLS__POSE_HISTORY_HPP
#define TOOLS__POSE_HISTORY_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>

namespace tools
{
// Queries do not consume samples: every consumer sees the same pose for a frame.
class PoseHistory
{
public:
  using Clock = std::chrono::steady_clock;
  struct Sample { Eigen::Quaterniond q; Clock::time_point time; };

  void push(Eigen::Quaterniond q, Clock::time_point time)
  {
    if (!q.coeffs().allFinite() || q.norm() < 1e-9) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!samples_.empty() && time <= samples_.back().time) return;
    samples_.push_back({q.normalized(), time});
    while (samples_.size() > 4096 ||
           (samples_.size() > 2 && time - samples_.front().time > std::chrono::seconds(2)))
      samples_.pop_front();
    changed_.notify_all();
  }

  std::optional<Sample> at(Clock::time_point time,
                          std::chrono::milliseconds wait = std::chrono::milliseconds(5)) const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait_for(lock, wait, [&] {
      return !samples_.empty() && samples_.back().time >= time;
    });
    if (samples_.empty() || time < samples_.front().time || time > samples_.back().time)
      return std::nullopt;
    for (std::size_t i = 0; i < samples_.size(); ++i) {
      const auto & b = samples_[i];
      if (b.time == time) return b;
      if (b.time < time || i == 0) continue;
      const auto & a = samples_[i - 1];
      if (b.time - a.time > std::chrono::milliseconds(40)) return std::nullopt;
      const double k = std::chrono::duration<double>(time - a.time).count() /
                       std::chrono::duration<double>(b.time - a.time).count();
      return Sample{a.q.slerp(k, b.q).normalized(), time};
    }
    return std::nullopt;
  }

  static Eigen::Quaterniond invalid()
  {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan};
  }
  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
  }

private:
  mutable std::mutex mutex_;
  mutable std::condition_variable changed_;
  std::deque<Sample> samples_;
};
}  // namespace tools
#endif
