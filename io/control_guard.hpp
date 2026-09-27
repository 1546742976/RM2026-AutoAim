#ifndef IO__CONTROL_GUARD_HPP
#define IO__CONTROL_GUARD_HPP

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include "io/command.hpp"

namespace io
{
// Enforced again at the transport boundary, even if a producer stalls.
class ControlGuard
{
public:
  using Clock = std::chrono::steady_clock;
  void configure(double shoot_ms, double control_ms)
  {
    if (!std::isfinite(shoot_ms) || !std::isfinite(control_ms) ||
        shoot_ms <= 0 || control_ms < shoot_ms)
      throw std::invalid_argument("Expected 0 < shoot_max_age_ms <= control_max_age_ms");
    shoot_age_ = shoot_ms;
    control_age_ = control_ms;
  }
  void observe(Clock::time_point time, bool valid)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (time < frame_time_) return;
    frame_time_ = time;
    pose_valid_ = valid;
  }
  void invalidate(Clock::time_point now = Clock::now())
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_since_ = now;
    pose_valid_ = false;
  }
  bool feedback_fresh(Clock::time_point time, Clock::time_point now = Clock::now()) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const double age = std::chrono::duration<double, std::milli>(now - time).count();
    return time != Clock::time_point{} && time >= mode_since_ && age >= 0 && age <= shoot_age_;
  }
  ControlIntent apply(ControlIntent intent, bool enabled, Clock::time_point now = Clock::now()) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & cmd = intent.command;
    if (cmd.source_time == Clock::time_point{}) cmd.source_time = frame_time_;
    const double age_ms = std::chrono::duration<double, std::milli>(now - cmd.source_time).count();
    const double pose_age_ms = std::chrono::duration<double, std::milli>(now - frame_time_).count();
    if (!enabled || !pose_valid_ || cmd.source_time < mode_since_ ||
        cmd.source_time == Clock::time_point{} || age_ms < 0 ||
        age_ms > control_age_ || pose_age_ms < 0 || pose_age_ms > control_age_)
      return {};
    if (age_ms > shoot_age_ || pose_age_ms > shoot_age_) cmd.shoot = false;
    intent.expire(now);
    return intent;
  }
private:
  mutable std::mutex mutex_;
  Clock::time_point frame_time_{}, mode_since_{};
  bool pose_valid_ = false;
  double shoot_age_ = 100, control_age_ = 200;
};
}  // namespace io
#endif
