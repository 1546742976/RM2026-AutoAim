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
  void configure(double shoot_ms, double control_ms, bool timing_calibrated = false)
  {
    if (!std::isfinite(shoot_ms) || !std::isfinite(control_ms) ||
        shoot_ms <= 0 || control_ms < shoot_ms)
      throw std::invalid_argument("Expected 0 < shoot_max_age_ms <= control_max_age_ms");
    std::lock_guard<std::mutex> lock(mutex_);
    shoot_age_ = shoot_ms;
    control_age_ = control_ms;
    timing_calibrated_ = timing_calibrated;
  }
  void observe(Clock::time_point time, bool valid)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (time < frame_time_) return;
    frame_time_ = time;
    pose_valid_ = valid;
  }
  void observe_pose_feedback(Clock::time_point time, bool valid)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (time < pose_feedback_time_) return;
    pose_feedback_time_ = time;
    pose_feedback_valid_ = valid;
  }
  void observe_status_feedback(Clock::time_point time, double bullet_speed)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (time < status_feedback_time_) return;
    status_feedback_time_ = time;
    status_feedback_valid_ = std::isfinite(bullet_speed) && bullet_speed > 0;
  }
  void observe_feedback(Clock::time_point time, bool pose_valid, double bullet_speed)
  {
    // A combined serial packet must not expose a mixture of old status and
    // new pose validity to the sender between two independently locked updates.
    std::lock_guard<std::mutex> lock(mutex_);
    if (time >= pose_feedback_time_) {
      pose_feedback_time_ = time;
      pose_feedback_valid_ = pose_valid;
    }
    if (time >= status_feedback_time_) {
      status_feedback_time_ = time;
      status_feedback_valid_ = std::isfinite(bullet_speed) && bullet_speed > 0;
    }
  }
  void invalidate(Clock::time_point now = Clock::now())
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_since_ = now;
    pose_valid_ = false;
    pose_feedback_valid_ = false;
    status_feedback_valid_ = false;
  }
  bool feedback_fresh(Clock::time_point time, Clock::time_point now = Clock::now()) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return feedback_fresh_locked(time, now);
  }
  ControlIntent apply(ControlIntent intent, bool enabled, Clock::time_point now = Clock::now()) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & cmd = intent.command;
    if (cmd.source_time == Clock::time_point{}) {
      // Legacy commands may track, but a frame timestamp cannot be invented
      // to renew an otherwise untraceable fire request.
      cmd.source_time = frame_time_;
      cmd.shoot = false;
    }
    const double age_ms = std::chrono::duration<double, std::milli>(now - cmd.source_time).count();
    const double pose_age_ms = std::chrono::duration<double, std::milli>(now - frame_time_).count();
    if (!enabled || !pose_valid_ || cmd.source_time < mode_since_ ||
        cmd.source_time == Clock::time_point{} || age_ms < 0 ||
        age_ms > control_age_ || pose_age_ms < 0 || pose_age_ms > control_age_)
      return {};
    if (!timing_calibrated_ || age_ms > shoot_age_ || pose_age_ms > shoot_age_ ||
        !pose_feedback_valid_ || !status_feedback_valid_ ||
        !feedback_fresh_locked(pose_feedback_time_, now) ||
        !feedback_fresh_locked(status_feedback_time_, now))
      cmd.shoot = false;
    intent.expire(now);
    return intent;
  }
private:
  mutable std::mutex mutex_;
  Clock::time_point frame_time_{}, mode_since_{};
  Clock::time_point pose_feedback_time_{}, status_feedback_time_{};
  bool pose_valid_ = false;
  bool pose_feedback_valid_ = false, status_feedback_valid_ = false;
  bool timing_calibrated_ = false;
  double shoot_age_ = 100, control_age_ = 200;

  bool feedback_fresh_locked(Clock::time_point time, Clock::time_point now) const
  {
    const double age = std::chrono::duration<double, std::milli>(now - time).count();
    return time != Clock::time_point{} && time >= mode_since_ && age >= 0 && age <= shoot_age_;
  }
};
}  // namespace io
#endif
