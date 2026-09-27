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
  enum Reason : uint32_t {
    disabled = 1u << 0, missing_source = 1u << 1, previous_mode = 1u << 2,
    future_time = 1u << 3, control_expired = 1u << 4, pose_invalid = 1u << 5,
    uncalibrated = 1u << 6, shoot_expired = 1u << 7,
    pose_feedback_invalid = 1u << 8, status_feedback_invalid = 1u << 9,
    pose_feedback_stale = 1u << 10, status_feedback_stale = 1u << 11,
    nonfinite = 1u << 12, deadline_expired = 1u << 13
  };
  struct Evaluation {
    ControlIntent intent;
    uint32_t reasons = 0;
  };
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
    return evaluate(intent, enabled, now).intent;
  }
  Evaluation evaluate(ControlIntent intent, bool enabled, Clock::time_point now = Clock::now()) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & cmd = intent.command;
    uint32_t reasons = intent.inhibit_reasons;
    if (cmd.source_time == Clock::time_point{}) {
      reasons |= missing_source;
      // Legacy commands may track, but a frame timestamp cannot be invented
      // to renew an otherwise untraceable fire request.
      cmd.source_time = frame_time_;
      cmd.shoot = false;
    }
    const double age_ms = std::chrono::duration<double, std::milli>(now - cmd.source_time).count();
    const double pose_age_ms = std::chrono::duration<double, std::milli>(now - frame_time_).count();
    if (!enabled) reasons |= disabled;
    if (!pose_valid_ || !cmd.pose_valid) reasons |= pose_invalid;
    if (cmd.source_time < mode_since_) reasons |= previous_mode;
    if (age_ms < 0 || pose_age_ms < 0) reasons |= future_time;
    if (age_ms > control_age_ || pose_age_ms > control_age_) reasons |= control_expired;
    if (!timing_calibrated_) reasons |= uncalibrated;
    if (age_ms > shoot_age_ || pose_age_ms > shoot_age_) reasons |= shoot_expired;
    if (!pose_feedback_valid_) reasons |= pose_feedback_invalid;
    if (!status_feedback_valid_) reasons |= status_feedback_invalid;
    if (!feedback_fresh_locked(pose_feedback_time_, now)) reasons |= pose_feedback_stale;
    if (!feedback_fresh_locked(status_feedback_time_, now)) reasons |= status_feedback_stale;
    if (!std::isfinite(cmd.yaw) || !std::isfinite(cmd.pitch) ||
        !std::isfinite(cmd.horizon_distance) || !std::isfinite(intent.yaw_vel) ||
        !std::isfinite(intent.yaw_acc) || !std::isfinite(intent.pitch_vel) ||
        !std::isfinite(intent.pitch_acc)) reasons |= nonfinite;
    if (cmd.valid_until != Clock::time_point{} && now > cmd.valid_until)
      reasons |= deadline_expired;
    // Retain only provenance on a stop, never the rejected numerical command.
    const auto source_time = cmd.source_time;
    const auto frame_id = cmd.frame_id;
    if (!enabled || !pose_valid_ || cmd.source_time < mode_since_ ||
        cmd.source_time == Clock::time_point{} || age_ms < 0 ||
        age_ms > control_age_ || pose_age_ms < 0 || pose_age_ms > control_age_)
      intent = {};
    else if (!timing_calibrated_ || age_ms > shoot_age_ || pose_age_ms > shoot_age_ ||
        !pose_feedback_valid_ || !status_feedback_valid_ ||
        !feedback_fresh_locked(pose_feedback_time_, now) ||
        !feedback_fresh_locked(status_feedback_time_, now))
      cmd.shoot = false;
    intent.expire(now);
    intent.command.source_time = source_time;
    intent.command.frame_id = frame_id;
    intent.inhibit_reasons = reasons;
    return {intent, reasons};
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
