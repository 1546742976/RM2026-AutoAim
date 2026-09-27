#ifndef AUTO_AIM__RUNTIME_HPP
#define AUTO_AIM__RUNTIME_HPP

#include "io/frame_packet.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"

namespace auto_aim
{
// The estimator snapshot is immutable once published to a planning thread.
struct TargetSnapshot
{
  std::optional<Target> target;
  uint64_t frame_id = 0;
  std::chrono::steady_clock::time_point source_time{};
  bool pose_valid = false;
};

inline TargetSnapshot snapshot(const std::list<Target> & targets, const io::FramePacket & frame)
{
  return {targets.empty() ? std::nullopt : std::optional<Target>(targets.front()),
          frame.frame_id, frame.exposure_time, frame.pose_valid};
}

inline io::ControlIntent control_intent(const Plan & plan, uint64_t frame_id = 0)
{
  io::ControlIntent result;
  // A plan without measured gimbal feedback may track, but cannot authorize fire.
  result.command = {plan.control, false, plan.yaw, plan.pitch};
  result.command.source_time = plan.source_time;
  result.command.valid_until = plan.valid_until;
  result.command.pose_valid = plan.pose_valid;
  result.command.frame_id = frame_id;
  result.yaw_vel = plan.yaw_vel;
  result.yaw_acc = plan.yaw_acc;
  result.pitch_vel = plan.pitch_vel;
  result.pitch_acc = plan.pitch_acc;
  return result;
}

inline io::ControlIntent control_intent(
  const Plan & plan, const io::GimbalState & feedback, uint64_t frame_id = 0)
{
  auto result = control_intent(plan, frame_id);
  const auto yaw_error = std::remainder(static_cast<double>(plan.yaw) - feedback.yaw, 2 * CV_PI);
  const auto pitch_error = static_cast<double>(plan.pitch) - feedback.pitch;
  result.command.shoot = plan.control && plan.fire &&
    std::isfinite(yaw_error) && std::isfinite(pitch_error) &&
    plan.fire_yaw_tolerance > 0 && plan.fire_pitch_tolerance > 0 &&
    std::abs(yaw_error) < plan.fire_yaw_tolerance &&
    std::abs(pitch_error) < plan.fire_pitch_tolerance;
  return result;
}
}  // namespace auto_aim
#endif
