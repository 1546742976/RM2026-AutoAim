#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

#include <chrono>
#include <cmath>
#include <cstdint>

namespace io
{
struct Command
{
  bool control = false;
  bool shoot = false;
  double yaw = 0;
  double pitch = 0;
  double horizon_distance = 0;  //无人机专有
  std::chrono::steady_clock::time_point source_time{};
  std::chrono::steady_clock::time_point valid_until{};
  bool pose_valid = true;
  uint64_t frame_id = 0;
};

// Metadata is internal only; no serial/CAN wire format changes.
struct ControlIntent
{
  Command command;
  double yaw_vel = 0, yaw_acc = 0, pitch_vel = 0, pitch_acc = 0;

  void expire(std::chrono::steady_clock::time_point now)
  {
    const bool finite = std::isfinite(command.yaw) && std::isfinite(command.pitch) &&
      std::isfinite(command.horizon_distance) && std::isfinite(yaw_vel) &&
      std::isfinite(yaw_acc) && std::isfinite(pitch_vel) && std::isfinite(pitch_acc);
    if (!finite || !command.pose_valid ||
        (command.valid_until != std::chrono::steady_clock::time_point{} && now > command.valid_until)) {
      *this = {};
    }
    if (!command.control || command.source_time == std::chrono::steady_clock::time_point{} ||
        command.source_time > now) command.shoot = false;
  }
};

}  // namespace io

#endif  // IO__COMMAND_HPP
