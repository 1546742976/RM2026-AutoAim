#include "shooter.hpp"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Shooter::Shooter(const std::string & config_path) : last_command_{false, false, 0, 0}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
  shoot_max_age_ms_ = yaml["shoot_max_age_ms"].as<double>(100);
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
  const auto now = std::chrono::steady_clock::now();
  if (!command.control || targets.empty() || !auto_fire_ || !command.pose_valid ||
      !aimer.fire_allowed() || !aimer.debug_aim_point.valid || !gimbal_pos.allFinite() ||
      !std::isfinite(command.yaw) || !std::isfinite(command.pitch) ||
      command.source_time == std::chrono::steady_clock::time_point{} ||
      command.source_time > now || now > command.valid_until ||
      tools::delta_time(now, command.source_time) * 1000 > shoot_max_age_ms_) {
    last_command_ = {};
    return false;
  }

  auto target_x = targets.front().ekf_x()[0];
  auto target_y = targets.front().ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;
  const auto & point = aimer.debug_aim_point.xyza;
  const double distance = point.head<2>().norm();
  const double incidence = std::max(0.0, std::cos(point[3] - std::atan2(point.y(), point.x())));
  const double half_width = targets.front().armor_type == ArmorType::big ? 0.1125 : 0.0675;
  // Configured angular tolerance can tighten, but not enlarge, the actual plate.
  const double yaw_tolerance = std::min(tolerance, std::atan2(half_width * incidence, distance));
  const double pitch_tolerance = std::min(tolerance, std::atan2(0.0275, distance));
  const bool stable = !last_command_.control ||
    std::abs(tools::limit_rad(last_command_.yaw - command.yaw)) < tolerance * 2;
  const bool fire = stable && incidence > 0.1 &&
    std::abs(tools::limit_rad(gimbal_pos[0] - command.yaw)) < yaw_tolerance &&
    std::abs(tools::limit_rad(gimbal_pos[1] - command.pitch)) < pitch_tolerance;
  last_command_ = command;
  return fire;
}

}  // namespace auto_aim
