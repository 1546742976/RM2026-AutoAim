#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control = false;
  bool fire = false;
  float target_yaw = 0;
  float target_pitch = 0;
  float yaw = 0;
  float yaw_vel = 0;
  float yaw_acc = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float pitch_acc = 0;
  std::chrono::steady_clock::time_point source_time{};
  std::chrono::steady_clock::time_point valid_until{};
  bool pose_valid = false;
  float fire_yaw_tolerance = 0;
  float fire_pitch_tolerance = 0;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza = Eigen::Vector4d::Zero();
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed,
            std::chrono::steady_clock::time_point now);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double shoot_max_age_ms_ = 100, control_max_age_ms_ = 200;
  bool auto_fire_ = false;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;
  Plan solve(Target target, double bullet_speed, std::chrono::steady_clock::time_point now);

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
