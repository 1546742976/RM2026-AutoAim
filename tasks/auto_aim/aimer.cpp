#include "aimer.hpp"
#include "interception.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;    // degree to rad
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // degree to rad
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // degree to rad
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  if (yaml["actuation_delay_s"])
    high_speed_delay_time_ = low_speed_delay_time_ = yaml["actuation_delay_s"].as<double>();
  shoot_max_age_ms_ = yaml["shoot_max_age_ms"].as<double>(100);
  control_max_age_ms_ = yaml["control_max_age_ms"].as<double>(200);
  if (!std::isfinite(high_speed_delay_time_) || !std::isfinite(low_speed_delay_time_) ||
      high_speed_delay_time_ < 0 || low_speed_delay_time_ < 0 ||
      !(shoot_max_age_ms_ > 0 && control_max_age_ms_ >= shoot_max_age_ms_))
    throw std::invalid_argument("Invalid aiming delay or command age limits");
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;    // degree to rad
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;  // degree to rad
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  debug_aim_point = {};
  fire_allowed_ = false;
  if (targets.empty() || !std::isfinite(bullet_speed) || bullet_speed <= 0) return {};
  auto target = targets.front();
  if (target.diverged()) return {};
  const auto now = to_now ? std::chrono::steady_clock::now() : timestamp;
  const auto source = target.last_observation_time();
  const double age_ms = tools::delta_time(now, source) * 1000;
  if (source != std::chrono::steady_clock::time_point{} &&
      (age_ms < 0 || age_ms > control_max_age_ms_)) return {};
  const double delay = std::abs(target.ekf_x()[7]) > decision_speed_
                         ? high_speed_delay_time_ : low_speed_delay_time_;
  const auto launch_time = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(delay));
  target.predict_mean(launch_time);
  const auto solution = intercept(target, bullet_speed, [this](const Target & predicted) {
    const auto point = choose_aim_point(predicted);
    return point.valid ? std::optional<Eigen::Vector4d>(point.xyza) : std::nullopt;
  });
  if (!solution) return {};
  debug_aim_point = {true, solution->xyza};
  const auto & xyz = solution->xyza;
  io::Command command;
  command.control = true;
  command.yaw = tools::limit_rad(std::atan2(xyz.y(), xyz.x()) + yaw_offset_);
  command.pitch = -(solution->trajectory.pitch + pitch_offset_);
  command.horizon_distance = xyz.head<2>().norm();
  command.source_time = source;
  command.valid_until = source + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double, std::milli>(control_max_age_ms_));
  command.pose_valid = true;
  fire_allowed_ = source != std::chrono::steady_clock::time_point{} && age_ms <= shoot_max_age_ms_ &&
                  solution->target.fire_confident();
  if (!std::isfinite(command.yaw) || !std::isfinite(command.pitch)) {
    debug_aim_point = {};
    fire_allowed_ = false;
    return {};
  }
  return command;
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  if (command.control) command.yaw = tools::limit_rad(command.yaw - yaw_offset_ + yaw_offset);

  return command;
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.aimable_armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (armor_xyza_list.empty()) { lock_id_ = -1; return {}; }
  if (armor_xyza_list.size() == 1) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 不考虑小陀螺
  if (std::abs(target.ekf_x()[7]) <= 2 && target.name != ArmorName::outpost) {
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    // 绝无可能
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];

      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个装甲板在可射击范围内时，退出锁定模式
    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
