#include "planner.hpp"
#include "tasks/auto_aim/interception.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  if (yaml["actuation_delay_s"])
    high_speed_delay_time_ = low_speed_delay_time_ = yaml["actuation_delay_s"].as<double>();
  shoot_max_age_ms_ = yaml["shoot_max_age_ms"].as<double>(100);
  control_max_age_ms_ = yaml["control_max_age_ms"].as<double>(200);
  auto_fire_ = yaml["auto_fire"].as<bool>(false);
  if (!std::isfinite(high_speed_delay_time_) || !std::isfinite(low_speed_delay_time_) ||
      high_speed_delay_time_ < 0 || low_speed_delay_time_ < 0 ||
      !(shoot_max_age_ms_ > 0 && control_max_age_ms_ >= shoot_max_age_ms_))
    throw std::invalid_argument("Invalid planner delay or command age limits");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // The epoch-tagged constructor is an offline synthetic fixture, not live input.
  const auto now = target.last_observation_time() == std::chrono::steady_clock::time_point{}
                     ? target.state_time() : std::chrono::steady_clock::now();
  return solve(std::move(target), bullet_speed, now);
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  return plan(std::move(target), bullet_speed, std::chrono::steady_clock::now());
}

Plan Planner::plan(
  std::optional<Target> target, double bullet_speed, std::chrono::steady_clock::time_point now)
{
  if (!target) { debug_xyza.setZero(); return {}; }
  return solve(std::move(*target), bullet_speed, now);
}

Plan Planner::solve(Target target, double bullet_speed, std::chrono::steady_clock::time_point now)
{
  debug_xyza.setZero();
  if (!std::isfinite(bullet_speed) || bullet_speed <= 0 || target.diverged()) return {};
  const auto source = target.last_observation_time();
  const double age_ms = tools::delta_time(now, source) * 1000;
  if (source != std::chrono::steady_clock::time_point{} &&
      (age_ms < 0 || age_ms > control_max_age_ms_)) return {};
  const double delay = std::abs(target.ekf_x()[7]) > decision_speed_
                         ? high_speed_delay_time_ : low_speed_delay_time_;
  const auto launch_time = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(delay));
  target.predict_mean(launch_time);
  const auto solution = intercept(target, bullet_speed, [](const Target & predicted) {
    return nearest_aimable_plate(predicted);
  });
  if (!solution) return {};
  target = solution->target;
  const auto impact_xyza = solution->xyza;
  const bool confidence = target.fire_confident();

  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception &) {
    debug_xyza.setZero();
    return {};
  }
  if (!traj.allFinite()) return {};

  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);
  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);
  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);
  if (!yaw_solver_->work->x.allFinite() || !yaw_solver_->work->u.allFinite() ||
      !pitch_solver_->work->x.allFinite() || !pitch_solver_->work->u.allFinite()) return {};

  Plan plan;
  plan.control = true;
  plan.source_time = source;
  plan.valid_until = source + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double, std::milli>(control_max_age_ms_));
  plan.pose_valid = true;
  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);
  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);
  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);
  debug_xyza = impact_xyza;

  const double distance = impact_xyza.head<2>().norm();
  const double incidence = std::max(0.0, std::cos(
    impact_xyza[3] - std::atan2(impact_xyza.y(), impact_xyza.x())));
  const double half_width = target.armor_type == ArmorType::big ? 0.1125 : 0.0675;
  const double yaw_tolerance = std::min(fire_thresh_, std::atan2(half_width * incidence, distance));
  const double pitch_tolerance = std::min(fire_thresh_, std::atan2(0.0275, distance));
  plan.fire_yaw_tolerance = yaw_tolerance;
  plan.fire_pitch_tolerance = pitch_tolerance;
  const int fire_index = HALF_HORIZON;
  plan.fire = auto_fire_ && confidence && source != std::chrono::steady_clock::time_point{} &&
    age_ms <= shoot_max_age_ms_ && incidence > 0.1 &&
    std::abs(traj(0, fire_index) - yaw_solver_->work->x(0, fire_index)) < yaw_tolerance &&
    std::abs(traj(2, fire_index) - pitch_solver_->work->x(0, fire_index)) < pitch_tolerance;
  return plan;
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  // Continuation samples may temporarily turn away. Visibility is enforced at
  // the intercept/fire point, not by invalidating the entire one-second horizon.
  const auto selected = nearest_aimable_plate(target, false);
  if (!selected) throw std::runtime_error("No visible aimable plate");
  const Eigen::Vector3d xyz = selected->head<3>();
  const double min_dist = xyz.head<2>().norm();
  debug_xyza = *selected;

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict_mean(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict_mean(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict_mean(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim
