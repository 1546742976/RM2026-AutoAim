#include "target.hpp"

#include <numeric>
#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  ekf_.nis_gate = 18.466827;  // 4 DoF, 99.9%; diagnostic remains 95%.
  last_observation_time_ = t;
  observed_xyz_ = armor.xyz_in_world;
  observed_normal_ = armor.R_armor_to_world.col(0);
  pose_reliable_ = armor.pose_valid && armor.pose_reliable;
  geometry_reliable_ = std::abs(armor.ypr_in_world[2]) < 10 * CV_PI / 180;
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  name = ArmorName::three;
  jumped = true;
  // Synthetic target constructor is for offline trajectory/control tests.
  pose_reliable_ = true;
  is_converged_ = true;
  update_count_ = 20;
  acceleration_variance_ = 0;
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  if (dt < 0) return;
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (!std::isfinite(dt) || dt < 0 || ekf_.x.size() != 11) return;
  prediction_horizon_ += dt;
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }
  v2 *= 1 + std::min(rejected_updates_, 4);
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] += 0.5 * angular_acceleration_ * dt * dt;
    x_prior[7] += angular_acceleration_ * dt;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  ekf_.predict(F, Q, f);
  mean_prediction_horizon_ = 0;
}

void Target::predict_mean(double dt)
{
  if (!std::isfinite(dt) || ekf_.x.size() != 11) return;
  for (int i : {0, 2, 4}) ekf_.x[i] += dt * ekf_.x[i + 1];
  ekf_.x[6] = tools::limit_rad(ekf_.x[6] + dt * ekf_.x[7] + 0.5 * angular_acceleration_ * dt * dt);
  ekf_.x[7] += angular_acceleration_ * dt;
  prediction_horizon_ += dt;
  mean_prediction_horizon_ += dt;
}

void Target::predict_mean(std::chrono::steady_clock::time_point t)
{
  predict_mean(tools::delta_time(t, t_));
  t_ = t;
}

void Target::update(const Armor & armor)
{
  last_update_accepted_ = false;
  if (!armor.pose_valid || !armor.xyz_in_world.allFinite() || !armor.ypr_in_world.allFinite() ||
      !armor.ypd_in_world.allFinite() || armor_num_ <= 0) return;
  if (!geometry_reliable_) { update_visible_board(armor); return; }
  // 装甲板匹配
  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板
  for (size_t i = 0; i < std::min<size_t>(3, xyza_i_list.size()); i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  const double previous_omega = last_observed_omega_;
  const double observation_dt = tools::delta_time(t_, last_observation_time_);
  update_ypda(armor, id);
  last_update_accepted_ = ekf_.last_update_accepted;
  if (!last_update_accepted_) {
    ++rejected_updates_;
    pose_reliable_ = false;
    if (rejected_updates_ >= 3) {
      geometry_reliable_ = false;
      update_visible_board(armor);
    }
    return;
  }
  rejected_updates_ = 0;
  last_observed_omega_ = ekf_.x[7];
  if (observation_dt > 1e-4 && observation_dt < 0.2) {
    const double sample = std::clamp((ekf_.x[7] - previous_omega) / observation_dt, -100.0, 100.0);
    const double gain = 1 - std::exp(-observation_dt / 0.08);
    const double error = sample - angular_acceleration_;
    angular_acceleration_ = std::clamp(angular_acceleration_ + gain * error, -100.0, 100.0);
    acceleration_variance_ = std::clamp((1 - gain) * acceleration_variance_ + gain * error * error, 1.0, 10000.0);
  }
  observed_xyz_ = armor.xyz_in_world;
  observed_normal_ = armor.R_armor_to_world.col(0);
  last_observation_time_ = t_;
  prediction_horizon_ = 0;
  mean_prediction_horizon_ = 0;
  pose_reliable_ = armor.pose_reliable;
  if (!geometry_ && std::abs(armor.ypr_in_world[2]) >= 10 * CV_PI / 180)
    geometry_reliable_ = false;
  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

}

void Target::update_visible_board(const Armor & armor)
{
  const double dt = tools::delta_time(t_, last_observation_time_);
  const Eigen::Vector3d displacement = armor.xyz_in_world - observed_xyz_;
  if (dt > 1e-4 && dt < 0.2 && displacement.norm() < 0.3) {
    const Eigen::Vector3d sample = displacement / dt;
    visible_board_velocity_ = sample.norm() < 10 ?
      (0.7 * visible_board_velocity_ + 0.3 * sample).eval() : Eigen::Vector3d::Zero();
  } else {
    visible_board_velocity_.setZero();
  }
  observed_xyz_ = armor.xyz_in_world;
  observed_normal_ = armor.R_armor_to_world.col(0);
  last_observation_time_ = t_;
  prediction_horizon_ = mean_prediction_horizon_ = 0;
  pose_reliable_ = armor.pose_reliable;
  last_update_accepted_ = true;
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  const double quality_scale = armor.pose_reliable ? 1.0 : 10.0;
  R_dig *= quality_scale * (1 + std::min(25.0, armor.reprojection_error * armor.reprojection_error) / 4);

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = armor_yaw(x, id);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = armor_yaw(ekf_.x, i);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  if (ekf_.x.size() != 11 || !ekf_.x.allFinite() || !ekf_.P.allFinite()) return true;
  if (!geometry_reliable_) return !observed_xyz_.allFinite();
  if (geometry_) return false;
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  if (geometry_) {
    return Eigen::Vector3d(x[0], x[2], x[4]) + axis_basis_ *
      Eigen::AngleAxisd(x[6], Eigen::Vector3d::UnitZ()) * geometry_->plate_offsets[id];
  }
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  if (geometry_) {
    Eigen::Matrix<double, 4, 11> jacobian = Eigen::Matrix<double, 4, 11>::Zero();
    constexpr double step = 1e-5;
    for (int col : {0, 2, 4, 6}) {
      Eigen::VectorXd plus = x, minus = x;
      plus[col] += step;
      minus[col] -= step;
      Eigen::Vector3d delta = tools::xyz2ypd(h_armor_xyz(plus, id)) - tools::xyz2ypd(h_armor_xyz(minus, id));
      delta[0] = tools::limit_rad(delta[0]);
      delta[1] = tools::limit_rad(delta[1]);
      jacobian.block<3, 1>(0, col) = delta / (2 * step);
      jacobian(3, col) = tools::limit_rad(armor_yaw(plus, id) - armor_yaw(minus, id)) / (2 * step);
    }
    return jacobian;
  }
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

double Target::armor_yaw(const Eigen::VectorXd & x, int id) const
{
  if (!geometry_) return tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  const Eigen::Vector3d normal = axis_basis_ * Eigen::AngleAxisd(x[6], Eigen::Vector3d::UnitZ()) *
                                 geometry_->plate_normals[id];
  return std::atan2(normal.y(), normal.x());
}

bool Target::configure_geometry(const GeometryProfile & profile)
{
  if (static_cast<int>(profile.plate_offsets.size()) != armor_num_ ||
      profile.plate_normals.size() != profile.plate_offsets.size() ||
      !profile.axis_in_world.allFinite() || profile.axis_in_world.norm() < 1e-6) return false;
  for (size_t i = 0; i < profile.plate_offsets.size(); ++i) {
    if (!profile.plate_offsets[i].allFinite() || !profile.plate_normals[i].allFinite() ||
        profile.plate_normals[i].head<2>().norm() < 1e-6 ||
        profile.plate_offsets[i].norm() > 2) return false;
  }
  geometry_ = profile;
  axis_basis_ = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), profile.axis_in_world.normalized()).toRotationMatrix();
  const Eigen::Vector3d body_normal = axis_basis_.transpose() * observed_normal_;
  ekf_.x[6] = tools::limit_rad(std::atan2(body_normal.y(), body_normal.x()) -
    std::atan2(profile.plate_normals[0].y(), profile.plate_normals[0].x()));
  const Eigen::Vector3d offset = axis_basis_ * Eigen::AngleAxisd(ekf_.x[6], Eigen::Vector3d::UnitZ()) * profile.plate_offsets[0];
  for (int i = 0; i < 3; ++i) ekf_.x[2 * i] = observed_xyz_[i] - offset[i];
  // Radius/alternating-height states are unused when mounting positions are calibrated.
  for (int i : {8, 9, 10}) { ekf_.P.row(i).setZero(); ekf_.P.col(i).setZero(); }
  geometry_reliable_ = true;
  return true;
}

std::chrono::steady_clock::time_point Target::last_observation_time() const
{
  return last_observation_time_;
}

bool Target::pose_reliable() const { return pose_reliable_; }

MotionMode Target::motion_mode() const
{
  if (ekf_.x.size() != 11 || rejected_updates_ || !geometry_reliable_ ||
      std::abs(angular_acceleration_) > 15 || acceleration_variance_ > 900)
    return MotionMode::uncertain;
  return std::abs(ekf_.x[7]) < 2 ? MotionMode::translating : MotionMode::rotating;
}

std::vector<Eigen::Vector4d> Target::aimable_armor_xyza_list() const
{
  if (geometry_reliable_) {
    const auto all = armor_xyza_list();
    if (jumped || all.empty()) return all;
    return {all[std::clamp(last_id, 0, static_cast<int>(all.size()) - 1)]};
  }
  // Unknown mounting geometry: only follow the observed board; never fabricate
  // the other three boards from one observation. This mode does not authorize fire.
  const Eigen::Vector3d xyz = observed_xyz_ + visible_board_velocity_ * std::max(0.0, prediction_horizon_);
  return {{xyz.x(), xyz.y(), xyz.z(), std::atan2(observed_normal_.y(), observed_normal_.x())}};
}

bool Target::fire_confident(double future_horizon) const
{
  if (!pose_reliable_ || !geometry_reliable_ || diverged() || update_count_ < 4 ||
      rejected_updates_ || !std::isfinite(future_horizon) || future_horizon < 0) return false;
  const double horizon = std::max(0.0, prediction_horizon_) + future_horizon;
  if (motion_mode() == MotionMode::uncertain && horizon > 0.10) return false;
  // A 3-sigma phase interval must fit within the plate's angular hit window.
  const double covariance_horizon = std::max(0.0, mean_prediction_horizon_ + future_horizon);
  const double phase_variance = std::max(0.0, ekf_.P(6, 6) +
    2 * covariance_horizon * ekf_.P(6, 7) + covariance_horizon * covariance_horizon * ekf_.P(7, 7)) +
    0.25 * std::pow(horizon, 4) * acceleration_variance_;
  double radius = std::max(0.05, std::abs(ekf_.x[8]) + std::abs(ekf_.x[9]));
  if (geometry_) {
    radius = 0.05;
    for (const auto & offset : geometry_->plate_offsets) radius = std::max(radius, offset.head<2>().norm());
  }
  const double half_width = armor_type == ArmorType::big ? 0.1125 : 0.0675;
  return std::isfinite(phase_variance) && 3 * std::sqrt(phase_variance) < std::atan2(half_width, radius);
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
