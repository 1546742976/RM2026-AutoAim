#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

// Calibrated board centres/normals in an axis-aligned body frame. Plate zero is
// the first observed board; profile ordering must follow positive rotation.
struct GeometryProfile
{
  Eigen::Vector3d axis_in_world = Eigen::Vector3d::UnitZ();
  std::vector<Eigen::Vector3d> plate_offsets;
  std::vector<Eigen::Vector3d> plate_normals;
};

enum class MotionMode { translating, rotating, uncertain };

class Target
{
public:
  ArmorName name = ArmorName::not_armor;
  ArmorType armor_type = ArmorType::small;
  ArmorPriority priority = ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  // Planning copies do not need covariance propagation, including backwards samples.
  void predict_mean(double dt);
  void predict_mean(std::chrono::steady_clock::time_point t);
  void update(const Armor & armor);
  bool configure_geometry(const GeometryProfile & profile);
  std::chrono::steady_clock::time_point last_observation_time() const;
  std::chrono::steady_clock::time_point state_time() const { return t_; }
  bool pose_reliable() const;
  bool fire_confident(double future_horizon = 0) const;
  MotionMode motion_mode() const;
  std::vector<Eigen::Vector4d> aimable_armor_xyza_list() const;
  bool last_update_accepted() const { return last_update_accepted_; }
  double prediction_horizon() const { return prediction_horizon_; }

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_ = 0;
  int switch_count_ = 0;
  int update_count_ = 0;

  bool is_switch_ = false, is_converged_ = false;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;
  std::chrono::steady_clock::time_point last_observation_time_{};
  Eigen::Vector3d observed_xyz_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d observed_normal_ = Eigen::Vector3d::UnitX();
  Eigen::Vector3d visible_board_velocity_ = Eigen::Vector3d::Zero();
  bool pose_reliable_ = false;
  bool geometry_reliable_ = true;
  bool last_update_accepted_ = false;
  std::optional<GeometryProfile> geometry_;
  Eigen::Matrix3d axis_basis_ = Eigen::Matrix3d::Identity();
  double angular_acceleration_ = 0;
  double last_observed_omega_ = 0;
  double acceleration_variance_ = 400;
  double prediction_horizon_ = 0;
  double mean_prediction_horizon_ = 0;
  int rejected_updates_ = 0;
  double armor_yaw(const Eigen::VectorXd & x, int id) const;
  void update_visible_board(const Armor & armor);

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
