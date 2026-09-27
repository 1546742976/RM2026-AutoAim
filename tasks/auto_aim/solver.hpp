#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "armor.hpp"

namespace auto_aim
{
class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // Prior must be the same physical plate in a recent frame, not merely the same robot class.
  void solve(Armor & armor, const Armor * prior = nullptr) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, const Eigen::Matrix3d & R_armor_to_world,
    ArmorType type) const;

  double oupost_reprojection_error(Armor armor, const double & picth);

  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;
  bool gimbal_pose_valid_ = true;
  double max_reprojection_error_ = 3.0;
  double max_corner_error_ = 6.0;
  double min_projected_edge_ = 2.0;
  double ambiguity_error_gap_ = 0.25;
  double min_view_cosine_ = 0.1;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP
