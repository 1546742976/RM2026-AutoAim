#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  auto check_calibration = [](const std::vector<double> & data, std::size_t size) {
    if (data.size() != size || !std::all_of(data.begin(), data.end(), [](double v) {
          return std::isfinite(v);
        })) throw std::invalid_argument("Invalid PnP calibration array");
  };
  check_calibration(R_gimbal2imubody_data, 9);
  check_calibration(R_camera2gimbal_data, 9);
  check_calibration(t_camera2gimbal_data, 3);
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  check_calibration(camera_matrix_data, 9);
  check_calibration(distort_coeffs_data, 5);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
  max_reprojection_error_ = yaml["pnp_max_reprojection_error"].as<double>(3.0);
  max_corner_error_ = yaml["pnp_max_corner_error"].as<double>(6.0);
  min_projected_edge_ = yaml["pnp_min_projected_edge"].as<double>(2.0);
  ambiguity_error_gap_ = yaml["pnp_ambiguity_error_gap"].as<double>(0.25);
  min_view_cosine_ = yaml["pnp_min_view_cosine"].as<double>(0.1);
  if (!(max_reprojection_error_ > 0 && std::isfinite(max_reprojection_error_)) ||
      !(max_corner_error_ > 0 && std::isfinite(max_corner_error_)) ||
      !(min_projected_edge_ > 0 && std::isfinite(min_projected_edge_)) ||
      !(ambiguity_error_gap_ >= 0 && std::isfinite(ambiguity_error_gap_)) ||
      !(min_view_cosine_ > 0 && min_view_cosine_ <= 1))
    throw std::invalid_argument("Invalid PnP quality thresholds");
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  gimbal_pose_valid_ = q.coeffs().allFinite() && q.norm() > 1e-6;
  if (!gimbal_pose_valid_) return;
  Eigen::Matrix3d R_imubody2imuabs = q.normalized().toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(Armor & armor, const Armor * prior) const
{
  armor.pose_valid = false;
  armor.pose_reliable = false;
  armor.reprojection_error = std::numeric_limits<double>::infinity();
  if (!gimbal_pose_valid_ || armor.points.size() != 4) return;
  for (const auto & p : armor.points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) return;
  if (!cv::isContourConvex(armor.points) || std::abs(cv::contourArea(armor.points)) < 4.0) return;
  for (std::size_t i = 0; i < 4; ++i)
    if (cv::norm(armor.points[i] - armor.points[(i + 1) % 4]) < min_projected_edge_) return;

  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  std::vector<cv::Mat> rvecs, tvecs;
  try {
    cv::solvePnPGeneric(
      object_points, armor.points, camera_matrix_, distort_coeffs_, rvecs, tvecs, false,
      cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception &) {
    return;
  }

  struct Candidate
  {
    Eigen::Vector3d camera_position;
    Eigen::Matrix3d camera_rotation;
    Eigen::Vector3d world_position;
    Eigen::Matrix3d world_rotation;
    double error;
    double prior_distance = std::numeric_limits<double>::infinity();
  };
  std::vector<Candidate> candidates;
  const bool use_prior = prior && prior->pose_reliable && prior->name == armor.name &&
    prior->color == armor.color && prior->type == armor.type &&
    prior->xyz_in_world.allFinite() && prior->R_armor_to_world.allFinite();
  for (std::size_t i = 0; i < rvecs.size(); ++i) {
    if (!cv::checkRange(rvecs[i]) || !cv::checkRange(tvecs[i])) continue;
    Candidate candidate;
    cv::Mat rotation;
    cv::Rodrigues(rvecs[i], rotation);
    cv::cv2eigen(rotation, candidate.camera_rotation);
    cv::cv2eigen(tvecs[i], candidate.camera_position);
    bool positive_depth = true;
    for (const auto & p : object_points) {
      const Eigen::Vector3d camera_point = candidate.camera_rotation *
        Eigen::Vector3d(p.x, p.y, p.z) + candidate.camera_position;
      positive_depth = positive_depth && camera_point.z() > 1e-4;
    }
    if (!positive_depth) continue;
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix_, distort_coeffs_, projected);
    double sum_squared = 0, max_error = 0;
    for (std::size_t j = 0; j < 4; ++j) {
      const double error = cv::norm(projected[j] - armor.points[j]);
      sum_squared += error * error;
      max_error = std::max(max_error, error);
    }
    candidate.error = std::sqrt(sum_squared / 4);
    if (!std::isfinite(candidate.error) || candidate.error > max_reprojection_error_ ||
        max_error > max_corner_error_) continue;
    candidate.world_position = R_gimbal2world_ *
      (R_camera2gimbal_ * candidate.camera_position + t_camera2gimbal_);
    candidate.world_rotation = R_gimbal2world_ * R_camera2gimbal_ * candidate.camera_rotation;
    if (use_prior) {
      const double distance = (candidate.world_position - prior->xyz_in_world).norm();
      const double angle = Eigen::AngleAxisd(
        prior->R_armor_to_world.transpose() * candidate.world_rotation).angle();
      // A prior cannot override a geometrically much better solution or another physical plate.
      if (distance < 0.3 && angle < CV_PI / 4)
        candidate.prior_distance = distance / 0.3 + angle / (CV_PI / 4);
    }
    candidates.push_back(candidate);
  }
  if (candidates.empty()) return;
  std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
    return a.error < b.error;
  });
  std::size_t selected = 0;
  const double equivalent_error = candidates.front().error + ambiguity_error_gap_;
  if (use_prior) {
    for (std::size_t i = 1; i < candidates.size(); ++i) {
      if (candidates[i].error <= equivalent_error &&
          candidates[i].prior_distance < candidates[selected].prior_distance) selected = i;
    }
  }
  const auto & best = candidates[selected];
  bool ambiguous = false;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i == selected || candidates[i].error > equivalent_error) continue;
    const auto & alternative = candidates[i];
    const double angle = Eigen::AngleAxisd(best.world_rotation.transpose() *
      alternative.world_rotation).angle();
    if (angle < 5.0 * CV_PI / 180 &&
        (best.world_position - alternative.world_position).norm() < 0.03) continue;
    const bool prior_resolves = std::isfinite(best.prior_distance) &&
      alternative.prior_distance > best.prior_distance + 0.2;
    ambiguous = ambiguous || !prior_resolves;
  }
  armor.xyz_in_gimbal = R_camera2gimbal_ * best.camera_position + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;
  armor.R_armor_to_gimbal = R_camera2gimbal_ * best.camera_rotation;
  armor.R_armor_to_world = best.world_rotation;
  armor.ypr_in_gimbal = tools::eulers(armor.R_armor_to_gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(armor.R_armor_to_world, 2, 1, 0);
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  armor.yaw_raw = armor.ypr_in_world[0];
  armor.reprojection_error = best.error;
  armor.pose_valid = armor.xyz_in_world.allFinite() && armor.ypr_in_world.allFinite();
  const double view_cosine = std::abs(best.camera_rotation.col(0).dot(
    best.camera_position.normalized()));
  armor.pose_reliable = armor.pose_valid && armor.corners_reliable && !ambiguous &&
    view_cosine >= min_view_cosine_;
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // This overload is only the legacy nominal-pitch rendering/calibration model.
  return reproject_armor(xyz_in_world, R_armor2world, type);
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, const Eigen::Matrix3d & R_armor2world,
  ArmorType type) const
{
  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  solve(armor);
  if (!armor.pose_valid) return std::numeric_limits<double>::infinity();

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  auto image_points = reproject_armor(xyz_in_world, _R_armor2world, armor.type);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

// 世界坐标到像素坐标的转换
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 如果没有有效点，返回空vector
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace auto_aim
