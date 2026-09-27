#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "tasks/auto_aim/corner_order.hpp"
#include "tasks/auto_aim/solver.hpp"

namespace
{
void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

auto_aim::Armor observation(const Eigen::Matrix3d & rotation, const Eigen::Vector3d & position)
{
  // Independent camera projection of the physical plate, ordered TL/TR/BR/BL.
  const std::vector<cv::Point3f> corners{
    {0, .0675f, .028f}, {0, -.0675f, .028f},
    {0, -.0675f, -.028f}, {0, .0675f, -.028f}};
  const cv::Mat camera = (cv::Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 512, 0, 0, 1);
  cv::Mat rotation_cv, rvec;
  cv::eigen2cv(rotation, rotation_cv);
  cv::Rodrigues(rotation_cv, rvec);
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(corners, rvec, cv::Vec3d(position.x(), position.y(), position.z()),
    camera, cv::Mat::zeros(1, 5, CV_64F), image_points);
  return auto_aim::Armor(0, .99f, cv::boundingRect(image_points), image_points);
}
}

int main(int argc, char ** argv)
{
  try {
    require(argc == 2, "Pass the synthetic pose calibration YAML");
    auto_aim::Solver solver(argv[1]);
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    Eigen::Matrix3d frontal;
    frontal << 0, -1, 0, 0, 0, -1, 1, 0, 0;
    const Eigen::Matrix3d tilted = Eigen::AngleAxisd(.6, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(.5, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(.2, Eigen::Vector3d::UnitX()) * frontal;
    const Eigen::Vector3d position(.04, -.03, 1.3);
    auto plate = observation(tilted, position);
    solver.solve(plate);
    require(plate.pose_valid && plate.pose_reliable, "Rolled, tilted plate should have a reliable pose");
    require((plate.xyz_in_world - position).norm() < 1e-3, "Incorrect reconstructed translation");
    require(Eigen::AngleAxisd(tilted.transpose() * plate.R_armor_to_world).angle() < .002,
      "PnP must retain full 3-D orientation, not fixed pitch/zero roll");

    auto unknown_labels = observation(tilted, position);
    unknown_labels.corners_reliable = false;
    solver.solve(unknown_labels);
    require(unknown_labels.pose_valid && !unknown_labels.pose_reliable,
      "Unknown physical corner labels must not certify a fire pose");

    auto invalid = observation(tilted, position);
    invalid.points[1] = invalid.points[0];
    solver.solve(invalid);
    require(!invalid.pose_valid, "Collapsed edge must be rejected");
    invalid = observation(tilted, position);
    std::swap(invalid.points[1], invalid.points[2]);
    solver.solve(invalid);
    require(!invalid.pose_valid, "Crossed corner ordering must be rejected");
    invalid = observation(tilted, position);
    invalid.points[0].x = std::numeric_limits<float>::quiet_NaN();
    solver.solve(invalid);
    require(!invalid.pose_valid, "NaN image corner must be rejected");

    const Eigen::Matrix3d far_rotation =
      Eigen::AngleAxisd(.35, Eigen::Vector3d::UnitY()).toRotationMatrix() * frontal;
    const Eigen::Vector3d far_position(0, 0, 8);
    auto ambiguous = observation(far_rotation, far_position);
    solver.solve(ambiguous);
    require(ambiguous.pose_valid && !ambiguous.pose_reliable,
      "Far planar two-solution ambiguity must be exposed");
    auto known_same_plate = ambiguous;
    known_same_plate.pose_reliable = true;
    known_same_plate.xyz_in_world = far_position;
    known_same_plate.R_armor_to_world = far_rotation;
    solver.solve(ambiguous, &known_same_plate);
    require(ambiguous.pose_reliable, "A same-plate prior should resolve equivalent reprojections");

    auto grazing = observation(
      Eigen::AngleAxisd(1.55, Eigen::Vector3d::UnitY()).toRotationMatrix() * frontal, position);
    solver.solve(grazing);
    require(!grazing.pose_reliable, "Grazing angle must not certify a fire pose");
    solver.set_R_gimbal2world(Eigen::Quaterniond(0, 0, 0, 0));
    solver.solve(plate);
    require(!plate.pose_valid, "Invalid IMU quaternion must invalidate pose");
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());
    solver.solve(plate);
    require(plate.pose_valid, "Pose should recover after a valid IMU sample");

    auto corners = observation(tilted, position).points;
    const auto physical = corners;
    std::swap(corners[1], corners[3]);
    const auto order = auto_aim::read_corner_order(YAML::Load("[0,3,2,1]"));
    require(auto_aim::order_corners(corners, order) && corners == physical,
      "Semantic corner mapping must survive roll");
    require(!auto_aim::order_corners(corners, {}), "Unspecified label mapping cannot be reliable");
    bool rejected = false;
    try { auto_aim::read_corner_order(YAML::Load("[0,0,2,3]")); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "Duplicate keypoint indices must fail configuration");
    auto_aim::Lightbar lamp(cv::RotatedRect({100, 100}, {80, 6}, 35), 0);
    require(std::abs(lamp.length - 80) < .01 && std::abs(lamp.width - 6) < .01,
      "Oblique lamp endpoints must follow the long edge");
    std::cout << "pose contracts passed\n";
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
