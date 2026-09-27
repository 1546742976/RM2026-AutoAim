#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/interception.hpp"
#include "tasks/auto_aim/runtime.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

using namespace std::chrono_literals;
namespace {
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
auto_aim::Armor observation(const Eigen::Vector3d & p, const Eigen::Matrix3d & r)
{
  auto_aim::Armor armor(0, .99f, cv::Rect(0, 0, 135, 56),
    {{0, 0}, {135, 0}, {135, 56}, {0, 56}});
  armor.name = auto_aim::ArmorName::three;
  armor.type = auto_aim::ArmorType::small;
  armor.color = auto_aim::Color::red;
  armor.xyz_in_world = p;
  armor.R_armor_to_world = r;
  armor.ypr_in_world = tools::eulers(r, 2, 1, 0);
  armor.ypd_in_world = tools::xyz2ypd(p);
  armor.pose_valid = armor.pose_reliable = true;
  armor.reprojection_error = 0;
  return armor;
}
}
int main(int argc, char ** argv)
{
  try {
    require(argc == 2, "Pass a runtime YAML for planner/aim configuration");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const double speed : {0.0, -1.0, nan}) {
      const tools::Trajectory invalid(speed, 3, 0);
      require(invalid.unsolvable && std::isfinite(invalid.fly_time), "Invalid speed produced a trajectory");
    }
    require(tools::Trajectory(1, 100, 100).unsolvable, "Impossible ballistic solution accepted");
    require(tools::Trajectory(20, 0, 0).unsolvable, "Zero horizontal distance accepted");
    const tools::Trajectory valid(25, 5, .3);
    require(!valid.unsolvable && valid.fly_time > 0, "Valid ballistic solution rejected");
    const double hit_height = 25 * std::sin(valid.pitch) * valid.fly_time -
      .5 * 9.7833 * valid.fly_time * valid.fly_time;
    require(std::abs(hit_height - .3) < 1e-9, "Ballistic solution misses requested height");

    const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(1);
    const Eigen::MatrixXd unit = Eigen::MatrixXd::Identity(1, 1);
    tools::ExtendedKalmanFilter ekf(x0, unit);
    Eigen::VectorXd z(1); z << 4;
    ekf.update(z, unit, 3 * unit);
    require(std::abs(ekf.last_nis - 4) < 1e-12, "NIS was not computed from the prior innovation");
    require(std::abs(ekf.x[0] - 1) < 1e-12 && std::abs(ekf.P(0,0) - .75) < 1e-12,
      "Scalar Kalman update or Joseph covariance incorrect");
    require(ekf.data.count("nees") == 0 && ekf.data.at("nis_fail") == 1 && ekf.last_update_accepted,
      "Diagnostic statistics changed generic EKF acceptance");
    tools::ExtendedKalmanFilter gated(x0, unit);
    gated.nis_gate = 3;
    gated.update(z, unit, 3 * unit);
    require(!gated.last_update_accepted && gated.x[0] == 0 && gated.P(0,0) == 1,
      "Rejected observation mutated prior state");
    z[0] = nan;
    gated.update(z, unit, unit);
    require(!gated.last_update_accepted && gated.x.allFinite(), "NaN measurement contaminated state");
    require(std::abs(tools::ExtendedKalmanFilter::nis_threshold_95(4) - 9.487729) < 1e-6,
      "NIS threshold has wrong degrees of freedom");

    const auto t = std::chrono::steady_clock::now();
    const Eigen::VectorXd variance = Eigen::VectorXd::Constant(11, .001);
    auto armor = observation({3, 0, .2}, Eigen::Matrix3d::Identity());
    armor.type = auto_aim::ArmorType::big;
    auto_aim::Target two(armor, t, .2, 2, variance);
    two.predict(t + 10ms);
    two.update(armor);
    require(two.last_update_accepted() && two.armor_xyza_list().size() == 2 && !two.diverged(),
      "Two-plate update failed or used a nonexistent third candidate");

    auto_aim::GeometryProfile geometry;
    geometry.axis_in_world = Eigen::Vector3d(.2, .3, 1).normalized();
    geometry.plate_offsets = {{-.2,0,0}, {0,-.22,.05}, {.2,0,.09}, {0,.22,-.03}};
    geometry.plate_normals = {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
      -Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitY()};
    const Eigen::Matrix3d basis = Eigen::Quaterniond::FromTwoVectors(
      Eigen::Vector3d::UnitZ(), geometry.axis_in_world).toRotationMatrix();
    const Eigen::Matrix3d rotation = basis * Eigen::AngleAxisd(.3, Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d center(3, .1, .3);
    armor = observation(center + rotation * geometry.plate_offsets[0], rotation);
    auto_aim::Target tilted(armor, t, .2, 4, variance);
    require(tilted.configure_geometry(geometry), "Valid calibrated tilted geometry rejected");
    const auto boards = tilted.armor_xyza_list();
    for (int i = 0; i < 4; ++i)
      require((boards[i].head<3>() - center - rotation * geometry.plate_offsets[i]).norm() < 1e-9,
        "Tilted-axis independent-height geometry reconstructed incorrectly");
    auto invalid_geometry = geometry;
    invalid_geometry.plate_offsets.pop_back();
    require(!tilted.configure_geometry(invalid_geometry), "Wrong plate-count calibration accepted");
    const auto covariance = tilted.ekf().P;
    const auto state = tilted.ekf_x();
    tilted.predict_mean(.02);
    tilted.predict_mean(-.02);
    require((tilted.ekf().P - covariance).norm() == 0 && (tilted.ekf_x() - state).norm() < 1e-9,
      "Mean-only planning changed covariance or was not reversible");

    auto_aim::Target unknown_geometry(armor, t, .2, 4, variance);
    require(!unknown_geometry.fire_confident(), "Unconfirmed geometry authorized firing");
    auto_aim::Aimer aimer(argv[1]);
    auto_aim::Planner planner(argv[1]);
    require(!aimer.aim({two}, t, 0, false).control, "Aimer substituted an invalid bullet speed");
    require(!planner.plan(std::optional<auto_aim::Target>(two), nan, t).control,
      "Planner substituted an invalid bullet speed");
    require(!planner.plan(std::optional<auto_aim::Target>(two), 25, t + 1s).control,
      "Planner accepted stale observation");
    require(!planner.plan(std::nullopt, 25, t).fire, "No-target plan allowed firing");
    auto_aim::Target synthetic(3, 0, .2, 0);
    const auto solution = auto_aim::intercept(synthetic, 25, [](const auto_aim::Target & target) {
      return auto_aim::nearest_aimable_plate(target);
    });
    require(solution && std::abs(solution->xyza.x() - 2.8) < 1e-9,
      "Stationary target intercept did not converge");
    require(!auto_aim::intercept(synthetic, -1, [](const auto_aim::Target & target) {
      return auto_aim::nearest_aimable_plate(target);
    }), "Invalid intercept produced a firing solution");

    auto_aim::Plan plan;
    plan.control = plan.fire = plan.pose_valid = true;
    plan.fire_yaw_tolerance = plan.fire_pitch_tolerance = .01;
    plan.source_time = t;
    plan.valid_until = t + 100ms;
    require(!auto_aim::control_intent(plan).command.shoot, "Missing measured feedback authorized fire");
    io::GimbalState feedback{};
    require(auto_aim::control_intent(plan, feedback).command.shoot, "Aligned finite feedback rejected");
    feedback.yaw = .1;
    require(!auto_aim::control_intent(plan, feedback).command.shoot, "Measured yaw error allowed fire");
    feedback.yaw = nan;
    require(!auto_aim::control_intent(plan, feedback).command.shoot, "NaN measured feedback allowed fire");
    std::cout << "estimation contracts passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
