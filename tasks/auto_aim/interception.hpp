#ifndef AUTO_AIM__INTERCEPTION_HPP
#define AUTO_AIM__INTERCEPTION_HPP

#include <cmath>
#include <optional>

#include "target.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
struct Intercept
{
  Target target;
  Eigen::Vector4d xyza;
  tools::Trajectory trajectory;
};

// Both control paths solve launch delay + flight time against the same moving
// target. The selector may choose a visible plate or a future shooting window.
template <typename Select>
std::optional<Intercept> intercept(
  const Target & launch_target, double bullet_speed, Select select)
{
  if (!std::isfinite(bullet_speed) || bullet_speed <= 0 || launch_target.diverged())
    return std::nullopt;
  double flight_time = 0;
  for (int iteration = 0; iteration < 10; ++iteration) {
    Target predicted = launch_target;
    predicted.predict_mean(flight_time);
    const auto selected = select(predicted);
    if (!selected || !selected->allFinite()) return std::nullopt;
    tools::Trajectory trajectory(bullet_speed, selected->template head<2>().norm(), (*selected)[2]);
    if (trajectory.unsolvable) return std::nullopt;
    if (std::abs(trajectory.fly_time - flight_time) < 0.001)
      return Intercept{std::move(predicted), *selected, trajectory};
    flight_time = trajectory.fly_time;
  }
  // Failure to converge is not a valid firing solution.
  return std::nullopt;
}

inline std::optional<Eigen::Vector4d> nearest_aimable_plate(const Target & target, bool visible_only = true)
{
  std::optional<Eigen::Vector4d> best;
  for (const auto & plate : target.aimable_armor_xyza_list()) {
    if (!plate.allFinite() || plate.head<2>().norm() <= 1e-6) continue;
    // Plate normal faces toward the vehicle centre in this model; an incidence
    // angle beyond 80 degrees is a grazing or hidden surface, not a hit window.
    const double bearing = std::atan2(plate.y(), plate.x());
    if (visible_only && std::cos(plate[3] - bearing) < std::cos(80 * CV_PI / 180)) continue;
    if (!best || plate.head<2>().squaredNorm() < best->head<2>().squaredNorm()) best = plate;
  }
  return best;
}
}  // namespace auto_aim

#endif
