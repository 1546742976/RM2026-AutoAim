#include "trajectory.hpp"

#include <cmath>

namespace tools
{
constexpr double g = 9.7833;

Trajectory::Trajectory(const double v0, const double d, const double h)
{
  if (!std::isfinite(v0) || !std::isfinite(d) || !std::isfinite(h) || v0 <= 0 || d <= 1e-6)
    return;
  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (!std::isfinite(delta) || delta < 0 || !std::isfinite(a) || a <= 0) return;
  // Rationalized low-arc root avoids catastrophic cancellation at short range.
  const auto tan_pitch = 2 * c / (d + std::sqrt(delta));
  const auto candidate_pitch = std::atan(tan_pitch);
  const auto candidate_time = d / (v0 * std::cos(candidate_pitch));
  if (!std::isfinite(candidate_pitch) || !std::isfinite(candidate_time) || candidate_time <= 0)
    return;
  pitch = candidate_pitch;
  fly_time = candidate_time;
  unsolvable = false;
}

}  // namespace tools
