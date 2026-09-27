#include "extended_kalman_filter.hpp"

#include <numeric>
#include <cmath>
#include <algorithm>

namespace tools
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nis_fail"] = 0.0;
  data["update_rejected"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  const Eigen::MatrixXd candidate_P = F * P * F.transpose() + Q;
  const Eigen::VectorXd candidate_x = f(x);
  if (!candidate_P.allFinite() || !candidate_x.allFinite()) return x;
  P = (candidate_P + candidate_P.transpose()) * 0.5;
  x = candidate_x;
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  last_update_accepted = false;
  data["update_rejected"] = 1;
  const Eigen::VectorXd residual = z_subtract(z, h(x));
  const Eigen::MatrixXd S = H * P * H.transpose() + R;
  Eigen::LDLT<Eigen::MatrixXd> factor(S);
  const bool valid = residual.allFinite() && S.allFinite() &&
                     factor.info() == Eigen::Success && (factor.vectorD().array() > 0).all();
  last_nis = valid ? residual.dot(factor.solve(residual)) : std::numeric_limits<double>::infinity();
  const bool failed = !std::isfinite(last_nis) || last_nis < 0 ||
                      last_nis > nis_threshold_95(static_cast<int>(z.size()));
  recent_nis_failures.push_back(failed ? 1 : 0);
  while (recent_nis_failures.size() > std::max<size_t>(window_size, 1))
    recent_nis_failures.pop_front();
  const char * labels[] = {"residual_yaw", "residual_pitch", "residual_distance", "residual_angle"};
  for (int i = 0; i < 4; ++i) data[labels[i]] = i < residual.size() ? residual[i] : 0;
  data["nis"] = last_nis;
  data["nis_fail"] = failed ? 1 : 0;
  data["recent_nis_failures"] =
    static_cast<double>(std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0)) /
    recent_nis_failures.size();
  if (!valid || !std::isfinite(last_nis) || last_nis < 0 || last_nis > nis_gate) return x;

  const Eigen::MatrixXd K = factor.solve(H * P).transpose();
  const Eigen::VectorXd candidate_x = x_add(x, K * residual);
  const Eigen::MatrixXd correction = I - K * H;
  const Eigen::MatrixXd candidate_P = correction * P * correction.transpose() + K * R * K.transpose();
  if (!candidate_x.allFinite() || !candidate_P.allFinite()) return x;
  x = candidate_x;
  P = (candidate_P + candidate_P.transpose()) * 0.5;
  last_update_accepted = true;
  data["update_rejected"] = 0;
  return x;
}

double ExtendedKalmanFilter::nis_threshold_95(int dimension)
{
  static constexpr double thresholds[] = {
    0, 3.841459, 5.991465, 7.814728, 9.487729, 11.070498, 12.591587,
    14.067140, 15.507313, 16.918978, 18.307038, 19.675138, 21.026070};
  if (dimension > 0 && dimension <= 12) return thresholds[dimension];
  if (dimension <= 0) return 0;
  // Wilson-Hilferty approximation for generic higher-dimensional consumers.
  const double n = dimension;
  return n * std::pow(1 - 2 / (9 * n) + 1.644853627 * std::sqrt(2 / (9 * n)), 3);
}

}  // namespace tools
