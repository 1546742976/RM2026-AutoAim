#ifndef IO__FRAME_PACKET_HPP
#define IO__FRAME_PACKET_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>

namespace io
{
struct FramePacket
{
  // cv::Mat owns/refcounts storage; drivers must not publish an SDK-owned view.
  cv::Mat image;
  uint64_t frame_id = 0;
  std::chrono::steady_clock::time_point received_at{};
  std::chrono::steady_clock::time_point exposure_time{};
  double timing_uncertainty_ms = 0;
  bool timing_calibrated = false;
  Eigen::Quaterniond pose = Eigen::Quaterniond::Identity();
  bool pose_valid = false;

  void set_pose(const Eigen::Quaterniond & q)
  {
    pose_valid = q.coeffs().allFinite() && q.norm() > 1e-9;
    pose = pose_valid ? q.normalized() : q;
  }
};
}  // namespace io
#endif
