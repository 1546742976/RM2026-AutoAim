#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "io/frame_packet.hpp"

namespace io
{
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;
  virtual bool read_for(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
                        std::chrono::milliseconds timeout)
  { read(img, timestamp); return !img.empty(); }
};

class Camera
{
public:
  Camera(const std::string & config_path);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  FramePacket read();
  bool read_for(FramePacket & frame, std::chrono::milliseconds timeout);

private:
  std::unique_ptr<CameraBase> camera_;
  std::atomic<uint64_t> next_frame_{1};
  double exposure_ms_ = 0;
  double transport_delay_ms_ = 0;
  double timing_uncertainty_ms_ = 5;
  bool timing_calibrated_ = false;
  void stamp(FramePacket & frame);
};

}  // namespace io

#endif  // IO__CAMERA_HPP
