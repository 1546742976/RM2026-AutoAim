#include "camera.hpp"

#include <stdexcept>
#include <cmath>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto camera_name = tools::read<std::string>(yaml, "camera_name");
  auto exposure_ms = tools::read<double>(yaml, "exposure_ms");
  exposure_ms_ = exposure_ms;
  transport_delay_ms_ = yaml["camera_transport_delay_ms"].as<double>(0.0);
  timing_uncertainty_ms_ = yaml["camera_timing_uncertainty_ms"].as<double>(5.0);
  timing_calibrated_ = yaml["camera_timing_calibrated"].as<bool>(false);
  if (!std::isfinite(exposure_ms_) || exposure_ms_ <= 0 ||
      !std::isfinite(transport_delay_ms_) || transport_delay_ms_ < 0 ||
      !std::isfinite(timing_uncertainty_ms_) || timing_uncertainty_ms_ < 0)
    throw std::invalid_argument("Invalid camera exposure/timing configuration");

  if (camera_name == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }

  else if (camera_name == "hikrobot") {
    auto gain = tools::read<double>(yaml, "gain");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  auto frame = read();
  img = std::move(frame.image);
  timestamp = frame.exposure_time;
}

FramePacket Camera::read()
{
  FramePacket frame;
  camera_->read(frame.image, frame.received_at);
  stamp(frame);
  return frame;
}

bool Camera::read_for(FramePacket & frame, std::chrono::milliseconds timeout)
{
  frame = {};
  if (!camera_->read_for(frame.image, frame.received_at, timeout)) return false;
  stamp(frame);
  return true;
}

void Camera::stamp(FramePacket & frame)
{
  frame.frame_id = next_frame_.fetch_add(1);
  // SDK time is currently receipt time. Device ticks alone do not establish a
  // host-clock offset; do not label this fallback as hardware synchronisation.
  frame.exposure_time = frame.received_at - std::chrono::duration_cast<
    std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(transport_delay_ms_ + exposure_ms_ * 0.5));
  frame.timing_uncertainty_ms = timing_uncertainty_ms_;
  frame.timing_calibrated = timing_calibrated_;
}

}  // namespace io
