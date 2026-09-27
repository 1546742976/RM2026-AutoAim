#include <fmt/core.h>

#include <chrono>
#include <exception>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[]) try
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  io::DM_IMU dm_imu{};

  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::GimbalMode::IDLE;
  auto last_mode = io::GimbalMode::IDLE;

  std::exception_ptr failure;
  try {
  while (!exiter.exit()) {
    gimbal.rethrow_if_failed();
    io::FramePacket frame;
    if (!camera.read_for(frame, 50ms)) { gimbal.send(io::Command{}); continue; }
    img = frame.image;
    t = frame.exposure_time;
    q = dm_imu.imu_at(t);
    frame.set_pose(q);
    gimbal.observe_frame(t, q);
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[static_cast<int>(mode)]);
      last_mode = mode;
      tracker.reset();
    }

    if (mode != io::GimbalMode::AUTO_AIM || !frame.pose_valid) {
      tracker.reset();
      gimbal.send(io::Command{});
      continue;
    }

    // recorder.record(img, q, t);

    solver.set_R_gimbal2world(q);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto armors = detector.detect(img);

    auto targets = tracker.track(armors, t);

    auto command = aimer.aim(targets, t, gimbal.state().bullet_speed);
    command.shoot = shooter.shoot(command, aimer, targets, ypr);
    command.frame_id = frame.frame_id;

    gimbal.send(command);
    gimbal.send_imu_forward(dm_imu);
  }

  } catch (...) { failure = std::current_exception(); }
  gimbal.close_control();
  try { gimbal.rethrow_if_failed(); }
  catch (...) { if (!failure) failure = std::current_exception(); }
  if (failure) std::rethrow_exception(failure);
  return 0;
} catch (const std::exception & error) {
  tools::logger()->error("Runtime stopped: {}", error.what());
  return 1;
} catch (...) {
  tools::logger()->error("Runtime stopped: unknown exception");
  return 1;
}
