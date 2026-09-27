#include <fmt/core.h>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv,
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path | configs/sentry.yaml | yaml配置文件路径}");
  const auto config = cli.get<std::string>(0);
  if (cli.has("help") || config.empty()) { cli.printMessage(); return 0; }

  tools::Exiter exiter;
  tools::Plotter plotter;
  io::CBoard cboard(config);
  io::Camera camera(config);
  auto_aim::multithread::MultiThreadDetector detector(config, false);
  auto_aim::Solver solver(config);
  auto_aim::Tracker tracker(config, solver);
  auto_aim::Aimer aimer(config);
  auto_aim::Shooter shooter(config);
  // Only the main thread plots and draws. The decision worker owns aimer/shooter mutable state.
  auto_aim::multithread::CommandGener commands(shooter, aimer, cboard, plotter, false);
  std::atomic<bool> stopping{false};
  std::mutex error_mutex;
  std::exception_ptr failure;
  const auto remember_failure = [&] {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!failure) failure = std::current_exception();
    stopping = true;
  };
  std::thread capture([&] {
    try {
      while (!stopping && !exiter.exit()) {
        io::FramePacket frame;
        if (!camera.read_for(frame, 50ms)) continue;
        const auto mode = cboard.mode.load();
        if (mode == io::Mode::auto_aim || mode == io::Mode::outpost)
          detector.push(frame.image, frame.exposure_time, frame.frame_id);
      }
    } catch (...) { remember_failure(); }
  });

  auto previous_mode = io::Mode::idle;
  auto mode_since = std::chrono::steady_clock::now();
  try {
    while (!stopping && !exiter.exit()) {
      const auto mode = cboard.mode.load();
      if (mode != previous_mode) {
        previous_mode = mode;
        mode_since = std::chrono::steady_clock::now();
        detector.reset();
        tracker.reset();
        commands.clear();
      }
      auto_aim::multithread::MultiThreadDetector::DetectionResult result;
      if (!detector.pop_for(result, 20ms)) continue;
      if ((mode != io::Mode::auto_aim && mode != io::Mode::outpost) ||
          result.timestamp < mode_since) continue;
      const auto q = cboard.imu_at(result.timestamp);
      if (!q.coeffs().allFinite()) { tracker.reset(); commands.clear(); continue; }
      solver.set_R_gimbal2world(q);
      const Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      auto targets = tracker.track(result.armors, result.timestamp);
      commands.push(targets, result.timestamp, cboard.bullet_speed.load(), ypr);

      auto image = result.image;  // DetectionResult owns this completed source frame.
      for (const auto & armor : result.armors) {
        tools::draw_points(image, armor.points, armor.pose_reliable ?
          cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255));
      }
      tools::draw_text(image, fmt::format("[{}] frame {}", tracker.state(), result.frame_id),
        {10, 30}, {255, 255, 255});
      nlohmann::json data{
        {"frame_id", result.frame_id}, {"armor_num", result.armors.size()},
        {"source_age_ms", 1000 * tools::delta_time(
          std::chrono::steady_clock::now(), result.timestamp)},
        {"bullet_speed", cboard.bullet_speed.load()}};
      if (!targets.empty()) {
        const auto & target = targets.front();
        const auto x = target.ekf_x();
        data["x"] = x[0]; data["y"] = x[2]; data["z"] = x[4];
        data["a"] = x[6]; data["w"] = x[7];
        const auto stats = target.ekf().data;
        for (const auto & key : {"nis", "nis_fail", "recent_nis_failures"}) {
          const auto it = stats.find(key);
          if (it != stats.end()) data[key] = it->second;
        }
      }
      plotter.plot(data);
      cv::resize(image, image, {}, 0.5, 0.5);
      cv::imshow("reprojection", image);
      if (cv::waitKey(1) == 'q') break;
    }
  } catch (...) { remember_failure(); }
  stopping = true;
  commands.clear();
  detector.close();
  capture.join();
  if (failure) std::rethrow_exception(failure);
  return 0;
}
