#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

using namespace std::chrono_literals;
int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv,
    "{help h||}{@config-path|configs/standard3.yaml|yaml configuration}");
  if (cli.has("help")) { cli.printMessage(); return 0; }
  const auto config = cli.get<std::string>(0);
  tools::Exiter exiter;
  tools::Plotter plotter;
  io::Camera camera(config);
  io::CBoard cboard(config);
  auto_aim::multithread::MultiThreadDetector detector(config, false);
  auto_aim::Solver solver(config);
  auto_aim::Tracker tracker(config, solver);
  auto_aim::Aimer aimer(config);
  auto_aim::Shooter shooter(config);
  auto_aim::multithread::CommandGener commands(shooter, aimer, cboard, plotter);
  auto_buff::Buff_Detector buff_detector(config);
  auto_buff::Solver buff_solver(config);
  auto_buff::SmallTarget small;
  auto_buff::BigTarget big;
  auto_buff::Aimer buff_aimer(config);

  std::atomic<bool> quit{false};
  std::mutex failure_mutex;
  std::exception_ptr failure;
  const auto fail = [&] {
    std::lock_guard<std::mutex> lock(failure_mutex);
    if (!failure) failure = std::current_exception();
    quit = true;
  };
  tools::ThreadSafeQueue<io::FramePacket, true> other_frames(1);
  std::thread capture([&] {
    try {
    while (!quit) {
      io::FramePacket frame;
      if (!camera.read_for(frame, 50ms)) continue;
      const auto mode = cboard.mode.load();
      if (mode == io::auto_aim || mode == io::outpost)
        detector.push(frame.image, frame.exposure_time, frame.frame_id);
      else
        other_frames.push(std::move(frame));
    }
    } catch (...) { fail(); }
  });

  auto previous = io::idle;
  auto mode_since = std::chrono::steady_clock::now();
  try {
  while (!quit && !exiter.exit()) {
    const auto mode = cboard.mode.load();
    if (mode != previous) {
      mode_since = std::chrono::steady_clock::now();
      detector.reset();
      commands.clear();
      tracker.reset();
      small = auto_buff::SmallTarget{};
      big = auto_buff::BigTarget{};
      buff_aimer.reset();
      other_frames.clear();
      previous = mode;
    }
    if (mode == io::auto_aim || mode == io::outpost) {
      auto_aim::multithread::MultiThreadDetector::DetectionResult result;
      if (!detector.pop_for(result, 20ms)) continue;
      if (result.timestamp < mode_since || cboard.mode.load() != mode) continue;
      const auto q = cboard.imu_at(result.timestamp);
      if (!q.coeffs().allFinite()) { tracker.reset(); commands.clear(); continue; }
      solver.set_R_gimbal2world(q);
      auto targets = tracker.track(result.armors, result.timestamp);
      const auto ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      commands.push(targets, result.timestamp, cboard.bullet_speed.load(), ypr);
    } else {
      io::FramePacket frame;
      if (!other_frames.pop_for(frame, 20ms)) continue;
      if (frame.exposure_time < mode_since || cboard.mode.load() != mode) continue;
      frame.set_pose(cboard.imu_at(frame.exposure_time));
      io::Command command;
      if (frame.pose_valid && (mode == io::small_buff || mode == io::big_buff)) {
        buff_solver.set_R_gimbal2world(frame.pose);
        auto runes = buff_detector.detect(frame.image);
        buff_solver.solve(runes);
        if (mode == io::small_buff) {
          small.get_target(runes, frame.exposure_time);
          auto target = small;
          command = buff_aimer.aim(target, frame.exposure_time, cboard.bullet_speed.load(), true);
        } else {
          big.get_target(runes, frame.exposure_time);
          auto target = big;
          command = buff_aimer.aim(target, frame.exposure_time, cboard.bullet_speed.load(), true);
        }
        command.source_time = frame.exposure_time;
        command.pose_valid = frame.pose_valid;
        command.frame_id = frame.frame_id;
      }
      cboard.send(command);
    }
  }
  } catch (...) { fail(); }
  quit = true;
  detector.close();
  other_frames.close();
  capture.join();
  commands.clear();
  if (failure) std::rethrow_exception(failure);
  return 0;
}
