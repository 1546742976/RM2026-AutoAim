#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/runtime.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

namespace {
struct PlanningInput {
  io::GimbalMode mode = io::GimbalMode::IDLE;
  auto_aim::TargetSnapshot snapshot;
  io::ControlIntent direct;
};
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv,
    "{help h||}{@config-path|configs/standard3.yaml|yaml configuration}");
  if (cli.has("help")) { cli.printMessage(); return 0; }
  const auto config_path = cli.get<std::string>(0);
  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget small_target;
  auto_buff::BigTarget big_target;
  auto_buff::Aimer buff_aimer(config_path);

  tools::ThreadSafeQueue<PlanningInput, true> pending(1);
  std::atomic<bool> quit{false};
  std::mutex failure_mutex;
  std::exception_ptr failure;
  const auto fail = [&] {
    std::lock_guard<std::mutex> lock(failure_mutex);
    if (!failure) failure = std::current_exception();
    quit = true;
  };
  std::thread planning([&] {
    try {
    PlanningInput input;
    auto next_plan = std::chrono::steady_clock::now();
    while (!quit) {
      PlanningInput update;
      if (pending.pop_for(update, 10ms)) input = std::move(update);
      if (quit) break;
      std::this_thread::sleep_until(next_plan);
      // Drain at the deadline: do not plan an input replaced during the wait.
      while (pending.try_pop(update)) input = std::move(update);
      const auto mode = gimbal.mode();
      io::ControlIntent intent;
      if (input.mode == mode && mode == io::GimbalMode::AUTO_AIM && input.snapshot.pose_valid) {
        const auto feedback = gimbal.state();
        intent = auto_aim::control_intent(
          planner.plan(input.snapshot.target, feedback.bullet_speed), feedback, input.snapshot.frame_id);
      } else if (input.mode == mode && mode != io::GimbalMode::IDLE) {
        intent = input.direct;
      }
      gimbal.send(intent);
      next_plan = std::chrono::steady_clock::now() + 10ms;
    }
    } catch (...) { fail(); }
    gimbal.send(io::Command{});
  });

  auto previous_mode = io::GimbalMode::IDLE;
  try {
  while (!quit && !exiter.exit()) {
    const auto mode = gimbal.mode();
    if (mode != previous_mode) {
      tracker.reset();
      small_target = auto_buff::SmallTarget{};
      big_target = auto_buff::BigTarget{};
      buff_aimer.reset();
      pending.push(PlanningInput{});
      previous_mode = mode;
    }
    io::FramePacket frame;
    if (!camera.read_for(frame, 50ms)) {
      pending.push(PlanningInput{});
      continue;
    }
    frame.set_pose(gimbal.q(frame.exposure_time));
    PlanningInput input;
    input.mode = mode;
    if (!frame.pose_valid || mode != gimbal.mode()) {
      tracker.reset();
      pending.push(std::move(input));
      continue;
    }
    if (mode == io::GimbalMode::AUTO_AIM) {
      solver.set_R_gimbal2world(frame.pose);
      auto armors = detector.detect(frame.image);
      auto targets = tracker.track(armors, frame.exposure_time);
      input.snapshot = auto_aim::snapshot(targets, frame);
    } else if (mode == io::GimbalMode::SMALL_BUFF || mode == io::GimbalMode::BIG_BUFF) {
      buff_solver.set_R_gimbal2world(frame.pose);
      auto runes = buff_detector.detect(frame.image);
      buff_solver.solve(runes);
      auto_aim::Plan plan{};
      if (mode == io::GimbalMode::SMALL_BUFF) {
        small_target.get_target(runes, frame.exposure_time);
        auto target = small_target;
        plan = buff_aimer.mpc_aim(target, frame.exposure_time, gimbal.state(), true);
      } else {
        big_target.get_target(runes, frame.exposure_time);
        auto target = big_target;
        plan = buff_aimer.mpc_aim(target, frame.exposure_time, gimbal.state(), true);
      }
      input.direct = auto_aim::control_intent(plan, gimbal.state(), frame.frame_id);
      input.direct.command.source_time = frame.exposure_time;
      input.direct.command.pose_valid = frame.pose_valid;
    }
    pending.push(std::move(input));
  }
  } catch (...) { fail(); }
  quit = true;
  pending.close();
  planning.join();
  gimbal.send(io::Command{});
  if (failure) std::rethrow_exception(failure);
  return 0;
}
