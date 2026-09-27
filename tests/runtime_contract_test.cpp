#include <atomic>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include "io/control_guard.hpp"
#include "io/control_publisher.hpp"
#include "tools/pose_history.hpp"
#include "tools/latency_stats.hpp"

using namespace std::chrono_literals;
static void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

static void require_tracking_only(const io::ControlIntent & intent, const char * message)
{
  require(intent.command.control && !intent.command.shoot, message);
}

int main()
{
  using Clock = std::chrono::steady_clock;
  const auto t = Clock::time_point{} + 1s;
  tools::PoseHistory history;
  history.push(Eigen::Quaterniond::Identity(), t);
  history.push(Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())), t + 20ms);
  const auto a = history.at(t + 10ms, 0ms);
  const auto b = history.at(t + 10ms, 0ms);
  require(a && b, "a repeated historical pose must remain available");
  require(a->q.angularDistance(b->q) < 1e-12, "pose queries consumed or changed data");
  require(std::abs(a->q.angularDistance(Eigen::Quaterniond::Identity()) - 0.1) < 1e-8,
          "slerp interpolation failed");
  require(!history.at(t - 1ms, 0ms) && !history.at(t + 21ms, 0ms), "pose extrapolation accepted");
  history.push(Eigen::Quaterniond::Identity(), t + 100ms);
  require(!history.at(t + 60ms, 0ms), "large missing-IMU gap accepted");

  io::ControlGuard guard;
  guard.configure(30, 60, true);
  guard.observe(t, true);
  guard.observe_pose_feedback(t, true);
  guard.observe_status_feedback(t, 1.0);
  require(guard.feedback_fresh(t, t + 10ms) && !guard.feedback_fresh(t, t + 40ms),
          "Feedback freshness did not use shoot age limit");
  io::ControlIntent command{{true, true, 0.1, 0.2}};
  command.command.source_time = t;
  require(guard.apply(command, true, t + 10ms).command.shoot, "fresh command denied");
  auto stale = guard.apply(command, true, t + 40ms);
  require(stale.command.control && !stale.command.shoot, "stale fire allowed");
  require(!guard.apply(command, true, t + 70ms).command.control, "expired control allowed");
  require(!guard.apply(command, false, t + 10ms).command.control, "idle mode controlled gimbal");

  io::ControlGuard uncalibrated;
  uncalibrated.configure(30, 60);
  uncalibrated.observe(t, true);
  uncalibrated.observe_pose_feedback(t, true);
  uncalibrated.observe_status_feedback(t, 1.0);
  require_tracking_only(uncalibrated.apply(command, true, t + 10ms),
                        "uncalibrated timing allowed shoot or disabled tracking");

  auto missing_source = command;
  missing_source.command.source_time = {};
  require_tracking_only(guard.apply(missing_source, true, t + 10ms),
                        "missing source timestamp allowed shoot or disabled tracking");

  io::ControlGuard feedback_guard;
  feedback_guard.configure(30, 60, true);
  feedback_guard.observe(t, true);
  require_tracking_only(feedback_guard.apply(command, true, t + 10ms),
                        "missing feedback allowed shoot or disabled tracking");
  feedback_guard.observe_pose_feedback(t, true);
  require_tracking_only(feedback_guard.apply(command, true, t + 10ms),
                        "missing status feedback allowed shoot or disabled tracking");
  feedback_guard.observe_status_feedback(t, 1.0);
  require(feedback_guard.apply(command, true, t + 10ms).command.shoot,
          "valid feedback did not restore eligibility");

  for (const bool stale_pose : {true, false}) {
    io::ControlGuard stale_feedback;
    stale_feedback.configure(30, 60, true);
    stale_feedback.observe(t + 40ms, true);
    stale_feedback.observe_pose_feedback(stale_pose ? t : t + 40ms, true);
    stale_feedback.observe_status_feedback(stale_pose ? t + 40ms : t, 1.0);
    auto fresh_command = command;
    fresh_command.command.source_time = t + 40ms;
    require_tracking_only(stale_feedback.apply(fresh_command, true, t + 40ms),
                          stale_pose ? "stale pose feedback allowed shoot or disabled tracking"
                                     : "stale status feedback allowed shoot or disabled tracking");
  }

  const double invalid_speeds[] = {
    0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  auto status_time = t;
  for (const double speed : invalid_speeds) {
    status_time += 1ms;
    feedback_guard.observe_status_feedback(status_time, speed);
    require_tracking_only(feedback_guard.apply(command, true, status_time),
                          "invalid speed did not immediately revoke shoot");
    feedback_guard.observe_status_feedback(status_time - 1ms, 1.0);
    require_tracking_only(feedback_guard.apply(command, true, status_time),
                          "older valid status restored an invalidated feedback stream");
    feedback_guard.observe_status_feedback(status_time, 1.0);
    require(feedback_guard.apply(command, true, status_time).command.shoot,
            "valid status at the same timestamp did not restore eligibility");
  }

  feedback_guard.observe_pose_feedback(t + 10ms, false);
  require_tracking_only(feedback_guard.apply(command, true, t + 10ms),
                        "invalid pose feedback did not immediately revoke shoot");
  feedback_guard.observe_pose_feedback(t + 9ms, true);
  require_tracking_only(feedback_guard.apply(command, true, t + 10ms),
                        "older valid pose restored an invalidated feedback stream");
  feedback_guard.observe_pose_feedback(t + 10ms, true);
  require(feedback_guard.apply(command, true, t + 10ms).command.shoot,
          "valid pose at the same timestamp did not restore eligibility");
  auto no_shoot = command;
  no_shoot.command.shoot = false;
  require_tracking_only(feedback_guard.apply(no_shoot, true, t + 10ms),
                        "feedback recovery escalated a non-shoot command");

  io::ControlGuard combined_feedback;
  combined_feedback.configure(30, 60, true);
  combined_feedback.observe(t, true);
  combined_feedback.observe_feedback(t, true, 1.0);
  require(combined_feedback.apply(command, true, t).command.shoot,
          "valid combined feedback denied");
  combined_feedback.observe_feedback(t + 1ms, true, 0.0);
  require_tracking_only(combined_feedback.apply(command, true, t + 1ms),
                        "invalid status in combined feedback did not revoke shoot");
  combined_feedback.observe_feedback(t + 1ms, true, 1.0);
  require(combined_feedback.apply(command, true, t + 1ms).command.shoot,
          "valid combined status did not restore eligibility");
  combined_feedback.observe_feedback(t + 2ms, false, 1.0);
  require_tracking_only(combined_feedback.apply(command, true, t + 2ms),
                        "invalid pose in combined feedback did not revoke shoot");
  combined_feedback.observe_feedback(t + 2ms, true, 1.0);
  require(combined_feedback.apply(command, true, t + 2ms).command.shoot,
          "valid combined pose did not restore eligibility");
  require_tracking_only(combined_feedback.apply(no_shoot, true, t + 2ms),
                        "combined feedback recovery escalated a non-shoot command");

  for (const bool refresh_pose_first : {true, false}) {
    io::ControlGuard mode_guard;
    mode_guard.configure(30, 60, true);
    mode_guard.observe(t, true);
    mode_guard.observe_pose_feedback(t, true);
    mode_guard.observe_status_feedback(t, 1.0);
    mode_guard.invalidate(t + 10ms);
    require(!mode_guard.apply(command, true, t + 10ms).command.control,
            "mode invalidation left control active");
    mode_guard.observe(t + 20ms, true);
    require(!mode_guard.apply(command, true, t + 21ms).command.control,
            "previous-mode command accepted");
    auto new_mode_command = command;
    new_mode_command.command.source_time = t + 20ms;
    require_tracking_only(mode_guard.apply(new_mode_command, true, t + 21ms),
                          "mode recovery reused invalidated feedback");
    mode_guard.observe_pose_feedback(t + 9ms, true);
    mode_guard.observe_status_feedback(t + 9ms, 1.0);
    if (refresh_pose_first) mode_guard.observe_pose_feedback(t + 20ms, true);
    else mode_guard.observe_status_feedback(t + 20ms, 1.0);
    require_tracking_only(mode_guard.apply(new_mode_command, true, t + 21ms),
                          "feedback from the previous mode allowed shoot");
    if (refresh_pose_first) mode_guard.observe_status_feedback(t + 20ms, 1.0);
    else mode_guard.observe_pose_feedback(t + 20ms, true);
    require(mode_guard.apply(new_mode_command, true, t + 21ms).command.shoot,
            "new-mode frame and feedback did not restore eligibility");
    new_mode_command.command.shoot = false;
    require_tracking_only(mode_guard.apply(new_mode_command, true, t + 21ms),
                          "mode recovery escalated a non-shoot command");
  }

  std::atomic<bool> active{false};
  std::atomic<int> writes{0};
  io::ControlGuard live_guard;
  live_guard.configure(15, 30, true);
  const auto live_time = Clock::now();
  live_guard.observe(live_time, true);
  live_guard.observe_pose_feedback(live_time, true);
  live_guard.observe_status_feedback(live_time, 1.0);
  {
    io::ControlPublisher publisher(
      [&](io::ControlIntent intent) { return live_guard.apply(intent, true); },
      [&](const io::ControlIntent & intent) { active = intent.command.control; ++writes; });
    command.command.source_time = live_time;
    publisher.publish(command);
    std::this_thread::sleep_for(80ms);
    require(writes > 1 && !active, "stalled producer left active command latched");
  }
  require(!active, "shutdown did not stop controller");

  {
    io::ControlGuard transmit_guard;
    transmit_guard.configure(60000, 120000, true);
    const auto source_time = Clock::now();
    transmit_guard.observe(source_time, true);
    transmit_guard.observe_feedback(source_time, true, 1.0);
    std::mutex emitted_mutex;
    std::condition_variable emitted_changed;
    io::ControlIntent first_emissions[3];
    io::ControlIntent last_emission;
    int emitted_count = 0;
    io::ControlPublisher publisher(
      [&](io::ControlIntent intent) { return transmit_guard.apply(intent, true); },
      [&](const io::ControlIntent & intent) {
        std::lock_guard<std::mutex> lock(emitted_mutex);
        if (emitted_count < 3) first_emissions[emitted_count] = intent;
        ++emitted_count;
        last_emission = intent;
        // Invalidate after the repeated write, before the next filtering pass.
        if (emitted_count == 2) transmit_guard.observe_status_feedback(Clock::now(), 0.0);
        emitted_changed.notify_one();
      });
    auto stored_command = command;
    stored_command.command.source_time = source_time;
    publisher.publish(stored_command);
    std::unique_lock<std::mutex> lock(emitted_mutex);
    require(emitted_changed.wait_for(lock, 2s, [&] { return emitted_count >= 3; }),
            "publisher did not repeat the stored command after feedback invalidation");
    require(first_emissions[0].command.control && first_emissions[0].command.shoot &&
              first_emissions[1].command.control && first_emissions[1].command.shoot,
            "publisher did not repeat a permitted stored shoot request");
    require_tracking_only(first_emissions[2],
                          "next transmitted command ignored invalid status feedback");
    lock.unlock();
    publisher.close();
    lock.lock();
    require(!last_emission.command.control && !last_emission.command.shoot,
            "publisher close did not emit a stop after feedback invalidation");
  }

  tools::LatencyStats disabled;
  disabled.record(5);
  require(!disabled.record_first_send(t, t + 1ms) && disabled.summary().samples == 0,
          "disabled metrics collected samples");
  tools::LatencyStats latency(true);
  require(latency.record_first_send(t, t + 5ms), "first source ignored");
  require(!latency.record_first_send(t, t + 6ms) &&
          !latency.record_first_send(t - 1ms, t), "repeated or older source counted");
  latency.record(-1);
  latency.record(std::numeric_limits<double>::quiet_NaN());
  require(latency.summary().samples == 1, "invalid metrics counted");
  tools::LatencyStats window(true);
  for (int i = 1; i <= 1024; ++i) window.record(i);
  const auto stats = window.summary();
  require(stats.samples == 512 && stats.total_samples == 1024 && stats.p50_ms == 768 &&
          stats.p95_ms == 999 && stats.max_ms == 1024, "bounded metrics or percentiles incorrect");
  std::cout << "runtime contracts passed\n";
}
