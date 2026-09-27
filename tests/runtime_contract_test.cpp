#include <atomic>
#include <cmath>
#include <iostream>
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

int main()
{
  using Clock = std::chrono::steady_clock;
  const auto t = Clock::now();
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
  guard.configure(30, 60);
  guard.observe(t, true);
  require(guard.feedback_fresh(t, t + 10ms) && !guard.feedback_fresh(t, t + 40ms),
          "Feedback freshness did not use shoot age limit");
  io::ControlIntent command{{true, true, 0.1, 0.2}};
  command.command.source_time = t;
  require(guard.apply(command, true, t + 10ms).command.shoot, "fresh command denied");
  auto stale = guard.apply(command, true, t + 40ms);
  require(stale.command.control && !stale.command.shoot, "stale fire allowed");
  require(!guard.apply(command, true, t + 70ms).command.control, "expired control allowed");
  require(!guard.apply(command, false, t + 10ms).command.control, "idle mode controlled gimbal");
  guard.invalidate(t + 10ms);
  guard.observe(t + 20ms, true);
  require(!guard.apply(command, true, t + 21ms).command.control, "previous-mode command accepted");

  std::atomic<bool> active{false};
  std::atomic<int> writes{0};
  io::ControlGuard live_guard;
  live_guard.configure(15, 30);
  const auto live_time = Clock::now();
  live_guard.observe(live_time, true);
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
