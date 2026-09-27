#include <atomic>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
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

static std::string exception_message(std::exception_ptr failure)
{
  if (!failure) return {};
  try { std::rethrow_exception(failure); }
  catch (const std::exception & error) { return error.what(); }
  catch (...) { return "non-standard exception"; }
}

static void test_guard_diagnostics()
{
  using Guard = io::ControlGuard;
  using Clock = Guard::Clock;
  const auto t = Clock::time_point{} + 1s;
  Guard guard;
  guard.configure(30, 60, true);
  guard.observe(t, true);
  guard.observe_feedback(t, true, 1.0);
  io::ControlIntent command{{true, false, 0.1, 0.2}};
  command.command.source_time = t;
  command.command.frame_id = 42;
  const auto fresh = guard.evaluate(command, true, t);
  require(fresh.reasons == 0 && fresh.intent.command.control && !fresh.intent.command.shoot,
          "diagnostic evaluation escalated an existing non-shoot request");

  auto missing = command;
  missing.command.source_time = {};
  const auto missing_result = guard.evaluate(missing, true, t);
  require((missing_result.reasons & Guard::missing_source) != 0,
          "missing source was not diagnosed");
  require_tracking_only(missing_result.intent, "missing source escalated during evaluation");

  guard.configure(30, 60, false);
  const auto combined = guard.evaluate(missing, true, t);
  require((combined.reasons & (Guard::missing_source | Guard::uncalibrated)) ==
            (Guard::missing_source | Guard::uncalibrated),
          "simultaneous rejection reasons were lost");
  guard.configure(30, 60, true);
  const auto repeated = guard.evaluate(combined.intent, true, t);
  require((repeated.reasons & combined.reasons) == combined.reasons &&
            !repeated.intent.command.shoot,
          "second evaluation lost submission reasons or restored shoot");

  const auto stopped = guard.evaluate(command, false, t);
  require((stopped.reasons & Guard::disabled) != 0 && !stopped.intent.command.control &&
            !stopped.intent.command.shoot && stopped.intent.command.yaw == 0 &&
            stopped.intent.command.source_time == t && stopped.intent.command.frame_id == 42 &&
            stopped.intent.inhibit_reasons == stopped.reasons,
          "stop diagnostics lost provenance or retained an active command");
  auto future = command;
  future.command.source_time = t + 1ms;
  const auto future_result = guard.evaluate(future, true, t);
  require((future_result.reasons & Guard::future_time) != 0 &&
            !future_result.intent.command.control, "future source was not stopped/diagnosed");
  const auto old = guard.evaluate(command, true, t + 61ms);
  require((old.reasons & (Guard::control_expired | Guard::shoot_expired |
                         Guard::pose_feedback_stale | Guard::status_feedback_stale)) ==
            (Guard::control_expired | Guard::shoot_expired |
             Guard::pose_feedback_stale | Guard::status_feedback_stale) &&
            !old.intent.command.control, "expired command reasons incomplete");

  auto bad_pose = command;
  bad_pose.command.pose_valid = false;
  const auto pose_result = guard.evaluate(bad_pose, true, t);
  require((pose_result.reasons & Guard::pose_invalid) != 0 &&
            !pose_result.intent.command.control, "invalid pose was not stopped/diagnosed");
  auto deadline = command;
  deadline.command.valid_until = t - 1ms;
  const auto deadline_result = guard.evaluate(deadline, true, t);
  require((deadline_result.reasons & Guard::deadline_expired) != 0 &&
            !deadline_result.intent.command.control && deadline_result.intent.command.frame_id == 42,
          "expired explicit deadline was not stopped/diagnosed");

  for (int field = 0; field < 7; ++field) {
    auto bad_number = command;
    double * values[] = {&bad_number.command.yaw, &bad_number.command.pitch,
      &bad_number.command.horizon_distance, &bad_number.yaw_vel, &bad_number.yaw_acc,
      &bad_number.pitch_vel, &bad_number.pitch_acc};
    *values[field] = std::numeric_limits<double>::quiet_NaN();
    const auto result = guard.evaluate(bad_number, true, t);
    require((result.reasons & Guard::nonfinite) != 0 && !result.intent.command.control &&
              result.intent.command.frame_id == 42 && result.intent.command.source_time == t,
            "nonfinite value escaped stop or lost diagnostic provenance");
  }
  guard.observe_feedback(t, false, 0.0);
  const auto bad_feedback = guard.evaluate(command, true, t);
  require((bad_feedback.reasons & (Guard::pose_feedback_invalid | Guard::status_feedback_invalid)) ==
            (Guard::pose_feedback_invalid | Guard::status_feedback_invalid),
          "invalid feedback reasons incomplete");
  guard.invalidate(t + 1ms);
  guard.observe(t + 2ms, true);
  const auto previous = guard.evaluate(command, true, t + 2ms);
  require((previous.reasons & Guard::previous_mode) != 0 && !previous.intent.command.control,
          "previous-mode result was not stopped/diagnosed");
}

enum class InjectedFailure { filter, incomplete_write, throwing_write };
enum class StopOutcome { complete, incomplete, throws };

static void test_publisher_failure(InjectedFailure failure, StopOutcome stop_outcome)
{
  using Publisher = io::ControlPublisher;
  std::mutex mutex;
  std::condition_variable changed;
  int normal_writes = 0, stop_writes = 0;
  Publisher publisher(
    [&](io::ControlIntent intent) {
      if (failure == InjectedFailure::filter) throw std::runtime_error("filter failed");
      return intent;
    },
    [&](const io::ControlIntent & intent) {
      std::lock_guard<std::mutex> lock(mutex);
      if (!intent.command.control) {
        ++stop_writes;
        changed.notify_one();
        if (stop_outcome == StopOutcome::throws) throw std::runtime_error("stop failed");
        return stop_outcome == StopOutcome::complete ? Publisher::WriteResult::complete :
          Publisher::WriteResult::failed;
      }
      ++normal_writes;
      if (failure == InjectedFailure::throwing_write) throw std::runtime_error("write failed");
      // A short write is reported by the transport as the same failed result.
      return Publisher::WriteResult::failed;
    });
  io::ControlIntent command{{true, false, 0.1, 0.2}};
  publisher.publish(command);
  {
    std::unique_lock<std::mutex> lock(mutex);
    require(changed.wait_for(lock, 2s, [&] { return stop_writes != 0; }),
            "publisher failure did not attempt a stop");
  }
  require(publisher.status().failure != nullptr, "publisher did not latch original fault");
  publisher.publish(command);
  publisher.close();
  publisher.close();
  const auto status = publisher.status();
  require(status.stop_attempted && status.stop_written == (stop_outcome == StopOutcome::complete) &&
            bool(status.stop_failure) == (stop_outcome != StopOutcome::complete),
          "stop outcome was recorded incorrectly");
  const std::string expected = failure == InjectedFailure::filter ? "filter failed" :
    failure == InjectedFailure::throwing_write ? "write failed" :
    "Control transport write incomplete";
  require(exception_message(status.failure) == expected,
          "stop failure replaced the original publisher fault");
  if (stop_outcome != StopOutcome::complete)
    require(exception_message(status.stop_failure) ==
              (stop_outcome == StopOutcome::throws ? "stop failed" :
               "Control transport stop write incomplete"),
            "stop error was not retained independently");
  bool rethrown = false;
  try { publisher.rethrow_if_failed(); }
  catch (const std::runtime_error & error) { rethrown = error.what() == expected; }
  require(rethrown, "owner could not retrieve original publisher failure");
  require(normal_writes == (failure == InjectedFailure::filter ? 0 : 1) && stop_writes == 1,
          "fault or repeated close resent a command/stop");
}

static void test_publisher_close_during_filter()
{
  using Publisher = io::ControlPublisher;
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  const auto release_future = release.get_future().share();
  std::atomic<int> active_writes{0}, stops{0};
  Publisher publisher(
    [&](io::ControlIntent intent) {
      entered.set_value();
      require(release_future.wait_for(2s) == std::future_status::ready,
              "test did not release blocked filter");
      return intent;
    },
    [&](const io::ControlIntent & intent) {
      if (intent.command.control) ++active_writes;
      else ++stops;
      return Publisher::WriteResult::complete;
    });
  publisher.publish(io::ControlIntent{{true, false, 0.1, 0.2}});
  require(entered_future.wait_for(2s) == std::future_status::ready,
          "publisher filter did not enter");
  auto closing = std::async(std::launch::async, [&] { publisher.close(); });
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!publisher.status().closed && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool closed_before_release = publisher.status().closed;
  release.set_value();
  require(closing.wait_for(2s) == std::future_status::ready,
          "close did not finish after filter release");
  closing.get();
  require(closed_before_release && active_writes == 0 && stops == 1,
          "close admitted an active command still inside the filter");
  require(!publisher.status().failure && publisher.status().stop_written,
          "normal close during filtering was reported as failure");
}

static void test_publisher_close_stop_failure()
{
  using Publisher = io::ControlPublisher;
  int stops = 0;
  Publisher publisher(
    [](io::ControlIntent intent) { return intent; },
    [&](const io::ControlIntent & intent) {
      require(!intent.command.control && !intent.command.shoot,
              "idle shutdown attempted active control");
      ++stops;
      return Publisher::WriteResult::failed;
    });
  publisher.close();
  publisher.close();
  const auto status = publisher.status();
  require(stops == 1 && status.failure && status.stop_failure && !status.stop_written &&
            exception_message(status.failure) == exception_message(status.stop_failure),
          "close-only stop failure was lost or retried");
}

int main()
{
  test_guard_diagnostics();
  for (const auto failure : {InjectedFailure::filter, InjectedFailure::incomplete_write,
                             InjectedFailure::throwing_write}) {
    for (const auto outcome : {StopOutcome::complete, StopOutcome::incomplete, StopOutcome::throws})
      test_publisher_failure(failure, outcome);
  }
  test_publisher_close_during_filter();
  test_publisher_close_stop_failure();
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

  std::mutex expiry_mutex;
  std::condition_variable expiry_changed;
  bool active = false;
  int writes = 0;
  io::ControlGuard live_guard;
  live_guard.configure(15, 30, true);
  const auto live_time = t;
  live_guard.observe(live_time, true);
  live_guard.observe_pose_feedback(live_time, true);
  live_guard.observe_status_feedback(live_time, 1.0);
  {
    io::ControlPublisher publisher(
      [&](io::ControlIntent intent) {
        std::lock_guard<std::mutex> lock(expiry_mutex);
        return live_guard.apply(intent, true, writes == 0 ? live_time : live_time + 31ms);
      },
      [&](const io::ControlIntent & intent) {
        std::lock_guard<std::mutex> lock(expiry_mutex);
        active = intent.command.control;
        ++writes;
        expiry_changed.notify_one();
        return io::ControlPublisher::WriteResult::complete;
      });
    command.command.source_time = live_time;
    publisher.publish(command);
    std::unique_lock<std::mutex> lock(expiry_mutex);
    require(expiry_changed.wait_for(lock, 2s, [&] { return writes > 1; }),
            "publisher did not recheck a stored command");
    require(writes > 1 && !active, "stalled producer left active command latched");
    lock.unlock();
    publisher.close();
    require(publisher.status().stop_written && !publisher.status().failure,
            "normal publisher close did not record a successful stop write");
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
        return io::ControlPublisher::WriteResult::complete;
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
