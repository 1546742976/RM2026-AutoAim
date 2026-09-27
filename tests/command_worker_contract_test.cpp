#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "tasks/auto_aim/multithread/commandgener.hpp"

using namespace std::chrono_literals;
using Worker = auto_aim::multithread::CommandGener;

static void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

static std::string exception_message(std::exception_ptr failure)
{
  if (!failure) return {};
  try { std::rethrow_exception(failure); }
  catch (const std::exception & error) { return error.what(); }
  catch (...) { return "non-standard exception"; }
}

static std::string worker_failure(const Worker & worker)
{
  try { worker.rethrow_if_failed(); }
  catch (...) { return exception_message(std::current_exception()); }
  return {};
}

static void push(Worker & worker, int marker = 1)
{
  // The marker is test data only; callbacks perform no target or ballistic calculation.
  worker.push({}, std::chrono::steady_clock::now(), 1.0, Eigen::Vector3d(marker, 0, 0));
}

static io::Command result(const Worker::Input & input)
{
  io::Command command;
  command.control = true;
  command.shoot = false;
  command.frame_id = static_cast<uint64_t>(input.gimbal_pos.x());
  return command;
}

enum class FailurePoint { enabled, compute, send };

static void test_worker_failure(FailurePoint point, bool fail_stop)
{
  std::mutex mutex;
  std::condition_variable changed;
  int stops = 0, active_sends = 0;
  std::atomic<int> computes{0};
  Worker worker(
    [&](const Worker::Input & input) {
      ++computes;
      if (point == FailurePoint::compute) throw std::runtime_error("compute failed");
      return result(input);
    },
    [&](io::Command command) {
      std::lock_guard<std::mutex> lock(mutex);
      if (command.control) {
        ++active_sends;
        throw std::runtime_error("send failed");
      }
      ++stops;
      changed.notify_one();
      if (fail_stop) throw std::runtime_error("stop failed");
    },
    [&] {
      if (point == FailurePoint::enabled) throw std::runtime_error("enabled failed");
      return true;
    });
  push(worker);
  {
    std::unique_lock<std::mutex> lock(mutex);
    require(changed.wait_for(lock, 2s, [&] { return stops != 0; }),
            "worker fault did not request stop");
  }
  push(worker, 2);
  worker.close();
  worker.close();
  const std::string expected = point == FailurePoint::enabled ? "enabled failed" :
    point == FailurePoint::compute ? "compute failed" : "send failed";
  require(worker_failure(worker) == expected, "worker did not preserve its original exception");
  require(exception_message(worker.stop_failure()) == (fail_stop ? "stop failed" : ""),
          "worker stop failure was not separately observable");
  require(computes == (point == FailurePoint::enabled ? 0 : 1),
          "worker computed new input after its failure");
  require(active_sends == (point == FailurePoint::send ? 1 : 0) && stops == 1,
          "worker retried a normal command or repeated terminal stop");
}

static void test_clear_suppresses_inflight_result()
{
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  const auto release_future = release.get_future().share();
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<uint64_t> emitted;
  int stops = 0;
  Worker worker(
    [&](const Worker::Input & input) {
      if (input.gimbal_pos.x() == 1) {
        entered.set_value();
        require(release_future.wait_for(2s) == std::future_status::ready,
                "test did not release compute after clear");
      }
      return result(input);
    },
    [&](io::Command command) {
      std::lock_guard<std::mutex> lock(mutex);
      if (command.control) emitted.push_back(command.frame_id);
      else ++stops;
      changed.notify_one();
    }, [] { return true; });
  push(worker);
  require(entered_future.wait_for(2s) == std::future_status::ready,
          "worker computation did not enter before clear");
  worker.clear();
  push(worker, 2);
  release.set_value();
  {
    std::unique_lock<std::mutex> lock(mutex);
    require(changed.wait_for(lock, 2s, [&] { return !emitted.empty(); }),
            "worker did not accept a new generation after clear");
  }
  worker.close();
  require(emitted.size() == 1 && emitted.front() == 2 && stops == 2,
          "clear allowed stale result or prevented new generation");
  require(worker_failure(worker).empty(), "normal clear/close created a worker failure");
}

static void test_close_suppresses_inflight_result()
{
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  const auto release_future = release.get_future().share();
  std::mutex mutex;
  std::condition_variable changed;
  int stops = 0, active_sends = 0;
  std::atomic<int> computes{0};
  Worker worker(
    [&](const Worker::Input & input) {
      ++computes;
      entered.set_value();
      require(release_future.wait_for(2s) == std::future_status::ready,
              "test did not release compute after close");
      return result(input);
    },
    [&](io::Command command) {
      std::lock_guard<std::mutex> lock(mutex);
      if (command.control) ++active_sends;
      else ++stops;
      changed.notify_one();
    }, [] { return true; });
  push(worker);
  require(entered_future.wait_for(2s) == std::future_status::ready,
          "worker computation did not enter before close");
  auto closing = std::async(std::launch::async, [&] { worker.close(); });
  bool stop_observed;
  {
    std::unique_lock<std::mutex> lock(mutex);
    stop_observed = changed.wait_for(lock, 2s, [&] { return stops != 0; });
  }
  push(worker, 2);
  release.set_value();
  require(closing.wait_for(2s) == std::future_status::ready,
          "worker close did not finish after computation returned");
  closing.get();
  worker.close();
  require(stop_observed && active_sends == 0 && stops == 1 && computes == 1,
          "close admitted in-flight result or accepted later input");
  require(worker_failure(worker).empty(), "normal in-flight close created a worker failure");
}

static void test_clear_stop_failure(bool recovery_stop_fails)
{
  std::atomic<int> computes{0};
  int stops = 0;
  Worker worker(
    [&](const Worker::Input & input) { ++computes; return result(input); },
    [&](io::Command command) {
      require(!command.control, "clear failure admitted a normal command");
      ++stops;
      if (stops == 1) throw std::runtime_error("clear stop failed");
      if (recovery_stop_fails) throw std::runtime_error("recovery stop failed");
    }, [] { return true; });
  std::string clear_error;
  try { worker.clear(); }
  catch (...) { clear_error = exception_message(std::current_exception()); }
  push(worker);
  worker.close();
  worker.close();
  require(clear_error == "clear stop failed" && worker_failure(worker) == clear_error,
          "clear did not latch and rethrow its first send failure");
  require(exception_message(worker.stop_failure()) ==
            (recovery_stop_fails ? "recovery stop failed" : ""),
          "clear recovery stop error replaced/lost original failure");
  require(stops == 2 && computes == 0,
          "failed clear accepted input or repeatedly attempted stop");
}

static void test_idle_close_stop_failure()
{
  int stops = 0;
  Worker worker(
    [](const Worker::Input & input) { return result(input); },
    [&](io::Command command) {
      require(!command.control && !command.shoot, "idle close issued active command");
      ++stops;
      throw std::runtime_error("idle stop failed");
    }, [] { return true; });
  worker.close();
  push(worker);
  worker.close();
  require(stops == 1 && worker_failure(worker) == "idle stop failed" &&
            exception_message(worker.stop_failure()) == "idle stop failed",
          "idle close stop failure was not observable or was retried");
}

int main()
{
  for (const auto point : {FailurePoint::enabled, FailurePoint::compute, FailurePoint::send}) {
    test_worker_failure(point, false);
    test_worker_failure(point, true);
  }
  test_clear_suppresses_inflight_result();
  test_close_suppresses_inflight_result();
  test_clear_stop_failure(false);
  test_clear_stop_failure(true);
  test_idle_close_stop_failure();
  std::cout << "command worker contracts passed\n";
}
