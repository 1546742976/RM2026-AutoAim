#ifndef IO__CONTROL_PUBLISHER_HPP
#define IO__CONTROL_PUBLISHER_HPP

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include "io/command.hpp"

namespace io
{
// One transport writer, latest command wins. A stalled producer still expires.
class ControlPublisher
{
public:
  // Host write completion only; this is not an acknowledgement from the device.
  enum class WriteResult { complete, failed };
  struct Status {
    std::exception_ptr failure;
    std::exception_ptr stop_failure;
    bool closed = false;
    bool stop_attempted = false;
    bool stop_written = false;
  };
  using Filter = std::function<ControlIntent(ControlIntent)>;
  using Write = std::function<WriteResult(const ControlIntent &)>;
  ControlPublisher(Filter filter, Write write) : filter_(std::move(filter)), write_(std::move(write)),
    worker_([this] { run(); }) {}
  ~ControlPublisher() noexcept { close(); }
  void publish(ControlIntent intent)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || status_.failure) return;
    latest_ = std::move(intent);
    dirty_ = true;
    changed_.notify_one();
  }
  Status status() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = status_;
    result.closed = closed_;
    return result;
  }
  void rethrow_if_failed() const
  {
    const auto state = status();
    if (state.failure) std::rethrow_exception(state.failure);
  }
  // Called by owners, never by a filter/write callback. Serialize concurrent closers.
  void close() noexcept
  {
    std::lock_guard<std::mutex> close_lock(close_mutex_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
      latest_ = {};
      dirty_ = false;
    }
    changed_.notify_one();
    if (worker_.joinable()) worker_.join();
  }
private:
  Filter filter_;
  Write write_;
  mutable std::mutex mutex_;
  std::mutex close_mutex_;
  std::condition_variable changed_;
  ControlIntent latest_{};
  bool dirty_ = false, closed_ = false;
  Status status_;
  std::thread worker_;
  void run() noexcept
  {
    using Clock = std::chrono::steady_clock;
    auto next_send = Clock::now();
    bool was_active = false;
    try {
    while (true) {
      ControlIntent intent;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait_until(lock, next_send, [&] { return closed_; });
        if (closed_) break;
        if (!dirty_ && !was_active) {
          changed_.wait(lock, [&] { return dirty_ || closed_; });
          if (closed_) break;
        }
        intent = latest_;
        dirty_ = false;
      }
      intent = filter_(intent);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        // A close requested during filtering must not admit this cached command.
        // A transport write already admitted here may finish before the final stop.
        if (closed_) break;
      }
      if (write_(intent) != WriteResult::complete)
        throw std::runtime_error("Control transport write incomplete");
      was_active = intent.command.control;
      next_send = Clock::now() + std::chrono::milliseconds(10);
    }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      status_.failure = std::current_exception();
      closed_ = true;
      latest_ = {};
      dirty_ = false;
    }
    // Exactly one best-effort stop, including when the normal write failed.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      status_.stop_attempted = true;
    }
    try {
      if (write_({}) != WriteResult::complete)
        throw std::runtime_error("Control transport stop write incomplete");
      std::lock_guard<std::mutex> lock(mutex_);
      status_.stop_written = true;
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      status_.stop_failure = std::current_exception();
      if (!status_.failure) status_.failure = status_.stop_failure;
    }
  }
};
}  // namespace io
#endif
