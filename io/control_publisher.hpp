#ifndef IO__CONTROL_PUBLISHER_HPP
#define IO__CONTROL_PUBLISHER_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include "io/command.hpp"

namespace io
{
// One transport writer, latest command wins. A stalled producer still expires.
class ControlPublisher
{
public:
  using Filter = std::function<ControlIntent(ControlIntent)>;
  using Write = std::function<void(const ControlIntent &)>;
  ControlPublisher(Filter filter, Write write) : filter_(std::move(filter)), write_(std::move(write)),
    worker_([this] { run(); }) {}
  ~ControlPublisher() { close(); }
  void publish(ControlIntent intent)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    latest_ = std::move(intent);
    dirty_ = true;
    changed_.notify_one();
  }
  void close()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    changed_.notify_one();
    if (worker_.joinable()) worker_.join();
  }
private:
  Filter filter_;
  Write write_;
  std::mutex mutex_;
  std::condition_variable changed_;
  ControlIntent latest_{};
  bool dirty_ = false, closed_ = false;
  std::thread worker_;
  void run()
  {
    using Clock = std::chrono::steady_clock;
    auto next_send = Clock::now();
    bool was_active = false;
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
      write_(intent);
      was_active = intent.command.control;
      next_send = Clock::now() + std::chrono::milliseconds(10);
    }
    write_({});
  }
};
}  // namespace io
#endif
