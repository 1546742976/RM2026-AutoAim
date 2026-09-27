#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>

namespace tools
{
template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue(
    size_t max_size, std::function<void(void)> full_handler = [] {})
  : max_size_(max_size), full_handler_(full_handler)
  {
    if (max_size_ == 0) throw std::invalid_argument("Queue capacity must be positive");
  }

  bool push(const T & value) { return push_impl(value, PopWhenFull); }

  bool push(T && value) { return push_impl(std::move(value), PopWhenFull); }

  // Camera/command mailboxes can explicitly replace the oldest queued value.
  bool push_latest(const T & value) { return push_impl(value, true); }

  bool push_latest(T && value) { return push_impl(std::move(value), true); }

  bool pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    return take_front(value);
  }

  bool try_pop(T & value)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return take_front(value);
  }

  template <typename Rep, typename Period>
  bool pop_for(T & value, const std::chrono::duration<Rep, Period> & timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_empty_condition_.wait_for(
          lock, timeout, [this] { return closed_ || !queue_.empty(); }))
      return false;
    return take_front(value);
  }

  T pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) throw std::runtime_error("Queue is closed");

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  T front()
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) throw std::runtime_error("Queue is closed");

    return queue_.front();
  }

  void back(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      std::cerr << "Error: Attempt to access the back of an empty queue." << std::endl;
      return;
    }

    value = queue_.back();
  }

  bool empty() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  // Existing entries remain drainable, but no new entries are accepted.
  void close()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_condition_.notify_all();
  }

  bool closed() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  void clear()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      queue_.pop();
    }
  }

private:
  template <typename U>
  bool push_impl(U && value, bool pop_when_full)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) return false;
    if (queue_.size() >= max_size_) {
      if (pop_when_full) {
        queue_.pop();
      } else {
        // A handler may inspect this queue; never invoke it under the mutex.
        lock.unlock();
        full_handler_();
        return false;
      }
    }
    queue_.push(std::forward<U>(value));
    lock.unlock();
    not_empty_condition_.notify_one();
    return true;
  }

  bool take_front(T & value)
  {
    if (queue_.empty()) return false;
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  std::queue<T> queue_;
  size_t max_size_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::function<void(void)> full_handler_;
  bool closed_ = false;
};

}  // namespace tools

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP
