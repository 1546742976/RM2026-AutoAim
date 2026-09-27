#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;
void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}
int main()
{
  try {
    tools::ThreadSafeQueue<int, true> latest(1);
    latest.push(1); latest.push(2); latest.push(3);
    int value = 0;
    require(latest.size() == 1 && latest.try_pop(value) && value == 3,
      "A latest-frame queue must replace stale entries");
    require(!latest.pop_for(value, 2ms), "Timed read should return when there is no new frame");
    auto waiter = std::async(std::launch::async, [&] { return latest.pop(value); });
    latest.close();
    require(waiter.wait_for(100ms) == std::future_status::ready && !waiter.get(),
      "Closing a queue must release blocked consumers");
    require(!latest.push(4), "Closed queue must reject frames");

    tools::ThreadSafeQueue<std::unique_ptr<int>, true> owned(1);
    owned.push(std::make_unique<int>(5));
    owned.push(std::make_unique<int>(6));
    std::unique_ptr<int> item;
    require(owned.try_pop(item) && *item == 6, "Queue must support owned move-only buffers");

    tools::ThreadSafeQueue<int> * inspect = nullptr;
    std::size_t observed_size = 0;
    tools::ThreadSafeQueue<int> bounded(1, [&] { observed_size = inspect->size(); });
    inspect = &bounded;
    bounded.push(7);
    require(!bounded.push(8) && observed_size == 1, "Overflow handler must run outside queue lock");
    bounded.close();
    require(bounded.pop(value) && value == 7 && !bounded.pop(value),
      "Close must preserve already queued values for draining");
    std::cout << "queue contracts passed\n";
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
