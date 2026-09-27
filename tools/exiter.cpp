#include "exiter.hpp"

#include <csignal>
#include <atomic>
#include <stdexcept>

namespace tools
{
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> exit_{false};
bool exiter_inited_ = false;

Exiter::Exiter()
{
  if (exiter_inited_) throw std::runtime_error("Multiple Exiter instances!");
  std::signal(SIGINT, [](int) { exit_ = true; });
  exiter_inited_ = true;
}

bool Exiter::exit() const { return exit_; }

}  // namespace tools
