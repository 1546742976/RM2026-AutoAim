#ifndef AUTO_AIM_MULTITHREAD__HPP
#define AUTO_AIM_MULTITHREAD__HPP

#include <optional>
#include <exception>
#include <functional>

#include "io/cboard.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/plotter.hpp"

namespace auto_aim
{
namespace multithread
{

class CommandGener
{
public:
  struct Input
  {
    std::list<auto_aim::Target> targets_;
    std::chrono::steady_clock::time_point t;
    double bullet_speed;
    Eigen::Vector3d gimbal_pos;
  };
  // Injectable boundaries allow lifecycle/failure tests without opening devices.
  using Compute = std::function<io::Command(const Input &)>;
  using Send = std::function<void(io::Command)>;
  using Enabled = std::function<bool()>;
  CommandGener(Compute compute, Send send, Enabled enabled);
  CommandGener(
    auto_aim::Shooter & shooter, auto_aim::Aimer & aimer, io::CBoard & cboard,
    tools::Plotter & plotter, bool debug = false);

  ~CommandGener() noexcept;
  void close() noexcept;
  void rethrow_if_failed() const;
  std::exception_ptr stop_failure() const;
  void clear();

  void push(
    const std::list<auto_aim::Target> & targets, const std::chrono::steady_clock::time_point & t,
    double bullet_speed, const Eigen::Vector3d & gimbal_pos);

private:
  Compute compute_;
  Send send_;
  Enabled enabled_;
  std::optional<Input> latest_;
  mutable std::mutex mtx_;
  std::mutex close_mutex_;
  std::condition_variable cv_;
  bool stop_ = false, stop_requested_ = false;
  uint64_t generation_ = 0;
  std::exception_ptr failure_, stop_failure_;
  std::thread thread_;

  void generate_command() noexcept;
  void request_stop() noexcept;
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM_MULTITHREAD__HPP
