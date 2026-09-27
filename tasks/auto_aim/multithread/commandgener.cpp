#include "commandgener.hpp"

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace multithread
{

CommandGener::CommandGener(Compute compute, Send send, Enabled enabled)
: compute_(std::move(compute)), send_(std::move(send)), enabled_(std::move(enabled)),
  thread_(&CommandGener::generate_command, this)
{}

CommandGener::CommandGener(
  auto_aim::Shooter & shooter, auto_aim::Aimer & aimer, io::CBoard & cboard,
  tools::Plotter & plotter, bool debug)
: CommandGener(
    [&shooter, &aimer, &plotter, debug, t0 = std::chrono::steady_clock::now()](const Input & input) {
      auto command = aimer.aim(input.targets_, input.t, input.bullet_speed);
      command.shoot = shooter.shoot(command, aimer, input.targets_, input.gimbal_pos);
      command.horizon_distance = input.targets_.empty() ? 0 : std::sqrt(
        tools::square(input.targets_.front().ekf_x()[0]) +
        tools::square(input.targets_.front().ekf_x()[2]));
      if (debug) {
        nlohmann::json data;
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
        data["cmd_yaw"] = command.yaw * 57.3;
        data["cmd_pitch"] = command.pitch * 57.3;
        data["shoot"] = command.shoot;
        data["horizon_distance"] = command.horizon_distance;
        plotter.plot(data);
      }
      return command;
    },
    [&cboard](io::Command command) { cboard.send(command); },
    [&cboard] {
      cboard.rethrow_if_failed();
      const auto mode = cboard.mode.load();
      return mode == io::Mode::auto_aim || mode == io::Mode::outpost;
    })
{}

CommandGener::~CommandGener() noexcept { close(); }

void CommandGener::close() noexcept
{
  std::lock_guard<std::mutex> close_lock(close_mutex_);
  {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_ = true;
    latest_.reset();
    ++generation_;
  }
  cv_.notify_all();
  request_stop();
  if (thread_.joinable()) thread_.join();
}

void CommandGener::rethrow_if_failed() const
{
  std::exception_ptr failure;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    failure = failure_;
  }
  if (failure) std::rethrow_exception(failure);
}

std::exception_ptr CommandGener::stop_failure() const
{
  std::lock_guard<std::mutex> lock(mtx_);
  return stop_failure_;
}

void CommandGener::request_stop() noexcept
{
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (stop_requested_) return;
    stop_requested_ = true;
  }
  try { send_({}); }
  catch (...) {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_failure_ = std::current_exception();
    if (!failure_) failure_ = stop_failure_;
  }
}

void CommandGener::clear()
{
  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (stop_) return;
    latest_.reset();
    ++generation_;
    try { send_({}); }
    catch (...) {
      error = std::current_exception();
      if (!failure_) failure_ = error;
      stop_ = true;
    }
  }
  cv_.notify_all();
  if (error) {
    request_stop();
    std::rethrow_exception(error);
  }
}

void CommandGener::push(
  const std::list<auto_aim::Target> & targets, const std::chrono::steady_clock::time_point & t,
  double bullet_speed, const Eigen::Vector3d & gimbal_pos)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (stop_) return;
  latest_ = {targets, t, bullet_speed, gimbal_pos};
  cv_.notify_one();
}

void CommandGener::generate_command() noexcept
{
  try {
    while (true) {
      std::optional<Input> input;
      uint64_t generation;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return stop_ || latest_.has_value(); });
        if (stop_) break;
        input = std::move(latest_);
        latest_.reset();
        generation = generation_;
      }
      io::Command command;
      if (input && enabled_() &&
          tools::delta_time(std::chrono::steady_clock::now(), input->t) < 0.2)
        command = compute_(*input);
      {
        std::lock_guard<std::mutex> lock(mtx_);
        if (generation == generation_ && !stop_) send_(command);
      }
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (!failure_) failure_ = std::current_exception();
      stop_ = true;
      latest_.reset();
      ++generation_;
    }
    cv_.notify_all();
    request_stop();
  }
}

}  // namespace multithread
}  // namespace auto_aim
