#include "commandgener.hpp"

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace multithread
{

CommandGener::CommandGener(
  auto_aim::Shooter & shooter, auto_aim::Aimer & aimer, io::CBoard & cboard,
  tools::Plotter & plotter, bool debug)
: shooter_(shooter), aimer_(aimer), cboard_(cboard), plotter_(plotter), stop_(false), debug_(debug)
{
  thread_ = std::thread(&CommandGener::generate_command, this);
}

CommandGener::~CommandGener()
{
  {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  cboard_.send({});
}

void CommandGener::clear()
{
  std::lock_guard<std::mutex> lock(mtx_);
  latest_.reset();
  ++generation_;
  cboard_.send({});
  cv_.notify_all();
}

void CommandGener::push(
  const std::list<auto_aim::Target> & targets, const std::chrono::steady_clock::time_point & t,
  double bullet_speed, const Eigen::Vector3d & gimbal_pos)
{
  std::lock_guard<std::mutex> lock(mtx_);
  latest_ = {targets, t, bullet_speed, gimbal_pos};
  cv_.notify_one();
}

void CommandGener::generate_command()
{
  auto t0 = std::chrono::steady_clock::now();
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
    if (input && (cboard_.mode.load() == io::Mode::auto_aim || cboard_.mode.load() == io::Mode::outpost) &&
        tools::delta_time(std::chrono::steady_clock::now(), input->t) < 0.2) {
      auto command = aimer_.aim(input->targets_, input->t, input->bullet_speed);
      command.shoot = shooter_.shoot(command, aimer_, input->targets_, input->gimbal_pos);
      command.horizon_distance = input->targets_.empty()
                                   ? 0
                                   : std::sqrt(
                                       tools::square(input->targets_.front().ekf_x()[0]) +
                                       tools::square(input->targets_.front().ekf_x()[2]));
      {
        std::lock_guard<std::mutex> lock(mtx_);
        if (generation == generation_ && !stop_) cboard_.send(command);
      }
      if (debug_) {
        nlohmann::json data;
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
        data["cmd_yaw"] = command.yaw * 57.3;
        data["cmd_pitch"] = command.pitch * 57.3;
        data["shoot"] = command.shoot;
        data["horizon_distance"] = command.horizon_distance;
        plotter_.plot(data);
      }
    } else {
      std::lock_guard<std::mutex> lock(mtx_);
      if (generation == generation_) cboard_.send({});
    }
  }
}

}  // namespace multithread

}  // namespace auto_aim
