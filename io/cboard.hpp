#ifndef IO__CBOARD_HPP
#define IO__CBOARD_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <atomic>
#include <memory>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "io/control_guard.hpp"
#include "io/control_publisher.hpp"
#include "tools/pose_history.hpp"
#include "tools/latency_stats.hpp"
#include "io/socketcan.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
enum Mode
{
  idle,
  auto_aim,
  small_buff,
  big_buff,
  outpost
};
const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};

// 哨兵专有
// 双头哨兵，我们不需要
enum ShootMode
{
  left_shoot,
  right_shoot,
  both_shoot
};
const std::vector<std::string> SHOOT_MODES = {"left_shoot", "right_shoot", "both_shoot"};

class CBoard
{
public:
  std::atomic<double> bullet_speed{0};
  std::atomic<Mode> mode{Mode::idle};
  std::atomic<ShootMode> shoot_mode{ShootMode::left_shoot};
  std::atomic<double> ft_angle{0};  //无人机专有

  CBoard(const std::string & config_path);
  ~CBoard();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

  virtual void send(Command command) const;

private:
  tools::PoseHistory pose_history_;
  mutable ControlGuard control_guard_;
  std::unique_ptr<ControlPublisher> publisher_;
  std::unique_ptr<tools::LatencyStats> send_latency_;
  SocketCAN can_;

  int quaternion_canid_, bullet_speed_canid_, send_canid_;

  void callback(const can_frame & frame);
  void write_control(const ControlIntent & intent) const;

  std::string read_yaml(const std::string & config_path);
};

}  // namespace io

#endif  // IO__CBOARD_HPP
