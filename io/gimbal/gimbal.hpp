#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <optional>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"
#include "io/command.hpp"
#include "io/control_guard.hpp"
#include "io/control_publisher.hpp"
#include "tools/pose_history.hpp"
#include "tools/latency_stats.hpp"
#include "io/dm_imu/dm_imu.hpp"


namespace io
{
constexpr uint8_t gimbal_struct_header[2] = {'A', 'B'};
constexpr uint8_t nav_struct_header[2] = {'C', 'D'};
using RefereeCallback = std::function<void(uint16_t cmd_id, const uint8_t* data, uint16_t len)>;

#pragma pack(push, 1)
struct NavData
{
  uint8_t head[2] = {nav_struct_header[0], nav_struct_header[1]};
  uint8_t mode;  // 0：正常， 1：小陀螺
  uint8_t empty1;
  uint8_t empty2;
  double linear_x;
  double linear_y;
  double linear_z;
  double angular_x;
  double angular_y;
  double angular_z;
  uint16_t crc16;
};
#pragma pack(pop)
static_assert(sizeof(NavData) <= 64);

#pragma pack(push, 1)
struct GimbalToVision
{
  uint8_t head[2] = {gimbal_struct_header[0], gimbal_struct_header[1]};
  uint8_t mode;  // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
  float q[4];    // wxyz顺序
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;  // 子弹累计发送次数
  uint16_t crc16;
};
#pragma pack(pop)
static_assert(sizeof(GimbalToVision) <= 64);

#pragma pack(push, 1)
struct VisionToGimbal
{
  const int8_t head[2] = {gimbal_struct_header[0], gimbal_struct_header[1]};
  uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火，2: 控制云台且开火
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  uint16_t crc16;
};
#pragma pack(pop)

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
  std::chrono::steady_clock::time_point timestamp{};
};

class Gimbal
{
public:
  // 裁判系统回调函数类型：参数依次为 cmd_id, 数据指针, 数据长度
  using RefereeCallback = std::function<void(uint16_t cmd_id, const uint8_t* data, uint16_t len)>;
  using NavRefereeCallback = std::function<void(uint16_t cmd_id, const std::vector<uint8_t>& data)>;
  using AfterSendGimbalData = std::function<void(void)>;

  Gimbal(const std::string & config_path);
  Gimbal(const std::string & config_path, AfterSendGimbalData after_send_gimbal_data, RefereeCallback referee_callback, NavRefereeCallback nav_referee_callback);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);
  void observe_frame(std::chrono::steady_clock::time_point t, const Eigen::Quaterniond & q);
  void send(io::ControlIntent intent);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  void send(const io::Command & cmd);
  void send(io::VisionToGimbal VisionToGimbal);
  void send_cmd_vel(const std::optional<const NavData> & nav_data);
  void send_imu_forward(const DM_IMU & imu) const;

private:
  serial::Serial serial_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;
  mutable std::mutex send_mutex_;
  io::ControlGuard control_guard_;
  tools::PoseHistory pose_history_;
  std::unique_ptr<ControlPublisher> publisher_;
  std::unique_ptr<tools::LatencyStats> send_latency_;
  bool use_packet_quaternion_ = false;

  VisionToGimbal tx_data_gimbal;
  NavData tx_data_nav;

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_{};

  bool read(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
  void parse_referee_data(uint16_t cmd_id, const uint8_t* data, uint16_t len);
  bool send_gimbal_data() const;
  void write_control(const io::ControlIntent & intent);

  RefereeCallback referee_callback_;
  NavRefereeCallback nav_referee_callback_;
  AfterSendGimbalData after_send_gimbal_data_;
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
