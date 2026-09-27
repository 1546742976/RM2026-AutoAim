#ifndef IO__Dm_Imu_HPP
#define IO__Dm_Imu_HPP

#include <math.h>
#include <serial/serial.h>

#include <Eigen/Geometry>
#include <array>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <thread>
#include <functional>
#include <optional>

#include "tools/thread_safe_queue.hpp"
#include "tools/pose_history.hpp"

namespace io
{
// IMU数据接收帧格式
#pragma pack(push, 1)
struct IMU_Receive_Frame
{
  uint8_t FrameHeader1;  // 0x55
  uint8_t flag1;         // 0xAA
  uint8_t slave_id1;     // ID
  uint8_t reg_acc;       // 0x01
  uint32_t accx_u32;     // 加速度X
  uint32_t accy_u32;     // 加速度Y
  uint32_t accz_u32;     // 加速度Z
  uint16_t crc1;         // CRC16校验码
  uint8_t FrameEnd1;     // 0xAA

  uint8_t FrameHeader2;  // ?
  uint8_t flag2;
  uint8_t slave_id2;
  uint8_t reg_gyro;    // 0x02
  uint32_t gyrox_u32;  // 角速度X
  uint32_t gyroy_u32;  // 角速度Y
  uint32_t gyroz_u32;  // 角速度Z
  uint16_t crc2;       // 128bit长度 len=16B
  uint8_t FrameEnd2;

  uint8_t FrameHeader3;
  uint8_t flag3;
  uint8_t slave_id3;
  uint8_t reg_euler;   // 0x03
  uint32_t roll_u32;   // 欧拉角roll
  uint32_t pitch_u32;  // 欧拉角pitch
  uint32_t yaw_u32;    // 欧拉角yaw
  uint16_t crc3;
  uint8_t FrameEnd3;
};
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
  float accx;   // 加速度X
  float accy;   // 加速度Y
  float accz;   // 加速度Z
  float gyrox;  // 角速度X
  float gyroy;  // 角速度Y
  float gyroz;  // 角速度Z
  float roll;   // 欧拉角roll
  float pitch;  // 欧拉角pitch
  float yaw;    // 欧拉角yaw
} IMU_Data;
#pragma pack(pop)

#pragma pack(push, 1)
struct IMU_Forward_Frame
{
  const uint8_t header = 0x55;
  IMU_Data data;
  uint16_t crc16;  // CRC16校验码
};
#pragma pack(pop)

class DM_IMU
{
public:
  using QTransform = std::function<Eigen::Quaterniond(Eigen::Quaterniond)>;

  DM_IMU(QTransform q_transform = [](Eigen::Quaterniond q) { return q; });
  ~DM_IMU();

  enum class IMU_COMMAND : uint16_t
  {
    RESTART              = 0x0000, // 重启IMU 
    DISABLE_RS485_ACTIVE = 0x0103, // 关闭485主动模式
    DISABLE_ACC_OUTPUT   = 0x0104, // 关闭加速度输出
    DISABLE_GYRO_OUTPUT  = 0x0105, // 关闭角速度输出
    DISABLE_EULER_OUTPUT = 0x0106, // 关闭欧拉角输出
    DISABLE_QUAT_OUTPUT  = 0x0107, // 关闭四元数输出
    DISABLE_CAN_ACTIVE   = 0x0108, // 关闭CAN主动模式
    ENABLE_RS485_ACTIVE  = 0x0113, // 开启485主动模式
    ENABLE_ACC_OUTPUT    = 0x0114, // 开启加速度输出
    ENABLE_GYRO_OUTPUT   = 0x0115, // 开启角速度输出
    ENABLE_EULER_OUTPUT  = 0x0116, // 开启欧拉角输出
    ENABLE_QUAT_OUTPUT   = 0x0117, // 开启四元数输出
    ENABLE_CAN_ACTIVE    = 0x0118, // 开启CAN主动模式
    SAVE_PARAMS          = 0x0301, // 保存参数
    START_GYRO_CALIBRATE = 0x0302, // 启动陀螺静态校准
    START_ACC_CALIBRATE  = 0x0303, // 启动加计六面校准
    DISABLE_TEMP_CONTROL = 0x0400, // 关闭温度控制
    ENABLE_TEMP_CONTROL  = 0x0401, // 开启温度控制
    SET_TEMP             = 0x0500, // 设置控制温度
    ENTER_NORMAL_MODE    = 0x0600, // 进入正常模式
    ENTER_SET_MODE       = 0x0601, // 进入设置模式
    SET_CAN_ID           = 0x0800, // 设置CANID
    SET_MST_ID           = 0x0900, // 设置MSTID
    SET_OUTPUT_INTERFACE = 0x0A00, // 设置输出接口
    RESTORE_FACTORY      = 0x0B01, // 恢复出厂设置
    ZERO_ANGLE           = 0x0C01  // 角度置零
  };
  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
  IMU_Data current_data() const;

  // IMU operations
  void send_command(IMU_COMMAND command, std::optional<uint8_t> param = std::nullopt);
  void forward_data(const serial::Serial & serial) const;

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  void init_serial();
  void get_imu_data_thread();

  serial::Serial serial_;
  std::thread rec_thread_;

  tools::PoseHistory pose_history_;
  mutable std::mutex data_mutex_;

  std::atomic<bool> stop_thread_{false};
  IMU_Receive_Frame receive_data{};  //receive data frame
  
  // 双缓冲区，实现无锁的读写分离
  IMU_Data data_buffer_[2]{};
  std::atomic<uint8_t> data_idx_{0};
  QTransform q_transform_;
};

}  // namespace io

#endif
