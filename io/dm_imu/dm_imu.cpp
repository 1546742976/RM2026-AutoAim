#include "dm_imu.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace io
{
DM_IMU::DM_IMU(QTransform q_transform) : q_transform_(q_transform)
{
  init_serial();
  rec_thread_ = std::thread(&DM_IMU::get_imu_data_thread, this);
  tools::logger()->info("[DM_IMU] initialized");
}

DM_IMU::~DM_IMU()
{
  stop_thread_ = true;
  if (rec_thread_.joinable()) {
    rec_thread_.join();
  }
  send_command(IMU_COMMAND::RESTART);
  if (serial_.isOpen()) {
    serial_.close();
  }
}

void DM_IMU::init_serial()
{
  try {
    serial_.setPort("/dev/ttyACM0");
    serial_.setBaudrate(921600);
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);  //default is parity_none
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial::Timeout time_out = serial::Timeout::simpleTimeout(20);
    serial_.setTimeout(time_out);
    serial_.open();
    send_command(IMU_COMMAND::ZERO_ANGLE);
    send_command(IMU_COMMAND::ENTER_SET_MODE);
    send_command(IMU_COMMAND::ENABLE_TEMP_CONTROL);
    send_command(IMU_COMMAND::SET_TEMP, 25);
    send_command(IMU_COMMAND::ENTER_NORMAL_MODE);
    // send_command(IMU_COMMAND::ZERO_ANGLE);
    usleep(1000000);  //1s

    tools::logger()->info("[DM_IMU] serial port opened");
  }

  catch (serial::IOException & e) {
    tools::logger()->warn("[DM_IMU] failed to open serial port ");
    exit(0);
  }
}

void DM_IMU::get_imu_data_thread()
{
  uint8_t header[4];
  while (!stop_thread_) {
    if (!serial_.isOpen()) {
      tools::logger()->warn("In get_imu_data_thread,imu serial port unopen");
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }

    try {
      // 逐字节读取寻找帧头 0x55
      if (serial_.read(header, 1) != 1) continue;
      if (header[0] != 0x55) continue;

      // 读取后续3字节校验头部
      if (serial_.read(header + 1, 3) != 3) continue;

      if (header[1] != 0xAA /*|| header[2] != 0x9B*/ || header[3] != 0x01) {
        continue;
      }

      // 头部校验通过，填充结构体
      receive_data.FrameHeader1 = header[0];
      receive_data.flag1 = header[1];
      receive_data.slave_id1 = header[2];
      receive_data.reg_acc = header[3];

      // 读取剩余数据 (57 - 4 = 53 bytes)
      if (serial_.read((uint8_t *)(&receive_data.accx_u32), 53) != 53) continue;

      std::lock_guard<std::mutex> data_lock(data_mutex_);
      uint8_t current_idx = data_idx_.load(std::memory_order_relaxed);
      IMU_Data temp_data = data_buffer_[current_idx];

      if (tools::get_imu_crc16((uint8_t *)(&receive_data.FrameHeader1), 16) == receive_data.crc1) {
        temp_data.accx = *((float *)(&receive_data.accx_u32));
        temp_data.accy = *((float *)(&receive_data.accy_u32));
        temp_data.accz = *((float *)(&receive_data.accz_u32));
      }
      if (tools::get_imu_crc16((uint8_t *)(&receive_data.FrameHeader2), 16) == receive_data.crc2) {
        temp_data.gyrox = *((float *)(&receive_data.gyrox_u32));
        temp_data.gyroy = *((float *)(&receive_data.gyroy_u32));
        temp_data.gyroz = *((float *)(&receive_data.gyroz_u32));
      }
      const bool angles_valid = tools::get_imu_crc16((uint8_t *)(&receive_data.FrameHeader3), 16) == receive_data.crc3;
      if (angles_valid) {
        temp_data.roll = *((float *)(&receive_data.roll_u32));
        temp_data.pitch = *((float *)(&receive_data.pitch_u32));
        temp_data.yaw = *((float *)(&receive_data.yaw_u32));
      }
      
      uint8_t next_idx = 1 - current_idx;
      data_buffer_[next_idx] = temp_data;
      data_idx_.store(next_idx, std::memory_order_release);

      if (!angles_valid) continue;  // Never stamp previous angles as a new sample.

      auto timestamp = std::chrono::steady_clock::now();
      Eigen::Quaterniond q = Eigen::AngleAxisd(temp_data.yaw * M_PI / 180, Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(temp_data.pitch * M_PI / 180, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(temp_data.roll * M_PI / 180, Eigen::Vector3d::UnitX());
      q.normalize();
      q = q_transform_(q);
      q.normalize();
      pose_history_.push(q, timestamp);
    } catch (const std::exception & e) { 
      tools::logger()->error("[DM_IMU] error: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

Eigen::Quaterniond DM_IMU::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  auto sample = pose_history_.at(timestamp);
  return sample ? sample->q : tools::PoseHistory::invalid();
}

IMU_Data DM_IMU::current_data() const {
  std::lock_guard<std::mutex> lock(data_mutex_);
  uint8_t read_idx = data_idx_.load(std::memory_order_acquire);
  return data_buffer_[read_idx]; 
}

void DM_IMU::send_command(IMU_COMMAND command, std::optional<uint8_t> param) {
  uint8_t cmd_buffer[4] = {0xAA, 0x00, 0x00, 0x0D};  // 基础命令格式
  
  uint16_t cmd_val = static_cast<uint16_t>(command);
  cmd_buffer[1] = (cmd_val >> 8) & 0xFF;
  cmd_buffer[2] = (cmd_val & 0xFF) | param.value_or(0);
  
  if (serial_.isOpen()) {
    serial_.write(cmd_buffer, 4);
  }
}

void DM_IMU::forward_data(const serial::Serial & serial) const {
  IMU_Forward_Frame forward_frame;
  // 直接从当前最新的索引中读取数据，无需加锁，不阻塞写入线程
  forward_frame.data = current_data();
  
  forward_frame.crc16 = tools::get_crc16((uint8_t *)(&forward_frame.header), sizeof(IMU_Forward_Frame) - sizeof(uint16_t));
  const_cast<serial::Serial &>(serial).write(reinterpret_cast<const uint8_t *>(&forward_frame), sizeof(IMU_Forward_Frame));
}

}  // namespace io
