#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

#ifndef NDEBUG
#include "tools/plotter.hpp"
#endif

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
: Gimbal(config_path, nullptr, nullptr, nullptr)
{}

Gimbal::Gimbal(const std::string & config_path, AfterSendGimbalData after_send_gimbal_data, RefereeCallback referee_callback, NavRefereeCallback nav_referee_callback)
: after_send_gimbal_data_(std::move(after_send_gimbal_data)), referee_callback_(std::move(referee_callback)), nav_referee_callback_(std::move(nav_referee_callback))
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  control_guard_.configure(yaml["shoot_max_age_ms"].as<double>(100),
                           yaml["control_max_age_ms"].as<double>(200));
  use_packet_quaternion_ = yaml["gimbal_use_packet_quaternion"].as<bool>(false);
  send_latency_ = std::make_unique<tools::LatencyStats>(yaml["runtime_metrics"].as<bool>(false));

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(115200);
    serial::Timeout to = serial::Timeout::simpleTimeout(1000);
    serial_.setTimeout(to);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  publisher_ = std::make_unique<ControlPublisher>(
    [this](ControlIntent intent) {
      auto result = control_guard_.apply(intent, mode() != GimbalMode::IDLE);
      const auto feedback = state();
      if (!control_guard_.feedback_fresh(feedback.timestamp) ||
          !std::isfinite(feedback.yaw) || !std::isfinite(feedback.pitch) ||
          !std::isfinite(feedback.bullet_speed) || feedback.bullet_speed <= 0)
        result.command.shoot = false;
      return result;
    },
    [this](const ControlIntent & intent) { write_control(intent); });
  thread_ = std::thread(&Gimbal::read_thread, this);

  tools::logger()->info("[Gimbal] Waiting for timestamped poses in the receive thread.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  publisher_->close();
  if (send_latency_->enabled()) {
    const auto stats = send_latency_->summary();
    tools::logger()->info("[Gimbal] estimated exposure-to-send: n={} window={} p50={:.2f} p95={:.2f} max={:.2f} ms",
      stats.total_samples, stats.samples, stats.p50_ms, stats.p95_ms, stats.max_ms);
  }
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  auto sample = pose_history_.at(t);
  auto pose = sample ? sample->q : tools::PoseHistory::invalid();
  observe_frame(t, pose);
  return pose;
}

void Gimbal::observe_frame(std::chrono::steady_clock::time_point t, const Eigen::Quaterniond & q)
{
  control_guard_.observe(t, q.coeffs().allFinite() && q.norm() > 1e-9);
}

void Gimbal::send(const io::Command & cmd) {
  send(io::ControlIntent{cmd});
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  send(VisionToGimbal.mode != 0, VisionToGimbal.mode == 2, VisionToGimbal.yaw,
       VisionToGimbal.yaw_vel, VisionToGimbal.yaw_acc, VisionToGimbal.pitch,
       VisionToGimbal.pitch_vel, VisionToGimbal.pitch_acc);
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  send(io::ControlIntent{{control, fire, yaw, pitch}, yaw_vel, yaw_acc, pitch_vel, pitch_acc});
}

void Gimbal::send(io::ControlIntent intent)
{
  intent = control_guard_.apply(intent, mode() != GimbalMode::IDLE);
  publisher_->publish(std::move(intent));
}

void Gimbal::write_control(const io::ControlIntent & intent)
{
  std::unique_lock<std::mutex> send_lock(send_mutex_);
  tx_data_gimbal.mode = intent.command.control ? (intent.command.shoot ? 2 : 1) : 0;
  tx_data_gimbal.yaw = intent.command.yaw;
  tx_data_gimbal.yaw_vel = intent.yaw_vel;
  tx_data_gimbal.yaw_acc = intent.yaw_acc;
  tx_data_gimbal.pitch = intent.command.pitch;
  tx_data_gimbal.pitch_vel = intent.pitch_vel;
  tx_data_gimbal.pitch_acc = intent.pitch_acc;
  tx_data_gimbal.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_gimbal), sizeof(tx_data_gimbal) - sizeof(tx_data_gimbal.crc16));
  if (send_gimbal_data()) send_latency_->record_first_send(intent.command.source_time);
  send_lock.unlock();
  if (after_send_gimbal_data_) after_send_gimbal_data_();
}

void Gimbal::send_cmd_vel(const std::optional<const NavData> & nav_data)
{
  if (!nav_data.has_value()) return;
  std::lock_guard<std::mutex> send_lock(send_mutex_);

  tx_data_nav.linear_x = nav_data->linear_x;
  tx_data_nav.linear_y = nav_data->linear_y;
  tx_data_nav.linear_z = nav_data->linear_z;
  tx_data_nav.angular_x = nav_data->angular_x;
  tx_data_nav.angular_y = nav_data->angular_y;
  tx_data_nav.angular_z = nav_data->angular_z;
  tx_data_nav.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_nav), sizeof(tx_data_nav) - sizeof(tx_data_nav.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_nav), sizeof(tx_data_nav));
  } catch (const std::exception & e) {
    tools::logger()->warn("[GimbalWithNav] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send_imu_forward(const DM_IMU & imu) const {
  std::lock_guard<std::mutex> send_lock(send_mutex_);
  imu.forward_data(serial_);
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

#ifndef NDEBUG
  auto plotter = tools::Plotter{};
#endif

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    uint8_t head_byte[2];
    if (!read(head_byte, 1)) continue;

#ifndef NDEBUG
    tools::logger()->debug("[Gimbal] Read first byte: 0x{:02X}", head_byte[0]);
    // if (head_byte[0] == gimbal_struct_header[0]) {
    //   tools::logger()->debug("[Gimbal] Detected potential GimbalToVision header byte.");
    // } else if (head_byte[0] == nav_struct_header[0]) {
    //   tools::logger()->debug("[Gimbal] Detected potential NavData header byte.");
    // } else if (head_byte[0] == 0xA5) {
    //   tools::logger()->debug("[Gimbal] Detected referee system packet header.");
    // }
#endif

    if (head_byte[0] == gimbal_struct_header[0]) {
      // ---- GimbalToVision packet ('A', 'B') ----
      if (!read(head_byte + 1, 1)) continue;
      if (head_byte[1] != gimbal_struct_header[1]) continue;

      auto t = std::chrono::steady_clock::now();

      // Use a local buffer to avoid member-variable aliasing across iterations
      GimbalToVision rx_pkt{};
      rx_pkt.head[0] = head_byte[0];
      rx_pkt.head[1] = head_byte[1];

      if (!read(
            reinterpret_cast<uint8_t *>(&rx_pkt) + sizeof(rx_pkt.head),
            sizeof(rx_pkt) - sizeof(rx_pkt.head))) {
        error_count++;
        continue;
      }
      auto crc16_passed = tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_pkt), sizeof(rx_pkt));
      tools::logger()->debug(
        "[Gimbal] Received GimbalToVision packet: yaw={:.2f}, pitch={:.2f}, crc16=0x{:04X}, received_crc16=0x{:04X}, crc16_passed={}",
        rx_pkt.yaw, rx_pkt.pitch, tools::get_crc16(reinterpret_cast<uint8_t *>(&rx_pkt), sizeof(rx_pkt) - sizeof(rx_pkt.crc16)), rx_pkt.crc16, crc16_passed);

      if (!crc16_passed) {
        tools::logger()->debug("[Gimbal] CRC16 check failed for GimbalToVision packet.");
        continue;
      }

      error_count = 0;
      // Preserve the existing yaw/pitch convention unless the firmware's
      // quaternion frame has been explicitly verified in configuration.
      double pitch = rx_pkt.pitch;
      double yaw = rx_pkt.yaw;
      
      double cy = std::cos(yaw * 0.5);
      double sy = std::sin(yaw * 0.5);
      double cp = std::cos(pitch * 0.5);
      double sp = std::sin(pitch * 0.5);
      
      double w = cp * cy;
      double x = -sp * sy;
      double y = sp * cy;
      double z = cp * sy;
      
      // Validate quaternion
      Eigen::Quaterniond packet_q(rx_pkt.q[0], rx_pkt.q[1], rx_pkt.q[2], rx_pkt.q[3]);
      if (use_packet_quaternion_) {
        if (packet_q.coeffs().allFinite() && std::abs(packet_q.squaredNorm() - 1) < 1e-2)
          pose_history_.push(packet_q, t);
      } else if (std::abs(w * w + x * x + y * y + z * z - 1) < 1e-2) {
          pose_history_.push({w, x, y, z}, t);
      } else {
          tools::logger()->warn("[CBoardUART] Invalid quaternion received");
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);

        state_.yaw = rx_pkt.yaw;
        state_.yaw_vel = rx_pkt.yaw_vel;
        state_.pitch = rx_pkt.pitch;
        state_.pitch_vel = rx_pkt.pitch_vel;
        state_.bullet_speed = rx_pkt.bullet_speed;
        state_.bullet_count = rx_pkt.bullet_count;
        state_.timestamp = t;

        const auto previous_mode = mode_;
        switch (rx_pkt.mode) {
          case 0:
            mode_ = GimbalMode::IDLE;
            break;
          case 1:
            mode_ = GimbalMode::AUTO_AIM;
            break;
          case 2:
            mode_ = GimbalMode::SMALL_BUFF;
            break;
          case 3:
            mode_ = GimbalMode::BIG_BUFF;
            break;
          default:
            mode_ = GimbalMode::IDLE;
            tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_pkt.mode);
            break;
        }
        if (mode_ != previous_mode) {
          control_guard_.invalidate(t);
          publisher_->publish({});
        }
#ifndef NDEBUG
        nlohmann::json data;
        data["mode"] = str(mode_);
        data["yaw"] = state_.yaw * 57.3;
        data["yaw_vel"] = state_.yaw_vel * 57.3;
        data["pitch"] = state_.pitch * 57.3;
        data["pitch_vel"] = state_.pitch_vel * 57.3;
        data["bullet_speed"] = state_.bullet_speed;
        data["bullet_count"] = state_.bullet_count;
        plotter.plot(data);
#endif
      }
    } 
    else if (head_byte[0] == 0xA5) {
      // ---- Referee system packet (0xA5 header) ----
      uint8_t header[4];  // length(2), seq(1), crc8(1)
      if (!read(header, 4)) {
        error_count++;
        continue;
      }

      uint16_t data_len = *reinterpret_cast<uint16_t *>(header);
      if (data_len > 256) {
        continue;  // 长度超限
      }

      // 校验帧头 CRC8
      uint8_t crc_header[5] = {0xA5, header[0], header[1], header[2], header[3]};
      if (!tools::check_crc8(crc_header, 5)) continue;

      uint8_t cmd_buff[2];
      if (!read(cmd_buff, 2)) {
        error_count++;
        continue;
      }
      uint16_t cmd_id = *reinterpret_cast<uint16_t *>(cmd_buff);

      std::vector<uint8_t> data_buff(data_len);
      if (data_len > 0) {
        if (!read(data_buff.data(), data_len)) {
          error_count++;
          continue;
        }
      }

      uint8_t tail[2];  // CRC16
      if (!read(tail, 2)) {
        error_count++;
        continue;
      }

      // 验证全包 CRC16
      std::vector<uint8_t> full_packet(1 + 4 + 2 + data_len);
      full_packet[0] = 0xA5;
      std::copy(header, header + 4, full_packet.begin() + 1);
      std::copy(cmd_buff, cmd_buff + 2, full_packet.begin() + 5);
      if (data_len > 0) {
        std::copy(data_buff.data(), data_buff.data() + data_len, full_packet.begin() + 7);
      }
      uint16_t crc_calculated = tools::get_crc16(full_packet.data(), full_packet.size());
      uint16_t crc_received = *reinterpret_cast<uint16_t *>(tail);
      if (crc_calculated != crc_received) {
        tools::logger()->debug("[Gimbal] Full packet CRC16 check failed for referee packet.");
        continue;
      }
      error_count = 0;
      parse_referee_data(cmd_id, data_buff.data(), data_len);
    }
  }
  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  pose_history_.clear();
  control_guard_.invalidate();
  publisher_->publish({});
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      {
        std::lock_guard<std::mutex> send_lock(send_mutex_);
        serial_.close();
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      {
        std::lock_guard<std::mutex> send_lock(send_mutex_);
        serial_.open();
      }
      pose_history_.clear();
      control_guard_.invalidate();
      publisher_->publish({});
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}


void Gimbal::parse_referee_data(uint16_t cmd_id, const uint8_t* data, uint16_t len)
{
  switch (cmd_id) {
    case 0x0201: // 机器人性能体系状态
    case 0x0202: // 实时功率热量数据
      if (referee_callback_) {
        referee_callback_(cmd_id, data, len);
      }
      // break;
    // case 0x0203: // 机器人位置数据
    // case 0x0204: // 机器人增益数据
    // case 0x0206: // 伤害状态数据
    // case 0x0101: // 场地事件数据
    // case 0x0102: // 补给站动作标识数据
    default: // 默认转发给导航决策器
      if (nav_referee_callback_) {
        std::vector<uint8_t> data_vec(data, data + len);
        nav_referee_callback_(cmd_id, data_vec);
      }
      break;
  }
}

bool Gimbal::send_gimbal_data() const {
  try {
    return const_cast<serial::Serial &>(serial_).write(
      reinterpret_cast<const uint8_t *>(&tx_data_gimbal), sizeof(tx_data_gimbal)) == sizeof(tx_data_gimbal);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
    return false;
  }
}

} // namespace io
