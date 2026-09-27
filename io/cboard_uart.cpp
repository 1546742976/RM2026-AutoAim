#include "cboard_uart.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"
#include "tools/math_tools.hpp"
#include "tools/crc.hpp"
#include "io/gimbal/gimbal.hpp"
#include <iostream>

#ifndef NDEBUG
#include "tools/plotter.hpp"

#endif

namespace io
{

#pragma pack(push, 1)
#ifdef NEW_UART_PROTOCOL
struct SendPacket
{
  uint8_t header = 0xA5;
  uint8_t length = sizeof(SendPacket);
  uint8_t control;
  uint8_t shoot;
  int16_t yaw;
  int16_t pitch;
  int16_t dist;
  uint16_t checksum;
  uint16_t tail = 0x7891;
};
#else
struct SendPacket
{
  uint8_t header = 0xA5;
  uint8_t control;
  uint8_t shoot;
  int16_t yaw;
  int16_t pitch;
  int16_t dist;
  uint16_t checksum;
};
#endif

#pragma pack(pop)

CBoardUART::CBoardUART(const std::string & config_path)
: bullet_speed(0),
  mode(Mode::idle),
  shoot_mode(ShootMode::left_shoot),
  ft_angle(0),
  stop_thread_(false)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  send_latency_ = std::make_unique<tools::LatencyStats>(yaml["runtime_metrics"].as<bool>(false));
  control_guard_.configure(yaml["shoot_max_age_ms"].as<double>(100),
                           yaml["control_max_age_ms"].as<double>(200));

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(115200);
    serial::Timeout to = serial::Timeout::simpleTimeout(1000);
    serial_.setTimeout(to);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[CBoardUART] Failed to open serial: {}", e.what());
    exit(1);
  }

  publisher_ = std::make_unique<ControlPublisher>(
    [this](ControlIntent intent) { return control_guard_.apply(intent, mode.load() != Mode::idle); },
    [this](const ControlIntent & intent) { write_control(intent); });
  thread_ = std::thread(&CBoardUART::read_thread, this);
  tools::logger()->info("[CBoardUART] Opened.");
}

CBoardUART::~CBoardUART()
{
  stop_thread_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
  publisher_->close();
  if (send_latency_->enabled()) {
    const auto stats = send_latency_->summary();
    tools::logger()->info("[CBoardUART] estimated exposure-to-send: n={} window={} p50={:.2f} p95={:.2f} max={:.2f} ms",
      stats.total_samples, stats.samples, stats.p50_ms, stats.p95_ms, stats.max_ms);
  }
  if (serial_.isOpen()) {
    serial_.close();
  }
}

Eigen::Quaterniond CBoardUART::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  auto sample = pose_history_.at(timestamp);
  control_guard_.observe(timestamp, sample.has_value());
  return sample ? sample->q : tools::PoseHistory::invalid();
}

void CBoardUART::send(Command command) const
{
  publisher_->publish(control_guard_.apply(ControlIntent{command}, mode.load() != Mode::idle));
}

void CBoardUART::write_control(const ControlIntent & intent) const
{
  auto command = intent.command;
  if (std::abs(command.yaw) > 3.2767 || std::abs(command.pitch) > 3.2767) command = {};
  command.horizon_distance = std::clamp(command.horizon_distance, 0.0, 3.2767);
  SendPacket packet;
  packet.control = command.control ? 1 : 0;
  packet.shoot = command.shoot ? 1 : 0;
  // Convert to fixed point as per CBoard protocol logic
  packet.yaw = static_cast<int16_t>(command.yaw * 1e4);
  packet.pitch = static_cast<int16_t>(command.pitch * 1e4);
  packet.dist = static_cast<int16_t>(command.horizon_distance * 1e4);
  
  // Calculate CRC (exclude checksum field itself)
#ifdef NEW_UART_PROTOCOL
  packet.checksum = tools::get_crc16(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet) - 4);
#else
  packet.checksum = tools::get_crc16(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet) - 2);
#endif
  try {
     // Serial write is thread-safe generally, but cast is needed if member is not mutable
     if (const_cast<serial::Serial&>(serial_).write(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet)) == sizeof(packet))
       send_latency_->record_first_send(command.source_time);
  } catch (const std::exception & e) {
    tools::logger()->warn("[CBoardUART] Send failed: {}", e.what());
  }
}

void CBoardUART::read_thread()
{
  const size_t PACKET_SIZE = sizeof(GimbalToVision);
  uint8_t buffer[PACKET_SIZE];
#ifndef NDEBUG
  tools::Plotter plotter;
  nlohmann::json plot_json;
#endif
  auto start_time = std::chrono::steady_clock::now();
  while (!stop_thread_) {
    try {
      if (!serial_.isOpen()) {
         std::this_thread::sleep_for(std::chrono::milliseconds(100));
         continue;
      }

      // Read header
      uint8_t header[2];
      if (serial_.read(header, 1) != 1) continue;
      if (header[0] != gimbal_struct_header[0]) continue;
      
      if (serial_.read(header + 1, 1) != 1) continue;
      if (header[1] != gimbal_struct_header[1]) continue;

      buffer[0] = gimbal_struct_header[0];
      buffer[1] = gimbal_struct_header[1];
      
      // Read rest of the packet
      if (serial_.read(buffer + 2, PACKET_SIZE - 2) == PACKET_SIZE - 2) {
         GimbalToVision* pkt = reinterpret_cast<GimbalToVision*>(buffer);
         
         // Verify CRC
         uint16_t cal_crc = tools::get_crc16(buffer, PACKET_SIZE - 2);
         tools::logger()->debug("[CBoardUART] Received packet yaw {:.4f}, pitch {:.4f}, Calculated CRC: {:04X}, Recived CRC: {:04X}, CRC Passed: {}", 
          (float)(pkt->yaw) * 57.3, 
          (float)(pkt->pitch) * 57.3, 
          cal_crc, 
          (uint16_t)(pkt->crc16), 
          (uint16_t)(pkt->crc16) == cal_crc
        );
#ifndef NDEBUG
          // For debugging: plot received data
          plot_json["timestamp"] = tools::delta_time(std::chrono::steady_clock::now(), start_time);
          plot_json["yaw"] = static_cast<double>(pkt->yaw) * 57.3;
          plot_json["pitch"] = static_cast<double>(pkt->pitch) * 57.3;
          //plot_json["bullet_speed"] = pkt->bullet_speed;
          plotter.plot(plot_json);
#endif
         if (cal_crc == pkt->crc16) {
             auto timestamp = std::chrono::steady_clock::now();
             
             // Process Quaternion from yaw and pitch (roll = 0)
             double pitch = pkt->pitch;
             double yaw = pkt->yaw;
             
             double cy = std::cos(yaw * 0.5);
             double sy = std::sin(yaw * 0.5);
             double cp = std::cos(pitch * 0.5);
             double sp = std::sin(pitch * 0.5);
             
             double w = cp * cy;
             double x = -sp * sy;
             double y = sp * cy;
             double z = cp * sy;
             
             // Validate quaternion
             if (std::abs(w * w + x * x + y * y + z * z - 1) < 1e-2) {
                 pose_history_.push({w, x, y, z}, timestamp);
             } else {
                 tools::logger()->warn("[CBoardUART] Invalid quaternion received");
             }
             
             // Process State
             bullet_speed = pkt->bullet_speed;
             
             // Map packet mode to internal Mode
             const auto previous_mode = mode.load();
             switch (pkt->mode) {
                 case 0: mode = Mode::idle; break;
                 case 1: mode = Mode::auto_aim; break;
                 case 2: mode = Mode::small_buff; break;
                 case 3: mode = Mode::big_buff; break;
                 default: mode = Mode::idle; break;
             }
             if (mode.load() != previous_mode) control_guard_.invalidate(timestamp);
             
             // GimbalToVision does not have shoot_mode or ft_angle, so we cannot update them.

                 // Log occasionally
             static auto last_log = std::chrono::steady_clock::time_point::min();
             if (bullet_speed > 0 && tools::delta_time(timestamp, last_log) >= 1.0) {
                std::string mode_str = (mode < MODES.size()) ? MODES[mode] : "Unknown";
                
                tools::logger()->info(
                    "[CBoardUART] Speed: {:.2f}, Mode: {}",
                    bullet_speed.load(), mode_str);
                last_log = timestamp;
             }

         } else {
             tools::logger()->warn("[CBoardUART] CRC mismatch");
         }
      }
    } catch (const std::exception & e) {
      tools::logger()->error("[CBoardUART] Read error: {}", e.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }
}
}  // namespace io
