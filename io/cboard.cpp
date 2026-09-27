#include "cboard.hpp"

#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
CBoard::CBoard(const std::string & config_path)
: mode(Mode::idle),
  shoot_mode(ShootMode::left_shoot),
  bullet_speed(0),
  can_(read_yaml(config_path), std::bind(&CBoard::callback, this, std::placeholders::_1))
// 注意: callback的运行会早于Cboard构造函数的完成
{
  publisher_ = std::make_unique<ControlPublisher>(
    [this](ControlIntent intent) { return control_guard_.apply(intent, mode.load() != Mode::idle); },
    [this](const ControlIntent & intent) { write_control(intent); });
  tools::logger()->info("[Cboard] Opened.");
}

CBoard::~CBoard()
{
  publisher_->close();
  if (send_latency_->enabled()) {
    const auto stats = send_latency_->summary();
    tools::logger()->info("[CBoard] estimated exposure-to-send: n={} window={} p50={:.2f} p95={:.2f} max={:.2f} ms",
      stats.total_samples, stats.samples, stats.p50_ms, stats.p95_ms, stats.max_ms);
  }
}

Eigen::Quaterniond CBoard::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  auto sample = pose_history_.at(timestamp);
  control_guard_.observe(timestamp, sample.has_value());
  return sample ? sample->q : tools::PoseHistory::invalid();
}

void CBoard::send(Command command) const
{
  publisher_->publish(control_guard_.apply(ControlIntent{command}, mode.load() != Mode::idle));
}

void CBoard::write_control(const ControlIntent & intent) const
{
  auto command = intent.command;
  // Fixed-point protocol cannot represent out-of-range values. Reject instead
  // of wrapping a valid angle/distance into an unrelated command.
  if (std::abs(command.yaw) > 3.2767 || std::abs(command.pitch) > 3.2767 ||
      !std::isfinite(command.horizon_distance)) command = {};
  command.horizon_distance = std::clamp(command.horizon_distance, 0.0, 3.2767);
  can_frame frame;
  frame.can_id = send_canid_;
  frame.can_dlc = 8;
  frame.data[0] = (command.control) ? 1 : 0;
  frame.data[1] = (command.shoot) ? 1 : 0;
  frame.data[2] = (int16_t)(command.yaw * 1e4) >> 8;
  frame.data[3] = (int16_t)(command.yaw * 1e4);
  frame.data[4] = (int16_t)(command.pitch * 1e4) >> 8;
  frame.data[5] = (int16_t)(command.pitch * 1e4);
  frame.data[6] = (int16_t)(command.horizon_distance * 1e4) >> 8;
  frame.data[7] = (int16_t)(command.horizon_distance * 1e4);

  try {
    can_.write(&frame);
    send_latency_->record_first_send(command.source_time);
  } catch (const std::exception & e) {
    tools::logger()->warn("{}", e.what());
  }
}

void CBoard::callback(const can_frame & frame)
{
  auto timestamp = std::chrono::steady_clock::now();

  if (frame.can_id == quaternion_canid_) {
    if (frame.can_dlc < 8) return;
    auto x = (int16_t)(frame.data[0] << 8 | frame.data[1]) / 1e4;
    auto y = (int16_t)(frame.data[2] << 8 | frame.data[3]) / 1e4;
    auto z = (int16_t)(frame.data[4] << 8 | frame.data[5]) / 1e4;
    auto w = (int16_t)(frame.data[6] << 8 | frame.data[7]) / 1e4;

    if (std::abs(x * x + y * y + z * z + w * w - 1) > 1e-2) {
      tools::logger()->warn("Invalid q: {} {} {} {}", w, x, y, z);
      return;
    }

    pose_history_.push({w, x, y, z}, timestamp);
  }

  else if (frame.can_id == bullet_speed_canid_) {
    if (frame.can_dlc < 6) return;
    bullet_speed = (int16_t)(frame.data[0] << 8 | frame.data[1]) / 1e2;
    const auto new_mode = frame.data[2] <= Mode::outpost ? Mode(frame.data[2]) : Mode::idle;
    if (mode.exchange(new_mode) != new_mode) control_guard_.invalidate(timestamp);
    shoot_mode = frame.data[3] <= ShootMode::both_shoot ? ShootMode(frame.data[3]) : ShootMode::left_shoot;
    ft_angle = (int16_t)(frame.data[4] << 8 | frame.data[5]) / 1e4;

    // 限制日志输出频率为1Hz
    static auto last_log_time = std::chrono::steady_clock::time_point::min();
    auto now = std::chrono::steady_clock::now();

    if (bullet_speed > 0 && tools::delta_time(now, last_log_time) >= 1.0) {
      tools::logger()->info(
        "[CBoard] Bullet speed: {:.2f} m/s, Mode: {}, Shoot mode: {}, FT angle: {:.2f} rad",
        bullet_speed.load(), MODES[mode.load()], SHOOT_MODES[shoot_mode.load()], ft_angle.load());
      last_log_time = now;
    }
  }
}

// 实现方式有待改进
std::string CBoard::read_yaml(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  send_latency_ = std::make_unique<tools::LatencyStats>(yaml["runtime_metrics"].as<bool>(false));
  control_guard_.configure(yaml["shoot_max_age_ms"].as<double>(100),
                           yaml["control_max_age_ms"].as<double>(200));

  quaternion_canid_ = tools::read<int>(yaml, "quaternion_canid");
  bullet_speed_canid_ = tools::read<int>(yaml, "bullet_speed_canid");
  send_canid_ = tools::read<int>(yaml, "send_canid");

  if (!yaml["can_interface"]) {
    throw std::runtime_error("Missing 'can_interface' in YAML configuration.");
  }

  return yaml["can_interface"].as<std::string>();
}

}  // namespace io
