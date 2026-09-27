#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/cboard_uart.hpp"
#include "io/command.hpp"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

// Conversion factor: degrees per radian (used throughout this file)
constexpr double DEG_PER_RAD = 180.0 / M_PI;

const std::string keys =
  "{help h usage ? |                     | usage}"
  "{@config-path   | configs/test2.yaml  | config path}";

struct TrajectoryState
{
  double pos;
  double vel;
  double acc;
};

// Sinusoidal yaw reference trajectory (degrees)
double yaw_cal(double t)
{
  constexpr double A = 120.0;  // amplitude in degrees
  constexpr double T = 4.0;    // period in seconds
  constexpr double w = 2 * M_PI / T;
  return A * std::sin(w * t);
}

// Sinusoidal pitch reference trajectory (degrees)
double pitch_cal(double t)
{
  constexpr double A = 30.0;
  constexpr double T = 5.0;
  constexpr double w = 2 * M_PI / T;
  return A * std::sin(w * t);
}

// // PD-based trajectory follower (mimics MPC step structure from gimbal_control_test)
// TrajectoryState mpc_like_step(
//   double ref_pos, double ref_vel, double dt, double kp, double kd, double max_acc,
//   double & state_pos, double & state_vel)
// {
//   double pos_err = ref_pos - state_pos;
//   double vel_err = ref_vel - state_vel;

//   double acc = kp * pos_err + kd * vel_err;
//   acc = std::clamp(acc, -max_acc, max_acc);

//   state_vel += acc * dt;
//   state_pos += state_vel * dt;

//   return {state_pos, state_vel, acc};
// }

// Periodic shoot command: fire for the first 2 s of every 5 s window
bool shoot_cal(double t)
{
  constexpr double shoot_interval = 5.0;
  return std::fmod(t, shoot_interval) < 2.0;
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);

  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  // Use CBoardUART as the verified communication interface
  io::CBoardUART cboard(config_path);

  std::cout << "[cboard_uart_control_test] Zeroing gimbal..." << std::endl;
  io::Command init_command{true, false, 0.0, 0.0};
  cboard.send(init_command);
  std::this_thread::sleep_for(500ms);

  std::cout << "[cboard_uart_control_test] Starting trajectory tracking..." << std::endl;

  // constexpr double dt = 0.01;           // 10 ms control period
  // constexpr double yaw_kp = 60.0;
  // constexpr double yaw_kd = 12.0;
  // constexpr double pitch_kp = 45.0;
  // constexpr double pitch_kd = 10.0;
  // constexpr double max_yaw_acc = 50.0;    // deg/s²
  // constexpr double max_pitch_acc = 100.0; // deg/s²

  // bool initialized = false;
  // double yaw_state_pos = 0.0;
  // double yaw_state_vel = 0.0;
  // double pitch_state_pos = 0.0;
  // double pitch_state_vel = 0.0;

  auto start_time = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    double t = tools::delta_time(now, start_time);

    // Reference trajectory at current time
    double yaw_ref_pos = yaw_cal(t);
    double pitch_ref_pos = pitch_cal(t);

    // // Numerical derivative for reference velocity
    // double yaw_ref_vel = (yaw_cal(t + dt) - yaw_cal(t - dt)) / (2.0 * dt);
    // double pitch_ref_vel = (pitch_cal(t + dt) - pitch_cal(t - dt)) / (2.0 * dt);

    // // Seed controller state from reference on first iteration
    // if (!initialized) {
    //   yaw_state_pos = yaw_ref_pos;
    //   yaw_state_vel = yaw_ref_vel;
    //   pitch_state_pos = pitch_ref_pos;
    //   pitch_state_vel = pitch_ref_vel;
    //   initialized = true;
    // }

    // // Compute trajectory step
    // auto yaw_traj = mpc_like_step(
    //   yaw_ref_pos, yaw_ref_vel, dt, yaw_kp, yaw_kd, max_yaw_acc,
    //   yaw_state_pos, yaw_state_vel);
    // auto pitch_traj = mpc_like_step(
    //   pitch_ref_pos, pitch_ref_vel, dt, pitch_kp, pitch_kd, max_pitch_acc,
    //   pitch_state_pos, pitch_state_vel);

    // Build command (CBoardUART uses yaw/pitch in radians)
    io::Command command;
    command.control = true;
    command.shoot = shoot_cal(t);
    // command.yaw = yaw_traj.pos / DEG_PER_RAD;
    // command.pitch = pitch_traj.pos / DEG_PER_RAD;
    command.yaw = yaw_ref_pos / DEG_PER_RAD;
    command.pitch = pitch_ref_pos / DEG_PER_RAD;
    command.horizon_distance = 0.0; // Not used in this test
    
    // Plot for debugging
    nlohmann::json json;
    json["t"] = t;
    json["send_yaw_deg"] = command.yaw * DEG_PER_RAD;
    json["send_pitch_deg"] = command.pitch * DEG_PER_RAD;
    json["yaw_ref_deg"] = yaw_ref_pos;
    json["pitch_ref_deg"] = pitch_ref_pos;
    json["send_shoot"] = command.shoot ? 1 : 0;
    json["bullet_speed"] = cboard.bullet_speed.load();
    plotter.plot(json);

    // Send command via CBoardUART
    cboard.send(command);

    std::this_thread::sleep_for(10ms);
  }

  return 0;
}
