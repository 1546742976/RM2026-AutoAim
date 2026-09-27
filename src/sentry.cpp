#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "io/ros2/publish2nav.hpp"
#include "io/ros2/ros2.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/sentry_decision/sentry_decision.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  // tools::Recorder recorder;

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  io::ROS2 ros2;
  
  sentry_decision::SentryDecider decider_sentry;

  // io::CBoard cboard(config_path);
  io::Gimbal gimbal(config_path, nullptr, [&](uint16_t cmd_id, const uint8_t* data, uint16_t len){
    decider_sentry.referee_callback(cmd_id, data, len);
  }, nullptr);
  io::Camera camera(config_path);
  io::DM_IMU dm_imu{};
  // io::Camera back_camera("configs/camera.yaml");
  // io::USBCamera usbcam1("video0", config_path);
  // io::USBCamera usbcam2("video2", config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  // omniperception::Decider decider(config_path);

  cv::Mat img;

  std::chrono::steady_clock::time_point timestamp;
  io::Command last_command;

  decider_sentry.start_nav_thread(&dm_imu);
  auto previous_mode = gimbal.mode();

  while (!exiter.exit()) {
    io::FramePacket frame;
    if (!camera.read_for(frame, std::chrono::milliseconds(50))) { gimbal.send(io::Command{}); continue; }
    img = frame.image;
    timestamp = frame.exposure_time;
    //Eigen::Quaterniond q = dm_imu.imu_at(timestamp);
    auto q = gimbal.q(timestamp);
    frame.set_pose(q);
    const auto mode = gimbal.mode();
    if (mode != previous_mode) { tracker.reset(); previous_mode = mode; }
    if (!frame.pose_valid || mode != io::GimbalMode::AUTO_AIM) {
      tracker.reset(); gimbal.send(io::Command{}); continue;
    }
    // recorder.record(img, q, timestamp);

    /// 自瞄核心逻辑
    solver.set_R_gimbal2world(q);

    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto armors = yolo.detect(img);

    // decider.get_invincible_armor(ros2.subscribe_enemy_status()); // 相关接口尚屏蔽

    // decider.armor_filter(armors);

    // decider.get_auto_aim_target(armors, ros2.subscribe_autoaim_target());

    // decider.set_priority(armors);

    auto targets = tracker.track(armors, timestamp);

    io::Command command{false, false, 0, 0};

    static auto last_seen_time = timestamp;
    if (tracker.state() != "lost") {
      last_seen_time = timestamp;
    }

    /// 全向感知逻辑
    if (tracker.state() == "lost") {
      if (std::chrono::duration<double>(timestamp - last_seen_time).count() > 1.0) {
        // command = decider.decide(yolo, gimbal_pos, camera);
        static auto t0 = timestamp;
        double t = std::chrono::duration<double>(timestamp - t0).count();
        
        command.control = true;
        // 使云台yaw以1.5弧度/秒不断旋转并映射至[-PI, PI)
        command.yaw = 0.5 * std::sin(t * 0.6) * M_PI;  // 1.5 rad/s的正弦波，幅值为PI
        // if (command.yaw > M_PI) command.yaw -= 2 * M_PI;
        
        // pitch以2 rad/s的速度，在-0.2到+0.2弧度上下摆动
        command.pitch = 15 / 57.3 * std::sin(t * 5.0) + 7.5 / 57.3;  // 2 rad/s的正弦波，幅值为0.2，中心位置为-7.5度
        command.shoot = false;
      }
    } else {
      command = aimer.aim(targets, timestamp, gimbal.state().bullet_speed);
    }

    /// 发射逻辑
    if (tracker.state() != "lost") {
      const auto gs = gimbal.state();
      Eigen::Vector3d gimbal_pos_shoot{gs.yaw, gs.pitch, 0};
      command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos_shoot);
    } else {
      command.shoot = false;
    }
    
    io::NavData nav_data = decider_sentry.get_nav_data();
    uint8_t cur_game_progress = decider_sentry.get_game_progress();

    // game_progress == 4 代表比赛中，未开始比赛前不进行任何动作
#ifdef NDEBUG
    if (cur_game_progress != 4) {
      command.control = false;
      command.shoot = false;
      nav_data.linear_x = 0;
      nav_data.linear_y = 0;
      nav_data.angular_z = 0;
    }
#endif

    gimbal.send(command);
    gimbal.send_imu_forward(dm_imu);
    gimbal.send_cmd_vel(nav_data);
#ifndef NDEBUG
    nlohmann::json json;
    json["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count();
    json["command"] = {
      {"control", command.control},
      {"shoot", command.shoot},
      {"yaw", command.yaw},
      {"pitch", command.pitch}
    };
    json["nav_data"] = {
      {"linear_x", nav_data.linear_x},
      {"linear_y", nav_data.linear_y},
      {"angular_z", nav_data.angular_z}
    };
    plotter.plot(json);
#endif

    /// ROS2通信
    // Eigen::Vector4d target_info = decider.get_target_info(armors, targets);

    // ros2.publish(target_info);
  }
  decider_sentry.stop_nav_thread();
  return 0;
}
