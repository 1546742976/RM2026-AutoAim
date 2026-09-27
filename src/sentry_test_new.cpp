#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/runtime.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/test2.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  io::ROS2 ros2;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  std::mutex planner_mutex;
  omniperception::Decider decider(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  
  std::cout << "Waiting for devices..." << std::endl;
  std::this_thread::sleep_for(1s);

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();

    std::optional<auto_aim::Target> target;
    while (!quit) {
      std::optional<auto_aim::Target> update;
      if (target_queue.pop_for(update, 10ms)) target = std::move(update);
      if (gimbal.mode() != io::GimbalMode::AUTO_AIM) target.reset();
      
      const auto feedback = gimbal.state();
      double bullet_speed = feedback.bullet_speed;
      
      auto_aim::Plan plan;
      { std::lock_guard<std::mutex> lock(planner_mutex); plan = planner.plan(target, bullet_speed); }

      // 发送控制指令
      gimbal.send(auto_aim::control_intent(plan, feedback));

      // 在向云台发送控制指令后尝试获取导航目标数据并发送（如果为空就不发送）
      auto nav_data = ros2.get_last_cmd_vel_data();
      if (nav_data.has_value()) {
        gimbal.send_cmd_vel(nav_data);
      }

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

      data["plan_yaw"] = plan.yaw * 57.3;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch * 57.3;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 10 : 0;
      data["control"] = plan.control ? 10 : 0;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z
        data["target_vz"] = target->ekf_x()[5];  //vz
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  auto previous_mode = gimbal.mode();
  while (!exiter.exit()) {
    io::FramePacket frame;
    if (!camera.read_for(frame, 50ms)) { target_queue.push(std::nullopt); continue; }
    img = frame.image;
    t = frame.exposure_time;
    
    // 获取云台位姿作为先验
    Eigen::Quaterniond q = gimbal.q(t);
    const auto mode = gimbal.mode();
    if (mode != previous_mode) { tracker.reset(); previous_mode = mode; }
    if (mode != io::GimbalMode::AUTO_AIM || !q.coeffs().allFinite()) {
      tracker.reset(); target_queue.push(std::nullopt); continue;
    }

    auto eulers = tools::eulers(q, 2, 1, 0);
    nlohmann::json data;
    data["imu_roll"] = eulers[0] * 57.3;
    data["imu_pitch"] = eulers[1] * 57.3;
    data["imu_yaw"] = eulers[2] * 57.3;
    plotter.plot(data);

    solver.set_R_gimbal2world(q);
    auto armors = yolo.detect(img);

    // 哨兵特有的决策过滤与装甲板优先级设置逻辑
    decider.get_invincible_armor(ros2.subscribe_enemy_status());
    decider.armor_filter(armors);
    decider.set_priority(armors);

    auto targets = tracker.track(armors, t);
    
    // 发送目标信息给导航
    Eigen::Vector4d target_info = decider.get_target_info(armors, targets);
    ros2.publish(target_info);

    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    if (!targets.empty()) {
      auto target = targets.front();

      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {255, 255, 0});
      }

      Eigen::Vector4d aim_xyza;
      { std::lock_guard<std::mutex> lock(planner_mutex); aim_xyza = planner.debug_xyza; }
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  quit = true;
  target_queue.close();
  if (plan_thread.joinable()) plan_thread.join();
  
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
