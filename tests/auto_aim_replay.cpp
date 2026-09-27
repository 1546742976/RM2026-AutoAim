#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
const std::string keys =
  "{help h ?             |                   | Show help }"
  "{config-path c        | configs/demo.yaml | Existing runtime YAML }"
  "{output o             | replay.csv        | Per-frame CSV output }"
  "{bullet-speed         | 27                | Measured bullet speed, m/s }"
  "{decision-delay-ms    | 0                 | Fixed simulated exposure-to-planning age }"
  "{warmup               | 20                | Exclude first N frames from timing percentiles }"
  "{limit                | 0                 | Maximum processed frames; 0 means all }"
  "{@input               |                   | Recorder .avi/.txt prefix, without extension }";

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool finite(const auto_aim::Plan & plan)
{
  return std::isfinite(plan.target_yaw) && std::isfinite(plan.target_pitch) &&
         std::isfinite(plan.yaw) && std::isfinite(plan.yaw_vel) &&
         std::isfinite(plan.yaw_acc) && std::isfinite(plan.pitch) &&
         std::isfinite(plan.pitch_vel) && std::isfinite(plan.pitch_acc);
}

void print_timing(const char * name, std::vector<double> values)
{
  if (values.empty()) {
    std::cout << name << ": no measured frames after warmup\n";
    return;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double p) {
    return values[static_cast<std::size_t>(std::ceil(p * values.size())) - 1];
  };
  std::cout << name << " ms: P50=" << percentile(0.50) << ", P95=" << percentile(0.95)
            << ", max=" << values.back() << ", n=" << values.size() << '\n';
}
}  // namespace

// Offline only: this executable constructs no Camera, Gimbal, CBoard or serial
// transport. `fire_requested` is diagnostic data, never sent to hardware.
int main(int argc, char * argv[])
{
  try {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
      cli.printMessage();
      return 0;
    }
    const auto input = cli.get<std::string>("@input");
    const auto config = cli.get<std::string>("config-path");
    const auto output = cli.get<std::string>("output");
    const auto bullet_speed = cli.get<double>("bullet-speed");
    const auto delay_ms = cli.get<double>("decision-delay-ms");
    const auto warmup = cli.get<int>("warmup");
    const auto limit = cli.get<int>("limit");
    if (!cli.check()) {
      cli.printErrors();
      return 2;
    }
    if (input.empty() || output.empty() || warmup < 0 || limit < 0 ||
        !std::isfinite(bullet_speed) || bullet_speed <= 0 ||
        !std::isfinite(delay_ms) || delay_ms < 0 || delay_ms > 60000)
      throw std::runtime_error("Invalid arguments; use --help.");

    cv::VideoCapture video(input + ".avi");
    std::ifstream metadata(input + ".txt");
    if (!video.isOpened() || !metadata)
      throw std::runtime_error("Cannot read recorder .avi/.txt pair: " + input);
    // Do not allow an output argument to destroy the recording itself.
    const auto output_path = std::filesystem::weakly_canonical(output);
    if (output_path == std::filesystem::weakly_canonical(input + ".txt") ||
        output_path == std::filesystem::weakly_canonical(input + ".avi") ||
        output_path == std::filesystem::weakly_canonical(config))
      throw std::runtime_error("CSV output must differ from input and configuration files.");
    std::ofstream csv(output);
    if (!csv) throw std::runtime_error("Cannot write CSV: " + output);

    const auto yaml = YAML::LoadFile(config);
    const double shoot_age_ms = yaml["shoot_max_age_ms"].as<double>(100.0);
    if (!std::isfinite(shoot_age_ms) || shoot_age_ms <= 0)
      throw std::runtime_error("Invalid shoot_max_age_ms.");
    auto_aim::YOLO detector(config, false);
    auto_aim::Solver solver(config);
    auto_aim::Tracker tracker(config, solver);
    auto_aim::Planner planner(config);
    tracker.reset();

    csv << "frame,recording_seconds,pose_input_valid,detections,solved_poses,reliable_poses,"
           "targets,tracker_state,control,fire_requested,command_finite,invalid_fire,"
           "target_age_ms,detect_ms,tracker_pnp_ms,planner_ms,processing_ms\n";
    csv << std::setprecision(12);
    std::vector<double> detection_times, tracking_times, planning_times, total_times;
    std::size_t frames = 0, no_detection = 0, no_target = 0, invalid_poses = 0;
    std::size_t control_count = 0, fire_count = 0, nonfinite_count = 0, invalid_fire_count = 0;
    double previous_t = -std::numeric_limits<double>::infinity();
    double first_t = 0;
    // Replay clock is independent of decoding / output speed. --decision-delay-ms
    // models a fixed source age so two runs compare the same prediction horizon.
    const auto epoch = Clock::time_point(std::chrono::seconds(1000));
    const auto delay = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double, std::milli>(delay_ms));
    bool exhausted = false;
    cv::Mat image;
    while (!limit || frames < static_cast<std::size_t>(limit)) {
      if (!video.read(image) || image.empty()) {
        exhausted = true;
        break;
      }
      std::string line;
      if (!std::getline(metadata, line))
        throw std::runtime_error("Timestamp rows end before video at frame " + std::to_string(frames));
      double t, w, x, y, z;
      std::istringstream row(line);
      if (!(row >> t >> w >> x >> y >> z))
        throw std::runtime_error("Malformed timestamp row " + std::to_string(frames));
      std::string extra;
      if (row >> extra)
        throw std::runtime_error("Extra timestamp columns at frame " + std::to_string(frames));
      if (!std::isfinite(t) || t <= previous_t)
        throw std::runtime_error("Non-monotonic timestamp at frame " + std::to_string(frames));
      if (frames == 0) first_t = t;
      if (t - first_t > 86400.0)
        throw std::runtime_error("Recording duration exceeds 24 hours.");
      previous_t = t;
      const auto timestamp = epoch + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(t - first_t));
      const auto decision_time = timestamp + delay;
      Eigen::Quaterniond pose(w, x, y, z);
      const auto pose_norm = pose.norm();
      const bool pose_valid = pose.coeffs().allFinite() && std::isfinite(pose_norm) &&
                              pose_norm > 1e-6;

      const auto begin = Clock::now();
      auto armors = detector.detect(image, static_cast<int>(frames));
      const auto detected = Clock::now();
      const auto detected_count = armors.size();
      std::list<auto_aim::Target> targets;
      if (pose_valid) {
        solver.set_R_gimbal2world(pose.normalized());
        targets = tracker.track(armors, timestamp);
      } else {
        ++invalid_poses;
        tracker.reset();
      }
      const auto tracked = Clock::now();
      std::optional<auto_aim::Target> target;
      if (!targets.empty()) target = targets.front();
      const auto plan = planner.plan(target, bullet_speed, decision_time);
      const auto finish = Clock::now();

      const bool command_finite = finite(plan);
      const double target_age_ms = plan.source_time == Clock::time_point{}
        ? -1.0 : elapsed_ms(plan.source_time, decision_time);
      const bool invalid_fire = plan.fire &&
        (!command_finite || !plan.control || !plan.pose_valid || !pose_valid ||
         plan.source_time == Clock::time_point{} || target_age_ms < 0 ||
         target_age_ms > shoot_age_ms || plan.valid_until == Clock::time_point{} ||
         decision_time > plan.valid_until);
      std::size_t solved = 0, reliable = 0;
      for (const auto & armor : armors) {
        solved += armor.pose_valid;
        reliable += armor.pose_reliable;
      }
      const auto detect_ms = elapsed_ms(begin, detected);
      const auto track_ms = elapsed_ms(detected, tracked);
      const auto plan_ms = elapsed_ms(tracked, finish);
      const auto total_ms = elapsed_ms(begin, finish);
      csv << frames << ',' << t << ',' << pose_valid << ',' << detected_count << ','
          << solved << ',' << reliable << ',' << targets.size() << ',' << tracker.state() << ','
          << plan.control << ',' << plan.fire << ',' << command_finite << ',' << invalid_fire << ','
          << target_age_ms << ',' << detect_ms << ',' << track_ms << ',' << plan_ms << ','
          << total_ms << '\n';
      if (!csv) throw std::runtime_error("CSV write failed: " + output);
      if (frames >= static_cast<std::size_t>(warmup)) {
        detection_times.push_back(detect_ms);
        tracking_times.push_back(track_ms);
        planning_times.push_back(plan_ms);
        total_times.push_back(total_ms);
      }
      no_detection += detected_count == 0;
      no_target += targets.empty();
      control_count += plan.control;
      fire_count += plan.fire;
      nonfinite_count += !command_finite;
      invalid_fire_count += invalid_fire;
      ++frames;
    }
    if (!frames) throw std::runtime_error("Recording contains no readable frames.");
    if (exhausted) {
      std::string remainder;
      while (std::getline(metadata, remainder)) {
        if (remainder.find_first_not_of(" \t\r") != std::string::npos)
          throw std::runtime_error("Timestamp rows outnumber video frames.");
      }
    }
    csv.close();
    std::cout << std::fixed << std::setprecision(3)
              << "Offline replay only; no hardware output. Config: " << config << '\n'
              << "Frames=" << frames << ", no_detection=" << no_detection
              << ", no_target=" << no_target << ", invalid_input_pose=" << invalid_poses
              << ", control_requested=" << control_count << ", fire_requested=" << fire_count
              << ", nonfinite_commands=" << nonfinite_count
              << ", invalid_fire=" << invalid_fire_count << '\n'
              << "Simulated decision age=" << delay_ms << " ms; timing excludes video decode,"
                 " CSV output, camera, transport and firing.\n";
    print_timing("Detector", detection_times);
    print_timing("Tracker including PnP", tracking_times);
    print_timing("Planner", planning_times);
    print_timing("Processing", total_times);
    std::cout << "CSV: " << output << '\n';
    return nonfinite_count || invalid_fire_count ? 1 : 0;
  } catch (const std::exception & error) {
    std::cerr << "Replay failed: " << error.what() << '\n';
    return 2;
  }
}
