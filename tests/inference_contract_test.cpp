#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <yaml-cpp/yaml.h>
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

using namespace std::chrono_literals;
namespace {
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
}

// No device IO. The supplied output prefix must reside in the build directory.
int main(int argc, char ** argv)
{
  try {
    require(argc == 3, "Usage: inference_contract_test runtime.yaml output-prefix");
    const std::string prefix(argv[2]);
    auto yaml = YAML::LoadFile(argv[1]);
    yaml["device"] = "CPU";  // Explicit test fixture, never rewrites a deployment config.
    const auto config = prefix + ".yaml";
    {
      std::ofstream output(config);
      require(static_cast<bool>(output), "Cannot create CPU test config");
      output << yaml;
    }
    cv::Mat black(1080, 1440, CV_8UC3, cv::Scalar::all(0));
    cv::Mat noise(black.size(), black.type());
    cv::RNG rng(12450);
    rng.fill(noise, cv::RNG::UNIFORM, 0, 100);
    {
      auto_aim::YOLO detector(config, false);
      require(detector.detect(black).empty(), "Black frame yielded armor");
      for (const auto & armor : detector.detect(noise))
        require(armor.points.size() == 4, "Noise result has invalid corner count");
    }
    {
      auto_aim::multithread::MultiThreadDetector detector(config, false);
      const auto t = std::chrono::steady_clock::now();
      cv::Mat caller_buffer = black.clone();
      detector.push(caller_buffer, t, 1);
      caller_buffer.setTo(cv::Scalar::all(255));
      auto_aim::multithread::MultiThreadDetector::DetectionResult result;
      require(detector.pop_for(result, 10s), "Async inference timed out");
      require(result.frame_id == 1 && result.timestamp == t && cv::norm(result.image) == 0,
        "Inference reused producer image memory or lost source identity");
      for (int i = 2; i <= 18; ++i) detector.push(noise, t + i * 1ms, i);
      std::uint64_t latest = 1;
      const auto deadline = std::chrono::steady_clock::now() + 10s;
      while (latest != 18 && std::chrono::steady_clock::now() < deadline) {
        if (!detector.pop_for(result, 100ms)) continue;
        require(result.frame_id > latest, "Async result ordering regressed");
        latest = result.frame_id;
      }
      require(latest == 18, "Latest pending frame was dropped instead of superseded old frames");
      detector.push(noise, t + 20ms, 20);
      detector.reset();
      detector.push(black, t + 21ms, 21);
      require(detector.pop_for(result, 10s) && result.frame_id == 21 && result.generation == 1,
        "Mode reset published previous-generation result");
      detector.close();
      require(!detector.pop_for(result, 1ms), "Closed detector published a result");
    }
    // A tiny reproducible .avi/.txt pair exercises the real offline replay path.
    cv::VideoWriter video(prefix + ".avi", cv::VideoWriter::fourcc('M','J','P','G'),
      100, black.size());
    std::ofstream times(prefix + ".txt");
    require(video.isOpened() && times, "Cannot create replay smoke fixture");
    for (int i = 0; i < 8; ++i) {
      video.write(i % 2 ? noise : black);
      times << i * .01 << " 1 0 0 0\n";
    }
    std::cout << "CPU inference ownership/reset and synthetic replay fixture passed\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
