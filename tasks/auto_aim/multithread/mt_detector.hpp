#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <optional>
#include <thread>
#include <tuple>

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  struct DetectionResult
  {
    cv::Mat image;
    std::list<Armor> armors;
    std::chrono::steady_clock::time_point timestamp{};
    std::uint64_t frame_id = 0;
    std::uint64_t generation = 0;
  };

  MultiThreadDetector(const std::string & config_path, bool debug = false);
  ~MultiThreadDetector();

  MultiThreadDetector(const MultiThreadDetector &) = delete;
  MultiThreadDetector & operator=(const MultiThreadDetector &) = delete;

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);
  void push(cv::Mat img, std::chrono::steady_clock::time_point t, std::uint64_t frame_id);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

  bool try_pop(DetectionResult & result);
  bool pop_for(DetectionResult & result, std::chrono::milliseconds timeout);

  // Discard old mode results without freeing buffers still used by inference.
  void reset();
  void close();

private:
  struct InputFrame
  {
    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp;
    std::uint64_t frame_id;
    std::uint64_t generation;
  };

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  YOLO yolo_;
  int input_side_ = 640;
  bool use_roi_ = false;
  cv::Rect roi_;

  std::mutex mutex_;
  std::mutex close_mutex_;
  std::condition_variable work_available_;
  std::condition_variable result_available_;
  std::optional<InputFrame> pending_;
  std::optional<DetectionResult> ready_;
  std::exception_ptr failure_;
  std::uint64_t generation_ = 0;
  std::uint64_t next_frame_id_ = 1;
  std::chrono::steady_clock::time_point newest_timestamp_{};
  bool have_timestamp_ = false;
  bool stopping_ = false;
  std::thread worker_;

  void run();
  bool take_result(DetectionResult & result);
  bool wait_result(DetectionResult & result);
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
