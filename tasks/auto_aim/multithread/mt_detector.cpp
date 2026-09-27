#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <stdexcept>

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug, false)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto yolo_name = yaml["yolo_name"].as<std::string>();
  const auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  input_side_ = yolo_name == "yolov8" ? 416 : 640;
  use_roi_ = yaml["use_roi"].as<bool>(false);
  if (use_roi_) {
    roi_ = cv::Rect(
      yaml["roi"]["x"].as<int>(), yaml["roi"]["y"].as<int>(),
      yaml["roi"]["width"].as<int>(), yaml["roi"]["height"].as<int>());
  }

  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();
  const auto side = static_cast<ov::Dimension::value_type>(input_side_);
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, side, side, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

  worker_ = std::thread(&MultiThreadDetector::run, this);
  tools::logger()->info("[MultiThreadDetector] initialized with two owned inference slots");
}

MultiThreadDetector::~MultiThreadDetector() { close(); }

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  push(std::move(img), t, 0);
}

void MultiThreadDetector::push(
  cv::Mat img, std::chrono::steady_clock::time_point t, std::uint64_t frame_id)
{
  if (img.empty()) return;
  std::uint64_t generation;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return;
    generation = generation_;
    if (frame_id == 0) frame_id = next_frame_id_++;
  }

  // The producer may reuse its camera buffer as soon as push returns.
  InputFrame frame{img.clone(), t, frame_id, generation};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || generation != generation_) return;
    if (have_timestamp_ && t <= newest_timestamp_) return;
    newest_timestamp_ = t;
    have_timestamp_ = true;
    pending_ = std::move(frame);
  }
  work_available_.notify_one();
}

void MultiThreadDetector::run()
{
  struct Slot
  {
    InputFrame frame;
    cv::Mat input;
    double scale = 1;
    // Destroy the request before the external image memory it references.
    ov::InferRequest request;
  };
  std::array<Slot, 2> slots;
  std::deque<std::size_t> active;
  std::optional<std::chrono::steady_clock::time_point> published_timestamp;
  std::uint64_t published_generation = 0;
  try {
    for (auto & slot : slots) slot.request = compiled_model_.create_infer_request();
    for (;;) {
      std::optional<InputFrame> frame;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (active.empty()) {
          work_available_.wait(lock, [this] { return stopping_ || pending_.has_value(); });
        }
        if (stopping_) break;
        if (pending_ && active.size() < slots.size()) {
          frame = std::move(pending_);
          pending_.reset();
        }
      }

      if (frame) {
        const std::size_t index = active.empty() ? 0 : 1 - active.front();
        auto & slot = slots[index];
        slot.frame = std::move(*frame);
        cv::Mat source = slot.frame.image;
        if (use_roi_) {
          auto roi = roi_;
          if (roi.width == -1) roi.width = source.cols - roi.x;
          if (roi.height == -1) roi.height = source.rows - roi.y;
          if (roi.width <= 0 || roi.height <= 0 ||
              (roi & cv::Rect(0, 0, source.cols, source.rows)) != roi)
            throw std::runtime_error("Configured detector ROI is outside the camera image");
          source = source(roi);
        }
        slot.scale = std::min(
          static_cast<double>(input_side_) / source.rows,
          static_cast<double>(input_side_) / source.cols);
        const int h = std::max(1, static_cast<int>(source.rows * slot.scale));
        const int w = std::max(1, static_cast<int>(source.cols * slot.scale));
        slot.input = cv::Mat(input_side_, input_side_, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::resize(source, slot.input(cv::Rect(0, 0, w, h)), {w, h});
        const auto side = static_cast<std::size_t>(input_side_);
        slot.request.set_input_tensor(
          ov::Tensor(ov::element::u8, {1, side, side, 3}, slot.input.data));
        // Track the slot before start_async so cleanup also covers a failed start.
        active.push_back(index);
        slot.request.start_async();
      }

      if (active.empty()) continue;
      // Publish the newest completed request, not the oldest submitted request.
      // Devices may complete concurrent inference requests out of order.
      auto completed = active.end();
      for (auto it = active.begin(); it != active.end(); ++it) {
        if (slots[*it].request.wait_for(std::chrono::milliseconds(0)) &&
            (completed == active.end() ||
             slots[*it].frame.timestamp > slots[*completed].frame.timestamp)) completed = it;
      }
      if (completed == active.end()) {
        slots[active.front()].request.wait_for(std::chrono::milliseconds(1));
        continue;
      }
      auto & slot = slots[*completed];
      slot.request.wait();

      bool publish;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        publish = !stopping_ && slot.frame.generation == generation_ &&
          (published_generation != generation_ || !published_timestamp ||
           slot.frame.timestamp > *published_timestamp);
      }
      if (publish) {
        auto tensor = slot.request.get_output_tensor();
        const auto shape = tensor.get_shape();
        if (shape.size() != 3) throw std::runtime_error("Expected rank-three YOLO output");
        cv::Mat output(
          static_cast<int>(shape[1]), static_cast<int>(shape[2]), CV_32F, tensor.data());
        const auto debug_frame = static_cast<int>(
          slot.frame.frame_id % static_cast<std::uint64_t>(std::numeric_limits<int>::max()));
        auto armors = yolo_.postprocess(slot.scale, output, slot.frame.image, debug_frame);
        DetectionResult result{
          std::move(slot.frame.image), std::move(armors), slot.frame.timestamp,
          slot.frame.frame_id, slot.frame.generation};
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!stopping_ && result.generation == generation_) {
            published_timestamp = result.timestamp;
            published_generation = result.generation;
            ready_ = std::move(result);
          }
        }
        result_available_.notify_one();
      }
      active.erase(completed);
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_ = std::current_exception();
    stopping_ = true;
    pending_.reset();
    ready_.reset();
  }

  // reset/close never destroy an input while the device can still read it.
  for (const auto index : active) {
    try {
      slots[index].request.wait();
    } catch (...) {
      // Keep the original exception, but still drain every other active slot.
    }
  }
  result_available_.notify_all();
}

bool MultiThreadDetector::take_result(DetectionResult & result)
{
  if (failure_) std::rethrow_exception(failure_);
  if (!ready_) return false;
  result = std::move(*ready_);
  ready_.reset();
  return true;
}

bool MultiThreadDetector::try_pop(DetectionResult & result)
{
  std::lock_guard<std::mutex> lock(mutex_);
  return take_result(result);
}

bool MultiThreadDetector::pop_for(DetectionResult & result, std::chrono::milliseconds timeout)
{
  std::unique_lock<std::mutex> lock(mutex_);
  result_available_.wait_for(lock, timeout, [this] { return stopping_ || ready_.has_value(); });
  return take_result(result);
}

bool MultiThreadDetector::wait_result(DetectionResult & result)
{
  std::unique_lock<std::mutex> lock(mutex_);
  result_available_.wait(lock, [this] { return stopping_ || ready_.has_value(); });
  return take_result(result);
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  DetectionResult result;
  wait_result(result);
  return {std::move(result.armors), result.timestamp};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  DetectionResult result;
  wait_result(result);
  return {std::move(result.image), std::move(result.armors), result.timestamp};
}

void MultiThreadDetector::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  ++generation_;
  pending_.reset();
  ready_.reset();
  have_timestamp_ = false;
}

void MultiThreadDetector::close()
{
  // Serializes explicit close and destruction without holding the worker mutex at join.
  std::lock_guard<std::mutex> close_lock(close_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    pending_.reset();
    ready_.reset();
  }
  work_available_.notify_all();
  result_available_.notify_all();
  if (worker_.joinable()) worker_.join();
}

}  // namespace multithread

}  // namespace auto_aim
