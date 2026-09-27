#include "logger.hpp"

#include <fmt/chrono.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>

namespace tools
{
std::shared_ptr<spdlog::logger> make_logger()
{
  auto file_name = fmt::format("logs/{:%Y-%m-%d_%H-%M-%S}.log", std::chrono::system_clock::now());
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name, true);
  file_sink->set_level(spdlog::level::debug);

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::debug);

  auto result = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink, console_sink});
#ifdef NDEBUG
  result->set_level(spdlog::level::info);
#else
  result->set_level(spdlog::level::debug);
#endif
  result->flush_on(spdlog::level::warn);
  return result;
}

std::shared_ptr<spdlog::logger> logger()
{
  static auto instance = make_logger();
  return instance;
}

}  // namespace tools
