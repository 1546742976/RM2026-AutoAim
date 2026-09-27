#ifndef IO__CONTROL_DIAGNOSTICS_HPP
#define IO__CONTROL_DIAGNOSTICS_HPP

#include <array>
#include <string>
#include "io/control_guard.hpp"
#include "io/control_publisher.hpp"
#include "tools/logger.hpp"

namespace io
{
// Owned and used only by the transport's single writer. No logging under guard locks.
class ControlDiagnostics
{
public:
  void observe(const ControlGuard::Evaluation & result)
  {
    latest_ = result;
    pending_ = true;
    ++evaluations_;
    for (std::size_t i = 0; i < counts_.size(); ++i)
      if (result.reasons & (1u << i)) ++counts_[i];
  }

  // Write the control packet before any potentially blocking log I/O.
  void report(const char * transport)
  {
    if (!pending_) return;
    pending_ = false;
    const auto & result = latest_;
    const auto now = ControlGuard::Clock::now();
    if (seen_ && previous_ == result.reasons && now - last_log_ < std::chrono::seconds(5)) return;
    // Active, uninhibited steady state needs no periodic log.
    if (seen_ && previous_ == 0 && result.reasons == 0) return;
    std::string reasons;
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (!(result.reasons & (1u << i))) continue;
      if (!reasons.empty()) reasons += ',';
      reasons += std::string(names_[i]) + ':' + std::to_string(counts_[i]);
    }
    const auto source_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      result.intent.command.source_time.time_since_epoch()).count();
    tools::logger()->info(
      "[{}] control guard reasons(counts)={} frame={} source_ns={} evaluations={}",
      transport, reasons.empty() ? "none" : reasons, result.intent.command.frame_id,
      source_ns, evaluations_);
    seen_ = true;
    previous_ = result.reasons;
    last_log_ = now;
  }

private:
  inline static constexpr std::array<const char *, 14> names_{{
    "disabled", "missing_source", "previous_mode", "future_time", "control_expired",
    "pose_invalid", "uncalibrated", "shoot_expired", "pose_feedback_invalid",
    "status_feedback_invalid", "pose_feedback_stale", "status_feedback_stale",
    "nonfinite", "deadline_expired"}};
  std::array<uint64_t, 14> counts_{};
  uint64_t evaluations_ = 0;
  uint32_t previous_ = 0;
  bool seen_ = false;
  bool pending_ = false;
  ControlGuard::Evaluation latest_{};
  ControlGuard::Clock::time_point last_log_{};
};

inline void log_control_shutdown(const char * transport, const ControlPublisher::Status & state) noexcept
{
  try {
    const auto log_error = [transport](const char * stage, std::exception_ptr error) {
      if (!error) return;
      try { std::rethrow_exception(error); }
      catch (const std::exception & e) { tools::logger()->error("[{}] {}: {}", transport, stage, e.what()); }
      catch (...) { tools::logger()->error("[{}] {}: unknown exception", transport, stage); }
    };
    log_error("control worker failed", state.failure);
    log_error("stop write failed", state.stop_failure);
    tools::logger()->info("[{}] stop attempted={} host_write_complete={} device_stop=unverified",
      transport, state.stop_attempted, state.stop_written);
  } catch (...) { /* Destruction must not throw if logging itself is unavailable. */ }
}
}  // namespace io
#endif
