#include "cboard.hpp"
#include "io/serial/include/serial/serial.h"
#include "tools/thread_safe_queue.hpp"
#include <atomic>
#include <thread>


namespace io
{
class CBoardUART
{
public:
    std::atomic<double> bullet_speed{0};
    std::atomic<Mode> mode{Mode::idle};
    std::atomic<ShootMode> shoot_mode{ShootMode::left_shoot};
    std::atomic<double> ft_angle{0};

    CBoardUART(const std::string & config_path);
    ~CBoardUART();

    Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
    void send(Command command) const;

private:
    serial::Serial serial_;
    std::thread thread_;
    std::atomic<bool> stop_thread_;

    struct IMUData
    {
        Eigen::Quaterniond q;
        std::chrono::steady_clock::time_point timestamp;
    };

    tools::PoseHistory pose_history_;
    mutable ControlGuard control_guard_;
    std::unique_ptr<ControlPublisher> publisher_;
    std::unique_ptr<tools::LatencyStats> send_latency_;
    void write_control(const ControlIntent & intent) const;

    void read_thread();
};
}  // namespace io
