# RM2026-AutoAim 修复版：数据契约、回放与 NUC 验收

## 1. 交付范围和使用边界

目标运行设备为 **Intel Core i5-12450H NUC**。Windows 工作站只用于编辑；测试和编译检查在 WSL 中完成。WSL 上的耗时不能代替这台 NUC 的实时性能或命中效果。

本轮保留 C++、OpenVINO、现有模型文件、串口/CAN 包格式和非 ROS2 运行路径。修复重点是错误位姿、过期数据、越界、非法弹道及不同入口的失效行为。没有把 CKF、大网络、端到端轨迹网络或完整 ROS2 迁移作为上线前提。

有界随机转速或无法辨识的隐藏转速规则并不会因为换滤波器而变得可准确长时预测。本轮使用短时状态外推和不确定度约束；未来相位区间过大时允许继续跟踪，但不应继续开火。

以下项目仍需实际设备和样本才能完成，不能把“代码存在”当作已验收：

- 相机曝光中点与姿态时钟的固定偏移及抖动标定。
- 云台坐标约定、反馈时效、通信/发弹延迟、真实弹速的标定。
- 倾斜旋转轴及不同安装高度的可靠多视角/多相位标定。
- 同一台 i5-12450H 上 CPU/GPU、YOLOv5/v8/11 的整链路 A/B。
- 真实目标、噪声、遮挡、强侧视、丢帧和模式切换的实机命中验收。

“P95 延迟降低至少 20%”是优化目标，不是已经测得的结果。未配置完整 ROS2 与 `sp_msgs` 时，CMake 会跳过哨兵相关目标；这种构建不能声称验证了全部哨兵入口。

## 2. 数据沿时间轴如何流动

```text
相机图像（拥有内存）+ 曝光时刻估计 + frame_id
    -> FramePacket + 对应时刻姿态
    -> 最新帧检测 / 角点与 PnP 质量
    -> TargetSnapshot（源时刻、目标状态与协方差、质量）
    -> 预测目标到达时刻 / 瞄准或 MPC
    -> ControlIntent（源时刻、有效截止、跟踪及开火许可）
    -> 唯一发送线程再次检查有效性 -> 兼容原协议的数据包
```

相关文件：`io/frame_packet.hpp`、`tasks/auto_aim/runtime.hpp`、`io/command.hpp`、`io/control_guard.hpp`、`io/control_publisher.hpp`。

`frame_id` 标识采集顺序；源时间标识这份结果实际观测了哪个时刻。不能在发送、重发、预测时把旧目标的源时间刷新成现在。模式切换会使上一模式的结果失效；最新队列不能代替这一检查。

姿态查询使用非消费式历史插值。同一图像被检测、瞄准、调试读取多次，不会因为第一次读取弹出样本而失去姿态。历史范围之外、不合理的采样间隔或非法四元数返回无效，不靠无限等待或外推制造同步成功。

发送线程只保存最新指令，限制最高发送频率。生产线程卡顿后，发送端仍能按源时刻撤销开火，超过跟踪时效后输出停止指令。这不能代替下位机自身的通信看门狗：如果整个主机或进程停止调度，主机无法保证继续发送停止包。

## 3. 配置与保守默认值

以下都是 YAML 顶层可选项。例子表示软件默认值，**不是 i5-12450H 的标定结果**：

```yaml
shoot_max_age_ms: 100.0
control_max_age_ms: 200.0
camera_transport_delay_ms: 0.0
camera_timing_uncertainty_ms: 5.0
camera_timing_calibrated: false
gimbal_use_packet_quaternion: false
runtime_metrics: false
```

`0 < shoot_max_age_ms <= control_max_age_ms`。开火时效必须比允许短暂跟随的时效更严格；先按实测帧周期、处理尾延迟和相位误差选择，再结合回放和实际效果调整，不能简单调大来掩盖卡顿。TTL 是最后的失效边界，不是允许每帧积压到这个年龄。

目前相机统一采用接收时间回退估计：

\[
t_\mathrm{exposure} = t_\mathrm{receive}
 - d_\mathrm{transport} - \frac{T_\mathrm{exposure}}{2}.
\]

`camera_transport_delay_ms` 必须和 SDK 所记录的接收时刻定义一致。设备时钟 tick 没有到主机单调时钟的映射时，不能直接与主机姿态时间相减。`camera_timing_calibrated: true` 只是使用者对标定完成的声明，不会执行标定或自动消除偏差；`camera_timing_uncertainty_ms` 是随帧保留的质量描述，目前不是完整的时钟误差协方差传播模型。

`gimbal_use_packet_quaternion` 默认关闭，以保留旧协议 yaw/pitch 构造姿态的坐标约定。只有确认包内四元数的轴方向、旋转方向和外参一致后才能启用完整四元数；仅验证其范数正确还不够。外置 IMU 路径使用其对应时刻的完整姿态。

保留 `high_speed_delay_time`、`low_speed_delay_time` 等配置用于兼容现有配置，但应把其物理含义统一为尚余通信/发弹等延迟，避免同时包含已经通过源时间补偿的图像处理时间。具体预测入口以 `Aimer`/`Planner` 的实现为准，不再由各入口额外随意减 1 ms 或增加提前量。

可选的 `actuation_delay_s` 会同时覆盖上述两个兼容字段，单位为秒。自瞄的串行瞄准和 MPC 共用 `tasks/auto_aim/interception.hpp` 的迭代拦截计算；飞行时间不收敛或弹速无效时返回无解，不代入虚构弹速。

## 4. 检测、PnP 与整车几何是三个独立层面

### 4.1 图像中的“上、下”不等于装甲板的物理角点

网络定位目标，传统灯条方法只做局部精修。精修失败应保留原始角点并降级质量，不能把失败结果包装成更精确的角点。图像旋转后仅按像素 y 排序会交换物理角点，对应关系一旦错了，PnP 得到低残差也未必是正确物理姿态。

四角顺序必须与模型训练标签和 `Solver` 物体点定义一致。不同 YOLO 导出模型的关键点索引不能互相套用。未能证明语义顺序的兼容路径只允许保守跟踪，不应授权自动开火。

YOLOv8 和 YOLO11 分别支持顶层 `yolov8_keypoint_order`、`yolo11_keypoint_order`：四个整数是 **物理 TL、TR、BR、BL** 在模型原始输出中的索引，必须是 `[0,1,2,3]` 的排列。只有确认训练标签顺序后才能填值，不提供可盲抄的默认映射。省略时保留旧图像 y 排序兼容路径，但 `corners_reliable=false`，不允许开火。YOLOv5 保留仓库现有的 `[0,3,2,1]` 映射。`use_traditional` 控制局部精修，v5/v8/11 统一应用失败降级规则。

### 4.2 PnP 选择候选，不强行把板立直

平面 PnP 可能给出多个候选。`tasks/auto_aim/solver.cpp` 使用正深度、逐角重投影误差、投影退化程度等质量条件；近期同一物理板的先验只能帮助区分候选，不能用另一块板或另一个目标的姿态作连续性依据。

输出分为 `pose_valid` 和 `pose_reliable`。它们分别表示存在可供后续处理的几何解和该解是否足够可靠，不应合并为一个“检测成功”布尔值。两个候选不可区分、强侧视或角点几乎重合时，应显式降低质量。取消对普通装甲板强制固定俯仰/零滚转的 yaw 修正，并不意味着斜轴整车运动已被正确建模。

PnP 顶层可选阈值如下，单位不能混用：

```yaml
pnp_max_reprojection_error: 3.0  # 四角 RMS 像素误差
pnp_max_corner_error: 6.0        # 单角最大像素误差
pnp_min_projected_edge: 2.0      # 最短投影边长度，像素
pnp_ambiguity_error_gap: 0.25    # 候选 RMS 误差差值，像素
pnp_min_view_cosine: 0.1        # 视线与法向夹角余弦绝对值
```

这些是保守的软件阈值，需要随分辨率、标定误差和角点质量校准。降低可靠性门限可能增加“能开火”的帧，却不代表位姿更正确。

### 4.3 整车几何采用有约束的校准，而非单帧猜四个高度

整车位置关系使用旋转中心、相位、轴方向和板在车体坐标中的偏移：

\[
p_i(t)=c(t)+B(a)R_z(\theta(t))o_i.
\]

`B(a)` 把轴坐标系 z 轴对齐到世界轴方向 `a`；`o_i` 是与板编号顺序一致的已标定偏移，板法向另行给出。不同高度体现在偏移的 z 分量；倾斜轴体现在 `B(a)`。位置和法向不能混为一谈。

未标定路径保留旧版的简化半径/交错高度参数。已标定路径可以描述同高、交错两组高度和独立高度，不会在线自由拟合所有高度及任意倾轴。只有一块板或短视角观测时，整车身份、相位与高度偏移存在不可观测性。几何证据不足时退回当前可见板保守跟随，而不是从一帧制造其余板的位置。

安装顺序和首块板身份必须与标定约定一致。若四块板外观无法区分，首板身份歧义必须通过额外相位观测或离线校准解决，不能因为提供了四个高度数组就认为问题已经消失。

代码配置接口位于 `Tracker` 与 `Target::configure_geometry`。下面只是几何回归用例的数值示意，不是可直接部署的机器人标定：

```yaml
auto_aim_geometry:
  three:  # ARMOR_NAMES 中的名称；这里是四板小装甲板目标
    axis_in_world: [0.2, 0.3, 1.0]  # 自动归一化；轴方向在跟踪期间固定
    plate_offsets:                # 单位 m，轴坐标系下，相位为零时的位置
      - [-0.20,  0.00,  0.00]
      - [ 0.00, -0.22,  0.05]
      - [ 0.20,  0.00,  0.09]
      - [ 0.00,  0.22, -0.03]
    plate_normals:                # 与现有 PnP 板坐标 x 轴一致的法向约定
      - [ 1.0,  0.0, 0.0]
      - [ 0.0,  1.0, 0.0]
      - [-1.0,  0.0, 0.0]
      - [ 0.0, -1.0, 0.0]
```

向量数必须等于该兵种的板数：平衡目标两板、前哨站/当前基地模型三板、普通目标四板。第一块被观测的板被当作 profile 的第 0 块；若无法保证这一对应，**不要开启不对称高度 profile**。本轮没有实现自动板身份辨识、在线三种安装方案选择或在线轴倾角估计。未配置 profile 时，明显滚转或连续创新拒绝会降级到当前板跟随，降级状态不允许开火。

## 5. 估计、规划与开火许可

`tools/extended_kalman_filter.cpp` 的 NIS 使用更新前创新及其协方差：

\[
\nu=z-h(x^-),\quad S=HP^-H^T+R,\quad
\mathrm{NIS}=\nu^T S^{-1}\nu.
\]

创新统计的自由度是观测维度。没有真值不能用“更新前后状态差”冒充 NEES。普通 EKF 使用者与自瞄 `Target` 的拒绝策略分开，避免修改共享滤波器时无意改变能量机关等模块的行为。

运动模型保留 EKF，增加有界的短时角加速度估计与运动不确定状态。规划采样仅需均值时，不重复传播不使用的协方差；开火判断仍使用不确定度，不能因为省掉 MPC 采样中的协方差计算而省掉火控风险判断。

CKF 仍适合作为同一观测、几何、噪声和数据关联条件下的离线对照。它不修复错角点、时间偏差、板身份混淆或错误旋转轴；本轮没有直接把所有 EKF 替换为 CKF。ESO 的云台扰动估计也未加入默认主路径。

当前火控约束包含位姿可靠性、目标年龄、有限指令、可解弹道、运动相位不确定度和瞄准误差。它仍不是完整的弹丸散布模型：真实弹速分布、枪口偏差、云台动态误差和三维板内落点分布还需标定。不能把“所有软件门限通过”宣传为保证命中。

MPC 的规划参考误差不能替代实际云台误差。`runtime.hpp` 中没有传入实测云台状态的转换接口只生成跟踪指令；带反馈的接口还检查板面投影尺寸约束下的 yaw/pitch 误差。串口发送线程继续检查反馈年龄与弹速有效性，防止外置 IMU 正常而云台反馈停更时沿用旧开火状态。

现阶段不确定度门限主要约束相位，并未完成位置、时钟、弹速、散布的统一三维命中概率计算。PnP 已提供可靠同板先验的候选消歧接口和单元测试，但 Tracker 尚未实现可证明同一物理板的数据关联，因此默认不向它传入上一块同编号装甲板冒充的先验。

## 6. 不连接硬件的回放

`tests/auto_aim_replay.cpp` 读取现有 `Recorder` 输出的一对同前缀文件：

- `sample.avi`：逐帧图像。
- `sample.txt`：每行 `t w x y z`，其中 t 是记录的相对秒数，四元数顺序为 **wxyz**。

它构造 `YOLO -> Tracker（包括 PnP）-> Planner`，不构造相机、云台、CBoard 或串口对象。`fire_requested` 只是 CSV 中的离线决策字段，不发送任何硬件命令。

在 WSL、仓库根目录运行，先按本机安装路径加载 OpenVINO 环境：

```bash
source /opt/intel/openvino_2026/setupvars.sh
cmake -S . -B build/codex-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/codex-release --target auto_aim_replay -j2
./build/codex-release/auto_aim_replay records/sample \
  --config-path=configs/demo.yaml --output=build/replay.csv \
  --bullet-speed=27 --warmup=20 --decision-delay-ms=30
```

`device`、`yolo_name` 和模型路径全部取自所选 YAML，不会根据 WSL 环境偷偷切换模型或 CPU/GPU。WSL 没有可用 GPU 插件时，应显式选用用于对比的 CPU 配置；这不是 GPU 验证成功。

`--decision-delay-ms` 是固定模拟源年龄，不是强制 sleep，也不是测得的端到端延迟。回放使用记录时间差驱动预测，避免录像解码速度或 CSV 写入速度改变运动轨迹；不同版本以相同参数回放才能比较预测结果。

输出包括：

- 每帧检测数、已解/可靠位姿数、目标数、跟踪状态。
- 是否请求控制/开火、指令有限性、非法开火数量、目标源年龄。
- `detect_ms`、`tracker_pnp_ms`、`planner_ms`、`processing_ms`。
- 去除前 N 帧预热后的处理耗时 P50/P95/max；所有帧仍保留在 CSV。

视频解码、CSV 写入、相机、姿态通信、指令发送和发弹不包含在上述处理耗时中，因此不能直接把 `1000 / processing_ms` 当成实机 FPS 或曝光到发送延迟。PnP 位于 Tracker 内部，单独声称这是纯 Tracker 时间同样不准确。

空录像、时间不递增、时间行与视频数量不匹配、配置/模型失败返回非零。非法/过期开火或者非有限控制结果返回 1；输入及运行错误返回 2。没有检测到目标不等于工具失败，空场和噪声本就是必测场景。

## 7. 运行时延迟统计及正确解释

`tools/latency_stats.hpp` 提供默认关闭的固定 512 样本环形统计。`record_first_send` 在实际成功写出控制包后记录源时间到首次发送的延迟；同一结果的周期重发和乱序旧结果不会增加样本数。

P50/P95/max 对应最近最多 512 个有效样本，而 `total_samples` 是累计接收样本数。不可把多个窗口的 P95 求平均当成全程 P95，也不可把 `total_samples` 误当成窗口容量。需要全程指标时，在同一测量方案中保留原始逐帧时刻或分段比较相同长度的稳态窗口。

将 `runtime_metrics: true` 后，Gimbal、CBoard、CBoardUART 在正常退出时输出统计摘要。进程被强杀不会触发这个析构摘要；样本只记录成功写出的包，不表示下位机已执行。

源时刻仍是回退曝光估计时，这个指标应称为“估计曝光到成功发送”，不能称为硬件同步曝光延迟。仅统计实际发送的有效目标结果还会有选择偏差：同时记录无目标比例、丢帧/过期数量和有效控制比例，避免通过少发或不发困难帧获得漂亮延迟。

## 8. 一轮针对性验证流程

只安排一轮基线、一轮针对性回放和一轮 NUC 实机 A/B。构建失败时修复对应问题再重跑受影响的检查，不反复重做全部检查。

1. **基线**：同一台 NUC 固定电源模式、散热、模型、分辨率、曝光、弹速和素材；记录实际入口、处理时间与估计曝光到发送时间，区分空场、目标和高噪声。
2. **WSL 无硬件检查**：运行新增自动化回归；用同一段录像进行一次 headless 回放。硬件控制/开火测试程序不作为自动化测试直接执行。
3. **NUC A/B**：相同场景、模型和参数对比修复前后；串行/轻量异步分别测量。启动程序需要具体设备和部署授权，回放通过不能替代这个步骤。

固定覆盖：两板目标、无目标和高噪声、强侧视与斜板、倾斜旋转轴、各板不同高度、稳定/变化/突变转速、掉帧、相机停滞和模式切换。

必须为零：越界、未初始化飞行时间、非有限发送指令、已过期开火结果。退化 PnP 无法判别时要正确降级，不能只统计成功解算帧。实机有效命中窗口和命中表现不得低于基线；延迟不得靠积压旧帧换取更高表面 FPS。

模型压缩、INT8、动态 ROI、短滑窗联合重投影优化仅在上述证据显示具体收益后进入下一轮。当前改动不自动开启这些方案，也不提供未经本机测试的 FPS 数字。

## 9. 本次 WSL 验证记录（2026-09-27）

- 环境：Ubuntu 22.04 / GCC 11.4 / Release；已加载 ROS2 Humble，已构建本仓库 `sp_msgs`。
- `cmake --build build/codex-release -j2` 成功。包含 `standard`、`mt_standard`、`standard_mpc`、自瞄/打符调试入口、`uav` / `uav_debug`，以及 `sentry`、`sentry_bp`、`sentry_debug`、`sentry_test_new`、`sentry_multithread`；不是跳过 ROS2 后的部分构建。
- `ctest --test-dir build/codex-release --output-on-failure`：**6/6 通过**。

| 测试 | 本轮覆盖 |
| --- | --- |
| `runtime_contracts` | 姿态插值和重复历史读取；指令/反馈时效；模式切换；处理停滞后停止；固定容量延迟统计 |
| `queue_contracts` | 最新帧替换；超时；关闭唤醒；独占内存；满队列回调不死锁 |
| `pose_contracts` | 合成完整三维斜板；强侧视；平面两解；错误/重合/NaN 角点；无效姿态；语义重排 |
| `estimation_contracts` | 两板候选；先验 NIS；非法弹道；已标定斜轴和独立高度；均值预测；过期目标；实测云台误差 |
| `inference_contracts` | 仓库 YOLOv5 的真实 OpenVINO CPU 推理；空场/合成噪声；异步输入内存所有权；新旧结果顺序；模式代际隔离 |
| `replay_smoke` | 生成的 8 帧空场/噪声录像与时间戳通过完整 headless 回放链路，无非有限指令或非法开火 |

测试程序未打开相机、串口或 CAN，未向机器人发送控制指令。生成的 CPU 配置、合成视频与 CSV 位于 `build/codex-release/`，原部署 YAML 没有被切换为 CPU。Windows 原生测试不计入本轮验收。

仍未验证：真实录像中的识别精度与预测误差、GPU 推理、实机驱动时序、下位机实际执行、真实弹道散布、NUC 延迟与命中率。合成空场回放只是接口冒烟测试，不能作为真实装甲板场景的性能基线。

### WSL 复现命令

ROS2 Humble 的旧生成器在本工作区中文构建路径下读取生成文件列表会失败；仅消息包在英文临时目录构建，再安装进项目 `build/`，不修改 ROS2 安装。

```bash
source /opt/ros/humble/setup.bash
source /opt/intel/openvino_2026/setupvars.sh
cmake -S sp_msgs -B /tmp/rm2026-msgs-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/codex-msgs-install" \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build /tmp/rm2026-msgs-build -j2
cmake --install /tmp/rm2026-msgs-build
cmake -S . -B build/codex-release -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -Dsp_msgs_DIR="$PWD/build/codex-msgs-install/share/sp_msgs/cmake"
cmake --build build/codex-release -j2
ctest --test-dir build/codex-release --output-on-failure
```

以上命令在 **WSL Bash 内**执行。不要通过多层 PowerShell/Bash 拼接让 `$PWD` 提前被其他 shell 展开。本次失败配置产生的临时根目录构建文件已保留移入 `build/codex-msgs-in-source-failed/`，没有删除项目源文件。
