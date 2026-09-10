# NOTES — 调研与设计取舍

## 为什么是"静态多姿态 + 线性最小二乘"

- 需要的只是**重力项**（作者控制器的前馈只补重力，速度很慢，惯性/科氏项可忽略）。重力矩对负载参数
  `[m, m·c]` 严格线性，静止时几个姿态就能解，不需要激励轨迹、不需要惯量辨识。
- 4 个未知数，每个姿态给 7 个方程；12 姿态 × 双向 = 168 个方程。D-最优挑姿态后条件数 ≈1.5，
  加 α 之后 ≈15（α 与 m 在部分方向上相关，正常）。
- 备选方案（放弃）：拿力矩传感器 —— H2 没有；用 IMU/相机看末端 —— 精度差且和重力无直接关系。

## τ_true 用 PD 重构而不是 tau_est

稳态下 `kp·(q_cmd−q) − kd·dq + τ_ff` 就是控制器要电机出的力矩，也正是"下垂"的直接原因。
用它辨识，再补进前馈，消掉的就是这个偏差本身，闭环自洽。`tau_est` 是电流估算，含电机常数误差，
仿真里两者一致，真机上建议两者都解一遍对照；差得多说明电机力矩常数或 kp 实际值与设定不符。

## 摩擦

静摩擦让手臂"停在哪都行"的带宽 ≈ τ_stick/kp（0.8 Nm / 80 ≈ 0.6°）。单向逼近样本会系统性偏一边，
双向逼近在最小二乘里对消。剩余 0.1–0.3° 就是摩擦死区，前馈补不掉，只能靠"到位修正"（量到偏差后把指令角反向偏一点）。

## 真机接入方式

- 不复制作者 arm.py，而是 `class PayloadArmController(H2ArmController)`：只替换 `self._grav_model` 引用。
  作者 `_compute_tau` 每周期读 `self._grav_model.torque(cmd_q, g_dir=...)`，我们的 `GravityWithPayload.torque`
  返回 `α·作者模型 + 负载项`。控制线程不加锁读引用，Python 赋值原子，切换安全。
- 作者 `grav_alpha` 保持 1.0，α 在我们这层乘，避免 α 也去缩放负载项。
- 作者 `_tau_cap = 0.6 × URDF effort` 仍然生效：负载太重时前馈会被钳位，网页上会看到残差降不下来。
- DDS 初始化走作者 `backend.dds.ensure_dds_initialized`，与作者类共享"进程内只初始化一次"的标志；
  只读模式 `ReadOnlyH2Arm` 也用它，之后切到可运动模式不会二次初始化。

## 仿真物理层参数（controllers.SimArmController）

关节惯量 J、粘滞 B、静摩擦 τ_stick 是量级猜测，只求复现现象：kp 80 + J 0.3 → 16 rad/s 自然频率，
半隐式欧拉 4 ms 子步稳定。权重 w 期间把本体视作"理想托举"，等效前馈 `w·τ_ff + (1−w)·τ_true`。

## 姿态集

- **示教**走作者的卸力模式：`disable_jog()` → `enter_hand_move()`（kp=0、kd=2，`grav_in_float=True` 时重力前馈按实测角继续给）。
  空格 → `stop()`（作者语义：从实测角抓取、刚性保持、点动关闭）→ `wait_still` → 15 帧均值 → 追加。
  再按空格 → `enter_hand_move()`（点动已关，允许）。结束 → `stop()` + `enable_jog()`。
  仿真里没有手，`SimArmController.push(q)` 把手臂钉在滑块位置模拟"手握着"；锁定后松手，物理层会像真机一样略下垂再停。
- **自动生成**：安全盒子 手腕 x∈(0.15,0.55)、同侧 |y|∈(0.10,0.50)、z∈(−0.30,0.50)，肘 ≥0.3 rad。贪心 D-最优
  （每次加入使 log det(YᵀY) 增量最大的候选）。没有自碰撞检查 —— 在 3D 视图里用"点姿态名单独走到"逐个确认。
- 每次保存都重算回归矩阵条件数（列归一化后），网页实时显示，示教时用它判断姿态是否够多样。

## 与 IK_replay 18002 重力标定台的对照（2026-09-10）

18002（`yx/project/IK_replay/api/gravity_calibration.py`）是同事做的实验台：手拖存路点 → 经 reach 18001 走点、静置、
10 Hz 采 2 s，记录指令/实测/`tau_est`/前馈/增益；参数是版本化 profile（`config/gravity_compensation.json`），
当前激活 `0.1.0 = α1.1 + payload 0.5 kg（挂 hand_link 质心）`，由人工在 (α, 质量) 两标量上搜出，质心不可调。
他们实测最大关节误差：0.0.0 平均 1.94° → 0.1.0 平均 0.83°（kp=140）。

把他们 `data/gravity_calibration/runs/*/*.json` 的 75 个 sample_points 喂进 `identify.solve_payload`（字段映射：
`aggregate.command_rad/measured_rad/measured_velocity_rad_s/estimated_joint_torque_nm/feedforward_torque_nm` 取 `mean`，
kp/kd 取 `samples[0].arm`）：

| 前馈 | 剩余力矩 RMS | 折算下垂 |
|---|---|---|
| 0.0.0 α1 无负载 | 1.85 Nm | 0.58° |
| 0.1.0 现用 | 0.92 Nm | 0.36° |
| 本项目 α=1 解 m+质心 | 0.84 Nm（m=1.10±0.05 kg，质心≈腕心） | 0.35° |
| 本项目再解 α | 0.72 Nm，但 m=−0.75、α=1.65 → 病态 | 0.33° |

结论：① 他们 9 个路点姿态太像，质心/α 方向不可辨识（条件数 20、负质量）——示教时必须看条件数、多翻腕变高度；
② 前馈天花板 ≈0.7–0.8 Nm，是静摩擦/迟滞量级（kp140 ≈0.3°，kp80 ≈0.6°），任何重力模型都补不掉，只能靠到位修正或提 kp；
③ 换新末端后 0.5 kg 已失效，质心偏离手掌中心时腕部会留系统误差，这是 4 参数模型能看出差别的地方。
没有做成正式导入命令（数据病态、格式属别人、随时会变）；需要时按上面字段映射写十几行脚本即可。

## 已知问题 / 待办

- 真机未测。`ReadOnlyH2Arm` / `PayloadArmController` 在沙箱里因无 DDS 网络只验证到"5 s 无 lowstate → 报错留在仿真"。
- `still_dq_rad_s=0.01` 静止阈值来自仿真；真机 dq 噪声若更大需调（`config/poses_*.json` 的 `motion` 段可加字段覆盖）。
- 验证里"末端误差 mm"是 FK 算的，不是外部测量；要做绝对精度对比可结合 hand_eye_checkerboard 的相机看棋盘。
