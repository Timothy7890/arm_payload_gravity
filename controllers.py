"""三种"手臂控制器"，对上层暴露同一套接口（与作者 H2ArmController 同名同语义）：

    start() / enable_jog() / set_target(q) / stop() / disable_jog() / shutdown()
    set_max_speed(v) / set_payload(mass, com, alpha) / payload() / status()

- SimArmController      仿真：同样的 50 Hz 环、权重渐入、矢量同步限速、URDF 钳位、重力前馈；
                        下面挂一个带**真实负载 + 库仑摩擦 + 质量误差**的关节动力学模型，
                        所以会复现真机的"纯 PD 下垂"和"摩擦死区"，可以无机器人测试整条辨识/验证流水线。
- PayloadArmController  真机：**子类化**作者的 H2ArmController，只替换 ``_grav_model`` 为
                        "作者原模型 + 末端负载项"（gravity.GravityWithPayload）。发令通道、限速、
                        钳位、权重渐入渐出、力矩上限全部沿用作者实现，未改一行。
                        ⚠ 构造即建立 rt/arm_sdk 发布者；start() 后开始发令接管手臂。
- ReadOnlyH2Arm         真机只读：只订阅 rt/lowstate 显示关节角/力矩，不能运动（没开 --allow-real-motion 时用）。

本文件不含任何自行组装 LowCmd_ 的代码：真机发令只通过作者的类。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from gravity import ArmGravity, GravityWithPayload

CONTROL_DT = 0.02
WEIGHT_RAMP_S = 1.0
DEFAULT_KP, DEFAULT_KD = 80.0, 1.5
DEFAULT_KP_WRIST, DEFAULT_KD_WRIST = 50.0, 2.0
EFFORT_MARGIN = 0.6

AUTHOR_DIR = Path("/home/robot/yx/project/calib/hand_eye_3D")   # 只读 import
UNITREE_SDK_PY = Path("/home/robot/unitree_sdk2_python")
MOTOR_INDICES = {"left": list(range(15, 22)), "right": list(range(22, 29))}


def _ensure_author_paths() -> None:
    for p in (AUTHOR_DIR, UNITREE_SDK_PY):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def gain_vectors(joint_names, kp=DEFAULT_KP, kd=DEFAULT_KD, kp_w=DEFAULT_KP_WRIST, kd_w=DEFAULT_KD_WRIST):
    is_wrist = np.array(["wrist" in n for n in joint_names])
    return np.where(is_wrist, kp_w, kp).astype(float), np.where(is_wrist, kd_w, kd).astype(float)


# ==================================================================== 仿真
class SimTruth:
    """仿真世界的"真值"，控制器不知道这些数。"""

    def __init__(self, payload_mass: float = 1.2, payload_com=(0.10, 0.0, -0.02),
                 arm_mass_scale: float = 1.04, friction_scale: float = 1.0,
                 tau_noise: float = 0.03, q_noise: float = 5e-5):
        self.payload_mass = float(payload_mass)
        self.payload_com = np.asarray(payload_com, float).reshape(3)
        self.arm_mass_scale = float(arm_mass_scale)     # URDF 质量参数与实物的偏差
        self.friction_scale = float(friction_scale)
        self.tau_noise = float(tau_noise)
        self.q_noise = float(q_noise)
        self.foreign_publisher = False      # 模拟"有别人在发 rt/arm_sdk"

    def as_dict(self) -> Dict:
        return {"payload_mass": self.payload_mass, "payload_com": self.payload_com.tolist(),
                "arm_mass_scale": self.arm_mass_scale, "friction_scale": self.friction_scale,
                "tau_noise": self.tau_noise, "q_noise": self.q_noise, "foreign_publisher": self.foreign_publisher}

    def update(self, d: Dict) -> None:
        if "foreign_publisher" in d:
            self.foreign_publisher = bool(d["foreign_publisher"])
        if "payload_mass" in d:
            self.payload_mass = max(0.0, float(d["payload_mass"]))
        if "payload_com" in d:
            self.payload_com = np.asarray(d["payload_com"], float).reshape(3)
        for k in ("arm_mass_scale", "friction_scale", "tau_noise", "q_noise"):
            if k in d:
                setattr(self, k, float(d[k]))


class SimArmController:
    """仿真控制器：控制律照搬作者 H2ArmController._loop / _compute_tau；物理层见 _physics。"""

    kind = "sim"
    controllable = True
    # 关节等效惯量 / 粘滞阻尼 / 静摩擦（Nm）——量级上的合理猜测，只为复现现象
    J = np.array([0.30, 0.30, 0.15, 0.15, 0.03, 0.03, 0.02])
    B = np.array([2.0, 2.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    TAU_STICK = np.array([0.8, 0.8, 0.5, 0.5, 0.15, 0.15, 0.12])
    SUBSTEPS = 5

    def __init__(self, arm: str = "right", max_speed_rad_s: float = 0.15, truth: Optional[SimTruth] = None,
                 q0=None, seed: int = 0):
        self.arm = arm
        self.grav = ArmGravity(arm)
        self.joint_names = list(self.grav.joint_names)
        self.n = 7
        self.limits = self.grav.limits
        self.max_speed = float(max_speed_rad_s)
        self._speed_ceiling = max(self.max_speed, 0.3)
        self.kp_vec, self.kd_vec = gain_vectors(self.joint_names)
        self._tau_cap = EFFORT_MARGIN * self.grav.effort_limits
        self.truth = truth or SimTruth()
        self._rng = np.random.default_rng(seed)
        self._gm = GravityWithPayload(self.grav, self.grav, 0.0, (0, 0, 0), 1.0)

        q0 = np.asarray(q0 if q0 is not None else [0.2, -0.25 if arm == "right" else 0.25, 0.0, 0.9, 0.0, -0.1, 0.0], float)
        self._q = self._clamp(q0)          # 物理真值
        self._dq = np.zeros(self.n)
        self._cmd_q = self._q.copy()
        self._desired_q = self._q.copy()
        self._weight = 0.0
        self._engaged = False
        self._jog = False
        self._float = False                 # 卸力拖动（示教）：kp=0，重力前馈按实测角给
        self._hand_q = None                 # 仿真里"人的手"：非 None 时把手臂钉在这里
        self.hand_move_kd = 2.0
        self._tau_grav = np.zeros(self.n)
        self._tau_ff_sent = np.zeros(self.n)
        self._tau_motor = np.zeros(self.n)
        self._seq = 0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # 未接管时也让物理层跑：模拟本体控制器托着（等效于刚性保持在 q0 且无下垂）
        self._phys_thread = threading.Thread(target=self._phys_loop, daemon=True)
        self._phys_thread.start()

    # ---- 物理 ----
    def _clamp(self, q):
        return np.clip(np.asarray(q, float).reshape(-1), self.limits[:, 0], self.limits[:, 1])

    def _tau_true_gravity(self, q) -> np.ndarray:
        t = self.truth
        tau = t.arm_mass_scale * self.grav.tau_arm(q)
        if t.payload_mass > 0:
            tau = tau + self.grav.tau_payload(q, t.payload_mass, t.payload_com)
        return tau

    def _physics(self, cmd_q, kp, kd, tau_ff, dt) -> None:
        """半隐式欧拉，SUBSTEPS 子步。电机力矩 = PD + 前馈；负载 = 真实重力 + 摩擦。"""
        h = dt / self.SUBSTEPS
        q, dq = self._q, self._dq
        for _ in range(self.SUBSTEPS):
            tau_motor = kp * (cmd_q - q) - kd * dq + tau_ff
            net = tau_motor - self._tau_true_gravity(q) - self.B * dq
            stick = self.TAU_STICK * self.truth.friction_scale
            moving = np.abs(dq) > 2e-3
            # 静摩擦：几乎静止且驱动力矩小于静摩擦 → 锁死；否则库仑摩擦反向
            hold = (~moving) & (np.abs(net) < stick)
            net = np.where(hold, 0.0, net - np.where(moving, np.sign(dq), np.sign(net)) * 0.7 * stick)
            dq = np.where(hold, 0.0, dq + h * net / self.J)
            q = q + h * dq
            lo, hi = self.limits[:, 0], self.limits[:, 1]
            hit = (q < lo) | (q > hi)
            q = np.clip(q, lo, hi); dq = np.where(hit, 0.0, dq)
        self._q, self._dq, self._tau_motor = q, dq, tau_motor

    def _phys_loop(self) -> None:
        """控制线程没跑时（未接管 / 已释放），把手臂"托"在当前位置。"""
        next_t = time.perf_counter()
        while True:
            if self._thread is None or not self._thread.is_alive():
                with self._lock:
                    hold_q = self._q.copy()
                    self._cmd_q = hold_q.copy(); self._desired_q = hold_q.copy()
                # 本体控制器：视作理想托举（前馈 = 真实重力）
                self._physics(hold_q, self.kp_vec, self.kd_vec, self._tau_true_gravity(hold_q), CONTROL_DT)
            next_t += CONTROL_DT
            s = next_t - time.perf_counter()
            time.sleep(s if s > 0 else 0.0)
            if s <= 0:
                next_t = time.perf_counter()

    # ---- 控制环（照搬作者） ----
    def _loop(self) -> None:
        next_t = time.perf_counter()
        while True:
            stopping = self._stop_evt.is_set()
            with self._lock:
                if stopping:
                    self._weight = max(0.0, self._weight - CONTROL_DT / WEIGHT_RAMP_S)
                else:
                    self._weight = min(1.0, self._weight + CONTROL_DT / WEIGHT_RAMP_S)
                w = self._weight
                float_mode = self._float
                if not float_mode:
                    step = self.max_speed * CONTROL_DT
                    delta = self._desired_q - self._cmd_q
                    worst = float(np.max(np.abs(delta)))
                    if worst > step:
                        delta = delta * (step / worst)
                    self._cmd_q = self._cmd_q + delta
                cmd_q = self._cmd_q.copy()
                gm = self._gm
                hand_q = self._hand_q
            # 重力前馈：保持/点动按指令角；卸力按实测角（作者 grav_in_float=True 的行为）
            tau_grav = gm.torque(self._q if float_mode else cmd_q)
            tau_ff = np.clip(tau_grav * w, -self._tau_cap, self._tau_cap)
            # 权重 w：SDK 与本体混合。本体视作理想托举 → 混合后的等效前馈
            tau_true = self._tau_true_gravity(self._q)
            eff_ff = w * tau_ff + (1 - w) * tau_true
            if float_mode:
                if hand_q is not None:            # 人手握着：钉住
                    self._q, self._dq = self._clamp(hand_q), np.zeros(self.n)
                    self._tau_motor = tau_ff
                else:                             # 松手：kp=0，只剩前馈与阻尼，模型误差会让它慢慢飘
                    self._physics(self._q, np.zeros(self.n), np.full(self.n, self.hand_move_kd), eff_ff, CONTROL_DT)
            else:
                eff_cmd = w * cmd_q + (1 - w) * self._q
                self._physics(eff_cmd, self.kp_vec, self.kd_vec, eff_ff, CONTROL_DT)
            with self._lock:
                self._tau_grav = tau_grav; self._tau_ff_sent = tau_ff; self._seq += 1
            if stopping and w <= 0.0:
                break
            next_t += CONTROL_DT
            s = next_t - time.perf_counter()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.perf_counter()

    # ---- 接口 ----
    def start(self) -> None:
        if self._engaged:
            return
        with self._lock:
            self._cmd_q = self._q.copy(); self._desired_q = self._q.copy(); self._weight = 0.0
        self._stop_evt.clear()
        self._engaged = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        if not self._engaged:
            return
        self._stop_evt.set()
        if self._thread:
            self._thread.join(WEIGHT_RAMP_S + 1.0)
        self._engaged = False
        self._jog = False

    def enable_jog(self) -> None:
        with self._lock:
            if self._float:
                self._cmd_q = self._q.copy()
            self._float = False; self._hand_q = None
            self._desired_q = self._cmd_q.copy(); self._jog = True

    def disable_jog(self) -> None:
        with self._lock:
            self._desired_q = self._cmd_q.copy(); self._jog = False

    def stop(self) -> None:
        """冻结 + 刚性保持（从卸力恢复时抓取当前实测角）——与作者语义一致。"""
        with self._lock:
            if self._float:
                self._cmd_q = self._q.copy()
            self._float = False; self._hand_q = None
            self._desired_q = self._cmd_q.copy(); self._jog = False

    def enter_hand_move(self) -> bool:
        """卸力拖动，仅点动关闭时允许（与作者一致）。"""
        with self._lock:
            if self._jog or not self._engaged:
                return False
            self._float = True
        return True

    def push(self, q) -> None:
        """仿真专用：模拟人手把手臂推到 q（只在卸力模式下有效）。"""
        q = self._clamp(np.asarray(q, float).reshape(-1))
        with self._lock:
            if not self._float:
                raise RuntimeError("只有卸力（示教）模式下才能手推")
            self._hand_q = q

    def set_max_speed(self, v: float) -> None:
        with self._lock:
            self.max_speed = float(np.clip(v, 0.05, self._speed_ceiling))

    def set_target(self, q) -> bool:
        q = np.asarray(q, float).reshape(-1)
        if q.size != self.n or not np.all(np.isfinite(q)):
            raise ValueError("目标关节角非法")
        with self._lock:
            if not self._jog:
                return False
            self._desired_q = self._clamp(q)
            return True

    def set_payload(self, mass: float, com, alpha: float = 1.0) -> None:
        with self._lock:
            self._gm = GravityWithPayload(self.grav, self.grav, mass, com, alpha)

    def payload(self) -> Dict:
        return self._gm.as_dict()

    def read_measured(self) -> np.ndarray:
        return self._q + self._rng.normal(0, self.truth.q_noise, self.n)

    def status(self) -> Dict:
        with self._lock:
            cmd, des, w, jog, fl = self._cmd_q.copy(), self._desired_q.copy(), self._weight, self._jog, self._float
            tg, tff, seq = self._tau_grav.copy(), self._tau_ff_sent.copy(), self._seq
            q, dq, tm = self._q.copy(), self._dq.copy(), self._tau_motor.copy()
        noise = self.truth
        return {
            "kind": self.kind, "arm": self.arm, "engaged": self._engaged, "jog_enabled": jog, "float": fl, "weight": w,
            "joint_names": self.joint_names,
            "measured_rad": (q + self._rng.normal(0, noise.q_noise, self.n)).tolist(),
            "measured_dq_rad_s": dq.tolist(),
            "tau_est_nm": (tm + self._rng.normal(0, noise.tau_noise, self.n)).tolist(),
            "cmd_rad": cmd.tolist(), "desired_rad": des.tolist(),
            "tau_grav_nm": tg.tolist(), "last_sent_tau_ff_nm": tff.tolist(), "last_sent_sequence": seq,
            "kp_vec": self.kp_vec.tolist(), "kd_vec": self.kd_vec.tolist(),
            "max_speed_rad_s": self.max_speed, "limits_rad": self.limits.tolist(),
            "payload": self.payload(), "sim_truth": self.truth.as_dict(),
        }


# ==================================================================== rt/arm_sdk 占用检测
class ArmSdkMonitor:
    """只读订阅 rt/arm_sdk，数别人（或我们自己）发令的频率。

    DDS 是多对多：任何进程往 rt/arm_sdk 发的 LowCmd_ 我们都能收到。接管前若已有消息在流 → 有人在控臂，
    此时再发令只会两边打架（手臂发抖 / 我们的卸力被对方 kp 顶住，看起来"没卸力"）。
    接管后我们自己也在发（50 Hz），所以运行期用 rate − 50 判断是否还有第三方。
    """

    OWN_RATE_HZ = 1.0 / CONTROL_DT

    def __init__(self, network_interface: Optional[str] = None):
        _ensure_author_paths()
        from backend.dds import ensure_dds_initialized  # type: ignore
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

        ensure_dds_initialized(network_interface)
        self._lock = threading.Lock()
        self._stamps: list = []
        self._last_weight = None
        self._last_kp = None
        self._mark = 0.0
        self._sub = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self._sub.Init(self._on_cmd, 50)

    def mark(self) -> None:
        """忽略此刻之前的消息（我们刚释放，窗口里还是自己的发令）。"""
        with self._lock:
            self._mark = time.monotonic()
            self._last_weight = self._last_kp = None

    def _on_cmd(self, msg) -> None:
        now = time.monotonic()
        with self._lock:
            self._stamps.append(now)
            if len(self._stamps) > 2000:
                del self._stamps[:1000]
            try:
                self._last_weight = float(msg.motor_cmd[31].q)
                self._last_kp = float(max(msg.motor_cmd[i].kp for i in range(15, 29)))
            except Exception:  # noqa: BLE001
                pass

    def rate_hz(self, window_s: float = 1.0) -> float:
        now = time.monotonic()
        with self._lock:
            n = sum(1 for t in self._stamps if now - t <= window_s and t > self._mark)
        return n / window_s

    def report(self, we_publish: bool) -> Dict:
        r = self.rate_hz()
        foreign = r - (self.OWN_RATE_HZ if we_publish else 0.0)
        with self._lock:
            w, kp = self._last_weight, self._last_kp
        return {"rate_hz": round(r, 1), "foreign_rate_hz": round(max(0.0, foreign), 1),
                "foreign": foreign > 8.0,          # 容忍抖动；第三方最少也是 50 Hz
                "last_weight": w, "last_kp": kp, "local_hints": local_arm_sdk_processes()}

    def close(self) -> None:
        try:
            self._sub.Close()
        except Exception:  # noqa: BLE001
            pass


# 只匹配参数部分（不含解释器路径，否则 .../envs/teleop/bin/python 会把所有进程都命中）。
# "python -u - ... 192.168.123.162" 是 teleop 从 stdin 喂脚本起的 holder（之前见过 250 Hz kp 80 常驻），只能靠机器人 IP 认。
_HINT_PATTERNS = ("reach_server", "calibration_replay", "h2_openpi", "ArmSdkTargetHold", "arm_sdk", "hand_eye_3D",
                  "--arm-control", "teleop_web", "192.168.123.")


def local_arm_sdk_processes() -> list:
    """本机上疑似在控臂的进程（只是提示，DDS 发布者也可能在别的机器上）。"""
    import os
    me = os.getpid()
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = " ".join(a.decode(errors="replace") for a in argv[1:] if a)
        if any(p in args for p in _HINT_PATTERNS) and "arm_payload_gravity" not in args:
            out.append({"pid": int(d.name), "cmd": (argv[0].decode(errors="replace").rsplit("/", 1)[-1] + " " + args)[:160]})
    return out[:10]


class SimArmSdkMonitor:
    """仿真：可用 truth.foreign_publisher 模拟"有人占用"来测网页逻辑。"""

    def __init__(self, truth: SimTruth):
        self.truth = truth

    def report(self, we_publish: bool) -> Dict:
        foreign = bool(getattr(self.truth, "foreign_publisher", False))
        r = (50.0 if we_publish else 0.0) + (250.0 if foreign else 0.0)
        return {"rate_hz": r, "foreign_rate_hz": 250.0 if foreign else 0.0, "foreign": foreign,
                "last_weight": 1.0 if foreign else None, "last_kp": 80.0 if foreign else None,
                "local_hints": [{"pid": 0, "cmd": "（仿真）模拟的第三方 holder"}] if foreign else []}

    def close(self) -> None:
        pass


# ==================================================================== 真机（可运动）
def make_payload_arm_controller(arm: str, network_interface: Optional[str], max_speed_rad_s: float = 0.15):
    """延迟 import 作者代码后再定义子类（避免无 SDK 环境 import 本模块就失败）。"""
    _ensure_author_paths()
    from backend.arm import H2ArmController  # type: ignore

    class PayloadArmController(H2ArmController):
        kind = "h2"
        controllable = True

        def __init__(self, arm: str, network_interface: Optional[str], max_speed_rad_s: float):
            # grav_in_float=True：示教卸力时重力前馈继续给（按实测角），手臂近似失重、推到哪停哪
            # （与作者 calibration_replay 引擎一致）。负载未补偿前它仍会往下坠，人要扶住。
            super().__init__(arm=arm, network_interface=network_interface, max_speed_rad_s=max_speed_rad_s,
                             grav_alpha=1.0, payload_kg=0.0, grav_in_float=True)
            self._our_grav = ArmGravity(arm)
            self._base_grav = self._grav_model           # 作者的 ArmGravityModel（臂自重）
            self._gm = GravityWithPayload(self._base_grav, self._our_grav, 0.0, (0, 0, 0), 1.0)
            self._grav_model = self._gm                  # _compute_tau 每周期读这个引用

        def set_payload(self, mass: float, com, alpha: float = 1.0) -> None:
            gm = GravityWithPayload(self._base_grav, self._our_grav, mass, com, float(np.clip(alpha, 0.0, 1.2)))
            self._gm = gm
            self._grav_model = gm                        # 原子替换引用，控制线程下一周期生效

        def payload(self) -> Dict:
            return self._gm.as_dict()

        def push(self, q) -> None:
            raise PermissionError("真机没有'模拟手推'：请直接用手推手臂")

        def status(self) -> Dict:
            st = super().status()
            st.update({"kind": self.kind, "kp_vec": self.kp_vec.tolist(), "kd_vec": self.kd_vec.tolist(),
                       "payload": self.payload()})
            return st

    return PayloadArmController(arm, network_interface, max_speed_rad_s)


# ==================================================================== 真机（只读）
class ReadOnlyH2Arm:
    """只订阅 rt/lowstate；status() 形状与上面一致，但不能动。"""

    kind = "h2-readonly"
    controllable = False

    def __init__(self, arm: str, network_interface: Optional[str] = None, timeout: float = 5.0):
        _ensure_author_paths()
        from backend.dds import ensure_dds_initialized  # type: ignore  # 与作者共用"进程内只初始化一次"
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self.arm = arm
        self.grav = ArmGravity(arm)
        self.joint_names = list(self.grav.joint_names)
        self.n = 7
        self.limits = self.grav.limits
        self.kp_vec, self.kd_vec = gain_vectors(self.joint_names)
        self._idx = MOTOR_INDICES[arm]
        ensure_dds_initialized(network_interface)
        self._lock = threading.Lock()
        self._state = None
        self._stamp = 0.0
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        deadline = time.monotonic() + timeout
        while self._state is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._state is None:
            raise TimeoutError(f"{timeout:.0f}s 内没收到 rt/lowstate（网卡对吗？机器人开机了吗？）")

    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg; self._stamp = time.monotonic()

    def _vals(self, field: str):
        with self._lock:
            st, stamp = self._state, self._stamp
        if st is None or time.monotonic() - stamp > 1.0:
            return None
        try:
            return [float(getattr(st.motor_state[i], field)) for i in self._idx]
        except AttributeError:
            return None

    # 不可运动：这些方法存在只是为了接口一致，全部拒绝
    def start(self): raise PermissionError("只读模式不能接管手臂（启动服务时加 --allow-real-motion）")
    def enable_jog(self): raise PermissionError("只读模式")
    def set_target(self, q): raise PermissionError("只读模式")
    def stop(self): pass
    def disable_jog(self): pass
    def shutdown(self):
        try:
            self._sub.Close()
        except Exception:  # noqa: BLE001
            pass
    def set_max_speed(self, v): pass
    def enter_hand_move(self): raise PermissionError("只读模式")
    def push(self, q): raise PermissionError("只读模式")
    def set_payload(self, mass, com, alpha=1.0): raise PermissionError("只读模式不能改前馈")
    def payload(self): return {"mass_kg": 0.0, "com_m": [0, 0, 0], "alpha": 1.0}

    def status(self) -> Dict:
        q = self._vals("q")
        return {"kind": self.kind, "arm": self.arm, "engaged": False, "jog_enabled": False, "weight": 0.0,
                "joint_names": self.joint_names, "measured_rad": q, "measured_dq_rad_s": self._vals("dq"),
                "tau_est_nm": self._vals("tau_est"), "cmd_rad": q, "desired_rad": q,
                "tau_grav_nm": None, "last_sent_tau_ff_nm": None, "kp_vec": self.kp_vec.tolist(),
                "kd_vec": self.kd_vec.tolist(), "max_speed_rad_s": 0.0, "limits_rad": self.limits.tolist(),
                "payload": self.payload(), "lowstate_ok": q is not None}
