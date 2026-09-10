"""走点 + 静止判据（测量、验证、单点到位共用）。

move_and_settle(ctrl, target, ...)：五次多项式插值逐点喂 set_target（与作者 calibration_replay 相同参数），
轨迹结束后等"实测角静止"（不是等"到达目标"——基线模式因为下垂根本到不了目标，
但它一样会静止，静止后的偏差正是我们要量的东西）。返回耗时与误差指标。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Optional

import numpy as np

CONTROL_DT = 0.02
QUINTIC_VELOCITY_PEAK = 1.875
QUINTIC_ACCELERATION_PEAK = 10.0 / math.sqrt(3.0)

DEFAULT_MOTION = {"vmax_rad_s": 0.25, "amax_rad_s2": 0.5, "min_segment_s": 1.0,
                  "still_dq_rad_s": 0.01, "still_window_s": 0.4, "still_drift_rad": 0.002, "settle_timeout_s": 8.0}


def quintic_scale(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5


def segment_duration(start, end, vmax: float, amax: float, min_duration: float) -> float:
    delta = float(np.max(np.abs(np.asarray(end, float) - np.asarray(start, float))))
    return max(min_duration, QUINTIC_VELOCITY_PEAK * delta / vmax, math.sqrt(QUINTIC_ACCELERATION_PEAK * delta / amax))


class Stopped(Exception):
    pass


def _check(stop_evt: Optional[threading.Event]) -> None:
    if stop_evt is not None and stop_evt.is_set():
        raise Stopped()


def wait_still(ctrl, params: Dict, stop_evt: Optional[threading.Event] = None) -> Dict:
    """等实测角静止：窗口内 |dq| 都小于阈值且位置漂移小于阈值。返回 {settle_s, timed_out, q, dq}。"""
    p = {**DEFAULT_MOTION, **(params or {})}
    t0 = time.monotonic()
    deadline = t0 + p["settle_timeout_s"]
    hist = []
    drift, max_dq = float("nan"), float("nan")
    while True:
        _check(stop_evt)
        st = ctrl.status()
        q = np.asarray(st["measured_rad"], float)
        dq = st.get("measured_dq_rad_s")
        now = time.monotonic()
        hist.append((now, q, None if dq is None else np.asarray(dq, float)))
        hist = [h for h in hist if now - h[0] <= p["still_window_s"]]
        if now - hist[0][0] >= p["still_window_s"] * 0.95 and len(hist) >= 3:
            qs = np.stack([h[1] for h in hist])
            drift = float(np.max(np.ptp(qs, axis=0)))
            dqs = [h[2] for h in hist if h[2] is not None]
            max_dq = float(np.max(np.abs(np.stack(dqs)))) if dqs else 0.0
            if drift < p["still_drift_rad"] and max_dq < p["still_dq_rad_s"]:
                return {"settle_s": now - t0, "timed_out": False, "q": q, "dq": hist[-1][2], "drift_rad": drift, "max_dq_rad_s": max_dq}
        if now >= deadline:
            return {"settle_s": now - t0, "timed_out": True, "q": q, "dq": hist[-1][2], "drift_rad": drift, "max_dq_rad_s": max_dq}
        time.sleep(0.02)


def move_and_settle(ctrl, target, params: Optional[Dict] = None, stop_evt: Optional[threading.Event] = None,
                    on_phase=None) -> Dict:
    """从当前指令角五次多项式走到 target，再等静止。返回耗时/误差指标。"""
    p = {**DEFAULT_MOTION, **(params or {})}
    target = np.asarray(target, float).reshape(-1)
    start = np.asarray(ctrl.status()["cmd_rad"], float)
    dur = segment_duration(start, target, p["vmax_rad_s"], p["amax_rad_s2"], p["min_segment_s"])
    steps = max(1, int(math.ceil(dur / CONTROL_DT)))
    if on_phase:
        on_phase(f"moving ({dur:.1f}s)")
    t0 = time.monotonic()
    for k in range(1, steps + 1):
        _check(stop_evt)
        s = quintic_scale(k / steps)
        if not ctrl.set_target(start + (target - start) * s):
            raise RuntimeError("控制器未处于点动模式（set_target 被拒）")
        time.sleep(max(0.0, t0 + k * CONTROL_DT - time.monotonic()))
    t_traj = time.monotonic() - t0
    # 等控制器内部限速层追上（cmd == desired）
    t1 = time.monotonic()
    while time.monotonic() - t1 < 5.0:
        _check(stop_evt)
        st = ctrl.status()
        if float(np.max(np.abs(np.asarray(st["cmd_rad"]) - np.asarray(st["desired_rad"])))) < 1e-6:
            break
        time.sleep(0.02)
    if on_phase:
        on_phase("settling")
    still = wait_still(ctrl, p, stop_evt)
    err = still["q"] - target
    return {"segment_s": dur, "traj_s": t_traj, "settle_s": still["settle_s"], "total_s": time.monotonic() - t0,
            "timed_out": still["timed_out"], "q_final": still["q"].tolist(), "err_rad": err.tolist(),
            "err_max_deg": float(np.max(np.abs(err)) * 180 / math.pi),
            "err_rms_deg": float(np.sqrt(np.mean(err ** 2)) * 180 / math.pi)}


def average_status(ctrl, n: int = 25, dt: float = 0.02, stop_evt: Optional[threading.Event] = None) -> Dict:
    """静止后连采 n 帧 status 取均值（q / dq / tau_est / cmd / tau_ff）。"""
    acc: Dict[str, list] = {k: [] for k in ("measured_rad", "measured_dq_rad_s", "tau_est_nm", "cmd_rad", "last_sent_tau_ff_nm", "tau_grav_nm")}
    for _ in range(n):
        _check(stop_evt)
        st = ctrl.status()
        for k in acc:
            v = st.get(k)
            if v is not None:
                acc[k].append(np.asarray(v, float))
        time.sleep(dt)
    out = {k: (np.mean(np.stack(v), axis=0).tolist() if v else None) for k, v in acc.items()}
    st = ctrl.status()
    out["kp_vec"] = st["kp_vec"]; out["kd_vec"] = st["kd_vec"]
    out["n_frames"] = n
    return out
