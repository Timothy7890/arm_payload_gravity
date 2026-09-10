"""到位精度验证：同一组姿态，分别用不同前馈方案跑一遍，对比稳态误差与耗时。

方案（modes）：
  baseline         作者现状：α=1 的 URDF 臂自重前馈，不含负载
  payload          本项目：URDF 臂自重 × α + 辨识出的末端负载前馈
  payload_correct  payload 基础上再做 k 次"到位修正"：静止后量偏差 e，把指令角改成 cmd − e 再等静止
                   （对付前馈补不掉的那点摩擦/模型误差；每次修正多花 ~0.5 s，可选）

速度：三种方案走的是同一条五次多项式轨迹、同一限速，所以运动时间一致；
差别只在"静止后误差"与"静止用时"。结果存 data/validate/<session>.json。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gravity import ArmGravity
from motion import move_and_settle, wait_still
from runner import BaseRunner
from urdf_fk import matrix_to_rpy

HERE = Path(__file__).resolve().parent
VALIDATE_DIR = HERE / "data" / "validate"

MODES = ("baseline", "payload", "payload_correct")
NO_PAYLOAD = {"mass_kg": 0.0, "com_m": [0.0, 0.0, 0.0], "alpha": 1.0}


def ee_error(grav: ArmGravity, q_final, q_target) -> Dict:
    Tf, Tt = grav.tip_T(q_final), grav.tip_T(q_target)
    dp = Tf[:3, 3] - Tt[:3, 3]
    dR = Tt[:3, :3].T @ Tf[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    return {"pos_mm": float(np.linalg.norm(dp) * 1000), "pos_vec_mm": (dp * 1000).tolist(), "rot_deg": ang}


def apply_payload(ctrl, p: Dict) -> None:
    ctrl.set_payload(float(p.get("mass_kg", 0.0)), p.get("com_m", [0, 0, 0]), float(p.get("alpha", 1.0)))


def goto_with_correction(ctrl, grav: ArmGravity, target, motion: Dict, iters: int, stop_evt, on_phase=None,
                         gain: float = 1.0) -> Dict:
    """走到 target，静止后可选做 iters 次指令偏移修正。返回完整指标。"""
    target = np.asarray(target, float)
    t0 = time.monotonic()
    r = move_and_settle(ctrl, target, motion, stop_evt, on_phase)
    steps = [{"iter": 0, "err_max_deg": r["err_max_deg"], "err_rms_deg": r["err_rms_deg"], "t_s": r["total_s"]}]
    q_final = np.asarray(r["q_final"], float)
    lo, hi = ctrl.limits[:, 0], ctrl.limits[:, 1]
    for k in range(1, int(iters) + 1):
        if on_phase:
            on_phase(f"correct {k}/{iters}")
        st = ctrl.status()
        cmd = np.asarray(st["cmd_rad"], float)
        e = q_final - target
        new_cmd = np.clip(cmd - gain * e, lo, hi)
        if not ctrl.set_target(new_cmd):
            raise RuntimeError("set_target 被拒")
        # 偏移很小（几度以内），限速层一两帧就追上；等静止
        time.sleep(0.1)
        still = wait_still(ctrl, {**motion, "still_window_s": 0.3}, stop_evt)
        q_final = still["q"]
        e2 = q_final - target
        steps.append({"iter": k, "err_max_deg": float(np.max(np.abs(e2)) * 180 / math.pi),
                      "err_rms_deg": float(np.sqrt(np.mean(e2 ** 2)) * 180 / math.pi), "t_s": time.monotonic() - t0})
    e = q_final - target
    out = dict(r)
    out.update({"q_final": q_final.tolist(), "err_rad": e.tolist(),
                "err_max_deg": float(np.max(np.abs(e)) * 180 / math.pi),
                "err_rms_deg": float(np.sqrt(np.mean(e ** 2)) * 180 / math.pi),
                "total_s": time.monotonic() - t0, "corrections": steps, "ee": ee_error(grav, q_final, target)})
    return out


class ValidateRunner(BaseRunner):
    name = "validate"

    def __init__(self, ctrl, arm: str, poses_cfg: Dict, opts: Dict):
        super().__init__()
        self.ctrl = ctrl
        self.arm = arm
        self.grav = ArmGravity(arm)
        self.poses_cfg = poses_cfg
        self.opts = {"modes": list(MODES), "payload": None, "correction_iters": 2, "pose_indices": None,
                     "return_home": True, **(opts or {})}
        self.motion = dict(poses_cfg.get("motion", {}))
        self.session = f"{time.strftime('%Y%m%d_%H%M%S')}_{arm}"
        self.state.update(session=self.session, rows=[], summary={})

    def run(self) -> Dict:
        poses = self.poses_cfg["poses"]
        idx = self.opts["pose_indices"] or list(range(len(poses)))
        modes = [m for m in self.opts["modes"] if m in MODES]
        payload = self.opts["payload"]
        if any(m.startswith("payload") for m in modes) and not payload:
            raise ValueError("要验证 payload 方案，需要先辨识/应用一个负载参数")
        original = self.ctrl.payload()
        rows: List[Dict] = []
        self.set(progress=[0, len(modes) * len(idx)])
        try:
            for m in modes:
                apply_payload(self.ctrl, NO_PAYLOAD if m == "baseline" else payload)
                self.log(f"方案 {m}：前馈 {self.ctrl.payload()}")
                self.phase(f"{m}: 回 home")
                move_and_settle(self.ctrl, self.poses_cfg["home_q"], self.motion, self.stop_evt)
                for i in idx:
                    p = poses[i]
                    iters = int(self.opts["correction_iters"]) if m == "payload_correct" else 0
                    r = goto_with_correction(self.ctrl, self.grav, p["q"], self.motion, iters, self.stop_evt,
                                             on_phase=lambda s, m=m, p=p: self.phase(f"{m} {p['name']} {s}"))
                    row = {"mode": m, "pose": p["name"], "index": i, "err_max_deg": r["err_max_deg"], "err_rms_deg": r["err_rms_deg"],
                           "ee_mm": r["ee"]["pos_mm"], "ee_rot_deg": r["ee"]["rot_deg"], "traj_s": r["traj_s"],
                           "settle_s": r["settle_s"], "total_s": r["total_s"], "timed_out": r["timed_out"],
                           "err_deg": (np.asarray(r["err_rad"]) * 180 / math.pi).round(3).tolist(),
                           "corrections": r.get("corrections")}
                    rows.append(row)
                    with self._lock:
                        self.state["rows"] = rows
                        self.state["progress"] = [len(rows), len(modes) * len(idx)]
                        self.state["summary"] = summarize(rows)
                    self.log(f"{m} {p['name']}: 最大关节误差 {r['err_max_deg']:.2f}°  末端 {r['ee']['pos_mm']:.1f} mm  总耗时 {r['total_s']:.1f}s")
                if self.opts["return_home"]:
                    move_and_settle(self.ctrl, self.poses_cfg["home_q"], self.motion, self.stop_evt)
        finally:
            apply_payload(self.ctrl, original)      # 恢复验证前的前馈设置
        result = {"session": self.session, "arm": self.arm, "controller": self.ctrl.status().get("kind"),
                  "poseset": self.poses_cfg.get("name"),
                  "payload": payload, "modes": modes, "rows": rows, "summary": summarize(rows),
                  "sim_truth": self.ctrl.status().get("sim_truth"), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        VALIDATE_DIR.mkdir(parents=True, exist_ok=True)
        (VALIDATE_DIR / f"{self.session}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        return result


def summarize(rows: List[Dict]) -> Dict:
    out = {}
    for m in MODES:
        rs = [r for r in rows if r["mode"] == m]
        if not rs:
            continue
        out[m] = {"n": len(rs),
                  "err_max_deg_mean": float(np.mean([r["err_max_deg"] for r in rs])),
                  "err_max_deg_worst": float(np.max([r["err_max_deg"] for r in rs])),
                  "err_rms_deg_mean": float(np.mean([r["err_rms_deg"] for r in rs])),
                  "ee_mm_mean": float(np.mean([r["ee_mm"] for r in rs])), "ee_mm_worst": float(np.max([r["ee_mm"] for r in rs])),
                  "traj_s_mean": float(np.mean([r["traj_s"] for r in rs])),
                  "settle_s_mean": float(np.mean([r["settle_s"] for r in rs])),
                  "total_s_mean": float(np.mean([r["total_s"] for r in rs]))}
    return out


def list_validations() -> List[Dict]:
    out = []
    if VALIDATE_DIR.exists():
        for p in sorted(VALIDATE_DIR.glob("*.json"), reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append({"name": p.stem, "arm": d.get("arm"), "controller": d.get("controller"), "summary": d.get("summary")})
            except Exception:  # noqa: BLE001
                continue
    return out


class GotoRunner(BaseRunner):
    """单点到位：给一个关节目标，走过去，报告误差（可选修正）。"""

    name = "goto"

    def __init__(self, ctrl, arm: str, target, motion: Dict, correction_iters: int = 0):
        super().__init__()
        self.ctrl = ctrl
        self.grav = ArmGravity(arm)
        self.target = np.asarray(target, float)
        self.motion = motion
        self.iters = int(correction_iters)

    def run(self) -> Dict:
        r = goto_with_correction(self.ctrl, self.grav, self.target, self.motion, self.iters, self.stop_evt, on_phase=self.phase)
        r["target"] = self.target.tolist()
        r["payload"] = self.ctrl.payload()
        r["err_deg"] = (np.asarray(r["err_rad"]) * 180 / math.pi).round(3).tolist()
        self.log(f"到位：最大关节误差 {r['err_max_deg']:.2f}°，末端 {r['ee']['pos_mm']:.1f} mm，耗时 {r['total_s']:.1f}s")
        return r
