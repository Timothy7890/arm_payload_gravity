"""末端负载辨识：多姿态静态测量 → 线性最小二乘解 [m, m·cx, m·cy, m·cz]（可选臂自重比例 α）。

原理
----
手臂静止时，电机实际输出的力矩恰好平衡了重力（加一点摩擦）：

    τ_true(q) = α · τ_arm(q) + Y(q) · θ          θ = [m, m·cx, m·cy, m·cz]

τ_true 有两个来源（都记下来，求解时选）：
  - "pd" : 按底层 PD 律重构  τ = kp·(q_cmd − q) − kd·dq + τ_ff_sent  —— 这正是造成下垂的那份力矩，
           用它辨识出的负载，补进前馈后消掉的就是下垂本身（推荐）；
  - "est": rt/lowstate 的 tau_est（电流估算，不是力传感器）。
τ_arm、Y 都用**实测角**算（真实姿态下的真实力矩）。

摩擦：静止时静摩擦会承担一部分重力，方向取决于最后是从哪边停下来的。
"双向逼近"对每个姿态各从 +δ、−δ 两侧停一次取两样本，摩擦项符号相反，最小二乘里自然抵消。

用法
----
  网页里点"开始测量"→"求解"→"应用到前馈"；或离线：
  python identify.py solve data/measure/<session> [--source est] [--fit-alpha]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gravity import ArmGravity, theta_to_payload
from motion import average_status, move_and_settle
from runner import BaseRunner

HERE = Path(__file__).resolve().parent
MEASURE_DIR = HERE / "data" / "measure"
CONFIG = HERE / "config"


def payload_config_path(arm: str) -> Path:
    return CONFIG / f"payload_{arm}.json"


def load_payload_config(arm: str) -> Optional[Dict]:
    p = payload_config_path(arm)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_payload_config(arm: str, payload: Dict) -> Path:
    CONFIG.mkdir(parents=True, exist_ok=True)
    p = payload_config_path(arm)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 测量
class MeasureRunner(BaseRunner):
    name = "measure"

    def __init__(self, ctrl, arm: str, poses_cfg: Dict, opts: Dict):
        super().__init__()
        self.ctrl = ctrl
        self.arm = arm
        self.poses_cfg = poses_cfg
        self.opts = {"bidirectional": True, "approach_delta_rad": 0.06, "dwell_s": 0.6, "avg_frames": 25,
                     "return_home": True, **(opts or {})}
        self.motion = dict(poses_cfg.get("motion", {}))
        self.session_dir = MEASURE_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{arm}"
        self.samples: List[Dict] = []
        self.state.update(session=self.session_dir.name, samples=[], current=-1)

    def _save(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "samples.json").write_text(json.dumps(self.samples, indent=1), encoding="utf-8")
        st = self.ctrl.status()
        meta = {"arm": self.arm, "controller": st.get("kind"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "poseset": self.poses_cfg.get("name"), "opts": self.opts,
                "payload_during_measure": st.get("payload"), "sim_truth": st.get("sim_truth")}
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def _take_sample(self, idx: int, pose: Dict, approach: str) -> Dict:
        q_t = np.asarray(pose["q"], float)
        r = move_and_settle(self.ctrl, q_t, self.motion, self.stop_evt, on_phase=lambda s: self.phase(f"{pose['name']}{approach} {s}"))
        time.sleep(self.opts["dwell_s"])
        avg = average_status(self.ctrl, int(self.opts["avg_frames"]), stop_evt=self.stop_evt)
        s = {"index": idx, "name": pose["name"], "approach": approach, "q_target": q_t.tolist(),
             "q_cmd": avg["cmd_rad"], "q_meas": avg["measured_rad"], "dq": avg["measured_dq_rad_s"],
             "tau_est": avg["tau_est_nm"], "tau_ff_sent": avg["last_sent_tau_ff_nm"], "tau_grav_sent": avg["tau_grav_nm"],
             "kp": avg["kp_vec"], "kd": avg["kd_vec"], "settle_s": r["settle_s"], "timed_out": r["timed_out"],
             "err_max_deg": r["err_max_deg"], "t": time.time()}
        s["tau_pd"] = tau_from_pd(s)
        return s

    def run(self) -> Dict:
        poses = self.poses_cfg["poses"]
        delta = float(self.opts["approach_delta_rad"])
        lo, hi = self.ctrl.limits[:, 0], self.ctrl.limits[:, 1]
        total = len(poses) * (2 if self.opts["bidirectional"] else 1)
        self.set(progress=[0, total])
        self.log(f"开始测量 {len(poses)} 个姿态，{'双向逼近' if self.opts['bidirectional'] else '单向'}，会话 {self.session_dir.name}")
        for i, pose in enumerate(poses):
            self.set(current=i)
            q_t = np.asarray(pose["q"], float)
            approaches = [("+", +1.0), ("-", -1.0)] if self.opts["bidirectional"] else [("", 0.0)]
            for tag, sgn in approaches:
                if sgn != 0.0:
                    pre = np.clip(q_t + sgn * delta, lo, hi)
                    move_and_settle(self.ctrl, pre, {**self.motion, "still_window_s": 0.2}, self.stop_evt,
                                    on_phase=lambda s: self.phase(f"{pose['name']} 预位{tag} {s}"))
                s = self._take_sample(i, pose, tag)
                self.samples.append(s)
                self._save()
                with self._lock:
                    self.state["samples"] = [sample_brief(x) for x in self.samples]
                    self.state["progress"] = [len(self.samples), total]
                self.log(f"{pose['name']}{tag}: 静止 {s['settle_s']:.2f}s，指令-实测最大差 {s['err_max_deg']:.2f}°")
        if self.opts["return_home"]:
            self.phase("回 home")
            move_and_settle(self.ctrl, self.poses_cfg["home_q"], self.motion, self.stop_evt)
        return {"session": self.session_dir.name, "n_samples": len(self.samples)}


def sample_brief(s: Dict) -> Dict:
    return {"index": s["index"], "name": s["name"], "approach": s["approach"], "err_max_deg": round(s["err_max_deg"], 3),
            "settle_s": round(s["settle_s"], 2), "timed_out": s["timed_out"]}


def tau_from_pd(s: Dict) -> Optional[List[float]]:
    if s.get("tau_ff_sent") is None or s.get("q_cmd") is None:
        return None
    kp, kd = np.asarray(s["kp"], float), np.asarray(s["kd"], float)
    dq = np.asarray(s["dq"], float) if s.get("dq") is not None else np.zeros_like(kp)
    tau = kp * (np.asarray(s["q_cmd"]) - np.asarray(s["q_meas"])) - kd * dq + np.asarray(s["tau_ff_sent"])
    return tau.tolist()


# ---------------------------------------------------------------- 求解
def solve_payload(samples: List[Dict], arm: str, source: str = "pd", fit_alpha: bool = False,
                  joint_mask: Optional[List[bool]] = None, reweight: bool = True) -> Dict:
    grav = ArmGravity(arm)
    rows_A, rows_b, rows_arm, tags = [], [], [], []
    mask = np.asarray(joint_mask if joint_mask is not None else [True] * 7, bool)
    used = 0
    for s in samples:
        tau_true = s.get("tau_pd") if source == "pd" else s.get("tau_est")
        if tau_true is None:
            continue
        q = np.asarray(s["q_meas"], float)
        Y = grav.payload_regressor(q)
        ta = grav.tau_arm(q)
        A = np.hstack([Y, ta[:, None]]) if fit_alpha else Y
        rows_A.append(A[mask]); rows_b.append(np.asarray(tau_true, float)[mask]); rows_arm.append(ta[mask])
        tags.append(f"{s['name']}{s.get('approach', '')}")
        used += 1
    if used < 4:
        raise ValueError(f"可用样本只有 {used} 个（需要 ≥4，建议 ≥8）")
    A = np.vstack(rows_A); b = np.concatenate(rows_b); ta = np.concatenate(rows_arm)
    nj = int(mask.sum())
    joint_idx = np.tile(np.arange(7)[mask], used)
    if not fit_alpha:
        b_fit = b - ta          # α=1：残差 = 真实力矩 − URDF 臂自重
    else:
        b_fit = b

    def wlsq(w):
        Aw, bw = A * w[:, None], b_fit * w
        x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        return x

    w = np.ones_like(b_fit)
    x = wlsq(w)
    if reweight:
        # 一次 IRLS：按关节残差标准差加权（腕关节力矩小、噪声相对大）
        res = b_fit - A @ x
        sig = np.array([max(np.std(res[joint_idx == j]), 0.02) if np.any(joint_idx == j) else 1.0 for j in range(7)])
        w = 1.0 / sig[joint_idx]
        x = wlsq(w)
    theta, alpha = (x[:4], float(x[4])) if fit_alpha else (x, 1.0)
    mass, com = theta_to_payload(theta)
    res_before = b - ta                       # 只用 URDF 前馈时的剩余力矩
    res_after = b_fit - A @ x
    # 参数不确定度（加权残差协方差）
    try:
        Aw = A * w[:, None]
        dof = max(1, Aw.shape[0] - Aw.shape[1])
        s2 = float(np.sum((res_after * w) ** 2)) / dof
        cov = s2 * np.linalg.inv(Aw.T @ Aw)
        std = np.sqrt(np.diag(cov))
        com_std = (np.sqrt(std[1:4] ** 2 + (com * std[0]) ** 2) / max(abs(mass), 1e-6)).tolist() if mass > 1e-6 else None
    except np.linalg.LinAlgError:
        std, com_std = np.full(x.size, np.nan), None
    scale = np.linalg.norm(A, axis=0) + 1e-9
    per_joint = lambda r: [float(np.sqrt(np.mean(r[joint_idx == j] ** 2))) if np.any(joint_idx == j) else None for j in range(7)]
    per_sample = [float(np.sqrt(np.mean(res_after[k * nj:(k + 1) * nj] ** 2))) for k in range(used)]
    # 换算：残差力矩 / kp ≈ 若只用此前馈会剩下的下垂角
    kp = np.asarray(samples[0]["kp"], float)
    sag_before = np.abs(res_before) / kp[joint_idx]
    sag_after = np.abs(res_after) / kp[joint_idx]
    return {
        "arm": arm, "source": source, "fit_alpha": fit_alpha, "n_samples": used, "joint_mask": mask.tolist(),
        "mass_kg": mass, "com_m": com.tolist(), "alpha": alpha, "theta": theta.tolist(),
        "mass_std_kg": float(std[0]), "com_std_m": com_std, "alpha_std": float(std[4]) if fit_alpha else None,
        "regressor_cond": float(np.linalg.cond(A / scale)),
        "residual_rms_before_nm": float(np.sqrt(np.mean(res_before ** 2))),
        "residual_rms_after_nm": float(np.sqrt(np.mean(res_after ** 2))),
        "residual_per_joint_before_nm": per_joint(res_before), "residual_per_joint_after_nm": per_joint(res_after),
        "sag_est_before_deg": float(np.mean(sag_before) * 180 / math.pi), "sag_est_after_deg": float(np.mean(sag_after) * 180 / math.pi),
        "per_sample_rms_nm": dict(zip(tags, per_sample)),
        "solved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def solve_session(session_dir: Path, **kw) -> Dict:
    samples = json.loads((session_dir / "samples.json").read_text(encoding="utf-8"))
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    for s in samples:
        if s.get("tau_pd") is None:
            s["tau_pd"] = tau_from_pd(s)
    r = solve_payload(samples, meta["arm"], **kw)
    r["session"] = session_dir.name
    if meta.get("sim_truth"):
        t = meta["sim_truth"]
        r["sim_truth"] = {"mass_kg": t["payload_mass"], "com_m": t["payload_com"], "arm_mass_scale": t["arm_mass_scale"]}
        r["sim_error"] = {"mass_kg": r["mass_kg"] - t["payload_mass"],
                          "com_mm": (1000 * (np.asarray(r["com_m"]) - np.asarray(t["payload_com"]))).tolist()}
    (session_dir / f"payload_{kw.get('source', 'pd')}{'_alpha' if kw.get('fit_alpha') else ''}.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    return r


def list_sessions() -> List[Dict]:
    out = []
    if MEASURE_DIR.exists():
        for d in sorted(MEASURE_DIR.iterdir(), reverse=True):
            if (d / "samples.json").exists():
                try:
                    n = len(json.loads((d / "samples.json").read_text()))
                    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
                except Exception:  # noqa: BLE001
                    continue
                out.append({"name": d.name, "n_samples": n, "arm": meta.get("arm"), "controller": meta.get("controller"),
                            "poseset": meta.get("poseset"),
                            "solved": sorted(p.name for p in d.glob("payload_*.json"))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve"); s.add_argument("session"); s.add_argument("--source", default="pd", choices=["pd", "est"])
    s.add_argument("--fit-alpha", action="store_true")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "list":
        print(json.dumps(list_sessions(), indent=2, ensure_ascii=False)); return 0
    d = Path(a.session)
    if not d.is_absolute() and not d.exists():
        d = MEASURE_DIR / a.session
    r = solve_session(d, source=a.source, fit_alpha=a.fit_alpha)
    print(json.dumps({k: v for k, v in r.items() if k != "per_sample_rms_nm"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
