"""末端负载辨识 + 重力前馈补偿 + 到位验证：网页服务（端口 10183）。

    conda activate abyss
    python server.py                       # 默认仿真模式，网页里可切真机
    python server.py --allow-real-motion   # ⚠ 允许真机接管手臂（rt/arm_sdk）；不加则真机只读

模式：
  sim   仿真控制器（含真实负载/摩擦/质量误差的物理层），完整跑通测量→求解→验证
  real  未加 --allow-real-motion：只读 rt/lowstate 显示；加了：网页"接管"后才发令

所有真机发令均经作者 H2ArmController（子类 PayloadArmController 只替换重力模型），见 controllers.py。
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from controllers import (ArmSdkMonitor, ReadOnlyH2Arm, SimArmController, SimArmSdkMonitor, SimTruth,
                         make_payload_arm_controller)
from gravity import ArmGravity
from identify import (MeasureRunner, MEASURE_DIR, list_sessions, load_payload_config, save_payload_config,
                      solve_session)
from motion import average_status, wait_still
from poses import (add_pose, create_set, delete_pose, delete_set, generate, list_sets, load_set, save_set,
                   update_set)
from validate import GotoRunner, VALIDATE_DIR, ValidateRunner, list_validations

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"

# ---------------------------------------------------------------- 全局状态
lock = threading.RLock()
mode = "sim"
arm = "right"
ctrl = None
mode_error: Optional[str] = None
allow_real_motion = False
dds_iface: Optional[str] = None
max_speed = 0.15
sim_truth = SimTruth()
task = None                      # 当前后台任务（MeasureRunner / ValidateRunner / GotoRunner）
last_solve: Optional[Dict] = None
gravs = {a: ArmGravity(a) for a in ("left", "right")}
poseset_name: Dict[str, Optional[str]] = {"left": None, "right": None}   # 每条手臂当前选中的姿态集
monitor = None                   # rt/arm_sdk 占用检测（ArmSdkMonitor / SimArmSdkMonitor）
# 示教静止判据：真机保持态有编码器噪声/kp 微抖，比走点用的 DEFAULT_MOTION 宽；网页可改
TEACH_STILL_DEFAULT = {"still_dq_rad_s": 0.03, "still_window_s": 0.4, "still_drift_rad": 0.004, "settle_timeout_s": 3.0}
teach: Dict = {"active": False, "locked": False, "last_pose": None, "message": "", "still": dict(TEACH_STILL_DEFAULT),
               "last_still": None}


def set_teach_still(body: Dict) -> Dict:
    s = dict(teach["still"])
    for k in TEACH_STILL_DEFAULT:
        if k in body and body[k] is not None:
            v = float(body[k])
            if not (v > 0):
                raise ValueError(f"{k} 必须 > 0")
            s[k] = v
    with lock:
        teach["still"] = s
    return teach_state()


def _task_running() -> bool:
    return task is not None and task.running


def _require_idle() -> None:
    if _task_running():
        raise RuntimeError(f"任务 {task.name} 正在运行，先停止")
    if teach["active"]:
        raise RuntimeError("正在示教（卸力拖动），先点「结束示教」")


def _require_ctrl() -> None:
    if ctrl is None:
        raise RuntimeError("没有控制器")
    if not getattr(ctrl, "controllable", False):
        raise PermissionError("当前为只读模式，不能运动（启动服务加 --allow-real-motion，并在网页接管）")
    if not ctrl.status().get("jog_enabled"):
        raise RuntimeError("手臂未接管/未进入点动模式，先点「接管」")


# ---------------------------------------------------------------- 姿态集
def current_set() -> Dict:
    """当前手臂选中的姿态集；没选就取该臂最近修改的一个。"""
    name = poseset_name.get(arm)
    if name is None:
        sets = list_sets(arm)
        if not sets:
            raise FileNotFoundError(f"{arm} 臂还没有姿态集：新建一个后示教，或自动生成")
        name = sets[0]["name"]
        poseset_name[arm] = name
    return load_set(name)


def select_set(name: str) -> Dict:
    d = load_set(name)
    if d["arm"] != arm:
        raise ValueError(f"姿态集 {name} 是 {d['arm']} 臂的，当前是 {arm} 臂")
    poseset_name[arm] = name
    return d


def set_summary() -> Optional[Dict]:
    try:
        d = current_set()
    except FileNotFoundError:
        return None
    return {"name": d["name"], "arm": d["arm"], "note": d.get("note", ""), "n": len(d["poses"]), "cond": d.get("regressor_cond"),
            "home_q": d["home_q"], "motion": d.get("motion"),
            "poses": [{"name": x["name"], "q": x["q"], "tip_xyz": x.get("tip_xyz"), "source": x.get("source"),
                       "recorded_at": x.get("recorded_at")} for x in d["poses"]]}


# ---------------------------------------------------------------- 示教（手推记姿态）
def teach_start() -> Dict:
    with lock:
        if _task_running():
            raise RuntimeError(f"任务 {task.name} 正在运行，先停止")
        if ctrl is None or not getattr(ctrl, "controllable", False):
            raise PermissionError("只读模式不能示教")
        if not ctrl.status().get("engaged"):
            raise RuntimeError("先「接管」再示教")
        current_set()                                  # 没有姿态集就报错，别白推
        ctrl.disable_jog()
        if not ctrl.enter_hand_move():
            raise RuntimeError("进入卸力模式失败")
        teach.update(active=True, locked=False, message="卸力中：手推到位后按空格锁定并记录")
        return teach_state()


def teach_lock(pose_name: Optional[str] = None) -> Dict:
    """空格：锁定当前姿态 → 等静止 → 均值读取 → 追加进当前姿态集。"""
    with lock:
        if not teach["active"]:
            raise RuntimeError("未在示教")
        if teach["locked"]:
            raise RuntimeError("已锁定，按空格继续拖动")
        ctrl.stop()                                    # 从实测角抓取，刚性保持
        teach.update(locked=True, message="已锁定，等待静止…")
    still = wait_still(ctrl, teach["still"])
    avg = average_status(ctrl, 15)
    r = add_pose(current_set()["name"], avg["measured_rad"], pose_name, source="teach")
    info = {"timed_out": still["timed_out"], "settle_s": round(still["settle_s"], 2),
            "max_dq_rad_s": still.get("max_dq_rad_s"), "drift_rad": still.get("drift_rad")}
    warn = (f"（{still['settle_s']:.1f}s 内未完全静止：max|dq|={info['max_dq_rad_s']:.4f} 阈 {teach['still']['still_dq_rad_s']}，"
            f"漂移={info['drift_rad']:.4f} 阈 {teach['still']['still_drift_rad']}；仍已按均值记录）") if still["timed_out"] else ""
    with lock:
        teach.update(last_pose=r["pose"], last_still=info, message=f"已记录 {r['pose']['name']}{warn}；按空格继续拖动")
    return {**teach_state(), "pose": r["pose"], "set": set_summary()}


def teach_float() -> Dict:
    with lock:
        if not teach["active"]:
            raise RuntimeError("未在示教")
        if not teach["locked"]:
            return teach_state()
        if not ctrl.enter_hand_move():
            raise RuntimeError("回到卸力模式失败")
        teach.update(locked=False, message="卸力中：手推到位后按空格锁定并记录")
        return teach_state()


def teach_toggle(pose_name: Optional[str] = None) -> Dict:
    return teach_float() if teach["locked"] else teach_lock(pose_name)


def teach_end() -> Dict:
    with lock:
        if ctrl is not None and getattr(ctrl, "controllable", False):
            ctrl.stop()                                # 刚性保持在当前位置
            ctrl.enable_jog()                          # 回到可走点状态
        teach.update(active=False, locked=False, message="")
        return teach_state()


def teach_push(q) -> Dict:
    if mode != "sim":
        raise PermissionError("真机没有'模拟手推'")
    if not teach["active"] or teach["locked"]:
        raise RuntimeError("只有示教卸力状态下才能模拟手推")
    ctrl.push(np.asarray(q, float))
    return teach_state()


def teach_state() -> Dict:
    return dict(teach)


def build_ctrl(new_mode: str, new_arm: str):
    """返回 (控制器, 占用监视器)。"""
    if new_mode == "sim":
        return SimArmController(new_arm, max_speed_rad_s=max_speed, truth=sim_truth), SimArmSdkMonitor(sim_truth)
    if new_mode == "real":
        mon = ArmSdkMonitor(dds_iface)
        if allow_real_motion:
            return make_payload_arm_controller(new_arm, dds_iface, max_speed), mon
        return ReadOnlyH2Arm(new_arm, dds_iface), mon
    raise ValueError("mode 必须是 sim/real")


def arm_sdk_report() -> Dict:
    we_publish = bool(ctrl is not None and ctrl.status().get("engaged"))
    if monitor is None:
        return {"rate_hz": 0.0, "foreign_rate_hz": 0.0, "foreign": False, "local_hints": []}
    return monitor.report(we_publish)


def switch_mode(new_mode: str, new_arm: str) -> Dict:
    """先构造新控制器，成功再替换；失败抛错并留在原模式。"""
    global mode, arm, ctrl, mode_error, last_solve, monitor
    if new_arm not in ("left", "right"):
        raise ValueError("arm 必须是 left/right")
    with lock:
        _require_idle()
        if ctrl is not None and mode == new_mode and arm == new_arm:
            return status_payload()
        try:
            new_ctrl, new_mon = build_ctrl(new_mode, new_arm)
        except Exception as exc:  # noqa: BLE001
            mode_error = f"切换到 {new_mode}/{new_arm} 失败：{exc}"
            print(f"[mode] {mode_error}")
            raise RuntimeError(mode_error) from exc
        old, old_mon = ctrl, monitor
        ctrl, monitor, mode, arm, mode_error = new_ctrl, new_mon, new_mode, new_arm, None
        last_solve = None
        for o in (old, old_mon):
            if o is not None:
                try:
                    o.shutdown() if hasattr(o, "shutdown") else o.close()   # 真机：权重渐出交还本体
                except Exception:  # noqa: BLE001
                    pass
        print(f"[mode] {mode} / {arm} / {ctrl.kind}")
        return status_payload()


def engage() -> Dict:
    with lock:
        if ctrl is None:
            raise RuntimeError("没有控制器")
        if not getattr(ctrl, "controllable", False):
            raise PermissionError("只读模式不能接管")
        if not ctrl.status().get("engaged"):
            # 接管前先听 1 s：rt/arm_sdk 上已经有人在发令就拒绝，避免两个发布者打架
            rep = arm_sdk_report()
            if rep["foreign"]:
                hints = "；本机疑似进程：" + "、".join(f"pid {h['pid']} {h['cmd'][:60]}" for h in rep["local_hints"]) if rep["local_hints"] else ""
                raise RuntimeError(f"检测到其他程序正在向 rt/arm_sdk 发令（约 {rep['rate_hz']:.0f} Hz，权重 {rep.get('last_weight')}，"
                                   f"kp {rep.get('last_kp')}），拒绝接管。请先停掉对方{hints}")
            ctrl.start()
        ctrl.enable_jog()
        # 接管后自动套用已保存的负载参数？——不。保持"作者现状"，由用户显式点「应用」
        return status_payload()


def release() -> Dict:
    global ctrl
    with lock:
        if _task_running():
            raise RuntimeError(f"任务 {task.name} 正在运行，先停止")
        if teach["active"]:
            teach_end()
        if ctrl is not None and getattr(ctrl, "controllable", False):
            ctrl.disable_jog()
            ctrl.shutdown()
            if mode == "real":
                # 作者的 H2ArmController 是一次性的：shutdown 后线程不能再 start，且 _engaged 仍为 True。
                # 释放后重建一个新实例（沿用当前负载前馈），否则再点接管只会 enable_jog 而没人发令。
                payload = ctrl.payload()
                new_ctrl = make_payload_arm_controller(arm, dds_iface, max_speed)
                try:
                    new_ctrl.set_payload(float(payload.get("mass_kg", 0.0)), payload.get("com_m", [0, 0, 0]),
                                         float(payload.get("alpha", 1.0)))
                except Exception as exc:  # noqa: BLE001
                    print(f"[release] 重建控制器后恢复负载失败：{exc}")
                ctrl = new_ctrl
            if monitor is not None and hasattr(monitor, "mark"):
                monitor.mark()          # 之前 1 s 里都是我们自己发的，别把它当成"别人占用"
        return status_payload()


def hold() -> Dict:
    """急停：停任务/示教，手臂冻结在当前位置（仍保持接管/点动）。"""
    if task is not None:
        task.stop()
    if teach["active"]:
        return {**teach_end(), **status_payload()}
    if ctrl is not None and getattr(ctrl, "controllable", False) and ctrl.status().get("engaged"):
        ctrl.stop()
        ctrl.enable_jog()
    return status_payload()


# ---------------------------------------------------------------- 任务
def start_task(make) -> Dict:
    global task
    with lock:
        _require_idle()
        _require_ctrl()
        task = make()
        task.start()
        return task.snapshot()


def start_measure(body: Dict) -> Dict:
    cfg = current_set()
    if len(cfg["poses"]) < 4:
        raise ValueError(f"姿态集 {cfg['name']} 只有 {len(cfg['poses'])} 个姿态，辨识至少要 4 个（建议 ≥8）")
    if body.get("pose_indices"):
        cfg["poses"] = [cfg["poses"][int(i)] for i in body["pose_indices"]]
    if body.get("motion"):
        cfg["motion"] = {**cfg.get("motion", {}), **body["motion"]}
    opts = {k: body[k] for k in ("bidirectional", "approach_delta_rad", "dwell_s", "avg_frames", "return_home") if k in body}
    return start_task(lambda: MeasureRunner(ctrl, arm, cfg, opts))


def start_validate(body: Dict) -> Dict:
    cfg = current_set()
    if not cfg["poses"]:
        raise ValueError(f"姿态集 {cfg['name']} 是空的")
    if body.get("motion"):
        cfg["motion"] = {**cfg.get("motion", {}), **body["motion"]}
    payload = body.get("payload") or (ctrl.payload() if ctrl.payload().get("mass_kg", 0) > 0 else None) or load_payload_config(arm)
    opts = {"modes": body.get("modes") or ["baseline", "payload", "payload_correct"],
            "payload": payload, "correction_iters": int(body.get("correction_iters", 2)),
            "pose_indices": body.get("pose_indices") or None, "return_home": bool(body.get("return_home", True))}
    return start_task(lambda: ValidateRunner(ctrl, arm, cfg, opts))


def start_goto(body: Dict) -> Dict:
    q = np.asarray(body.get("q"), float).reshape(-1)
    if q.size != 7 or not np.all(np.isfinite(q)):
        raise ValueError("q 需要 7 个有限数")
    try:
        cfg = current_set()
    except FileNotFoundError:
        cfg = {"motion": {}}
    motion = {**cfg.get("motion", {}), **(body.get("motion") or {})}
    return start_task(lambda: GotoRunner(ctrl, arm, q, motion, int(body.get("correction_iters", 0))))


def do_solve(body: Dict) -> Dict:
    global last_solve
    name = body.get("session")
    if not name:
        if task is not None and task.name == "measure":
            name = task.snapshot().get("session")
    if not name:
        raise ValueError("缺少 session")
    d = MEASURE_DIR / name
    if not d.exists():
        raise FileNotFoundError(f"没有会话 {name}")
    r = solve_session(d, source=body.get("source", "pd"), fit_alpha=bool(body.get("fit_alpha", True)))
    last_solve = r
    return r


def apply_payload(body: Dict) -> Dict:
    with lock:
        if ctrl is None or not getattr(ctrl, "controllable", False):
            raise PermissionError("只读模式不能改前馈")
        mass = float(body.get("mass_kg", 0.0)); com = [float(v) for v in body.get("com_m", [0, 0, 0])]
        alpha = float(body.get("alpha", 1.0))
        if not (0.0 <= mass <= 5.0):
            raise ValueError("质量需在 0～5 kg")
        if any(abs(v) > 0.5 for v in com):
            raise ValueError("质心分量需在 ±0.5 m 内")
        if not (0.5 <= alpha <= 1.2):
            raise ValueError("alpha 需在 0.5～1.2")
        ctrl.set_payload(mass, com, alpha)
        p = {"mass_kg": mass, "com_m": com, "alpha": alpha, "arm": arm, "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "source_session": body.get("source_session")}
        if body.get("save", True):
            save_payload_config(arm, p)
        return {"payload": ctrl.payload(), "saved": bool(body.get("save", True))}


# ---------------------------------------------------------------- 状态 / 场景
def status_payload() -> Dict:
    st = ctrl.status() if ctrl is not None else None
    return {"mode": mode, "arm": arm, "mode_error": mode_error, "allow_real_motion": allow_real_motion,
            "controllable": bool(getattr(ctrl, "controllable", False)), "ctrl": st,
            "task": task.snapshot() if task is not None else None, "teach": teach_state(),
            "arm_sdk": arm_sdk_report(),
            "payload_saved": load_payload_config(arm), "last_solve": last_solve,
            "poses": set_summary(), "posesets": list_sets(arm),
            "sim_truth": sim_truth.as_dict() if mode == "sim" else None,
            "sessions": list_sessions(), "validations": list_validations()[:20]}


def scene() -> Dict:
    g = gravs[arm]
    st = ctrl.status() if ctrl is not None else None
    out: Dict = {"arm": arm, "kind": st.get("kind") if st else None}
    if st and st.get("measured_rad"):
        q = np.asarray(st["measured_rad"], float)
        out["links"] = [[0, 0, 0]] + [T[:3, 3].tolist() for _, T in g.model.fk_chain(q)]
        out["wrist_T"] = g.tip_T(q).tolist()
        pl = st.get("payload") or {}
        if pl.get("mass_kg", 0) > 0:
            out["payload_point"] = g.payload_point(q, pl["com_m"]).tolist(); out["payload_mass"] = pl["mass_kg"]
        tr = st.get("sim_truth")
        if tr and tr.get("payload_mass", 0) > 0:
            out["truth_point"] = g.payload_point(q, tr["payload_com"]).tolist(); out["truth_mass"] = tr["payload_mass"]
        if st.get("cmd_rad"):
            qc = np.asarray(st["cmd_rad"], float)
            if float(np.max(np.abs(qc - q))) > 1e-4:
                out["cmd_links"] = [[0, 0, 0]] + [T[:3, 3].tolist() for _, T in g.model.fk_chain(qc)]
        if st.get("desired_rad"):
            qd = np.asarray(st["desired_rad"], float)
            if float(np.max(np.abs(qd - q))) > 1e-3:
                out["target_links"] = [[0, 0, 0]] + [T[:3, 3].tolist() for _, T in g.model.fk_chain(qd)]
    # 姿态集的手腕点（小点云）
    try:
        out["pose_tips"] = [p.get("tip_xyz") or g.tip_T(p["q"])[:3, 3].tolist() for p in current_set()["poses"]]
    except FileNotFoundError:
        out["pose_tips"] = []
    out["teach"] = teach["active"]
    out["teach_locked"] = teach["locked"]
    return out


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False, default=_default).encode(), "application/json; charset=utf-8")

    def _err(self, exc: Exception) -> None:
        code = {ValueError: 400, LookupError: 409, FileNotFoundError: 404, PermissionError: 403,
                RuntimeError: 409, TimeoutError: 504}.get(type(exc), 500)
        if code == 500:
            traceback.print_exc()
        self._json(code, {"error": str(exc)})

    def _read_json(self) -> Dict:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError("请求体不是合法 JSON") from exc

    def _static(self, rel: str) -> None:
        p = (FRONTEND / rel).resolve()
        if not str(p).startswith(str(FRONTEND)) or not p.is_file():
            return self._json(404, {"error": "not found"})
        ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}.get(p.suffix, "application/octet-stream")
        self._send(200, p.read_bytes(), ctype)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/status":
                return self._json(200, status_payload())
            if url.path == "/api/ctrl":
                return self._json(200, ctrl.status() if ctrl else None)
            if url.path == "/api/task":
                return self._json(200, task.snapshot() if task else None)
            if url.path == "/api/scene":
                return self._json(200, scene())
            if url.path == "/api/poses":
                return self._json(200, current_set())
            if url.path == "/api/posesets":
                return self._json(200, {"posesets": list_sets(qs.get("arm", [None])[0]), "current": poseset_name.get(arm)})
            if url.path == "/api/poseset":
                return self._json(200, load_set(qs.get("name", [""])[0]))
            if url.path == "/api/teach":
                return self._json(200, teach_state())
            if url.path == "/api/sessions":
                return self._json(200, {"sessions": list_sessions()})
            if url.path == "/api/session":
                d = MEASURE_DIR / qs.get("name", [""])[0]
                if not (d / "samples.json").exists():
                    raise FileNotFoundError("没有该会话")
                return self._json(200, {"samples": json.loads((d / "samples.json").read_text()),
                                        "meta": json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {},
                                        "solved": {p.stem: json.loads(p.read_text()) for p in d.glob("payload_*.json")}})
            if url.path == "/api/validations":
                return self._json(200, {"validations": list_validations()})
            if url.path == "/api/validation":
                p = VALIDATE_DIR / (qs.get("name", [""])[0] + ".json")
                if not p.exists():
                    raise FileNotFoundError("没有该验证记录")
                return self._json(200, json.loads(p.read_text()))
            self._json(404, {"error": f"未知路径 {url.path}"})
        except Exception as exc:  # noqa: BLE001
            self._err(exc)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            body = self._read_json()
            p = url.path
            if p == "/api/mode":
                return self._json(200, switch_mode(body.get("mode", mode), body.get("arm", arm)))
            if p == "/api/engage":
                return self._json(200, engage())
            if p == "/api/release":
                return self._json(200, release())
            if p == "/api/hold":
                return self._json(200, hold())
            if p == "/api/task/stop":
                if task is not None:
                    task.stop()
                return self._json(200, {"ok": True})
            if p == "/api/sim/truth":
                if mode != "sim":
                    raise RuntimeError("只有仿真模式能改真值")
                sim_truth.update(body)
                return self._json(200, sim_truth.as_dict())
            if p == "/api/speed":
                v = float(body.get("max_speed_rad_s"))
                if ctrl is None:
                    raise RuntimeError("没有控制器")
                ctrl.set_max_speed(v)
                return self._json(200, {"max_speed_rad_s": ctrl.status().get("max_speed_rad_s")})
            # ---- 姿态集 ----
            if p == "/api/posesets/create":
                d = create_set(body.get("name", ""), arm, body.get("note", ""))
                poseset_name[arm] = d["name"]
                return self._json(200, set_summary())
            if p == "/api/posesets/select":
                select_set(body.get("name", ""))
                return self._json(200, set_summary())
            if p == "/api/posesets/delete":
                _require_idle()
                name = body.get("name", "")
                delete_set(name)
                if poseset_name.get(arm) == name:
                    poseset_name[arm] = None
                return self._json(200, {"ok": True, "posesets": list_sets(arm)})
            if p == "/api/posesets/update":
                update_set(current_set()["name"], body)
                return self._json(200, set_summary())
            if p == "/api/posesets/pose/delete":
                _require_idle()
                delete_pose(current_set()["name"], int(body.get("index", -1)))
                return self._json(200, set_summary())
            if p == "/api/posesets/add_current":
                # 不进示教也能记：把当前实测角追加进当前姿态集（例如用走点面板摆好后）
                st = ctrl.status() if ctrl else None
                if not st or not st.get("measured_rad"):
                    raise RuntimeError("读不到实测关节角")
                r = add_pose(current_set()["name"], st["measured_rad"], body.get("pose_name"), source="manual")
                return self._json(200, {"pose": r["pose"], "set": set_summary()})
            if p == "/api/posesets/gen":
                _require_idle()
                d = generate(arm, int(body.get("n", 12)), int(body.get("seed", 1)), name=body.get("name") or None)
                save_set(d)
                poseset_name[arm] = d["name"]
                return self._json(200, set_summary())
            # ---- 示教 ----
            if p == "/api/teach/start":
                return self._json(200, teach_start())
            if p == "/api/teach/toggle":
                return self._json(200, teach_toggle(body.get("pose_name")))
            if p == "/api/teach/lock":
                return self._json(200, teach_lock(body.get("pose_name")))
            if p == "/api/teach/float":
                return self._json(200, teach_float())
            if p == "/api/teach/end":
                return self._json(200, teach_end())
            if p == "/api/teach/push":
                return self._json(200, teach_push(body.get("q")))
            if p == "/api/teach/still":
                return self._json(200, set_teach_still(body))
            if p == "/api/measure/start":
                return self._json(200, start_measure(body))
            if p == "/api/solve":
                return self._json(200, do_solve(body))
            if p == "/api/payload/apply":
                return self._json(200, apply_payload(body))
            if p == "/api/payload/clear":
                return self._json(200, apply_payload({"mass_kg": 0, "com_m": [0, 0, 0], "alpha": 1.0, "save": False}))
            if p == "/api/validate/start":
                return self._json(200, start_validate(body))
            if p == "/api/goto":
                return self._json(200, start_goto(body))
            self._json(404, {"error": f"未知路径 {p}"})
        except Exception as exc:  # noqa: BLE001
            self._err(exc)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _kill_child_processes() -> None:
    import os
    me = os.getpid()
    kids = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == me:
                kids.append(int(d.name))
        except (OSError, IndexError, ValueError):
            continue
    for pid in kids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def main() -> int:
    global allow_real_motion, dds_iface, max_speed
    ap = argparse.ArgumentParser(description="末端负载辨识 / 重力前馈补偿 / 到位验证")
    ap.add_argument("--port", type=int, default=10183)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--mode", default="sim", choices=["sim", "real"])
    ap.add_argument("--arm", default="right", choices=["left", "right"])
    ap.add_argument("--iface", default=None, help="DDS 网卡名，如 enp86s0")
    ap.add_argument("--max-speed", type=float, default=0.15, help="控制器限速天花板 rad/s（真机建议 ≤0.15）")
    ap.add_argument("--allow-real-motion", action="store_true", help="⚠ 允许真机模式接管手臂（rt/arm_sdk）")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    allow_real_motion, dds_iface, max_speed = bool(args.allow_real_motion), args.iface, float(args.max_speed)
    if allow_real_motion:
        print("[motion] ⚠⚠ 已启用真机运动。真机模式点「接管」即发布 rt/arm_sdk，务必确认没有别的程序在控臂、有人守着。")

    switch_mode("sim", args.arm)
    if args.mode == "real":
        try:
            switch_mode("real", args.arm)
        except RuntimeError:
            print("[mode] 留在仿真模式，可在网页里重试")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"[http] http://{args.host}:{args.port}/   (mode={mode} arm={arm})")
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=httpd.shutdown, daemon=True).start())
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if task is not None and task.running:
            task.stop(); task.join(5)
        if ctrl is not None:
            try:
                ctrl.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if monitor is not None:
            monitor.close()
        _kill_child_processes()
    return 0


if __name__ == "__main__":
    sys.exit(main())
