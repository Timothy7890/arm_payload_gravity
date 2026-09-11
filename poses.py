"""姿态集：config/posesets/<名字>.json，可有多个（对应不同末端负载 / 不同用途），可命名。

两种来路：
1. **示教**（推荐，网页「示教」面板）：手臂卸力，人手推到想要的位置，按空格 → 手臂锁定当前姿态、
   静止后读取实测关节角追加进当前姿态集。姿态是你自己定的，所见即所得。
2. **自动生成**（``python poses.py gen --arm right --n 12 --name auto_right``）：URDF 限位内随机采样，
   FK 后要求手腕在躯干前方的安全盒子里，再贪心 D-最优挑 n 个（让 [m, m·cx, m·cy, m·cz] 都可辨识）。
   只做了工作空间约束，**没有自碰撞检查**，上真机前必须在 3D 视图里逐个走一遍。

不论哪种来路，网页都会显示该姿态集的回归矩阵条件数：越接近 1 越好，>50 说明姿态太雷同
（示教时多变换腕部的翻转/俯仰、手臂的抬高/外展）。

文件格式：
  {"name", "arm", "note", "created", "home_q", "motion": {...}, "poses": [{"name", "q", "tip_xyz", "recorded_at", "source"}]}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gravity import ArmGravity

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config"
SETS_DIR = CONFIG / "posesets"

HOME_Q = {"right": [0.2, -0.25, 0.0, 0.9, 0.0, -0.1, 0.0], "left": [0.2, 0.25, 0.0, 0.9, 0.0, -0.1, 0.0]}
DEFAULT_MOTION = {"vmax_rad_s": 0.25, "amax_rad_s2": 0.5, "min_segment_s": 1.0}
_NAME_RE = re.compile(r"^[\w\-\u4e00-\u9fff.]{1,48}$")


def _check_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError("姿态集名字只能用中英文、数字、_ - .，1～48 个字符")
    return name


def set_path(name: str) -> Path:
    return SETS_DIR / f"{_check_name(name)}.json"


def _migrate_legacy() -> None:
    """老版本 config/poses_<arm>.json → posesets/auto_<arm>.json。"""
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ("left", "right"):
        old = CONFIG / f"poses_{arm}.json"
        new = SETS_DIR / f"auto_{arm}.json"
        if old.exists() and not new.exists():
            d = json.loads(old.read_text(encoding="utf-8"))
            d.setdefault("name", f"auto_{arm}")
            for p in d["poses"]:
                p.setdefault("source", "gen")
            new.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            old.unlink()


def list_sets(arm: Optional[str] = None) -> List[Dict]:
    _migrate_legacy()
    out = []
    for p in sorted(SETS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if arm and d.get("arm") != arm:
            continue
        out.append({"name": d.get("name", p.stem), "arm": d.get("arm"), "n": len(d.get("poses", [])),
                    "note": d.get("note", ""), "created": d.get("created"), "cond": d.get("regressor_cond")})
    return out


def load_set(name: str) -> Dict:
    _migrate_legacy()
    p = set_path(name)
    if not p.exists():
        raise FileNotFoundError(f"没有姿态集 {name}")
    d = json.loads(p.read_text(encoding="utf-8"))
    d.setdefault("motion", dict(DEFAULT_MOTION))
    d.setdefault("home_q", HOME_Q[d["arm"]])
    return d


def save_set(d: Dict) -> Path:
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    d["regressor_cond"] = round(condition_number(d["arm"], [p["q"] for p in d["poses"]]), 2) if len(d["poses"]) >= 2 else None
    d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p = set_path(d["name"])
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def create_set(name: str, arm: str, note: str = "") -> Dict:
    name = _check_name(name)
    if set_path(name).exists():
        raise ValueError(f"姿态集 {name} 已存在")
    if arm not in HOME_Q:
        raise ValueError("arm 必须是 left/right")
    d = {"name": name, "arm": arm, "note": note, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "home_q": HOME_Q[arm], "motion": dict(DEFAULT_MOTION), "poses": []}
    save_set(d)
    return d


def delete_set(name: str) -> None:
    p = set_path(name)
    if not p.exists():
        raise FileNotFoundError(f"没有姿态集 {name}")
    p.unlink()


def add_pose(name: str, q, pose_name: Optional[str] = None, source: str = "teach") -> Dict:
    d = load_set(name)
    q = [round(float(v), 5) for v in np.asarray(q, float).reshape(-1)]
    if len(q) != 7:
        raise ValueError("需要 7 个关节角")
    grav = ArmGravity(d["arm"])
    pose = {"name": pose_name or f"p{len(d['poses']):02d}", "q": q,
            "tip_xyz": [round(float(v), 4) for v in grav.tip_T(q)[:3, 3]],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source}
    d["poses"].append(pose)
    save_set(d)
    return {"pose": pose, "set": d}


def delete_pose(name: str, index: int) -> Dict:
    d = load_set(name)
    if not (0 <= index < len(d["poses"])):
        raise IndexError("姿态下标越界")
    d["poses"].pop(index)
    save_set(d)
    return d


def update_set(name: str, patch: Dict) -> Dict:
    d = load_set(name)
    if "note" in patch:
        d["note"] = str(patch["note"])
    if "home_q" in patch:
        h = [float(v) for v in patch["home_q"]]
        if len(h) != 7:
            raise ValueError("home_q 需要 7 个数")
        d["home_q"] = h
    if "motion" in patch:
        d["motion"] = {**d.get("motion", {}), **{k: float(v) for k, v in patch["motion"].items()}}
    save_set(d)
    return d


# ---------------------------------------------------------------- 左右镜像
# H2 左右臂关于 y=0 面镜像：绕 y 轴的关节（肩 pitch、肘、腕 pitch）符号不变，
# 绕 x/z 轴的关节（肩 roll、肩 yaw、腕 roll、腕 yaw）取反。URDF 限位也正好对应（右肩 roll [-2.49,0.52] ↔ 左 [-0.52,2.49]）。
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def mirror_q(q) -> List[float]:
    return [round(float(v), 5) + 0.0 for v in np.asarray(q, float).reshape(7) * MIRROR_SIGN]   # +0.0 去掉 -0.0


def _load_any(src: str) -> Dict:
    """src 可以是姿态集名字，也可以是任意路径的 .json 文件。"""
    p = Path(src).expanduser()
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return load_set(src)


def mirror_set(src: str, name: Optional[str] = None, note: str = "", out: Optional[str] = None,
               force: bool = False) -> Path:
    """把一个姿态集镜像到另一条手臂。

    src   源姿态集（名字或 .json 路径）；name 新名字（默认 <源名>_<左|右>）；
    out   输出目录或 .json 文件路径（默认 config/posesets/）。
    只镜像姿态与 home_q，不镜像负载参数（左右末端/电机/摩擦各不相同，左臂要重新测量求解）。
    """
    d = _load_any(src)
    src_arm = d.get("arm")
    if src_arm not in HOME_Q:
        raise ValueError(f"源姿态集 arm 字段无效：{src_arm!r}")
    dst_arm = "left" if src_arm == "right" else "right"
    grav = ArmGravity(dst_arm)
    lo, hi = grav.limits[:, 0], grav.limits[:, 1]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    poses = []
    for p in d.get("poses", []):
        q = np.asarray(mirror_q(p["q"]), float)
        bad = np.where((q < lo - 1e-6) | (q > hi + 1e-6))[0]
        if len(bad):
            raise ValueError(f"姿态 {p.get('name')} 镜像后超出 {dst_arm} 臂限位（关节 {bad.tolist()}）")
        poses.append({"name": p.get("name", f"p{len(poses):02d}"), "q": q.round(5).tolist(),
                      "tip_xyz": [round(float(v), 4) for v in grav.tip_T(q)[:3, 3]],
                      "recorded_at": now, "source": f"mirror:{p.get('source', '?')}"})
    new_name = _check_name(name or f"{d.get('name', 'set')}_{'左' if dst_arm == 'left' else '右'}")
    out_d = {"name": new_name, "arm": dst_arm,
             "note": note or f"由 {src_arm} 臂姿态集「{d.get('name')}」镜像而来。" + (f" {d['note']}" if d.get("note") else ""),
             "created": now, "mirrored_from": {"name": d.get("name"), "arm": src_arm, "path": str(src)},
             "home_q": mirror_q(d.get("home_q", HOME_Q[src_arm])), "motion": dict(d.get("motion") or DEFAULT_MOTION),
             "poses": poses}
    out_d["regressor_cond"] = round(condition_number(dst_arm, [p["q"] for p in poses]), 2) if len(poses) >= 2 else None
    out_d["updated"] = now
    if out:
        op = Path(out).expanduser()
        path = op if op.suffix == ".json" else op / f"{new_name}.json"
    else:
        path = set_path(new_name)
    if path.exists() and not force:
        raise FileExistsError(f"{path} 已存在，加 --force 覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out_d, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def condition_number(arm: str, qs: List) -> float:
    grav = ArmGravity(arm)
    Y = np.vstack([grav.payload_regressor(np.asarray(q, float)) for q in qs])
    scale = np.linalg.norm(Y, axis=0) + 1e-9
    return float(np.linalg.cond(Y / scale))


# ---------------------------------------------------------------- 自动生成
def in_workspace(arm: str, p: np.ndarray) -> bool:
    x, y, z = p
    side = -1.0 if arm == "right" else 1.0
    return (0.15 < x < 0.55) and (0.10 < side * y < 0.50) and (-0.30 < z < 0.50)


def generate(arm: str, n: int = 12, seed: int = 1, candidates: int = 4000, name: Optional[str] = None) -> Dict:
    grav = ArmGravity(arm)
    rng = np.random.default_rng(seed)
    lo, hi = grav.limits[:, 0], grav.limits[:, 1]
    margin = 0.08 * (hi - lo)
    cands: List[np.ndarray] = []
    tries = 0
    while len(cands) < candidates and tries < candidates * 20:
        tries += 1
        q = rng.uniform(lo + margin, hi - margin)
        q[3] = rng.uniform(0.3, hi[3] - margin[3])          # 肘别伸直
        if in_workspace(arm, grav.tip_T(q)[:3, 3]):
            cands.append(q)
    if len(cands) < n:
        raise RuntimeError(f"只采到 {len(cands)} 个可行候选")
    Ys = [grav.payload_regressor(q) for q in cands]
    scale = np.linalg.norm(np.vstack(Ys), axis=0) / np.sqrt(len(Ys)) + 1e-9
    Ys = [Y / scale for Y in Ys]
    chosen: List[int] = []
    M = 1e-6 * np.eye(4)
    for _ in range(n):
        best, best_val = -1, -np.inf
        for i, Y in enumerate(Ys):
            if i in chosen:
                continue
            val = np.linalg.slogdet(M + Y.T @ Y)[1]
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
        M = M + Ys[best].T @ Ys[best]
    qs = [cands[i] for i in chosen]
    ordered: List[np.ndarray] = []
    cur = np.asarray(HOME_Q[arm], float)
    rest = qs[:]
    while rest:
        i = int(np.argmin([np.linalg.norm(p - cur) for p in rest]))
        cur = rest.pop(i)
        ordered.append(cur)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    d = {
        "name": _check_name(name or f"auto_{arm}_{time.strftime('%m%d_%H%M')}"), "arm": arm, "created": now,
        "note": "自动生成：安全盒子内随机采样 + 贪心 D-最优。未做自碰撞检查，上真机前逐个走一遍。",
        "home_q": HOME_Q[arm], "motion": dict(DEFAULT_MOTION),
        "poses": [{"name": f"p{i:02d}", "q": [round(float(v), 4) for v in q],
                   "tip_xyz": [round(float(v), 3) for v in grav.tip_T(q)[:3, 3]], "recorded_at": now, "source": "gen"}
                  for i, q in enumerate(ordered)],
    }
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--arm", default="right", choices=["left", "right"])
    g.add_argument("--n", type=int, default=12); g.add_argument("--seed", type=int, default=1)
    g.add_argument("--name", default=None); g.add_argument("--force", action="store_true")
    sub.add_parser("list")
    s = sub.add_parser("show"); s.add_argument("name")
    m = sub.add_parser("mirror", help="把姿态集镜像到另一条手臂（右↔左）")
    m.add_argument("src", help="源姿态集名字，或 .json 文件路径")
    m.add_argument("--name", default=None, help="新姿态集名字（默认 <源名>_左/右）")
    m.add_argument("--out", default=None, help="输出目录或 .json 路径（默认 config/posesets/）")
    m.add_argument("--note", default=""); m.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "mirror":
        try:
            path = mirror_set(a.src, a.name, a.note, a.out, a.force)
        except (FileExistsError, FileNotFoundError, ValueError) as e:
            raise SystemExit(f"[mirror] {e}")
        d = json.loads(path.read_text(encoding="utf-8"))
        print(f"[mirror] {d['mirrored_from']['arm']} → {d['arm']}，{len(d['poses'])} 个姿态 → {path}，条件数 {d['regressor_cond']}")
    elif a.cmd == "gen":
        d = generate(a.arm, a.n, a.seed, name=a.name)
        if set_path(d["name"]).exists() and not a.force:
            raise SystemExit(f"{set_path(d['name'])} 已存在，加 --force 覆盖")
        print(f"[gen] {len(d['poses'])} 个姿态 → {save_set(d)}，条件数 {d['regressor_cond']}")
    elif a.cmd == "list":
        print(json.dumps(list_sets(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(load_set(a.name), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
