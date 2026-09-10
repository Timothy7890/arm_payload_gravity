"""手臂重力模型 + 末端负载回归矩阵（只依赖 numpy + 标准库）。

两件事：
1. ``ArmGravity.tau_arm(q)``  —— URDF 自带连杆质量产生的重力力矩 τ_g(q)，7 维。
   算法与作者 hand_eye_3D/backend/gravity.py 相同（势能对关节角求导，
   τ_i = −Σ_{j∈下游(i)} m_j · g⃗ · (z_i × (p_cj − p_i))），已对拍到 1e-12 Nm，见 ``python gravity.py``。
2. ``ArmGravity.payload_regressor(q)`` —— 末端**点质量负载**的力矩对参数是线性的：

       τ_payload(q) = Y(q) · θ ,   θ = [m, m·cx, m·cy, m·cz]

   其中 (cx,cy,cz) 是负载质心在末端连杆（<arm>_wrist_yaw_link）坐标系下的位置。
   推导：负载位置 p = p_L + R_L c，τ_i = −m g⃗·(z_i × (p − p_i))
        = −[g⃗·(z_i×(p_L−p_i))]·m − Σ_k [g⃗·(z_i × R_L[:,k])]·(m c_k)。
   于是静态多姿态测量 → 一个线性最小二乘就能同时解出质量和质心，这就是 identify.py 的核心。

坐标系：一切在 torso_link 下算；重力默认 (0,0,−g)，即躯干直立。作者模型根在 pelvis、
腰关节取 0，pelvis→torso_link 只有平移，所以两者数值完全一致。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from urdf_fk import URDFRobotModel, axis_angle_to_matrix, make_transform

HERE = Path(__file__).resolve().parent
URDF = HERE / "assets" / "h2" / "robot.urdf"
G_ACCEL = 9.81

ARM_JOINT_NAMES = {
    arm: [f"{arm}_{s}_joint" for s in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                        "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    for arm in ("left", "right")
}
TIP_LINK = {arm: f"{arm}_wrist_yaw_link" for arm in ("left", "right")}
BASE_LINK = "torso_link"


@dataclass
class _Joint:
    name: str
    jtype: str
    parent: str
    child: str
    T_origin: np.ndarray
    axis: np.ndarray


def _parse_urdf(urdf_path: Path):
    root = ET.parse(str(urdf_path)).getroot()
    joints: Dict[str, _Joint] = {}
    children: Dict[str, List[_Joint]] = {}
    for el in root.findall("joint"):
        o = el.find("origin")
        xyz = [float(v) for v in (o.attrib.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.attrib.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        a = el.find("axis")
        axis = np.array([float(v) for v in a.attrib.get("xyz", "1 0 0").split()]) if a is not None else np.array([1.0, 0, 0])
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
        j = _Joint(el.attrib["name"], el.attrib.get("type", "fixed"), el.find("parent").attrib["link"],
                   el.find("child").attrib["link"], make_transform(xyz, rpy), axis)
        joints[j.name] = j
        children.setdefault(j.parent, []).append(j)
    inertials: Dict[str, Tuple[float, np.ndarray]] = {}
    efforts: Dict[str, float] = {}
    for el in root.findall("link"):
        ine = el.find("inertial")
        if ine is None:
            continue
        m_el = ine.find("mass")
        mass = float(m_el.attrib.get("value", 0.0)) if m_el is not None else 0.0
        if mass <= 0:
            continue
        o = ine.find("origin")
        com = np.array([float(v) for v in (o.attrib.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        inertials[el.attrib["name"]] = (mass, com)
    for el in root.findall("joint"):
        lim = el.find("limit")
        if lim is not None and "effort" in lim.attrib:
            efforts[el.attrib["name"]] = float(lim.attrib["effort"])
    return joints, children, inertials, efforts


class ArmGravity:
    """某条手臂（left/right）的重力模型；base = torso_link。"""

    def __init__(self, arm: str, urdf_path: Path = URDF):
        if arm not in ARM_JOINT_NAMES:
            raise ValueError("arm 必须是 left/right")
        self.arm = arm
        self.joint_names = ARM_JOINT_NAMES[arm]
        self.tip_link = TIP_LINK[arm]
        self.n = 7
        joints, children, inertials, efforts = _parse_urdf(Path(urdf_path))
        self._joints = joints
        self._children = children
        self._chain = [joints[n] for n in self.joint_names]
        self.effort_limits = np.array([efforts.get(n, 10.0) for n in self.joint_names])

        # 链基座（第一个关节的 parent）到 torso_link 的静态变换：用 URDFRobotModel 走一遍 fixed 链
        first_parent = self._chain[0].parent
        self._T_base_firstparent = self._static_T(BASE_LINK, first_parent)

        # 每个链关节的下游连杆（含 fixed 子链，如 hand_link）
        self._downstream: List[List[str]] = []
        for j in self._chain:
            sub = [ln for ln in self._subtree(j.child) if ln in inertials]
            self._downstream.append(sub)
        self.bodies = {ln: inertials[ln] for sub in self._downstream for ln in sub}
        # FK 只需展开第一个链关节 child 的子树
        self._fk_root = self._chain[0]
        self.model = URDFRobotModel(str(urdf_path), BASE_LINK, self.tip_link)
        self.limits = self.model.joint_limits()

    # ---- 树 ----
    def _subtree(self, link: str) -> List[str]:
        out, stack = [], [link]
        while stack:
            ln = stack.pop()
            out.append(ln)
            stack.extend(j.child for j in self._children.get(ln, []))
        return out

    def _static_T(self, from_link: str, to_link: str) -> np.ndarray:
        if from_link == to_link:
            return np.eye(4)
        # 深搜路径（只允许 fixed / 零位关节）
        stack = [(from_link, np.eye(4))]
        seen = set()
        while stack:
            ln, T = stack.pop()
            if ln == to_link:
                return T
            if ln in seen:
                continue
            seen.add(ln)
            for j in self._children.get(ln, []):
                stack.append((j.child, T @ j.T_origin))
        raise ValueError(f"URDF 中找不到 {from_link}→{to_link}")

    # ---- FK：返回链上所有下游连杆位姿 ----
    def _fk(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        qmap = dict(zip(self.joint_names, [float(v) for v in q]))
        T0 = self._T_base_firstparent
        out: Dict[str, np.ndarray] = {self._fk_root.parent: T0}
        stack = [self._fk_root.parent]
        while stack:
            ln = stack.pop()
            Tp = out[ln]
            for j in self._children.get(ln, []):
                if ln == self._fk_root.parent and j is not self._fk_root:
                    continue           # 基座下其它分支（头、另一臂）不展开
                T = Tp @ j.T_origin
                if j.jtype in ("revolute", "continuous"):
                    v = qmap.get(j.name, 0.0)
                    if v != 0.0:
                        R = axis_angle_to_matrix(j.axis, v)
                        M = np.eye(4); M[:3, :3] = R
                        T = T @ M
                elif j.jtype == "prismatic":
                    T = T.copy(); T[:3, 3] += T[:3, :3] @ (j.axis * qmap.get(j.name, 0.0))
                out[j.child] = T
                stack.append(j.child)
        return out

    @staticmethod
    def _g_vec(g_dir) -> np.ndarray:
        if g_dir is None:
            return np.array([0.0, 0.0, -G_ACCEL])
        g = np.asarray(g_dir, float).reshape(3)
        n = float(np.linalg.norm(g))
        return g / n * G_ACCEL if n > 1e-9 else np.array([0.0, 0.0, -G_ACCEL])

    def _axes(self, fk: Dict[str, np.ndarray]):
        zs, ps = [], []
        for j in self._chain:
            T = fk[j.child]
            zs.append(T[:3, :3] @ j.axis)
            ps.append(T[:3, 3])
        return zs, ps

    # ---- 公开接口 ----
    def tau_arm(self, q, g_dir=None) -> np.ndarray:
        """URDF 连杆自重的重力力矩（Nm，7 维）。"""
        q = np.asarray(q, float).reshape(-1)
        if q.size != self.n:
            raise ValueError(f"需要 {self.n} 个关节角，收到 {q.size}")
        g = self._g_vec(g_dir)
        fk = self._fk(q)
        coms = {ln: fk[ln][:3, :3] @ com + fk[ln][:3, 3] for ln, (m, com) in self.bodies.items()}
        zs, ps = self._axes(fk)
        tau = np.zeros(self.n)
        for i, j in enumerate(self._chain):
            tot = 0.0
            if j.jtype == "prismatic":
                for ln in self._downstream[i]:
                    tot += self.bodies[ln][0] * float(g @ zs[i])
            else:
                for ln in self._downstream[i]:
                    tot += self.bodies[ln][0] * float(g @ np.cross(zs[i], coms[ln] - ps[i]))
            tau[i] = -tot
        return tau

    def payload_regressor(self, q, g_dir=None) -> np.ndarray:
        """Y(q)：7×4，使 τ_payload = Y @ [m, m·cx, m·cy, m·cz]（质心在末端连杆系）。"""
        q = np.asarray(q, float).reshape(-1)
        g = self._g_vec(g_dir)
        fk = self._fk(q)
        zs, ps = self._axes(fk)
        TL = fk[self.tip_link]
        pL, RL = TL[:3, 3], TL[:3, :3]
        Y = np.zeros((self.n, 4))
        for i, j in enumerate(self._chain):
            if j.jtype == "prismatic":
                Y[i, 0] = -float(g @ zs[i])
                continue
            Y[i, 0] = -float(g @ np.cross(zs[i], pL - ps[i]))
            for k in range(3):
                Y[i, 1 + k] = -float(g @ np.cross(zs[i], RL[:, k]))
        return Y

    def tau_payload(self, q, mass: float, com, g_dir=None) -> np.ndarray:
        theta = payload_theta(mass, com)
        return self.payload_regressor(q, g_dir) @ theta

    def tau_total(self, q, mass: float = 0.0, com=(0, 0, 0), alpha: float = 1.0, g_dir=None) -> np.ndarray:
        """alpha·τ_arm + τ_payload —— 新的重力前馈。"""
        tau = alpha * self.tau_arm(q, g_dir)
        if mass > 0:
            tau = tau + self.tau_payload(q, mass, com, g_dir)
        return tau

    def tip_T(self, q) -> np.ndarray:
        return self.model.fk_T(q)

    def payload_point(self, q, com) -> np.ndarray:
        """负载质心在 torso_link 下的位置（画图用）。"""
        T = self.tip_T(q)
        return T[:3, :3] @ np.asarray(com, float) + T[:3, 3]

    def describe(self) -> Dict:
        return {"arm": self.arm, "moving_links": {ln: round(m, 4) for ln, (m, _) in self.bodies.items()},
                "moving_mass_kg": round(sum(m for m, _ in self.bodies.values()), 4),
                "tip_link": self.tip_link, "effort_limits": self.effort_limits.tolist()}


def payload_theta(mass: float, com) -> np.ndarray:
    com = np.asarray(com, float).reshape(3)
    return np.array([mass, mass * com[0], mass * com[1], mass * com[2]])


def theta_to_payload(theta) -> Tuple[float, np.ndarray]:
    theta = np.asarray(theta, float).reshape(4)
    m = float(theta[0])
    com = theta[1:] / m if abs(m) > 1e-6 else np.zeros(3)
    return m, com


class GravityWithPayload:
    """给作者 H2ArmController 用的 ``_grav_model`` 替身：``torque(q, g_dir)`` = 基础模型 + 负载项。

    base 可以是作者的 ArmGravityModel（真机，保持与原实现逐位一致），也可以是本模块的 ArmGravity。
    """

    def __init__(self, base, grav: ArmGravity, mass: float = 0.0, com=(0.0, 0.0, 0.0), alpha: float = 1.0):
        self.base = base
        self.grav = grav
        self.mass = float(mass)
        self.com = np.asarray(com, float).reshape(3)
        self.alpha = float(alpha)          # 只缩放臂自重项，不缩放负载项

    def torque(self, q, g_dir=None) -> np.ndarray:
        base_tau = self.base.torque(q, g_dir=g_dir) if hasattr(self.base, "torque") else self.base.tau_arm(q, g_dir)
        tau = self.alpha * base_tau
        if self.mass > 0:
            tau = tau + self.grav.tau_payload(q, self.mass, self.com, g_dir)
        return tau

    def as_dict(self) -> Dict:
        return {"mass_kg": self.mass, "com_m": self.com.tolist(), "alpha": self.alpha}

    def describe(self) -> Dict:
        d = dict(self.base.describe()) if hasattr(self.base, "describe") else {}
        d.update({"payload_mass_kg": self.mass, "payload_com_m": self.com.tolist(), "alpha": self.alpha})
        return d


# ---------------------------------------------------------------- 自检：和作者模型对拍
def _selfcheck() -> int:
    import sys
    rng = np.random.default_rng(0)
    ok = True
    for arm in ("left", "right"):
        g = ArmGravity(arm)
        print(f"[{arm}] 参与质量 {g.describe()['moving_mass_kg']} kg，连杆 {list(g.bodies)}")
        # 1) 负载回归矩阵 vs 直接把点质量当作连杆加进去（数值验证线性化）
        for _ in range(5):
            q = rng.uniform(g.limits[:, 0], g.limits[:, 1])
            m, c = 1.3, np.array([0.06, -0.02, 0.03])
            direct = _payload_direct(g, q, m, c)
            lin = g.tau_payload(q, m, c)
            err = float(np.max(np.abs(direct - lin)))
            ok &= err < 1e-9
        print(f"[{arm}] 回归矩阵线性化 vs 直接算：最大误差 {err:.2e} Nm")
        # 2) 与作者 ArmGravityModel 对拍
        try:
            sys.path.insert(0, "/home/robot/yx/project/calib/hand_eye_3D")
            from backend.gravity import ArmGravityModel  # type: ignore
            from backend.paths import H2_ROBOT_CONFIG_PATH  # type: ignore
            from backend.robotics import RobotModel, load_robot_config  # type: ignore
            model = RobotModel(load_robot_config(H2_ROBOT_CONFIG_PATH))
            ref = ArmGravityModel(model, f"{arm}_arm")
            worst = 0.0
            for _ in range(20):
                q = rng.uniform(g.limits[:, 0], g.limits[:, 1])
                worst = max(worst, float(np.max(np.abs(ref.torque(q) - g.tau_arm(q)))))
            print(f"[{arm}] 与作者 ArmGravityModel 对拍 20 姿态：最大差 {worst:.2e} Nm  {'PASS' if worst < 1e-9 else 'FAIL'}")
            ok &= worst < 1e-9
        except Exception as exc:  # noqa: BLE001
            print(f"[{arm}] 作者模型不可用，跳过对拍：{exc}")
    return 0 if ok else 1


def _payload_direct(g: ArmGravity, q, m, c) -> np.ndarray:
    gv = np.array([0.0, 0.0, -G_ACCEL])
    fk = g._fk(q)
    zs, ps = g._axes(fk)
    T = fk[g.tip_link]
    p = T[:3, :3] @ c + T[:3, 3]
    return np.array([-m * float(gv @ np.cross(zs[i], p - ps[i])) for i in range(g.n)])


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
