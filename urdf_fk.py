"""URDF 单链正运动学（只依赖 numpy + 标准库）。

改自 /home/robot/yx/project/calib/hand_eye_2D/urdf_robot_model.py（去掉了对
ik_5d_suction_solver 的依赖，其余逻辑一致）。从 base_link 到 tip_link 抽出一条
串联链，支持 fixed / revolute / continuous / prismatic 关节。

    model = URDFRobotModel("assets/h2/robot.urdf", "torso_link", "right_wrist_yaw_link")
    p, R = model.fk(q)          # tip 在 base 下的位置(3,)与旋转(3,3)
    T = model.fk_T(q)           # 4x4
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def _parse_xyz(value: Optional[str], default) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    parts = [float(x) for x in value.split()]
    if len(parts) != 3:
        raise ValueError(f"需要 3 个数，得到: {value}")
    return np.asarray(parts, dtype=float)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF 固定轴 RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = [float(x) for x in rpy]
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """rpy_to_matrix 的逆。"""
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def axis_angle_to_matrix(axis, angle: float) -> np.ndarray:
    k = normalize(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def make_transform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return T


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _motion_transform(joint_type: str, axis, q: float) -> np.ndarray:
    T = np.eye(4)
    if joint_type in ("revolute", "continuous"):
        T[:3, :3] = axis_angle_to_matrix(axis, q)
    elif joint_type == "prismatic":
        T[:3, 3] = normalize(axis) * float(q)
    elif joint_type == "fixed":
        pass
    else:
        raise ValueError(f"链中有不支持的关节类型: {joint_type}")
    return T


@dataclass
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit: Optional[Tuple[float, float]]

    @property
    def is_active(self) -> bool:
        return self.joint_type in ("revolute", "continuous", "prismatic")


class URDFRobotModel:
    def __init__(self, urdf_path: str, base_link: str, tip_link: str):
        self.urdf_path = str(urdf_path)
        self.base_link = base_link
        self.tip_link = tip_link
        self.joints = self._load_chain(self.urdf_path, base_link, tip_link)
        self.active_joints = [j for j in self.joints if j.is_active]
        self.joint_names = [j.name for j in self.active_joints]
        self.n = len(self.active_joints)
        if self.n == 0:
            raise ValueError("所选 URDF 链没有活动关节")

    @staticmethod
    def _load_chain(urdf_path: str, base_link: str, tip_link: str) -> List[URDFJoint]:
        root = ET.parse(urdf_path).getroot()
        children: Dict[str, List[URDFJoint]] = {}
        for el in root.findall("joint"):
            parent_el, child_el = el.find("parent"), el.find("child")
            if parent_el is None or child_el is None:
                continue
            origin = el.find("origin")
            xyz = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0])
            rpy = _parse_xyz(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0])
            axis_el = el.find("axis")
            axis = _parse_xyz(axis_el.attrib.get("xyz"), [1, 0, 0]) if axis_el is not None else np.array([1.0, 0, 0])
            jtype = el.attrib.get("type", "fixed")
            limit_el = el.find("limit")
            limit = None
            if limit_el is not None and "lower" in limit_el.attrib and "upper" in limit_el.attrib:
                limit = (float(limit_el.attrib["lower"]), float(limit_el.attrib["upper"]))
            elif jtype == "continuous":
                limit = (-math.pi, math.pi)
            j = URDFJoint(el.attrib["name"], jtype, parent_el.attrib["link"], child_el.attrib["link"],
                          xyz, rpy, axis, limit)
            children.setdefault(j.parent, []).append(j)

        stack = [(base_link, [])]
        visited = set()
        while stack:
            link, path = stack.pop()
            if link == tip_link:
                return path
            if link in visited:
                continue
            visited.add(link)
            for j in children.get(link, []):
                stack.append((j.child, path + [j]))
        raise ValueError(f"URDF 中找不到 {base_link!r} → {tip_link!r} 的链")

    def joint_limits(self) -> np.ndarray:
        return np.asarray([j.limit if j.limit is not None else (-math.pi, math.pi)
                           for j in self.active_joints], dtype=float)

    def fk_T(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n:
            raise ValueError(f"需要 {self.n} 个关节角，得到 {q.size}")
        T = np.eye(4)
        k = 0
        for j in self.joints:
            T = T @ make_transform(j.origin_xyz, j.origin_rpy)
            if j.is_active:
                T = T @ _motion_transform(j.joint_type, j.axis, q[k])
                k += 1
        return T

    def fk(self, q) -> Tuple[np.ndarray, np.ndarray]:
        T = self.fk_T(q)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def fk_chain(self, q) -> List[Tuple[str, np.ndarray]]:
        """沿链每个关节 child link 的位姿 [(link_name, T4x4), ...]，用于画骨架。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        out: List[Tuple[str, np.ndarray]] = []
        T = np.eye(4)
        k = 0
        for j in self.joints:
            T = T @ make_transform(j.origin_xyz, j.origin_rpy)
            if j.is_active:
                T = T @ _motion_transform(j.joint_type, j.axis, q[k])
                k += 1
            out.append((j.child, T.copy()))
        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    urdf = Path(__file__).resolve().parent / "assets" / "h2" / "robot.urdf"
    arm = sys.argv[1] if len(sys.argv) > 1 else "right"
    m = URDFRobotModel(str(urdf), "torso_link", f"{arm}_wrist_yaw_link")
    print("joints:", m.joint_names)
    p, R = m.fk(np.zeros(m.n))
    print("零位腕心 (torso_link):", np.round(p, 4))
