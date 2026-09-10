"""后台任务基类：一个线程、可停止、带状态快照与日志（测量 / 验证 / 单点到位共用）。"""

from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Dict

from motion import Stopped


class BaseRunner:
    name = "task"

    def __init__(self):
        self._lock = threading.Lock()
        self.stop_evt = threading.Event()
        self._thread = None
        self.state: Dict = {"task": self.name, "running": False, "phase": "idle", "progress": [0, 0],
                            "log": [], "error": None, "result": None}

    # ---- 状态 ----
    def log(self, s: str) -> None:
        with self._lock:
            self.state["log"].append(f"{time.strftime('%H:%M:%S')} {s}")
            self.state["log"] = self.state["log"][-300:]

    def set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def phase(self, s: str) -> None:
        self.set(phase=s)

    def snapshot(self) -> Dict:
        with self._lock:
            return json.loads(json.dumps(self.state, default=str))

    @property
    def running(self) -> bool:
        return bool(self.state["running"])

    # ---- 生命周期 ----
    def start(self) -> None:
        with self._lock:
            if self.state["running"]:
                raise RuntimeError(f"{self.name} 已在运行")
            self.state.update(running=True, phase="starting", error=None, result=None, log=[])
        self.stop_evt.clear()
        self._thread = threading.Thread(target=self._guard, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_evt.set()

    def join(self, timeout: float = 30.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _guard(self) -> None:
        try:
            result = self.run()
            self.set(result=result)
            self.log("完成")
        except Stopped:
            self.log("已停止")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.set(error=str(exc))
            self.log(f"[error] {exc}")
        finally:
            self.set(running=False, phase="idle")

    def run(self):
        raise NotImplementedError
