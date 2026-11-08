"""Plain logging to stderr plus a JSONL trajectory file (every LLM call)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

_lock = threading.Lock()
_t0 = time.time()


def quiet_framework_logs() -> None:
    """openJiuwen logs every LLM request/response at INFO; keep stderr readable."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for lg in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(lg, logging.Logger):
            lg.setLevel(logging.WARNING)
    try:
        from loguru import logger as _lg
        _lg.remove()
        _lg.add(sys.stderr, level="WARNING")
    except Exception:
        pass


TRAJ_PATH = os.environ.get("TRAJ_PATH", "/tmp/trajectories.jsonl")


def log(msg: str) -> None:
    with _lock:
        sys.stderr.write(f"[{time.time() - _t0:7.1f}s] {msg}\n")
        sys.stderr.flush()


def traj(rec: dict) -> None:
    rec = dict(rec, t=round(time.time() - _t0, 2))
    try:
        with _lock, open(TRAJ_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
