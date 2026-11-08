"""Run model-written code inside a task workspace with a timeout."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

PY = sys.executable or "python3"


@dataclass
class Run:
    code: int
    out: str
    err: str
    timeout: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timeout


def _env():
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("LLM_"):
            env.pop(k)  # model-written code never sees the key
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_python(code: str, cwd: str, timeout: int = 30) -> Run:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="agent_", dir=tempfile.gettempdir())
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        return _run([PY, path], cwd, timeout)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_shell(cmd: str, cwd: str, timeout: int = 30) -> Run:
    return _run(["bash", "-lc", cmd] if os.name != "nt" else ["cmd", "/c", cmd], cwd, timeout)


def run_argv(argv, cwd, timeout: int = 30) -> Run:
    return _run(argv, cwd, timeout)


def _run(argv, cwd, timeout) -> Run:
    try:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout,
                           env=_env(), stdin=subprocess.DEVNULL)
        return Run(p.returncode, p.stdout.decode("utf-8", "replace"),
                   p.stderr.decode("utf-8", "replace"))
    except subprocess.TimeoutExpired as e:
        return Run(-1, (e.stdout or b"").decode("utf-8", "replace"),
                   (e.stderr or b"").decode("utf-8", "replace"), timeout=True)
    except OSError as e:
        return Run(-1, "", str(e))


def clip(text: str, n: int = 1500) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n // 2] + "\n...[cut]...\n" + text[-n // 2:]


CODE_RE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    blocks = CODE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    m = re.search(r"```[\w+-]*[ \t]*\n(.*)", text, re.S)  # unclosed (truncated) block
    if m:
        return m.group(1).strip()
    # model answered with bare code
    return text.strip().strip("`").strip()
