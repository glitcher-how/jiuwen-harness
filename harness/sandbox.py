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


_RAN = re.compile(r"^Ran (\d+) tests? in", re.M)


def run_test_file(tf: str, cwd: str, timeout: int = 60) -> Run:
    """Run one test file the way the tasks say: python3 FILE. A file without a
    unittest.main() call exits with 0 without running a single test, so when no test
    reported itself the file goes through the unittest runner instead."""
    r = _run([PY, tf], cwd, timeout)
    if r.ok and not _RAN.search(r.err + r.out):
        r2 = _run([PY, "-m", "unittest", tf], cwd, timeout)
        m = _RAN.search(r2.err + r2.out)
        if m and int(m.group(1)) > 0:
            return r2
        # plain test_* functions without unittest: call them one by one
        r3 = _run([PY, "-c", _PLAIN_TESTS, os.path.splitext(os.path.basename(tf))[0]], cwd, timeout)
        m = _RAN.search(r3.err + r3.out)
        if m and int(m.group(1)) > 0:
            return r3
    return r


_PLAIN_TESTS = """
import importlib, sys, traceback
sys.path.insert(0, '.')
m = importlib.import_module(sys.argv[1])
n = bad = 0
for k, f in sorted(vars(m).items()):
    own = getattr(f, '__module__', None) == m.__name__
    if k.startswith('test') and callable(f) and own and not isinstance(f, type):
        n += 1
        try:
            f()
        except Exception:
            bad += 1
            traceback.print_exc()
print('Ran %d tests in 0s' % n, file=sys.stderr)
sys.exit(1 if bad else 0)
"""


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
