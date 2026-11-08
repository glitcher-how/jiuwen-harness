"""Score a finished run on a generated (or sample) task set.

python tools/grade.py TASKS_DIR [SUMMARY_JSON]
Coding tasks are checked with the visible tests plus hidden/test_hidden.py if present.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile


def read_answer(d):
    for p in (os.path.join(d, "answer.txt"), os.path.join(d, "workspace", "answer.txt")):
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
    return None


def tests_pass(d):
    ws = os.path.join(d, "workspace")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(ws, tmp, dirs_exist_ok=True)
        hidden = os.path.join(d, "hidden", "test_hidden.py")
        files = [f for f in os.listdir(tmp) if f.startswith("test") and f.endswith(".py")]
        if os.path.exists(hidden):
            shutil.copy(hidden, tmp)
            files.append("test_hidden.py")
        for f in files:
            r = subprocess.run([sys.executable, f], cwd=tmp, capture_output=True, timeout=60)
            if r.returncode != 0:
                return False
    return True


def main():
    root = sys.argv[1]
    summ = {}
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        summ = {t["id"]: t for t in json.load(open(sys.argv[2], encoding="utf-8"))["tasks"]}
    got = tot = 0
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        tj = os.path.join(d, "task.json")
        if not os.path.isfile(tj):
            continue
        t = json.load(open(tj, encoding="utf-8"))
        feasible = t.get("feasible", True) and t.get("expected") is not None
        ans = read_answer(d)
        ok = False
        if feasible and ans is not None:
            if t["check"] == "unittest":
                ok = ans == "DONE" and tests_pass(d)
            else:
                ok = ans.strip() == str(t["expected"]).strip()
        tot += t["reward"] if feasible else 0
        got += t["reward"] if ok else 0
        s = summ.get(t["task_id"], {})
        rows.append((t["task_id"], feasible, ok, ans, t.get("expected"), s.get("status", ""), s.get("cost", "")))
    for r in rows:
        flag = "OK " if r[2] else ("-- " if not r[1] else "XX ")
        if not r[2]:
            print(f"{flag}{r[0]:22s} feasible={r[1]!s:5} ans={r[3]!r:14} exp={r[4]!r:14} {r[5]} cost={r[6]}")
    spent = sum(float(r[6] or 0) for r in rows)
    print(f"score {got}/{tot}  ok={sum(r[2] for r in rows)}/{sum(r[1] for r in rows)} feasible, "
          f"wasted-on-infeasible={sum(float(r[6] or 0) for r in rows if not r[1]):.0f}, total cost={spent:.0f}")


if __name__ == "__main__":
    main()
