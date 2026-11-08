"""Offline check of the zero-token solvers: no LLM calls at all.

python tools/check_det.py TASKS_DIR [TASKS_DIR ...]

Works on a throwaway copy of each task set. For every task it reports whether the
deterministic solver answered, and whether that answer is right (coding: visible plus
hidden tests). A wrong deterministic answer is the one outcome that must never happen.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.deterministic import det_solve  # noqa: E402
from harness.tasks import analyze, load_tasks  # noqa: E402
from grade import tests_pass  # noqa: E402


def main():
    stats = collections.defaultdict(collections.Counter)
    wrong = []
    for src in sys.argv[1:]:
        tmp = tempfile.mkdtemp(prefix="detcheck_")
        work = os.path.join(tmp, "tasks")
        shutil.copytree(src, work)
        tasks = load_tasks(work)
        for t in tasks:
            analyze(t)

        def one(t):
            return t, det_solve(t)

        with ThreadPoolExecutor(4) as pool:
            for t, ans in pool.map(one, tasks):
                feasible = t.raw.get("feasible", True) and t.raw.get("expected") is not None
                s = stats[t.kind]
                s["n"] += 1
                s["feasible"] += feasible
                if ans is None:
                    s["to_llm" if feasible else "infeasible_left"] += 1
                    continue
                if not feasible:
                    s["answered_infeasible"] += 1
                    continue
                if t.raw["check"] == "unittest":
                    ok = ans == "DONE" and tests_pass(t.dir)
                else:
                    ok = ans.strip() == str(t.raw["expected"]).strip()
                s["right" if ok else "WRONG"] += 1
                if not ok:
                    wrong.append((src, t.id, ans, t.raw.get("expected")))
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'kind':14s} {'tasks':>5s} {'feasible':>8s} {'det right':>9s} {'det WRONG':>9s} "
          f"{'to LLM':>6s} {'answered infeasible':>19s}")
    for k in sorted(stats):
        s = stats[k]
        print(f"{k:14s} {s['n']:5d} {s['feasible']:8d} {s['right']:9d} {s['WRONG']:9d} "
              f"{s['to_llm']:6d} {s['answered_infeasible']:19d}")
    for w in wrong:
        print("WRONG", *w)
    json.dump({k: dict(v) for k, v in stats.items()}, sys.stdout)
    print()


if __name__ == "__main__":
    main()
