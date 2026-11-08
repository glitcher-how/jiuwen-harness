"""Compare models per task kind on the same generated task set.

python tools/bench.py TASKS_DIR WORK_DIR model1 [model2 ...] [--workers 3]

For each model: fresh copy of the tasks, run_agent.py with FAST_MODEL=<model> and no
escalation to the reasoning model (so each model is measured on its own), then
per-kind accuracy, first-attempt accuracy, cost and latency from the trajectories.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import read_answer, tests_pass  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def correct(d):
    t = json.load(open(os.path.join(d, "task.json"), encoding="utf-8"))
    ans = read_answer(d)
    if ans is None or t.get("expected") is None:
        return t, False
    if t["check"] == "unittest":
        return t, ans == "DONE" and tests_pass(d)
    return t, ans.strip() == str(t["expected"]).strip()


def run_model(tasks, work, model, workers):
    safe = model.replace("/", "_")
    d = os.path.join(work, safe)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(tasks, d)
    traj = os.path.join(work, f"{safe}.traj.jsonl")
    summ = os.path.join(work, f"{safe}.summary.json")
    for p in (traj, summ):
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ, FAST_MODEL=model, CODING_ESCALATE="0",
               TRAJ_PATH=traj, SUMMARY_PATH=summ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(os.path.join(work, f"{safe}.log"), "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, os.path.join(ROOT, "run_agent.py"), "--tasks-dir", d,
                        "--workers", str(workers)], env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0

    calls = [json.loads(x) for x in open(traj, encoding="utf-8")] if os.path.exists(traj) else []
    by_task = collections.defaultdict(list)
    for c in calls:
        by_task[c["tag"].split("#")[0]].append(c)

    stats = collections.defaultdict(lambda: collections.Counter())
    lat = collections.defaultdict(list)
    for name in sorted(os.listdir(d)):
        td = os.path.join(d, name)
        if not os.path.isfile(os.path.join(td, "task.json")):
            continue
        t, ok = correct(td)
        k = t["task_type"]
        cs = by_task.get(t["task_id"], [])
        s = stats[k]
        s["n"] += 1
        s["ok"] += ok
        s["ok_first"] += ok and len(cs) <= 1
        s["calls"] += len(cs)
        s["cost"] += sum(c["cost"] for c in cs)
        s["out"] += sum(c["out"] for c in cs)
        lat[k] += [c.get("sec", 0) for c in cs]
    return {"model": model, "wall": wall, "stats": stats, "lat": lat}


def report(results):
    kinds = sorted({k for r in results for k in r["stats"]})
    print(f"\n{'kind':14s} {'model':26s} {'acc':>6s} {'1st try':>8s} {'calls/t':>8s} "
          f"{'cost/t':>7s} {'out/t':>6s} {'sec/call':>8s}")
    for k in kinds + ["ALL"]:
        for r in results:
            if k == "ALL":
                s = sum((r["stats"][x] for x in kinds), collections.Counter())
                lat = [v for x in kinds for v in r["lat"][x]]
            else:
                s, lat = r["stats"][k], r["lat"][k]
            n = max(s["n"], 1)
            print(f"{k:14s} {r['model']:26s} {s['ok'] / n:6.1%} {s['ok_first'] / n:8.1%} "
                  f"{s['calls'] / n:8.2f} {s['cost'] / n:7.1f} {s['out'] / n:6.1f} "
                  f"{(sum(lat) / len(lat) if lat else 0):8.2f}")
        print()
    for r in results:
        print(f"{r['model']}: wall {r['wall']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("work")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    results = [run_model(a.tasks, a.work, m, a.workers) for m in a.models]
    report(results)
    json.dump([{"model": r["model"], "wall": r["wall"],
                "stats": {k: dict(v) for k, v in r["stats"].items()}} for r in results],
              open(os.path.join(a.work, "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
