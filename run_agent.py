"""Budget-aware harness entry point.

python run_agent.py --tasks-dir /data/tasks [--out FILE] [--budget N] [--workers 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _writable_cwd():
    """openJiuwen writes ./logs relative to the cwd at import time; the container runs
    with --read-only and --workdir /app, so move to a writable place first."""
    if os.access(os.getcwd(), os.W_OK):
        return
    for d in (os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent"), os.path.expanduser("~/agent")):
        try:
            os.makedirs(d, exist_ok=True)
            os.chdir(d)
            return
        except OSError:
            continue


from harness.llm import LLM, Ledger, BudgetExhausted, ProxyDown, probe_budget  # noqa: E402
from harness.log import log  # noqa: E402
from harness.solvers import SOLVERS, Infeasible  # noqa: E402
from harness.tasks import load_tasks, analyze  # noqa: E402

SKIP_BELOW = 0.1          # p_feasible under this -> do not spend tokens
TIME_LIMIT = float(os.environ.get("TIME_LIMIT_SEC", 3300))  # hard stop is 1h


class Results:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                self.data = json.load(open(path, encoding="utf-8"))
            except ValueError:
                self.data = {}

    def save(self, task, answer: str):
        self.data[task.id] = answer
        _atomic_write(self.path, json.dumps(self.data, ensure_ascii=False, indent=1))
        for d in {task.dir, task.cwd}:
            _atomic_write(os.path.join(d, "answer.txt"), answer + "\n")


def _atomic_write(path: str, text: str):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError as e:
        log(f"[save] cannot write {path}: {e}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass


def resolve_out(arg: str | None, tasks_dir: str) -> str:
    out = arg or os.environ.get("OUT") or ""
    if out and (out.endswith(os.sep) or os.path.isdir(out)):
        out = os.path.join(out, "answers.json")
    if out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "a", encoding="utf-8"):
                pass
            return out
        except OSError as e:
            log(f"[out] {out} not writable ({e}); falling back")
    return os.path.join(tasks_dir, "answers.json")


async def main_async(args):
    t_start = time.time()
    tasks = load_tasks(args.tasks_dir)
    log(f"loaded {len(tasks)} tasks from {args.tasks_dir}")
    for t in tasks:
        analyze(t)
    budget = args.budget or float(os.environ.get("TOKEN_BUDGET") or os.environ.get("BUDGET") or 0) or None
    ledger = Ledger(budget=budget)
    llm = LLM(ledger)
    if os.environ.get("LLM_BASE_URL"):
        probe_budget(os.environ["LLM_BASE_URL"], os.environ.get("LLM_API_KEY", ""))
    results = Results(resolve_out(args.out, args.tasks_dir))
    log(f"budget={budget} out={results.path}")

    todo = [t for t in tasks if t.p_feasible >= SKIP_BELOW and t.reward > 0]
    for t in tasks:
        if t.p_feasible < SKIP_BELOW:
            t.status = "skipped"
            log(f"skip {t.id}: {t.why_infeasible}")
    todo.sort(key=lambda t: t.priority, reverse=True)
    for t in todo:
        log(f"plan {t.id:22s} kind={t.kind:13s} r={t.reward:g} p={t.p_feasible:.2f} est={t.est_cost:.0f}")

    # online calibration: actual / estimated cost per kind
    ratio = {}
    queue = asyncio.Queue()
    for t in todo:
        queue.put_nowait(t)
    sem_workers = max(1, min(args.workers, 5))

    async def worker(wid: int):
        while True:
            try:
                t = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if ledger.dead or time.time() - t_start > TIME_LIMIT:
                t.status = "not_started"
                continue
            est = t.est_cost * ratio.get(t.kind, 1.0)
            if ledger.remaining() < est:
                t.status = "too_expensive"
                log(f"[w{wid}] {t.id}: est {est:.0f} > remaining {ledger.remaining():.0f}")
                continue
            t0 = time.time()
            try:
                ans = await asyncio.wait_for(SOLVERS[t.kind](t, llm), timeout=600)
                ans = (ans or "").strip()
                if ans:
                    t.answer, t.status = ans, "answered"
                    results.save(t, ans)
            except Infeasible as e:
                t.status = f"infeasible: {e}"
            except (BudgetExhausted, ProxyDown) as e:
                t.status = f"stopped: {type(e).__name__}"
            except Exception as e:  # noqa: BLE001 - one task must not kill the run
                t.status = f"error: {type(e).__name__}: {str(e)[:200]}"
            t.spent = ledger.by_task.get(t.id, 0.0)
            if t.spent > 0:
                r = t.spent / max(t.est_cost, 1.0)
                ratio[t.kind] = 0.7 * ratio.get(t.kind, 1.0) + 0.3 * r
            log(f"[w{wid}] {t.id}: {t.status} ans={t.answer!r} cost={t.spent:.0f} "
                f"total={ledger.spent:.0f} {time.time() - t0:.1f}s")

    await asyncio.gather(*(worker(i) for i in range(sem_workers)))

    solved = [t for t in tasks if t.status == "answered"]
    log(f"DONE: answered {len(solved)}/{len(tasks)}, reward claimed "
        f"{sum(t.reward for t in solved):g}, spent {ledger.spent:.0f} "
        f"(in={ledger.in_tok} out={ledger.out_tok} calls={ledger.calls}) "
        f"in {time.time() - t_start:.0f}s")
    summary = [{"id": t.id, "kind": t.kind, "reward": t.reward, "status": t.status,
                "answer": t.answer, "cost": round(t.spent, 1), "est": round(t.est_cost, 1),
                "p_feasible": t.p_feasible, "why": t.why_infeasible} for t in tasks]
    try:
        with open(os.environ.get("SUMMARY_PATH", "/tmp/run_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"spent": ledger.spent, "tasks": summary}, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    args.tasks_dir = os.path.abspath(args.tasks_dir)
    if args.out:
        args.out = os.path.abspath(args.out)
    _writable_cwd()  # before openJiuwen is imported (lazily, inside LLM)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
