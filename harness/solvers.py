"""Per-kind solvers. Each makes as few and as short LLM calls as possible:
the harness gathers context itself, the model only writes the decisive bit,
and the harness executes and checks it."""
from __future__ import annotations

import ast
import hashlib
import os
import re
from typing import Optional

from .llm import LLM, MODELS
from .sandbox import clip, extract_code, run_python, run_shell, run_argv, PY
from .tasks import Task

IMPOSSIBLE = "IMPOSSIBLE"

SYS_CODE = ("You write minimal Python 3 programs. Reply with exactly one ```python``` code block "
            "and nothing else, no comments. The program must print only the final answer (one line). "
            f"If the problem is contradictory or cannot be answered from the given data, "
            f"print {IMPOSSIBLE}.")


class Infeasible(Exception):
    pass


def _answer_line(out: str) -> Optional[str]:
    lines = [x.strip() for x in out.strip().splitlines() if x.strip()]
    return lines[-1] if lines else None


def _num_clean(ans: str) -> str:
    a = ans.strip().strip("'\"")
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


async def _program(task: Task, llm: LLM, user: str, model: str, max_tokens: int = 400) -> str:
    msgs = [{"role": "system", "content": SYS_CODE}, {"role": "user", "content": user}]
    feedback = None
    for attempt in range(2):
        m = msgs if feedback is None else msgs + feedback
        reply = await llm.chat(m, model=model if attempt == 0 else MODELS["fast"],
                               max_tokens=max_tokens, tag=f"{task.id}#{attempt}")
        code = extract_code(reply)
        if not code:
            feedback = [{"role": "assistant", "content": reply},
                        {"role": "user", "content": "Reply with one ```python``` block only."}]
            continue
        r = run_python(PRELUDE.get(task.kind, "") + code, task.cwd, timeout=30)
        ans = _answer_line(r.out) if r.ok else None
        if ans and ans.upper().startswith(IMPOSSIBLE):
            raise Infeasible("model says impossible")
        if r.ok and ans:
            return _num_clean(ans)
        if "ZeroDivisionError" in r.err:
            raise Infeasible("division by zero")
        if task.kind == "data_analysis" and re.search(r"KeyError|FileNotFoundError", r.err):
            raise Infeasible("data lacks the column/file the question needs")
        feedback = [{"role": "assistant", "content": f"```python\n{code}\n```"},
                    {"role": "user", "content": "It failed:\n" + clip(r.err or r.out or "no output", 800)
                     + "\nFix it. One ```python``` block only."}]
    raise RuntimeError("program failed twice")


# ------------------------------------------------------------------ math

async def solve_math(task: Task, llm: LLM) -> Optional[str]:
    return await _program(task, llm, task.question, MODELS["fast"], max_tokens=350)


# ------------------------------------------------------------------ data analysis

# Runs before the model's code, so the model writes one expression instead of csv
# boilerplate (about half the output tokens). Numbers are already converted.
PRELUDE = {"data_analysis": r'''
import csv as _csv, glob as _glob
def _conv(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
tables = {}
for _f in sorted(_glob.glob('*.csv') + _glob.glob('*.tsv')):
    with open(_f, newline='', encoding='utf-8', errors='replace') as _h:
        tables[_f] = [{k: _conv(v) for k, v in r.items()}
                      for r in _csv.DictReader(_h, delimiter='\t' if _f.endswith('.tsv') else ',')]
rows = next(iter(tables.values())) if len(tables) == 1 else None
'''}


async def solve_data(task: Task, llm: LLM) -> Optional[str]:
    note = f"\nNote: {task.context['note']}" if task.context.get("note") else ""
    one = task.context.get("data", "").count("data rows") == 1
    pre = ("Preloaded variables: `tables` = {filename: list of row dicts}, numeric fields already int/float"
           + ("; `rows` = rows of the only table" if one else "")
           + ". Write the shortest code that prints the answer; do not reread files.")
    user = (f"Files in the working directory (summary):\n{task.context.get('data', '')}{note}\n\n"
            f"Task: {task.question}\n{pre}")
    return await _program(task, llm, user, MODELS["fast"], max_tokens=300)


# ------------------------------------------------------------------ search

SYS_SEARCH = ("Answer the question using only the knowledge base. Facts may need several hops "
              "(person -> employer -> company facts). Reply with only the answer value, exactly as "
              f"written in the KB, no sentence. If the KB does not contain it, reply {IMPOSSIBLE}.")


async def solve_search(task: Task, llm: LLM) -> Optional[str]:
    user = f"Knowledge base:\n{task.context.get('kb', '')}\n\nQuestion: {task.question}"
    reply = await llm.chat([{"role": "system", "content": SYS_SEARCH},
                            {"role": "user", "content": user}],
                           model=MODELS["fast"], max_tokens=24, tag=task.id)
    ans = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    ans = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", ans, flags=re.I).strip().strip("*`'\". ")
    if not ans or ans.upper().startswith(IMPOSSIBLE) or ans.lower() in ("unknown", "none", "n/a"):
        raise Infeasible("not in KB")
    return ans


# ------------------------------------------------------------------ coding

SYS_FIX = ("You fix bugs in Python code. Hidden tests check other inputs too, so make each function "
           "correct for what its name and the tests intend, not just for the visible cases. "
           "Reply with exactly one ```python``` block that contains ONLY the top-level functions/classes "
           "you changed (complete definitions, plus any imports they need); omit unchanged ones. "
           "Do not touch tests. "
           f"If the tests contradict each other and cannot all pass, reply {IMPOSSIBLE}.")


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _test_files(cwd: str):
    return sorted(f for f in os.listdir(cwd) if f.endswith(".py") and
                  (f.startswith("test") or f.endswith("_test.py")))


def _run_tests(cwd: str, tests):
    outs, ok = [], True
    for tf in tests:
        r = run_argv([PY, tf], cwd, timeout=60)
        if not r.ok:
            ok = False
        outs.append(f"$ python3 {tf}\n{r.err}{r.out}")
    return ok, "\n".join(outs)


def splice(original: str, patch: str) -> str:
    """Replace top-level defs/classes of `original` by same-named ones in `patch`."""
    try:
        po = ast.parse(original)
        pp = ast.parse(patch)
    except SyntaxError:
        raise
    lines = original.splitlines(keepends=True)
    plines = patch.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    def span(node, src_lines):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        return start, node.end_lineno

    repl = {}
    new_imports, appended = [], []
    for n in pp.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s, e = span(n, plines)
            repl[n.name] = "".join(plines[s:e]).rstrip() + "\n"
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            seg = "".join(plines[n.lineno - 1:n.end_lineno])
            if seg.strip() not in original:
                new_imports.append(seg.rstrip() + "\n")
        elif isinstance(n, ast.Assign):
            seg = "".join(plines[n.lineno - 1:n.end_lineno]).rstrip() + "\n"
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            for nm in names:
                repl["=" + nm] = seg
    edits = []
    seen = set()
    for n in po.body:
        key = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = n.name
        elif isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            key = "=" + names[0] if len(names) == 1 else None
        if key and key in repl:
            s, e = span(n, lines)
            edits.append((s, e, repl[key]))
            seen.add(key)
    for s, e, text in sorted(edits, reverse=True):
        lines[s:e] = [text]
    for k, v in repl.items():
        if k not in seen:
            appended.append("\n\n" + v)
    out = "".join(new_imports) + "".join(lines) + "".join(appended)
    ast.parse(out)
    return out


async def solve_coding(task: Task, llm: LLM) -> Optional[str]:
    cwd = task.cwd
    tests = _test_files(cwd)
    if not tests:
        return await solve_generic(task, llm)
    guard = {t: _sha(os.path.join(cwd, t)) for t in tests}
    backup = {t: open(os.path.join(cwd, t), "rb").read() for t in tests}
    ok, out = _run_tests(cwd, tests)
    if ok:
        return "DONE"
    targets = [f for f in os.listdir(cwd) if f.endswith(".py") and f not in tests]
    m = re.search(r"bugs? in ([\w./-]+\.py)", task.instruction)
    target = m.group(1) if m and m.group(1) in targets else (targets[0] if len(targets) == 1 else None)
    if target is None:
        return await solve_generic(task, llm)
    tpath = os.path.join(cwd, target)
    original = open(tpath, encoding="utf-8").read()
    test_src = "\n\n".join(f"### {t}\n" + open(os.path.join(cwd, t), encoding="utf-8").read()
                           for t in tests)
    base = (f"### {target}\n```python\n{original}\n```\n\n{test_src}\n\n"
            f"Test output:\n{clip(out, 1800)}")
    msgs = [{"role": "system", "content": SYS_FIX}, {"role": "user", "content": base}]
    plan = [(MODELS["fast"], 700), (MODELS["fast"], 900), (MODELS["think"], 3000)]
    current = original
    try:
        for i, (model, mt) in enumerate(plan):
            reply = await llm.chat(msgs, model=model, max_tokens=mt, tag=f"{task.id}#{i}")
            if IMPOSSIBLE in reply and "```" not in reply:
                raise Infeasible("model says tests contradict")
            code = extract_code(reply)
            try:
                current = splice(current, code)
            except SyntaxError as e:
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"SyntaxError: {e}. Resend the corrected functions."}]
                continue
            with open(tpath, "w", encoding="utf-8") as f:
                f.write(current)
            ok, out = _run_tests(cwd, tests)
            for t, h in guard.items():  # tests must stay untouched
                if _sha(os.path.join(cwd, t)) != h:
                    with open(os.path.join(cwd, t), "wb") as f:
                        f.write(backup[t])
                    ok, out = _run_tests(cwd, tests)
            if ok:
                return "DONE"
            # next round: show the current file and the new failures only
            msgs = [{"role": "system", "content": SYS_FIX},
                    {"role": "user", "content": f"### {target}\n```python\n{current}\n```\n\n{test_src}"
                                                f"\n\nTest output:\n{clip(out, 1800)}"}]
        raise RuntimeError("tests still failing")
    finally:
        for t in tests:
            if _sha(os.path.join(cwd, t)) != guard[t]:
                with open(os.path.join(cwd, t), "wb") as f:
                    f.write(backup[t])


# ------------------------------------------------------------------ generic fallback (text ReAct)

SYS_AGENT = ("You are a terse coding agent in a Linux shell. To act, reply with one ```bash``` block; "
             "you will get its output. Keep commands short. When finished, reply with a line "
             "`FINAL: <answer>` (the exact content for answer.txt) and no code block.")


async def solve_generic(task: Task, llm: LLM, max_steps: int = 8) -> Optional[str]:
    msgs = [{"role": "system", "content": SYS_AGENT},
            {"role": "user", "content": f"Working directory files:\n{task.context.get('files') or ''}"
                                        f"\n\nTask: {task.instruction}"}]
    for step in range(max_steps):
        reply = await llm.chat(msgs, model=MODELS["fast"], max_tokens=600, tag=f"{task.id}#{step}")
        fm = re.search(r"FINAL:\s*(.+)", reply)
        cm = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", reply, re.S)
        if fm and not cm:
            ans = fm.group(1).strip()
            if ans.upper().startswith(IMPOSSIBLE):
                raise Infeasible("agent gave up")
            return ans
        if not cm:
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Use a ```bash``` block or FINAL: <answer>."}]
            continue
        r = run_shell(cm.group(1), task.cwd, timeout=60)
        obs = clip(f"exit={r.code}\n{r.out}\n{r.err}", 1500)
        msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": obs}]
        if len(msgs) > 12:  # context control: keep system, task, and the last exchanges
            msgs = msgs[:2] + msgs[-8:]
    # the agent may have written answer.txt itself
    p = os.path.join(task.cwd, "answer.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    raise RuntimeError("no answer")


SOLVERS = {"math": solve_math, "data_analysis": solve_data, "search": solve_search,
           "coding": solve_coding, "generic": solve_generic}
