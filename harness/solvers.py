"""Per-kind solvers. Each one is an openJiuwen ReAct agent with at most one tool. The
harness gathers the context itself, the model only writes the decisive bit (a short
program, a patch, a value) as a tool call, and the tool executes and checks it."""
from __future__ import annotations

import ast
import hashlib
import os
import re
from typing import Optional

from .llm import LLM, MODELS, NATIVE_TOOLS
from .sandbox import clip, run_python, run_shell, run_argv, PY
from .tasks import Task

IMPOSSIBLE = "IMPOSSIBLE"

def _via(tool_name: str, what: str) -> str:
    """How the model hands over its code: a tool call, or a fenced block for the same tool."""
    if NATIVE_TOOLS:
        return f"Call {tool_name} exactly once with {what}"
    return f"Reply with exactly one ```python``` block and nothing else, containing {what}"


_NO = "do not call the tool: reply" if NATIVE_TOOLS else "reply only"

SYS_CODE = (_via("run_python", "a minimal Python 3 program, no comments") + ". The program must print "
            "only the final answer (one line) and must not write files. If the problem is "
            f"contradictory or cannot be answered from the given data, {_NO} {IMPOSSIBLE}.")


class Infeasible(Exception):
    pass


# Assured mode: the run has far more budget than tasks (the two-task qualifying run, where
# one wrong answer disqualifies the team). Every answer is then confirmed by independent
# solutions, and the task is assumed to be solvable. Set by run_agent.
ASSURE = False
SOLVABLE = "\nThis task is known to be solvable with the given data: always give an answer."


async def _vote(candidates):
    """Run independent solvers until two of them agree; return that answer.
    Without agreement the last answer wins (candidates go from cheap to strong)."""
    from collections import Counter
    answers = []
    for make in candidates:
        try:
            a = await make()
        except Infeasible:
            a = None
        except RuntimeError:
            a = None
        if a:
            answers.append(a)
            top, n = Counter(answers).most_common(1)[0]
            if n >= 2:
                return top
    if answers:
        return answers[-1]
    raise Infeasible("no solver produced an answer")


def _answer_line(out: str) -> Optional[str]:
    lines = [x.strip() for x in out.strip().splitlines() if x.strip()]
    return lines[-1] if lines else None


def _num_clean(ans: str) -> str:
    a = ans.strip().strip("'\"")
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def _budget(text: str, base: int, per_token: float, top: int) -> int:
    """Output cap for one call. A cap is a guard against a model that loops, not a
    target: it has to leave room for a long problem (a nine-step word problem needs a
    longer program than a one-line sum), so it grows with the size of the input."""
    return int(min(top, base + per_token * len(text) / 3.5))


async def _program(task: Task, llm: LLM, user: str, model: str, max_tokens: int = 400,
                   retry_model: str = "", retry_tokens: int = 0, system: str = "") -> str:
    state = {}

    def handler(code: str) -> str:
        r = run_python(PRELUDE.get(task.kind, "") + code, task.cwd, timeout=30)
        state["run"] = r
        return clip(r.out if r.ok else (r.err or r.out or "no output"), 800)

    def fresh():
        return llm.session(task.id, system or SYS_CODE, "run_python", handler)

    agent = fresh()
    cap = max_tokens
    # A deferred (probably infeasible) task gets one try with the normal cap: a model
    # that sees the gap answers IMPOSSIBLE in a few tokens, and if the guess was wrong
    # the task is still solved.
    text, use = user, model
    failures = regrown = 0
    while True:
        state.clear()
        reply = await agent.ask(text, use, cap)
        r = state.get("run")
        ans = _answer_line(r.out) if r is not None and r.ok else None
        if ans and ans.upper().startswith(IMPOSSIBLE):
            raise Infeasible("model says impossible")
        if ans:
            return _num_clean(ans)
        if r is None and IMPOSSIBLE in reply.content.upper():
            raise Infeasible("model says impossible")
        if r is not None and task.kind == "math" and "ZeroDivisionError" in r.err:
            raise Infeasible("division by zero")
        if task.deferred:
            break
        if reply.truncated and regrown < 2:
            # Cut off by the cap, not wrong: ask again from scratch with more room,
            # instead of paying to resend half a program and its syntax error.
            regrown += 1
            cap = min(cap * 3, 16000)
            agent, text = fresh(), user
            continue
        failures += 1
        if failures >= 2:
            break
        use, cap = retry_model or MODELS["fast"], max(cap, retry_tokens)
        if r is None:
            text = _via("run_python", "the program") + "."
        else:
            text = ("It failed. Fix the program and call run_python again." if NATIVE_TOOLS else
                    "It failed:\n" + clip(r.err or r.out or "no output", 800)
                    + "\nFix it. One ```python``` block only.")
    raise RuntimeError("program failed")


# ------------------------------------------------------------------ math
SYS_STEPS = (_via("run_python", "a Python 3 program, no comments") + ". Simulate the problem literally: "
             "one variable per quantity it tracks, and one statement per operation or sentence, in "
             "the stated order, using every number as written. Do not merge, reorder or skip "
             "operations and do not compute anything in your head. Print only the requested value. "
             f"If the problem is contradictory or lacks a number it needs, {_NO} {IMPOSSIBLE}.")


async def solve_math(task: Task, llm: LLM) -> Optional[str]:
    """Two cheap independent solutions, and the reasoning model only if they disagree.

    Only wordings the zero-token solver did not recognise get here. Measured on 71 such
    tasks: each single solver loses or misreads a step in 1-4% of them (the reasoning
    model too), but never the same way, so agreement of two differently prompted cheap
    solvers is a free proof. Disagreement goes to the reasoning model with the stricter
    prompt. Result: 71/71 at 161 tokens per task, against 70/71 at 286 for the reasoning
    model alone."""
    fast_cap = _budget(task.question, 350, 2.5, 4000)
    think_cap = _budget(task.question, 1500, 10, 12000)

    async def attempt(**kw):
        try:
            return await _program(task, llm, task.question, **kw)
        except Infeasible:
            return IMPOSSIBLE
        except RuntimeError:
            return None  # no usable program: counts as a disagreement

    if ASSURE:
        q = task.question + SOLVABLE

        def run(**kw):
            return lambda: _program(task, llm, q, **kw)
        return await _vote([run(model=MODELS["fast"], max_tokens=fast_cap),
                            run(model=MODELS["fast"], max_tokens=fast_cap, system=SYS_STEPS),
                            run(model=MODELS["think"], max_tokens=think_cap, system=SYS_STEPS),
                            run(model=MODELS["think"], max_tokens=think_cap)])
    first = await attempt(model=MODELS["fast"], max_tokens=fast_cap)
    if task.deferred:  # probably infeasible: one cheap try only
        if first in (None, IMPOSSIBLE):
            raise Infeasible("model says impossible")
        return first
    second = await attempt(model=MODELS["fast"], max_tokens=fast_cap, system=SYS_STEPS)
    answer = first if first is not None and first == second else await attempt(
        model=MODELS["think"], max_tokens=think_cap, system=SYS_STEPS,
        retry_model=MODELS["think"], retry_tokens=think_cap)
    if answer is None:
        raise RuntimeError("program failed")
    if answer == IMPOSSIBLE:
        raise Infeasible("model says impossible")
    return answer


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
for _f in sorted(_glob.glob('**/*.csv', recursive=True) + _glob.glob('**/*.tsv', recursive=True)):
    with open(_f, newline='', encoding='utf-8', errors='replace') as _h:
        _head = _h.readline()
        _h.seek(0)
        _delim = max(',;\t|', key=_head.count)
        _delim = _delim if _head.count(_delim) else ','
        tables[_f] = [{k: _conv(v) for k, v in r.items()}
                      for r in _csv.DictReader(_h, delimiter=_delim)]
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
    fast_cap = _budget(task.question, 300, 2.5, 4000)
    think_cap = _budget(task.question, 1500, 10, 12000)
    if ASSURE:
        q = user + SOLVABLE

        def run(**kw):
            return lambda: _program(task, llm, q, **kw)
        return await _vote([run(model=MODELS["fast"], max_tokens=fast_cap),
                            run(model=MODELS["think"], max_tokens=think_cap),
                            run(model=MODELS["think"], max_tokens=think_cap, retry_model=MODELS["think"],
                                retry_tokens=think_cap)])
    return await _program(task, llm, user + " If a value the answer depends on is blank or "
                          f"missing, print {IMPOSSIBLE}.", MODELS["fast"], max_tokens=fast_cap)


# ------------------------------------------------------------------ search

SYS_SEARCH = ("Answer the question using only the knowledge base. Facts may need several hops "
              "(person -> employer -> company facts). Reply with only the answer value, exactly as "
              "written in the KB, no sentence. Apply every condition of the question (day, role, "
              "earliest/latest) before choosing. If the KB does not contain the answer or does not "
              f"single out one value, reply {IMPOSSIBLE}.")


async def solve_search(task: Task, llm: LLM) -> Optional[str]:
    # Standard questions are answered by the zero-token solver. What reaches the model is
    # unfamiliar wording, where the hop estimate behind the narrow context cannot be
    # trusted, so the model gets the wider set of linked cards straight away.
    if ASSURE:
        wide = task.context.get("kb_wide") or task.context.get("kb", "")
        whole = task.context.get("kb_all") or wide

        def run(kb, model, limit):
            return lambda: _search_once(task, llm, kb, model, limit, SOLVABLE)
        cap = _budget(whole + task.question, 1500, 4, 12000)
        return await _vote([run(wide, MODELS["think"], cap), run(whole, MODELS["think"], cap),
                            run(whole, MODELS["fast"], 40), run(whole, MODELS["think"], cap * 2)])
    contexts = [task.context.get("kb_wide") or task.context.get("kb", "")]
    # The model without reasoning answers lookups well but picks the wrong card when the
    # question needs filtering or comparing, and guesses when the KB is silent. Unfamiliar
    # questions therefore go to the reasoning model; a probably-infeasible one stays cheap.
    model, limit = (MODELS["fast"], 24) if task.deferred else (
        MODELS["think"], _budget(contexts[0] + task.question, 1200, 4, 12000))
    for kb in contexts:
        try:
            return await _search_once(task, llm, kb, model, limit)
        except Infeasible:
            pass
    raise Infeasible("not in KB")


async def _search_once(task: Task, llm: LLM, kb: str, model: str, limit: int, note: str = "") -> str:
    user = f"Knowledge base:\n{kb}\n\nQuestion: {task.question}{note}"
    got = await llm.session(task.id, SYS_SEARCH).ask(user, model, limit)
    if got.truncated and model == MODELS["think"] and not got.content.strip():
        # the reasoning ran into the cap before any answer: once more with room
        got = await llm.session(task.id, SYS_SEARCH).ask(user, model, min(limit * 3, 16000))
    reply = got.content
    ans = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    ans = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", ans, flags=re.I).strip().strip("*`'\". ")
    if ans and not ans.upper().startswith(IMPOSSIBLE) and ans.lower() not in ("unknown", "none", "n/a"):
        return kb_spelling(ans, kb)
    raise Infeasible("not in KB")


def kb_spelling(ans: str, kb: str) -> str:
    """The check is an exact string match, so return the value as the KB spells it:
    'osaka' -> 'Osaka', 'The city is Osaka' -> 'Osaka'. Unknown values pass through."""
    values = {}
    for line in kb.splitlines():
        m = re.match(r"^(?:#\s*|[\w ]+:\s*)(.+?)\s*$", line)
        if m and not line.startswith("---"):
            values.setdefault(m.group(1).lower(), m.group(1))
    if ans.lower() in values:
        return values[ans.lower()]
    inside = [v for k, v in values.items()
              if re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", ans.lower())]
    longest = max(inside, key=len, default=None)
    # a sentence around exactly one KB value (shorter hits inside it do not count)
    if longest and all(v.lower() in longest.lower() for v in inside) and len(ans.split()) > len(longest.split()):
        return longest
    return ans


# ------------------------------------------------------------------ coding

SYS_FIX = ("You fix bugs in Python code. Hidden tests check other inputs too, so make each function "
           "correct for what its name and the tests intend, not just for the visible cases. "
           + _via("apply_fix", "ONLY the top-level functions/classes you changed (complete definitions, "
                  "plus any imports they need); omit unchanged ones") + ". Do not touch tests. "
           "Never create or edit data files. If the tests cannot pass by editing this code alone "
           f"(they contradict each other, or need a file that is missing), {_NO} {IMPOSSIBLE}.")


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


def _compact(target: str, src: str, cwd: str, tests, out: str) -> Optional[str]:
    """Prompt with only the failing tests and the functions they call. Used only when
    that view is complete: every function of the file is covered by some test (so a bug
    cannot hide in code no test names) and the module has no other top-level state."""
    import textwrap
    from .deterministic import _FAILED, _test_map
    failed = set(_FAILED.findall(out))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    defs = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[n.name] = n
        elif not isinstance(n, (ast.Import, ast.ImportFrom)):
            return None  # classes, constants, module code: show the whole file
    uses, methods = {}, {}
    for tf in tests:
        with open(os.path.join(cwd, tf), encoding="utf-8") as f:
            tsrc = f.read()
        u, _ = _test_map(tsrc, os.path.splitext(target)[0])
        uses.update(u)
        for n in ast.walk(ast.parse(tsrc)):
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test"):
                methods[n.name] = textwrap.dedent(ast.get_source_segment(tsrc, n, padded=True))
    covered = set().union(*uses.values()) if uses else set()
    if not failed or not set(defs) <= covered:
        return None
    need = set()
    for t in failed:
        hit = uses.get(t, set()) & set(defs)
        if not hit or t not in methods:
            return None
        need |= hit
    while True:  # helpers the failing functions call
        more = {x.id for f in need for x in ast.walk(defs[f])
                if isinstance(x, ast.Name) and x.id in defs} - need
        if not more:
            break
        need |= more
    errors = []
    for block in re.split(r"^={20,}\s*$", out, flags=re.M):
        m = _FAILED.search(block)
        e = re.search(r"^\w*(?:Error|Exception)\b.*$", block, re.M)
        if m and e:
            errors.append(f"{m.group(1)}: {e.group(0).strip()[:200]}")
    imports = [ast.get_source_segment(src, n) for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    funcs = [ast.get_source_segment(src, defs[f]) for f in defs if f in need]
    return ("### " + target + " (the failing functions)\n```python\n"
            + "".join(i + "\n" for i in imports) + "\n\n\n".join(funcs) + "\n```\n\n"
            "### failing tests\n" + "\n\n".join(methods[t] for t in sorted(failed))
            + "\n\n" + "\n".join(errors))


def _fail_signature(out: str):
    """Which tests fail and with what error, ignoring line numbers and timings."""
    return tuple(sorted(set(re.findall(
        r"^(?:FAIL|ERROR): \w+|^\w*(?:Error|Exception)\b.*$", out, re.M))))


def _coding_prompt(target: str, src: str, cwd: str, tests, out: str) -> str:
    short = _compact(target, src, cwd, tests, out)
    if short:
        return short
    parts = []
    for t in tests:
        with open(os.path.join(cwd, t), encoding="utf-8") as f:
            parts.append(f"### {t}\n" + f.read())
    test_src = "\n\n".join(parts)
    return f"### {target}\n```python\n{src}\n```\n\n{test_src}\n\nTest output:\n{clip(out, 1800)}"


# The graders run their own, larger test file. After the visible tests pass, one more
# agent reviews the whole module for bugs the visible tests do not catch.
AUDIT = os.environ.get("CODE_AUDIT", "1") != "0"
AUDIT_MODEL = os.environ.get("AUDIT_MODEL", "fast")

SYS_AUDIT = ("You review a Python module whose visible tests already pass. Hidden tests check every "
             "function on more inputs: other sizes and orders, zero and negative numbers, duplicates, "
             "mixed case, extra whitespace. A function's name, docstring, the task text and its test "
             "define what it must do in general; a test shows one example, not the whole contract. "
             "Find functions that return a wrong result for some valid input (for example a hard-coded "
             "size, a missing case, a wrong operator). "
             + _via("apply_fix", "corrected definitions of ONLY those functions") + ". Keep correct "
             "functions untouched and do not add input validation. If every function is already "
             f"correct, {_NO} OK.")


async def _audit(task: Task, llm: LLM, cwd: str, target: str, tests, model: str) -> None:
    """Review the fixed module for bugs the visible tests miss. A proposed change is kept
    only if the visible tests still pass; anything else leaves the file as it was."""
    tpath = os.path.join(cwd, target)
    with open(tpath, encoding="utf-8") as f:
        src = f.read()
    guard = {t: _sha(os.path.join(cwd, t)) for t in tests}

    def handler(code: str) -> str:
        try:
            patched = splice(src, code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        try:
            with open(tpath, "w", encoding="utf-8") as f:
                f.write(patched)
            ok, _ = _run_tests(cwd, tests)
            if ok and all(_sha(os.path.join(cwd, t)) == h for t, h in guard.items()):
                return "accepted"
        except Exception:
            pass
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(src)
        return "rejected: the visible tests fail with this change"

    parts = []
    for t in tests:
        with open(os.path.join(cwd, t), encoding="utf-8") as f:
            parts.append(f"### {t}\n" + f.read())
    prompt = (f"Task text: {task.question}\n\n### {target}\n```python\n{src}\n```\n\n" + "\n\n".join(parts))
    cap = _budget(src, 3000, 6, 16000) if model == "think" else _budget(src, 600, 1.5, 4000)
    await llm.session(task.id, SYS_AUDIT, "apply_fix", handler).ask(prompt, MODELS[model], cap)


async def solve_coding(task: Task, llm: LLM) -> Optional[str]:
    cwd = task.cwd
    tests = _test_files(cwd)
    if not tests:
        return await solve_generic(task, llm)
    targets = [f for f in os.listdir(cwd) if f.endswith(".py") and f not in tests]
    m = re.search(r"bugs? in ([\w./-]+\.py)", task.instruction)
    target = m.group(1) if m and m.group(1) in targets else (targets[0] if len(targets) == 1 else None)
    answer = await _fix_until_tests_pass(task, llm, cwd, tests, target)
    if answer == "DONE" and AUDIT and target and not task.deferred:
        # Measured on 40 tasks with a hidden second bug and 40 without: the cheap model
        # finds 36 of 40 at ~80 tokens per task and breaks nothing; the reasoning model
        # finds 39 but costs 540-1070 per task, so it reviews only in assured mode.
        for model in ([AUDIT_MODEL, "think"] if ASSURE else [AUDIT_MODEL]):
            try:
                await _audit(task, llm, cwd, target, tests, model)
            except Exception:
                pass  # visible tests pass: the answer stands whatever happened to the review
    return answer


async def _fix_until_tests_pass(task: Task, llm: LLM, cwd: str, tests, target) -> Optional[str]:
    guard = {t: _sha(os.path.join(cwd, t)) for t in tests}
    backup = {t: open(os.path.join(cwd, t), "rb").read() for t in tests}
    ok, out = _run_tests(cwd, tests)
    if ok:
        return "DONE"
    if target is None:
        return await solve_generic(task, llm)
    tpath = os.path.join(cwd, target)
    original = open(tpath, encoding="utf-8").read()
    fast_cap = _budget(original, 700, 1.5, 5000)  # a patch is at most about the file itself
    plan = [(MODELS["fast"], fast_cap)] + ([] if task.deferred else [(MODELS["fast"], fast_cap)])
    if os.environ.get("CODING_ESCALATE", "1") != "0" and not task.deferred:
        plan.append((MODELS["think"], _budget(original, 3000, 6, 16000)))
    state = {"current": original, "ok": False, "out": out}

    def handler(code: str) -> str:
        try:
            patched = splice(state["current"], code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        state["current"] = patched
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(patched)
        state["ok"], state["out"] = _run_tests(cwd, tests)
        for t, h in guard.items():  # tests must stay untouched
            if _sha(os.path.join(cwd, t)) != h:
                with open(os.path.join(cwd, t), "wb") as f:
                    f.write(backup[t])
                state["ok"], state["out"] = _run_tests(cwd, tests)
        return "all tests pass" if state["ok"] else "tests still fail"

    try:
        sig0 = _fail_signature(out)
        for model, mt in plan:
            if model == MODELS["think"] and _fail_signature(state["out"]) == sig0:
                # two patches changed nothing about how the tests fail: the cause is not
                # in this file, and the expensive model would only burn budget on it
                raise Infeasible("patches do not change the failure")
            # every round is a fresh agent: it sees the current file and current failures only
            agent = llm.session(task.id, SYS_FIX, "apply_fix", handler)
            reply = await agent.ask(_coding_prompt(target, state["current"], cwd, tests, state["out"]),
                                    model, mt)
            if reply.truncated and not state["ok"] and not task.deferred:
                # the patch was cut off by the cap: same request again with more room
                agent = llm.session(task.id, SYS_FIX, "apply_fix", handler)
                reply = await agent.ask(
                    _coding_prompt(target, state["current"], cwd, tests, state["out"]),
                    model, min(mt * 3, 16000))
            if not reply.called and IMPOSSIBLE in reply.content:
                raise Infeasible("model says tests contradict")
            if state["ok"]:
                return "DONE"
        raise RuntimeError("tests still failing")
    finally:
        for t in tests:
            if _sha(os.path.join(cwd, t)) != guard[t]:
                with open(os.path.join(cwd, t), "wb") as f:
                    f.write(backup[t])


# ------------------------------------------------------------------ generic fallback (text ReAct)

SYS_AGENT = ("You are a terse coding agent in a Linux shell. Use the bash tool to inspect files and "
             "compute; keep commands short. When you know the answer, stop calling tools and reply "
             "with one line `FINAL: <answer>` (the exact content for answer.txt). If the task cannot "
             f"be solved with the given data, reply `FINAL: {IMPOSSIBLE}`.")


async def solve_generic(task: Task, llm: LLM, max_steps: int = 8) -> Optional[str]:
    def handler(command: str) -> str:
        r = run_shell(command, task.cwd, timeout=60)
        return clip(f"exit={r.code}\n{r.out}\n{r.err}", 1500)

    agent = llm.session(task.id, SYS_AGENT, "bash", handler, max_iterations=max_steps)
    reply = await agent.ask(f"Working directory files:\n{task.context.get('files') or ''}"
                            f"\n\nTask: {task.instruction}", MODELS["fast"], 600)
    fm = re.search(r"FINAL:\s*(.+)", reply.content)
    if fm:
        ans = fm.group(1).strip()
        if ans.upper().startswith(IMPOSSIBLE):
            raise Infeasible("agent gave up")
        return ans
    # the agent may have written answer.txt itself
    p = os.path.join(task.cwd, "answer.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    raise RuntimeError("no answer")


SOLVERS = {"math": solve_math, "data_analysis": solve_data, "search": solve_search,
           "coding": solve_coding, "generic": solve_generic}
