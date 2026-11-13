"""Per-kind solvers. Each one is an openJiuwen ReAct agent with at most one tool. The
harness gathers the context itself, the model only writes the decisive bit (a short
program, a patch, a value) as a tool call, and the tool executes and checks it."""
from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import tempfile
import threading
from typing import Optional

from .llm import LLM, MODELS, NATIVE_TOOLS, BudgetExhausted, ProxyDown
from .sandbox import Run, clip, run_python, run_shell, run_test_file
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


async def _vote(candidates, prior=None):
    """Run independent solvers until two of them agree; return that answer.
    Without agreement the last answer wins (candidates go from cheap to strong).
    `prior` is the answer of the zero-token solver: it counts as one voice."""
    from collections import Counter
    answers = [prior] if prior else []
    for make in candidates:
        try:
            a = await make()
        except Infeasible:
            a = None
        except RuntimeError:
            a = None
        except (BudgetExhausted, ProxyDown):
            # out of budget in the middle of the vote: an unconfirmed answer is still
            # worth more than none
            if answers:
                break
            raise
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
        if task.kind == "data_analysis" and not re.search(r"\b(?:rows|tables)\b|\bopen\s*\(|read_csv|glob", code):
            # the model saw three sample rows and typed the rest of the table from imagination
            r = Run(1, "", "The program does not read the data. Use the preloaded `tables`; "
                           "never type table contents into the program.")
        else:
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
        # a reply cut off inside the code block is half a program: whatever it printed
        # is not the answer, even if it ran
        cut = reply.truncated and reply.content.count("```") % 2 == 1
        if ans and not cut:
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
                            run(model=MODELS["think"], max_tokens=think_cap)], task.prior)
    first = await attempt(model=MODELS["fast"], max_tokens=fast_cap)
    if task.deferred:  # probably infeasible: one cheap try only
        if first in (None, IMPOSSIBLE):
            raise Infeasible("model says impossible")
        return first
    second = None
    try:
        second = await attempt(model=MODELS["fast"], max_tokens=fast_cap, system=SYS_STEPS)
        answer = first if first is not None and first == second else await attempt(
            model=MODELS["think"], max_tokens=think_cap, system=SYS_STEPS,
            retry_model=MODELS["think"], retry_tokens=think_cap)
        seen = [a for a in (first, second) if a is not None]
        if answer is not None and len(seen) == 2 and answer not in seen:
            # three solutions, three answers: the arbiter is as likely wrong as the others.
            # One more reasoning solution with the other prompt; an answer given twice wins.
            fourth = await attempt(model=MODELS["think"], max_tokens=think_cap)
            if fourth in seen:
                answer = fourth
    except (BudgetExhausted, ProxyDown):
        # out of budget before the answer was confirmed: an unconfirmed answer can
        # still score, no answer cannot
        answer = next((a for a in (first, second) if a not in (None, IMPOSSIBLE)), None)
        if answer is None:
            raise
    if answer in (None, IMPOSSIBLE):
        # A wrong answer costs nothing more than no answer, and the tokens are spent
        # already: if any solution produced a value, the value is submitted even when
        # the others called the problem unsolvable (they misread it as often as not).
        answer = next((a for a in (first, second) if a not in (None, IMPOSSIBLE)), answer)
    if answer is None:
        raise RuntimeError("program failed")
    if answer == IMPOSSIBLE:
        raise Infeasible("model says impossible")
    return answer


# ------------------------------------------------------------------ data analysis

# Runs before the model's code, so the model writes one expression instead of csv
# boilerplate (about half the output tokens). Numbers are already converted.
# 1.1 * 50 is 55.00000000000001 in binary floating point, so math.ceil gives 56, and
# 0.35 * 180 is 62.99999999999999, so math.floor gives 62. "ceil(110% of the balance)" is
# an ordinary step of these problems; a float a few units in the last place away from an
# integer is that integer.
# round() is replaced by rounding half away from zero for the same reason.
_EXACT = r'''
import math as _m, builtins as _bi
from decimal import Decimal as _D, ROUND_HALF_UP as _HU
def _near(f):
    def g(x):
        # only a float that differs from an integer by rounding error of the arithmetic
        # (a few units in the last place) is snapped; 1/2000000000 stays what it is
        if type(x) is float and x == x and abs(x) < 1e15:
            r = _bi.round(x)
            if abs(x - r) <= 64 * _m.ulp(max(abs(r), 1.0)):
                return r
        return f(x)
    return g
_m.floor, _m.ceil, _m.trunc = _near(_m.floor), _near(_m.ceil), _near(_m.trunc)
def round(x, n=None):
    # Python rounds halves to the even neighbour: round(20.5) is 20 and round(2.5) is 2.
    # The problems mean school rounding ("an exact midpoint goes up"), and the programs
    # written for them call round() expecting exactly that. Anything unusual (other
    # types, huge values, odd digit counts) keeps the built-in behaviour.
    try:
        if type(x) in (int, float) and x == x and abs(x) < 1e15 \
                and (n is None or (type(n) is int and -15 <= n <= 15)):
            d = _D(repr(x)).quantize(_D(1).scaleb(-(n or 0)), rounding=_HU)
            return int(d) if n is None or type(x) is int else float(d)
    except Exception:
        pass
    return _bi.round(x) if n is None else _bi.round(x, n)
'''

PRELUDE = {"math": _EXACT, "data_analysis": _EXACT + r'''
import csv as _csv, glob as _glob, re as _re
def _conv(v):
    s = v.strip() if isinstance(v, str) else ''
    if _re.fullmatch(r'-?(0|[1-9]\d*)', s):
        return int(s)
    if _re.fullmatch(r'-?(0|[1-9]\d*)\.\d+', s):
        return float(s)
    return v  # '007', '1e3', 'nan' are codes and words, not numbers
class _Cell:
    # A numeric cell is a number, but programs written for text cells call .strip() or
    # .lower() on it first ("trim every cell"); those calls get the original text back.
    def strip(self, *a): return self._s.strip(*a)
    def lstrip(self, *a): return self._s.lstrip(*a)
    def rstrip(self, *a): return self._s.rstrip(*a)
    def lower(self): return self._s.lower()
    def upper(self): return self._s.upper()
    def replace(self, *a): return self._s.replace(*a)
    def split(self, *a): return self._s.split(*a)
    def startswith(self, *a): return self._s.startswith(*a)
    def endswith(self, *a): return self._s.endswith(*a)
    def isdigit(self): return self._s.strip().isdigit()
class _Int(_Cell, int):
    def __new__(cls, v, s):
        o = int.__new__(cls, v); o._s = s; return o
class _Float(_Cell, float):
    def __new__(cls, v, s):
        o = float.__new__(cls, v); o._s = s; return o
def _cell(v):
    c = _conv(v)
    return _Int(c, v) if type(c) is int else _Float(c, v) if type(c) is float else c
class _Tables(dict):
    # 'data/x.csv', 'data\\x.csv', './data/x.csv', 'workspace/data/x.csv' and a unique 'x.csv' are one file
    def _key(self, k):
        k = _re.sub(r'^(?:[.]/)?(?:workspace/)?', '', str(k).replace(chr(92), '/'))
        if dict.__contains__(self, k):
            return k
        hits = [x for x in self if x.rsplit('/', 1)[-1] == k.rsplit('/', 1)[-1]]
        return hits[0] if len(hits) == 1 else k
    def __getitem__(self, k): return dict.__getitem__(self, self._key(k))
    def __contains__(self, k): return dict.__contains__(self, self._key(k))
    def get(self, k, d=None): return dict.get(self, self._key(k), d)
_raw, _js = {}, {}
for _f in sorted(_glob.glob('**/*.csv', recursive=True) + _glob.glob('**/*.tsv', recursive=True)):
    with open(_f, newline='', encoding='utf-8', errors='replace') as _h:
        _head = _h.readline()
        _h.seek(0)
        _delim = max(',;\t|', key=_head.count)
        _delim = _delim if _head.count(_delim) else ','
        _raw[_f.replace(chr(92), '/')] = list(_csv.DictReader(_h, delimiter=_delim))
for _f in sorted(_glob.glob('**/*.json', recursive=True)):
    try:
        import json as _json
        with open(_f, encoding='utf-8') as _h:
            _d = _json.load(_h)
        if isinstance(_d, list) and _d and all(isinstance(_r, dict) for _r in _d):
            _js[_f.replace(chr(92), '/')] = _d
    except Exception:
        pass
# A column is a column of codes, and stays text as a whole, if one of its values has a
# leading zero (007 is not 7) or if every one of its values is spelled as a string in a JSON file (the key
# of a join: "7" there must equal the cell here). The same column name in another table
# follows, so that a join between two tables compares like with like.
_jstr = {_v.strip() for _t in _js.values() for _r in _t for _v in _r.values() if isinstance(_v, str)}
_cols = {}
for _t in _raw.values():
    for _r in _t:
        for _k, _v in _r.items():
            if isinstance(_v, str) and _v.strip():
                _cols.setdefault(_k, set()).add(_v.strip())
_jkeys = {_k for _t in _js.values() for _r in _t for _k, _v in _r.items() if isinstance(_v, str)}
_text = {_k for _k, _vs in _cols.items()
         if any(_re.fullmatch(r'-?0\d+', _v) for _v in _vs)
         or (_vs <= _jstr and (len(_vs) > 1 or _k in _jkeys))}
tables = _Tables()
for _f, _t in _raw.items():
    tables[_f] = [{k: (v if k in _text else _cell(v)) for k, v in r.items()} for r in _t]
for _f, _t in _js.items():
    tables[_f] = _t
rows = next(iter(tables.values())) if len(tables) == 1 and not _glob.glob('**/*.json', recursive=True) else None
import builtins as _b, os as _os
def open(f, *a, **k):  # the task says "workspace/x.csv", the program already runs inside workspace
    if isinstance(f, str) and not _os.path.exists(f):
        g = _re.sub(r'^(?:[.]/)?workspace/', '', f.replace(chr(92), '/'))
        if _os.path.exists(g):
            f = g
    return _b.open(f, *a, **k)
'''}


async def solve_data(task: Task, llm: LLM) -> Optional[str]:
    note = f"\nNote: {task.context['note']}" if task.context.get("note") else ""
    found = [f for _, _, fs in os.walk(task.cwd) for f in fs if f.lower().endswith((".csv", ".tsv", ".json"))]
    one = len(found) == 1 and not found[0].lower().endswith(".json")  # same rule as `rows` in PRELUDE
    pre = ("Preloaded variables: `tables` = {filename: list of row dicts}, plain numbers already int/float "
           "(a column of codes like 007 stays text)"
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
                                retry_tokens=think_cap)], task.prior)
    # A first program that crashed means the table is not what the cheap model assumed;
    # the second attempt goes to the reasoning model instead of repeating the guess.
    tail = f" If a value the answer depends on is blank or missing, print {IMPOSSIBLE}."
    q = user + tail
    try:
        first = await _program(task, llm, q, MODELS["fast"], max_tokens=fast_cap,
                               retry_model=MODELS["think"], retry_tokens=think_cap)
    except Infeasible:
        if task.deferred:
            raise
        # The cheap model gives up too easily on a messy table (an empty cell the task
        # itself tells how to read). Nothing else pointed at an unsolvable task, so
        # "impossible" is accepted only if the reasoning model, asked from scratch, agrees.
        return await _program(task, llm, q, MODELS["think"], max_tokens=think_cap)
    if task.deferred:
        return first
    # One cheap program is right most of the time, and wrong in ways nothing flags: a
    # one-liner that takes max minus min instead of the largest step between neighbours.
    # As in math, a second cheap solution written under a different instruction is the
    # check, and the reasoning model is called only when the two disagree.
    q2 = user.replace("Write the shortest code that prints the answer; do not reread files.",
                      "Write a plain step-by-step program: one statement per rule of the task, in the "
                      "stated order, with explicit loops; print only the answer. Use the preloaded "
                      "tables.") + tail
    try:
        try:
            second = await _program(task, llm, q2, MODELS["fast"], max_tokens=fast_cap)
        except Infeasible:
            second = IMPOSSIBLE
        except RuntimeError:
            return first  # a crash of the checker says nothing about the answer
        if _same_value(first, second):
            return first
        try:
            return await _program(task, llm, q, MODELS["think"], max_tokens=think_cap)
        except (RuntimeError, Infeasible):
            return first  # a value in hand beats "impossible": a wrong answer is not penalised
    except (BudgetExhausted, ProxyDown):
        return first


def _same_value(a: str, b: str) -> bool:
    if a.strip() == b.strip():
        return True
    try:
        from decimal import Decimal, InvalidOperation
        return Decimal(a.strip()) == Decimal(b.strip())  # exact: float would merge 2**53 + 1 with 2**53
    except (InvalidOperation, ValueError):
        return False


# ------------------------------------------------------------------ search

SYS_SEARCH = ("Answer the question using only the knowledge base. Facts may need several hops "
              "(person -> employer -> company facts). Reply with only the answer value, exactly as "
              "written in the KB, no sentence. Apply every condition of the question (day, role, "
              "earliest/latest) before choosing. If the KB does not contain the answer or does not "
              f"single out one value, reply {IMPOSSIBLE}.")


_WHOLE_BASE = re.compile(r"\b(how many|number of|count|total|all|every|each|any|most|least|fewest|"
                         r"oldest|youngest|earliest|latest|newest|largest|smallest|highest|lowest|"
                         r"longest|shortest|more than|fewer than|less than|at least|at most|"
                         r"which (?:person|people|employee|employees|company|companies|team|teams)|"
                         r"who (?:works|work|is employed|reports|leads|manages|mentors))\b", re.I)


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
                            run(whole, MODELS["fast"], 40), run(whole, MODELS["think"], cap * 2)],
                           task.prior)
    contexts = [task.context.get("kb_wide") or task.context.get("kb", "")]
    whole = task.context.get("kb_all")
    if whole and whole != contexts[0]:
        # Linked cards are found by walking forward from the names in the question.
        # Counts, extremes and reverse links ("who works at X") need cards nobody
        # points to, and "not in these cards" is not "not in the base".
        if _WHOLE_BASE.search(task.question):
            contexts = [whole]
        elif len(whole) <= 20000 and not task.deferred:
            contexts.append(whole)
    # The model without reasoning answers lookups well but picks the wrong card when the
    # question needs filtering or comparing, and guesses when the KB is silent. Unfamiliar
    # questions therefore go to the reasoning model. A task marked doubtful by its wording
    # gets the same model: the mark is a guess (a capitalised field name looks like an
    # unknown entity), and a cheap wrong answer costs the whole task.
    model = MODELS["think"]
    for kb in contexts:
        limit = _budget(kb + task.question, 1200, 4, 12000)
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
    # the cleaning above strips a final period; "Pier-7 / East." is the value, not a sentence
    tail = [v for k, v in values.items() if k.rstrip(".!?;:, ") == ans.lower().rstrip(".!?;:, ")]
    if len(tail) == 1:
        return tail[0]
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
        r = run_test_file(tf, cwd, timeout=60)
        if not r.ok:
            ok = False
        outs.append(f"$ python3 {tf}\n{r.err}{r.out}")
    return ok, "\n".join(outs)


def _assigned(n) -> list:
    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
    return [t.id for t in targets if isinstance(t, ast.Name)]


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
    have_imports = {ast.dump(n) for n in po.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    for n in pp.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s, e = span(n, plines)
            repl[n.name] = "".join(plines[s:e]).rstrip() + "\n"
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            seg = "".join(plines[n.lineno - 1:n.end_lineno])
            if ast.dump(n) not in have_imports:
                new_imports.append(seg.rstrip() + "\n")
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            seg = "".join(plines[n.lineno - 1:n.end_lineno]).rstrip() + "\n"
            for nm in _assigned(n):
                repl["=" + nm] = seg
    edits = []
    seen = set()
    for n in po.body:
        key = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = n.name
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            names = _assigned(n)
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
    # new imports go below the module docstring and the __future__ imports
    at = 0
    for k, n in enumerate(po.body):
        if (k == 0 and isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)
                and isinstance(n.value.value, str)) or (
                isinstance(n, ast.ImportFrom) and n.module == "__future__"):
            at = n.end_lineno
        else:
            break
    out = "".join(lines[:at]) + "".join(new_imports) + "".join(lines[at:]) + "".join(appended)
    compile(out, "<patched>", "exec")  # stricter than ast.parse: catches a misplaced import
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


def _coding_prompt(task: Task, target: str, src: str, cwd: str, tests, out: str) -> str:
    # the task text may hold the contract (what the function must do) that tests only sample
    head = f"Task: {task.question}\n\n"
    short = _compact(target, src, cwd, tests, out)
    if short:
        return head + short
    parts = []
    for t in tests:
        with open(os.path.join(cwd, t), encoding="utf-8") as f:
            parts.append(f"### {t}\n" + f.read())
    test_src = "\n\n".join(parts)
    return (head + f"### {target}\n```python\n{src}\n```\n\n{test_src}"
            f"\n\nTest output:\n{clip(out, 1800)}")


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


_COMMIT = threading.Lock()


async def _audit(task: Task, llm: LLM, cwd: str, target: str, tests, model: str) -> None:
    """Review the fixed module for bugs the visible tests miss. A proposed change is kept
    only if the visible tests still pass; anything else leaves the file as it was."""
    tpath = os.path.join(cwd, target)
    with open(tpath, encoding="utf-8") as f:
        src = f.read()
    guard = {t: _sha(os.path.join(cwd, t)) for t in tests}
    # The handler runs in a worker thread and outlives a cancelled review (timeout, end
    # of the run). Once the review is over it must not touch the graded file any more.
    over = threading.Event()

    def handler(code: str) -> str:
        try:
            patched = splice(src, code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        # The change is tried in a copy: the file that will be graded is replaced only
        # after the tests passed, so a kill in the middle leaves the tested version.
        tmp = tempfile.mkdtemp(prefix="audit_")
        try:
            d = os.path.join(tmp, "w")
            shutil.copytree(cwd, d, ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))
            with open(os.path.join(d, target), "w", encoding="utf-8") as f:
                f.write(patched)
            ok, _ = _run_tests(d, tests)
        except Exception:
            ok = False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not ok or any(_sha(os.path.join(cwd, t)) != h for t, h in guard.items()):
            return "rejected: the visible tests fail with this change"
        with _COMMIT:
            if over.is_set():
                return "rejected: the review is over"
            with open(tpath + ".new", "w", encoding="utf-8") as f:
                f.write(patched)
            os.replace(tpath + ".new", tpath)
        return "accepted"

    parts = []
    for t in tests:
        with open(os.path.join(cwd, t), encoding="utf-8") as f:
            parts.append(f"### {t}\n" + f.read())
    prompt = (f"Task text: {task.question}\n\n### {target}\n```python\n{src}\n```\n\n" + "\n\n".join(parts))
    cap = _review_cap(src) if model == "think" else _budget(src, 600, 1.5, 4000)
    try:
        await llm.session(task.id, SYS_AUDIT, "apply_fix", handler).ask(prompt, MODELS[model], cap)
    finally:
        with _COMMIT:
            over.set()


def _target(task: Task, tests) -> Optional[str]:
    targets = [f for f in os.listdir(task.cwd) if f.endswith(".py") and f not in tests]
    if len(targets) == 1:
        return targets[0]
    # "the bugs are in X.py" / "all three defects are in X.py", then "Repair X.py"
    for pat in (r"(?:bugs?|defects?|errors?|mistakes?|faults?)\s+(?:(?:is|are|lie|lies|sit|sits)\s+)?(?:all\s+)?in\s+`?([\w./-]+\.py)",
                r"\b(?:repair|fix|correct|debug)\s+(?:the\s+)?(?:module\s+|file\s+)?`?([\w./-]+\.py)"):
        m = re.search(pat, task.instruction, re.I)
        if m and m.group(1) in targets:
            return m.group(1)
    # otherwise the module the others import: the tests reach it through them
    srcs = {f: open(os.path.join(task.cwd, f), encoding="utf-8", errors="replace").read() for f in targets}
    used = [f for f in targets
            if any(re.search(r"^\s*(?:from|import)\s+" + re.escape(f[:-3]) + r"\b", s, re.M)
                   for g, s in srcs.items() if g != f)]
    return used[0] if len(used) == 1 else None


def _review_cap(src: str) -> int:
    return _budget(src, 3000, 6, 16000)


def review_cost(task: Task) -> Optional[float]:
    """Worst-case cost of review_coding, or None if the task has nothing to review."""
    tests = _test_files(task.cwd)
    target = _target(task, tests) if tests else None
    if not target:
        return None
    size = len(task.question) + 600
    for f in [target] + list(tests):
        size += os.path.getsize(os.path.join(task.cwd, f))
    return _review_cap(open(os.path.join(task.cwd, target), encoding="utf-8").read()) + size / 30.0 + 100


async def review_coding(task: Task, llm: LLM) -> None:
    """Second review of a solved coding task by the reasoning model. run_agent calls it
    after every task was attempted, with the budget that is left over."""
    tests = _test_files(task.cwd)
    target = _target(task, tests) if tests else None
    if target:
        await _audit(task, llm, task.cwd, target, tests, "think")


async def solve_coding(task: Task, llm: LLM) -> Optional[str]:
    cwd = task.cwd
    tests = _test_files(cwd)
    if not tests:
        return await solve_generic(task, llm)
    target = _target(task, tests)
    answer = await _fix_until_tests_pass(task, llm, cwd, tests, target)
    if answer == "DONE" and AUDIT and target and not task.deferred:
        # The cheap model reviews here (~100 tokens per task). The reasoning model finds
        # more but costs ~1000 per task: in assured mode it reviews right away, otherwise
        # run_agent gives it whatever budget is left once every task was attempted.
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
        for model, mt in plan:
            # every round is a fresh agent: it sees the current file and current failures only
            agent = llm.session(task.id, SYS_FIX, "apply_fix", handler)
            reply = await agent.ask(_coding_prompt(task, target, state["current"], cwd, tests,
                                                   state["out"]), model, mt)
            if reply.truncated and not state["ok"] and not task.deferred:
                # the patch was cut off by the cap: same request again with more room
                agent = llm.session(task.id, SYS_FIX, "apply_fix", handler)
                reply = await agent.ask(
                    _coding_prompt(task, target, state["current"], cwd, tests, state["out"]),
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
