"""Zero-token solvers. Each one answers only when it understands the task completely;
anything else returns None and the task goes to the LLM solver.

- math: every sentence of the problem must match a known step pattern;
- data_analysis: the question must match a known aggregate over existing columns;
- search: the question must parse into a chain of card lookups that all resolve;
- coding: single-edit mutations of the failing function, accepted only if the tests
  pass and all passing candidates behave identically on extra probe inputs.
"""
from __future__ import annotations

import ast
import copy
import csv
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List, Optional

from .sandbox import run_python, run_test_file
from .tasks import Task, sniff_delimiter

ENABLED = os.environ.get("DET", "1") != "0"


def det_solve(t: Task) -> Optional[str]:
    if not ENABLED:
        return None
    fn = {"math": solve_math, "data_analysis": solve_data, "search": solve_search,
          "coding": solve_coding}.get(t.kind)
    try:
        return fn(t) if fn else None
    except Exception:  # a zero-token shortcut must never break the task
        return None


# ------------------------------------------------------------------ math

_STEPS = [
    (r"It ships (\d+) boxes? per day for (\d+) days?\.", lambda x, a, d: x - a * d),
    (r"It receives (\d+) deliver(?:y|ies) of (\d+) boxes? each\.", lambda x, k, m: x + k * m),
    (r"the stock is split equally into (\d+) warehouses? and it keeps one share \(rounded down\)\.",
     lambda x, k: x // k),
    (r"the stock is doubled by merging with a sister warehouse\.", lambda x: x * 2),
]


def solve_math(t: Task) -> Optional[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.?])\s+", t.question) if s.strip()]
    if len(sents) < 2:
        return None
    m = re.fullmatch(r"A warehouse starts with (\d+) boxes?\.", sents[0])
    if not m or not re.fullmatch(r"How many boxes does it have at the end\?", sents[-1]):
        return None
    x = int(m.group(1))
    for s in sents[1:-1]:
        s = re.sub(r"^Then,?\s+", "", s)
        s = s[0].upper() + s[1:] if s.startswith("it ") else s
        for pat, fn in _STEPS:
            mm = re.fullmatch(pat, s)
            if mm:
                args = [int(g) for g in mm.groups()]
                if "split" in pat and args[0] == 0:
                    return None
                x = fn(x, *args)
                break
        else:
            return None  # an unknown step: let the model handle it
    return str(x)


# ------------------------------------------------------------------ data analysis

_Q_UNITS = re.compile(
    r"How many (\w+) of (\w+) '([^']+)' were sold"
    r"(?: in (\w+?)s? (\d+)(?: through (\d+) inclusive)?)?\?")


def solve_data(t: Task) -> Optional[str]:
    files = re.findall(r"\b([\w-]+\.csv)\b", t.instruction)
    if len(set(files)) != 1:
        return None
    path = os.path.join(t.cwd, files[0])
    if not os.path.isfile(path):
        return None
    m = _Q_UNITS.search(t.question)
    if not m or t.question.rstrip()[-len(m.group(0)):] != m.group(0):
        return None
    # Nothing but the file description may precede the question: an extra sentence
    # ("Exclude cancelled rows.") is a condition this solver does not understand.
    before = t.question.rstrip()[:-len(m.group(0))].strip()
    if not re.fullmatch(r"(?:The file [\w.-]+ has columns [\w,]+(?: \(\d+ rows\))?\.)?", before):
        return None
    measure, col, value, rcol, lo, hi = m.groups()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=sniff_delimiter(path))
        head = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return None
    need = [measure, col] + ([rcol] if rcol else [])
    if any(c not in head for c in need) or len(set(head)) != len(head):
        return None
    lo_i = int(lo) if lo else None
    hi_i = int(hi) if hi else lo_i
    total = 0
    for r in rows:
        if None in r.values() or None in r:
            return None  # ragged row
        try:
            units = int(r[measure])
            rv = int(r[rcol]) if rcol else None
        except (TypeError, ValueError):
            return None
        if r[col] == value and (rcol is None or lo_i <= rv <= hi_i):
            total += units
    return str(total)


# ------------------------------------------------------------------ search

_REL = {"employer": "Employer", "mentor": "Mentor"}
_QUESTIONS = [
    (r"In which year was (.+) born\?", "Born"),
    (r"In which year was (.+) founded\?", "Founded"),
    (r"In which city is the headquarters of (.+)\?", "Headquarters"),
    (r"In which city is (.+) headquartered\?", "Headquarters"),
    (r"Which company does (.+) work for\?", "Employer"),
    (r"Which company employs (.+)\?", "Employer"),
    (r"Who is (.+)'s employer\?", "Employer"),
    (r"Who is (.+)'s mentor\?", "Mentor"),
    (r"Who is the mentor of (.+)\?", "Mentor"),
]


def _load_cards(root: str) -> Optional[Dict[str, dict]]:
    cards: Dict[str, dict] = {}
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if not fn.endswith(".md"):
                continue
            with open(os.path.join(dp, fn), encoding="utf-8") as f:
                text = f.read()
            h = re.search(r"^#\s*(.+?)\s*$", text, re.M)
            if not h:
                return None
            fields = {}
            for k, v in re.findall(r"^([A-Za-z][\w ]*):[ \t]*(.+?)[ \t]*$", text, re.M):
                if k in fields:
                    return None  # two values for one field: not ours to decide
                fields[k] = v
            key = h.group(1).lower()
            if key in cards:
                return None      # two cards with one title
            cards[key] = {"title": h.group(1), "fields": fields}
    return cards or None


def _resolve(expr: str, cards: Dict[str, dict]) -> Optional[str]:
    expr = expr.strip()
    if expr.lower() in cards:
        return expr.lower()
    # Try every reading: "the company that employs X's mentor" parses two ways and
    # only one of them resolves to cards.
    for pat in (r"the (employer|mentor) of (.+)", r"the company that employs (.+)",
                r"(.+)'s (employer|mentor)"):
        m = re.fullmatch(pat, expr)
        if not m:
            continue
        if pat.startswith("the company"):
            rel, inner = "employer", m.group(1)
        elif pat.startswith("the ("):
            rel, inner = m.group(1), m.group(2)
        else:
            inner, rel = m.group(1), m.group(2)
        base = _resolve(inner, cards)
        if base is None:
            continue
        val = cards[base]["fields"].get(_REL[rel])
        if val and val.lower() in cards:
            return val.lower()
    return None


def solve_search(t: Task) -> Optional[str]:
    kb = os.path.join(t.cwd, "kb")
    if not os.path.isdir(kb):
        return None
    q = re.search(r"((?:In which|Which|Who)\b[^.?]*\?)\s*$", t.question)
    if not q:
        return None
    # the question must be the whole task: a sentence before it may carry a condition
    # (the standard sentence about the knowledge base is already removed by the cleaning;
    # any other sentence is unknown text, and unknown text goes to the model)
    if t.question[:q.start()].strip():
        return None
    cards = _load_cards(kb)
    if not cards:
        return None
    for pat, field in _QUESTIONS:
        m = re.fullmatch(pat, q.group(1))
        if not m:
            continue
        ent = _resolve(m.group(1), cards)
        if ent is None:
            return None
        return cards[ent]["fields"].get(field) or None
    return None


# ------------------------------------------------------------------ coding

_CMP = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
_ARITH = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod]
_SWAP = {"sorted": ["reversed"], "reversed": ["sorted"], "min": ["max"], "max": ["min"],
         "any": ["all"], "all": ["any"], "upper": ["lower"], "lower": ["upper"],
         "startswith": ["endswith"], "endswith": ["startswith"],
         "lstrip": ["rstrip", "strip"], "rstrip": ["lstrip", "strip"],
         "isdigit": ["isalpha"], "isalpha": ["isdigit"], "keys": ["values"], "values": ["keys"]}


def _become(node: ast.AST, other: ast.AST) -> None:
    node.__class__ = other.__class__
    node.__dict__ = dict(other.__dict__)


def _edits(n: ast.AST):
    """(rank, mutate) pairs for one node; mutate edits the node in place."""
    out = []
    if isinstance(n, ast.Compare):
        for k, op in enumerate(n.ops):
            alts = ([c for c in _CMP if not isinstance(op, c)] if type(op) in _CMP else
                    [ast.NotIn] if isinstance(op, ast.In) else
                    [ast.In] if isinstance(op, ast.NotIn) else
                    [ast.IsNot] if isinstance(op, ast.Is) else
                    [ast.Is] if isinstance(op, ast.IsNot) else [])
            for alt in alts:
                out.append((0, lambda x, k=k, alt=alt: x.ops.__setitem__(k, alt())))
    if isinstance(n, ast.AugAssign) and type(n.op) in _ARITH:
        for alt in (ast.Add, ast.Sub, ast.Mult):
            if not isinstance(n.op, alt):
                out.append((1, lambda x, alt=alt: setattr(x, "op", alt())))
    if isinstance(n, ast.BinOp) and type(n.op) in _ARITH:
        for alt in _ARITH:
            if not isinstance(n.op, alt):
                out.append((1, lambda x, alt=alt: setattr(x, "op", alt())))
        out.append((3, lambda x: _become(x, x.left)))
        out.append((3, lambda x: _become(x, x.right)))
    if isinstance(n, ast.BoolOp):
        alt = ast.Or if isinstance(n.op, ast.And) else ast.And
        out.append((1, lambda x, alt=alt: setattr(x, "op", alt())))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.USub)):
        out.append((3, lambda x: _become(x, x.operand)))
    if isinstance(n, ast.Name) and n.id in _SWAP:
        for alt in _SWAP[n.id]:
            out.append((2, lambda x, alt=alt: setattr(x, "id", alt)))
    if isinstance(n, ast.Attribute) and n.attr in _SWAP:
        for alt in _SWAP[n.attr]:
            out.append((2, lambda x, alt=alt: setattr(x, "attr", alt)))
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool):
            out.append((4, lambda x: setattr(x, "value", not x.value)))
        elif isinstance(n.value, int):
            for alt in dict.fromkeys([n.value + 1, n.value - 1, 0, 1]):
                if alt != n.value:
                    out.append((4, lambda x, alt=alt: setattr(x, "value", alt)))
    if isinstance(n, (ast.If, ast.While, ast.IfExp)):
        out.append((5, lambda x: setattr(x, "test", ast.UnaryOp(op=ast.Not(), operand=x.test))))
    if isinstance(n, ast.comprehension):
        for k in range(len(n.ifs)):
            out.append((5, lambda x, k=k: x.ifs.__setitem__(
                k, ast.UnaryOp(op=ast.Not(), operand=x.ifs[k]))))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sorted":
        if any(k.arg == "reverse" for k in n.keywords):
            out.append((2, lambda x: setattr(x, "keywords",
                                             [k for k in x.keywords if k.arg != "reverse"])))
        else:
            out.append((2, lambda x: x.keywords.append(
                ast.keyword(arg="reverse", value=ast.Constant(True)))))
    return out


def mutants(func_src: str, limit: int = 120) -> List[str]:
    tree = ast.parse(func_src)
    base = ast.unparse(tree)
    found = []
    for i, n in enumerate(list(ast.walk(tree))):
        for rank, edit in _edits(n):
            t2 = copy.deepcopy(tree)
            try:
                edit(list(ast.walk(t2))[i])
                src = ast.unparse(ast.fix_missing_locations(t2))
                ast.parse(src)
            except Exception:
                continue
            if src != base:
                found.append((rank, len(found), src))
    seen, out = set(), []
    for _, _, src in sorted(found):
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out[:limit]


_FAILED = re.compile(r"^(?:FAIL|ERROR): (\w+) \(", re.M)


def _failing(cwd: str, tests: List[str], timeout: int = 15) -> Optional[set]:
    """Names of failing tests; None if the run broke in a way we cannot attribute."""
    bad = set()
    for tf in tests:
        r = run_test_file(tf, cwd, timeout=timeout)
        if r.timeout:
            return None
        names = set(_FAILED.findall(r.err + r.out))
        if r.code != 0 and not names:
            return None
        bad |= names
    return bad


def _test_map(test_src: str, module: str):
    """test name -> functions of `module` it calls; and literal call args per function."""
    tree = ast.parse(test_src)
    aliases, direct = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == module:
                    aliases.add(a.asname or a.name)
        elif isinstance(n, ast.ImportFrom) and n.module == module:
            for a in n.names:
                direct.add(a.asname or a.name)
    uses: Dict[str, set] = {}
    calls: Dict[str, list] = {}

    def target(node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in aliases:
            return node.attr
        if isinstance(node, ast.Name) and node.id in direct:
            return node.id
        return None

    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test"):
            for n in ast.walk(fn):
                name = target(n)
                if name:
                    uses.setdefault(fn.name, set()).add(name)
                if isinstance(n, ast.Call) and target(n.func) and not n.keywords:
                    try:
                        calls.setdefault(target(n.func), []).append(
                            [ast.literal_eval(a) for a in n.args])
                    except Exception:
                        pass
    return uses, calls


def _variants(v):
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [v - 1, v + 1, 0, v + 5]
    if isinstance(v, float):
        return [v + 0.5, -v]
    if isinstance(v, str):
        return [v + " zz", "B " + v, v.upper(), v.replace(" ", "  "), "", v[::-1]]
    if isinstance(v, list):
        out = [v[::-1], v + v, v[:1], []]
        if v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            out += [sorted(v), v + [max(v) + 5], [min(v) - 3] + v, [-x for x in v],
                    v + [v[0]], [x + 1 for x in v]]
        elif v:
            out += [v + [v[0]], sorted(v, key=repr)]
        return out
    return []


_PROBE = r'''
import ast, json
data = json.loads(%r)
res = []
for src in data["sources"]:
    ns = {}
    try:
        exec(compile(src, "candidate", "exec"), ns)
        fn = ns[data["func"]]
    except Exception as e:
        res.append(["LOAD " + type(e).__name__])
        continue
    outs = []
    for p in data["probes"]:
        try:
            r = fn(*ast.literal_eval(p))
            if hasattr(r, "__next__"):
                r = list(r)
            outs.append(repr(r))
        except Exception as e:
            outs.append("EXC " + type(e).__name__)
    res.append(outs)
print(json.dumps(res))
'''


def _pick(sources: List[str], func: str, calls: list, workdir: str) -> Optional[str]:
    """Several fixes pass the tests: compare them on inputs the tests do not cover.
    Equivalent spellings of the real fix (> and >=, == 0 and != 1) behave alike, while
    a fix that only fits the test (x % 3 == 1) stands alone. Returns the best-ranked
    member of a clear majority, or None when there is no such majority."""
    probes = []
    for args in calls:
        probes.append(repr(args))
        for i, a in enumerate(args):
            for v in _variants(a):
                probes.append(repr(args[:i] + [v] + args[i + 1:]))
    probes = list(dict.fromkeys(probes))[:60]
    if not probes:
        return None
    payload = json.dumps({"sources": sources, "func": func, "probes": probes})
    r = run_python(_PROBE % payload, workdir, timeout=15)
    if not r.ok:
        return None
    try:
        res = json.loads(r.out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    groups: Dict[str, List[int]] = {}
    for i, outs in enumerate(res):
        groups.setdefault(json.dumps(outs), []).append(i)
    best = max(groups.values(), key=len)
    if len(best) == len(res) or (len(best) >= 2 and len(best) * 2 > len(res)):
        return sources[best[0]]
    return None


def repair(cwd: str, target: str, tests: List[str], budget_sec: float = 60.0) -> bool:
    """Try to fix `target` in place with single-edit mutations. True only if every test
    passes afterwards; on any doubt the file is left exactly as it was."""
    from .solvers import splice  # late import: solvers imports this module
    deadline = time.time() + budget_sec
    tpath = os.path.join(cwd, target)
    with open(tpath, encoding="utf-8") as f:
        original = f.read()
    module = os.path.splitext(target)[0]
    uses: Dict[str, set] = {}
    calls: Dict[str, list] = {}
    for tf in tests:
        with open(os.path.join(cwd, tf), encoding="utf-8") as f:
            u, c = _test_map(f.read(), module)
        uses.update(u)
        for k, v in c.items():
            calls.setdefault(k, []).extend(v)
    tree = ast.parse(original)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    tmp = tempfile.mkdtemp(prefix="repair_")
    try:
        dirs: Queue = Queue()
        for i in range(4):
            d = os.path.join(tmp, f"w{i}")
            shutil.copytree(cwd, d, ignore=shutil.ignore_patterns("__pycache__", "answer.txt"))
            dirs.put(d)

        def failing_with(src: str) -> Optional[set]:
            d = dirs.get()
            try:
                with open(os.path.join(d, target), "w", encoding="utf-8") as f:
                    f.write(src)
                return _failing(d, tests)
            finally:
                dirs.put(d)

        current = original
        bad = failing_with(current)
        if not bad:
            return False
        bad0 = set(bad)
        todo: Dict[str, set] = {}
        for test in bad:
            funcs = uses.get(test, set()) & set(defs)
            if len(funcs) == 1:  # otherwise we cannot tell which function the test blames
                todo.setdefault(next(iter(funcs)), set()).add(test)

        for func, its_tests in sorted(todo.items()):
            node = defs[func]
            func_src = ast.get_source_segment(original, node)
            cands = [splice(current, m) for m in mutants(func_src)]
            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(
                    lambda s: failing_with(s) if time.time() < deadline else None, cands))
            good = [s for s, f in zip(cands, results)
                    if f is not None and not (f & its_tests) and f <= bad]
            if time.time() > deadline:
                break
            chosen = (None if not good else good[0] if len(good) == 1
                      else _pick(good[:12], func, calls.get(func, []), tmp))
            if chosen is None:
                continue  # no single-edit fix, or the tests do not pin it down: the model's job
            current = chosen
            bad = bad - its_tests

        if current == original:
            return False
        # Keep what was fixed even if something is left: the model then gets a smaller job.
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(current)
        left = _failing(cwd, tests)
        if left == set():
            return True
        if left is None or not left <= bad0 or left == bad0:
            with open(tpath, "w", encoding="utf-8") as f:
                f.write(original)
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def solve_coding(t: Task) -> Optional[str]:
    cwd = t.cwd
    tests = sorted(f for f in os.listdir(cwd) if f.endswith(".py") and
                   (f.startswith("test") or f.endswith("_test.py")))
    targets = [f for f in os.listdir(cwd) if f.endswith(".py") and f not in tests]
    if not tests or len(targets) != 1:
        return None
    if _failing(cwd, tests) == set():
        return "DONE"
    return "DONE" if repair(cwd, targets[0], tests) else None
