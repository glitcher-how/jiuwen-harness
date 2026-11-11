"""Task loading, zero-token analysis (context building, feasibility) and cost estimates."""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Task:
    id: str
    dir: str
    kind: str
    instruction: str
    reward: float
    raw: dict
    question: str = ""          # instruction without harness boilerplate
    cwd: str = ""               # "current directory" of the instruction
    answer_mode: str = "value"  # value | done
    p_feasible: float = 1.0
    why_infeasible: str = ""
    certain: bool = False       # infeasibility proven, not guessed: never attempt
    deferred: bool = False      # guessed infeasible: attempt last, once, if budget remains
    prior: Optional[str] = None  # answer of the zero-token solver, to be confirmed (assured mode)
    det: bool = False           # solved by code, without the model
    context: Dict[str, str] = field(default_factory=dict)
    est_cost: float = 0.0
    status: str = "todo"
    answer: Optional[str] = None
    spent: float = 0.0

    @property
    def priority(self) -> float:
        return self.reward * self.p_feasible * P_SUCCESS.get(self.kind, 0.6) / max(self.est_cost, 1.0)


# prior success rates and expected output tokens per kind (updated online by planner)
P_SUCCESS = {"math": 0.95, "data_analysis": 0.95, "search": 0.95, "coding": 0.85, "generic": 0.5}
EXPECTED_OUT = {"math": 110, "data_analysis": 140, "search": 8, "coding": 260, "generic": 900}
ATTEMPTS = {"math": 1.1, "data_analysis": 1.1, "search": 1.0, "coding": 1.4, "generic": 1.0}

_BOILER = [
    r"Work only inside the current directory\.?",
    r"The directory [\w./-]+ contains a small knowledge base of markdown files\.",
    r"echo\s+\S+\s*>+\s*`?answer\.txt`?",
]
# Words that say nothing about the problem itself: tools, delivery of the answer.
_FILLER = set("""a an the to of in into inside with and or then it its this that your you are is be as by for
no not use using compute computed calculate find get run do done when after all once finished work only just
single final answer result value line file files tool tools bash python python3 c py current directory write
save put store print output word number integer text other nothing else must should contain containing named
called exactly please make sure them there txt ls grep cat tests pass edit example e g eg such like via
here""".split())
_TRIGGER = re.compile(r"answer\.txt|\bbash\b|\bpython3?\b|file tools?|current directory|\bwrite only\b", re.I)
_TOOL = r"(?:the )?(?:bash tool|bash|python3?(?: -c)?|file tools?)"


def _housekeeping(sent: str) -> bool:
    """A sentence only about tools or about delivering the answer. One that also carries
    a condition of the problem ("Use bash to sum the rows excluding cancelled ones",
    "Write the negated result to answer.txt") is not housekeeping: dropping it would
    change the task, so a single unknown word is enough to keep the sentence."""
    if not _TRIGGER.search(sent):
        return False
    words = re.findall(r"[A-Za-z0-9_']+", re.sub(r"`?[\w./-]+\.(?:py|txt)\b`?", " ", sent.lower()))
    return all(w in _FILLER for w in words)


def clean_instruction(text: str) -> str:
    q = text
    for pat in _BOILER:
        q = re.sub(pat, "", q, flags=re.I)
    # "(for example, python3 -c)" goes; "(for example, 2.5 rounds to 3)" is a rule and stays
    q = re.sub(r"\s*\(([^()]*)\)", lambda m: "" if _housekeeping(m.group(1)) else m.group(0), q)
    # How to deliver the answer and which tools to use is the harness's business, and the
    # model must not act on it. Only the housekeeping words go; conditions stay.
    keep = []
    for sent in re.split(r"(?<=[.?!])\s+(?=[A-Z`])", q):
        if _housekeeping(sent):
            continue
        if _TRIGGER.search(sent):
            s = re.sub(r"\s+(?:to|in|into)\s+(?:the\s+)?(?:file\s+)?`?answer\.txt`?"
                       r"(?:\s+in the current directory)?", "", sent)
            if s != sent:
                s = re.sub(r"^(?:Write|Save|Put|Store)\b", "Give", s)
            s = re.sub(rf"^(?:Use|Using) {_TOOL}(?: (?:with|or|and) {_TOOL})*(?: tool)? to ", "", s)
            sent = s[:1].upper() + s[1:]
        keep.append(sent)
    return re.sub(r"\s+", " ", " ".join(keep)).strip()


def load_tasks(root: str) -> List[Task]:
    tasks = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        tj = os.path.join(d, "task.json")
        if not os.path.isfile(tj):
            continue
        try:
            with open(tj, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            continue
        instr = str(raw.get("instruction", ""))
        kind = str(raw.get("task_type", "")).lower()
        if kind not in ("math", "data_analysis", "search", "coding"):
            kind = guess_kind(instr, d)
        try:
            reward = float(raw.get("reward", 1) or 0)
        except (TypeError, ValueError):
            reward = 1.0
        ws = os.path.join(d, "workspace")
        t = Task(id=str(raw.get("task_id", name)), dir=d, kind=kind, instruction=instr,
                 reward=reward, raw=raw, cwd=ws if os.path.isdir(ws) else d)
        t.question = clean_instruction(instr)
        t.answer_mode = "done" if re.search(r"ONLY the word DONE", instr) else "value"
        tasks.append(t)
    return tasks


def guess_kind(instr: str, d: str) -> str:
    s = instr.lower()
    if "test" in s and ("fix" in s or "bug" in s):
        return "coding"
    if "kb/" in s or "knowledge base" in s:
        return "search"
    if ".csv" in s:
        return "data_analysis"
    if re.search(r"how many|compute|calculate", s) and not os.path.isdir(os.path.join(d, "workspace")):
        return "math"
    return "generic"


# ---------------------------------------------------------------- per-kind analysis

def analyze(t: Task) -> None:
    fn = {"search": _an_search, "data_analysis": _an_data, "coding": _an_coding,
          "math": _an_math}.get(t.kind, _an_generic)
    try:
        fn(t)
    except Exception as e:  # analysis must never kill the run
        t.context.setdefault("note", f"analysis error: {e}")
    t.est_cost = estimate_cost(t)


def estimate_cost(t: Task) -> float:
    """Expected spend on the task, in budget units, before any model call.

    The harness builds the prompt itself, so the input side is known. The output side
    grows with the problem: a nine-operation word problem needs a longer program than a
    one-line sum. Fitted on collected trajectories (short tasks of the organisers' kind
    and long bureaucratic ones): about 2.7 output tokens per token of a math problem
    and 2.2 per token of a data question."""
    inp_chars = len(t.question) + sum(len(v) for v in t.context.values()) + 450
    base = (inp_chars / 3.3 / 10.0 + EXPECTED_OUT.get(t.kind, 500)) * ATTEMPTS.get(t.kind, 1.2)
    q_tokens = len(t.question) / 3.5
    if t.kind == "math":
        return max(base, 2.7 * q_tokens - 100)
    if t.kind == "data_analysis":
        return max(base, 2.2 * q_tokens - 70)
    return base


def _an_math(t: Task) -> None:
    if not re.search(r"\d", t.question):
        t.p_feasible, t.why_infeasible = 0.2, "no numbers in problem"
    # The zero must belong to the division itself: the first "into/by/among" after the
    # verb. "Divide 10 by 2 and increase the result by 0" is an ordinary problem. Even so
    # this is a guess from the wording, so the task is postponed, not dropped.
    if re.search(r"\b(?:split|divided?|shared?|distributed?)\b"
                 r"(?:(?!\b(?:into|by|among|between)\b)[^.?!])*"
                 r"\b(?:into|by|among|between)\s+0\b(?![.,]?\d)", t.question, re.I):
        t.p_feasible, t.why_infeasible = 0.05, "division by zero in the statement"
    first = re.split(r"(?<=[.!?])\s", t.question, maxsplit=1)[0]
    if not re.search(r"\d", first) and re.search(
            r"\b(unknown|unrecorded|unspecified|undisclosed|not recorded|some number|an? (?:certain|unknown) (?:number|amount))\b",
            first, re.I):
        t.p_feasible, t.why_infeasible = 0.05, "starting quantity is not given"


def _an_generic(t: Task) -> None:
    t.context["files"] = _listing(t.cwd)


def _listing(root: str, limit: int = 60) -> str:
    out = []
    for dp, dn, fns in os.walk(root):
        dn[:] = [x for x in dn if not x.startswith(".") and x != "__pycache__"]
        for f in fns:
            p = os.path.join(dp, f)
            out.append(f"{os.path.relpath(p, root)} ({os.path.getsize(p)} B)")
            if len(out) >= limit:
                return "\n".join(out) + "\n..."
    return "\n".join(out)


# --- search: entity-linked retrieval over the markdown KB

# multi-word capitalised names, including "Granite AI", "Bank of Kazan", "O'Neil"
_NAME = re.compile(r"\b([A-Z][\w&.'-]*(?:[ -](?:[A-Z][\w&.'-]*|of|and|de|von|van|the))*[ -][A-Z][\w&.'-]*)")
# "Key: Value" lines of a card; values are the usual links to other cards
_FIELD = re.compile(r"^[\w ]+:\s*(.+?)\s*$", re.M)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _an_search(t: Task) -> None:
    kb = os.path.join(t.cwd, "kb")
    if not os.path.isdir(kb):
        cands = [os.path.join(t.cwd, x) for x in os.listdir(t.cwd)
                 if os.path.isdir(os.path.join(t.cwd, x))]
        kb = cands[0] if cands else t.cwd
    docs = {}
    for dp, _, fns in os.walk(kb):
        for f in sorted(fns):
            if f.endswith((".md", ".txt", ".json", ".csv", ".yaml", ".yml")):
                p = os.path.join(dp, f)
                try:
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        docs[os.path.relpath(p, t.cwd)] = fh.read()
                except OSError:
                    pass
    if not docs:
        t.p_feasible, t.why_infeasible = 0.05, "knowledge base is empty or missing"
        t.certain = True
        return
    total = sum(len(v) for v in docs.values())

    def title(path, text):
        h = re.search(r"^#\s*(.+)$", text, re.M)
        return _norm(h.group(1)) if h else _norm(os.path.splitext(os.path.basename(path))[0].replace("_", " "))

    titles = {p: title(p, x) for p, x in docs.items()}
    by_title = {}
    for p, ti in titles.items():
        by_title.setdefault(ti, []).append(p)
        # filenames like person_irina_smith.md -> "irina smith"
        stem = _norm(re.sub(r"^(person|company|org|city|product)_", "",
                            os.path.splitext(os.path.basename(p))[0]))
        by_title.setdefault(stem, []).append(p)

    # A name never runs across a sentence boundary ("... named Jordan Pike. Capacity is")
    # and is not a placeholder in capitals ("YYYY-MM-DD", "ID. Follow"): such false names
    # are found nowhere in the base and would mark a solvable task as doubtful.
    names = set()
    for part in re.split(r"[.;:!?,()]\s+", t.question.replace("'s ", " ")):
        for n in _NAME.findall(part):
            words = re.split(r"[ -]", n)
            if any(re.search(r"[a-z]", w) for w in words if w not in ("of", "and", "de", "von", "van", "the")):
                names.add(_norm(n.replace("'s", "")))
    names.discard("")
    # "Find Alice Smith" is the name "Alice Smith" behind a capitalised verb: every
    # tail of at least two words counts as the name too
    for n in list(names):
        w = n.split()
        names.update(" ".join(w[i:]) for i in range(1, len(w) - 1))
    seeds = [p for n in names for p in by_title.get(n, [])]
    if not seeds:
        # unknown entity -> very likely an infeasible question; still try body search
        hits = [p for p, x in docs.items() if any(n and n in _norm(x) for n in names)]
        if names and not hits:
            t.p_feasible, t.why_infeasible = 0.05, f"entities {sorted(names)} not in KB"
        seeds = hits
    def walk(max_hops: int) -> List[str]:
        picked: List[str] = []
        frontier = list(dict.fromkeys(seeds))
        for _hop in range(max_hops):
            nxt = []
            for p in frontier:
                if p in picked:
                    continue
                picked.append(p)
                refs = _FIELD.findall(docs[p]) + _NAME.findall(docs[p])
                for v in refs:
                    for n in [v] + re.split(r"\s*(?:,|;|\band\b)\s*", v):
                        for q in by_title.get(_norm(n), []):
                            if q not in picked:
                                nxt.append(q)
            frontier = nxt
            if not frontier:
                break
        return picked

    # Cards the question needs: the person, one per person->person link
    # ("mentor of the mentor" counts twice), and one more if a fact about the
    # company is asked (the employer's name itself is on the person's card).
    # Extra cards are not free: the model sometimes walks one link too far.
    ql = t.question.lower()
    person_links = len(re.findall(
        r"\b(mentor|manager|supervisor|advisor|spouse|friend|colleague|parent|boss)s?\b", ql))
    company_fact = bool(re.search(r"headquarter|founded|located|\bcity\b|\bceo\b|founder|industry", ql))
    need = 1 + person_links + int(company_fact)

    def render(paths, limit=25):
        if paths:
            chosen = paths[:limit]        # only the entity-linked cards
        elif total <= 6000:
            chosen = list(docs)           # nothing matched: small KB goes in whole
        else:
            chosen = list(docs)[:25]
        return "\n".join(f"--- {p}\n{docs[p].strip()}" for p in chosen)

    narrow, wide = walk(need), walk(need + 2)
    t.context["kb"] = render(narrow)
    if wide != narrow:
        t.context["kb_wide"] = render(wide)  # retry context if the narrow one is not enough
    t.context["kb_linked"] = ", ".join(narrow)
    if total <= 60000:
        # every card: for questions over the whole base (counts, extremes, reverse
        # links) and, while the base is small, as a second look when the linked cards
        # do not hold the answer
        t.context["kb_all"] = render(list(docs), limit=None)


# --- data analysis: schema summary instead of raw file

def sniff_delimiter(path: str) -> str:
    """Field separator of a table file, judged by its header line."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
    except OSError:
        return ","
    best = max(",;\t|", key=head.count)
    return best if head.count(best) else ","


def _an_data(t: Task) -> None:
    files = [f for f in os.listdir(t.cwd) if f.lower().endswith((".csv", ".tsv", ".json", ".txt"))
             and f != "answer.txt"]
    mentioned = []
    for f in dict.fromkeys(re.findall(r"([\w./-]*\w\.(?:csv|tsv|json))\b", t.instruction)):
        if os.path.exists(os.path.join(t.cwd, f)):
            mentioned.append(f)
            continue
        # maybe it sits in a subfolder (data/sales.csv): look through the whole task
        base = os.path.basename(f)
        hits = [os.path.relpath(os.path.join(dp, base), t.cwd)
                for dp, _, fns in os.walk(t.dir) if base in fns]
        if hits:
            mentioned.append(hits[0])
        else:
            t.p_feasible, t.why_infeasible = 0.05, f"{f} is missing"
            t.certain = True
            return
    summ = []
    for f in (mentioned or files)[:4]:
        p = os.path.join(t.cwd, f)
        if not f.lower().endswith((".csv", ".tsv")):
            with open(p, encoding="utf-8", errors="replace") as fh:
                summ.append(f"{f}: {fh.read(600)}")
            continue
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh, delimiter=sniff_delimiter(p)))
        if not rows:
            summ.append(f"{f}: empty")
            t.p_feasible, t.why_infeasible = 0.05, f"{f} is empty"
            t.certain = True
            continue
        head, body = rows[0], rows[1:]
        lines = [f"{f}: {len(body)} data rows, columns: {','.join(head)}",
                 "first rows:"] + [",".join(r) for r in body[:3]]
        values = {}
        for i, col in enumerate(head):
            col_vals = [r[i] for r in body if i < len(r)]
            uniq = sorted(set(col_vals))
            numeric = all(re.fullmatch(r"-?\d+(\.\d+)?", v or "0") for v in col_vals)
            if numeric:
                nums = [float(v) for v in col_vals if v]
                if nums:
                    blank = sum(1 for v in col_vals if not v.strip())
                    lines.append(f"{col}: numeric {min(nums):g}..{max(nums):g}"
                                 + (f", {blank} BLANK (value missing)" if blank else ""))
            elif len(uniq) <= 25:
                lines.append(f"{col}: values {uniq}")
                values[col] = set(uniq)
            else:
                lines.append(f"{col}: {len(uniq)} distinct text values")
        bad = [r for r in body if len(r) != len(head)]
        if bad:
            lines.append(f"warning: {len(bad)} malformed rows")
        summ.append("\n".join(lines))
        # quoted literals that appear in no categorical column -> probably infeasible
        # a short literal in quotes; an apostrophe inside a word (employee's) is not a quote
        quoted = re.findall(r"(?<![A-Za-z])'([^'\n]{1,40})'(?![A-Za-z])|\"([^\"\n]{1,40})\"", t.question)
        allvals = set().union(*values.values()) if values else set()
        for a, b in quoted:
            v = a or b
            if values and v not in allvals and v not in head:
                t.p_feasible, t.why_infeasible = 0.08, f"value {v!r} not present in {f}"
        for c in re.findall(r"\b(?:the\s+)?([A-Za-z_]+)\s+column\b", t.question):
            if c.lower() not in ("the", "a", "this") and c not in head:
                t.p_feasible, t.why_infeasible = 0.05, f"column {c!r} not in {f}"
        # explicit claims like "(20 rows)" that disagree with the file
        mrows = re.search(r"\((\d+) rows\)", t.instruction)
        if mrows and int(mrows.group(1)) != len(body):
            t.context["note"] = f"instruction says {mrows.group(1)} rows, file has {len(body)}"
        cols = re.search(r"has columns ([\w,]+)", t.instruction)
        if cols:
            missing = [c for c in cols.group(1).split(",") if c and c not in head]
            if missing:
                t.p_feasible, t.why_infeasible = 0.2, f"columns {missing} missing in {f}"
    t.context["data"] = "\n\n".join(summ)


# --- coding: just collect sources; tests are run by the solver

def _an_coding(t: Task) -> None:
    srcs = {}
    for f in sorted(os.listdir(t.cwd)):
        if f.endswith(".py"):
            with open(os.path.join(t.cwd, f), encoding="utf-8", errors="replace") as fh:
                srcs[f] = fh.read()
    if not srcs:
        t.p_feasible, t.why_infeasible = 0.05, "no python sources"
        t.certain = True
        return
    t.context["sources"] = "\n\n".join(f"### {k}\n{v}" for k, v in srcs.items())
    # A test that reads a data file which is not there usually cannot be satisfied by
    # editing the library. Not proof (the fix may be to stop needing it), so: one try.
    for k, v in srcs.items():
        if k.startswith("test") or k.endswith("_test.py"):
            for name in re.findall(r"""['"]([\w./-]+\.(?:json|csv|tsv|txt|ya?ml|dat|ini))['"]""", v):
                if name != "answer.txt" and not any(
                        os.path.basename(name) in fns for _, _, fns in os.walk(t.dir)):
                    t.p_feasible, t.why_infeasible = 0.05, f"{k} needs {name}, which is missing"
    libs = {k: v for k, v in srcs.items() if not (k.startswith("test") or k.endswith("_test.py"))}
    for k, v in srcs.items():
        if k.startswith("test") or k.endswith("_test.py"):
            why, proven = contradictory_asserts(v, libs)
            if why:
                t.p_feasible, t.why_infeasible = 0.05, f"{k}: {why}"
                t.certain = proven


def _pure_functions(libs: Dict[str, str]) -> set:
    """Top-level functions that read nothing but their own arguments, locals and
    builtins: calling one twice with the same literals must give the same result."""
    import ast
    import builtins
    pure = set()
    for src in libs.values():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or fn.decorator_list:
                continue
            a = fn.args
            known = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            known |= {x.arg for x in (a.vararg, a.kwarg) if x}
            if a.defaults or a.kw_defaults:
                continue  # a mutable default can carry state between calls
            ok = True
            for n in ast.walk(fn):
                if n is not fn and isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom,
                                                  ast.Import, ast.ImportFrom, ast.FunctionDef,
                                                  ast.Lambda, ast.ClassDef)):
                    ok = False
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    known.add(n.id)
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                        and n.id not in known and not hasattr(builtins, n.id):
                    ok = False
                if isinstance(n, ast.Name) and n.id in ("open", "input", "print", "eval", "exec",
                                                         "globals", "locals", "id", "hash", "iter", "next"):
                    ok = False
            if ok:
                pure.add(fn.name)
    return pure


def contradictory_asserts(src: str, libs: Optional[Dict[str, str]] = None):
    """Finds tests that can never all pass. Returns (reason, proven).

    proven=True only for a pure library function called with the same literal arguments
    and expected to return two different literals. Anything else that merely looks
    contradictory (method calls on objects, which may be stateful; checks of __name__)
    is reported with proven=False and gets a model attempt instead of being skipped."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "", False
    pure = _pure_functions(libs or {})
    modules = {os.path.splitext(k)[0] for k in (libs or {})}
    aliases, direct = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            aliases |= {a.asname or a.name for a in n.names if a.name in modules}
        elif isinstance(n, ast.ImportFrom) and n.module in modules:
            direct |= {a.name for a in n.names if not a.asname}

    def lib_func(call):
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases:
            return f.attr
        if isinstance(f, ast.Name) and f.id in direct:
            return f.id
        return None

    def literal_args(call):
        try:
            for a in call.args:
                ast.literal_eval(a)
            return not call.keywords
        except Exception:
            return False

    guess = ""
    seen = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        meth, args = node.func.attr, node.args
        if meth in ("assertEqual", "assertEquals") and len(args) >= 2:
            call, exp = args[0], args[1]
            if not isinstance(call, (ast.Call, ast.Attribute)):
                call, exp = args[1], args[0]
        elif meth in ("assertTrue", "assertFalse") and args:
            call, exp = args[0], ast.Constant(meth == "assertTrue")
        else:
            continue
        try:
            val = ast.literal_eval(exp)
        except Exception:
            continue
        if (isinstance(call, ast.Attribute) and call.attr == "__name__"
                and isinstance(call.value, ast.Attribute) and call.value.attr != val):
            guess = guess or f"{ast.unparse(call)} must equal {val!r}"
            continue
        if not isinstance(call, ast.Call) or lib_func(call) is None:
            continue  # methods of objects built in the test are stateful by nature
        # assertTrue(f(1)) and assertEqual(f(1), 2) do not contradict: 2 is truthy
        key = ("truth:" if meth in ("assertTrue", "assertFalse") else "eq:") + ast.dump(call)
        if key in seen and seen[key] != val:
            why = f"{ast.unparse(call)} expected both {seen[key]!r} and {val!r}"
            if lib_func(call) in pure and literal_args(call):
                return why, True
            guess = guess or why
        seen.setdefault(key, val)
    return guess, False
