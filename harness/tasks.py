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
    r"When you are done,[^.]*?answer\.txt[^.]*?(\([^)]*\))?\.",
    r"Compute it with the bash tool[^.]*\.",
    r"Use bash with python3? to compute it\.",
    r"Use the file tools or bash[^.]*?(\([^)]*\))?[^.]*\.",
]


def clean_instruction(text: str) -> str:
    q = text
    for pat in _BOILER:
        q = re.sub(pat, "", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


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
    inp_chars = len(t.question) + sum(len(v) for v in t.context.values()) + 450
    return (inp_chars / 3.3 / 10.0 + EXPECTED_OUT.get(t.kind, 500)) * ATTEMPTS.get(t.kind, 1.2)


def _an_math(t: Task) -> None:
    if not re.search(r"\d", t.question):
        t.p_feasible, t.why_infeasible = 0.2, "no numbers in problem"
    if re.search(r"\b(?:into|by|among|between|over)\s+0\s+\w+", t.question):
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

    names = {_norm(n.replace("'s", "")) for n in _NAME.findall(t.question.replace("'s ", " "))}
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

    def render(paths):
        if paths:
            chosen = paths[:25]           # only the entity-linked cards
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


# --- data analysis: schema summary instead of raw file

def _an_data(t: Task) -> None:
    files = [f for f in os.listdir(t.cwd) if f.lower().endswith((".csv", ".tsv", ".json", ".txt"))
             and f != "answer.txt"]
    mentioned = re.findall(r"\b([\w-]+\.(?:csv|tsv|json))\b", t.instruction)
    for f in mentioned:
        if not os.path.exists(os.path.join(t.cwd, f)):
            t.p_feasible, t.why_infeasible = 0.05, f"{f} is missing"
            return
    summ = []
    for f in (mentioned or files)[:4]:
        p = os.path.join(t.cwd, f)
        if not f.lower().endswith((".csv", ".tsv")):
            with open(p, encoding="utf-8", errors="replace") as fh:
                summ.append(f"{f}: {fh.read(600)}")
            continue
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh, delimiter="\t" if f.endswith(".tsv") else ","))
        if not rows:
            summ.append(f"{f}: empty")
            t.p_feasible, t.why_infeasible = 0.05, f"{f} is empty"
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
                    lines.append(f"{col}: numeric {min(nums):g}..{max(nums):g}")
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
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", t.question)
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
        return
    t.context["sources"] = "\n\n".join(f"### {k}\n{v}" for k, v in srcs.items())
    for k, v in srcs.items():
        if k.startswith("test") or k.endswith("_test.py"):
            why = contradictory_asserts(v)
            if why:
                t.p_feasible, t.why_infeasible = 0.05, f"{k}: {why}"


def contradictory_asserts(src: str) -> str:
    """Same call asserted to give two different literal results -> tests can never all pass.
    Also catches asserts on a function's __name__ that differ from the def name."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
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
            return f"{ast.unparse(call)} must equal {val!r}"
        if not isinstance(call, ast.Call):
            continue
        key = ast.dump(call)
        if key in seen and seen[key] != val:
            return f"{ast.unparse(call)} expected both {seen[key]!r} and {val!r}"
        seen.setdefault(key, val)
    return ""
