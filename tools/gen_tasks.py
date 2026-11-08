"""Generate a training set in the organisers' format, including infeasible tasks.

python tools/gen_tasks.py OUT_DIR [--n 100] [--seed 1] [--infeasible 0.2]

task.json gets "expected" (null for infeasible) and "feasible" so tools/grade.py
can score a run. The harness never reads these fields.
"""
from __future__ import annotations

import argparse
import json
import os
import random

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer "
        "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
TAIL_DONE = (" Work only inside the current directory. When you are done, write ONLY the word DONE "
             "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")

FIRST = "Alice Boris Chen Dmitri Farid Grace Hiro Irina Jonas Kira Maya Nikolai Olga Pavel Quinn Rosa Sergei Tanya".split()
LAST = "Smith Novak Haddad Wang Kowalski Petrov Sato Garcia Muller Ivanova".split()
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio",
             "Fjord Data", "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Tbilisi", "Osaka", "Lisbon", "Oslo", "Austin", "Porto Alegre", "Kazan", "Seoul",
          "Lyon", "Novosibirsk", "Bergen", "Krakow"]


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ------------------------------------------------------------------ math
def gen_math(rng, feasible):
    n = rng.randint(800, 3000)
    x = n
    parts = [f"A warehouse starts with {n} boxes."]
    for _ in range(rng.randint(2, 5)):
        op = rng.choice(["ship", "recv", "split", "double", "lose", "sell_half_up"])
        if op == "ship":
            a, d = rng.randint(5, 60), rng.randint(2, 6)
            x -= a * d
            parts.append(f"It ships {a} boxes per day for {d} days.")
        elif op == "recv":
            k, m = rng.randint(1, 4), rng.randint(20, 150)
            x += k * m
            parts.append(f"It receives {k} deliveries of {m} boxes each.")
        elif op == "split":
            k = rng.randint(2, 4)
            x //= k
            parts.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down).")
        elif op == "double":
            x *= 2
            parts.append("Then the stock is doubled by merging with a sister warehouse.")
        elif op == "lose":
            p = rng.choice([10, 20, 25, 50])
            x -= x * p // 100
            parts.append(f"Then {p}% of the boxes are damaged and thrown away (round the damaged count down).")
        else:
            x -= (x + 1) // 2
            parts.append("Then half of the boxes are sold (round the sold count up).")
    if not feasible:
        bad = rng.choice(["zero", "unknown"])
        if bad == "zero":
            parts.insert(2, "Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).")
        else:
            parts[0] = "A warehouse starts with an unrecorded number of boxes."
        x = None
    q = " ".join(parts) + " How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    return q + TAIL, ("exact_int", None if x is None else str(x)), {}


# ------------------------------------------------------------------ data
def gen_data(rng, feasible):
    nrows = rng.choice([20, 50, 300, 1000])
    regions, products = ["north", "south", "east", "west"], ["widget", "gadget", "sprocket", "gizmo"]
    use_products = products[:3]
    rows = []
    for i in range(1, nrows + 1):
        rows.append([i, rng.choice(regions), rng.choice(use_products), rng.randint(1, 12),
                     rng.randint(1, 50), rng.randint(5, 99)])
    csvtext = "order_id,region,product,month,units,price\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n"
    kind = rng.choice(["units_prod_months", "revenue_region", "count_region_prod", "max_units_prod", "avg_price_prod"])
    p, r = rng.choice(use_products), rng.choice(regions)
    a = rng.randint(1, 10)
    b = rng.randint(a, 12)
    if kind == "units_prod_months":
        q = f"How many units of product '{p}' were sold in months {a} through {b} inclusive?"
        ans = sum(x[4] for x in rows if x[2] == p and a <= x[3] <= b)
    elif kind == "revenue_region":
        q = f"What is the total revenue (units * price) of region '{r}'?"
        ans = sum(x[4] * x[5] for x in rows if x[1] == r)
    elif kind == "count_region_prod":
        q = f"How many orders of product '{p}' came from region '{r}'?"
        ans = sum(1 for x in rows if x[1] == r and x[2] == p)
    elif kind == "max_units_prod":
        q = f"What is the largest number of units in a single order of product '{p}'?"
        ans = max([x[4] for x in rows if x[2] == p] or [0])
    else:
        q = f"What is the average price of product '{p}' orders, rounded down to an integer?"
        sel = [x[5] for x in rows if x[2] == p]
        ans = sum(sel) // len(sel) if sel else 0
    files = {"sales.csv": csvtext}
    if not feasible:
        bad = rng.choice(["product", "column", "file"])
        if bad == "product":
            q = f"How many units of product 'gizmo-pro' were sold in months {a} through {b} inclusive?"
        elif bad == "column":
            q = f"What is the total discount given in region '{r}' (use the discount column)?"
        else:
            files = {"sales_2025.csv": csvtext}
        ans = None
    instr = (f"The file sales.csv has columns order_id,region,product,month,units,price ({nrows} rows). "
             f"{q} Use bash with python3 to compute it.")
    return instr + TAIL, ("exact_int", None if ans is None else str(ans)), files


# ------------------------------------------------------------------ search
def gen_search(rng, feasible):
    people = set()
    while len(people) < rng.randint(20, 30):
        people.add(f"{rng.choice(FIRST)} {rng.choice(LAST)}")
    people = sorted(people)
    comp = {c: {"Headquarters": rng.choice(CITIES), "Founded": rng.randint(1950, 2020)} for c in COMPANIES}
    info = {}
    for pn in people:
        info[pn] = {"Born": rng.randint(1950, 2002), "Employer": rng.choice(COMPANIES),
                    "Mentor": rng.choice([x for x in people if x != pn])}
    files = {}
    for c, d in comp.items():
        files[f"kb/company_{c.lower().replace(' ', '_')}.md"] = f"# {c}\n\nHeadquarters: {d['Headquarters']}\nFounded: {d['Founded']}\n"
    for pn, d in info.items():
        files[f"kb/person_{pn.lower().replace(' ', '_')}.md"] = (
            f"# {pn}\n\nBorn: {d['Born']}\nEmployer: {d['Employer']}\nMentor: {d['Mentor']}\n")
    pn = rng.choice(people)
    d = info[pn]
    kind = rng.choice(["born", "employer", "hq", "founded", "mentor_employer", "mentor_born", "mentor_hq"])
    if kind == "born":
        q, ans = f"In which year was {pn} born?", d["Born"]
    elif kind == "employer":
        q, ans = f"Which company does {pn} work for?", d["Employer"]
    elif kind == "hq":
        q, ans = f"In which city is the headquarters of {pn}'s employer?", comp[d["Employer"]]["Headquarters"]
    elif kind == "founded":
        q, ans = f"In which year was {pn}'s employer founded?", comp[d["Employer"]]["Founded"]
    elif kind == "mentor_employer":
        q, ans = f"Which company employs the mentor of {pn}?", info[d["Mentor"]]["Employer"]
    elif kind == "mentor_born":
        q, ans = f"In which year was {pn}'s mentor born?", info[d["Mentor"]]["Born"]
    else:
        q, ans = (f"In which city is the headquarters of the company that employs {pn}'s mentor?",
                  comp[info[d["Mentor"]]["Employer"]]["Headquarters"])
    if not feasible:
        bad = rng.choice(["person", "field"])
        if bad == "person":
            ghost = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            while ghost in people:
                ghost = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            q = f"In which year was {ghost} born?"
        else:
            q = f"What is the annual salary of {pn}?"
        ans = None
    instr = (f"The directory kb/ contains a small knowledge base of markdown files. {q} "
             f"Use the file tools or bash (ls, grep, cat) to find it.")
    return instr + TAIL, ("exact_str", None if ans is None else str(ans)), files


# ------------------------------------------------------------------ coding
FUNCS = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                      "self.assertEqual(m.reverse_words('c a b'), 'b a c')"),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
              "self.assertEqual(m.total([1, 2, 3]), 6)"),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                "self.assertEqual(m.maximum([3, 9, 2]), 9)"),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                "def average(xs):\n    return sum(xs) / (len(xs) + 1)\n",
                "self.assertEqual(m.average([2, 4, 6]), 4)"),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                "def first_n(xs, n):\n    return xs[:n - 1]\n",
                "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
                   "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"),
    "is_palindrome": ("def is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n",
                      "def is_palindrome(s):\n    t = ''.join(c for c in s if c.isalnum())\n    return t == t[::-1]\n",
                      "self.assertTrue(m.is_palindrome('Never odd or even'))"),
    "dedupe": ("def dedupe(xs):\n    seen, out = set(), []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
               "def dedupe(xs):\n    return sorted(set(xs))\n",
               "self.assertEqual(m.dedupe([3, 1, 3, 2, 1]), [3, 1, 2])"),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              "def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
              "self.assertEqual(m.clamp(15, 0, 10), 10)\n        self.assertEqual(m.clamp(-3, 0, 10), 0)"),
    "word_count": ("def word_count(s):\n    return len(s.split())\n",
                   "def word_count(s):\n    return len(s.split(' '))\n",
                   "self.assertEqual(m.word_count('  two   words '), 2)"),
}


def gen_coding(rng, feasible):
    names = rng.sample(sorted(FUNCS), rng.randint(4, 6))
    buggy = set(rng.sample(names, rng.randint(1, 3)))
    src = "\n\n".join(FUNCS[n][1] if n in buggy else FUNCS[n][0] for n in names)
    tests = [f"    def test_{n}(self):\n        {FUNCS[n][2]}\n" for n in names]
    if not feasible:
        n = rng.choice(names)
        tests.append(f"    def test_{n}_again(self):\n        "
                     + {"total": "self.assertEqual(m.total([1, 2, 3]), 7)",
                        "maximum": "self.assertEqual(m.maximum([3, 9, 2]), 3)",
                        "average": "self.assertEqual(m.average([2, 4, 6]), 5)",
                        }.get(n, f"self.assertEqual(m.{n}.__name__, 'other_name')") + "\n")
    test_src = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n" + "\n".join(tests) + \
               "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    instr = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
             "(do not edit the tests) until all tests pass.")
    return instr + TAIL_DONE, ("unittest", "tests_pass" if feasible else None), \
        {"mylib.py": src, "test_mylib.py": test_src}


GENS = {"math": gen_math, "data_analysis": gen_data, "search": gen_search, "coding": gen_coding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--infeasible", type=float, default=0.2)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    kinds = sorted(GENS)
    for i in range(a.n):
        kind = kinds[i % 4] if i < 8 else rng.choice(kinds)
        feasible = rng.random() >= a.infeasible
        instr, (check, expected), files = GENS[kind](rng, feasible)
        tid = f"{kind}_{i:04d}"
        d = os.path.join(a.out, tid)
        os.makedirs(d, exist_ok=True)
        for rel, text in files.items():
            w(os.path.join(d, "workspace", rel), text)
        w(os.path.join(d, "task.json"), json.dumps({
            "task_id": tid, "task_type": kind, "instruction": instr,
            "reward": rng.randint(1, 5), "expected": expected, "check": check,
            "feasible": feasible}, indent=2))
        if check == "unittest" and feasible:
            # hidden tests: different inputs than the visible ones
            w(os.path.join(d, "hidden", "test_hidden.py"), _hidden_tests(files["mylib.py"]))
    print(f"wrote {a.n} tasks to {a.out}")


HIDDEN = {
    "reverse_words": "self.assertEqual(m.reverse_words('one two three'), 'three two one')",
    "total": "self.assertEqual(m.total([5, -2]), 3)",
    "maximum": "self.assertEqual(m.maximum([-5, -1, -9]), -1)",
    "average": "self.assertEqual(m.average([1, 2]), 1.5)",
    "first_n": "self.assertEqual(m.first_n([1, 2], 5), [1, 2])",
    "count_even": "self.assertEqual(m.count_even([0, 3]), 1)",
    "is_palindrome": "self.assertFalse(m.is_palindrome('abca'))",
    "dedupe": "self.assertEqual(m.dedupe(['b', 'a', 'b']), ['b', 'a'])",
    "clamp": "self.assertEqual(m.clamp(5, 0, 10), 5)",
    "word_count": "self.assertEqual(m.word_count(''), 0)",
}


def _hidden_tests(src):
    names = [n for n in HIDDEN if f"def {n}(" in src]
    body = "\n".join(f"    def test_{n}(self):\n        {HIDDEN[n]}\n" for n in names)
    return "import unittest\n\nimport mylib as m\n\n\nclass H(unittest.TestCase):\n" + body + \
        "\n\nif __name__ == '__main__':\n    unittest.main()\n"


if __name__ == "__main__":
    main()
