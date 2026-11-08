"""Stress set: the same four task kinds, but worded and laid out differently from the
organisers' samples (other file names, column and field names, paraphrased questions,
classes instead of functions). The zero-token solvers should decline these and the model
path should still solve them. All tasks are feasible.

python tools/gen_variants.py OUT_DIR [--n 40] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_tasks import CITIES, COMPANIES, FIRST, LAST, w  # noqa: E402

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer "
        "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
TAIL_DONE = TAIL.replace("the final answer", "the word DONE")


def gen_math(rng):
    n = rng.randint(900, 4000)
    x = n
    noun = rng.choice(["crates", "pallets", "parcels"])
    parts = [f"A depot begins the week holding {n} {noun}."]
    for _ in range(rng.randint(2, 4)):
        op = rng.choice(["out", "in", "third", "triple", "tax"])
        if op == "out":
            a, d = rng.randint(5, 40), rng.randint(2, 5)
            x -= a * d
            parts.append(f"Over the next {d} days it sends out {a} {noun} every day.")
        elif op == "in":
            k, m = rng.randint(2, 5), rng.randint(30, 120)
            x += k * m
            parts.append(f"A supplier then brings {k} truckloads, {m} {noun} in each.")
        elif op == "third":
            k = rng.randint(3, 5)
            x -= x // k
            parts.append(f"One {k}th of the stock (rounded down) is moved to another site.")
        elif op == "triple":
            x *= 3
            parts.append("After a merger the stock becomes three times as large.")
        else:
            p = rng.choice([10, 20, 25])
            x -= x * p // 100
            parts.append(f"An audit writes off {p} percent of the {noun}, rounding the written-off amount down.")
    q = " ".join(parts) + f" What is the number of {noun} left in the depot? Compute it with the bash tool (python3 -c)."
    return q + TAIL, "exact_int", str(x), {}


def gen_data(rng):
    nrows = rng.choice([40, 250])
    cities, items = ["Lyon", "Oslo", "Kazan"], ["bolt", "nut", "washer", "gear"]
    rows = [[i, rng.choice(cities), rng.choice(items), rng.randint(1, 4), rng.randint(1, 30),
             round(rng.uniform(1, 50), 2)] for i in range(1, nrows + 1)]
    text = "id;city;item;quarter;qty;unit_cost\n" + "\n".join(";".join(map(str, r)) for r in rows) + "\n"
    it, c, qn = rng.choice(items), rng.choice(cities), rng.randint(1, 4)
    kind = rng.choice(["qty_city", "orders_q", "top_item", "distinct"])
    if kind == "qty_city":
        q = f"What is the total qty of item \"{it}\" shipped to {c}?"
        ans = sum(r[4] for r in rows if r[2] == it and r[1] == c)
    elif kind == "orders_q":
        q = f"Count the rows where quarter is {qn} and qty is greater than 15."
        ans = sum(1 for r in rows if r[3] == qn and r[4] > 15)
    elif kind == "top_item":
        q = f"Which single-row qty is the highest for city {c}? Give the number."
        ans = max(r[4] for r in rows if r[1] == c)
    else:
        q = f"How many different items were shipped to {c} in quarter {qn}?"
        ans = len({r[2] for r in rows if r[1] == c and r[3] == qn})
    instr = (f"The file shipments.csv is semicolon-separated with columns id;city;item;quarter;qty;unit_cost "
             f"({nrows} rows). {q} Use bash with python3 to compute it.")
    return instr + TAIL, "exact_int", str(ans), {"shipments.csv": text}


def gen_search(rng):
    people = sorted({f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(40)})[:24]
    comp = {c: {"city": rng.choice(CITIES), "since": rng.randint(1950, 2020)} for c in COMPANIES}
    info = {p: {"year": rng.randint(1950, 2002), "org": rng.choice(COMPANIES),
                "boss": rng.choice([x for x in people if x != p])} for p in people}
    files = {}
    for c, d in comp.items():
        files[f"docs/orgs/{c.lower().replace(' ', '-')}.md"] = (
            f"{c}\n{'=' * len(c)}\n\n- Main office: {d['city']}\n- Operating since: {d['since']}\n")
    for p, d in info.items():
        files[f"docs/staff/{p.lower().replace(' ', '-')}.md"] = (
            f"{p}\n{'=' * len(p)}\n\n- Year of birth: {d['year']}\n- Works at: {d['org']}\n- Reports to: {d['boss']}\n")
    p = rng.choice(people)
    d = info[p]
    kind = rng.choice(["year", "office", "boss_org", "since"])
    if kind == "year":
        q, ans = f"What is {p}'s year of birth?", d["year"]
    elif kind == "office":
        q, ans = f"Where is the main office of the organisation {p} works at?", comp[d["org"]]["city"]
    elif kind == "boss_org":
        q, ans = f"{p} reports to someone. Which organisation does that person work at?", info[d["boss"]]["org"]
    else:
        q, ans = f"Since which year has the employer of {p} been operating?", comp[d["org"]]["since"]
    instr = (f"The folder docs/ holds notes about staff and organisations. {q} "
             f"Use the file tools or bash (ls, grep, cat) to find it.")
    return instr + TAIL, "exact_str", str(ans), files


CLASS_SRC = '''class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        return self.items.pop(%(pop)s)

    def peek(self):
        return self.items[%(peek)s]

    def size(self):
        return len(self.items)%(size)s


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n %% 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / %(med)s


def initials(name):
    return ''.join(%(ini)s for p in name.split())
'''

CLASS_TESTS = '''import unittest

from textutils import Stack, median, initials


class StackTests(unittest.TestCase):
    def test_lifo(self):
        s = Stack()
        for v in (1, 2, 3):
            s.push(v)
        self.assertEqual(s.peek(), 3)
        self.assertEqual(s.pop(), 3)
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.size(), 1)


class FuncTests(unittest.TestCase):
    def test_median(self):
        self.assertEqual(median([5, 1, 3]), 3)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_initials(self):
        self.assertEqual(initials('ada king lovelace'), 'AKL')


if __name__ == '__main__':
    unittest.main()
'''


def gen_coding(rng):
    good = {"pop": "", "peek": "-1", "size": "", "med": "2", "ini": "p[0].upper()"}
    bad = {"pop": "0", "peek": "0", "size": " - 1", "med": "3", "ini": "p[-1].upper()"}
    broken = set(rng.sample(sorted(good), rng.randint(1, 3)))
    src = CLASS_SRC % {k: (bad[k] if k in broken else good[k]) for k in good}
    instr = ("The test suite in test_textutils.py does not pass (run `python3 test_textutils.py`). "
             "Repair textutils.py so that it does; the test file must stay as it is.")
    return instr + TAIL_DONE, "unittest", "tests_pass", {"textutils.py": src, "test_textutils.py": CLASS_TESTS}


GENS = {"math": gen_math, "data_analysis": gen_data, "search": gen_search, "coding": gen_coding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    kinds = sorted(GENS)
    for i in range(a.n):
        kind = kinds[i % 4]
        instr, check, expected, files = GENS[kind](rng)
        tid = f"{kind}_{i:04d}"
        d = os.path.join(a.out, tid)
        os.makedirs(d, exist_ok=True)
        for rel, text in files.items():
            w(os.path.join(d, "workspace", rel), text)
        w(os.path.join(d, "task.json"), json.dumps({
            "task_id": tid, "task_type": kind, "instruction": instr, "reward": rng.randint(1, 5),
            "expected": expected, "check": check, "feasible": True}, indent=2))
    print(f"wrote {a.n} variant tasks to {a.out}")


if __name__ == "__main__":
    main()
