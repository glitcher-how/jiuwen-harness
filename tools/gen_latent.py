"""Coding tasks with a trap: besides the bug that makes a visible test fail, one more
function is wrong in a way its (weak) visible test does not catch. Hidden tests are
stricter. An agent that only makes the visible tests pass fails these.

python tools/gen_latent.py OUT_DIR [--n 40] [--seed 1] [--clean]

--clean generates the same tasks without the hidden bug (the weak test stays), to measure
whether a review step damages code that was already right.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_tasks import FUNCS, HIDDEN, TAIL_DONE, w  # noqa: E402

# name -> (wrong implementation that passes the weak test, weak visible test)
LATENT = {
    "average": ("def average(xs):\n    return sum(xs) // len(xs)\n",
                "self.assertEqual(m.average([2, 4, 6]), 4)"),
    "first_n": ("def first_n(xs, n):\n    return xs[:3]\n",
                "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0 and x > 0)\n",
                   "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"),
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(sorted(s.split(), reverse=True))\n",
                      "self.assertEqual(m.reverse_words('a b c'), 'c b a')"),
    "is_palindrome": ("def is_palindrome(s):\n    return s == s[::-1]\n",
                      "self.assertTrue(m.is_palindrome('level'))"),
    "dedupe": ("def dedupe(xs):\n    return sorted(set(xs))\n",
               "self.assertEqual(m.dedupe([1, 1, 2, 3]), [1, 2, 3])"),
    "clamp": ("def clamp(x, lo, hi):\n    return min(x, hi)\n",
              "self.assertEqual(m.clamp(15, 0, 10), 10)"),
    "total": ("def total(xs):\n    return sum(abs(x) for x in xs)\n",
              "self.assertEqual(m.total([1, 2, 3]), 6)"),
    "maximum": ("def maximum(xs):\n    return max(xs[:3])\n",
                "self.assertEqual(m.maximum([3, 9, 2]), 9)"),
    "word_count": ("def word_count(s):\n    return len(s.split(' '))\n",
                   "self.assertEqual(m.word_count('two words'), 2)"),
}
EXTRA_HIDDEN = {
    "reverse_words": "self.assertEqual(m.reverse_words('c a b'), 'b a c')",
    "total": "self.assertEqual(m.total([1, 2, 3]), 6)",
    "maximum": "self.assertEqual(m.maximum([1, 2, 3, 10]), 10)",
    "average": "self.assertEqual(m.average([2, 4, 6]), 4)",
    "first_n": "self.assertEqual(m.first_n([1, 2, 3, 4], 2), [1, 2])",
    "count_even": "self.assertEqual(m.count_even([0, 3, -2]), 2)",
    "is_palindrome": "self.assertTrue(m.is_palindrome('Never odd or even'))",
    "dedupe": "self.assertEqual(m.dedupe([3, 1, 3, 2, 1]), [3, 1, 2])",
    "clamp": "self.assertEqual(m.clamp(-3, 0, 10), 0)",
    "word_count": "self.assertEqual(m.word_count('  two   words '), 2)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for i in range(a.n):
        names = rng.sample(sorted(FUNCS), rng.randint(4, 6))
        failing = names[0]
        latent = names[1]
        src, tests = [], []
        for nm in names:
            good, bad, test = FUNCS[nm]
            if nm == failing:
                src.append(bad)
            elif nm == latent:
                src.append(good if a.clean else LATENT[nm][0])
                test = LATENT[nm][1]
            else:
                src.append(good)
            tests.append(f"    def test_{nm}(self):\n        {test}\n")
        hidden = "\n".join(f"    def test_{nm}(self):\n        {HIDDEN[nm]}\n        {EXTRA_HIDDEN[nm]}\n"
                           for nm in names)
        head = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
        tail = "\n\nif __name__ == '__main__':\n    unittest.main()\n"
        tid = f"coding_{i:04d}"
        d = os.path.join(a.out, tid)
        w(os.path.join(d, "workspace", "mylib.py"), "\n\n".join(src))
        w(os.path.join(d, "workspace", "test_mylib.py"), head + "\n".join(tests) + tail)
        w(os.path.join(d, "hidden", "test_hidden.py"), head.replace("class T", "class H") + hidden + tail)
        w(os.path.join(d, "task.json"), json.dumps({
            "task_id": tid, "task_type": "coding",
            "instruction": "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the "
                           "bugs in mylib.py (do not edit the tests) until all tests pass." + TAIL_DONE,
            "reward": rng.randint(1, 5), "expected": "tests_pass", "check": "unittest",
            "feasible": True, "latent": None if a.clean else latent}, indent=2))
    print(f"wrote {a.n} tasks to {a.out}")


if __name__ == "__main__":
    main()
