"""The prompt in one pass (llama/prompt.bend) against the prompt position by position, in the same program
(demo/gen.bend with GEN_SEQ=1 in the environment), for the model MODEL (models.py).

  reference/.venv/bin/python reference/test_prompt.py <compiled gen>

For prompts of 1, 2, 7 and 77 ids and one chat turn, on one thread and on 16: the two runs must print the same
prompt ids, the same generated ids with the same logit bits, and the same text. Eight tokens are generated after
each prompt, so every layer's cache is read back: a cache that differed in one number would move a logit bit.
A prompt of one id takes the position-by-position path in both runs; it is here so that the switch between the two
paths is tested at its edge.
"""
import os, re, subprocess, sys

from models import ROOT, D

EXE = sys.argv[1]
N_NEW = 8
P56 = ("The history of computing is a history of abstraction. Early machines were programmed by rewiring them, then "
       "with punched cards, then with assembly language, and later with compilers that turned formulas into "
       "instructions. Each step hid a layer of detail and let people think about larger problems.")
CASES = [
    (1, ["Hello"]),
    (2, ["Hello world"]),
    (7, ["The capital of France is Paris,"]),
    (77, [P56 + " A language that proves what it runs is one more step on the same road, and it starts here with"]),
    (37, ["--chat", "What is the capital of France?"]),
]


def run(args, threads, seq):
    env = dict(os.environ, BENDRTS=f"--threads {threads}", BEND_NO_TELEMETRY="1")
    env.pop("GEN_SEQ", None)
    if seq:
        env["GEN_SEQ"] = "1"
    r = subprocess.run([EXE, D] + args + [str(N_NEW)], capture_output=True, cwd=ROOT, env=env)
    out = r.stdout.decode("utf-8", "replace")
    ids = re.search(r"^prompt ids: (.*)$", out, re.M)
    toks = re.findall(r"^  id (\d+)  logit (\d+) ", out, re.M)
    first = re.search(r"^  id \d+  logit \d+ \S+  t=([0-9.]+)s", out, re.M)
    text = re.search(r"^text: (.*)\n\Z", out, re.M | re.S)
    return dict(exit=r.returncode, ids=ids.group(1).split() if ids else None, toks=toks,
                text=text.group(1) if text else None, first=float(first.group(1)) if first else float("nan"))


def main():
    failures = 0
    for n_ids, args in CASES:
        for threads in (1, 16):
            a, b = run(args, threads, False), run(args, threads, True)
            ok = (a["exit"] == 0 and b["exit"] == 0 and a["ids"] is not None and len(a["ids"]) == n_ids
                  and a["ids"] == b["ids"] and len(a["toks"]) > 0 and a["toks"] == b["toks"] and a["text"] == b["text"])
            failures += not ok
            desc = " ".join(args)
            print(f"{'ok  ' if ok else 'ERROR'} {n_ids:3} ids, {threads:2} threads: {len(a['toks'])} tokens, ids, logit bits and "
                  f"text {'same' if ok else 'DIFFER'}; first token after {a['first']:.2f} s in one pass, {b['first']:.2f} s "
                  f"position by position  {desc[:40]!r}")
            if not ok:
                print(f"  one pass:    exit {a['exit']}, {a['ids'] and len(a['ids'])} prompt ids, {a['toks']}\n"
                      f"  by position: exit {b['exit']}, {b['ids'] and len(b['ids'])} prompt ids, {b['toks']}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
