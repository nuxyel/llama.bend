"""The whole forward pass in Bend (demo/forward.bend) against transformers (reference/forward_ref.py), for the model
MODEL (models.py).

  reference/.venv/bin/python reference/test_forward.py <compiled forward>

For every prompt and position: the logits within TOL relative to the largest logit (max|bend - ref| / max|ref|)
and the same argmax. The output of every layer is printed too (worst relative error over the positions), to show
how the error grows with depth; it is not a pass criterion.
"""
import os, subprocess, sys, time
import numpy as np

from models import ROOT, D, T_DIR, sizes

EXE = sys.argv[1]
HIDDEN, LAYERS, VOCAB = sizes()[0], sizes()[4], sizes()[5]
TOL = 2e-4
N_PROMPTS = 11


def ref(i, name, width):
    return np.fromfile(os.path.join(T_DIR, f"fwd{i}_{name}.bin"), "<f4").reshape(-1, width)


def rel(got, want):
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


def run(ids):
    t0 = time.time()
    r = subprocess.run([EXE, D] + [str(x) for x in ids], capture_output=True, text=True, cwd=ROOT)
    dt = time.time() - t0
    rows = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) > 2:
            rows[(p[0], int(p[1]))] = np.array([int(x) for x in p[2:]], dtype=np.uint32).view("<f4")
    if r.returncode != 0 or not rows:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return rows, dt


def main():
    failures = 0
    for i in range(N_PROMPTS):
        ids = [int(x) for x in open(os.path.join(T_DIR, f"fwd{i}_ids.txt")).read().split()]
        t = len(ids)
        rows, dt = run(ids)
        lg, nf, hs = ref(i, "logits", VOCAB), ref(i, "nf", HIDDEN), ref(i, "h", HIDDEN).reshape(LAYERS, t, HIDDEN)
        worst, where, same = 0.0, -1, True
        for p in range(t):
            got = rows.get(("logits", p))
            if got is None or got.shape != lg[p].shape:
                worst, where, same = float("inf"), p, False
                break
            e = rel(got, lg[p])
            if e > worst:
                worst, where = e, p
            same &= int(got.argmax()) == int(lg[p].argmax())
        ok = worst <= TOL and same
        failures += not ok
        print(f"{'ok  ' if ok else 'ERROR'} fwd{i} ({t} positions, {dt:.1f} s): logits worst {worst:.2e} (position {where}), argmax {'same' if same else 'DIFFERS'}")
        drift = []
        for l in range(LAYERS):
            e = max((rel(rows[(f'h{l}', p)], hs[l, p]) for p in range(t) if (f'h{l}', p) in rows), default=float("inf"))
            drift.append(e)
        e_nf = max((rel(rows[("nf", p)], nf[p]) for p in range(t) if ("nf", p) in rows), default=float("inf"))
        print("      layers " + " ".join(f"{e:.0e}" for e in drift) + f" | nf {e_nf:.1e}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
