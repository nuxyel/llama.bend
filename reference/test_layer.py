"""One layer in Bend (demo/layer_test.bend) against transformers (reference/layer_ref.py), for the model MODEL.

  reference/.venv/bin/python reference/test_layer.py <compiled layer_test>

The Bend program prints `name t b0 b1 ...` (the exact F32 bits of an intermediate at position t) and
`inv_freq b0 ... b31`. For every intermediate and position: max|bend - ref| / max|ref| must be <= TOL.

TOL is 1e-5 for SmolLM2-135M (milestone 3) and grows with the square root of the longest product, max(hidden,
MLP width) / 1536: the products sum their terms in a fixed sequential order (bend-ml-tensor-array's kernel), whose
rounding error grows like sqrt(n), while PyTorch's blocked sums err less. Against a float64 run of the same layer,
the remaining difference is Bend's, in the products (NOTES, 2026-10-10).
"""
import os, subprocess, sys, time
import numpy as np

from models import D, T_DIR, config, sizes

EXE = sys.argv[1]
CASES = [("A_L0", 0), ("A_L15", 15), ("B_L0", 0)]
HIDDEN, _, KV, HD = sizes()[:4]
TOL = 1e-5 * max(1.0, max(HIDDEN, config()["intermediate_size"]) / 1536) ** 0.5
WIDTH = {"k": KV * HD, "v": KV * HD, "kr": KV * HD}
ORDER = ["n1", "q", "k", "v", "qr", "kr", "attn", "a", "n2", "m", "y"]


def ref(case, name):
    return np.fromfile(os.path.join(T_DIR, f"{case}_{name}.bin"), "<f4").reshape(-1, WIDTH.get(name, HIDDEN))


def run(case, layer):
    t_in = ref(case, "in").shape[0]
    t0 = time.time()
    r = subprocess.run([EXE, D, str(layer), os.path.join(T_DIR, f"{case}_in.bin"), str(t_in)], capture_output=True, text=True)
    dt = time.time() - t0
    rows, inv = {}, None
    for line in r.stdout.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "inv_freq":
            inv = np.array([int(x) for x in p[1:]], dtype=np.uint32).view("<f4")
        elif p[0] in ORDER:
            rows.setdefault(p[0], {})[int(p[1])] = np.array([int(x) for x in p[2:]], dtype=np.uint32).view("<f4")
    if r.returncode != 0 or not rows:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return rows, inv, dt, t_in


def main():
    failures = 0
    want_inv = np.fromfile(os.path.join(T_DIR, "inv_freq.bin"), "<f4")
    for case, layer in CASES:
        rows, inv, dt, t_in = run(case, layer)
        print(f"== {case} (layer {layer}, {t_in} positions, {dt:.1f} s, tolerance {TOL:.2e})")
        if inv is not None:
            same = np.array_equal(inv.view(np.uint32), want_inv.view(np.uint32))
            diff = int((inv.view(np.uint32) != want_inv.view(np.uint32)).sum())
            print(f"  {'ok  ' if same else 'DIFF'} inv_freq bit for bit ({diff} of 32 differ)")
        if not rows:
            failures += 1
            print("  ERROR no output")
            continue
        for name in ORDER:
            if name not in rows:
                continue
            want = ref(case, name)
            worst, where = 0.0, -1
            for t in range(t_in):
                got = rows[name].get(t)
                if got is None or got.shape != want[t].shape:
                    worst, where = float("inf"), t
                    break
                rel = float(np.abs(got - want[t]).max() / max(np.abs(want[t]).max(), 1e-30))
                if rel > worst:
                    worst, where = rel, t
            ok = worst <= TOL
            failures += not ok
            print(f"  {'ok  ' if ok else 'ERROR'} {name:5s} worst max|d|/max|y| = {worst:.2e} (position {where})")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
