"""The smallest program on the package (demo/hello.bend) against transformers' greedy tokens, for the model MODEL
(models.py).

  reference/.venv/bin/python reference/test_hello.py <compiled hello>

hello must print, one per line, the ids of bench/expected-<MODEL>.txt: transformers' greedy continuation of the
benchmark prompt (bench_ref.py writes that file). `make check` builds it from a copy of the package's files alone
(.scratch/pkg), so an import from outside the package does not compile. Its weights come through main.bend's
plain-Bend loader, which no other test generates on.
"""
import os, subprocess, sys, time

from models import ROOT, D, MODEL

EXE = sys.argv[1]
PROMPT = "Once upon a time"  # bench_ref.py's


def main():
    exp = os.path.join(ROOT, "bench", f"expected-{MODEL}.txt")
    want = [int(x) for line in open(exp) if not line.startswith("#") for x in line.split()]
    t0 = time.time()
    r = subprocess.run([EXE, D, PROMPT, str(len(want))], capture_output=True, text=True, cwd=ROOT)
    dt = time.time() - t0
    lines = r.stdout.split()
    got = [int(x) for x in lines] if r.returncode == 0 and all(x.isdigit() for x in lines) else None
    ok = got == want and len(want) > 0
    print(f"{'ok  ' if ok else 'ERROR'} {PROMPT!r} ({len(want)} ids, {dt:.1f} s): ids {'same' if ok else 'DIFFER'}")
    if not ok:
        print(f"  bend:  {got}\n  torch: {want}\n{r.stdout[-500:]}{r.stderr[-2000:]}")
    print("FAILURES:", int(not ok))
    sys.exit(0 if ok else 1)


main()
