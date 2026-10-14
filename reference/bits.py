"""Saves, or checks, the exact output of the Bend forward pass and generation, to show that a refactor changes no
bit (milestone 6 checks the typed API against milestone 5's code this way).

  reference/.venv/bin/python reference/bits.py save  <compiled forward> <compiled gen>
  reference/.venv/bin/python reference/bits.py check <compiled forward> <compiled gen>

`save` writes data/<MODEL>/test/bits/ (MODEL: models.py): the forward output (every layer, the final norm and the
logits, as F32 bits) for the 11 prompts of forward_ref.py, and the generated ids with the bits of their logits and
the text for the 12 prompts of gen_ref.py. `check` runs the binaries again and compares byte for byte. Timings printed by gen are left out.
"""
import json, os, re, subprocess, sys

from models import ROOT, D, T_DIR

OUT = os.path.join(T_DIR, "bits")


def outputs(fwd, gen):
    for i in range(11):
        ids = open(os.path.join(T_DIR, f"fwd{i}_ids.txt")).read().split()
        yield f"fwd{i}.txt", subprocess.run([fwd, D] + ids, capture_output=True, cwd=ROOT).stdout
    for i, c in enumerate(json.load(open(os.path.join(T_DIR, "gen.json")))):
        out = subprocess.run([gen, D, c["prompt"], str(c["n"])], capture_output=True, cwd=ROOT).stdout
        out = re.sub(rb"  t=[0-9.]+s", b"", out)
        out = re.sub(rb"^weights loaded in .*\n", b"", out, flags=re.M)
        yield f"gen{i}.txt", out


def main():
    mode, fwd, gen = sys.argv[1:4]
    os.makedirs(OUT, exist_ok=True)
    failures = 0
    for name, out in outputs(fwd, gen):
        path = os.path.join(OUT, name)
        if not out:
            failures += 1
            print(f"ERROR {name}: no output")
        elif mode == "save":
            open(path, "wb").write(out)
            print(f"saved {name} ({len(out):,} bytes)")
        else:
            same = open(path, "rb").read() == out
            failures += not same
            print(f"{'ok  ' if same else 'ERROR'} {name} ({len(out):,} bytes) {'identical' if same else 'DIFFERS'}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
