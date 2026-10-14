"""Greedy generation in Bend (demo/gen.bend) against transformers' generate (reference/gen_ref.py), for the model
MODEL (models.py).

  reference/.venv/bin/python reference/test_gen.py <compiled gen>

For every prompt: the prompt ids (Bend's tokenizer), the generated ids and the decoded text must be identical; the
logit of each chosen id within TOL relative to it.
"""
import json, os, re, subprocess, sys, time

from models import ROOT, D, T_DIR

EXE = sys.argv[1]
TOL = 2e-4


def main():
    cases = json.load(open(os.path.join(T_DIR, "gen.json")))
    failures = 0
    for c in cases:
        t0 = time.time()
        r = subprocess.run([EXE, D, c["prompt"], str(c["n"])], capture_output=True, cwd=ROOT)
        dt = time.time() - t0
        out = r.stdout.decode("utf-8")
        m_ids = re.search(r"^prompt ids: (.*)$", out, re.M)
        p_ids = [int(x) for x in m_ids.group(1).split()] if m_ids else None
        gen = [int(x) for x in re.findall(r"^  id (\d+) ", out, re.M)]
        lgs = [float(x) for x in re.findall(r"^  id \d+  logit \d+ (\S+)", out, re.M)]
        m_txt = re.search(r"^text: (.*)\n\Z", out, re.M | re.S)
        text = m_txt.group(1) if m_txt else None
        ids_ok = p_ids == c["ids"] and gen == c["gen"]
        lg_err = max((abs(a - b) / abs(b) for a, b in zip(lgs, c["logits"])), default=float("inf"))
        lg_ok = len(lgs) == len(c["logits"]) and lg_err <= TOL
        txt_ok = text == c["text"]
        ok = ids_ok and lg_ok and txt_ok
        failures += not ok
        print(f"{'ok  ' if ok else 'ERROR'} {c['prompt']!r} ({len(gen)} ids, {dt:.1f} s): ids {'same' if ids_ok else 'DIFFER'}, "
              f"text {'same' if txt_ok else 'DIFFERS'}, chosen logits within {lg_err:.1e}")
        if not ok:
            print(f"  bend:  {p_ids} {gen}\n  torch: {c['ids']} {c['gen']}\n  bend text:  {text!r}\n  torch text: {c['text']!r}")
            if r.returncode != 0:
                print(r.stderr.decode("utf-8", "replace")[-2000:])
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
