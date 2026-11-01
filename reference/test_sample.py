"""Sampling in Bend (llama/main.bend's sample and uniform) against the same sampler in Python (sample_ref.py).

  reference/.venv/bin/python reference/test_sample.py <compiled sample_test> <compiled gen>

1. demo/sample_test.bend: logits made from the hash itself (with and without ties, 1 to 49,152 of them), every
   draw's id and logit bits equal to sample_ref.sample's from the same F32 logits and seed.
2. demo/gen.bend on MODEL with GEN_CANDS=1, for a few seeds and settings: every draw equal to
   sample_ref.sample_cands on the candidates Bend printed (the same F32 logits, the same seed), and the candidates
   the top-k of transformers' logits for the same tokens (a teacher-forced forward pass), within TOL; the same seed
   gives the same tokens on 1 thread and on the default count; temp 0 is greedy (gen.json's tokens); invalid
   settings are refused.
"""
import json, os, re, subprocess, sys

import numpy as np
import sample_ref as S
from models import D, HF, ROOT, T_DIR

F = S.F
TOL = 2e-4

# the list of demo/sample_test.bend's cases(): seed, logits, ties, temperature, k, p, draws
SYNTH = [(1, 1, True, 1.0, 1, 1.0, 4), (2, 2, True, 1.0, 2, 1.0, 8), (3, 7, True, 0.5, 3, 0.9, 8),
         (4, 7, False, 1.0, 9, 1.0, 8), (5, 100, True, 1.0, 100, 1.0, 16), (6, 100, True, 2.0, 10, 0.5, 16),
         (7, 1000, False, 0.7, 40, 0.9, 16), (8, 1000, True, 0.7, 40, 0.9, 16), (9, 1000, True, 1.3, 1, 1.0, 4),
         (10, 1000, False, 0.01, 50, 1.0, 8), (11, 1000, False, 100.0, 50, 0.3, 16), (12, 1000, True, 1.0, 200, 0.05, 8),
         (13, 1000, True, 1.0, 1200, 0.95, 16), (14, 49152, False, 0.8, 50, 0.95, 12),
         (15, 49152, True, 1.0, 50, 1.0, 12), (4294967295, 49152, False, 0.6, 20, 0.8, 12)]

# prompt (or --chat message), new tokens, temperature, top-k, top-p, seed
GEN = [("Once upon a time", False, 16, "0.8", "50", "0.95", "1"),
       ("Once upon a time", False, 16, "0.8", "50", "0.95", "2"),
       ("The capital of France is", False, 16, "1.0", "50", "1.0", "3"),
       ("def fibonacci(n):", False, 16, "0.6", "10", "0.8", "4"),
       ("Machine learning is", False, 16, "1.3", "200", "0.9", "4294967295"),
       ("What is the capital of France?", True, 12, "0.7", "20", "0.9", "5")]


def synth(exe):
    out = subprocess.run([exe], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    failures, k = 0, 0
    for c, (seed, n, ties, temp, kk, p, d) in enumerate(SYNTH):
        hs = [S.hash32(seed + j) for j in range(n)]
        xs = [F(F(F(h >> 26) * F(0.25)) - F(4.0)) if ties else F(F(F(h >> 8) * F(9.5367431640625e-7)) - F(8.0)) for h in hs]
        bad, ids = 0, set()
        for j in range(d):
            i, v = S.sample(xs, F(temp), kk, F(p), S.uniform(seed, j))
            ids.add(i)
            want = f"case {c} draw {j} id {i} logit {S.bits(v)}"
            got = out[k] if k < len(out) else None
            k += 1
            if got != want:
                bad += 1
                print(f"  bend: {got!r}\n  python: {want!r}")
        failures += bad > 0
        print(f"{'ok  ' if not bad else 'ERROR'} synthetic case {c}: {n} logits{' with ties' if ties else ''}, temp {temp}, "
              f"k {kk}, p {p}: {d} draws, {len(ids)} distinct ids{'' if not bad else f', {bad} differ'}")
    if k != len(out):
        failures += 1
        print(f"ERROR sample_test printed {len(out)} lines, expected {k}")
    return failures


def run_gen(exe, prompt, chat, n, rest, threads=None, cands=True):
    args = [exe] + (["+RTS", "--threads", str(threads), "-RTS"] if threads else []) + [D] + (["--chat"] if chat else [])
    env = dict(os.environ, GEN_CANDS="1" if cands else "0")
    r = subprocess.run(args + [prompt, str(n)] + list(rest), capture_output=True, cwd=ROOT, env=env)
    return r.stdout.decode("utf-8")


def parse(out):
    m = re.search(r"^sampling: temp (\d+) \S+ top_k (\d+) top_p (\d+) \S+ seed (\d+)$", out, re.M)
    prm = (S.from_bits(int(m.group(1))), int(m.group(2)), S.from_bits(int(m.group(3))), int(m.group(4))) if m else None
    pids = [int(x) for x in re.search(r"^prompt ids: (.*)$", out, re.M).group(1).split()]
    cands = [[(S.from_bits(int(b)), int(i)) for i, b in re.findall(r"(\d+):(\d+)", line)]
             for line in re.findall(r"^  cands (.*)$", out, re.M)]
    draws = [(int(i), int(b)) for i, b in re.findall(r"^  id (\d+)  logit (\d+) ", out, re.M)]
    return prm, pids, cands, draws


def gen(exe):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    failures = 0
    for prompt, chat, n, *rest in GEN:
        out = run_gen(exe, prompt, chat, n, rest)
        prm, pids, cands, draws = parse(out)
        temp, k, p, seed = prm
        same = len(cands) == len(draws) and len(draws) > 0
        for j, (cs, (i, b)) in enumerate(zip(cands, draws)):
            pi, pv = S.sample_cands(cs, temp, p, S.uniform(seed, j))
            same &= (pi, S.bits(pv)) == (i, b)
        ids = pids + [i for i, _ in draws]
        with torch.no_grad():
            lg = model(torch.tensor([ids])).logits[0].numpy()
        err, topk_ok = 0.0, True
        for j, cs in enumerate(cands):
            row = lg[len(pids) - 1 + j]
            chosen = {i for _, i in cs}
            err = max(err, max(abs(float(v) - row[i]) / max(abs(row[i]), 1.0) for v, i in cs))
            rest_max = max((row[i] for i in range(len(row)) if i not in chosen), default=-np.inf)
            topk_ok &= len(cs) == min(k, len(row)) and min(row[i] for i in chosen) >= rest_max - TOL * max(abs(rest_max), 1.0)
        ok = same and err <= TOL and topk_ok
        failures += not ok
        print(f"{'ok  ' if ok else 'ERROR'} {'chat ' if chat else ''}{prompt!r} seed {seed}, temp {float(temp):g}, k {k}, "
              f"p {float(p):g}: {len(draws)} draws {'equal to Python' if same else 'DIFFER from Python'}, candidates "
              f"{'the top-k' if topk_ok else 'NOT the top-k'} of transformers' logits within {err:.1e}")
        if not ok:
            print(out[-3000:])
    # the same seed, the same tokens on one thread
    prompt, chat, n, *rest = GEN[0]
    a = parse(run_gen(exe, prompt, chat, n, rest, cands=False))[3]
    b = parse(run_gen(exe, prompt, chat, n, rest, threads=1, cands=False))[3]
    ok = a == b and len(a) == n
    failures += not ok
    print(f"{'ok  ' if ok else 'ERROR'} seed {GEN[0][-1]}: the same {len(a)} tokens on 1 thread and the default count")
    # temp 0 is greedy
    case = json.load(open(os.path.join(T_DIR, "gen.json")))[1]
    out = run_gen(exe, case["prompt"], False, 8, ["0", "5", "0.5", "7"], cands=False)
    ok = "sampling:" not in out and [i for i, _ in parse(out)[3]] == case["gen"][:8]
    failures += not ok
    print(f"{'ok  ' if ok else 'ERROR'} temp 0 gives the greedy tokens of gen.json for {case['prompt']!r}")
    for rest in (["0.8", "0", "0.9", "1"], ["0.8", "5", "0", "1"], ["0.8", "5", "1.5", "1"], ["-1", "5", "0.9", "1"],
                 ["x", "5", "0.9", "1"]):
        out = run_gen(exe, "hello", False, 2, rest, cands=False)
        ok = out.startswith("usage: ")
        failures += not ok
        print(f"{'ok  ' if ok else 'ERROR'} refused: temp {rest[0]}, top_k {rest[1]}, top_p {rest[2]}")
    return failures


def main():
    failures = synth(sys.argv[1]) + gen(sys.argv[2])
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
