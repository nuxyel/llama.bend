"""Decoding in Bend (demo/tok.bend --decode) against `tokenizers`' decode(skip_special_tokens = True).

  reference/.venv/bin/python reference/test_detok.py <compiled tok>

Cases: the ids of test_tok.py's fixed texts (decoding gives the text back), and random id sequences drawn from
the whole vocabulary, the special tokens (0-16) and the byte tokens (17-251), so that many of them are invalid
UTF-8 and exercise the replacement character. The output is compared byte for byte.
"""
import os, random, subprocess, sys, tempfile
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from models import D  # noqa: E402  (the tokenizer tables of MODEL)
EXE = sys.argv[1]
from test_tok import CASES  # noqa: E402


def run(ids):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(" ".join(map(str, ids)))
    try:
        r = subprocess.run([EXE, os.path.join(D, "merges_num.txt"), os.path.join(D, "perm.txt"), "--decode", f.name],
                           capture_output=True)
    finally:
        os.unlink(f.name)
    out = r.stdout
    return out[:-1] if out.endswith(b"\n") else out


def main():
    tok = Tokenizer.from_file(os.path.join(D, "hf", "tokenizer.json"))
    rng = random.Random(int(os.environ.get("FUZZ_SEED", "7")))
    cases = [tok.encode(t, add_special_tokens=False).ids for t in CASES]
    for _ in range(int(os.environ.get("FUZZ_CASES", "40"))):
        n = rng.randint(1, 30)
        pick = [lambda: rng.randrange(49152), lambda: rng.randrange(17), lambda: rng.randrange(17, 252)]
        cases.append([rng.choice(pick)() for _ in range(n)])
    failures = 0
    for ids in cases:
        want = tok.decode(ids, skip_special_tokens=True).encode("utf-8")
        got = run(ids)
        if got != want:
            failures += 1
            print(f"ERROR {ids}\n  bend:  {got!r}\n  want:  {want!r}")
    print(f"{len(cases)} cases ({len(CASES)} texts, {len(cases) - len(CASES)} random)")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
