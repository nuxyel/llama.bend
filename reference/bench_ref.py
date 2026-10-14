"""The ids `make bench` expects: transformers' greedy continuation of the benchmark prompt, in F32 with eager attention
(the same settings as gen_ref.py), for the model MODEL (models.py).

  MODEL=smollm2-135m reference/.venv/bin/python reference/bench_ref.py [n]   -> bench/expected-<MODEL>.txt

The file is committed, so `make bench` needs no Python at run time.
"""
import os, sys
import torch
from forward_ref import HF
from models import ROOT, MODEL

PROMPT = "Once upon a time"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(HF)
    ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=n, pad_token_id=tok.eos_token_id)
    new = g.sequences[0, len(ids):].tolist() if hasattr(g, "sequences") else g[0, len(ids):].tolist()
    out = os.path.join(ROOT, "bench", f"expected-{MODEL}.txt")
    with open(out, "w") as f:
        f.write(f"# transformers, greedy, F32, eager attention: {PROMPT!r}, {len(new)} new tokens\n")
        f.write(" ".join(map(str, new)) + "\n")
    print(out, len(new), "ids:", tok.decode(new)[:80])


if __name__ == "__main__":
    main()
