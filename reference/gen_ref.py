"""Expected greedy generations (milestone 5) of the model MODEL (models.py), from `transformers`'
generate(do_sample = False) in F32 with eager attention.

  reference/.venv/bin/python reference/gen_ref.py

Writes data/<MODEL>/test/gen.json: for every prompt of forward_ref.PROMPTS and EXTRA, the prompt ids, the N_NEW generated
ids (fewer if the end-of-text token comes first; generate stops after it), the logit of each chosen id and the
decoded text (prompt and continuation, special tokens skipped).
"""
import json, os
import torch
from forward_ref import HF, OUT, PROMPTS

N_NEW = 32
# generate stops after the end-of-text token (id 0): this prompt produces it as its first new token
EXTRA = ["Copyright 2024. All rights reserved."]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(HF)
    cases = []
    for prompt in PROMPTS + EXTRA:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            g = model.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=N_NEW, output_logits=True,
                               return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        new = g.sequences[0, len(ids):].tolist()
        logits = [float(lg[0, i]) for lg, i in zip(g.logits, new)]
        text = tok.decode(ids + new, skip_special_tokens=True)
        cases.append({"prompt": prompt, "n": N_NEW, "ids": ids, "gen": new, "logits": logits, "text": text})
        print(f"{prompt!r}: {len(new)} new ids, {text!r}")
    with open(os.path.join(OUT, "gen.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
