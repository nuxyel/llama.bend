"""Expected greedy answers to one chat turn, for the model MODEL (models.py), from `transformers`' generate
(do_sample = False) in F32 with eager attention over apply_chat_template's ids.

  reference/.venv/bin/python reference/chat_gen_ref.py

Writes data/<MODEL>/test/chat_gen.json: for every message of MESSAGES as one user turn (the template's default
system message first, the assistant's turn opened), the prompt ids, the N_NEW generated ids (fewer if <|im_end|>, id
2, or the end-of-text token, id 0, comes first: generation stops after either, as demo/gen.bend --chat does), the
logit of each chosen id and the decoded text (prompt and answer, special tokens skipped).

An instruct checkpoint answers the message; a base model continues the template's text as it would any other.
"""
import json, os
import torch
from models import CHAT_TEMPLATE, HF, T_DIR as OUT

N_NEW = 32
MESSAGES = [
    "What is the capital of France?",
    "Hi there!",
    "What is 2 + 2?",
    "Write a haiku about the sea.",
    "List three colors.",
    "Explain what a transformer model is in one sentence.",
    "Translate 'good morning' to Portuguese.",
    "Who wrote Romeo and Juliet?",
    "What does this function do?\ndef add(a, b):\n    return a + b",
    "ação, coração e café: em que língua está esta frase?",
    "Tell me a short story about a robot.",
    "Is 17 a prime number? Answer yes or no.",
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(HF)
    cases = []
    for msg in MESSAGES:
        ids = list(tok.apply_chat_template([{"role": "user", "content": msg}], chat_template=CHAT_TEMPLATE,
                                           add_generation_prompt=True, tokenize=True, return_dict=False))
        with torch.no_grad():
            g = model.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=N_NEW, output_logits=True,
                               return_dict_in_generate=True, eos_token_id=[0, 2], pad_token_id=0)
        new = g.sequences[0, len(ids):].tolist()
        logits = [float(lg[0, i]) for lg, i in zip(g.logits, new)]
        text = tok.decode(ids + new, skip_special_tokens=True)
        cases.append({"message": msg, "n": N_NEW, "ids": ids, "gen": new, "logits": logits, "text": text})
        print(f"{msg!r}: {len(new)} new ids, {tok.decode(new, skip_special_tokens=True)!r}")
    with open(os.path.join(OUT, "chat_gen.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
