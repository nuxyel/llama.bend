"""Expected greedy answers to conversations of several turns, for the model MODEL (models.py), from `transformers`'
generate (do_sample = False) in F32 with eager attention: every turn is generated from the whole conversation so far.

  reference/.venv/bin/python reference/chat_turns_ref.py

Writes data/<MODEL>/test/chat_turns.json: for every conversation of CONVERSATIONS, and for every user message in it,
the ids of the conversation up to that message (the assistant's turn opened), the N_NEW generated ids (fewer if
<|im_end|>, id 2, or the end-of-text token, id 0, comes first), the logit of each chosen id, the answer's text
(special tokens skipped), and `template`: whether those ids are apply_chat_template's for the messages and the
answers' texts.

The conversation is kept in ids, as a program that keeps its caches holds it (demo/gen.bend --chat -): the ids
before, the last answer's ids without a stop token, then <|im_end|> and what the template puts after it. Those are
apply_chat_template's ids for the texts whenever every answer's text encodes back to its ids. An instruct checkpoint
answers in such text; a base model continues the template's text as it would any other, and what it writes does not
always encode back (`template` is then false from the next turn on).
"""
import json, os
import torch
from models import CHAT_TEMPLATE, HF, T_DIR as OUT

N_NEW = 24
CONVERSATIONS = [
    ["What is the capital of France?", "And of Italy?", "Which of the two is further north?"],
    ["Hi there!", "What is 2 + 2?", "Now multiply that by 3."],
    ["ação, coração e café: em que língua está esta frase?", "Translate it to English.", "Thanks!"],
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(HF)

    def template(messages):
        return list(tok.apply_chat_template(messages, chat_template=CHAT_TEMPLATE, add_generation_prompt=True,
                                            tokenize=True, return_dict=False))

    nl = tok.encode("\n", add_special_tokens=False)
    out = []
    for conv in CONVERSATIONS:
        messages, turns, kept = [], [], []
        for msg in conv:
            messages.append({"role": "user", "content": msg})
            alone = template([{"role": "user", "content": msg}])
            if turns:
                # the last answer's turn closed, then this message's turn and the assistant's opening: the template
                # of the message alone, past its default system message
                ids = kept + [2] + nl + alone[alone.index(2) + 1 + len(nl):]
            else:
                ids = alone
            same = ids == template(messages)
            with torch.no_grad():
                g = model.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=N_NEW, output_logits=True,
                                   return_dict_in_generate=True, eos_token_id=[0, 2], pad_token_id=0)
            new = g.sequences[0, len(ids):].tolist()
            logits = [float(lg[0, i]) for lg, i in zip(g.logits, new)]
            answer = tok.decode(new, skip_special_tokens=True)
            turns.append({"message": msg, "ids": ids, "gen": new, "logits": logits, "answer": answer, "template": same})
            messages.append({"role": "assistant", "content": answer})
            kept = ids + [i for i in new if i not in (0, 2)]
            print(f"{msg!r}: {len(ids)} ids{'' if same else ' (not the template of the texts)'}, {len(new)} new, {answer!r}")
        out.append({"n": N_NEW, "turns": turns})
    with open(os.path.join(OUT, "chat_turns.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
