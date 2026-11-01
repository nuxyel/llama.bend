"""SmolLM2-Instruct's chat template in Bend (demo/tok.bend's chat_ids) against transformers' apply_chat_template.

  reference/.venv/bin/python reference/test_chat.py <compiled tok>          the conversations below
  reference/.venv/bin/python reference/test_chat.py --hub                   models.CHAT_TEMPLATE against the Hub

For every conversation, with and without the generation prompt, the ids Bend writes must equal those of
apply_chat_template(tokenize = True) with models.CHAT_TEMPLATE over the tokenizer of MODEL (the instruct checkpoints'
tokenizer.json is the same file). --hub downloads tokenizer_config.json (4 KB) of the three instruct checkpoints at
their pinned revisions and checks that the template is the one in models.py (needs the network; not in make check).
"""
import json, os, subprocess, sys, tempfile

from models import CHAT_TEMPLATE, D, HF, INSTRUCT, ROOT

SYS = {"role": "system", "content": "You are a pirate. Answer in one line."}
CASES = [
    [{"role": "user", "content": "Hi there!"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [SYS, {"role": "user", "content": "Where is the treasure?"}],
    [{"role": "user", "content": "Hi there!"}, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "2+2?"}],
    [SYS, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"},
     {"role": "assistant", "content": "d"}],
    [{"role": "user", "content": "line one\nline two\n\n  indented\ttab"}],
    [{"role": "user", "content": ""}],
    [{"role": "user", "content": "ação, coração, café; Привет 你好 😀 123456 $19.99"}],
    [{"role": "user", "content": "def add(a, b):\n    return a + b\n"}],
    [{"role": "assistant", "content": "I start."}, {"role": "user", "content": " leading and trailing spaces "}],
    [{"role": "system", "content": ""}, {"role": "user", "content": "empty system"}],
    [{"role": "user", "content": "first"}, {"role": "system", "content": "a system message later"}],
    [{"role": "tool", "content": "{\"temp\": 21}"}],
]


def hub():
    from huggingface_hub import hf_hub_download
    bad = 0
    for model, (repo, rev) in INSTRUCT.items():
        with tempfile.TemporaryDirectory() as d:
            p = hf_hub_download(repo, "tokenizer_config.json", revision=rev, local_dir=d)
            same = json.load(open(p))["chat_template"] == CHAT_TEMPLATE
        bad += not same
        print(f"{'ok  ' if same else 'ERROR'} {repo}@{rev[:8]}: chat template {'same' if same else 'DIFFERS'}")
    return bad


def main():
    if sys.argv[1:] == ["--hub"]:
        bad = hub()
        print("FAILURES:", bad)
        sys.exit(1 if bad else 0)
    exe = sys.argv[1]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "messages")
        for msgs in CASES:
            with open(path, "wb") as f:
                for m in msgs:
                    f.write(m["role"].encode() + b"\0" + m["content"].encode() + b"\0")
            for gen in (True, False):
                want = list(tok.apply_chat_template(msgs, chat_template=CHAT_TEMPLATE, add_generation_prompt=gen,
                                                    tokenize=True, return_dict=False))
                r = subprocess.run([exe, os.path.join(D, "merges_num.txt"), os.path.join(D, "perm.txt"),
                                    "--chat" if gen else "--chat-nogen", path], capture_output=True, text=True, cwd=ROOT)
                got = [int(x) for x in r.stdout.split()] if r.returncode == 0 and "invalid" not in r.stdout else None
                ok = got == want
                failures += not ok
                desc = " / ".join(f"{m['role']}: {m['content'][:24]!r}" for m in msgs) or "(no messages)"
                print(f"{'ok  ' if ok else 'ERROR'} {'gen  ' if gen else 'nogen'} {len(want):3} ids  {desc}")
                if not ok:
                    print(f"  bend:         {got}\n  transformers: {want}\n{r.stderr[-1000:]}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
