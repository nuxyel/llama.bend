"""Conversations of several turns in Bend (demo/gen.bend --chat -, which keeps the caches between the turns) against
transformers' generate from the whole conversation at every turn (reference/chat_turns_ref.py), for the model MODEL
(models.py).

  reference/.venv/bin/python reference/test_chat_turns.py <compiled gen>

For every conversation, on 1 and 16 threads, turn by turn:
  - the ids the program has run when the answer starts (the turns' ids and the answers before, as the caches hold
    them) must be the reference's ids of the conversation;
  - the answer's ids and text must be identical, the logit of each chosen id within TOL relative to it;
  - why the answer ended must be what its ids say (a stop token, or the count).
The two thread counts must print the same ids and logit bits. Then, in the same binary:
  - every turn again as one prompt, the whole conversation's ids run from nothing (GEN_IDS=1): the same answer ids
    and logit bits as with the caches kept;
  - the first conversation sampled (temperature 1.5, top-k 50, top-p 1, seed 42, which leaves the greedy answers),
    every answer against one prompt of the conversation so far with the same seed: the same ids and logit bits;
  - the first conversation as a person sees it (no GEN_TRACE): the answers' texts after `> `, written in pieces;
  - the first conversation with room for the first turn and three positions more (GEN_CTX): the answer is cut after
    four tokens, the same four, the program says the caches are full, and the next turn does not run;
  - and with room for eight positions, fewer than a first turn needs: nothing runs.
"""
import json, os, re, subprocess, sys, time

from models import ROOT, D, T_DIR

EXE = sys.argv[1]
TOL = 2e-4
SAMPLING = ["1.5", "50", "1.0", "42"]        # temperature, top-k, top-p, seed: hot enough to leave the greedy path
failures = 0


def run(args, stdin=None, env=None, threads=None):
    rts = ["+RTS", "--threads", str(threads), "-RTS"] if threads else []
    r = subprocess.run([EXE] + rts + [D] + args, input=stdin, capture_output=True, cwd=ROOT,
                       env=dict(os.environ, **(env or {})))
    return r.returncode, r.stdout.decode("utf-8"), r.stderr.decode("utf-8", "replace")


def turns_of(out):
    """The turns a traced conversation printed: the ids each one added, its tokens (id, logit bits, logit), its text
    and why it ended."""
    turns = []
    for block in out.split("\nturn ids: ")[1:]:
        m = re.match(r"([^\n]*)\n((?:  (?:id|cands) [^\n]*\n)*)text: (.*)\nover: ([^\n]*?) room (\d+)\n?\Z", block, re.S)
        if not m:
            return None
        toks = re.findall(r"^  id (\d+)  logit (\d+) (\S+)", m.group(2), re.M)
        turns.append({"ids": [int(x) for x in m.group(1).split()], "gen": [int(i) for i, _, _ in toks],
                      "bits": [int(b) for _, b, _ in toks], "logits": [float(v) for _, _, v in toks],
                      "text": m.group(3), "why": m.group(4), "room": int(m.group(5))})
    return turns


def check(ok, what, detail=""):
    global failures
    failures += not ok
    print(f"{'ok  ' if ok else 'ERROR'} {what}" + (f"\n  {detail}" if detail and not ok else ""))
    return ok


def conversation(k, conv, threads):
    """One conversation on a thread count; returns its turns, or None when the run is not a conversation."""
    n = conv["n"]
    msgs = [t["message"] for t in conv["turns"]]
    t0 = time.time()
    code, out, err = run(["--chat", "-", str(n)], stdin=("\n".join(msgs) + "\n").encode("utf-8"),
                         env={"GEN_TRACE": "1"}, threads=threads)
    dt = time.time() - t0
    turns = turns_of(out) if code == 0 else None
    if not check(turns is not None and len(turns) == len(msgs), f"conversation {k + 1}, {threads} threads: "
                 f"{len(msgs)} turns ({dt:.1f} s)", (out + err)[-2000:]):
        return None
    ran, pend = [], []
    for j, (t, ref) in enumerate(zip(turns, conv["turns"])):
        ran += pend + t["ids"]
        ids_ok = ran == ref["ids"]
        gen_ok = t["gen"] == ref["gen"]
        lg_err = max((abs(a - b) / abs(b) for a, b in zip(t["logits"], ref["logits"])), default=float("inf"))
        lg_ok = len(t["logits"]) == len(ref["logits"]) and lg_err <= TOL
        txt_ok = t["text"] == ref["answer"]
        stop = bool(ref["gen"]) and ref["gen"][-1] in (0, 2)
        why_ok = t["why"] == ("stop" if stop else "count")
        check(ids_ok and gen_ok and lg_ok and txt_ok and why_ok,
              f"  turn {j + 1} {ref['message']!r}: {len(ref['ids'])} ids in the conversation, {len(t['gen'])} new: "
              f"conversation ids {'same' if ids_ok else 'DIFFER'}, answer ids {'same' if gen_ok else 'DIFFER'}, text "
              f"{'same' if txt_ok else 'DIFFERS'}, chosen logits within {lg_err:.1e}, over: {t['why']}"
              f"{'' if ref['template'] else ' (ids kept from a text that does not encode back to them)'}",
              f"bend:  {ran} {t['gen']} {t['text']!r}\n  torch: {ref['ids']} {ref['gen']} {ref['answer']!r}")
        ran += t["gen"][:-1]
        pend = [] if not t["gen"] or t["gen"][-1] in (0, 2) else t["gen"][-1:]
    return turns


def main():
    convs = json.load(open(os.path.join(T_DIR, "chat_turns.json")))
    for k, conv in enumerate(convs):
        one = conversation(k, conv, 1)
        many = conversation(k, conv, 16)
        if one is None or many is None:
            continue
        same = [(a["gen"], a["bits"]) for a in one] == [(a["gen"], a["bits"]) for a in many]
        check(same, f"conversation {k + 1}: the same ids and logit bits on 1 and 16 threads")
        # every turn as one prompt: the whole conversation's ids from nothing
        for j, (t, ref) in enumerate(zip(one, conv["turns"])):
            code, out, err = run(["--chat", " ".join(str(i) for i in ref["ids"]), str(conv["n"])], env={"GEN_IDS": "1"})
            toks = re.findall(r"^  id (\d+)  logit (\d+) ", out, re.M)
            got = ([int(i) for i, _ in toks], [int(b) for _, b in toks])
            check(code == 0 and got == (t["gen"], t["bits"]),
                  f"  turn {j + 1} as one prompt of {len(ref['ids'])} ids: the same answer ids and logit bits",
                  f"with the caches: {t['gen']} {t['bits']}\n  as one prompt:   {got[0]} {got[1]}\n{err[-500:]}")

    conv = convs[0]
    n, refs = conv["n"], conv["turns"]
    stdin = ("\n".join(t["message"] for t in refs) + "\n").encode("utf-8")

    # sampled: every answer of the conversation against one prompt of the conversation so far, with the same seed
    code, out, err = run(["--chat", "-", str(n)] + SAMPLING, stdin=stdin, env={"GEN_TRACE": "1"})
    turns = turns_of(out) if code == 0 else None
    if check(turns is not None and len(turns) == len(refs), "conversation 1, sampled " + " ".join(SAMPLING),
             (out + err)[-1500:]):
        ran, pend = [], []
        for j, t in enumerate(turns):
            ran += pend + t["ids"]
            code, out, err = run(["--chat", " ".join(str(i) for i in ran), str(n)] + SAMPLING, env={"GEN_IDS": "1"})
            toks = re.findall(r"^  id (\d+)  logit (\d+) ", out, re.M)
            got = ([int(i) for i, _ in toks], [int(b) for _, b in toks])
            check(code == 0 and got == (t["gen"], t["bits"]),
                  f"  turn {j + 1}, sampled: {len(t['gen'])} tokens, the same ids and logit bits as one prompt of "
                  f"{len(ran)} ids", f"with the caches: {t['gen']} {t['bits']}\n  as one prompt:   {got[0]} {got[1]}")
            ran += t["gen"][:-1]
            pend = [] if not t["gen"] or t["gen"][-1] in (0, 2) else t["gen"][-1:]

    # as a person sees it
    code, out, err = run(["--chat", "-", str(n)], stdin=stdin)
    want = "".join("> " + t["answer"] + "\n" + ("" if t["gen"][-1] in (0, 2) else f"(stopped at {n} tokens)\n")
                   for t in refs) + "> "
    body = out.split("\n", 1)[1] if "\n" in out else out          # past `weights loaded in ...`
    check(code == 0 and body == want, "conversation 1 without GEN_TRACE: the answers' texts after `> `",
          f"printed: {body!r}\n  wanted:  {want!r}\n{err[-500:]}")

    # the room: the first turn and three positions more
    first = refs[0]
    if len(first["gen"]) >= 5:
        ctx = len(first["ids"]) + 3
        code, out, err = run(["--chat", "-", str(n)], stdin=stdin, env={"GEN_TRACE": "1", "GEN_CTX": str(ctx)})
        turns = turns_of(out) if code == 0 else None
        ok = (turns is not None and len(turns) == len(refs) and turns[0]["gen"] == first["gen"][:4]
              and turns[0]["why"] == "full" and turns[0]["room"] == 0
              and all(t["gen"] == [] and t["why"] == f"unfit {len(t['ids']) + 1}" and t["room"] == 0 for t in turns[1:]))
        check(ok, f"conversation 1 with room for {ctx} positions: four tokens, the same four, then full; the next "
              f"turns do not run", (out + err)[-1500:])
        code, out, err = run(["--chat", "-", str(n)], stdin=stdin, env={"GEN_CTX": str(ctx)})
        body = out.split("\n", 1)[1] if "\n" in out else out
        check(code == 0 and body.endswith(f"\n(the conversation ends here: its {ctx} positions are full)\n")
              and body.count("> ") == 1, f"conversation 1 with room for {ctx} positions, without GEN_TRACE: it says "
              f"the caches are full and ends", repr(body) + err[-500:])
    else:
        print(f"skip the full-room case: the first answer has {len(first['gen'])} tokens, fewer than five")
    ctx = 8
    code, out, err = run(["--chat", "-", str(n)], stdin=stdin, env={"GEN_TRACE": "1", "GEN_CTX": str(ctx)})
    turns = turns_of(out) if code == 0 else None
    ok = (turns is not None and len(turns) == len(refs) and turns[0]["ids"] == first["ids"]
          and all(t["gen"] == [] and t["why"] == f"unfit {len(t['ids'])}" and t["room"] == ctx for t in turns))
    check(ok, f"conversation 1 with room for {ctx} positions, fewer than any turn that opens a conversation: nothing "
          f"runs, and every turn is offered as the first", (out + err)[-1500:])

    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


main()
