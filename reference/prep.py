"""Prepares a model for Bend (milestone 1) and checks the result. The model is MODEL (models.py; default
smollm2-135m), downloaded at its pinned revision into data/<MODEL>/.

  reference/.venv/bin/python reference/prep.py          download (pinned revision) and write data/<MODEL>/
  reference/.venv/bin/python reference/prep.py --check  verify what is in data/<MODEL>/

Writes to data/<MODEL>/:
  hf/            config.json, tokenizer.json, model.safetensors as downloaded
  w/<name>.bin   one little-endian float32 file per tensor, row = output (the PyTorch Linear layout), the layout
                 bend-ml's loader reads; names in WEIGHTS below
  merges_num.txt one BPE rule per line, "a b" in the bend-ml scheme (byte b -> b, the token of rule k -> 256 + k)
  ids.txt        the model id of every symbol of that scheme: 256 bytes, then one per rule; -1 for the 21 bytes
                 that have no token (0xC0, 0xC1, 0xF5-0xFF and six control bytes): `tokenizers` drops them
                 from a piece before the merges, and so does encode() below
  cfg.txt        the config the Bend package reads (bend-ml-llama's Cfg): kv g half f layers vocab, then the F32
                 bits of rms_norm_eps and rope_theta, in decimal; cfg_line() checks the structure it assumes
  perm.txt       what the Bend tokenizer reads: the model id of byte b, 0 for the 21 bytes without a token (0 is
                 <|endoftext|>, never a byte's id; the Bend number parser does not read "-1"); merge k is 252 + k
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref

from models import REPO, REV, D, HF, W, config

# the GPT-2 regex of the ByteLevel pre-tokenizer (use_regex = true)
GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

SAMPLES = [
    "Hello, world! The quick brown fox jumps over the lazy dog.",
    "Machine learning in Bend: types that catch shape errors at compile time.",
    "ação, coração e informação não cabem num ASCII; café déjà vu.",
    "  multiple   spaces\nand\nnewlines\t tabs it's we're they'll I'd",
    "Prices: $3.50, €1,234.99 and 1e-5; dates 2026-10-09, 09/10/2026; pi = 3.14159265358979",
    "A long number 12345678901234567890 and a phone +55 (11) 98765-4321.",
    "Non-ASCII digits: Arabic ٣٤٥, Devanagari १२३, fractions ½ ¾, Roman Ⅻ, superscript x² and ①②.",
    "Emoji 😀🎉 and CJK 漢字 かな 한국어, with digits ４５ (full width).",
    "Control bytes \x04 and \x1d are dropped: a\x04b, !!\x04!!, ...\x1d..., x\x04\x06y.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n",
]


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def weights(c):
    """(short name, safetensors name, shape) for every tensor; the LM head is tied to emb."""
    h, kv = c["hidden_size"], c["num_key_value_heads"] * (c["hidden_size"] // c["num_attention_heads"])
    f = c["intermediate_size"]
    out = [("emb", "model.embed_tokens.weight", (c["vocab_size"], h))]
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        out += [
            (f"h{i}.ln1", p + "input_layernorm.weight", (h,)),
            (f"h{i}.q", p + "self_attn.q_proj.weight", (h, h)),
            (f"h{i}.k", p + "self_attn.k_proj.weight", (kv, h)),
            (f"h{i}.v", p + "self_attn.v_proj.weight", (kv, h)),
            (f"h{i}.o", p + "self_attn.o_proj.weight", (h, h)),
            (f"h{i}.ln2", p + "post_attention_layernorm.weight", (h,)),
            (f"h{i}.gate", p + "mlp.gate_proj.weight", (f, h)),
            (f"h{i}.up", p + "mlp.up_proj.weight", (f, h)),
            (f"h{i}.down", p + "mlp.down_proj.weight", (h, f)),
        ]
    out.append(("lnf", "model.norm.weight", (h,)))
    return out


def cfg_line(c):
    """The Cfg of bend-ml-llama for a config.json, after checking that the model is the Llama the package runs."""
    heads, kv, hidden = c["num_attention_heads"], c["num_key_value_heads"], c["hidden_size"]
    hd = c.get("head_dim") or hidden // heads
    assert c["architectures"] == ["LlamaForCausalLM"] and c["hidden_act"] == "silu", "not a Llama with SwiGLU"
    assert heads % kv == 0 and hd % 2 == 0 and heads * hd == hidden, "heads, key/value heads and head size"
    assert not c.get("attention_bias") and not c.get("mlp_bias"), "biases are not supported"
    assert c.get("rope_scaling") is None and not c.get("rope_interleaved", False), "only plain rotate-half RoPE"
    assert c["tie_word_embeddings"], "the LM head must be the embedding"
    bits = lambda x: int(np.array([x], dtype="<f4").view("<u4")[0])
    return " ".join(map(str, [kv, heads // kv, hd // 2, c["intermediate_size"], c["num_hidden_layers"],
                              c["vocab_size"], bits(c["rms_norm_eps"]), bits(c["rope_theta"])])) + "\n"


def download():
    from huggingface_hub import hf_hub_download
    for name in ["config.json", "tokenizer.json", "model.safetensors"]:
        hf_hub_download(REPO, name, revision=REV, local_dir=HF)
    print(f"downloaded {REPO}@{REV[:7]} into {os.path.relpath(HF)}/")


def write_weights():
    from safetensors import safe_open
    os.makedirs(W, exist_ok=True)
    total = 0
    with safe_open(os.path.join(HF, "model.safetensors"), "pt") as f:
        for short, name, shape in weights(config()):
            t = f.get_tensor(name)
            assert tuple(t.shape) == shape, (name, tuple(t.shape), shape)
            a = t.float().numpy().astype("<f4")
            a.tofile(os.path.join(W, short + ".bin"))
            total += a.size
    print(f"{os.path.relpath(W)}/: {total:,} numbers")


def tables():
    """merges_num.txt and ids.txt from tokenizer.json; returns (table, ids, report)."""
    tok = json.load(open(os.path.join(HF, "tokenizer.json")))
    vocab, merges = tok["model"]["vocab"], tok["model"]["merges"]
    merges = [m.split() if isinstance(m, str) else m for m in merges]
    b2u = bytes_to_unicode()
    sym = {b2u[b]: b for b in range(256)}  # token string -> bend-ml symbol
    ids = [vocab.get(b2u[b], -1) for b in range(256)]
    table, missing, repeated = [], 0, 0
    for k, (a, b) in enumerate(merges):
        table.append((sym[a], sym[b]))
        s = a + b
        if s in sym:
            repeated += 1  # a later rule rebuilds a token an earlier one made; the earlier one wins
        else:
            sym[s] = 256 + k
        if s in vocab:
            ids.append(vocab[s])
        else:
            ids.append(-1); missing += 1
    open(os.path.join(D, "merges_num.txt"), "w").write("".join(f"{a} {b}\n" for a, b in table))
    open(os.path.join(D, "ids.txt"), "w").write(" ".join(map(str, ids)) + "\n")
    open(os.path.join(D, "perm.txt"), "w").write(" ".join(str(max(i, 0)) for i in ids[:256]) + "\n")
    added = sorted(t["id"] for t in tok["added_tokens"])
    nobyte = [b for b in range(256) if ids[b] < 0]
    report = dict(rules=len(table), vocab=len(vocab), missing=missing, repeated=repeated, added=added,
                  bytes_without_id=nobyte,
                  offset=sum(1 for k in range(len(table)) if ids[256 + k] == 17 + 256 - len(nobyte) + k))
    return table, ids, report


def pretokenize(s):
    """Digits(individual_digits) then the GPT-2 regex: every \\p{N} character is its own piece first."""
    import regex
    pat = regex.compile(GPT2_PAT)
    pieces = []
    for seg in regex.split(r"(\p{N})", s):
        if seg:
            pieces += regex.findall(pat, seg)
    return pieces


def encode(table, ids, s):
    out = []
    for w in pretokenize(s):
        bs = bytes(b for b in w.encode("utf-8") if ids[b] >= 0)
        out += [ids[x] for x in bpe_ref.encode(table, bs)]
    return out


def check():
    from safetensors import safe_open
    from tokenizers import Tokenizer
    import torch
    ok = True
    c = config()
    # 1. weights: bit for bit, shapes, total
    total = 0
    with safe_open(os.path.join(HF, "model.safetensors"), "pt") as f:
        for short, name, shape in weights(c):
            t = f.get_tensor(name)
            want = (t.view(torch.int16).numpy().astype(np.int32) & 0xFFFF) << 16  # bf16 bits << 16
            got = np.fromfile(os.path.join(W, short + ".bin"), dtype="<u4")
            same = got.size == int(np.prod(shape)) and np.array_equal(got, want.astype(np.uint32).ravel())
            ok &= same
            if not same:
                print("MISMATCH", short)
            total += got.size
    n_tensors = len(weights(c))
    print(f"weights: {n_tensors} files bit for bit = bf16 << 16: {'ok' if ok else 'FAILED'}; {total:,} numbers")
    # 2. the config the Bend package reads
    want_cfg = cfg_line(c)
    path = os.path.join(D, "cfg.txt")
    cfg_ok = os.path.exists(path) and open(path).read() == want_cfg
    print(f"cfg.txt: {want_cfg.strip()}: {'ok' if cfg_ok else 'FAILED (run prep again)'}")
    ok &= cfg_ok
    # 3. tokenizer: rules in order (what Bend will run) against tokenizers
    table, ids, rep = tables()
    print(f"tokenizer: {rep['rules']} rules, {rep['vocab']} ids, special ids {rep['added'][0]}..{rep['added'][-1]}; "
          f"rules whose token is missing from the vocabulary: {rep['missing']}; rules that rebuild an earlier "
          f"token: {rep['repeated']}; bytes without an id: {len(rep['bytes_without_id'])}; rules k with id 252 + k: "
          f"{rep['offset']}")
    perm = list(map(int, open(os.path.join(D, "perm.txt")).read().split()))
    fixed = len(perm) == 256 and perm.count(0) == 21 and rep["offset"] == rep["rules"]
    print(f"perm.txt: 256 ids, {perm.count(0)} zeros; merge k = 252 + k for all {rep['rules']} rules: "
          f"{'ok' if fixed else 'FAILED'}")
    ok &= fixed
    hf = Tokenizer.from_file(os.path.join(HF, "tokenizer.json"))
    tok_ok = True
    for s in SAMPLES:
        pieces_hf = [p for p, _ in hf.pre_tokenizer.pre_tokenize_str(s)]
        b2u = bytes_to_unicode()
        pieces_me = ["".join(b2u[b] for b in p.encode("utf-8")) for p in pretokenize(s)]
        mine, ref = encode(table, ids, s), hf.encode(s, add_special_tokens=False).ids
        same = mine == ref and pieces_me == pieces_hf
        tok_ok &= same
        print(f"  {'ok  ' if same else 'FAIL'} pieces {'=' if pieces_me == pieces_hf else '!='} ids "
              f"{'=' if mine == ref else '!='} | {s[:50]!r}")
        if mine != ref:
            print("     mine:", mine[:20], "\n     ref: ", ref[:20])
    print("TOKENIZER EQUIVALENCE:", "ok" if tok_ok else "FAILED")
    ok &= tok_ok
    print("PREP CHECK:", "ok" if ok else "FAILED")
    return ok


def main():
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    os.makedirs(D, exist_ok=True)
    download()
    write_weights()
    open(os.path.join(D, "cfg.txt"), "w").write(cfg_line(config()))
    print("cfg.txt:", cfg_line(config()).strip())
    _, _, rep = tables()
    print("tables:", rep)


if __name__ == "__main__":
    main()
