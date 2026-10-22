# Plan: SmolLM2-135M in Bend

Written 2026-10-09 from my planning notes of 2026-10-05, updated with what bend-ml
v1.2.1 now offers.

## Goal and success bar

- **Token for token against PyTorch** (`transformers`, greedy), logits within a small tolerance, as bend-ml's
  GPT-2 demo does.
- Typed Llama layers that other people can build models on: the BendHub package `bend-ml-llama`.
- The reference Llama implementation in Bend: generic over the Llama family (a second model with other sizes
  runs with only the config changed), its structure proved, its numbers tested.
- A post replying to the "llama.bend" line of the wishlist once it works.

## What I reuse from bend-ml

- `bend-ml-tensor-array@0.1.6.0`: weight matrices as `Bands<r, c>` (row bands, parallel matrix · vector without
  copying weights) with `Bands.matvec_l` (lists in and out), `Bands.read_row` for the embedding lookup, and the
  tied LM head as one more `matvec_l` over the embedding. With ≤ 2 threads one band (bend-ml `depth_for`).
- `bend-ml-bpe-tokenizer@0.1.2.0`: encode/decode with the proved roundtrip.
- bend-ml `demos/gpt2/`: `tok.bend` (pre-tokenizer, UTF-8), `unicode.bend` (character classes; class 1 is
  `\p{N}`), the loader (`get_bands`, `stream_exact`, `fill_bytes`, `check_size`), the KV-cache loop, argmax.
- bend-ml `reference/`: `gpt2_prep.py`, `bpe_ref.py`, `test_gpt2.py`, `test_gpt2_tok.py` as patterns.

## What is new

1. **RMSNorm**: `x / sqrt(mean(x²) + eps) * g`, no bias.
2. **RoPE** (`rope_theta` 100000, rotate-half layout, pairs (i, i + 32) in a head of 64). Risk: `F32.sin/cos/pow`
   against PyTorch's; fallback: read the cos/sin table from a file written by the reference.
3. **SwiGLU MLP**: `down(silu(gate(x)) * up(x))`, `silu(x) = x * sigmoid(x)`.
4. **Grouped-query attention**: 9 query heads read 3 KV heads (query head h uses KV head h / 3); KV width 192.
5. **bf16 weights** (exact in F32: the bits shifted left by 16) and 30 layers.
6. **Tokenizer**: the GPT-2 byte-level regex plus `Digits(individual_digits)` first; 17 special tokens (ids
   0-16) before the byte tokens; oracle `tokenizers` instead of `tiktoken`.

## Laws (the structure; numbers are tested, never proved)

- **GQA head mapping**: with `n_heads = n_kv × g` in the type, every query head maps to an existing KV head:
  `h < n_kv × g → h / g < n_kv` (a `div` bound lemma).
- **Even RoPE head size by construction**: the head size is `2 × half` in the type; splitting a `2 × half`
  vector into halves and joining them gives it back.
- **Causality by type**: a KV cache indexed by the number of positions, `KV<t>`; step t appends one row and the
  attention scores have type `Vec<t + 1>`, so a step that reads a future position does not type-check.
- **Weight layout covers the file**: the sum of the tensor sizes of the config equals the number of values;
  proved by computation for the concrete config and checked at load time against `File.size`.

## Milestones (each with its pass criterion)

1. **Prep** (`reference/prep.py`, **done 2026-10-09**): download at a pinned revision, bf16 → F32 one file per tensor, tokenizer
   tables. Pass: every weight file equals `tensor.float()` bit for bit; sizes match the config; the merges
   applied in order give the same ids as `tokenizers` on a test set.
2. **Tokenizer in Bend**: digits rule + GPT-2 regex + SmolLM2 merges. Pass: ids equal to `tokenizers` on
   bend-ml's GPT-2 tokenizer texts plus texts with numbers (prices, dates, long numbers, non-ASCII digits). **Done 2026-10-09** (v0.0.1).
3. **One layer**: RMSNorm, RoPE, GQA, SwiGLU on a fixed input. Pass: layer output within 1e-5 of PyTorch. **Done 2026-10-09** (v0.0.2): every intermediate within 2.2e-6.

Milestones 4-11 were rewritten on 2026-10-10 (NOTES): the goal is now the reference Llama implementation in
Bend, generic over the Llama family, and a base that later helps build a training framework. Phase A makes it
work on the current code (fixed sizes); phase B turns it into the typed, generic API and proves the laws on that
API; phase C publishes it.

**Phase A: it works (current code)**

4. **Full forward**: the 30 layers, the final RMSNorm and the tied LM head (`Bands.matvec_l` over the
   embedding); the oracle also writes the hidden state after every layer, to trace drift. Pass: logits within
   ~2e-4 and the same argmax on 10+ prompts. **Done 2026-10-10**: within 2.5e-5, the same argmax at all 88
   positions of 11 prompts.
5. **Generation**: greedy, 32 tokens, key/value cache per layer, tokenizer in and out. Pass: the same ids as
   `transformers` with `do_sample = False`. First timings in NOTES (not a benchmark). **Done 2026-10-10**: the
   same ids, text and chosen logits (within 3.2e-5) on 12 prompts, one of them ending at the end-of-text token.

**Phase B: the reference API**

6. **Generic typed API**, in one file (the future package): a `Config` whose structure is in the types
   (`n_heads = n_kv × g`, `head = 2 × half`, `hidden = n_heads × head`), typed heads, `KV<t>`, one record per
   layer instead of the `b1` … `b13` chain. Each block returns its output and an activations record (today's
   `Tr`), which serves the tests now and a backward pass later. Pass: logits and ids bit for bit equal to
   phase A (a pure refactor). **Done 2026-10-10**: `llama/main.bend`; all 23 outputs of `reference/bits.py` identical.
7. **Laws** on that API, `--verdict`: the four above, plus "the SmolLM2 configs satisfy `Config`", each in the
   README in plain language. **Done 2026-10-10**: GQA mapping through `Nat.div`, groups and columns, `Cache(t)`,
   head split, layout (in `U32`) and config sizes; `--verdict` ok.
8. **Generality**: `prep.py --model`, and SmolLM2-360M with only the config changed. Pass: the same 32 greedy
   tokens as `transformers`. Stretch: SmolLM2-1.7B if the memory allows (measure first). **Done 2026-10-10**, the
   stretch too: `MODEL=` picks the model (`reference/models.py`); SmolLM2-360M and SmolLM2-1.7B (32 key/value heads
   of one query head, RoPE theta 130000) give the same tokens on the 12 prompts with no code change. Found on the
   way: RMSNorm now sums its squares pairwise (NOTES).

8b. **Key/value cache in flat Arrays** (inserted 2026-10-11 after the first benchmark; "milestone 9" in NOTES): one
   `Array<F32>` of keys and one of values per key/value head, attention through tensor-array's `gemm`, the heads in
   parallel. Pass: a per-token cost that does not grow with the position over 128 and 512 tokens on the three models,
   the same tokens as transformers. **Done 2026-10-12**: slope 0.00-0.18 ms per position over 512 tokens (15-20 before),
   the 1.7B 4.0x faster at 16 threads. The capacity is proved as `kv_room` (2026-10-12): a position below `cap` is
   written inside the arrays. Since 2026-10-13 the type carries it too (`KV<cap, t>`): `KV.push` takes a proof that
   `t < cap`, so a write past the capacity does not compile (`tests/bad_cap.bend`).

**Phase C: publish**

9. **Package and measurements**: `bend-ml-llama@0.1.0.0` (one file, `LICENSE`, the laws in the README);
   `BENCHMARK.md` per rule 7 (idle, `taskset -c 3`, 1 and 16 threads, load time and resident peak against
   PyTorch); CI modeled on bend-ml's. **Done 2026-10-13**: `bend-ml-llama@0.1.0.0` on BendHub (0.1.1.0 since),
   `BENCHMARK.md` against transformers, llama.cpp and candle, CI (NOTES).
10. **Public**: the repository goes public, bend-ml's README links it, a post answering the wishlist line. **Done
    2026-10-13**: v1.0.0, the repository public, the launch posts (NOTES).
11. **Bridge to a training framework**: `docs/bendtorch.md`, for each op what its backward pass needs from the
    activations record and what the affine rules made hard. A document, no code. **Done 2026-10-12**: `docs/bendtorch.md`, every operation's
    backward pass checked against `torch.autograd` (RMSNorm, SwiGLU, softmax).

Optional, after 8: read `model.safetensors` directly in Bend (bf16 → F32 is a shift by 16), so running needs no
Python; decided after measuring, since `File.read_at` is the cost of loading (bend-ml exp. 13).

## Risks and open questions

- ~~The token id scheme~~ settled in milestone 1: 21 bytes have no token, the bytes are ids 17-251 and merge k
  is id 252 + k; the Bend tokenizer must drop the 21 bytes before the merges, as `tokenizers` does (NOTES).
- ~~RoPE float drift over 30 layers~~: `inv_freq` is bit for bit equal to PyTorch's (milestone 3), and the error
  does not grow over the 30 layers (milestone 4, NOTES).
- 30 layers: compile time (a data point for the launch question about deeper nets).
- The chat template (`<|im_start|>` …) is out of scope; the base model is enough.
- Larger Llama models later (SmolLM2-360M): only the config should change if the package stays generic.
