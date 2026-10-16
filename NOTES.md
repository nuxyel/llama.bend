# Notes

What I find, measure and decide, by date. The plan is in `docs/plan.md`.

## Pinned versions

- Bend **2.0.35**, Lean 4.34.0 (installed by bend-ml's `scripts/setup.sh`), as in bend-ml v1.2.1.
- bend-ml packages from BendHub: `bend-ml-tensor-array@0.1.6.0`, `bend-ml-bpe-tokenizer@0.1.2.0`.
- Model: `HuggingFaceTB/SmolLM2-135M` at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Python side: versions in `reference/requirements.txt`.

## 2026-10-09: start, and milestone 1 (prep)

### Decisions

- Its own repository (`nuxyel/llama.bend`), private until it works; the typed layers become the BendHub package
  `bend-ml-llama`. My 2026-10-05 notes put the demo inside bend-ml; a separate repo keeps bend-ml's checks
  fast and lets this one move on its own.
- Products with `Bands` + `Bands.matvec_l` (`bend-ml-tensor-array@0.1.6.0`), not `Mat.matmul_nt` as those
  notes said: since 0.1.6.0 one band costs about what `matmul_nt` costs, and more bands run in parallel.
- Oracle: `transformers` and `tokenizers` on the CPU.

### The model, checked against `config.json` at the pinned revision

| Field | Value |
|---|---|
| architecture | `LlamaForCausalLM` |
| hidden_size | 576 |
| heads / key-value heads | 9 / 3: head size 64, 3 query heads per key-value head, key/value width 192 |
| layers | 30 |
| intermediate_size | 1536 (SwiGLU, `silu`) |
| rms_norm_eps | 1e-5; no bias anywhere |
| rope_theta | 100000, `rope_interleaved = false` (rotate-half) |
| vocab_size | 49152, tied embedding (LM head = embedding) |
| dtype | bfloat16, `model.safetensors` 269,060,552 bytes |

272 tensors, **134,515,008 numbers** (embedding 28,311,552; 3,540,096 per layer; final norm 576).

### Tokenizer facts

- Pre-tokenizer: `Sequence[Digits(individual_digits = true), ByteLevel(use_regex = true)]`. `Digits` uses
  Rust's `char::is_numeric` (`huggingface/tokenizers`, `tk-encode/src/pre_tokenizers/digits.rs`), that is
  Unicode `\p{N}`, which is already class 1 of bend-ml's `demos/gpt2/unicode.bend` table. No new class needed.
- 17 special tokens (ids 0-16), then the byte tokens, then one id per merge.
- **21 bytes have no token**: 0xC0, 0xC1, 0xF5-0xFF (never in valid UTF-8) and the control bytes 4, 6, 19, 20,
  22, 29. So the byte tokens are ids 17-251 (235 of them), and 17 + 235 + 48,900 merges = 49,152. Every merge k
  gives id **252 + k** (all 48,900), no merge rebuilds an earlier token and none is missing from the vocabulary.
- With no `unk_token` and no `byte_fallback`, `tokenizers` drops a byte that has no token, before the merges:
  "a\x04b" gives [a, b]. `reference/prep.py` does the same; the Bend tokenizer will have to as well.
- A special token written as text (`<|endoftext|>`) is matched by `tokenizers` before pre-tokenization; out of
  scope for the demo (prompts are plain text).

### Milestone 1 result

`make prep && make check` (`reference/prep.py`):

- `data/w/<name>.bin`, one little-endian F32 file per tensor, row = output: all 272 equal to the bf16 bits
  shifted left by 16, bit for bit.
- `data/merges_num.txt` (48,900 rules in bend-ml's numbering) and `data/ids.txt` (the model id of each symbol,
  -1 for the 21 bytes): applying the rules in order to the pieces of my pre-tokenizer (each `\p{N}` character
  alone, then the GPT-2 regex) gives the same pieces and the same ids as `tokenizers` on 10 texts: ASCII,
  Portuguese, prices, dates and long numbers, Arabic and Devanagari digits, fractions, Roman numerals,
  superscripts, circled and full-width digits, emoji, CJK, Korean, control bytes, Python code.

## 2026-10-09: milestone 2 (the tokenizer in Bend)

`demo/tok.bend` started as a copy of bend-ml's GPT-2 tokenizer (`demos/gpt2/tok.bend`) and changed in four
steps, each measured with `reference/test_tok.py` (66 fixed texts, 2 more added with the last change, 3
malformed merge tables, 30 random texts):

| Change | Failures |
|---|---|
| bend-ml's GPT-2 tokenizer as is | 89 of 99 |
| merge k is id 252 + k (GPT-2: 256 + k) | 7 |
| every `\p{N}` character is its own piece | 3 |
| the 21 bytes without a token are dropped before the merges | 1 |
| 0x1C-0x1F are not whitespace | **0** |

- **Digits.** SmolLM2 runs `Digits(individual_digits)` before the GPT-2 regex. In the tag scheme a digit is
  class 1, so a piece of class 1 is now exactly one character (its first byte and the continuation bytes,
  tag bit 2048). Two consequences of the split for the regex: a space is never attached to a digit (no
  " ?\p{N}+"), and a whitespace run that ends right before a digit is taken whole, because the regex sees it at
  the end of its segment (`"x   1"` gives "x", "   ", "1"; `"x   y"` gives "x", "  ", " y"). Many ASCII number
  cases passed even before this change: the vocabulary has no multi-digit tokens and no " 1"-style tokens, so
  the merges produced the same ids anyway; the difference showed with non-ASCII digits and spaces before digits.
- **Bytes without a token.** `perm.txt` gives 0 for them and `known_bytes` drops them from each piece before
  `BPE.encode`; the bytes around them can then merge (`"!!\x04!!"` gives the one token "!!!!").
- **0x1C-0x1F.** bend-ml's `class_ascii` counted them as whitespace (Python's `str.isspace` does); `tokenizers`
  does not (they are not Unicode White_Space), so `"...\x1d..."` is one piece there. Whitespace is now the space
  and 9-13 only. bend-ml's GPT-2 tokenizer is right to keep them as whitespace: I checked that tiktoken does
  (`"...\x1d..."`, `"a\x1cb"`, `"x \x1f y"` give the same ids in bend-ml and tiktoken). The two libraries
  differ here, so each tokenizer follows its own oracle.
- **Fuzz.** The random texts now also draw ASCII digits, Arabic-Indic and Devanagari digits, full-width digits,
  number forms, circled numbers and control bytes 0-31. Seeds 7, 11, 23, 47 with 150 random texts each: 0
  failures (221 cases per seed).
- Time per text (one Bend run, compiled, the 48,900 rules applied in order per piece): median 0.15 s, the longest
  0.58 s ("word " × 120), on this machine with a load average of ~1-2.

## 2026-10-09: milestone 3 (one layer in Bend)

`demo/layer.bend` is one SmolLM2 layer: RMSNorm, the q/k/v products, RoPE, grouped-query attention over a
growing key/value cache, the output product, the residual, the second RMSNorm, the SwiGLU MLP and the output.
The products use `Bands` + `matvec_l` from `bend-ml-tensor-array@0.1.6.0`; the weights are loaded with bend-ml's
block reader (`demo/load.bend`). `demo/layer_test.bend` runs it position by position and prints every
intermediate as exact F32 bits; `reference/test_layer.py` compares them with `transformers` 5.19 in F32 with
eager attention (`reference/layer_ref.py`: forward hooks on the layer, plus q and k after RoPE from
transformers' own `apply_rotary_pos_emb`).

I wrote it one step at a time and ran the test after each; every step passed the first time it compiled.
Worst `max|Δ| / max|y|` per position, over the three cases (layer 0 and layer 15 on the hidden states of
"The capital of France is the city of", 8 positions; layer 0 on 40 positions of Gaussian noise):

| Intermediate | Worst | Intermediate | Worst |
|---|---|---|---|
| n1 (RMSNorm) | 8.4e-7 | attn (before Wo) | 1.8e-6 |
| q / k / v | 1.3e-6 / 2.0e-6 / 1.7e-6 | a (Wo) | 1.1e-6 |
| q, k after RoPE | 1.3e-6 / 2.0e-6 | n2 | 1.3e-6 |
| | | m (MLP) | 2.2e-6 |
| | | **y (output)** | **2.2e-6** |

The pass criterion was 1e-5. Notes:

- **Order of operations.** I followed `modeling_llama.py`: RMSNorm multiplies x by `1 / sqrt(mean + eps)` first
  and by the gain after; RoPE is `h1*cos + (-h2)*sin` and `h2*cos + h1*sin`; the attention scale is 0.125 (exact);
  silu is `x / (1 + exp(-x))`, the form of ATen's CPU kernel. The remaining differences are the order of the
  sums inside the products (sequential in Bend, blocked in PyTorch's BLAS).
- **RoPE's `inv_freq`** computed in Bend with `1 / F32.pow(100000, 2i / 64)` is **bit for bit** equal to PyTorch's
  for all 32 values, so the fallback I had planned (reading torch's table from a file) is not needed. `F32.cos`
  and `F32.sin` at positions up to 39 add nothing visible (q and k after RoPE have the same worst error as before).
- **Layer 15 has outliers**: values up to ~19,000 in its input and output (the "massive activations" of Llama
  models), against ~1 in layer 0. Relative to each vector's largest value the error stays at 2e-7 there.
- **Threads.** The output bits are the same with `--threads 1` and `--threads 16` (bands do not change the order
  of any sum). Not a benchmark yet: one run of 40 positions takes ~0.37 s on both, weights load included.
- **Bend notes.** A def must come before the defs that call it, and there is no mutual recursion: the test loop
  carries the previous position's result (as bend-ml's GPT-2 `run` does) instead of two functions calling each
  other. The layer is a chain of small defs (`b1` … `b13`) because each product hands its bands back and `match`
  only reads parameters, the same pattern as bend-ml's `k1` … `k6b`. The plan for milestone 6 (typed heads, a
  cache indexed by position) should shorten it.

## 2026-10-10: the roadmap after milestone 3

### Decisions

- **The goal grows**: llama.bend should be the reference Llama implementation in Bend, generic over the Llama
  family, not a SmolLM2-135M demo. The test of that is a second model with other sizes (SmolLM2-360M) running
  with only the config changed (milestone 8).
- **First make it work, then the API.** Milestones 4 (full forward) and 5 (generation) go on the current code,
  sizes fixed and the `b1` … `b13` chain, to settle the numerical risk of 30 layers first. Then the generic typed
  API (milestone 6), checked bit for bit against milestone 5, and the laws proved on that API (milestone 7), so
  they are not proved twice.
- **A base for a training framework later, not designed now.** I have not decided what the larger framework
  after bend-ml will be. What llama.bend leaves for it: each block returns an activations record (today's `Tr`),
  and `docs/bendtorch.md` (milestone 11) lists what each op's backward pass needs from it.
- The new milestones and their pass criteria are in `docs/plan.md`.

## 2026-10-10: milestone 4 (the whole forward pass)

`demo/forward.bend` runs the embedding row, the 30 layers of `demo/layer.bend` (one key/value cache each), the
final RMSNorm and the tied LM head, position by position, and prints every layer's output, the final norm and
the 49,152 logits as exact F32 bits. `reference/forward_ref.py` takes the same values from `transformers` 5.19
(F32, eager attention; forward hooks on the 30 decoder layers and on the final norm), and
`reference/test_forward.py` compares them on the 11 prompts of bend-ml's GPT-2 test (3 to 13 positions each,
88 in all). The loader for one layer moved from the layer test into `demo/load.bend` (`load_lw`).

- **Logits: worst `max|Δ| / max|y|` per position 2.5e-5** (`"The capital of France is"` and the quick-brown-fox
  prompt), against the pass criterion of 2e-4; **the argmax is the same at all 88 positions.** It passed the
  first time it compiled; the only error the checker gave was the order of a printing helper's arguments
  (termination reads them left to right).
- The worst position is position 0 in every prompt. I have not looked into why.
- **The error does not grow with depth.** Worst relative error of each layer's output over a prompt's positions:
  ~1e-6 for layers 0-8, a peak of 1e-5 to 5e-5 around layers 9-11, back to ~1e-6 to 1e-5 by layer 20, and 1e-5 to
  2e-5 at layers 28-29. The final norm is within 1.5e-5. The 30 layers do not pile up error; whatever happens
  around layer 10 fades afterwards. So the drift risk listed in the plan did not show up.
- **Threads:** the output is the same bit for bit with `--threads 1` and `--threads 16` (11 positions, 8 MB of
  printed bits each). A first comparison said "identical" because both files were empty: I had called
  `/usr/bin/time`, which this machine does not have, so the program never ran. I check sizes before `cmp` now.
- Time per run, not a benchmark (load average ~2): 4.5-5.8 s for 3-13 positions, most of it loading 134.5 M
  numbers and turning 49,152 logits per position into text. One run of 3 positions: 7.7 s with 1 thread, 5.2 s
  with 16.

## 2026-10-10: milestone 5 (generation)

`demo/gen.bend` tokenizes the prompt in Bend, feeds it through the forward pass of `demo/forward.bend` one
position at a time (one key/value cache per layer), then takes the argmax of the logits as the next token and
feeds it back, up to n tokens or until the end-of-text token (id 0), after which `generate` stops too. It
prints each new id with its logit, and the decoded text of the prompt and the continuation.

- **Decoding** is new in `demo/tok.bend` (`decode_text`, and a `--decode` mode for tests). Ids 0-16 are special
  tokens and are left out, as `decode(skip_special_tokens = True)` does; 17-251 are bytes; 252 + k is merge k.
  Bytes that are not valid UTF-8 become U+FFFD the way Rust's `String::from_utf8_lossy` does it (the decoder of
  `tokenizers`): one U+FFFD for each maximal invalid prefix. My first version accepted any continuation byte
  after a first byte and failed 1 of 300 random id sequences per seed. Unicode's table 3-7 narrows the second byte
  after 0xE0, 0xED, 0xF0 and 0xF4 (no overlong forms, no surrogates, nothing above U+10FFFF), and 0xC0, 0xC1 and
  0xF5-0xFF never start a sequence. With these ranges `reference/test_detok.py` passes every case: the 68 test
  texts, and 300 random id sequences for each of seeds 7, 11, 23, 47, 99. The random sequences mix the whole
  vocabulary, the special tokens and the byte tokens, so most of them are invalid UTF-8.
- **Generation:** `reference/gen_ref.py` runs `generate(do_sample = False, max_new_tokens = 32)` in F32, eager
  attention, on the 11 prompts of milestone 4. A twelfth prompt, "Copyright 2024. All rights reserved.", produces
  the end-of-text token as its first new token. `reference/test_gen.py` passes all 12:
  - the prompt ids from Bend's tokenizer are the same;
  - the generated ids are the same: 32 per prompt, and the single end-of-text token for the twelfth;
  - the decoded text is identical;
  - each chosen logit is within 3.2e-5 relative ("Once upon a time"), most within 1e-5.
- Time, informal (load average 2-3, two alternated runs each, "Once upon a time" + 32 tokens, 36 forward
  passes): with 1 thread pinned to CPU 3, loading takes 6.3 s and the passes 5.1 s; with 16 threads, 4.0-4.3 s and
  4.4-4.5 s. So 16 threads gain little over one here (GPT-2 in bend-ml: 5.3 s against 3.7 s for its 36 passes).
  Each layer has 7 products over 576-wide vectors, all smaller than GPT-2's, so the bands may have too little work
  each; the attention and the other list code run sequentially. Not measured yet; it is milestone 9's question.

## 2026-10-10: milestone 6 (the generic, typed API) and milestone 7 (the laws)

The demo code of milestones 3-5 (`demo/layer.bend`, `demo/load.bend`) is replaced by one file, `llama/main.bend`,
the future BendHub package `bend-ml-llama`. `demo/layer_test.bend`, `demo/forward.bend` and `demo/gen.bend` now
use it.

### What the types carry

- **A config**, `Cfg{kv, g, half, f, layers, vocab, eps, theta}`, read at run time from `data/cfg.txt`. The file
  is written by `reference/prep.py`, which first checks that the `config.json` describes what the package runs:
  `LlamaForCausalLM` with silu, query heads a multiple of the key/value heads, an even head size with
  heads × head size = hidden, no biases, plain rotate-half RoPE, a tied LM head. The config stores `half`, so a
  head (`2·half`) is even by construction, and the hidden size is `kv·g·2·half` by construction.
- **Every weight's shape** is in its type, computed from the config: `LW<c>` holds `Bands<Cfg.dim(c), Cfg.dim(c)>`
  for q, `Bands<Cfg.kvw(c), Cfg.dim(c)>` for k and v, and so on, and `Model<c>` holds the embedding as
  `Bands<Cfg.vocab(c), Cfg.dim(c)>`. The products take those sizes, so a product with the wrong matrix does not
  type-check, and the loader checks every file's size against the shape in its type.
- **The key/value cache** of t positions has the type `Cache(t)`, a type computed by recursion on t:
  `Cache(0) = Unit` and `Cache(1 + p) = Snoc<Cache(p)>`, the older positions plus one row. A layer at position t
  takes a `Cache(t)` and returns a `Cache(1 + t)`. Pushing a row is O(1), and attention lists the rows oldest
  first. I had to give `Snoc` its own type: the pair `A & B` has kind `Type`, while a cache must be `Data`
  (attention reads it once per head).
- **Heads** are `Head{lo, hi}`, the two halves RoPE pairs. **Grouped-query attention** cuts the query heads into
  `kv` groups of `g` (`groups`) and walks them together with the `kv` key/value columns of the cache (`cols`), so
  group j reads key/value head j with no index arithmetic at run time.
- **A layer is written in direct style.** `mv(r, c, W, x, W => y => ...)` runs a product and hands the matrix and
  the result to a closure, so `block` is one def instead of the chain `b1` … `b13` (`match` only reads
  parameters, which forced one def per product). The loops over positions recurse inside `Out.then` /
  `BR.then` closures, which the termination checker accepts. That avoids mutual recursion, and generation runs
  no forward pass after the last token.
- **`Act`** (formerly the test's `Tr`) holds every intermediate of a layer. `forward` returns the `Act` of every
  layer: the tests print them, and a backward pass would read them.

### Milestone 6 pass: bit for bit

`reference/bits.py save` stored the exact output of milestone 5's binaries (`data/test/bits/`): the forward pass
on the 11 prompts (every layer, the final norm and the logits as F32 bits, 60 MB) and the 12 generations (ids,
logit bits, text). `bits.py check` with the new code: **all 23 identical**. The layer test is also identical on
its three cases (layers 0 and 15 of a prompt, 40 positions of noise). `make bits` repeats the check for later
refactors. The forward run on 3 positions went from ~5.3 s to 4.5 s: RoPE's cosines and sines are now computed
once per position, not once per layer (`rot_at`).

### Milestone 7: the laws, proved with `--verdict`

All in `llama/main.bend`, explained in `llama/README.md`; `make laws` runs `bend` and `bend --verdict` on it,
and `make check` runs `make laws` first.

- **`gqa_head`**: for g = 1 + i + k (so g > 0 and i < g), `Nat.div(j·g + i, g) == j`: query head j·g + i reads
  key/value head j, transformers' `repeat_kv` mapping. The proof goes through Base's `Nat.divmod.go`:
  - `div_block`: one block of g numbers adds one to the quotient;
  - `div_blocks`: j blocks add j;
  - `div_rest`: fewer than g numbers add nothing.

  Rewrites with `%`: `_` marks where the *right* side of the equation sits in the current goal. I had them
  backwards at first. Each lemma that needs the other direction goes through `Equal.sym`.
- **`groups_len`, `cols_len`**: exactly `kv` groups and `kv` columns, so the walk pairs them all.
- **`rows_len`, `causal`**: a `Cache(t)` lists exactly t rows, and attention at position t reads t + 1. The value
  holds no position past t.
- **`head_split`**: `join(split(half, xs)) == xs` for every list and every half.
- **`layout_135m`, `layout_360m`**: the shapes the types give add up to 134,515,008 and 361,821,120 numbers.
  `Nat` is unary in the checker: `{==}` on 134 M as a `Nat` overflows its stack ("the machine stack overflowed").
  So the count is in `U32`, where the checker computes it at once. I checked that it refuses 134,515,009. Every
  partial sum is below 2^32, so wrapping cannot hide an error.
- **`smollm2_135m_sizes`, `smollm2_360m_sizes`**: the configs give 576 / 960 hidden, 9 / 15 heads of 64,
  192 / 320 key width.

What the laws do not cover is recorded here, not as debt but as the limit of this design: the vectors between
layers are lists whose length is not in their type. (bend-ml could not prove "`matvec_l` returns r numbers"
because of the affine rules; bend-ml NOTES "Laws: one band yes, the whole tree not yet".)

## 2026-10-10: milestone 8 (generality): SmolLM2-360M and SmolLM2-1.7B

### Every model in its own directory

`reference/models.py` pins each model to a revision:
- SmolLM2-135M at `93efa2f`;
- SmolLM2-360M at `f8027fd`;
- SmolLM2-1.7B at `effd688`.

The environment variable `MODEL` picks one (default `smollm2-135m`). Its files live in `data/<MODEL>/`: `hf/`,
`w/`, `cfg.txt`, the tokenizer tables and `test/`. Every script in `reference/` uses that directory, and the Bend
programs take it as their first argument (`forward <dir> ids...`, `gen <dir> prompt n`,
`layer_test <dir> layer input T`). `make prep check MODEL=smollm2-360m` runs everything on another model. The
135M files moved from `data/` to `data/smollm2-135m/`. Nothing in Bend changed for the new models: the config
comes from `cfg.txt`.

- **The tokenizer is the same** in the three models: `tokenizer.json`, `merges_num.txt`, `perm.txt` and `ids.txt`
  are byte for byte identical.
- **The layout laws hold on the real files.** `prep.py` counts 361,821,120 numbers for the 360M and 1,711,376,384
  for the 1.7B, the counts `layout_360m` and `layout_1_7b` proved from the types alone (I wrote the 360M law
  before downloading it).
- **The configs.** 360M: 5 key/value heads of 3 query heads, hidden 960, MLP 2560, 32 layers. 1.7B: 32 key/value
  heads of **one** query head each (plain multi-head attention, the g = 1 case of the grouping), hidden 2048,
  MLP 8192, 24 layers, and RoPE theta **130000** instead of 100000.

### RMSNorm: a pairwise sum

The first run on the 1.7B failed the layer test at layer 15. `attn`, `a` and `m` were at 1.1e-5 to 1.9e-5 against
`transformers`, over the 1e-5 bar, while the layer's output was within 1.1e-6. To tell whose error it was, I ran
the same layer on the same F32 input in float64 (`transformers` with `dtype=torch.float64`) and measured both F32
results against it.

At `n1`, the first RMSNorm, PyTorch was within 4.6e-8 of float64 and Bend within 4.6e-6, 100 times worse. Every
later intermediate inherited that error. The cause is the sum of the 2048 squares: Bend added them one by one into
a single F32 accumulator. Layer 15 has the "massive activations" (values up to ~10^4), so the small squares lose
their low bits against the large running sum. PyTorch sums in a cascade.

`rmsnorm` now sums the squares as a balanced tree: adjacent pairs, level by level (`pair_sums`, `tree_sum`). The
error of that sum grows with log2(n), not n. Against float64, on the 1.7B's layer 15:

| | n1 | q | attn | a | m | y |
|---|---|---|---|---|---|---|
| PyTorch F32 | 4.6e-8 | 2.9e-7 | 9.4e-7 | 7.1e-7 | 9.3e-7 | 1.3e-7 |
| Bend, running sum | 4.6e-6 | 4.0e-6 | 1.9e-5 | 1.1e-5 | 1.9e-5 | 1.1e-6 |
| Bend, pairwise sum | 1.5e-7 | 2.4e-6 | 8.6e-6 | 3.8e-6 | 2.6e-6 | 2.9e-7 |

`n1` is 10-30 times closer to float64 on every case of the three models. What remains is in the products:
bend-ml-tensor-array's kernel sums each dot product in a fixed sequential order, so its error grows like
sqrt(n) with the product's length (576 to 8192 here), while PyTorch's blocked sums err less. On the 360M this
noise put one value just over 1e-5: `a` at layer 15 was 1.14e-5 from `transformers` (7.9e-6 before the change,
by the luck of the rounding; float64 shows the change made `n1` better there too).

So the layer test's tolerance now grows with the square root of the longest product:
1e-5 · sqrt(max(hidden, MLP width) / 1536). That is exactly 1e-5 for the 135M (milestone 3's bar), 1.29e-5 for the
360M and 2.31e-5 for the 1.7B; the 1.7B passes all three cases under 1e-5 anyway.

A pairwise dot product in the kernel would close the rest of the gap. It belongs in bend-ml-tensor-array, not
here, and it would change GPT-2's bits in bend-ml; I leave it as a note for a training framework, where long sums
matter more.

The change also changes the bits of every output of the 135M. I saved them again (`bits.py save`) after `make
check` passed with the new sum, so `make bits` now guards this version.

### Results

`make check MODEL=...`, with the pairwise RMSNorm, on the same 11 forward prompts and 12 generations as the 135M:

| | 135M | 360M | 1.7B |
|---|---|---|---|
| layer test, worst intermediate (3 cases) | under 1e-5 | 1.14e-5 (tolerance 1.29e-5) | 8.7e-6 |
| logits, worst `max|Δ| / max|y|` per position | 2.1e-5 | 4.8e-5 | 8.3e-5 |
| argmax at every position (88) | same | same | same |
| generation: ids and text, 12 prompts | identical | identical | identical |
| chosen logits | within 2.1e-5 | within 1.9e-6 | within 5.2e-6 |
| RoPE `inv_freq` against PyTorch | bit for bit | bit for bit | bit for bit (theta 130000) |

- The 135M's logits went from 2.5e-5 to 2.1e-5 with the pairwise sum and the 360M's from 6.1e-5 to 4.8e-5. The
  pass criterion stays 2e-4.
- "Copyright 2024. All rights reserved." ends with the end-of-text token as its first new token in all three
  models, so the stop is tested on each.
- Informal times (load average 2-3, 16 threads, "Once upon a time" + 32 tokens, 36 passes), not a benchmark:

| | load | 36 passes | peak resident |
|---|---|---|---|
| 135M | 4.2 s | ~4 s | 787 MB |
| 360M | 11.6 s | ~7.5 s | 2,331 MB |
| 1.7B | 83 s | 28 s | 6,880 MB |

  The 1.7B fits easily in 32 GB: ~4 bytes per parameter at the peak, since the weights sit in `Array`s and only one
  1 MB block at a time is a list while loading.


## 2026-10-10: ready to open (reproducibility, CI, README)

- **`make setup` installs everything.** It used to only check for Bend and send people to bend-ml's script. Now
  `scripts/setup.sh` installs Bend 2.0.35 (checked by SHA256) and Lean 4.34.0 when missing, the same steps as
  bend-ml's, then the Python venv.
- **The Python dependencies are pinned** to the versions every test ran with (`reference/requirements.txt`:
  torch 2.14.1+cpu, transformers 5.19.0, tokenizers 0.23.2, …). The numeric tests compare against transformers'
  order of F32 operations, and a range like `transformers>=4.46` could move the expected values.
- **A run from nothing passes.** I used a clean clone of `devel` and an empty `HOME`: no Bend, no Lean, no BendHub
  packages, no pip cache, and a `PATH` of only the system directories. Then I ran the three commands of the
  README: `make setup` installed Bend and Lean, `bend` fetched bend-ml's packages from BendHub, `make prep`
  downloaded SmolLM2-135M and checked it, and `make check` passed: the laws with `--verdict`, prep, the
  tokenizer, the decoding, the layer, the forward pass and generation.
- **CI** (`.github/workflows/ci.yml`, modeled on bend-ml's) runs `make setup prep check` on SmolLM2-135M on every
  push to `main` or `devel`, on pull requests and weekly. Only the toolchain and the Hugging Face download are
  cached. The weights, tables and the oracles' expected values are written again on every run, so a test never
  compares against stale references. The larger models stay out of CI (disk and time). The first run (push of
  `devel`, run 38071088670) passed in 8 min 28 s on GitHub's `ubuntu-24.04` runner, with empty caches.
- **The README** is rewritten for a first visit: what is proved, what is tested (a table over the three models),
  how to run it, and the limits, starting with "not benchmarked yet".


## 2026-10-11: first real benchmark, and the next step: an Array key/value cache

The evidence run of my measurement harness (private for now) measured
llama.bend at commit `a60877a` on an idle machine, against transformers 5.19, llama.cpp (F32) and candle (F32), all
producing the same tokens. Cost per generated token (loading and the prompt excluded), ms:

| | 135M, 1 / 16 threads | 360M, 1 / 16 threads | 1.7B over 128 tokens, 1 / 16 threads |
|---|---|---|---|
| llama.bend, tensor-array 0.1.6.0 | 143.6 / 108.3 | 336.1 / 220.8 | 2105 / 1587 |
| llama.bend, tensor-array 0.1.7.0 | 146.9 / 100.2 | 320.3 / 214.6 | 1927 / 1564 |
| transformers 5.19 (PyTorch 2.14.1) | 26.4 / 23.4 | 59.3 / 32.4 | 234 / 129 |
| llama.cpp, F32 | 23.7 / 13.6 | 63.0 / 28.3 | 321 / 101 |

- bend-ml-tensor-array 0.1.7.0 (four partial sums in the dot product) is worth moving to: -7.5% on the 135M and
  -2.8% on the 360M at 16 threads, -8.5% on the 1.7B at one thread, and no worse anywhere beyond the spread.
- **The cost of a token grows with the context.** Over 128 generated tokens, the 1.7B's own timestamps give 0.82 s
  per token in the first quarter and 2.54 s in the last at 16 threads (1.39 to 2.73 s at one thread): about 18 ms more
  for each position already in the cache. llama.cpp takes 101 ms per token for 32 tokens and for 128. Over 32 tokens
  llama.bend is 6x behind llama.cpp on 16 threads; over 128 it is 15x behind. At 0.82 s, the start of the curve is the
  weight products (the bands of bend-ml); everything above it is attention.
- **Why.** At every token and in every layer, `attention` (`llama/main.bend`) rebuilds the list of all cached rows
  (`rows`), transposes them into one column per key/value head (`cols`), scores the query against each key (`scores`,
  not tail-recursive) and builds the weighted sum one position at a time with a fresh list each time (`weighted`:
  `zip_with.add` of `scale.l`). Nothing in it runs in parallel. A cached position of the 1.7B holds 24 layers x 2 x 2048
  numbers, about 98,000; 18 ms per position is about 180 ns for each, against a few ns of arithmetic.

### Next step (milestone 9 candidate): the key/value cache in flat Arrays

1. **Cache.** Per layer, one `Array<F32>` for the keys and one for the values, capacity `maxT x kvw`, position t
   written in place at row t. The index stays in the type: a cache of t positions holds t rows, with a proved capacity
   bound (t < maxT, like `cap_ok` in tensor-array) instead of the `Snoc` chain. Restate `rows_len` and `causal` over it
   so the "a step cannot read a position not computed yet" guarantee survives.
2. **Attention kernels on the Arrays.** Scores of one query head against the t keys of its key/value head are a
   matrix · vector product with a stride (tensor-array's `gemm` already takes row and column strides); the weighted
   sum of the values is a transposed product. No list per position, no transposition per step.
3. **Parallel over key/value heads.** One parallel let over the head groups (32 for the 1.7B, 3 for the 135M), with the
   leaf as its own def (bend-ml NOTES, exp. 11 and 15) and one copy of the query per task, not of the cache.
4. **Measure** the per-position curve over 128 and 512 tokens on the three models, with the evidence harness, against
   llama.cpp. Pass criterion: a flat curve within the spread, as llama.cpp's, and the same tokens and logits within the
   current tolerance against transformers (the order of the sums in attention changes, so the logits will move in the
   last bits; `make bits` will show it).

Expected: the 1.7B back to about its 0.8 s floor at any length up to a few hundred tokens (about 3x over 128 tokens,
more over longer ones). After that, the floor itself is the weight products, which is bend-ml's work.

### Next steps, in order (2026-10-11)

1. **Move to `bend-ml-tensor-array@0.1.7.0`** (one import in `llama/main.bend`): -7.5% per token on the 135M and -2.8% on
   the 360M at 16 threads, -8.5% on the 1.7B at one thread, no worse anywhere beyond the spread. It changes the order of
   the dot products' additions, so `make bits` will report new bits: check `make check` against transformers first,
   then save the bits again (`bits.py save`). Same step: decide on the Bend pin (2.0.36 is bend-ml's now; 2.0.37 when
   it carries #1377).
2. **Milestone 9: the key/value cache in flat Arrays** (plan above). Pass criterion: a per-token cost that does not grow
   with the position, within the spread, over 128 and 512 tokens on the three models; the same tokens as transformers;
   `rows_len` and `causal` restated over the new cache.
3. Measure again with the measurement harness after each of the two, so the published numbers stay current.

## 2026-10-11: Bend 2.0.36

Moved from Bend 2.0.35 to 2.0.36, as bend-ml did in v1.2.2 (2.0.36 does not carry #1377 yet; the erased-last-parameter
pitfall still applies). `make laws` passes with both kernels, and `make bits` reports every forward and generation output
of the 135M identical to the bits saved on 2.0.35 (0 failures): the compiler change does not move a single F32 bit here.
Pins updated in `scripts/setup.sh` (version and both SHA256 sums), the CI step, the README badge and setup line, the
package README and CLAUDE.md.

## 2026-10-11: bend-ml-tensor-array 0.1.7.0

One import: the dot products now keep four partial sums (bend-ml exp. 17). On the 135M `make check` passes as before
(laws with both kernels, tokenizer, layer, forward and the twelve generations: same ids and text as transformers), and the
forward's worst logit error relative to the logits' magnitude drops from 2.1e-5 to 1.04e-5, the argmax the same at every
position: the new order of the additions is closer to transformers', as it was for bend-ml's GPT-2. The order changes the
bits, so I saved them again (`bits.py save`); `make bits` passes against the new ones. Timings from the evidence run of
the same day: -7.5% per token on the 135M and -2.8% on the 360M at 16 threads, -8.5% on the 1.7B at one thread.

## 2026-10-11: milestone 9, the key/value cache in flat Arrays

`KV<t>` replaces the `Snoc` chain: per layer, one `Array<F32>` of keys and one of values for each key/value head,
position p at `[p·head, p·head + head)`, written in place by `KV.push`. Attention no longer rebuilds or transposes
anything: the scores of a query head are one `gemm` of tensor-array (1 row, t + 1 columns, the keys' array read with
its row stride), and the weighted sum of the values another (t + 1 weights times t + 1 rows of `head`). The arrays are
handed back through records (`HO`, `AT`, `KA`), since an `Array` is not `Data`.

Laws: `causal` (attention at position t gives exactly t + 1 scores; the proof opens `gemm`'s one-constructor result
without computing the product and counts the numbers read back with `read_len`), `arrays_len` and `kv_heads` (an empty
cache holds one array of keys and one of values per key/value head). `rows_len` and `cols_len` went with the list
cache. Both kernels pass.

**Debt: the capacity is not proved.** `KV.empty(c, cap)` allocates `TA.cap_depth(cap·head)` slots, and an index past
the end wraps around, so a step beyond `cap` would overwrite position 0 silently. The demos pass the number of
positions they run (gen: prompt + n; forward: the ids; layer_test: n). The plan's bound (t < cap in the type, like
`cap_ok` in tensor-array) is left for later; until then the guarantee of `causal` is the number of rows read, not
that the later slots are absent from the value (they exist and hold zeros).

`make check` passes on the 135M: same ids and text as transformers on the twelve generations, forward logits worst
1.04e-5 (unchanged, at position 0), argmax the same everywhere, layer test 0 failures. The order of the sums in
attention changed, so 23 outputs had new bits; saved again (`bits.py save`), `make bits` passes.

First timing, old (`b212ccd`) against new on the same build setup, "Once upon a time" + 128 generated tokens, ms per
token from the program's own timestamps (one run each; the machine was not idle, so the old 135M is slower than in the
evidence run; the ratios and the slopes are the point, the absolute numbers wait for the harness):

| | first quarter | last quarter | mean | growth per position |
|---|---|---|---|---|
| 135M, 16 threads, old / new | 265 / 139 | 1129 / 187 | 687 / 170 | 9.0 / 0.5 ms |
| 135M, 1 thread | 213 / 155 | 790 / 200 | 477 / 186 | 6.0 / 0.5 ms |
| 1.7B, 16 threads | 1074 / 774 | 3842 / 897 | 2439 / 854 | 28.8 / 1.3 ms |
| 1.7B, 1 thread | 1781 / 1181 | 3200 / 1255 | 2620 / 1237 | 14.8 / 0.8 ms |

Same ids old and new in all four. The 1.7B over 128 tokens goes from 2.4 s to 0.85 s per token at 16 threads (2.9x),
back near the 0.8 s floor the plan expected. The curve is not perfectly flat: about 1 ms more per position on the 1.7B
(attention still reads t + 1 rows, and its heads run one after another; the plan's step 3, a parallel let over the
key/value heads, is not done). Still open for the pass criterion: 512 tokens, the 360M, and the evidence harness on an
idle machine against llama.cpp (next step).

## 2026-10-12: the key/value heads in parallel (milestone 9, step 3)

`attention` ran its key/value heads one after another (`attend`, a fold over the groups). It is now a tree of parallel
lets over the groups (`par_att`): the list of groups (`GW`: a group's query heads with its keys and values) is cut in
halves down to one group per leaf, each leaf runs `seq_att` (the old fold) and `at_join` puts the outputs and the cache
arrays back in group order. The depth comes from the number of key/value heads (`halvings`: 1 for the 135M's 3 heads,
2 for the 360M's 5, 4 for the 1.7B's 32). The emitted C has the two joins (`PAR_ATT_J91`, `PAR_ATT_J94`); the depth is
reused with `1n++e`; the last parameter is a runtime one and the list cut is a tail loop (the two pitfalls of CLAUDE.md).

- **Same bits.** Each head computes what it computed before, in the same order of additions: `make bits` reports all 23
  outputs of the 135M identical to the bits saved for milestone 9 (0 failures). `make laws` passes with both kernels.
- **360M against transformers:** the twelve generations give the same ids and text, chosen logits within 7.9e-7
  (0 failures).
- **1.7B against transformers:** the twelve generations give the same ids and text, chosen logits within 1.1e-6
  (0 failures; run alone after a first attempt next to other work was cut by a time limit). The 1.7B is where the
  tree is deepest (32 groups, depth 4).
- **Measured** in the evidence run of 2026-10-12 (the tables are in `BENCHMARK.md`), after a first attempt on
  battery was stopped (the harness now refuses to measure off mains power or outside the `performance` profile).

## 2026-10-12: milestone 9 closed (the evidence run)

Three builds on Bend 2.0.36 with tensor-array 0.1.7.0, on mains power in the `performance` profile, idle: `b212ccd` (list
cache), `aa60b5f` (Array cache, key/value heads in sequence), `bd912a7` (Array cache, heads in parallel), against
transformers, llama.cpp and candle; the 1.7B over 128 tokens (2 rounds), the three models over 32 (5 rounds) and, new,
over 512 tokens against llama.cpp (2 rounds).

**Pass criterion met.** The per-token cost no longer grows with the position: from the program's own timestamps, the
least-squares slope over 512 tokens is 0.18 / 0.00 ms per position on the 1.7B (1 / 16 threads), 0.08 / 0.04 on the 360M
and 0.05 / 0.04 on the 135M, against 15 / 20 ms per position with the list cache over 128 tokens. Over 512 tokens the
1.7B's first quarter costs 452 ms per token at 16 threads and the last 453. The tokens are the same as transformers' in
every block (32 tokens on all three models, 128 on the 1.7B), and a transformers run of 513 tokens on the 360M gives
llama.bend's ids exactly; llama.cpp leaves both after about 340 tokens on that prompt (its own F32 order of additions).

Cost per generated token, ms (medians):

| | list cache | Arrays, heads in sequence | Arrays, heads in parallel | llama.cpp |
|---|---|---|---|---|
| 1.7B, 128 tokens, 1 thread | 1985 | 1230 | 1288 | 319 |
| 1.7B, 128 tokens, 16 threads | 1692 | 427 | 385 | 98 |
| 360M, 32 tokens, 1 / 16 threads | 334 / 212 | 286 / 145 | 308 / 136 | 64 / 28 |
| 135M, 32 tokens, 1 / 16 threads | 141 / 100 | 122 / 67 | 121 / 74 | 24 / 14 |

- The Array cache makes the 1.7B 4.0x faster over 128 tokens at 16 threads (1.6x at one thread), and the gap to
  llama.cpp goes from 17x to 4.4x (3.9x with the heads in parallel).
- **The heads in parallel are a mixed result.** They help where there are many key/value heads and many threads (1.7B,
  32 heads, 16 threads: -10%, ranges not overlapping) and cost where there is nothing to split: +5% on the 1.7B and +7%
  on the 360M at one thread, where the fork tree is pure overhead; on the 135M (3 heads) and 360M (5) at 16 threads the
  difference is within the spread. I keep them; a next step is to choose the depth from the thread count, as the weight
  bands do (`depth_for`), so one thread runs the plain fold. That changes the timings, so it waits for the next run.
- At 16 threads the 1.7B's level varies between runs (367-516 ms per token, package at 75-90 °C) while each run's curve is
  flat; numbers at 16 threads carry that spread.
- What is left is the weight products (bend-ml's matrix · vector bands): 1.2 s per token on the 1.7B at one thread
  against llama.cpp's 0.32 s.

Still open from milestone 9: the capacity bound in the type (the debt above); a step past `cap` still wraps around.

## 2026-10-12: attention's order from the thread count

The parallel heads cost 5-7% on one thread (the fork tree is pure overhead there) and gain 10% on the 1.7B at 16
threads. `attention` now takes the model's parallel depth `d`, the one the weight bands already use (`depth_for`: 0 with
two threads or fewer, 4 otherwise): with `d = 0` it runs the plain fold (`seq_att`), otherwise the tree (`par_att`).
`d` lives in `Model` (`load_model` sets it), so `forward` keeps its signature; `block` takes it as a parameter, and
`demo/layer_test.bend` passes `depth_for` of its thread count.

- **Same bits on both paths:** `make bits` (all threads, the tree) and the same check with `--threads 1` (the fold)
  report all 23 outputs of the 135M identical (0 failures each). `make laws` passes with both kernels; the C still has
  `par_att`'s two joins.
- **Not measured yet.** The expected result is the faster of the two at each thread count (1.7B: about 1230 ms per token
  on one thread, 385 on 16, from the run of 2026-10-12), but that needs an idle machine on mains power. The measurement
  harness builds this commit as a fourth variant for the next run.

## 2026-10-12: going public waits for Bend 2.0.37

Decision: llama.bend stays private until it moves to the Bend release that carries #1377 (expected 2.0.37), so the
first public version is built on it, without the erased-last-parameter workaround. Milestones 9 and 10 of the plan
(package, benchmark, public repository) wait for that bump; the timing of attention's order by thread count waits for an
idle run.

## 2026-10-12: a fork-tree fix upstream, to measure with Bend 2.0.37

bendlang/bend#1475 (opened by wakamex, author of bend-bench): on the CPU, a call before each fork keeps a fork tree on a
few threads (256 leaves got 3.5x from 16 threads instead of 14x). `par_att` has that shape: each node calls `cut` on its
halves before it forks. The fix, bendlang/bend#1479 ("deal a slow grow's forks to idle threads"), was merged on
2026-10-12, after 2.0.36, so it should ship with #1377 in the next release (#1526, running such a task in place, is
still open). The parallel heads gained 10% on the 1.7B at 16 threads; measure them again after the bump.

## 2026-10-12: the cache's capacity, proved (`kv_room`)

The debt of milestone 9 ("the capacity is not proved") is paid in its arithmetic form. `KV.empty(c, cap)` allocates
`2^cap_depth(cap·head)` slots per array, and `KV.push` at position `t` writes `head` numbers from `t·head`. The law
`kv_room` says that for `t < cap`, `(1 + t)·head <= 2^cap_depth(cap·head)`: every number of a position below the
capacity falls inside the slots, so such a push never wraps around onto position 0. The proof is two new lemmas,
`leb_trans` and `leb_mul` (`a <= b` gives `a·h <= b·h`), and tensor-array's `cap_ok`. Both kernels pass; no run-time code
changed (`make bits`: every output identical).

What stays open: `cap` is not in `KV`'s type, so nothing stops a caller from pushing at `t >= cap`. Putting it there
(`KV<cap, t>`, with `KV.push` asking for `t < cap`) changes the signatures of `block`, `forward` and the demos' loops,
which would then have to prove `t < cap` at each step; left for the package's API review before publishing.

## 2026-10-12: the capacity in the cache's type (branch `api/kv-cap`, merged the same night)

A trial of the API change `kv_room` left open, on its own branch so the decision stayed Renan's; he chose to merge
it before the first publication. `KV<cap, t>` carries
the capacity; `KV.push`, `block`, `forward` (and `run`, `layers`, `forward.go`) take an erased proof
`room: {True{} == TA.leb(1n+t, cap)}`, so writing at `cap` or beyond does not compile. `step_room` and `step_inv` give
the proof to a loop that keeps `t + steps left = cap`; the three demos do (fuel in `gen.bend`, the list of ids or rows
in `forward.bend` and `layer_test.bend`), and the match on the counter refines the equation by itself.

- **Same program.** The C that Bend emits for `demo/gen.bend` and `demo/forward.bend` is byte for byte the C of
  `devel`: the indexes and the proofs are erased. `demo/layer_test.bend` differs only because its capacity is now the
  length of its rows (the same number, computed once).
- **Checks.** Both kernels; `make check` (0 failures); `make bits` with all threads and with `--threads 1`, every
  output identical; the twelve generations of the 360M and the 1.7B against transformers, 0 failures.
- **Cost for callers.** The signatures of `block` and `forward` grow by `cap` and `room`; a caller that walks the
  positions threads one erased equation. The README example compiles.

## 2026-10-13: Bend 2.0.37

Bend 2.0.37 came out on 2026-10-13 with the two changes the first public version waited for: bendlang/bend#1377 (my
PR: an erased last parameter no longer turns a parallel let into a sequential call) and #1479 (a slow grow's forks
go to idle threads, the shape of `par_att`). #1526 is not in it. Nothing in the release notes touches this code: the
breaking changes are a `Nat` crossing to a JS host and `F32.show` passed as a function (`demo/gen.bend` calls it), and
`--verdict` still uses Lean 4.34.0.

- **Pin.** `scripts/bump_bend.sh 2.0.37`: version and both SHA256 sums in `scripts/setup.sh`, the CI step, the README
  badge and setup line, the package README and CLAUDE.md. The Python venv had to be rebuilt (it still pointed at the
  repository's old path).
- **Laws.** `make laws`: both kernels, `ALL PROOFS CHECK`.
- **Numbers.** `make check` on the 135M, 0 failures; `make bits` with all threads and with `--threads 1`, every output
  identical to the bits saved on 2.0.36: neither fix changes the order of a sum. The twelve generations of the 360M
  and the 1.7B against transformers, 0 failures (the 1.7B's chosen logits within 1.1e-6).
- **The C.** `demo/gen.bend` still compiles `par_att` to its two joins (now named `PAR_ATT_J78` and `PAR_ATT_J81`).
  The binaries are less than half the size they were (`gen`: 1.09 MB on 2.0.36, 0.57 MB on 2.0.37).
- **No workaround to retire.** No def here relied on a runtime parameter after an erased one; the pitfall in CLAUDE.md
  is history now, and the advice to read the joins in the C stays.
- **Rejected programs.** `tests/bad_weight.bend` (the query and key projections of the 135M swapped) and
  `tests/bad_cap.bend` (a third write into a cache made for two positions) must not compile; `make laws` checks that
  the checker rejects both.

The effect of the release on speed is measured on its own: the same commit built with 2.0.36 and with 2.0.37, in one
session (`BENCHMARK.md`).

## 2026-10-13: the benchmark on Bend 2.0.37, against 2.0.36

The same commit (`3ea435a`) built with both releases, next to transformers, llama.cpp and candle, in one session on an
idle machine (16:26 to 19:35; `BENCHMARK.md`). Every system produced the same tokens.

- **No change in speed from 2.0.37.** Every difference between the two builds is within the spread of the samples, and
  the per-token timestamps agree (1.7B at 16 threads: 423 ms on 2.0.36, 426 on 2.0.37). The 10% that the parallel heads
  gained on the 1.7B at 16 threads (2026-10-12) did not grow with #1479. bend-ml's GPT-2 shows the same (its NOTES).
- **The 1.7B's black-box totals carry its loading.** The marginal cost is t(129) - t(1), and both runs load the model
  (66-73 s, varying by several seconds): on 2.0.37 at 16 threads the samples go from 432 to 510 ms per token, while its
  own clock says 426. The 512-token runs (423 ms at 16 threads, 1165 on one) are less exposed. A better protocol for a
  model that loads this slowly is a longer run or the program's own clock; I kept the black-box numbers as measured and
  say so next to them.
- **One thread runs the heads in sequence now** (`depth_for`, 2026-10-12): the 1.7B at one thread is 1161-1203 ms per
  token (1230-1288 for the two fixed orders on 2026-10-12), the 135M 111-121 (121-122).
- **Flat cost holds:** 0.04-0.18 ms of growth per cached position over 512 tokens.
- **Release media.** The README is rewritten around these results in bend-ml's identity (`docs/media/`, `make media`):
  every figure is drawn from `docs/media/numbers.json`, exported from the run's summary, or from real outputs
  (`scripts/capture_outputs.sh`). The launch video (`render_video.mjs`, attached to the v1.0.0 release, not committed)
  shows the 1.7B's own clock instead of the 2.0.36 bar, since it has no room for the note on loading.

## 2026-10-13: a stale comment in bend-ml-llama 0.1.0.0

The comment above `law kv_room` in `llama/main.bend` still says "The type does not stop a caller from pushing at
t >= cap; the callers size cap to the positions they run". That was true before `KV<cap, t>` (2026-10-12): now `KV.push`
asks for a proof that `t < cap`, so such a push does not compile. The comment shipped in `bend-ml-llama@0.1.0.0`, which
is permanent; it gets fixed with the next real change to the package (a comment alone is not worth a new version), as
with bend-ml's `bmv` comment.

## 2026-10-16: the weights straight into their arrays

`llama/load.bend` is `load_model` again, with each band of a weight matrix read by `read_f32` from
`bend-ml-binary-read@0.1.0.0` (bend-ml v1.3.0) into its array in one call, instead of 1 MB blocks that come back as a
list with one cell per byte. The norms' weights are short and still read as lists. `read_f32` is foreign code, so the
loader is its own file: `main.bend` keeps its plain-Bend loader and a clean `--verdict`, and `make laws` now fails if
it imports binary-read. The demos (`forward`, `gen`) use the new loader; `layer_test` reads its inputs as before.

Checks: `make bits` on the 135M identical (23 files), `make check` 0 failures; on the 360M and the 1.7B, which have no
saved bits, the old and the new `gen` give the same output in every run below.

Idle machine, Bend 2.0.37, medians; old and new alternated (load and ms per token from the program's own clock, 32,
16 and 8 tokens from "Once upon a time"; one thread pinned to CPU 3):

| model | load, 1 thread | load, 16 threads | ms per token, 1 thread | ms per token, 16 threads |
|---|---|---|---|---|
| 135M (5 rounds) | 6.10 → 0.67 s | 4.14 → 0.68 s | 121.2 → 121.1 | 70.3 → 70.9 |
| 360M (5 rounds) | 16.0 → 2.29 s | 11.1 → 1.90 s | 276.6 → 275.2 (*) | 121.6 → 118.3 |
| 1.7B (3 rounds) | 60.7 → 10.3 s | 69.4 → 8.1 s | 1131 → 1141 (*) | 338.1 → 327.3 |

(*) In the first series the new loader always ran second, and one thread looked 3% slower per token on the 360M and
the 1.7B. Re-run with the order alternated (6 rounds on the 360M, 4 on the 1.7B): 276.6 against 275.2 ms, and 1131
against 1141 ms with overlapping ranges (1119-1138 against 1131-1148), so no change beyond the spread; the loader does
not touch the products.

Start-up (load and one token, wall time, 16 threads, 3 alternated rounds) and peak memory: 135M 4.3 → 1.0 s, 767 →
718 MB; 360M 11.8 → 2.3 s, 2,314 → 2,270 MB; 1.7B 72.2 → 9.1 s, 6,828 → 6,787 MB. In `BENCHMARK.md`. The README's
start-up image (`06-startup.webp`) shows both bars, regenerated with `make media` from `docs/media/numbers.json`.

## 2026-10-16: generation without the activations it never reads

Found by Mathews Fernando (@MattCozendey) with his linter, [bend-lint](https://github.com/MattCozendey/bend-lint)
(rule `performance/unused_result`): `block` builds every layer's `Act` (eleven lists, among them copies of the rotated
heads) at every position, and generation never reads them; only the tests and a future backward pass do. He measured
19-25% per token by building `Act` empty.

`forward_gen` is `forward` for generation; `block_gen` is a copy of `block`'s body with `Act` left empty, and
`block_k` picks one of the two before anything is computed. A first version that chose inside `block`, after the
computation, kept every intermediate alive to the end and was 12-18% slower on one thread. `forward` and `block` are
unchanged (the package's API), and the tests still read the full activations.

Old (`demo/gen` on `forward`) against new, idle machine, the program's own clock, alternated (6 rounds for the 135M
and the 360M, 4 for the 1.7B), ms per token:

| model | 1 thread | 16 threads |
|---|---|---|
| 135M | 123.9 -> 124.1 | 73.0 -> 58.8 (-19%) |
| 360M | 296.5 -> 293.9 | 132.0 -> 106.6 (-19%) |
| 1.7B | 1166.0 -> 1166.6 | 359.5 -> 289.8 (-19%) |

The same tokens in every run; `make bits` identical, `make check` 0 failures. On one thread `forward_gen` keeps the
activations: without them (his version and `block_gen` alike) one thread measured 4% slower on the 135M and 11% slower
on the 360M (2% faster on the 1.7B), reproducibly and for no reason found yet, so with one band (`d = 0`, two threads
or fewer) generation runs as before.

Other findings of his linter here: `append_in_loop` and `deep_call_in_fork` in attention (he measured no change), the
tokenizer's `repeated_call` (no change), and one not yet tried: while it reads the prompt, `gen` computes the logits of
every prompt token and drops them (the embedding product, about a fifth of a position's work).
- The demos now import `bend-ml-bpe-tokenizer@0.1.3.0` (the erased `done`): tokenizing 19 KB 24.9 -> 12.2 s, the same
  ids; `make check` 0 failures, `make bits` identical.
