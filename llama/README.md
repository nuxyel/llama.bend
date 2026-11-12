# bend-ml-llama

Llama-architecture language models in Bend 2: RMSNorm, rotary position embeddings, grouped-query attention with a
key/value cache, and SwiGLU. The sizes of a model come from a config (`Cfg`), and **every weight matrix has its
shape in its type**, computed from that config, so a model of other sizes is the same code with another config:
SmolLM2-135M, SmolLM2-360M and SmolLM2-1.7B (plain multi-head attention, another RoPE base) all run on it. The
structure is proved by Bend's checker and again by its kernel proved in Lean. The numbers are tested against
PyTorch: all three models give the same greedy tokens as `transformers` in the
[llama.bend](https://github.com/nuxyel/llama.bend) repository.

- Bend: **2.0.40** · License: MIT · Depends on `bend-ml-tensor-array@0.1.20.0`.
- `bend llama/main.bend --verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`, no axioms).

On BendHub as `bend-ml-llama@0.2.1.1`. A local copy works the same way (`import ./llama/main.bend as M`, as the
programs in `demo/` do with `../llama/main.bend`).

```python
import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import bend-ml-llama@0.2.1.1/main.bend as M

# one position: token tok at position t, with the caches of positions 0..t-1 (one per layer) and room for cap
# positions; room is the proof that t < cap (M.step_room gives it to a loop that keeps t + steps left = cap)
def step(+c: M.Cfg, -cap: Nat, +t: Nat, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +tok: Nat) -> M.Out<c, cap, t>:
  M.forward(c, cap, t, m, cs, room, tok)
```

## The laws, in plain language

| Law | What it says |
|---|---|
| `head_split` | A head is stored as its two halves, the pairs that RoPE rotates together. Cutting a list at `half` and joining the two parts gives the list back, so splitting a head loses nothing and moves nothing. The config stores `half`, not the head size, so a head (`2·half` numbers) is always even. |
| `gqa_head` | With `g` query heads per key/value head, query head `j·g + i` (for `i < g`) reads key/value head `(j·g + i) / g = j`. That division is how `transformers` maps query heads to key/value heads (`repeat_kv`). This package groups the query heads instead (group `j` reads key/value head `j`), and the law says both choose the same key/value head. The proof goes through `Nat.div` itself: `j·g` is `j` whole blocks of `g`, and `i < g` more numbers do not complete another one. |
| `groups_len`, `kv_heads`, `arrays_len` | The query heads form exactly `kv` groups, and an empty cache holds exactly `kv` arrays of keys and `kv` of values, one per key/value head. Attention walks the groups and the arrays together, so every group meets its key/value head; none is dropped, and none reads a missing head. |
| `causal`, `read_len` | The cache has the type `KV<cap, t>`: the number of positions written is in its type, and a layer at position `t` takes `KV<cap, t>` and returns `KV<cap, t + 1>`. At position `t`, attention scores the query against exactly `t + 1` keys, positions `0..t`: one score per position written. The scores come from `bend-ml-tensor-array`'s `gemm` over the first `t + 1` rows of the keys' array; `read_len` counts the numbers read back from its output. |
| `kv_room` | The cache is created for `cap` positions: each key/value head gets arrays of `2^d` slots, with `d` chosen by `bend-ml-tensor-array` so that `2^d` is at least `cap·head` (its law `cap_ok`). Writing position `t` fills `head` numbers from `t·head`. The law says that for every position below the capacity (`t < cap`) those numbers fall inside the slots: `(t + 1)·head <= 2^d`. So a push below the capacity never wraps around onto position 0. And a push needs that proof: `KV.push`, `block` and `forward` take a proof that `t < cap` (checked, then erased: it costs nothing when the program runs), so writing at `cap` or beyond does not compile. `step_room` and `step_inv` give the proof to a loop that counts its steps left. |
| `layout_135m`, `layout_360m`, `layout_1_7b` | The shapes that the types give to the weights (the embedding, every layer's nine tensors and the final norm) add up to 134,515,008 numbers for SmolLM2-135M, 361,821,120 for SmolLM2-360M and 1,711,376,384 for SmolLM2-1.7B, the number of values in each checkpoint. The loader stops on any file whose size differs from the shape in its type, so together the files cover the checkpoint exactly. |
| `sample_below` | From logits with at least one value, the token `sample` draws is an id of those logits: its index is below their number, whatever the temperature, `k`, `p` and draw. The top-k pass only ever stores the index of the value it is reading, and the walk keeps "index + values left = length", so each stored index is below the length (`topk_below`, through `ins_below` and `take_below`: inserting and cutting the list keep that property). The draw then returns one of the stored candidates (`pick_below`), whatever top-p kept. The proof never looks at an `F32`: every comparison of logits is a `Bool` the lemmas take as given, so it holds for any outcome of them. |
| `smollm2_135m_sizes`, `smollm2_360m_sizes`, `smollm2_1_7b_sizes` | The configs give the sizes of their `config.json`: SmolLM2-135M has 9 query heads of 64 numbers, a hidden size of 576 and keys 192 wide; SmolLM2-360M has 15 heads of 64, a hidden size of 960 and keys 320 wide; SmolLM2-1.7B has 32 heads of 64, a hidden size of 2048 and keys 2048 wide (one query head per key/value head). |

`add_zero`, `add_succ`, `add_one`, `add_assoc`, `div_block`, `div_blocks`, `div_rest`, `leb_trans` and `leb_mul` are the
arithmetic lemmas that the proofs above use; `below`, `and_l`, `and_r`, `ins_below`, `take_below`, `topk_below` and
`pick_below` are `sample_below`'s.

What is **not** proved: the numbers. `F32` is not a real number. RMSNorm, RoPE, softmax and silu follow the order
of operations of `transformers`' `modeling_llama.py`, and the tests check the results: logits within 8.3e-5 of
PyTorch (relative to the largest logit, over 11 prompts and the three models), the same greedy tokens. RMSNorm sums
its squares as a balanced tree, which keeps it within ~1e-7 of float64 next to the large activations of deep
layers. The vectors between layers are lists whose length is not in their type. Sampling is tested against a Python
twin that repeats it operation for operation (llama.bend's `reference/sample_ref.py`): the same ids and logit bits.

## How a model is described

```python
type Cfg is Data:
  Cfg{kv: Nat, g: Nat, half: Nat, f: Nat, layers: Nat, vocab: Nat, eps: F32, theta: F32}
```

`kv` key/value heads, `g` query heads per key/value head, heads of `2·half` numbers, MLP width `f`. From these
fields: `Cfg.heads = kv·g`, `Cfg.dim = heads·2·half`, `Cfg.kvw = kv·2·half`. `smollm2_135m()`, `smollm2_360m()`
and `smollm2_1_7b()` are given, and `read_cfg(dir)` reads `<dir>/cfg.txt`, written by llama.bend's
`reference/prep.py`, which checks that the `config.json` describes a model this package runs: a Llama with
SwiGLU, no biases, plain RoPE and a tied LM head.

## API

| Name | What |
|---|---|
| `LW<c>`, `Model<c>` | a layer's weights and the whole model; each matrix is a `bend-ml-tensor-array` `Bands<rows, cols>` with its shape from `c`, and the model keeps its parallel depth `d` (`depth_in` of the config and the thread count) |
| `KV<cap, t>`, `KV.empty(c, cap)`, `KV.push`, `caches(c, n, cap)` | the key/value cache of `t` positions, in arrays with room for `cap` positions; a push needs a proof that `t < cap` |
| `step_room`, `step_inv` | for a loop with `t + steps left = cap`: the proof that `t < cap`, and the same equation one step later |
| `block(c, d, lw, cap, t, kv, room, rot, x) -> BR<c, cap, t>` | one layer at position `t` (`d = 0`: attention's key/value heads one after another; otherwise in parallel): the output, the cache grown to `t + 1` positions, the weights handed back, and the activations (`Act`) |
| `forward(c, cap, t, m, cs, room, tok) -> Out<c, cap, t>` | one position of the whole model: logits, the final norm, every layer's `Act`, the caches, the model |
| `forward_gen(c, cap, t, m, cs, room, tok)` | `forward` for generation: the same logits, caches and model, with every layer's activations left empty, about 20% faster per token on 16 threads; `block_gen` is `block` without them |
| `forward_pre(c, cap, t, m, cs, room, tok)` | a prompt position, whose logits nobody reads: the layers and caches as `forward_gen` gives them, without the final norm and the product by the embedding (`lg` and `nf` empty) |
| `BR.then`, `Out.then` | continue with the fields of a result; a loop over positions recurses inside them |
| `argmax`, `AM.i`, `AM.v` | the first index of the largest logit, and that logit |
| `sample(xs, temp, k, p, u) -> Cand` | a token drawn from the logits `xs` (its id `Cand.i` and logit `Cand.v`): the `k` largest logits (ties: the lower id first), weights `exp(v/temp - top/temp)`, top-p `p` over them, and the uniform draw `u` in [0, 1); every step is one F32 operation in a fixed order, so a seed gives the same token wherever the C library's `exp` (which Bend's `F32.exp` calls in double precision) gives the same results; tested with glibc on x86-64 |
| `uniform(seed, n)`, `hash32` | draw `n` of a seed: the top 24 bits of `hash32(hash32(seed) + n)` (Chris Wellons' lowbias32) times 2^-24; no state between draws |
| `topk(k, xs)`, `sample.cs`, `Cand` | the two halves of `sample`: the candidates, largest first, and the draw from them |
| `rmsnorm`, `inv_freq`, `rot_at`, `rope`, `swiglu`, `attention` | the pieces of a layer |
| `read_cfg`, `load_model`, `load_lw`, `depth_in`, `depth_for` | loading a model prepared by `prep.py`, in plain Bend: every tensor read from the downloaded `<dir>/hf/model.safetensors` with `File.read_at`, BF16 widened exactly to F32, at the byte offsets `prep.py` writes into `<dir>/w/st.txt` (since 0.2.1.0; 0.2.0.0 read F32 copies, `<dir>/w/<name>.bin`, which `make prep F32=1` still writes). `depth_in(c, threads)` is the band depth `load_model` gives a model: one band on one thread, two on two, and from three threads on 16 bands for a model at least 1024 wide, or for a narrower one 4 on four threads, 8 on three and on five to eight and 16 beyond; `depth_for(threads)` is the narrow model's rule (up to 0.2.1.1 it gave one band up to two threads and 16 above) |
| `load.bend`: `load_model` | the same model, read from the downloaded `hf/model.safetensors` itself: each band of weights read straight into its array by `bend-ml-binary-read`'s `read_bf16_at` (the same numbers as `main.bend`'s loader, faster: the 1.7B in about 8 s instead of 60-70 with the F32 copies). It is foreign code (C and JavaScript), so it lives in its own file: `main.bend` does not import it and keeps `--verdict` clean |

An `Array` has a single owner, so every product hands its matrix back (`block` returns its `LW`, `forward` its
`Model`). `Act` holds every intermediate of a layer (`n1`, `q`, `k`, `v`, `qr`, `kr`, `attn`, `a`, `n2`, `m`,
`y`). The tests compare them with `transformers`, and a backward pass would read them.

## The tokenizer and generation

Since 2026-11-12 the package holds what a program needs to go from a prompt to tokens, in files of their own beside
`main.bend`. They are in the repository and not on BendHub yet: `bend-ml-llama@0.2.1.1` is `main.bend` alone.

| File | What |
|---|---|
| `tok.bend` | SmolLM2's tokenizer: the pre-tokenizer, the merges (through `bend-ml-bpe-tokenizer@0.1.3.0`), decoding, and SmolLM2-Instruct's chat template |
| `unicode.bend` | the character classes `tok.bend` reads |
| `prompt.bend` | the prompt in one pass: its positions through each layer together, each weight matrix read once for all of them, with the bits of the position-by-position path |
| `gen.bend` | the generation loop over `main.bend`'s forward pass and `prompt.bend`'s: greedy or sampled, one token at a time or all at once; and a conversation that keeps its caches between the turns |
| `package.bend` | the entry to publish: `--publish` sends a file and what it imports, so this one imports the five above; a program imports the files it uses |

```python
import Base
import ./llama/main.bend as M
import ./llama/tok.bend as K
import ./llama/gen.bend as G

# n greedy tokens after a prompt, one id per line (dir: a model directory prepared by llama.bend's reference/prep.py)
def run(+dir: String, prompt: String, n: Nat) -> IO(Unit):
  do IO<Unit>:
    tb : K.Tables <- K.load(dir)
    +c : M.Cfg <- M.read_cfg(dir)
    model : M.Model<c> <- M.load_model(c, dir)
    g : G.Gen<c> = G.generate(c, model, G.Greedy{}, False{}, K.encode(tb, prompt), n)
    List.for_each(~&2, ~Nat, ~(id => IO.print(Nat.show(id))), G.ids_of(G.Gen.toks(c, g)))
```

llama.bend's `demo/hello.bend` is this program with a command line, and `make check` compares its 128 tokens with
`transformers`'.

`tok.bend`:

| Name | What |
|---|---|
| `Tables`, `load(dir)`, `read_tables(merges, perm)` | the merge rules and the byte permutation of a model, built once for encoding and decoding: from a model directory (a malformed merges table stops the program), or from the two files `prep.py` writes, `merges_num.txt` and `perm.txt` (`None` for a malformed table) |
| `encode(tb, text)` | the ids of a text |
| `decode(tb, ids)` | the text of ids: special tokens left out, bytes that are not valid UTF-8 as U+FFFD |
| `chat(tb, ms, gen)` | the ids of a conversation in SmolLM2-Instruct's chat template; `ms` is role, content, role, content, ...; `gen` ends with `<|im_start|>assistant`, which asks for the answer |
| `chat_next(tb, msg)` | the ids that continue a conversation after an answer: `<|im_end|>` and a newline close the assistant's turn, `msg` is the user's next turn, and `<|im_start|>assistant` asks for the next answer |
| `encode_text(rules, perm, bytes)`, `decode_text(rules, perm, ids)`, `chat_ids(rules, perm, ms, gen)` | the same three on UTF-8 bytes and on the two fields of `Tables` |
| `utf8_of(chars)` | the characters of a `String` as UTF-8 bytes |
| `pretok`, `tag_all`, `ids_ok`, `nums`, `rules.go` | the pre-tokenizer, the check of a merges table, and the parsing of the two files |

`gen.bend`:

| Name | What |
|---|---|
| `Pick`: `Greedy{}`, `Draw{temp, k, p, seed}` | how a token is chosen: the argmax, or `main.bend`'s `sample` with draw `j` of the seed for the `j`-th new token |
| `choose(pk, lg, j) -> Chosen` | the `j`-th new token from the logits `lg`: its id and logit (`Chosen.am`) and, for a draw, the top-k candidates it came from (`Chosen.cs`) |
| `generate(c, m, pk, chat, ids, n) -> Gen<c>` | `n` new tokens after the prompt `ids`; fewer after the end-of-text token (id 0) and, when `chat` is set, after `<|im_end|>` (id 2). `Gen{toks, m}`: the tokens (the id and logit of each) and the model handed back |
| `Gen.then`, `Gen.toks`, `ids_of` | continue with the tokens and the model; the tokens alone; their ids |
| `tokens(c, m, pk, chat, ids, n) -> Toks<c>` | the same loop one token at a time. `Tok{am, cs, m, rest}` is a new token, the candidates of a draw, the model, and the rest of the generation, which runs when it is given the model back (`rest(m)`); `End{m}` is the end. A program that shows each token as it comes walks this value, as llama.bend's `demo/gen.bend` does A prompt of two ids or more runs in one pass (`prompt.bend`) |
| `tokens.seq(c, m, pk, chat, ids, n) -> Toks<c>` | `tokens` with the prompt position by position (`main.bend`'s `forward_pre`), the path `tokens` itself takes for a prompt of one id; the same tokens and logit bits |
| `take(n, c, s) -> Gen<c>` | the first `n` tokens of a `Toks`, with no forward pass after the `n`-th |
| `left_after`, `left_chat` | the stop rule: the tokens left to generate after a token |
| `Chat<c, cap>`, `Chat.start(c, cap)` | a conversation between two answers: the caches of the positions it has run (one per layer, with room for `cap`), and the last answer's last token when it has not gone through the model yet. `Chat.start` has no turn and empty caches |
| `turn(c, pk, cap, chat, m, ids, n) -> Turn<c, cap>` | the conversation's next answer, `n` tokens at most, one at a time. `ids` continue the conversation: `tok.bend`'s `chat` for the first turn, `chat_next` after an answer. They go through each layer in one pass (`prompt.bend`) on top of the caches, so a turn never runs the conversation before it again; then each new token runs at its position, until `<|im_end|>` (id 2), the end-of-text token (id 0), the count, or the end of the room. A `Draw` counts its draws from the answer's first token, so an answer is the tokens that `tokens` gives for the whole conversation up to it |
| `Turn`: `Said{am, cs, m, rest}`, `Over{why, m, room, chat}` | an answer as `Toks` hands tokens over: `Said` is a new token and the rest of the answer (`rest(m)`); `Over` is its end, with the positions the caches still have room for and the conversation for the next turn |
| `Why`: `Stop{}`, `Count{}`, `Full{}`, `Unfit{need}`, `Idle{}` | why an answer ended: a stop token; the count; the caches are full; and two cases in which nothing ran and the conversation comes back as it was: the turn's `need` ids are more than the room, or the turn had no id or asked for no token |
| `Fit<k, room>`, `fit(k, room)`, `room_inv`, `room_le` | how a turn gets its proof of room: `fit` counts the turn's `k` ids against the positions left and answers `Short{}`, or `Fits{rest, e}` with the equation `k + rest = room`; with the conversation's `t + room = cap`, `room_inv` gives `(t + k) + rest = cap` and `room_le` gives `t + k <= cap`, which is what `forward_prompt` asks for. A turn that does not fit cannot be written into the caches: there is no proof to pass |

`prompt.bend`:

| Name | What |
|---|---|
| `forward_prompt(c, cap, t0, n, tT, e, le, m, cs, ids) -> OutP<c, cap, tT>` | the `n >= 1` tokens `ids` at positions `t0 .. t0 + n - 1`, with the caches of the positions before them (`cs`, one per layer, at `KV<cap, t0>`): the logits of the last position, the caches at `KV<cap, tT>` and the model. `e` is the proof that `t0 + n = tT` and `le` that `tT <= cap`, so the caches are never written past their room. `OutP.then` continues with the three fields |
| `block_p(c, d, lw, cap, t0, n, tT, e, le, kv, x) -> BP<c, cap, tT>` | one layer on the `n` positions whose inputs are the rows of `x`: `block_gen` with each of its seven products done once for every position |
| `mm(r, c, b, n, x) -> BM<r, c>` | `W` (in bands) times the `n` rows of `x`: tensor-array's `gemm_t8` with the weights as its first operand, band by band in parallel; `vec(outs, i, n)` reads position `i`'s vector out of the result |

`gen.bend` takes the model as a parameter, so a program chooses its loader (`main.bend`'s `load_model`, or
`load.bend`'s), and none of these files prints or reads a clock. They hold no law of their own yet; `bend <file> --verdict`
accepts each (`ALL PROOFS CHECK`: every def checked by the kernel, no `@unsafe`, no foreign code).

MIT · Renan Vinícius (@nuxyel)
