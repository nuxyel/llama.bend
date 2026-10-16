# bend-ml-llama

Llama-architecture language models in Bend 2: RMSNorm, rotary position embeddings, grouped-query attention with a
key/value cache, and SwiGLU. The sizes of a model come from a config (`Cfg`), and **every weight matrix has its
shape in its type**, computed from that config, so a model of other sizes is the same code with another config:
SmolLM2-135M, SmolLM2-360M and SmolLM2-1.7B (plain multi-head attention, another RoPE base) all run on it. The
structure is proved by Bend's checker and again by its kernel proved in Lean. The numbers are tested against
PyTorch: all three models give the same greedy tokens as `transformers` in the
[llama.bend](https://github.com/nuxyel/llama.bend) repository.

- Bend: **2.0.37** · License: MIT · Depends on `bend-ml-tensor@0.1.2.0` and `bend-ml-tensor-array@0.1.7.0`.
- `bend llama/main.bend --verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`, no axioms).

On BendHub as `bend-ml-llama@0.1.0.0`. A local copy works the same way (`import ./llama/main.bend as M`, as the
programs in `demo/` do with `../llama/main.bend`).

```python
import Base
import bend-ml-tensor-array@0.1.7.0/main.bend as TA
import bend-ml-llama@0.1.0.0/main.bend as M

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
| `smollm2_135m_sizes`, `smollm2_360m_sizes`, `smollm2_1_7b_sizes` | The configs give the sizes of their `config.json`: SmolLM2-135M has 9 query heads of 64 numbers, a hidden size of 576 and keys 192 wide; SmolLM2-360M has 15 heads of 64, a hidden size of 960 and keys 320 wide; SmolLM2-1.7B has 32 heads of 64, a hidden size of 2048 and keys 2048 wide (one query head per key/value head). |

`add_zero`, `add_succ`, `add_one`, `add_assoc`, `div_block`, `div_blocks`, `div_rest`, `leb_trans` and `leb_mul` are the
arithmetic lemmas that the proofs above use.

What is **not** proved: the numbers. `F32` is not a real number. RMSNorm, RoPE, softmax and silu follow the order
of operations of `transformers`' `modeling_llama.py`, and the tests check the results: logits within 8.3e-5 of
PyTorch (relative to the largest logit, over 11 prompts and the three models), the same greedy tokens. RMSNorm sums
its squares as a balanced tree, which keeps it within ~1e-7 of float64 next to the large activations of deep
layers. The vectors between layers are lists whose length is not in their type.

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
| `LW<c>`, `Model<c>` | a layer's weights and the whole model; each matrix is a `bend-ml-tensor-array` `Bands<rows, cols>` with its shape from `c`, and the model keeps its parallel depth `d` (`depth_for` of the thread count) |
| `KV<cap, t>`, `KV.empty(c, cap)`, `KV.push`, `caches(c, n, cap)` | the key/value cache of `t` positions, in arrays with room for `cap` positions; a push needs a proof that `t < cap` |
| `step_room`, `step_inv` | for a loop with `t + steps left = cap`: the proof that `t < cap`, and the same equation one step later |
| `block(c, d, lw, cap, t, kv, room, rot, x) -> BR<c, cap, t>` | one layer at position `t` (`d = 0`: attention's key/value heads one after another; otherwise in parallel): the output, the cache grown to `t + 1` positions, the weights handed back, and the activations (`Act`) |
| `forward(c, cap, t, m, cs, room, tok) -> Out<c, cap, t>` | one position of the whole model: logits, the final norm, every layer's `Act`, the caches, the model |
| `forward_gen(c, cap, t, m, cs, room, tok)` | `forward` for generation: the same logits, caches and model, with every layer's activations left empty when the weights are in bands (more than two threads), about 20% faster per token there; `block_gen` is `block` without them |
| `BR.then`, `Out.then` | continue with the fields of a result; a loop over positions recurses inside them |
| `argmax`, `AM.i`, `AM.v` | the first index of the largest logit, and that logit |
| `rmsnorm`, `inv_freq`, `rot_at`, `rope`, `swiglu`, `attention` | the pieces of a layer |
| `read_cfg`, `load_model`, `load_lw`, `depth_for` | loading a model written by `prep.py` (`<dir>/w/<name>.bin`, little-endian F32, one file per tensor) |
| `load.bend`: `load_model` | the same model, with each band of weights read straight into its array by `bend-ml-binary-read` (the 1.7B in about 8 s instead of 60-70). It is foreign code (C and JavaScript), so it lives in its own file: `main.bend` does not import it and keeps `--verdict` clean, and `make bits` checks that both loaders give the same outputs |

An `Array` has a single owner, so every product hands its matrix back (`block` returns its `LW`, `forward` its
`Model`). `Act` holds every intermediate of a layer (`n1`, `q`, `k`, `v`, `qr`, `kr`, `attn`, `a`, `n2`, `m`,
`y`). The tests compare them with `transformers`, and a backward pass would read them.

MIT · Renan Vinícius (@nuxyel)
