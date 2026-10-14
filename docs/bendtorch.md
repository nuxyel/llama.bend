# From inference to training: what a backward pass needs

Milestone 11 of the plan: for every operation of the Llama forward pass in `llama/main.bend`, what its backward pass
needs, whether the forward pass already keeps it, and what Bend's rules make hard. A document, no code. It is the
bridge between this package and a training framework on bend-ml (the "BendTorch" step of my plans: typed tensor
autograd and a tiny transformer trained in Bend against a PyTorch twin).

## What exists to build on

- **bend-ml-autograd**: scalar reverse mode over F32 expressions (`G`, `gbwd`), and the law `reverse_eq_forward`: on
  expressions built from constants, x, sums and products, the reverse-mode derivative equals the forward-mode one.
  It also has `linear_bwd` for a typed dense layer.
- **bend-ml-tensor-array**: `Mat.matmul_tn` (Aᵀ·B), `Mat.matmul_nt` (A·Bᵀ) and `Mat.relu_bwd`. MNIST trains with
  these hand-written backward passes and matches PyTorch's loss.
- **llama.bend**: every layer returns an activations record, `Act`, with every intermediate of one position:

  | field | what it is | numbers |
  |---|---|---|
  | `n1` | the first RMSNorm's output | dim |
  | `q`, `k`, `v` | the projections, before RoPE | dim, kvw, kvw |
  | `qr`, `kr` | the query and key heads after RoPE | dim, kvw |
  | `attn` | attention's output, before the output projection | dim |
  | `a` | the output projection | dim |
  | `n2` | the second RMSNorm's output | dim |
  | `m` | the MLP's output (after `down`) | dim |
  | `y` | the layer's output | dim |

  That is 8·dim + 3·kvw numbers per position and layer: 5,184 for SmolLM2-135M (dim 576, kvw 192), 155,520 per
  position over its 30 layers. The tests read `Act` today; a backward pass would read it tomorrow.

## Operation by operation

Notation for one position: x the input, y the output, dy the gradient arriving from above.

| Operation (code) | Backward | Needs | In `Act`? |
|---|---|---|---|
| Embedding row, `Bands.read_row` | dEmb[tok] += dy | the token id | the caller has it |
| RMSNorm, `rmsnorm` | with x̂ = x / rms: dx = (g·dy − x̂·mean(g·dy·x̂)) / rms; dg += dy·x̂ | the input x and rms(x) | no: `n1`/`n2` are outputs. The first norm's input is the previous layer's `y` (or the embedding row); the second's is x + `a`. rms is recomputed in O(dim) |
| Linear, `mv` over `Bands.matvec_l` | dx = Wᵀ·dy; dW += dy ⊗ x | the input x | `n1` for q, k, v; `attn` for o; `n2` for gate and up; **not** the down projection's input, silu(g)·u |
| RoPE, `rope` | the inverse rotation of dy (angle −θ) | the angles of position t | recomputed from t (`rot_at`); nothing to keep |
| Attention scores, `scores` (a `gemm`) | dq = scale·ds·K; dK += scale·dsᵀ·q | q (after RoPE) and the keys of positions 0..t | `qr`; the keys live in the cache arrays |
| Softmax | ds = p ⊙ (dp − ⟨dp, p⟩) | the probabilities p | **no**: only the weighted sum (`attn`) is kept |
| Weighted sum, `weighted` (a `gemm`) | dp = do·Vᵀ; dV += pᵀ·do | p and the values of positions 0..t | the values live in the cache arrays |
| SwiGLU, `swiglu` | du = dz ⊙ silu(g); dg = dz ⊙ u ⊙ silu′(g), with silu′(x) = σ(x)(1 + x(1 − σ(x))) | g and u | **no**: recomputable from `n2` with two products, or kept at 2·ffn numbers more per position |
| Residual adds | dy flows to both branches unchanged | nothing | n/a |
| Final norm and LM head (the embedding, tied) | dnf = Embᵀ·dlogits; dEmb += dlogits ⊗ nf | nf | `Out.nf` |
| Cross-entropy (not in llama.bend) | dlogits = softmax(logits) − onehot(target) | the logits | `Out.lg` |

What `Act` lacks for a full backward pass: the softmax probabilities per head, the SwiGLU's g and u, and the input of
the down projection. Each can be recomputed from what is kept, at the cost of a forward step, or stored, at the cost
of memory. That is a decision to measure, not to guess.

## What Bend's rules make hard

These come from building bend-ml and llama.bend, not from theory.

1. **An `Array` has one owner.** Every product already hands its matrix back (`mv` returns the `Bands` with the
   result). A backward pass makes this heavier: each step reads the weights (for Wᵀ·dy) and writes their gradient
   accumulator (for dW += dy ⊗ x), so both have to be threaded through every call and returned. The typed pair from
   my notes, a layer as `forward : P, X -> Y & saved` and `backward : P, saved, dY -> dX & dP`, makes that threading
   explicit in the types; whether the affine rules allow it without copies is the first thing to prototype.
2. **`Array` cannot sit inside a `Data` value.** `Act` is `Data` (made of lists) so a test can read it twice; the
   cache (`KV`) is a `Type` because it holds arrays. Activations kept as lists are cheap to reuse and slow to compute
   with; kept as arrays they must be threaded once, like the cache. Attention's backward needs the keys and values of
   every earlier position, which are in the cache, so the backward pass takes the cache and hands it back too.
3. **No transposed product over row bands yet.** `Bands` stores a matrix as row bands so that W·x runs in parallel
   without copying W (bend-ml exp. 11). Wᵀ·dy needs each band's columns: every band computes its slice's contribution
   and the results are summed up the tree, which is a new operation with its own law (the sum covers every row, as
   `half_cover` does for the forward). dW = dy ⊗ x splits along the same bands naturally.
4. **No read-only sharing across a parallel let.** Every band gets its own copy of x today (bendlang/bend#1376, open).
   A backward pass shares two vectors per product (x and dy), so the copies double.
5. **Gradients into earlier positions.** In training, position t's attention sends gradient to the keys and values of
   every position p ≤ t. That needs a gradient cache shaped like the cache, written at earlier rows; per key/value head
   it is a plain array, so the heads stay independent and can run in parallel as they do now.
6. **F32 is not a real number.** As in the forward pass, numbers are tested, not proved: gradients against PyTorch's
   autograd on random shapes, as bend-ml does for its scalar autograd.

## Laws worth stating

- **Gradient shapes**: the gradient of every parameter has that parameter's type (`LW<c>`), so a transposed gradient
  does not compile. This comes from the types, with no proof to write.
- **Adjoint law** for the linear core: on a `Nat` model of matrices as lists, dot(g, M·v) = dot(Mᵀ·g, v), so the
  backward of a product is the adjoint of its forward. It generalizes `reverse_eq_forward` from scalars to matrices.
- **Causal gradients**: the gradient that reaches position p comes only from positions t ≥ p, the mirror of the
  forward law `causal`.
- **The transposed band product covers every row**, as the forward band product does.

## An order of work

1. One position, no attention: RMSNorm, the linear layers and SwiGLU backward, tested against `torch.autograd` on
   random inputs (the pattern of bend-ml's `test_autograd_random.py`).
2. The transposed band product, with its covering law.
3. Attention backward over a whole sequence, with the gradient cache; the causal-gradient law.
4. A full layer, then the model, with cross-entropy.
5. A tiny character-level Llama trained in Bend against a PyTorch twin with the same initial weights and batches, the
   way MNIST is checked today: the same loss, step by step, within F32 noise.
