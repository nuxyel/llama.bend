# bend-ml-llama: Llama-architecture language models (SmolLM2 and the rest of the family) in Bend 2, with the
# structure in the types and laws checked by Bend's kernel.
#
# The sizes of a model live in a config (Cfg), and every weight matrix has its shape in its type, computed from
# the config: a model of other sizes is the same code with another config. The numbers (F32) are tested against
# PyTorch / transformers by the scripts of the llama.bend repository; the structure is proved here:
#
# - a head has 2·half numbers and is split into two halves of half numbers (RoPE rotates each number with the
#   one half a head later), and splitting a head and joining the halves gives it back (head_split);
# - the query heads are cut into kv groups of g heads, and group j reads key/value head j: there are exactly kv
#   groups (groups_len) and an empty cache holds one array of keys and one of values per key/value head
#   (kv_heads, arrays_len), and query head j·g + i reads key/value head (j·g + i) / g = j, transformers'
#   repeat_kv mapping (gqa_head);
# - the key/value cache at position t has the type KV<t>: t positions written in flat arrays, so attention at
#   position t scores exactly the positions 0..t and nothing after (causal, read_len);
# - the shapes of the weights in the types add up to the number of values in the weight files (layout laws).
#
# The order of every F32 operation follows transformers' modeling_llama.py (5.19), so the results match it to a
# few ulps and the argmax exactly (llama.bend's NOTES.md).

import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-tensor-array@0.1.7.0/main.bend as TA

# =====================================================================================================
# Arithmetic lemmas (used by the laws below)
# =====================================================================================================

law add_zero:
  for a: Nat
  {Nat.add(a, 0n) == a : Nat}

def add_zero(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

law add_succ:
  for  a: Nat
  for -b: Nat
  {1n+Nat.add(a, b) == Nat.add(a, 1n+b) : Nat}

def add_succ(a, b):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, b) : {2n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

law add_one:
  for a: Nat
  {Nat.add(a, 1n) == 1n+a : Nat}

def add_one(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_one(p) : {1n+Nat.add(p, 1n) == 1n+_ : Nat}
      {==}

law add_assoc:
  for  a: Nat
  for -b: Nat
  for -c: Nat
  {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}

def add_assoc(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

# =====================================================================================================
# The configuration
# =====================================================================================================

# kv: key/value heads; g: query heads per key/value head; half: half a head (a head has 2·half numbers, so its
# size is even by construction); f: the MLP's inner width; layers; vocab: the vocabulary (the embedding has
# vocab rows, and the LM head is the embedding); eps: RMSNorm's epsilon; theta: RoPE's base.
type Cfg is Data:
  Cfg{kv: Nat, g: Nat, half: Nat, f: Nat, layers: Nat, vocab: Nat, eps: F32, theta: F32}

# a head: 2·half numbers
def head(+half: Nat) -> Nat:
  Nat.add(half, half)

def Cfg.half(+c: Cfg) -> Nat:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      half

def Cfg.kv(+c: Cfg) -> Nat:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      kv

def Cfg.g(+c: Cfg) -> Nat:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      g

def Cfg.ffn(+c: Cfg) -> Nat:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      f

def Cfg.layers(+c: Cfg) -> Nat:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      l

def Cfg.vocab(+c: Cfg) -> Nat:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      v

def Cfg.eps(+c: Cfg) -> F32:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      e

def Cfg.theta(+c: Cfg) -> F32:
  match c:
    case Cfg{kv, g, half, f, l, v, e, t}:
      t

# the size of a head
def Cfg.head(+c: Cfg) -> Nat:
  head(Cfg.half(c))

# the query heads: kv groups of g
def Cfg.heads(+c: Cfg) -> Nat:
  Nat.mul(Cfg.kv(c), Cfg.g(c))

# the hidden size: one head per query head
def Cfg.dim(+c: Cfg) -> Nat:
  Nat.mul(Cfg.heads(c), Cfg.head(c))

# the width of the keys (and of the values): one head per key/value head
def Cfg.kvw(+c: Cfg) -> Nat:
  Nat.mul(Cfg.kv(c), Cfg.head(c))

# SmolLM2-135M (HuggingFaceTB/SmolLM2-135M): 3 key/value heads of 3 query heads each, heads of 64
def smollm2_135m() -> Cfg:
  Cfg{3n, 3n, 32n, 1536n, 30n, 49152n, 0.00001, 100000.0}

# SmolLM2-360M: 5 key/value heads of 3 query heads each, heads of 64
def smollm2_360m() -> Cfg:
  Cfg{5n, 3n, 32n, 2560n, 32n, 49152n, 0.00001, 100000.0}

# SmolLM2-1.7B: 32 key/value heads of one query head each (plain multi-head attention), heads of 64
def smollm2_1_7b() -> Cfg:
  Cfg{32n, 1n, 32n, 8192n, 24n, 49152n, 0.00001, 130000.0}

# LAW (the configs): SmolLM2-135M has 9 query heads of 64, a hidden size of 576 and keys of 192; SmolLM2-360M
# has 15 query heads of 64, a hidden size of 960 and keys of 320; SmolLM2-1.7B has 32 query heads of 64, a hidden
# size of 2048 and keys of 2048 (as many key/value heads as query heads), as their config.json files say.
law smollm2_135m_sizes:
  {Cfg.dim(smollm2_135m()) == 576n : Nat} & {Cfg.heads(smollm2_135m()) == 9n : Nat} & {Cfg.head(smollm2_135m()) == 64n : Nat} & {Cfg.kvw(smollm2_135m()) == 192n : Nat}

def smollm2_135m_sizes():
  ({==}, {==}, {==}, {==})

law smollm2_360m_sizes:
  {Cfg.dim(smollm2_360m()) == 960n : Nat} & {Cfg.heads(smollm2_360m()) == 15n : Nat} & {Cfg.head(smollm2_360m()) == 64n : Nat} & {Cfg.kvw(smollm2_360m()) == 320n : Nat}

def smollm2_360m_sizes():
  ({==}, {==}, {==}, {==})

law smollm2_1_7b_sizes:
  {Cfg.dim(smollm2_1_7b()) == 2048n : Nat} & {Cfg.heads(smollm2_1_7b()) == 32n : Nat} & {Cfg.head(smollm2_1_7b()) == 64n : Nat} & {Cfg.kvw(smollm2_1_7b()) == 2048n : Nat}

def smollm2_1_7b_sizes():
  ({==}, {==}, {==}, {==})

# ---- the layout: the shapes in the types add up to the number of values in the checkpoint ----

def n32(+n: Nat) -> U32:
  U32.from_nat(n)

# the numbers of one layer, from the shapes of LW<c>: two norms, q and o (dim × dim), k and v (kvw × dim),
# gate and up (ffn × dim) and down (dim × ffn)
def Cfg.layer_size(+c: Cfg) -> U32:
  ((2 : U32) * n32(Cfg.dim(c)) + (2 : U32) * n32(Cfg.dim(c)) * n32(Cfg.dim(c)) + (2 : U32) * n32(Cfg.kvw(c)) * n32(Cfg.dim(c)) + (3 : U32) * n32(Cfg.ffn(c)) * n32(Cfg.dim(c)) : U32)

# the numbers of the model, from the shapes of Model<c>: the embedding (vocab × dim), the layers, the final norm
def Cfg.size(+c: Cfg) -> U32:
  (n32(Cfg.vocab(c)) * n32(Cfg.dim(c)) + n32(Cfg.layers(c)) * Cfg.layer_size(c) + n32(Cfg.dim(c)) : U32)

# LAW (layout): the shapes the types give to SmolLM2-135M's weights add up to 134,515,008 numbers,
# SmolLM2-360M's to 361,821,120 and SmolLM2-1.7B's to 1,711,376,384: the number of values in each model.safetensors (reference/prep.py counts them).
# The loader checks every file's size against the shape in its type, so together the files cover the checkpoint
# exactly. In U32, whose sums wrap at 2^32; every partial sum here is below it, so this is the exact count.
law layout_135m:
  {Cfg.size(smollm2_135m()) == (134515008 : U32) : U32}

def layout_135m():
  {==}

law layout_360m:
  {Cfg.size(smollm2_360m()) == (361821120 : U32) : U32}

def layout_360m():
  {==}

law layout_1_7b:
  {Cfg.size(smollm2_1_7b()) == (1711376384 : U32) : U32}

def layout_1_7b():
  {==}

# =====================================================================================================
# Heads
# =====================================================================================================

# a head as its two halves: RoPE rotates number i of the first half with number i of the second
type Head is Data:
  Head{lo: List<&2, F32>, hi: List<&2, F32>}

def split(+half: Nat, +xs: List<&2, F32>) -> Head:
  Head{List.take(&2, F32, xs, half), List.drop(&2, F32, xs, half)}

def join(h: Head) -> List<&2, F32>:
  match h:
    case Head{lo, hi}:
      List.append(&2, F32, lo, hi)

# n heads of 2·half numbers from xs, in order
def heads(n: Nat, +half: Nat, +xs: List<&2, F32>) -> List<&2, Head>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      split(half, List.take(&2, F32, xs, head(half))) <> heads(p, half, List.drop(&2, F32, xs, head(half)))

# n plain heads (the values, which RoPE does not touch)
def chunks(n: Nat, +w: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, w) <> chunks(p, w, List.drop(&2, F32, xs, w))

def joins(hs: List<&2, Head>) -> List<&2, List<&2, F32>>:
  match hs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      join(h) <> joins(t)

def flat(hs: List<&2, Head>) -> List<&2, F32>:
  List.concat(&2, F32, joins(hs))

# LAW (head_split): splitting a list at half and joining the two parts gives the list back, so cutting a head
# into the halves that RoPE pairs up loses and reorders nothing.
law head_split:
  for +xs: List<&2, F32>
  for +half: Nat
  {join(split(half, xs)) == xs : List<&2, F32>}

# PROOF: by induction on the list and half together: an empty list stays empty; at half 0 the first part is
# empty and the second is the whole list; otherwise the first number is kept in front and the rest is the
# induction hypothesis.
def head_split(xs, half):
  match xs half:
    case Nil{} _:
      {==}
    case Con{h, t} 0n:
      {==}
    case Con{h, t} 1n+p:
      %head_split(t, p) : {h <> List.append(&2, F32, List.take(&2, F32, t, p), List.drop(&2, F32, t, p)) == h <> _ : List<&2, F32>}
      {==}

# =====================================================================================================
# The numeric pieces (tested against transformers, never proved: F32 is not a real number)
# =====================================================================================================

# the squares of xs
def squares(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      (h * h : F32) <> squares(t)

# adjacent pairs added: [a, b, c, d, e] -> [a + b, c + d, e]
def pair_sums(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Con{+a, Con{+b, t}}:
      (a + b : F32) <> pair_sums(t)
    case _:
      xs

# the sum as a balanced tree of additions (pairwise summation): its rounding error grows with log2(n), not n, as
# in PyTorch's cascade sum. A running sum loses the small squares next to a large one (NOTES, 2026-10-10).
def tree_sum(fuel: Nat, xs: List<&2, F32>) -> F32:
  match fuel xs:
    case _ Nil{}:
      0.0
    case _ Con{+a, Nil{}}:
      a
    case 0n Con{+a, t}:
      T.sum.go(t, a)
    case 1n+p Con{+a, t}:
      tree_sum(p, pair_sums(Con{a, t}))

# RMSNorm: g * (x * (1 / sqrt(mean(x²) + eps))), in this order (LlamaRMSNorm.forward)
def rmsnorm(+c: Cfg, +x: List<&2, F32>, +g: List<&2, F32>) -> List<&2, F32>:
  T.zip_with.mul(g, T.scale.l((1.0 / F32.sqrt(((tree_sum(Cfg.dim(c), squares(x)) / F32.from_nat(Cfg.dim(c)) : F32) + Cfg.eps(c) : F32)) : F32), x))

# inv_freq[i] = 1 / (theta ^ (2i / head)), i = 0..half-1 (LlamaRotaryEmbedding, default rope)
def inv_freq.go(n: Nat, +i: Nat, +theta: F32, +hd: F32) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      (1.0 / F32.pow(theta, (F32.from_nat(Nat.mul(2n, i)) / hd : F32)) : F32) <> inv_freq.go(p, 1n+i, theta, hd)

def inv_freq(+c: Cfg) -> List<&2, F32>:
  inv_freq.go(Cfg.half(c), 0n, Cfg.theta(c), F32.from_nat(Cfg.head(c)))

def angles(+t: F32, fs: List<&2, F32>) -> List<&2, F32>:
  match fs:
    case Nil{}:
      Nil{}
    case Con{+f, r}:
      (t * f : F32) <> angles(t, r)

def cos.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, r}:
      F32.cos(x) <> cos.l(r)

def sin.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, r}:
      F32.sin(x) <> sin.l(r)

# the cosines and sines of position t, one per frequency (shared by every head of every layer)
type Rot is Data:
  Rot{cs: List<&2, F32>, sn: List<&2, F32>}

def rot_at(+c: Cfg, +t: Nat) -> Rot:
  Rot{cos.l(angles(F32.from_nat(t), inv_freq(c))), sin.l(angles(F32.from_nat(t), inv_freq(c)))}

# first half: lo*cos + (-hi)*sin; second half: hi*cos + lo*sin (q*cos + rotate_half(q)*sin)
def rot_lo(h1: List<&2, F32>, h2: List<&2, F32>, cs: List<&2, F32>, sn: List<&2, F32>) -> List<&2, F32>:
  match h1 h2 cs sn:
    case Con{+a, ta} Con{+b, tb} Con{+c, tc} Con{+s, ts}:
      ((a * c : F32) + (F32.neg(b) * s : F32) : F32) <> rot_lo(ta, tb, tc, ts)
    case _ _ _ _:
      Nil{}

def rot_hi(h1: List<&2, F32>, h2: List<&2, F32>, cs: List<&2, F32>, sn: List<&2, F32>) -> List<&2, F32>:
  match h1 h2 cs sn:
    case Con{+a, ta} Con{+b, tb} Con{+c, tc} Con{+s, ts}:
      ((b * c : F32) + (a * s : F32) : F32) <> rot_hi(ta, tb, tc, ts)
    case _ _ _ _:
      Nil{}

def rope(rt: Rot, h: Head) -> Head:
  match rt h:
    case Rot{+cs, +sn} Head{+lo, +hi}:
      Head{rot_lo(lo, hi, cs, sn), rot_hi(lo, hi, cs, sn)}

def ropes(+rt: Rot, hs: List<&2, Head>) -> List<&2, Head>:
  match hs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      rope(rt, h) <> ropes(rt, t)

# silu(x) = x / (1 + exp(-x)), the form ATen's CPU kernel uses
def silu(+x: F32) -> F32:
  (x / (1.0 + F32.exp(F32.neg(x)) : F32) : F32)

# silu(gate) * up
def swiglu(g: List<&2, F32>, u: List<&2, F32>) -> List<&2, F32>:
  match g u:
    case Con{+a, ga} Con{+b, ub}:
      (silu(a) * b : F32) <> swiglu(ga, ub)
    case _ _:
      Nil{}

# =====================================================================================================
# The key/value cache in flat arrays, indexed by the number of positions it holds
# =====================================================================================================

# One pair of arrays per key/value head: the keys (after RoPE) and the values of positions 0..t-1, position p at
# [p·head, p·head + head). The type indexes are the capacity cap (the positions the arrays have room for, set by
# KV.empty) and t, the number of positions written. KV.push asks for a proof that t < cap, so a cache is never
# written past its room (kv_room says why that is enough).
type KV<-cap: Nat, -t: Nat> is Type:
  KV{ks: List<&1, Array<F32>>, vs: List<&1, Array<F32>>}

# n arrays of 2^d slots
def arrays(n: Nat, +d: Nat) -> List<&1, Array<F32>>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      Array.new(F32, d, 0.0) <> arrays(p, d)

# a cache with no position yet, with room for cap positions per key/value head
def KV.empty(+c: Cfg, +cap: Nat) -> KV<cap, 0n>:
  KV{arrays(Cfg.kv(c), TA.cap_depth(Nat.mul(cap, Cfg.head(c)))), arrays(Cfg.kv(c), TA.cap_depth(Nat.mul(cap, Cfg.head(c))))}

# writes row j of rs into array j, at off
def write_rows(arrs: List<&1, Array<F32>>, rs: List<&2, List<&2, F32>>, +off: U32) -> List<&1, Array<F32>>:
  match arrs rs:
    case Con{a, at} Con{r, rt}:
      TA.fill_list(r, off, a) <> write_rows(at, rt, off)
    case _ _:
      arrs

# position t: the keys and values of every key/value head written at t·head; room is the proof that t < cap
def KV.push(+c: Cfg, -cap: Nat, +t: Nat, kv: KV<cap, t>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +ks: List<&2, List<&2, F32>>, +vs: List<&2, List<&2, F32>>) -> KV<cap, 1n+t>:
  match kv:
    case KV{ka, va}:
      KV{write_rows(ka, ks, U32.from_nat(Nat.mul(t, Cfg.head(c)))), write_rows(va, vs, U32.from_nat(Nat.mul(t, Cfg.head(c))))}

# the number of key/value heads a cache holds
def kv_count(-cap: Nat, -t: Nat, kv: KV<cap, t>) -> Nat:
  match kv:
    case KV{ka, va}:
      List.length(&1, Array<F32>, ka)

# LAW (arrays_len): arrays(n, d) makes exactly n arrays.
law arrays_len:
  for +n: Nat
  for +d: Nat
  {List.length(&1, Array<F32>, arrays(n, d)) == n : Nat}

# PROOF: by induction on n: none for 0; one more than the rest for 1 + p.
def arrays_len(n, d):
  match n:
    case 0n:
      {==}
    case 1n+p:
      %arrays_len(p, d) : {1n+List.length(&1, Array<F32>, arrays(p, d)) == 1n+_ : Nat}
      {==}

# LAW (kv_heads): an empty cache holds one array of keys (and one of values) per key/value head, so key/value head j
# of the model reads column j and no other.
law kv_heads:
  for +c: Cfg
  for +cap: Nat
  {kv_count(cap, 0n, KV.empty(c, cap)) == Cfg.kv(c) : Nat}

def kv_heads(c, cap):
  arrays_len(Cfg.kv(c), TA.cap_depth(Nat.mul(cap, Cfg.head(c))))

# a <= b  and  b <= c  ->  a <= c
def leb_trans(a: Nat, b: Nat, c: Nat, e1: {True{} == TA.leb(a, b) : Bool}, e2: {True{} == TA.leb(b, c) : Bool}) -> {True{} == TA.leb(a, c) : Bool}:
  match a b c:
    case 0n _ _:
      {==}
    case 1n+p 0n _:
      %e1 : TA.BD(_, Unit, {True{} == TA.leb(1n+p, c) : Bool})
      Unit{}
    case 1n+p 1n+q 0n:
      %e2 : TA.BD(_, Unit, {True{} == TA.leb(1n+p, 0n) : Bool})
      Unit{}
    case 1n+p 1n+q 1n+r:
      leb_trans(p, q, r, e1, e2)

# a <= b  ->  a·h <= b·h
def leb_mul(a: Nat, b: Nat, +h: Nat, e: {True{} == TA.leb(a, b) : Bool}) -> {True{} == TA.leb(Nat.mul(a, h), Nat.mul(b, h)) : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      %e : TA.BD(_, Unit, {True{} == TA.leb(Nat.mul(1n+p, h), 0n) : Bool})
      Unit{}
    case 1n++p 1n++q:
      TA.leb_add(h, h, Nat.mul(p, h), Nat.mul(q, h), TA.leb_refl(h), leb_mul(p, q, h, e))

# LAW (kv_room): KV.empty(c, cap) gives each key/value head arrays of 2^cap_depth(cap·head) slots, and KV.push at
# position t writes head numbers from t·head. For a position below the capacity (t < cap) they all fall inside the
# slots: (1 + t)·head <= 2^cap_depth(cap·head), so a push never wraps around onto position 0. (What stays trusted is
# tensor-array's: Array.new allocates 2^d slots.) The type does not stop a caller from pushing at t >= cap; the
# callers size cap to the positions they run.
law kv_room:
  for +cap: Nat
  for +h: Nat
  for +t: Nat
  for e: {True{} == TA.leb(1n+t, cap) : Bool}
  {True{} == TA.leb(Nat.mul(1n+t, h), TA.pow2(TA.cap_depth(Nat.mul(cap, h)))) : Bool}

# For callers that walk the positions with a counter of steps left (fuel), keeping t + fuel = cap: with 1 + p steps
# left, position t is below the capacity (step_room), and after the step t + 1 + p = cap still holds (step_inv).
def step_room(+t: Nat, +p: Nat, -cap: Nat, e: {Nat.add(t, 1n+p) == cap : Nat}) -> {True{} == TA.leb(1n+t, cap) : Bool}:
  %e : {True{} == TA.leb(1n+t, _) : Bool}
  %add_succ(t, p) : {True{} == TA.leb(1n+t, _) : Bool}
  TA.leb_up(t, t, p, TA.leb_refl(t))

def step_inv(+t: Nat, -p: Nat, -cap: Nat, e: {Nat.add(t, 1n+p) == cap : Nat}) -> {Nat.add(1n+t, p) == cap : Nat}:
  %e : {1n+Nat.add(t, p) == _ : Nat}
  add_succ(t, p)

# PROOF: (1 + t)·head <= cap·head by leb_mul, and cap·head <= 2^cap_depth(cap·head) by tensor-array's cap_ok; then
# leb_trans.
def kv_room(cap, h, t, e):
  leb_trans(Nat.mul(1n+t, h), Nat.mul(cap, h), TA.pow2(TA.cap_depth(Nat.mul(cap, h))), leb_mul(1n+t, cap, h, e), TA.cap_ok(Nat.mul(cap, h)))

# =====================================================================================================
# Grouped-query attention
# =====================================================================================================
# the query heads in n groups of g, in order: group j holds query heads j·g .. j·g + g - 1
def groups(n: Nat, +g: Nat, +hs: List<&2, Head>) -> List<&2, List<&2, Head>>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, Head, hs, g) <> groups(p, g, List.drop(&2, Head, hs, g))

# LAW (groups_len): the query heads make exactly as many groups as there are key/value heads, so attention
# (which walks the groups and the columns together) pairs every group with a key/value head.
law groups_len:
  for n: Nat
  for +g: Nat
  for +hs: List<&2, Head>
  {List.length(&2, List<&2, Head>, groups(n, g, hs)) == n : Nat}

def groups_len(n, g, hs):
  match n:
    case 0n:
      {==}
    case 1n+p:
      %groups_len(p, g, List.drop(&2, Head, hs, g)) : {1n+List.length(&2, List<&2, Head>, groups(p, g, List.drop(&2, Head, hs, g))) == 1n+_ : Nat}
      {==}

# one block of 1 + m numbers in Nat.divmod: the counter m runs down to 0, then the quotient d grows by one
law div_block:
  for +m: Nat
  for -n: Nat
  for -d: Nat
  for -r: Nat
  {Nat.divmod.go(Nat.add(1n+m, n), m, d, r) == Nat.divmod.go(n, Nat.add(m, r), 1n+d, 0n) : Nat & Nat}

def div_block(m, n, d, r):
  match m:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, 1n+Nat.add(p, r), Nat.add(p, 1n+r), add_succ(p, r)) : {Nat.divmod.go(Nat.add(1n+p, n), p, d, 1n+r) == Nat.divmod.go(n, _, 1n+d, 0n) : Nat & Nat}
      div_block(p, n, d, 1n+r)

# j whole blocks of 1 + s numbers add j to the quotient
law div_blocks:
  for +j: Nat
  for +s: Nat
  for +n: Nat
  for +d: Nat
  {Nat.divmod.go(Nat.add(Nat.mul(j, 1n+s), n), s, d, 0n) == Nat.divmod.go(n, s, Nat.add(j, d), 0n) : Nat & Nat}

def div_blocks(j, s, n, d):
  match j:
    case 0n:
      {==}
    case 1n+jp:
      %Equal.sym(Nat, Nat.add(Nat.add(1n+s, Nat.mul(jp, 1n+s)), n), Nat.add(1n+s, Nat.add(Nat.mul(jp, 1n+s), n)), add_assoc(1n+s, Nat.mul(jp, 1n+s), n)) : {Nat.divmod.go(_, s, d, 0n) == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(1n+s, Nat.add(Nat.mul(jp, 1n+s), n)), s, d, 0n), Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), Nat.add(s, 0n), 1n+d, 0n), div_block(s, Nat.add(Nat.mul(jp, 1n+s), n), d, 0n)) : {_ == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, Nat.add(s, 0n), s, add_zero(s)) : {Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), _, 1n+d, 0n) == Nat.divmod.go(n, s, Nat.add(1n+jp, d), 0n) : Nat & Nat}
      %Equal.sym(Nat, 1n+Nat.add(jp, d), Nat.add(jp, 1n+d), add_succ(jp, d)) : {Nat.divmod.go(Nat.add(Nat.mul(jp, 1n+s), n), s, 1n+d, 0n) == Nat.divmod.go(n, s, _, 0n) : Nat & Nat}
      div_blocks(jp, s, n, 1n+d)

# fewer than 1 + s numbers after the last whole block leave the quotient alone
law div_rest:
  for +i: Nat
  for -k: Nat
  for -d: Nat
  for -r: Nat
  {Pair.fst(Nat, Nat, Nat.divmod.go(i, Nat.add(i, k), d, r)) == d : Nat}

def div_rest(i, k, d, r):
  match i:
    case 0n:
      {==}
    case 1n+p:
      div_rest(p, k, d, 1n+r)

# LAW (gqa_head): with g = 1 + i + k query heads per key/value head (g > 0 and i < g), query head j·g + i, the
# head i of group j, reads key/value head (j·g + i) / g = j. That division is how transformers maps query heads
# to key/value heads (repeat_kv), so grouping the heads computes the same attention.
law gqa_head:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {Nat.div(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}

# PROOF: Nat.div counts how many whole blocks of g numbers fit. j·g is j whole blocks (div_blocks, which uses
# div_block for each block), and i < g numbers after them do not complete another one (div_rest).
def gqa_head(j, i, k):
  %Equal.sym(Nat & Nat, Nat.divmod.go(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), Nat.add(i, k), 0n, 0n), Nat.divmod.go(i, Nat.add(i, k), Nat.add(j, 0n), 0n), div_blocks(j, Nat.add(i, k), i, 0n)) : {Pair.fst(Nat, Nat, _) == j : Nat}
  %add_zero(j) : {Pair.fst(Nat, Nat, Nat.divmod.go(i, Nat.add(i, k), Nat.add(j, 0n), 0n)) == _ : Nat}
  div_rest(i, k, Nat.add(j, 0n), 0n)

# =====================================================================================================
# Grouped-query attention over the arrays
# =====================================================================================================

# the scores of one query head against the keys of its key/value head, and the keys handed back
type SR is Type:
  SR{k: Array<F32>, s: List<&2, F32>}

def sr_len(sr: SR) -> Nat:
  match sr:
    case SR{k, s}:
      List.length(&2, F32, s)

def sr_of(k: Array<F32>, al: TA.AL) -> SR:
  match al:
    case TA.AL{c, l}:
      SR{k, l}

# LAW (read_len): reading m + 1 numbers of an array onto acc gives m + 1 more scores.
# PROOF: by induction on m. With 0 left it reads one number and reverses the list (rev_len counts it); with 1 + p
# left it reads one and recurses, and len_step moves the 1 out of the sum (tensor-array's leaf_len, for scores).
law read_len:
  for k: Array<F32>
  for +m: Nat
  for +idx: U32
  for +acc: List<&2, F32>
  for rc: Array<F32> & F32
  {sr_len(sr_of(k, TA.read_l(m, idx, acc, rc))) == Nat.add(1n+m, List.length(&2, F32, acc)) : Nat}

def read_len(k, m, idx, acc, rc):
  match m rc:
    case 0n Tuple{d, +v}:
      TA.zero_out(List.length(&2, F32, List.reverse.go(&2, F32, v <> acc, Nil{})), 1n+List.length(&2, F32, acc), TA.rev_len(v <> acc, Nil{}))
    case 1n+p Tuple{d, +v}:
      TA.len_step(sr_len(sr_of(k, TA.read_l(p, (idx + 1 : U32), v <> acc, Array.get(F32, d, (idx + 1 : U32))))), p, List.length(&2, F32, acc), read_len(k, p, (idx + 1 : U32), v <> acc, Array.get(F32, d, (idx + 1 : U32))))

# read the 1 + t scores the gemm wrote
def sr_read(+t: Nat, g: TA.G) -> SR:
  match g:
    case TA.G{qa, k, s}:
      sr_of(k, TA.read_l(t, 0, Nil{}, Array.get(F32, s, 0)))

# q·k for the keys of positions 0..t, oldest first: K (t + 1 rows of `head` numbers) times q, the product
# Mat.matmul_nt computes for one row (bend-ml-tensor-array's gemm)
def scores(+hd: Nat, +t: Nat, k: Array<F32>, +q: List<&2, F32>) -> SR:
  sr_read(t, TA.gemm(1n, 1n+t, hd, U32.from_nat(hd), 1, 1, U32.from_nat(hd), TA.fill_list(q, 0, Array.new(F32, TA.cap_depth(hd), 0.0)), k, Array.new(F32, TA.cap_depth(1n+t), 0.0)))

def sr_len_g(+t: Nat, g: TA.G) -> {sr_len(sr_read(t, g)) == 1n+t : Nat}:
  match g:
    case TA.G{qa, k, s}:
      TA.zero_out(sr_len(sr_of(k, TA.read_l(t, 0, Nil{}, Array.get(F32, s, 0)))), 1n+t, read_len(k, t, 0, Nil{}, Array.get(F32, s, 0)))

# LAW (causal): attention at position t scores exactly t + 1 keys, the positions 0..t: one score per position
# written, none for a position not yet computed.
# PROOF: the gemm's result has one constructor, so matching it opens it without computing the product
# (sr_len_g); read_len counts the 1 + t numbers read back.
law causal:
  for +hd: Nat
  for +t: Nat
  for k: Array<F32>
  for +q: List<&2, F32>
  {sr_len(scores(hd, t, k, q)) == 1n+t : Nat}

def causal(hd, t, k, q):
  sr_len_g(t, TA.gemm(1n, 1n+t, hd, U32.from_nat(hd), 1, 1, U32.from_nat(hd), TA.fill_list(q, 0, Array.new(F32, TA.cap_depth(hd), 0.0)), k, Array.new(F32, TA.cap_depth(1n+t), 0.0)))

# the weighted sum of the values of positions 0..t: w (t + 1 weights) times V (t + 1 rows of `head` numbers)
type WR is Type:
  WR{v: Array<F32>, o: List<&2, F32>}

def wr_of(v: Array<F32>, al: TA.AL) -> WR:
  match al:
    case TA.AL{c, l}:
      WR{v, l}

def wr_read(+hd: Nat, g: TA.G) -> WR:
  match g:
    case TA.G{wa, v, o}:
      wr_of(v, TA.read_l(Nat.sub(hd, 1n), 0, Nil{}, Array.get(F32, o, 0)))

def weighted(+hd: Nat, +t: Nat, v: Array<F32>, +w: List<&2, F32>) -> WR:
  wr_read(hd, TA.gemm(1n, hd, 1n+t, U32.from_nat(1n+t), 1, U32.from_nat(hd), 1, TA.fill_list(w, 0, Array.new(F32, TA.cap_depth(1n+t), 0.0)), v, Array.new(F32, TA.cap_depth(hd), 0.0)))

# one query head: softmax(q·K · scale) · V, and the two arrays handed back
type HO is Type:
  HO{k: Array<F32>, v: Array<F32>, o: List<&2, F32>}

def HO.then(-R: Type, h: HO, then: Array<F32> -> Array<F32> -> List<&2, F32> -> R) -> R:
  match h:
    case HO{k, v, o}:
      then(k, v, o)

def head_wr(k: Array<F32>, wr: WR) -> HO:
  match wr:
    case WR{v, o}:
      HO{k, v, o}

def head_sr(+hd: Nat, +t: Nat, +scale: F32, v: Array<F32>, sr: SR) -> HO:
  match sr:
    case SR{k, s}:
      head_wr(k, weighted(hd, t, v, T.softmax.l(T.scale.l(scale, s))))

def head_att(+hd: Nat, +t: Nat, +scale: F32, k: Array<F32>, v: Array<F32>, +q: List<&2, F32>) -> HO:
  head_sr(hd, t, scale, v, scores(hd, t, k, q))

# the query heads of one group over its key/value head, in order; the arrays are threaded through
def group_att(hs: List<&2, Head>, +hd: Nat, +t: Nat, +scale: F32, k: Array<F32>, v: Array<F32>, +acc: List<&2, F32>) -> HO:
  match hs:
    case Nil{}:
      HO{k, v, acc}
    case Con{h, rest}:
      HO.then(HO, head_att(hd, t, scale, k, v, join(h)), k2 => v2 => +o =>
        group_att(rest, hd, t, scale, k2, v2, List.append(&2, F32, acc, o)))

# every group, in order: the outputs of query heads 0, 1, ..., and the caches handed back
type AT is Type:
  AT{ks: List<&1, Array<F32>>, vs: List<&1, Array<F32>>, o: List<&2, F32>}

def at_join(p: AT, q: AT) -> AT:
  match p q:
    case AT{k1, v1, o1} AT{k2, v2, o2}:
      AT{List.append(&1, Array<F32>, k1, k2), List.append(&1, Array<F32>, v1, v2), List.append(&2, F32, o1, o2)}

# one key/value head's work: its group of query heads, its keys and its values
type GW is Type:
  GW{g: List<&2, Head>, k: Array<F32>, v: Array<F32>}

def gws(gs: List<&2, List<&2, Head>>, ks: List<&1, Array<F32>>, vs: List<&1, Array<F32>>) -> List<&1, GW>:
  match gs ks vs:
    case Con{g, gt} Con{k, kt} Con{v, vt}:
      GW{g, k, v} <> gws(gt, kt, vt)
    case _ _ _:
      Nil{}

# some groups one after another, in order
def seq_att(ws: List<&1, GW>, +hd: Nat, +t: Nat, +scale: F32, aks: List<&1, Array<F32>>, avs: List<&1, Array<F32>>, +acc: List<&2, F32>) -> AT:
  match ws:
    case Con{w, wt}:
      match w:
        case GW{g, k, v}:
          HO.then(AT, group_att(g, hd, t, scale, k, v, Nil{}), k2 => v2 => +o =>
            seq_att(wt, hd, t, scale, k2 <> aks, v2 <> avs, List.append(&2, F32, acc, o)))
    case Nil{}:
      AT{List.reverse(&1, Array<F32>, aks), List.reverse(&1, Array<F32>, avs), acc}

# a list of groups cut after its first n (a tail loop: a helper in the fork tree must not hold the stack)
type GS is Type:
  GS{l: List<&1, GW>, r: List<&1, GW>}

def cut(n: Nat, xs: List<&1, GW>, acc: List<&1, GW>) -> GS:
  match n xs:
    case 1n+m Con{x, xt}:
      cut(m, xt, x <> acc)
    case _ _:
      GS{List.reverse(&1, GW, acc), xs}

# how many times n groups are halved (the larger half kept) until one or none is left
def halvings(fuel: Nat, +n: Nat) -> Nat:
  match fuel:
    case 0n:
      0n
    case 1n+f:
      match n:
        case 0n:
          0n
        case 1n+p:
          match p:
            case 0n:
              0n
            case 1n+q:
              1n+halvings(f, Nat.sub(2n+q, TA.half(2n+q)))

# the n groups of gs (cut in two halves) in a tree of parallel lets of depth d, each leaf a half of a node; the
# outputs and the caches come back in group order, so every head computes what it computed one after another
def par_att(d: Nat, gs: GS, +n: Nat, +hd: Nat, +t: Nat, +scale: F32) -> AT:
  match d gs:
    case 0n GS{l, r}:
      p q = seq_att(l, hd, t, scale, Nil{}, Nil{}, Nil{}) seq_att(r, hd, t, scale, Nil{}, Nil{}, Nil{})
      at_join(p, q)
    case 1n++e GS{l, r}:
      p q = par_att(e, cut(TA.half(TA.half(n)), l, Nil{}), TA.half(n), hd, t, scale) par_att(e, cut(TA.half(Nat.sub(n, TA.half(n))), r, Nil{}), Nat.sub(n, TA.half(n)), hd, t, scale)
      at_join(p, q)

# the attention output and the cache handed back
type KA<-cap: Nat, -t: Nat> is Type:
  KA{kv: KV<cap, t>, o: List<&2, F32>}

def ka_of(-cap: Nat, -t: Nat, at: AT) -> KA<cap, t>:
  match at:
    case AT{ks, vs, o}:
      KA{KV{ks, vs}, o}

def KA.then(-R: Type, -cap: Nat, -t: Nat, ka: KA<cap, t>, then: KV<cap, t> -> List<&2, F32> -> R) -> R:
  match ka:
    case KA{kv, o}:
      then(kv, o)

# attention of the (rotated) query heads at position t over the cache of positions 0..t (the newest included).
# d is the model's parallel depth (depth_for): with d = 0 (two threads or fewer) the key/value heads run one after
# another, since a fork tree on one thread is only overhead (+5% on the 1.7B, NOTES 2026-10-12); otherwise they run
# in parallel, one group per leaf. Both orders compute every head the same way, so the bits are the same.
def attention(+c: Cfg, +d: Nat, -cap: Nat, +t: Nat, qs: List<&2, Head>, kv: KV<cap, 1n+t>) -> KA<cap, 1n+t>:
  match d kv:
    case 0n KV{ks, vs}:
      ka_of(cap, 1n+t, seq_att(gws(groups(Cfg.kv(c), Cfg.g(c), qs), ks, vs), Cfg.head(c), t, (1.0 / F32.sqrt(F32.from_nat(Cfg.head(c))) : F32), Nil{}, Nil{}, Nil{}))
    case 1n+e KV{ks, vs}:
      ka_of(cap, 1n+t, par_att(Nat.sub(halvings(Cfg.kv(c), Cfg.kv(c)), 1n), cut(TA.half(Cfg.kv(c)), gws(groups(Cfg.kv(c), Cfg.g(c), qs), ks, vs), Nil{}), Cfg.kv(c), Cfg.head(c), t, (1.0 / F32.sqrt(F32.from_nat(Cfg.head(c))) : F32)))

# =====================================================================================================
# A layer
# =====================================================================================================

# the weights of one layer, with their shapes from the config (row = output, as PyTorch stores them)
type LW<-c: Cfg> is Type:
  LW{ln1: List<&2, F32>, q: TA.Bands<Cfg.dim(c), Cfg.dim(c)>, k: TA.Bands<Cfg.kvw(c), Cfg.dim(c)>, v: TA.Bands<Cfg.kvw(c), Cfg.dim(c)>, o: TA.Bands<Cfg.dim(c), Cfg.dim(c)>, ln2: List<&2, F32>, gate: TA.Bands<Cfg.ffn(c), Cfg.dim(c)>, up: TA.Bands<Cfg.ffn(c), Cfg.dim(c)>, down: TA.Bands<Cfg.dim(c), Cfg.ffn(c)>}

# what a layer computes at one position, every intermediate: the tests compare them with transformers, and a
# backward pass would read them
type Act is Data:
  Act{n1: List<&2, F32>, q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>, qr: List<&2, F32>, kr: List<&2, F32>, attn: List<&2, F32>, a: List<&2, F32>, n2: List<&2, F32>, m: List<&2, F32>, y: List<&2, F32>}

# the layer's output at position t, its cache grown to t + 1 positions, its weights handed back, and the
# activations
type BR<-c: Cfg, -cap: Nat, -t: Nat> is Type:
  BR{y: List<&2, F32>, kv: KV<cap, 1n+t>, lw: LW<c>, act: Act}

# continues with the fields of a layer's result (a computed value cannot be matched; a loop over positions
# recurses inside k)
def BR.then(-R: Type, -c: Cfg, -cap: Nat, -t: Nat, b: BR<c, cap, t>, k: List<&2, F32> -> KV<cap, 1n+t> -> LW<c> -> Act -> R) -> R:
  match b:
    case BR{y, kv, lw, act}:
      k(y, kv, lw, act)

def mv.go(-B: Type, -r: Nat, -k: Nat, bl: TA.BL<r, k>, then: TA.Bands<r, k> -> List<&2, F32> -> B) -> B:
  match bl:
    case TA.BL{w, y}:
      then(w, y)

# y = W · x, then `then` with W handed back and y (an Array has one owner, so a product returns its matrix)
def mv(-B: Type, +r: Nat, +k: Nat, w: TA.Bands<r, k>, +x: List<&2, F32>, then: TA.Bands<r, k> -> List<&2, F32> -> B) -> B:
  mv.go(B, r, k, TA.Bands.matvec_l(r, k, w, x), then)

# the layer at position t (0-based) on input x, with the cache of positions 0..t-1 and the rotation of t; d is the
# parallel depth of attention (see attention)
def block(+c: Cfg, +d: Nat, lw: LW<c>, -cap: Nat, +t: Nat, kv: KV<cap, t>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +rt: Rot, +x: List<&2, F32>) -> BR<c, cap, t>:
  match lw:
    case LW{+ln1, wq, wk, wv, wo, +ln2, wg, wu, wd}:
      +n1 = rmsnorm(c, x, ln1)
      mv(BR<c, cap, t>, Cfg.dim(c), Cfg.dim(c), wq, n1, wq => +q =>
        mv(BR<c, cap, t>, Cfg.kvw(c), Cfg.dim(c), wk, n1, wk => +k =>
          mv(BR<c, cap, t>, Cfg.kvw(c), Cfg.dim(c), wv, n1, wv => +v =>
            +qs = ropes(rt, heads(Cfg.heads(c), Cfg.half(c), q))
            +ks = ropes(rt, heads(Cfg.kv(c), Cfg.half(c), k))
            KA.then(BR<c, cap, t>, cap, 1n+t, attention(c, d, cap, t, qs, KV.push(c, cap, t, kv, room, joins(ks), chunks(Cfg.kv(c), Cfg.head(c), v))), kv2 => +at =>
            mv(BR<c, cap, t>, Cfg.dim(c), Cfg.dim(c), wo, at, wo => +a =>
              +x1 = T.zip_with.add(x, a)
              +n2 = rmsnorm(c, x1, ln2)
              mv(BR<c, cap, t>, Cfg.ffn(c), Cfg.dim(c), wg, n2, wg => +g =>
                mv(BR<c, cap, t>, Cfg.ffn(c), Cfg.dim(c), wu, n2, wu => +u =>
                  mv(BR<c, cap, t>, Cfg.dim(c), Cfg.ffn(c), wd, swiglu(g, u), wd => +m =>
                    +y = T.zip_with.add(x1, m)
                    BR{y, kv2, LW{ln1, wq, wk, wv, wo, ln2, wg, wu, wd}, Act{n1, q, k, v, flat(qs), flat(ks), at, a, n2, m, y}}))))))))

# =====================================================================================================
# The model
# =====================================================================================================

# emb: the embedding (vocab rows; also the LM head, tied); lnf: the final norm's gains
type Model<-c: Cfg> is Type:
  Model{d: Nat, emb: TA.Bands<Cfg.vocab(c), Cfg.dim(c)>, layers: List<&1, LW<c>>, lnf: List<&2, F32>}

# the last layer's output, the layers and caches handed back, and every layer's activations
type Run<-c: Cfg, -cap: Nat, -t: Nat> is Type:
  Run{x: List<&2, F32>, layers: List<&1, LW<c>>, cs: List<&1, KV<cap, 1n+t>>, acts: List<&2, Act>}

# pend is the previous layer's result: its output is the next layer's input
def run(+c: Cfg, +d: Nat, -cap: Nat, +t: Nat, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +rt: Rot, ls: List<&1, LW<c>>, cs: List<&1, KV<cap, t>>, al: List<&1, LW<c>>, ac: List<&1, KV<cap, 1n+t>>, aa: List<&2, Act>, pend: BR<c, cap, t>) -> Run<c, cap, t>:
  match ls cs pend:
    case Con{l, lt} Con{k, kt} BR{+y, kv, l0, act}:
      run(c, d, cap, t, room, rt, lt, kt, l0 <> al, kv <> ac, act <> aa, block(c, d, l, cap, t, k, room, rt, y))
    case _ _ BR{+y, kv, l0, act}:
      Run{y, List.reverse(&1, LW<c>, l0 <> al), List.reverse(&1, KV<cap, 1n+t>, kv <> ac), List.reverse(&2, Act, act <> aa)}

def layers(+c: Cfg, +d: Nat, -cap: Nat, +t: Nat, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +rt: Rot, ls: List<&1, LW<c>>, cs: List<&1, KV<cap, t>>, +x: List<&2, F32>) -> Run<c, cap, t>:
  match ls cs:
    case Con{l, lt} Con{k, kt}:
      run(c, d, cap, t, room, rt, lt, kt, Nil{}, Nil{}, Nil{}, block(c, d, l, cap, t, k, room, rt, x))
    case _ _:
      Run{x, Nil{}, Nil{}, Nil{}}

# what one position produces: the logits, the final norm, every layer's activations, the caches grown to t + 1
# positions and the model handed back
type Out<-c: Cfg, -cap: Nat, -t: Nat> is Type:
  Out{lg: List<&2, F32>, nf: List<&2, F32>, acts: List<&2, Act>, cs: List<&1, KV<cap, 1n+t>>, m: Model<c>}

# continues with the fields of a position's result: logits, final norm, activations, caches, model
def Out.then(-R: Type, -c: Cfg, -cap: Nat, -t: Nat, o: Out<c, cap, t>, k: List<&2, F32> -> List<&2, F32> -> List<&2, Act> -> List<&1, KV<cap, 1n+t>> -> Model<c> -> R) -> R:
  match o:
    case Out{lg, nf, acts, cs, m}:
      k(lg, nf, acts, cs, m)

# the final norm, then logits = emb · nf (the LM head is the embedding matrix)
def head_of(+c: Cfg, -cap: Nat, -t: Nat, +d: Nat, emb: TA.Bands<Cfg.vocab(c), Cfg.dim(c)>, +lnf: List<&2, F32>, r: Run<c, cap, t>) -> Out<c, cap, t>:
  match r:
    case Run{+x, ls, cs, acts}:
      +nf = rmsnorm(c, x, lnf)
      mv(Out<c, cap, t>, Cfg.vocab(c), Cfg.dim(c), emb, nf, emb => +lg => Out{lg, nf, acts, cs, Model{d, emb, ls, lnf}})

def forward.go(+c: Cfg, -cap: Nat, +t: Nat, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +d: Nat, ls: List<&1, LW<c>>, +lnf: List<&2, F32>, cs: List<&1, KV<cap, t>>, r: TA.BRow<Cfg.vocab(c), Cfg.dim(c)>) -> Out<c, cap, t>:
  match r:
    case TA.BRow{emb, row}:
      head_of(c, cap, t, d, emb, lnf, layers(c, d, cap, t, room, rot_at(c, t), ls, cs, row))

# token tok at position t (0-based), with the caches of positions 0..t-1 (one per layer) and room for cap positions;
# room is the proof that t < cap
def forward(+c: Cfg, -cap: Nat, +t: Nat, m: Model<c>, cs: List<&1, KV<cap, t>>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +tok: Nat) -> Out<c, cap, t>:
  match m:
    case Model{+d, emb, ls, +lnf}:
      forward.go(c, cap, t, room, d, ls, lnf, cs, TA.Bands.read_row(Cfg.vocab(c), Cfg.dim(c), emb, tok))

# n empty caches, one per layer, each with room for cap positions
def caches(+c: Cfg, n: Nat, +cap: Nat) -> List<&1, KV<cap, 0n>>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      KV.empty(c, cap) <> caches(c, p, cap)

# the first index of the largest value (torch.argmax does the same), and the value
type AM is Data:
  AM{i: Nat, v: F32}

def AM.i(am: AM) -> Nat:
  match am:
    case AM{i, v}:
      i

def AM.v(am: AM) -> F32:
  match am:
    case AM{i, v}:
      v

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax.go(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> AM:
  match xs g:
    case Nil{} _:
      AM{bi, bv}
    case Con{+h, +t} True{}:
      argmax.go(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax.go(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> AM:
  match xs:
    case Nil{}:
      AM{0n, 0.0}
    case Con{+h, +t}:
      argmax.go(t, gt_head(t, h), 1n, 0n, h)

# =====================================================================================================
# Loading (the weights as written by llama.bend's reference/prep.py: <dir>/w/<name>.bin, little-endian F32,
# one file per tensor, row = output; the config in <dir>/cfg.txt). The block reader is bend-ml's (MIT).
# =====================================================================================================

# 4 little-endian bytes -> F32 (U32 and F32 hold the same Word(32n): it is a bitcast)
def bitcast(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def f32_le(+b0: U32, +b1: U32, +b2: U32, +b3: U32) -> F32:
  bitcast((b0 + (b1 * 256 : U32) + (b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32))

def f32s(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Con{b0, Con{b1, Con{b2, Con{b3, t}}}}:
      f32_le(b0, b1, b2, b3) <> f32s(t)
    case _:
      Nil{}

# reads the file in blocks of `blk` bytes; k blocks remaining; m is the block already read
def rf_loop(k: Nat, +off: U32, +blk: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: List<&2, List<&2, F32>>) -> IO(List<&2, F32>):
  match k m:
    case 0n Tuple{f, r}:
      do IO<List<&2, F32>>:
        File.close(f)
        return List.concat(&2, F32, List.reverse(&2, List<&2, F32>, acc))
    case 1n+p Tuple{f, r}:
      do IO<List<&2, F32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + blk : U32), blk)
        rf_loop(p, (off + blk : U32), blk, m2, f32s(bs) <> acc)

def blocks_of(n: Nat, +per: Nat) -> Nat:
  Nat.div(Nat.add(n, Nat.sub(per, 1n)), per)

# n F32 numbers (little-endian) from a file
def read_floats(path: String, +n: Nat) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 0, 1048576)
    fl : List<&2, F32> <- rf_loop(blocks_of(Nat.mul(n, 4n), 1048576n), 0, 1048576, m, Nil{})
    return List.take(&2, F32, fl, n)

# the weight matrices are kept in 2^d row bands (TA.Bands); each matrix · vector product runs its bands in
# parallel without copying the weights (bend-ml NOTES exp. 11). With 2 threads or fewer the bands cannot help,
# so d = 0: a single band, which computes what the sequential Mat.matmul_nt computes (bend-ml exp. 14).
def depth_sel(few: Bool) -> Nat:
  match few:
    case True{}:
      0n
    case False{}:
      4n

def depth_for(+threads: U32) -> Nat:
  depth_sel(U32.is_le(threads, 2))

def nm(+i: Nat, f: String) -> String:
  "h" ++ Nat.show(i) ++ "." ++ f

# ---- the boundary: every weight file must hold exactly the numbers its shape needs ----

def size_cmp(ok: Bool, +path: String, +got: U32, +want: Nat) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, path ++ ": expected " ++ Nat.show(want) ++ " bytes, found " ++ U32.show(got) ++ "; run make prep again")

def size_ok(m: File & Result<&1, &1, U32 & String, U32>, +want: Nat, +path: String) -> IO(Unit):
  match m:
    case Tuple{f, r}:
      do IO<Unit>:
        +n : U32 <- IO.pass(U32, r)
        File.close(f)
        size_cmp(U32.is_eq(n, U32.from_nat(want)), path, n, want)

# stops the program unless the file holds exactly n F32 numbers (n * 4 bytes)
def check_size(+path: String, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    size_ok(m, Nat.mul(n, 4n), path)

def wpath(dir: String, name: String) -> String:
  dir ++ "/w/" ++ name ++ ".bin"

def get_list(+dir: String, +name: String, +n: Nat) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    check_size(wpath(dir, name), n)
    read_floats(wpath(dir, name), n)

# reads the file in 1 MB blocks and writes each block straight into the matrix, so only one
# block is a list at any time
# ---- matrices kept in bands: each band is read from its own range of the file ----

# writes the little-endian F32 numbers held in bs (4 bytes each) at index i, without building a
# list of F32 first
def fill_bytes(bs: List<&2, U32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match bs:
    case Con{+b0, Con{+b1, Con{+b2, Con{+b3, t}}}}:
      fill_bytes(t, (i + 1 : U32), Array.set(F32, a, i, f32_le(b0, b1, b2, b3)))
    case _:
      a

def fill_bytes_mat(-r: Nat, -c: Nat, +i: U32, bs: List<&2, U32>, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{d}:
      TA.Mat{fill_bytes(bs, i, d)}

# reads blocks of at most 1 MB until rem bytes have been read, so a band never takes numbers
# that belong to the next one
def stream_exact(k: Nat, -r: Nat, -c: Nat, +rem: Nat, +off: U32, +idx: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: TA.Mat<r, c>) -> IO(TA.Mat<r, c>):
  match k m:
    case 0n Tuple{f, rs}:
      do IO<TA.Mat<r, c>>:
        File.close(f)
        return acc
    case 1n+p Tuple{f, rs}:
      do IO<TA.Mat<r, c>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, rs)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + 1048576 : U32), U32.from_nat(Nat.min(1048576n, Nat.sub(rem, 1048576n))))
        stream_exact(p, r, c, Nat.sub(rem, 1048576n), (off + 1048576 : U32), (idx + 262144 : U32), m2, fill_bytes_mat(r, c, idx, bs, acc))

# rr x c numbers starting at number `start` of the file, into a band of r rows (rr == r, see
# load_tree; the Array is sized for rr * c numbers as Mat.zeros would size it)
def get_band(path: String, -r: Nat, +rr: Nat, +c: Nat, +start: Nat) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, U32.from_nat(Nat.mul(start, 4n)), U32.from_nat(Nat.min(1048576n, Nat.mul(Nat.mul(rr, c), 4n))))
    stream_exact(blocks_of(Nat.mul(Nat.mul(rr, c), 4n), 1048576n), r, c, Nat.mul(Nat.mul(rr, c), 4n), U32.from_nat(Nat.mul(start, 4n)), 0, m, TA.Mat{Array.new(F32, TA.cap_depth(Nat.mul(rr, c)), 0.0)})

# fills the bands of b (whose shape comes from TA.Bands.zeros) in row order; rr is r at run time
def load_tree(+c: Nat, -r: Nat, b: TA.Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, +path: String, +start: Nat) -> IO(TA.Bands<r, c>):
  match b:
    case TA.BLeaf{m}:
      do IO<TA.Bands<r, c>>:
        band : TA.Mat<r, c> <- get_band(path, r, rr, c, start)
        return TA.BLeaf{band}
    case TA.BNode{x, y}:
      do IO<TA.Bands<r, c>>:
        x2 : TA.Bands<TA.half(r), c> <- load_tree(c, TA.half(r), x, TA.half(rr), TA.eq_half(rr, r, e), path, start)
        y2 : TA.Bands<Nat.sub(r, TA.half(r)), c> <- load_tree(c, Nat.sub(r, TA.half(r)), y, Nat.sub(rr, TA.half(rr)), TA.eq_rest(rr, r, e), path, Nat.add(start, Nat.mul(TA.half(rr), c)))
        return TA.BNode{x2, y2}

def get_bands(+dir: String, +name: String, +d: Nat, +r: Nat, +c: Nat) -> IO(TA.Bands<r, c>):
  do IO<TA.Bands<r, c>>:
    check_size(wpath(dir, name), Nat.mul(r, c))
    load_tree(c, r, TA.Bands.zeros(d, r, c), r, {==}, wpath(dir, name), 0n)

# one layer's weights; d: the band depth (depth_for)
def load_lw(+c: Cfg, +dir: String, +i: Nat, +d: Nat) -> IO(LW<c>):
  do IO<LW<c>>:
    ln1 : List<&2, F32> <- get_list(dir, nm(i, "ln1"), Cfg.dim(c))
    q : TA.Bands<Cfg.dim(c), Cfg.dim(c)> <- get_bands(dir, nm(i, "q"), d, Cfg.dim(c), Cfg.dim(c))
    k : TA.Bands<Cfg.kvw(c), Cfg.dim(c)> <- get_bands(dir, nm(i, "k"), d, Cfg.kvw(c), Cfg.dim(c))
    v : TA.Bands<Cfg.kvw(c), Cfg.dim(c)> <- get_bands(dir, nm(i, "v"), d, Cfg.kvw(c), Cfg.dim(c))
    o : TA.Bands<Cfg.dim(c), Cfg.dim(c)> <- get_bands(dir, nm(i, "o"), d, Cfg.dim(c), Cfg.dim(c))
    ln2 : List<&2, F32> <- get_list(dir, nm(i, "ln2"), Cfg.dim(c))
    gate : TA.Bands<Cfg.ffn(c), Cfg.dim(c)> <- get_bands(dir, nm(i, "gate"), d, Cfg.ffn(c), Cfg.dim(c))
    up : TA.Bands<Cfg.ffn(c), Cfg.dim(c)> <- get_bands(dir, nm(i, "up"), d, Cfg.ffn(c), Cfg.dim(c))
    down : TA.Bands<Cfg.dim(c), Cfg.ffn(c)> <- get_bands(dir, nm(i, "down"), d, Cfg.dim(c), Cfg.ffn(c))
    return LW{ln1, q, k, v, o, ln2, gate, up, down}

def load_layers(k: Nat, +c: Cfg, +dir: String, +d: Nat, +i: Nat, acc: List<&1, LW<c>>) -> IO(List<&1, LW<c>>):
  match k:
    case 0n:
      IO.pure(List<&1, LW<c>>, List.reverse(&1, LW<c>, acc))
    case 1n+p:
      do IO<List<&1, LW<c>>>:
        l : LW<c> <- load_lw(c, dir, i, d)
        load_layers(p, c, dir, d, 1n+i, l <> acc)

# the whole model; the embedding gets one more level of bands than the layers (2^5 with 16 threads)
def load_model(+c: Cfg, +dir: String) -> IO(Model<c>):
  do IO<Model<c>>:
    threads : U32 <- IO.thread_count()
    +d : Nat = depth_for(threads)
    emb : TA.Bands<Cfg.vocab(c), Cfg.dim(c)> <- get_bands(dir, "emb", Nat.add(d, Nat.min(d, 1n)), Cfg.vocab(c), Cfg.dim(c))
    ls : List<&1, LW<c>> <- load_layers(Cfg.layers(c), c, dir, d, 0n, Nil{})
    lnf : List<&2, F32> <- get_list(dir, "lnf", Cfg.dim(c))
    return Model{d, emb, ls, lnf}

# ---- the config file: kv g half f layers vocab, then the bits of eps and theta as F32, in decimal ----

def rb_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

def read_bytes(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 1048576)
    rb_done(m)

def digit_head(xs: List<&2, U32>) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+b, t}:
      Bool.and(U32.is_le(48, b), U32.is_le(b, 57))

# the decimal numbers in bs, separated by anything that is not a digit; dg says whether bs starts with a digit
def u32s.go(bs: List<&2, U32>, dg: Bool, +cur: U32, have: Bool, acc: List<&2, U32>) -> List<&2, U32>:
  match bs dg have:
    case Nil{} _ True{}:
      List.reverse(&2, U32, cur <> acc)
    case Nil{} _ False{}:
      List.reverse(&2, U32, acc)
    case Con{+b, +t} True{} _:
      u32s.go(t, digit_head(t), ((cur * 10 : U32) + U32.sub(b, 48) : U32), True{}, acc)
    case Con{b, +t} False{} True{}:
      u32s.go(t, digit_head(t), 0, False{}, cur <> acc)
    case Con{b, +t} False{} False{}:
      u32s.go(t, digit_head(t), 0, False{}, acc)

def u32s(+bs: List<&2, U32>) -> List<&2, U32>:
  u32s.go(bs, digit_head(bs), 0, False{}, Nil{})

def cfg_of(+path: String, xs: List<&2, U32>) -> IO(Cfg):
  match xs:
    case Con{+kv, Con{+g, Con{+half, Con{+f, Con{+l, Con{+v, Con{+e, Con{+t, Nil{}}}}}}}}}:
      IO.pure(Cfg, Cfg{U32.to_nat(kv), U32.to_nat(g), U32.to_nat(half), U32.to_nat(f), U32.to_nat(l), U32.to_nat(v), bitcast(e), bitcast(t)})
    case _:
      IO.die(Cfg, 1, path ++ ": expected 8 numbers (kv g half f layers vocab eps_bits theta_bits); run make prep again")

# the config of the model in dir (written by reference/prep.py)
def read_cfg(+dir: String) -> IO(Cfg):
  do IO<Cfg>:
    bs : List<&2, U32> <- read_bytes(dir ++ "/cfg.txt")
    cfg_of(dir ++ "/cfg.txt", u32s(bs))
