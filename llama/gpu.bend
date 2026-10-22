# Generation with every matrix · vector product on the GPU: main.bend's forward pass for generation (forward_gen,
# forward_pre and block_gen's layer) with its products run by bend-ml-tensor-gpu instead of tensor-array's Bands.
#
#   import ../llama/main.bend as M
#   import ../llama/gpu.bend as G
#   model : G.Model<c> <- G.load_model(c, dir)
#   G.forward_gen(c, cap, t, model, cs, room, tok)     # as M.forward_gen, with G.Out for M.Out
#
# A product is TG.matmul_nt(1n, k, r, ...): one `!` call (a bang), which runs on the GPU when the program is built with
# CUDA and on the CPU threads with `--gpu off`. tensor-gpu gives Mat.matmul_nt's bits, and a single band of
# TA.Bands.matvec_l computes what Mat.matmul_nt computes for one row, so the logits and the tokens are main.bend's, bit
# for bit (make bench-gpu checks the tokens). Each bang has a fixed cost of about a millisecond, so the products that
# read the same vector are one product here: q, k and v are one matrix of dim + 2·kvw rows, gate and up one of 2·ffn
# rows (four bangs a layer instead of seven). Every output row is still its own dot product, computed in the same order,
# so stacking the rows changes no bit. Everything else (norms, RoPE, attention, the cache) is main.bend's own code,
# run on the host.
#
# Outside the proofs, on purpose: tensor-gpu shares arrays through an unsafe primitive of Base (Array.fork), so `bend`
# names the defs below that reach it, and no file with laws imports this one (`make laws` checks it). It also imports
# load.bend (bend-ml-binary-read, foreign code) for the weights. All of the computation is Bend.

import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bend-ml-tensor-gpu@0.1.2.1/main.bend as TG
import ./main.bend as M
import ./load.bend as L

# ---- the weights, with the products that read one vector stacked ----

# the rows of the stacked q, k, v matrix, and of the stacked gate, up matrix
def qkv(+c: M.Cfg) -> Nat:
  Nat.add(M.Cfg.dim(c), Nat.add(M.Cfg.kvw(c), M.Cfg.kvw(c)))

def gu(+c: M.Cfg) -> Nat:
  Nat.add(M.Cfg.ffn(c), M.Cfg.ffn(c))

# one layer: wqkv holds q's rows, then k's, then v's; wgu holds gate's rows, then up's
type LW<-c: M.Cfg> is Type:
  LW{ln1: List<&2, F32>, wqkv: TA.Mat<qkv(c), M.Cfg.dim(c)>, wo: TA.Mat<M.Cfg.dim(c), M.Cfg.dim(c)>, ln2: List<&2, F32>, wgu: TA.Mat<gu(c), M.Cfg.dim(c)>, wd: TA.Mat<M.Cfg.dim(c), M.Cfg.ffn(c)>}

# emb: the embedding, also the LM head (tied)
type Model<-c: M.Cfg> is Type:
  Model{emb: TA.Mat<M.Cfg.vocab(c), M.Cfg.dim(c)>, layers: List<&1, LW<c>>, lnf: List<&2, F32>}

# ---- one product ----

def gmv.out(-B: Type, -r: Nat, -k: Nat, w: TA.Mat<r, k>, ml: TA.ML<1n, r>, then: TA.Mat<r, k> -> List<&2, F32> -> B) -> B:
  match ml:
    case TA.ML{y, l}:
      then(w, l)

def gmv.nt(-B: Type, +r: Nat, -k: Nat, p: TA.MMulNT<1n, k, r>, then: TA.Mat<r, k> -> List<&2, F32> -> B) -> B:
  match p:
    case TA.MMulNT{x, w, y}:
      gmv.out(B, r, k, w, TA.Mat.to_list(1n, r, y), then)

# y = W · x (W: r x k, one row per output), then `then` with W handed back and y, as main.bend's mv
def gmv(-B: Type, +r: Nat, +k: Nat, w: TA.Mat<r, k>, +x: List<&2, F32>, then: TA.Mat<r, k> -> List<&2, F32> -> B) -> B:
  gmv.nt(B, r, k, TG.matmul_nt(1n, k, r, 0n, TA.Mat.from_list(1n, k, x), w), then)

# ---- the forward pass for generation ----

# the layer's output, its cache grown to t + 1 positions and its weights handed back
type BR<-c: M.Cfg, -cap: Nat, -t: Nat> is Type:
  BR{y: List<&2, F32>, kv: M.KV<cap, 1n+t>, lw: LW<c>}

# main.bend's block_gen with the stacked products; attention runs its heads one after another (d = 0): with d > 0 its
# parallel lets fork the host's evaluation, and a bang that follows them in the same evaluation runs on the CPU threads,
# not on the GPU (Bend's SHADERS guide; on 16 threads only 48 of 4,007 products reached the device, nsys, 2026-10-18)
def block(+c: M.Cfg, lw: LW<c>, -cap: Nat, +t: Nat, kv: M.KV<cap, t>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +rt: M.Rot, +x: List<&2, F32>) -> BR<c, cap, t>:
  match lw:
    case LW{+ln1, wqkv, wo, +ln2, wgu, wd}:
      +n1 = M.rmsnorm(c, x, ln1)
      gmv(BR<c, cap, t>, qkv(c), M.Cfg.dim(c), wqkv, n1, wqkv => +y3 =>
        +kv2 = List.drop(&2, F32, y3, M.Cfg.dim(c))
        +qs = M.ropes(rt, M.heads(M.Cfg.heads(c), M.Cfg.half(c), List.take(&2, F32, y3, M.Cfg.dim(c))))
        +ks = M.ropes(rt, M.heads(M.Cfg.kv(c), M.Cfg.half(c), List.take(&2, F32, kv2, M.Cfg.kvw(c))))
        M.KA.then(BR<c, cap, t>, cap, 1n+t, M.attention(c, 0n, cap, t, qs, M.KV.push(c, cap, t, kv, room, M.joins(ks), M.chunks(M.Cfg.kv(c), M.Cfg.head(c), List.drop(&2, F32, kv2, M.Cfg.kvw(c))))), kv3 => +at =>
        gmv(BR<c, cap, t>, M.Cfg.dim(c), M.Cfg.dim(c), wo, at, wo => +a =>
          +x1 = M.list.add(x, a)
          +n2 = M.rmsnorm(c, x1, ln2)
          gmv(BR<c, cap, t>, gu(c), M.Cfg.dim(c), wgu, n2, wgu => +y2 =>
            gmv(BR<c, cap, t>, M.Cfg.dim(c), M.Cfg.ffn(c), wd, M.swiglu(List.take(&2, F32, y2, M.Cfg.ffn(c)), List.drop(&2, F32, y2, M.Cfg.ffn(c))), wd => +m =>
              BR{M.list.add(x1, m), kv3, LW{ln1, wqkv, wo, ln2, wgu, wd}})))))

# the last layer's output, the layers and caches handed back
type Run<-c: M.Cfg, -cap: Nat, -t: Nat> is Type:
  Run{x: List<&2, F32>, layers: List<&1, LW<c>>, cs: List<&1, M.KV<cap, 1n+t>>}

def run(+c: M.Cfg, -cap: Nat, +t: Nat, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +rt: M.Rot, ls: List<&1, LW<c>>, cs: List<&1, M.KV<cap, t>>, al: List<&1, LW<c>>, ac: List<&1, M.KV<cap, 1n+t>>, pend: BR<c, cap, t>) -> Run<c, cap, t>:
  match ls cs pend:
    case Con{l, lt} Con{k, kt} BR{+y, kv, l0}:
      run(c, cap, t, room, rt, lt, kt, l0 <> al, kv <> ac, block(c, l, cap, t, k, room, rt, y))
    case _ _ BR{+y, kv, l0}:
      Run{y, List.reverse(&1, LW<c>, l0 <> al), List.reverse(&1, M.KV<cap, 1n+t>, kv <> ac)}

def layers(+c: M.Cfg, -cap: Nat, +t: Nat, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +rt: M.Rot, ls: List<&1, LW<c>>, cs: List<&1, M.KV<cap, t>>, +x: List<&2, F32>) -> Run<c, cap, t>:
  match ls cs:
    case Con{l, lt} Con{k, kt}:
      run(c, cap, t, room, rt, lt, kt, Nil{}, Nil{}, block(c, l, cap, t, k, room, rt, x))
    case _ _:
      Run{x, Nil{}, Nil{}}

# what one position produces: the logits (empty for a prompt position), the caches grown to t + 1 positions and the
# model handed back
type Out<-c: M.Cfg, -cap: Nat, -t: Nat> is Type:
  Out{lg: List<&2, F32>, cs: List<&1, M.KV<cap, 1n+t>>, m: Model<c>}

def Out.then(-R: Type, -c: M.Cfg, -cap: Nat, -t: Nat, o: Out<c, cap, t>, k: List<&2, F32> -> List<&1, M.KV<cap, 1n+t>> -> Model<c> -> R) -> R:
  match o:
    case Out{lg, cs, m}:
      k(lg, cs, m)

# the final norm, then logits = emb · nf
def head_of(+c: M.Cfg, -cap: Nat, -t: Nat, emb: TA.Mat<M.Cfg.vocab(c), M.Cfg.dim(c)>, +lnf: List<&2, F32>, r: Run<c, cap, t>) -> Out<c, cap, t>:
  match r:
    case Run{+x, ls, cs}:
      gmv(Out<c, cap, t>, M.Cfg.vocab(c), M.Cfg.dim(c), emb, M.rmsnorm(c, x, lnf), emb => +lg => Out{lg, cs, Model{emb, ls, lnf}})

# a prompt position: no final norm and no product by the embedding (main.bend's pre_of)
def pre_of(-c: M.Cfg, -cap: Nat, -t: Nat, emb: TA.Mat<M.Cfg.vocab(c), M.Cfg.dim(c)>, +lnf: List<&2, F32>, r: Run<c, cap, t>) -> Out<c, cap, t>:
  match r:
    case Run{x, ls, cs}:
      Out{Nil{}, cs, Model{emb, ls, lnf}}

def forward.go(+gen: Bool, +c: M.Cfg, -cap: Nat, +t: Nat, -room: {True{} == TA.leb(1n+t, cap) : Bool}, ls: List<&1, LW<c>>, +lnf: List<&2, F32>, cs: List<&1, M.KV<cap, t>>, r: TA.MRow<M.Cfg.vocab(c), M.Cfg.dim(c)>) -> Out<c, cap, t>:
  match gen r:
    case True{} TA.MRow{emb, row}:
      head_of(c, cap, t, emb, lnf, layers(c, cap, t, room, M.rot_at(c, t), ls, cs, row))
    case False{} TA.MRow{emb, row}:
      pre_of(c, cap, t, emb, lnf, layers(c, cap, t, room, M.rot_at(c, t), ls, cs, row))

def forward_k(+gen: Bool, +c: M.Cfg, -cap: Nat, +t: Nat, m: Model<c>, cs: List<&1, M.KV<cap, t>>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +tok: Nat) -> Out<c, cap, t>:
  match m:
    case Model{emb, ls, +lnf}:
      forward.go(gen, c, cap, t, room, ls, lnf, cs, TA.Mat.read_row(M.Cfg.vocab(c), M.Cfg.dim(c), emb, U32.from_nat(tok)))

# token tok at position t: the logits, the caches grown to t + 1 positions and the model (main.bend's forward_gen)
def forward_gen(+c: M.Cfg, -cap: Nat, +t: Nat, m: Model<c>, cs: List<&1, M.KV<cap, t>>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +tok: Nat) -> Out<c, cap, t>:
  forward_k(True{}, c, cap, t, m, cs, room, tok)

# a prompt position whose logits nobody reads (main.bend's forward_pre): lg is empty
def forward_pre(+c: M.Cfg, -cap: Nat, +t: Nat, m: Model<c>, cs: List<&1, M.KV<cap, t>>, -room: {True{} == TA.leb(1n+t, cap) : Bool}, +tok: Nat) -> Out<c, cap, t>:
  forward_k(False{}, c, cap, t, m, cs, room, tok)

# ---- loading ----

# n numbers of src from 0 into dst from o; rv is src with its number at i
def copy(n: Nat, +i: U32, +o: U32, dst: Array<F32>, rv: Array<F32> & F32) -> Array<F32>:
  match n rv:
    case 0n Tuple{src, x}:
      dst
    case 1n+p Tuple{src, +x}:
      copy(p, (i + 1 : U32), (o + 1 : U32), Array.set(F32, dst, o, x), Array.get(F32, src, (i + 1 : U32)))

# the r x c numbers of m at row `at` of s
def put(-r: Nat, -c: Nat, -s: Nat, +rr: Nat, +cc: Nat, +at: Nat, m: TA.Mat<r, c>, st: TA.Mat<s, c>) -> TA.Mat<s, c>:
  match m st:
    case TA.Mat{a} TA.Mat{d}:
      TA.Mat{copy(Nat.mul(rr, cc), 0, U32.from_nat(Nat.mul(at, cc)), d, Array.get(F32, a, 0))}

# one weight file as one Mat (r x c), with the size check of main.bend's loader
def get_mat(+dir: String, +name: String, +r: Nat, +c: Nat) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    M.check_size(M.wpath(dir, name), Nat.mul(r, c))
    L.get_band(M.wpath(dir, name), r, r, c, 0n)

def load_lw(+c: M.Cfg, +dir: String, +i: Nat) -> IO(LW<c>):
  do IO<LW<c>>:
    ln1 : List<&2, F32> <- M.get_list(dir, M.nm(i, "ln1"), M.Cfg.dim(c))
    q : TA.Mat<M.Cfg.dim(c), M.Cfg.dim(c)> <- get_mat(dir, M.nm(i, "q"), M.Cfg.dim(c), M.Cfg.dim(c))
    k : TA.Mat<M.Cfg.kvw(c), M.Cfg.dim(c)> <- get_mat(dir, M.nm(i, "k"), M.Cfg.kvw(c), M.Cfg.dim(c))
    v : TA.Mat<M.Cfg.kvw(c), M.Cfg.dim(c)> <- get_mat(dir, M.nm(i, "v"), M.Cfg.kvw(c), M.Cfg.dim(c))
    o : TA.Mat<M.Cfg.dim(c), M.Cfg.dim(c)> <- get_mat(dir, M.nm(i, "o"), M.Cfg.dim(c), M.Cfg.dim(c))
    ln2 : List<&2, F32> <- M.get_list(dir, M.nm(i, "ln2"), M.Cfg.dim(c))
    gate : TA.Mat<M.Cfg.ffn(c), M.Cfg.dim(c)> <- get_mat(dir, M.nm(i, "gate"), M.Cfg.ffn(c), M.Cfg.dim(c))
    up : TA.Mat<M.Cfg.ffn(c), M.Cfg.dim(c)> <- get_mat(dir, M.nm(i, "up"), M.Cfg.ffn(c), M.Cfg.dim(c))
    down : TA.Mat<M.Cfg.dim(c), M.Cfg.ffn(c)> <- get_mat(dir, M.nm(i, "down"), M.Cfg.dim(c), M.Cfg.ffn(c))
    wqkv : TA.Mat<qkv(c), M.Cfg.dim(c)> = put(M.Cfg.kvw(c), M.Cfg.dim(c), qkv(c), M.Cfg.kvw(c), M.Cfg.dim(c), Nat.add(M.Cfg.dim(c), M.Cfg.kvw(c)), v, put(M.Cfg.kvw(c), M.Cfg.dim(c), qkv(c), M.Cfg.kvw(c), M.Cfg.dim(c), M.Cfg.dim(c), k, put(M.Cfg.dim(c), M.Cfg.dim(c), qkv(c), M.Cfg.dim(c), M.Cfg.dim(c), 0n, q, TA.Mat.zeros(qkv(c), M.Cfg.dim(c)))))
    wgu : TA.Mat<gu(c), M.Cfg.dim(c)> = put(M.Cfg.ffn(c), M.Cfg.dim(c), gu(c), M.Cfg.ffn(c), M.Cfg.dim(c), M.Cfg.ffn(c), up, put(M.Cfg.ffn(c), M.Cfg.dim(c), gu(c), M.Cfg.ffn(c), M.Cfg.dim(c), 0n, gate, TA.Mat.zeros(gu(c), M.Cfg.dim(c))))
    return LW{ln1, wqkv, o, ln2, wgu, down}

def load_layers(k: Nat, +c: M.Cfg, +dir: String, +i: Nat, acc: List<&1, LW<c>>) -> IO(List<&1, LW<c>>):
  match k:
    case 0n:
      IO.pure(List<&1, LW<c>>, List.reverse(&1, LW<c>, acc))
    case 1n+p:
      do IO<List<&1, LW<c>>>:
        l : LW<c> <- load_lw(c, dir, i)
        load_layers(p, c, dir, 1n+i, l <> acc)

def load_model(+c: M.Cfg, +dir: String) -> IO(Model<c>):
  do IO<Model<c>>:
    emb : TA.Mat<M.Cfg.vocab(c), M.Cfg.dim(c)> <- get_mat(dir, "emb", M.Cfg.vocab(c), M.Cfg.dim(c))
    ls : List<&1, LW<c>> <- load_layers(M.Cfg.layers(c), c, dir, 0n, Nil{})
    lnf : List<&2, F32> <- M.get_list(dir, "lnf", M.Cfg.dim(c))
    return Model{emb, ls, lnf}
