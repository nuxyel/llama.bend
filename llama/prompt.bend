# bend-ml-llama, the prompt in one pass: the n positions of a prompt go through each layer together, so that every
# weight matrix is read once for all of them (main.bend's forward_pre reads it once per position). Between the
# products the positions sit in arrays (n rows, row-major) and are read one row at a time by the code main.bend
# already has (rmsnorm, RoPE, attention over the cache, SwiGLU, the residual sums), so every number is computed by
# the same operations in the same order as position by position: the caches and the last position's logits have the
# same bits (reference/bits.py and the generation tests go through this file).
#
#   import ./main.bend as M
#   import ./prompt.bend as P
#   P.forward_prompt(c, cap, 0n, n, n, {==}, le, model, M.caches(c, M.Cfg.layers(c), cap), ids)   # P.OutP{lg, cs, m}
#
# The product is tensor-array's gemm_t8 (cells of 8) with the weights as its first operand: each row of a weight
# matrix meets every position before the next row is read. It has the bits of the matrix · vector product
# (tensor-array's tests/t8.bend; a product x · y commutes exactly).

import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./main.bend as M

# =====================================================================================================
# Rows of an array as lists
# =====================================================================================================

# continues with an array and the list read from it
def al_then(-R: Type, al: TA.AL, k: Array<F32> -> List<&2, F32> -> R) -> R:
  match al:
    case TA.AL{a, l}:
      k(a, l)

# continues with the two arrays
def p2_then(-R: Type, p: TA.P2, k: Array<F32> -> Array<F32> -> R) -> R:
  match p:
    case TA.P2{a, b}:
      k(a, b)

# continues with an array and its copy
def with2(-R: Type, p: Array<F32> & Array<F32>, k: Array<F32> -> Array<F32> -> R) -> R:
  match p:
    case Tuple{a, b}:
      k(a, b)

# n rows of w zeros
def zeros(+n: Nat, +w: Nat) -> Array<F32>:
  Array.new(F32, TA.cap_depth(Nat.mul(n, w)), 0.0)

# row i of a (rows of w >= 1 numbers, row-major) as a list, and a handed back
def row(+w: Nat, +i: U32, a: Array<F32>) -> TA.AL:
  TA.read_l(Nat.sub(w, 1n), (((i + 1 : U32) * U32.from_nat(w) : U32) - 1 : U32), Nil{}, Array.get(F32, a, (((i + 1 : U32) * U32.from_nat(w) : U32) - 1 : U32)))

# =====================================================================================================
# W · every row of X, over the bands
# =====================================================================================================

# one band's results: rr rows (the band's) by n columns (the positions), row j at j·n
type Leaf is Type:
  Leaf{rr: Nat, out: Array<F32>}

# the bands handed back, and their results in band order
type BM<-r: Nat, -c: Nat> is Type:
  BM{b: TA.Bands<r, c>, outs: List<&1, Leaf>}

def bm_then(-R: Type, -r: Nat, -c: Nat, bm: BM<r, c>, k: TA.Bands<r, c> -> List<&1, Leaf> -> R) -> R:
  match bm:
    case BM{b, outs}:
      k(b, outs)

def bm_g(-r: Nat, -c: Nat, rr: Nat, g: TA.G) -> BM<r, c>:
  match g:
    case TA.G{w, x, out}:
      BM{TA.BLeaf{TA.Mat{w}}, Leaf{rr, out} <> Nil{}}

# a band (rr rows of c numbers) times the n rows of x: out[j·n + i] = row j of the band · row i of x
def bm_mat(-r: Nat, +c: Nat, +rr: Nat, +n: Nat, x: Array<F32>, m: TA.Mat<r, c>) -> BM<r, c>:
  match m:
    case TA.Mat{w}:
      bm_g(r, c, rr, TA.gemm_t8(0n, rr, n, c, U32.from_nat(c), 1, 1, U32.from_nat(c), w, x, Array.new(F32, TA.cap_depth(Nat.mul(rr, n)), 0.0)))

def bm_join(-r: Nat, -c: Nat, p: BM<TA.half(r), c>, q: BM<Nat.sub(r, TA.half(r)), c>) -> BM<r, c>:
  match p q:
    case BM{bx, ox} BM{by, oy}:
      BM{TA.BNode{bx, by}, List.append(&1, Leaf, ox, oy)}

# the two bands of a node run in parallel, one with x and the other with a copy of it (n·c numbers, small next to
# the weights): a band's product owns its operand, so every band needs its own. A fork with two copies in hand (as
# tensor-array's bmv has them for one vector) would make a copy more than it uses at every node
def bmm(+c: Nat, -r: Nat, b: TA.Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, +n: Nat, x: Array<F32>) -> BM<r, c>:
  match b:
    case TA.BLeaf{m}:
      bm_mat(r, c, rr, n, x, m)
    case TA.BNode{bx, by}:
      with2(BM<r, c>, Array.clone(F32, x), x1 => x2 =>
        p q = bmm(c, TA.half(r), bx, TA.half(rr), TA.eq_half(rr, r, e), n, x1) bmm(c, Nat.sub(r, TA.half(r)), by, Nat.sub(rr, TA.half(rr)), TA.eq_rest(rr, r, e), n, x2)
        bm_join(r, c, p, q))

# W (r x c, in bands) times the n rows of x (n x c): the bands handed back and every band's results
def mm(+r: Nat, +c: Nat, b: TA.Bands<r, c>, +n: Nat, x: Array<F32>) -> BM<r, c>:
  bmm(c, r, b, r, {==}, n, x)

# one band's column: its numbers from idx down in steps of nu, consed onto acc, so they come out in row order
def col(mleft: Nat, +idx: U32, +nu: U32, acc: List<&2, F32>, rc: Array<F32> & F32) -> TA.AL:
  match mleft rc:
    case 0n Tuple{a, +v}:
      TA.AL{a, v <> acc}
    case 1n+p Tuple{a, +v}:
      col(p, (idx - nu : U32), nu, v <> acc, Array.get(F32, a, (idx - nu : U32)))

# the bands' results handed back, and one position's vector
type RO is Type:
  RO{outs: List<&1, Leaf>, v: List<&2, F32>}

def ro_then(-R: Type, ro: RO, k: List<&1, Leaf> -> List<&2, F32> -> R) -> R:
  match ro:
    case RO{outs, v}:
      k(outs, v)

# rev: the bands' results, last band first; done collects them back in band order; acc is the vector's tail so far
def vec.go(rev: List<&1, Leaf>, +i: U32, +nu: U32, done: List<&1, Leaf>, acc: List<&2, F32>) -> RO:
  match rev:
    case Nil{}:
      RO{done, acc}
    case Con{Leaf{rr, out}, t}:
      match rr:
        case 0n:
          vec.go(t, i, nu, Leaf{0n, out} <> done, acc)
        case 1n++p:
          al_then(RO, col(p, ((U32.from_nat(p) * nu : U32) + i : U32), nu, acc, Array.get(F32, out, ((U32.from_nat(p) * nu : U32) + i : U32))), out2 => l =>
            vec.go(t, i, nu, Leaf{1n+p, out2} <> done, l))

# position i's vector (of the nu positions): every band's column i, as one list in row order, read from the last
# band's last row back so that nothing is appended or reversed; the results are handed back
def vec(outs: List<&1, Leaf>, +i: U32, +nu: U32) -> RO:
  vec.go(List.reverse(&1, Leaf, outs), i, nu, Nil{}, Nil{})

# =====================================================================================================
# A layer over the n positions
# =====================================================================================================

# nrm's row i = rmsnorm of x's row i, for the rows i .. i + left - 1 (w numbers each); both arrays come back
def norms(left: Nat, +c: M.Cfg, +g: List<&2, F32>, +w: Nat, +i: U32, x: Array<F32>, nrm: Array<F32>) -> TA.P2:
  match left:
    case 0n:
      TA.P2{x, nrm}
    case 1n+p:
      al_then(TA.P2, row(w, i, x), x2 => l =>
        norms(p, c, g, w, (i + 1 : U32), x2, TA.fill_list(M.rmsnorm(c, l, g), (i * U32.from_nat(w) : U32), nrm)))

# t + 0 = tT, so t = tT
def end_eq(t: Nat, -tT: Nat, e: {Nat.add(t, 0n) == tT : Nat}) -> {t == tT : Nat}:
  %M.add_zero(t) : {_ == tT : Nat}
  e

# a cache of t positions is a cache of tT positions when t = tT. The count is in the type only (KV.push rebuilds
# the cache at 1 + t the same way), so the proof is asked for and nothing is done with it at run time.
def kv_cast(-cap: Nat, -t: Nat, -tT: Nat, -e: {t == tT : Nat}, kv: M.KV<cap, t>) -> M.KV<cap, tT>:
  match kv:
    case M.KV{ks, vs}:
      M.KV{ks, vs}

# the cache with the positions written, and their attention outputs (n rows of dim numbers)
type AP<-cap: Nat, -t: Nat> is Type:
  AP{kv: M.KV<cap, t>, at: Array<F32>}

def ap_then(-R: Type, -cap: Nat, -t: Nat, ap: AP<cap, t>, k: M.KV<cap, t> -> Array<F32> -> R) -> R:
  match ap:
    case AP{kv, at}:
      k(kv, at)

# Position by position, as main.bend's block does it: the queries and keys of position t rotated, its keys and
# values pushed into the cache, attention over positions 0..t, the output written to at's row i. qo, ko and vo are
# the three products' results; left positions remain, the cache holds t of them, and t + left = tT <= cap (e, le),
# which gives each push its proof of room (step_room, leb_trans) and the cache its final index (end_eq).
def atts(left: Nat, +c: M.Cfg, +d: Nat, -cap: Nat, +t: Nat, -tT: Nat, -e: {Nat.add(t, left) == tT : Nat}, -le: {True{} == TA.leb(tT, cap) : Bool}, +i: U32, +nu: U32, qo: List<&1, Leaf>, ko: List<&1, Leaf>, vo: List<&1, Leaf>, kv: M.KV<cap, t>, at: Array<F32>) -> AP<cap, tT>:
  match left:
    case 0n:
      AP{kv_cast(cap, t, tT, end_eq(t, tT, e), kv), at}
    case 1n++p:
      ro_then(AP<cap, tT>, vec(qo, i, nu), qo2 => q =>
        ro_then(AP<cap, tT>, vec(ko, i, nu), ko2 => k =>
          ro_then(AP<cap, tT>, vec(vo, i, nu), vo2 => v =>
            +rt = M.rot_at(c, t)
            M.KA.then(AP<cap, tT>, cap, 1n+t, M.attention(c, d, cap, t, M.ropes(rt, M.heads(M.Cfg.heads(c), M.Cfg.half(c), q)), M.KV.push(c, cap, t, kv, M.leb_trans(1n+t, tT, cap, M.step_room(t, p, tT, e), le), M.joins(M.ropes(rt, M.heads(M.Cfg.kv(c), M.Cfg.half(c), k))), M.chunks(M.Cfg.kv(c), M.Cfg.head(c), v))), kv2 => a =>
              atts(p, c, d, cap, 1n+t, tT, M.step_inv(t, p, tT, e), le, (i + 1 : U32), nu, qo2, ko2, vo2, kv2, TA.fill_list(a, (i * U32.from_nat(M.Cfg.dim(c)) : U32), at))))))

# x's row i becomes x + a (a: the output projection's results, position i), and nrm's row i its rmsnorm
def res_norm(left: Nat, +c: M.Cfg, +g: List<&2, F32>, +w: Nat, +i: U32, +nu: U32, ao: List<&1, Leaf>, x: Array<F32>, nrm: Array<F32>) -> TA.P2:
  match left:
    case 0n:
      TA.P2{x, nrm}
    case 1n+p:
      ro_then(TA.P2, vec(ao, i, nu), ao2 => a =>
        al_then(TA.P2, row(w, i, x), x2 => l =>
          +x1 = M.list.add(l, a)
          res_norm(p, c, g, w, (i + 1 : U32), nu, ao2, TA.fill_list(x1, (i * U32.from_nat(w) : U32), x2), TA.fill_list(M.rmsnorm(c, x1, g), (i * U32.from_nat(w) : U32), nrm))))

# s's row i = swiglu of the gate's and the up projection's results at position i (f numbers)
def swis(left: Nat, +f: Nat, +i: U32, +nu: U32, go: List<&1, Leaf>, uo: List<&1, Leaf>, s: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      s
    case 1n+p:
      ro_then(Array<F32>, vec(go, i, nu), go2 => g =>
        ro_then(Array<F32>, vec(uo, i, nu), uo2 => u =>
          swis(p, f, (i + 1 : U32), nu, go2, uo2, TA.fill_list(M.swiglu(g, u), (i * U32.from_nat(f) : U32), s))))

# x's row i becomes x + m (m: the down projection's results, position i)
def res(left: Nat, +w: Nat, +i: U32, +nu: U32, mo: List<&1, Leaf>, x: Array<F32>) -> Array<F32>:
  match left:
    case 0n:
      x
    case 1n+p:
      ro_then(Array<F32>, vec(mo, i, nu), mo2 => m =>
        al_then(Array<F32>, row(w, i, x), x2 => l =>
          res(p, w, (i + 1 : U32), nu, mo2, TA.fill_list(M.list.add(l, m), (i * U32.from_nat(w) : U32), x2))))

# the layer's outputs for the n positions (n rows of dim numbers), its cache grown by them, its weights handed back
type BP<-c: M.Cfg, -cap: Nat, -t: Nat> is Type:
  BP{x: Array<F32>, kv: M.KV<cap, t>, lw: M.LW<c>}

def bp_then(-R: Type, -c: M.Cfg, -cap: Nat, -t: Nat, b: BP<c, cap, t>, k: Array<F32> -> M.KV<cap, t> -> M.LW<c> -> R) -> R:
  match b:
    case BP{x, kv, lw}:
      k(x, kv, lw)

# The layer on the n positions t0 .. t0 + n - 1, whose inputs are x's rows: main.bend's block_gen with each of its
# seven products done once for every position. tT = t0 + n (e) and tT <= cap (le).
def block_p(+c: M.Cfg, +d: Nat, lw: M.LW<c>, -cap: Nat, +t0: Nat, +n: Nat, -tT: Nat, -e: {Nat.add(t0, n) == tT : Nat}, -le: {True{} == TA.leb(tT, cap) : Bool}, kv: M.KV<cap, t0>, x: Array<F32>) -> BP<c, cap, tT>:
  match lw:
    case M.LW{+ln1, wq, wk, wv, wo, +ln2, wg, wu, wd}:
      p2_then(BP<c, cap, tT>, norms(n, c, ln1, M.Cfg.dim(c), 0, x, zeros(n, M.Cfg.dim(c))), x1 => n1 =>
        with2(BP<c, cap, tT>, Array.clone(F32, n1), na => nb =>
          bm_then(BP<c, cap, tT>, M.Cfg.dim(c), M.Cfg.dim(c), mm(M.Cfg.dim(c), M.Cfg.dim(c), wq, n, na), wq2 => qo =>
            with2(BP<c, cap, tT>, Array.clone(F32, nb), nc => nd =>
              bm_then(BP<c, cap, tT>, M.Cfg.kvw(c), M.Cfg.dim(c), mm(M.Cfg.kvw(c), M.Cfg.dim(c), wk, n, nc), wk2 => ko =>
                bm_then(BP<c, cap, tT>, M.Cfg.kvw(c), M.Cfg.dim(c), mm(M.Cfg.kvw(c), M.Cfg.dim(c), wv, n, nd), wv2 => vo =>
                  ap_then(BP<c, cap, tT>, cap, tT, atts(n, c, d, cap, t0, tT, e, le, 0, U32.from_nat(n), qo, ko, vo, kv, zeros(n, M.Cfg.dim(c))), kv2 => at =>
                    bm_then(BP<c, cap, tT>, M.Cfg.dim(c), M.Cfg.dim(c), mm(M.Cfg.dim(c), M.Cfg.dim(c), wo, n, at), wo2 => ao =>
                      p2_then(BP<c, cap, tT>, res_norm(n, c, ln2, M.Cfg.dim(c), 0, U32.from_nat(n), ao, x1, zeros(n, M.Cfg.dim(c))), x2 => n2 =>
                        with2(BP<c, cap, tT>, Array.clone(F32, n2), ne => nf =>
                          bm_then(BP<c, cap, tT>, M.Cfg.ffn(c), M.Cfg.dim(c), mm(M.Cfg.ffn(c), M.Cfg.dim(c), wg, n, ne), wg2 => go =>
                            bm_then(BP<c, cap, tT>, M.Cfg.ffn(c), M.Cfg.dim(c), mm(M.Cfg.ffn(c), M.Cfg.dim(c), wu, n, nf), wu2 => uo =>
                              bm_then(BP<c, cap, tT>, M.Cfg.dim(c), M.Cfg.ffn(c), mm(M.Cfg.dim(c), M.Cfg.ffn(c), wd, n, swis(n, M.Cfg.ffn(c), 0, U32.from_nat(n), go, uo, zeros(n, M.Cfg.ffn(c)))), wd2 => mo =>
                                BP{res(n, M.Cfg.dim(c), 0, U32.from_nat(n), mo, x2), kv2, M.LW{ln1, wq2, wk2, wv2, wo2, ln2, wg2, wu2, wd2}})))))))))))))

# =====================================================================================================
# The model over the prompt
# =====================================================================================================

# the last layer's outputs, the layers and the caches handed back
type RP<-c: M.Cfg, -cap: Nat, -t: Nat> is Type:
  RP{x: Array<F32>, layers: List<&1, M.LW<c>>, cs: List<&1, M.KV<cap, t>>}

def rp_then(-R: Type, -c: M.Cfg, -cap: Nat, -t: Nat, r: RP<c, cap, t>, k: Array<F32> -> List<&1, M.LW<c>> -> List<&1, M.KV<cap, t>> -> R) -> R:
  match r:
    case RP{x, ls, cs}:
      k(x, ls, cs)

# every layer in turn; al and ac collect the layers and the caches already done, newest first
def run_p(+c: M.Cfg, +d: Nat, -cap: Nat, +t0: Nat, +n: Nat, -tT: Nat, -e: {Nat.add(t0, n) == tT : Nat}, -le: {True{} == TA.leb(tT, cap) : Bool}, ls: List<&1, M.LW<c>>, cs: List<&1, M.KV<cap, t0>>, al: List<&1, M.LW<c>>, ac: List<&1, M.KV<cap, tT>>, x: Array<F32>) -> RP<c, cap, tT>:
  match ls cs:
    case Con{l, lt} Con{k, kt}:
      bp_then(RP<c, cap, tT>, c, cap, tT, block_p(c, d, l, cap, t0, n, tT, e, le, k, x), x2 => kv => l2 =>
        run_p(c, d, cap, t0, n, tT, e, le, lt, kt, l2 <> al, kv <> ac, x2))
    case _ _:
      RP{x, List.reverse(&1, M.LW<c>, al), List.reverse(&1, M.KV<cap, tT>, ac)}

def brow_then(-R: Type, -r: Nat, -c: Nat, br: TA.BRow<r, c>, k: TA.Bands<r, c> -> List<&2, F32> -> R) -> R:
  match br:
    case TA.BRow{b, l}:
      k(b, l)

# the embedding handed back, and x with the embedding row of each id (row i at i·dim)
type EX<-c: M.Cfg> is Type:
  EX{emb: TA.Bands<M.Cfg.vocab(c), M.Cfg.dim(c)>, x: Array<F32>}

def ex_then(-R: Type, -c: M.Cfg, ex: EX<c>, k: TA.Bands<M.Cfg.vocab(c), M.Cfg.dim(c)> -> Array<F32> -> R) -> R:
  match ex:
    case EX{emb, x}:
      k(emb, x)

def embed(ids: List<&2, Nat>, +c: M.Cfg, +i: U32, emb: TA.Bands<M.Cfg.vocab(c), M.Cfg.dim(c)>, x: Array<F32>) -> EX<c>:
  match ids:
    case Nil{}:
      EX{emb, x}
    case Con{id, t}:
      brow_then(EX<c>, M.Cfg.vocab(c), M.Cfg.dim(c), TA.Bands.read_row(M.Cfg.vocab(c), M.Cfg.dim(c), emb, id), emb2 => r =>
        embed(t, c, (i + 1 : U32), emb2, TA.fill_list(r, (i * U32.from_nat(M.Cfg.dim(c)) : U32), x)))

# what a prompt produces: the logits of its last position, the caches grown by its positions, the model handed back
type OutP<-c: M.Cfg, -cap: Nat, -t: Nat> is Type:
  OutP{lg: List<&2, F32>, cs: List<&1, M.KV<cap, t>>, m: M.Model<c>}

# continues with the fields of a prompt's result: logits, caches, model
def OutP.then(-R: Type, -c: M.Cfg, -cap: Nat, -t: Nat, o: OutP<c, cap, t>, k: List<&2, F32> -> List<&1, M.KV<cap, t>> -> M.Model<c> -> R) -> R:
  match o:
    case OutP{lg, cs, m}:
      k(lg, cs, m)

# The n >= 1 tokens ids at positions t0 .. t0 + n - 1, with the caches of positions 0..t0-1 (one per layer) and room
# for cap positions: tT = t0 + n (e) and tT <= cap (le). The final norm and the product by the embedding run on the
# last position only, as main.bend's head_of does them. ids must hold n ids (a shorter list leaves rows of zeros).
def forward_prompt(+c: M.Cfg, -cap: Nat, +t0: Nat, +n: Nat, -tT: Nat, -e: {Nat.add(t0, n) == tT : Nat}, -le: {True{} == TA.leb(tT, cap) : Bool}, m: M.Model<c>, cs: List<&1, M.KV<cap, t0>>, ids: List<&2, Nat>) -> OutP<c, cap, tT>:
  match m:
    case M.Model{+d, emb, ls, +lnf}:
      ex_then(OutP<c, cap, tT>, c, embed(ids, c, 0, emb, zeros(n, M.Cfg.dim(c))), emb2 => x =>
        rp_then(OutP<c, cap, tT>, c, cap, tT, run_p(c, d, cap, t0, n, tT, e, le, ls, cs, Nil{}, Nil{}, x), x2 => ls2 => cs2 =>
          al_then(OutP<c, cap, tT>, row(M.Cfg.dim(c), U32.from_nat(Nat.sub(n, 1n)), x2), x3 => l =>
            M.mv(OutP<c, cap, tT>, M.Cfg.vocab(c), M.Cfg.dim(c), emb2, M.rmsnorm(c, l, lnf), emb3 => lg =>
              OutP{lg, cs2, M.Model{d, emb3, ls2, lnf}}))))
