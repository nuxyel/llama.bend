# The fast weight loader: the same model as main.bend's load_model, read from the downloaded hf/model.safetensors
# itself. Every matrix band and norm is read by bend-ml-binary-read's read_bf16_at straight into its Array, each BF16
# number widened exactly to F32, at the byte offset that reference/prep.py wrote into w/st.txt (no F32 copy needed: the
# numbers are those of w/<name>.bin, which main.bend's plain-Bend loader reads).
#
#   import ../llama/main.bend as M
#   import ../llama/load.bend as L
#   ...
#   model : M.Model<c> <- L.load_model(c, dir)
#
# read_f32 is foreign code (C and JavaScript), so --verdict fails for this file and for anything that
# imports it. That is why it is a file of its own: main.bend, which holds the laws, does not import it
# and keeps its own loader in plain Bend, which this one must match bit for bit (make bits).
import Base
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bend-ml-binary-read@0.2.0.0/main.bend as R
import ./main.bend as M

# the numbers a read must have answered, or the program stops
def read_cmp(ok: Bool, +path: String, +got: U32, +want: Nat) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, path ++ ": read " ++ U32.show(got) ++ " numbers, expected " ++ Nat.show(want) ++ "; run make prep again")

def read_done(-r: Nat, -c: Nat, p: File & (Array<F32> & Result<&1, &1, U32 & String, U32>), +want: Nat, +path: String) -> IO(TA.Mat<r, c>):
  match p:
    case Tuple{f, Tuple{a, res}}:
      do IO<TA.Mat<r, c>>:
        +n : U32 <- IO.pass(U32, res)
        File.close(f)
        read_cmp(U32.is_eq(n, U32.from_nat(want)), path, n, want)
        return TA.Mat{a}

# ---- w/st.txt: tensor t's BF16 data starts at byte hi * 2^32 + lo of hf/model.safetensors (numbers 2t and 2t + 1) ----

def st_path(+dir: String) -> String:
  dir ++ "/hf/model.safetensors"

def nth(xs: List<&2, U32>, k: Nat) -> U32:
  match xs k:
    case Con{+x, t} 0n:
      x
    case Con{x, t} 1n+p:
      nth(t, p)
    case Nil{} _:
      0

# the tensors in w/st.txt's order (reference/prep.py's weights()): emb, then layer i's ln1 q k v o ln2 gate up down
# (j = 0..8), then lnf
def t_layer(+i: Nat, +j: Nat) -> Nat:
  Nat.add(Nat.add(1n, Nat.mul(9n, i)), j)

def st_check(ok: Bool, +dir: String, xs: List<&2, U32>) -> IO(List<&2, U32>):
  match ok:
    case True{}:
      IO.pure(List<&2, U32>, xs)
    case False{}:
      IO.die(List<&2, U32>, 1, dir ++ "/w/st.txt: not two numbers for each of the model's tensors; run reference/prep.py --index")

# w/st.txt's numbers, two for each of the 2 + 9 * layers tensors
def read_st(+dir: String, +layers: Nat) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- M.read_bytes(dir ++ "/w/st.txt")
    +xs : List<&2, U32> = M.u32s(bs)
    st_check(Nat.is_eq(List.length(&2, U32, xs), Nat.mul(2n, Nat.add(2n, Nat.mul(9n, layers)))), dir, xs)

def st_hi(+st: List<&2, U32>, +t: Nat) -> U32:
  nth(st, Nat.mul(2n, t))

def st_lo(+st: List<&2, U32>, +t: Nat) -> U32:
  nth(st, 1n+Nat.mul(2n, t))

# hi * 2^32 + lo + x, as (hi, lo): the carry when lo + x wraps
def hi_add(+hi: U32, +lo: U32, +x: U32) -> U32:
  (hi + Bool.pick(U32, U32.is_lt((lo + x : U32), lo), 1, 0) : U32)

# rr x c BF16 numbers from number `start` of the tensor at (hi, lo), into a band of r rows (rr == r, see load_tree),
# read in one call into an Array sized as Mat.zeros would size it
def get_band(+path: String, +hi: U32, +lo: U32, -r: Nat, +rr: Nat, +c: Nat, +start: Nat) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    +x : U32 = U32.from_nat(Nat.mul(start, 2n))
    f : File <- IO.try(File, File.open(path, "r"))
    p : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_bf16_at(f, hi_add(hi, lo, x), (lo + x : U32), U32.from_nat(Nat.mul(rr, c)), Array.new(F32, TA.cap_depth(Nat.mul(rr, c)), 0.0))
    read_done(r, c, p, Nat.mul(rr, c), path)

# cells i, i - 1, ..., 0 of an array onto acc, so the list comes out in cell order
def cells(k: Nat, +i: U32, acc: List<&2, F32>, r: Array<F32> & F32) -> List<&2, F32>:
  match k r:
    case 0n Tuple{a, +x}:
      x <> acc
    case 1n+p Tuple{a, +x}:
      cells(p, (i - 1 : U32), x <> acc, Array.get(F32, a, (i - 1 : U32)))

def m_list(+n: Nat, m: TA.Mat<1n, n>) -> List<&2, F32>:
  match m:
    case TA.Mat{a}:
      cells(Nat.sub(n, 1n), U32.from_nat(Nat.sub(n, 1n)), Nil{}, Array.get(F32, a, U32.from_nat(Nat.sub(n, 1n))))

# a norm's n weights (n >= 1) as a list, from the tensor at (hi, lo)
def get_list(+path: String, +hi: U32, +lo: U32, +n: Nat) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    m : TA.Mat<1n, n> <- get_band(path, hi, lo, 1n, 1n, n, 0n)
    return m_list(n, m)

# fills the bands of b (whose shape comes from TA.Bands.zeros) in row order; rr is r at run time
def load_tree(+c: Nat, -r: Nat, b: TA.Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, +path: String, +hi: U32, +lo: U32, +start: Nat) -> IO(TA.Bands<r, c>):
  match b:
    case TA.BLeaf{m}:
      do IO<TA.Bands<r, c>>:
        band : TA.Mat<r, c> <- get_band(path, hi, lo, r, rr, c, start)
        return TA.BLeaf{band}
    case TA.BNode{x, y}:
      do IO<TA.Bands<r, c>>:
        x2 : TA.Bands<TA.half(r), c> <- load_tree(c, TA.half(r), x, TA.half(rr), TA.eq_half(rr, r, e), path, hi, lo, start)
        y2 : TA.Bands<Nat.sub(r, TA.half(r)), c> <- load_tree(c, Nat.sub(r, TA.half(r)), y, Nat.sub(rr, TA.half(rr)), TA.eq_rest(rr, r, e), path, hi, lo, Nat.add(start, Nat.mul(TA.half(rr), c)))
        return TA.BNode{x2, y2}

# tensor t (w/st.txt's order) as bands of depth d; a short read stops the program (read_done)
def get_bands(+dir: String, +st: List<&2, U32>, +t: Nat, +d: Nat, +r: Nat, +c: Nat) -> IO(TA.Bands<r, c>):
  load_tree(c, r, TA.Bands.zeros(d, r, c), r, {==}, st_path(dir), st_hi(st, t), st_lo(st, t), 0n)

def get_norm(+dir: String, +st: List<&2, U32>, +t: Nat, +n: Nat) -> IO(List<&2, F32>):
  get_list(st_path(dir), st_hi(st, t), st_lo(st, t), n)

# one layer's weights (tensors t_layer(i, 0..8)); d: the band depth (M.depth_for). The norms are short lists
def load_lw(+c: M.Cfg, +dir: String, +st: List<&2, U32>, +i: Nat, +d: Nat) -> IO(M.LW<c>):
  do IO<M.LW<c>>:
    ln1 : List<&2, F32> <- get_norm(dir, st, t_layer(i, 0n), M.Cfg.dim(c))
    q : TA.Bands<M.Cfg.dim(c), M.Cfg.dim(c)> <- get_bands(dir, st, t_layer(i, 1n), d, M.Cfg.dim(c), M.Cfg.dim(c))
    k : TA.Bands<M.Cfg.kvw(c), M.Cfg.dim(c)> <- get_bands(dir, st, t_layer(i, 2n), d, M.Cfg.kvw(c), M.Cfg.dim(c))
    v : TA.Bands<M.Cfg.kvw(c), M.Cfg.dim(c)> <- get_bands(dir, st, t_layer(i, 3n), d, M.Cfg.kvw(c), M.Cfg.dim(c))
    o : TA.Bands<M.Cfg.dim(c), M.Cfg.dim(c)> <- get_bands(dir, st, t_layer(i, 4n), d, M.Cfg.dim(c), M.Cfg.dim(c))
    ln2 : List<&2, F32> <- get_norm(dir, st, t_layer(i, 5n), M.Cfg.dim(c))
    gate : TA.Bands<M.Cfg.ffn(c), M.Cfg.dim(c)> <- get_bands(dir, st, t_layer(i, 6n), d, M.Cfg.ffn(c), M.Cfg.dim(c))
    up : TA.Bands<M.Cfg.ffn(c), M.Cfg.dim(c)> <- get_bands(dir, st, t_layer(i, 7n), d, M.Cfg.ffn(c), M.Cfg.dim(c))
    down : TA.Bands<M.Cfg.dim(c), M.Cfg.ffn(c)> <- get_bands(dir, st, t_layer(i, 8n), d, M.Cfg.dim(c), M.Cfg.ffn(c))
    return M.LW{ln1, q, k, v, o, ln2, gate, up, down}

def load_layers(k: Nat, +c: M.Cfg, +dir: String, +st: List<&2, U32>, +d: Nat, +i: Nat, acc: List<&1, M.LW<c>>) -> IO(List<&1, M.LW<c>>):
  match k:
    case 0n:
      IO.pure(List<&1, M.LW<c>>, List.reverse(&1, M.LW<c>, acc))
    case 1n+p:
      do IO<List<&1, M.LW<c>>>:
        l : M.LW<c> <- load_lw(c, dir, st, i, d)
        load_layers(p, c, dir, st, d, 1n+i, l <> acc)

# the whole model, as M.load_model builds it; the embedding gets one more level of bands than the layers
def load_model(+c: M.Cfg, +dir: String) -> IO(M.Model<c>):
  do IO<M.Model<c>>:
    threads : U32 <- IO.thread_count()
    +d : Nat = M.depth_for(threads)
    +st : List<&2, U32> <- read_st(dir, M.Cfg.layers(c))
    emb : TA.Bands<M.Cfg.vocab(c), M.Cfg.dim(c)> <- get_bands(dir, st, 0n, Nat.add(d, Nat.min(d, 1n)), M.Cfg.vocab(c), M.Cfg.dim(c))
    ls : List<&1, M.LW<c>> <- load_layers(M.Cfg.layers(c), c, dir, st, d, 0n, Nil{})
    lnf : List<&2, F32> <- get_norm(dir, st, t_layer(M.Cfg.layers(c), 0n), M.Cfg.dim(c))
    return M.Model{d, emb, ls, lnf}
