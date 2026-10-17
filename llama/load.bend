# The fast weight loader: the same model as main.bend's load_model, with every matrix band read by
# bend-ml-binary-read's read_f32 straight into its Array, instead of a list of bytes per 1 MB block.
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
import bend-ml-tensor-array@0.1.12.0/main.bend as TA
import bend-ml-binary-read@0.1.0.0/main.bend as R
import ./main.bend as M

# the bytes a read must have answered: n numbers of 4 bytes, or the program stops
def read_cmp(ok: Bool, +path: String, +got: U32, +want: Nat) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, path ++ ": read " ++ U32.show(got) ++ " bytes, expected " ++ Nat.show(want) ++ "; run make prep again")

def read_done(-r: Nat, -c: Nat, p: File & (Array<F32> & Result<&1, &1, U32 & String, U32>), +want: Nat, +path: String) -> IO(TA.Mat<r, c>):
  match p:
    case Tuple{f, Tuple{a, res}}:
      do IO<TA.Mat<r, c>>:
        +n : U32 <- IO.pass(U32, res)
        File.close(f)
        read_cmp(U32.is_eq(n, U32.from_nat(want)), path, n, want)
        return TA.Mat{a}

# rr x c numbers starting at number `start` of the file, into a band of r rows (rr == r, see
# load_tree), read in one call into an Array sized as Mat.zeros would size it
def get_band(+path: String, -r: Nat, +rr: Nat, +c: Nat, +start: Nat) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    f : File <- IO.try(File, File.open(path, "r"))
    p : File & (Array<F32> & Result<&1, &1, U32 & String, U32>) <- R.read_f32(f, U32.from_nat(Nat.mul(start, 4n)), U32.from_nat(Nat.mul(rr, c)), Array.new(F32, TA.cap_depth(Nat.mul(rr, c)), 0.0))
    read_done(r, c, p, Nat.mul(Nat.mul(rr, c), 4n), path)

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
    M.check_size(M.wpath(dir, name), Nat.mul(r, c))
    load_tree(c, r, TA.Bands.zeros(d, r, c), r, {==}, M.wpath(dir, name), 0n)

# one layer's weights; d: the band depth (M.depth_for). The norms' weights are short lists, read as before
def load_lw(+c: M.Cfg, +dir: String, +i: Nat, +d: Nat) -> IO(M.LW<c>):
  do IO<M.LW<c>>:
    ln1 : List<&2, F32> <- M.get_list(dir, M.nm(i, "ln1"), M.Cfg.dim(c))
    q : TA.Bands<M.Cfg.dim(c), M.Cfg.dim(c)> <- get_bands(dir, M.nm(i, "q"), d, M.Cfg.dim(c), M.Cfg.dim(c))
    k : TA.Bands<M.Cfg.kvw(c), M.Cfg.dim(c)> <- get_bands(dir, M.nm(i, "k"), d, M.Cfg.kvw(c), M.Cfg.dim(c))
    v : TA.Bands<M.Cfg.kvw(c), M.Cfg.dim(c)> <- get_bands(dir, M.nm(i, "v"), d, M.Cfg.kvw(c), M.Cfg.dim(c))
    o : TA.Bands<M.Cfg.dim(c), M.Cfg.dim(c)> <- get_bands(dir, M.nm(i, "o"), d, M.Cfg.dim(c), M.Cfg.dim(c))
    ln2 : List<&2, F32> <- M.get_list(dir, M.nm(i, "ln2"), M.Cfg.dim(c))
    gate : TA.Bands<M.Cfg.ffn(c), M.Cfg.dim(c)> <- get_bands(dir, M.nm(i, "gate"), d, M.Cfg.ffn(c), M.Cfg.dim(c))
    up : TA.Bands<M.Cfg.ffn(c), M.Cfg.dim(c)> <- get_bands(dir, M.nm(i, "up"), d, M.Cfg.ffn(c), M.Cfg.dim(c))
    down : TA.Bands<M.Cfg.dim(c), M.Cfg.ffn(c)> <- get_bands(dir, M.nm(i, "down"), d, M.Cfg.dim(c), M.Cfg.ffn(c))
    return M.LW{ln1, q, k, v, o, ln2, gate, up, down}

def load_layers(k: Nat, +c: M.Cfg, +dir: String, +d: Nat, +i: Nat, acc: List<&1, M.LW<c>>) -> IO(List<&1, M.LW<c>>):
  match k:
    case 0n:
      IO.pure(List<&1, M.LW<c>>, List.reverse(&1, M.LW<c>, acc))
    case 1n+p:
      do IO<List<&1, M.LW<c>>>:
        l : M.LW<c> <- load_lw(c, dir, i, d)
        load_layers(p, c, dir, d, 1n+i, l <> acc)

# the whole model, as M.load_model builds it; the embedding gets one more level of bands than the layers
def load_model(+c: M.Cfg, +dir: String) -> IO(M.Model<c>):
  do IO<M.Model<c>>:
    threads : U32 <- IO.thread_count()
    +d : Nat = M.depth_for(threads)
    emb : TA.Bands<M.Cfg.vocab(c), M.Cfg.dim(c)> <- get_bands(dir, "emb", Nat.add(d, Nat.min(d, 1n)), M.Cfg.vocab(c), M.Cfg.dim(c))
    ls : List<&1, M.LW<c>> <- load_layers(M.Cfg.layers(c), c, dir, d, 0n, Nil{})
    lnf : List<&2, F32> <- M.get_list(dir, "lnf", M.Cfg.dim(c))
    return M.Model{d, emb, ls, lnf}
