# Runs one layer of a Llama model (bend-ml-llama, llama/main.bend) on T input rows and prints every intermediate
# as exact F32 bits, one line per intermediate and position: `name t b0 b1 ...`, after a line with RoPE's
# inv_freq. reference/test_layer.py compares them with transformers.
#
#   layer_test <dir> <layer> <input.bin> <T>   (the model in dir: cfg.txt and w/, from reference/prep.py;
#                                               input: T x hidden little-endian F32, from reference/layer_ref.py)

import Base
import ../llama/main.bend as M

def bits(f: F32) -> U32:
  match f:
    case F32{w}:
      U32{w}

def show_bits(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ U32.show(bits(h)) ++ show_bits(t)

def line(name: String, +t: Nat, xs: List<&2, F32>) -> IO(Unit):
  IO.print(name ++ " " ++ Nat.show(t) ++ show_bits(xs))

def rows(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> rows(p, c, List.drop(&2, F32, xs, c))

def show_act(+t: Nat, act: M.Act) -> IO(Unit):
  match act:
    case M.Act{n1, q, k, v, qr, kr, attn, a, n2, m, y}:
      do IO<Unit>:
        line("n1", t, n1)
        line("q", t, q)
        line("k", t, k)
        line("v", t, v)
        line("qr", t, qr)
        line("kr", t, kr)
        line("attn", t, attn)
        line("a", t, a)
        line("n2", t, n2)
        line("m", t, m)
        line("y", t, y)

# xs[0] runs at position t with the cache of positions 0..t-1; then the rest. The cache has room for cap positions
# with t + length(xs) = cap (inv), which gives each position its proof that t < cap.
def go(xs: List<&2, List<&2, F32>>, +c: M.Cfg, +d: Nat, -cap: Nat, +t: Nat, -inv: {Nat.add(t, List.length(&2, List<&2, F32>, xs)) == cap : Nat}, lw: M.LW<c>, kv: M.KV<cap, t>) -> IO(Unit):
  match xs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{+x, +rest}:
      M.BR.then(IO(Unit), c, cap, t, M.block(c, d, lw, cap, t, kv, M.step_room(t, List.length(&2, List<&2, F32>, rest), cap, inv), M.rot_at(c, t), x), y => kv2 => lw2 => act =>
        do IO<Unit>:
          show_act(t, act)
          go(rest, c, d, cap, 1n+t, M.step_inv(t, List.length(&2, List<&2, F32>, rest), cap, inv), lw2, kv2))

def run(+dir: String, +layer: Nat, path: String, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    threads : U32 <- IO.thread_count()
    +d : Nat = M.depth_for(threads)
    +c : M.Cfg <- M.read_cfg(dir)
    lw : M.LW<c> <- M.load_lw(c, dir, layer, d)
    IO.print("inv_freq" ++ show_bits(M.inv_freq(c)))
    xs : List<&2, F32> <- M.read_floats(path, Nat.mul(n, M.Cfg.dim(c)))
    +rs : List<&2, List<&2, F32>> = rows(n, M.Cfg.dim(c), xs)
    go(rs, c, d, List.length(&2, List<&2, F32>, rs), 0n, {==}, lw, M.KV.empty(c, List.length(&2, List<&2, F32>, rs)))

def run2(+dir: String, +layer: Maybe<&2, Nat>, path: String, +n: Maybe<&2, Nat>) -> IO(Unit):
  match layer n:
    case Some{+l} Some{+k}:
      run(dir, l, path, k)
    case _ _:
      IO.print("usage: layer_test <dir> <layer> <input.bin> <T>")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{d, Con{l, Con{p, Con{k, Nil{}}}}}}:
      run2(d, Nat.read(l), p, Nat.read(k))
    case _:
      IO.print("usage: layer_test <dir> <layer> <input.bin> <T>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
