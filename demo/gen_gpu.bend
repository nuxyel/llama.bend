# demo/gen.bend with its matrix · vector products on the GPU (llama/gpu.bend, over bend-ml-tensor-gpu): the same
# arguments, the same output and, since the products give tensor-array's bits, the same tokens and logits.
#
#   gen_gpu <dir> "prompt" n          (built with CUDA: `bend demo/gen_gpu.bend -o x` writes x and x.gpu)
#   gen_gpu --gpu off <dir> "prompt" n   (the same bangs on the CPU threads)
#
# make bench-gpu checks its tokens against transformers'.

import Base
import ./tok.bend as K
import ./gen.bend as D
import ../llama/main.bend as M
import ../llama/gpu.bend as G
import bend-ml-bpe-tokenizer@0.1.3.0/main.bend as BPE

# demo/gen.bend's loop, with llama/gpu.bend's model and forward pass
def loop(fuel: Nat, +c: M.Cfg, +t0: Nat, -cap: Nat, +t: Nat, -inv: {Nat.add(t, fuel) == cap : Nat}, +tok: Nat, queue: List<&2, Nat>, left: Nat, gen: List<&2, Nat>, m: G.Model<c>, cs: List<&1, M.KV<cap, t>>) -> IO(List<&2, Nat>):
  match fuel queue left:
    case 0n _ _:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))
    case 1n+p _ 0n:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))
    case 1n++p Con{+q, qt} 1n+lp:
      G.Out.then(IO(List<&2, Nat>), c, cap, t, G.forward_pre(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => cs2 => m2 =>
        loop(p, c, t0, cap, 1n+t, M.step_inv(t, p, cap, inv), q, qt, 1n+lp, gen, m2, cs2))
    case 1n++p Nil{} 1n+lp:
      G.Out.then(IO(List<&2, Nat>), c, cap, t, G.forward_gen(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => cs2 => m2 =>
        +am = M.argmax(lg)
        +id = M.AM.i(am)
        do IO<List<&2, Nat>>:
          t1 : Nat <- IO.now()
          IO.print("  id " ++ Nat.show(id) ++ "  logit " ++ U32.show(D.bits(M.AM.v(am))) ++ " " ++ F32.show(M.AM.v(am)) ++ "  t=" ++ D.tsec(t1, t0))
          loop(p, c, t0, cap, 1n+t, M.step_inv(t, p, cap, inv), id, Nil{}, D.left_after(id, lp), id <> gen, m2, cs2))

def start2(+c: M.Cfg, model: G.Model<c>, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("empty prompt")
    case Con{+t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        IO.print("prompt ids: " ++ D.show_nats(ids))
        gen : List<&2, Nat> <- loop(Nat.add(List.length(&2, Nat, ids), n), c, t0ms, Nat.add(List.length(&2, Nat, ids), n), 0n, {==}, t0, qt, n, Nil{}, model, M.caches(c, M.Cfg.layers(c), Nat.add(List.length(&2, Nat, ids), n)))
        IO.print("text: " ++ K.decode_text(table, perm, List.append(&2, Nat, ids, gen)))

def generate(+c: M.Cfg, model: G.Model<c>, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  start2(c, model, K.rules.go(K.nums(mb), 0n), K.nums(pb), K.encode_text(K.rules.go(K.nums(mb), 0n), K.nums(pb), D.utf8_of(String.to_list(prompt))), n)

def run3(ok: Bool, +dir: String, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +c : M.Cfg <- M.read_cfg(dir)
        model : G.Model<c> <- G.load_model(c, dir)
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ D.tsec(t1, t0))
        generate(c, model, mb, pb, prompt, n)
    case False{}:
      IO.print("invalid merges table")

def run2(+dir: String, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  run3(K.ids_ok(K.nums(mb), 0n), dir, mb, pb, prompt, n)

def run1(+dir: String, prompt: String, mn: Maybe<&2, Nat>) -> IO(Unit):
  match mn:
    case None{}:
      IO.print("usage: gen_gpu <dir> <prompt> <n_tokens>")
    case Some{n}:
      do IO<Unit>:
        mb : List<&2, U32> <- K.read_bytes(dir ++ "/merges_num.txt")
        pb : List<&2, U32> <- K.read_bytes(dir ++ "/perm.txt")
        run2(dir, mb, pb, prompt, n)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{d, Con{prompt, Con{n, Nil{}}}}}:
      run1(d, prompt, Nat.read(n))
    case _:
      IO.print("usage: gen_gpu <dir> <prompt> <n_tokens>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
