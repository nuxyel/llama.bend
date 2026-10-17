# Greedy generation with a Llama model in Bend (bend-ml-llama, llama/main.bend; SmolLM2's tokenizer,
# demo/tok.bend): the prompt is tokenized in Bend, runs through the forward pass position by position (one
# key/value cache per layer), and then each new token is the argmax of the logits, fed back in. Generation
# stops after n tokens, or after the end-of-text token (id 0), as transformers' generate does.
# reference/test_gen.py compares the ids, the chosen logits and the text with transformers.
#
#   gen <dir> "prompt" n         (the model in dir: cfg.txt, w/ and the tokenizer tables, from reference/prep.py)
#
# prints `prompt ids: ...`, one line per new token (`id <id> logit <F32 bits> <logit> t=<seconds>`) and
# `text: <prompt and continuation>`.

import Base
import ./tok.bend as K
import ../llama/main.bend as M
import ../llama/load.bend as L
import bend-ml-bpe-tokenizer@0.1.3.0/main.bend as BPE

# ---- the prompt as UTF-8 bytes ----

def selb(b: Bool, x: List<&2, U32>, y: List<&2, U32>) -> List<&2, U32>:
  match b:
    case True{}:
      x
    case False{}:
      y

def enc1(+cp: U32) -> List<&2, U32>:
  selb(U32.is_lt(cp, 128), cp <> Nil{}, selb(U32.is_lt(cp, 2048), (192 + U32.div(cp, 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{}), selb(U32.is_lt(cp, 65536), (224 + U32.div(cp, 4096) : U32) <> ((128 + U32.mod(U32.div(cp, 64), 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{})), (240 + U32.div(cp, 262144) : U32) <> ((128 + U32.mod(U32.div(cp, 4096), 64) : U32) <> ((128 + U32.mod(U32.div(cp, 64), 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{}))))))

def utf8_of(cs: List<&2, Char>) -> List<&2, U32>:
  match cs:
    case Nil{}:
      Nil{}
    case Con{Chr{cp}, t}:
      List.append(&2, U32, enc1(cp), utf8_of(t))

# ---- the generation loop ----

def bits(f: F32) -> U32:
  match f:
    case F32{w}:
      U32{w}

# r < 1000 milliseconds, as three digits
def ms3(small: Bool, mid: Bool, +r: Nat) -> String:
  match small mid:
    case True{} _:
      "00" ++ Nat.show(r)
    case False{} True{}:
      "0" ++ Nat.show(r)
    case False{} False{}:
      Nat.show(r)

# dt milliseconds as seconds with three decimals (make bench reads the per-token times at this resolution)
def tsec2(+dt: Nat) -> String:
  Nat.show(Nat.div(dt, 1000n)) ++ "." ++ ms3(Nat.is_lt(Nat.mod(dt, 1000n), 10n), Nat.is_lt(Nat.mod(dt, 1000n), 100n), Nat.mod(dt, 1000n)) ++ "s"

def tsec(t1: Nat, t0: Nat) -> String:
  tsec2(Nat.sub(t1, t0))

# after the end-of-text token nothing more is generated
def left_after(+id: Nat, +left: Nat) -> Nat:
  K.sel(Nat.is_eq(id, 0n), 0n, left)

# tok runs at position t; queue: the prompt ids still to feed; left: tokens still to generate; gen: the generated
# ids, newest first. Returns the generated ids in order. The caches have room for cap positions, and fuel counts
# the steps left with t + fuel = cap (inv), which gives each step its proof that t < cap.
def loop(fuel: Nat, +c: M.Cfg, +t0: Nat, -cap: Nat, +t: Nat, -inv: {Nat.add(t, fuel) == cap : Nat}, +tok: Nat, queue: List<&2, Nat>, left: Nat, gen: List<&2, Nat>, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> IO(List<&2, Nat>):
  match fuel queue left:
    case 0n _ _:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))
    case 1n+p _ 0n:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))
    case 1n++p Con{+q, qt} 1n+lp:
      M.Out.then(IO(List<&2, Nat>), c, cap, t, M.forward_pre(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        loop(p, c, t0, cap, 1n+t, M.step_inv(t, p, cap, inv), q, qt, 1n+lp, gen, m2, cs2))
    case 1n++p Nil{} 1n+lp:
      M.Out.then(IO(List<&2, Nat>), c, cap, t, M.forward_gen(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        +am = M.argmax(lg)
        +id = M.AM.i(am)
        do IO<List<&2, Nat>>:
          t1 : Nat <- IO.now()
          IO.print("  id " ++ Nat.show(id) ++ "  logit " ++ U32.show(bits(M.AM.v(am))) ++ " " ++ F32.show(M.AM.v(am)) ++ "  t=" ++ tsec(t1, t0))
          loop(p, c, t0, cap, 1n+t, M.step_inv(t, p, cap, inv), id, Nil{}, left_after(id, lp), id <> gen, m2, cs2))

# ---- main program ----

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def start2(+c: M.Cfg, model: M.Model<c>, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("empty prompt")
    case Con{+t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        IO.print("prompt ids: " ++ show_nats(ids))
        gen : List<&2, Nat> <- loop(Nat.add(List.length(&2, Nat, ids), n), c, t0ms, Nat.add(List.length(&2, Nat, ids), n), 0n, {==}, t0, qt, n, Nil{}, model, M.caches(c, M.Cfg.layers(c), Nat.add(List.length(&2, Nat, ids), n)))
        IO.print("text: " ++ K.decode_text(table, perm, List.append(&2, Nat, ids, gen)))

def generate(+c: M.Cfg, model: M.Model<c>, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  start2(c, model, K.rules.go(K.nums(mb), 0n), K.nums(pb), K.encode_text(K.rules.go(K.nums(mb), 0n), K.nums(pb), utf8_of(String.to_list(prompt))), n)

# refuse to run on a malformed merges table (checked before the weights are loaded)
def run3(ok: Bool, +dir: String, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +c : M.Cfg <- M.read_cfg(dir)
        model : M.Model<c> <- L.load_model(c, dir)
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ tsec(t1, t0))
        generate(c, model, mb, pb, prompt, n)
    case False{}:
      IO.print("invalid merges table")

def run2(+dir: String, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  run3(K.ids_ok(K.nums(mb), 0n), dir, mb, pb, prompt, n)

def run1(+dir: String, prompt: String, mn: Maybe<&2, Nat>) -> IO(Unit):
  match mn:
    case None{}:
      IO.print("usage: gen <dir> <prompt> <n_tokens>")
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
      IO.print("usage: gen <dir> <prompt> <n_tokens>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
