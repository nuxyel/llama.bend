# Generation with a Llama model in Bend (bend-ml-llama, llama/main.bend; SmolLM2's tokenizer, demo/tok.bend): the
# prompt is tokenized in Bend, runs through the forward pass position by position (one key/value cache per layer),
# and then each new token is chosen from the logits and fed back in: the argmax (greedy, the default), or a draw
# with temperature, top-k and top-p from a seed (llama/main.bend's sample). Generation stops after n tokens, or
# after the end-of-text token (id 0), as transformers' generate does; in chat mode after <|im_end|> (id 2) too.
# reference/test_gen.py compares greedy ids, chosen logits and text with transformers; reference/test_sample.py
# checks the draws.
#
#   gen <dir> "prompt" n                              greedy (the model in dir: cfg.txt, w/ and the tokenizer
#                                                     tables, from reference/prep.py)
#   gen <dir> "prompt" n <temp> <top_k> <top_p> <seed>   sampled; temp 0 is greedy, top_k >= 1, 0 < top_p <= 1
#   gen <dir> --chat "message" n [<temp> <top_k> <top_p> <seed>]
#                                                     the message as one user turn in SmolLM2-Instruct's chat
#                                                     template (demo/tok.bend's chat_ids), the default system
#                                                     message first and the assistant's turn opened
#
# prints `prompt ids: ...`, one line per new token (`id <id> logit <F32 bits> <logit> t=<seconds>`) and
# `text: <prompt and continuation>` (special tokens left out); a sampled run first prints `sampling: ...` with the
# parameters as read (F32 bits). With GEN_CANDS=1 in the environment, each draw is preceded by `cands <id>:<logit
# bits> ...`, the top-k candidates largest first (for reference/test_sample.py).

import Base
import ../llama/tok.bend as K
import ../llama/main.bend as M
import ../llama/load.bend as L
import bend-ml-bpe-tokenizer@0.1.3.0/main.bend as BPE

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

# how each new token is chosen: the argmax (greedy, transformers' do_sample = False), or a draw (llama/main.bend's
# sample) with temperature temp, the k largest logits, top-p p, and draw j of the seed for the j-th new token
type Pick is Data:
  Greedy{}
  Draw{temp: F32, k: Nat, p: F32, seed: U32}

def show_cands(cs: List<&2, M.Cand>) -> String:
  match cs:
    case Nil{}:
      ""
    case Con{M.Cand{+v, i}, t}:
      " " ++ Nat.show(i) ++ ":" ++ U32.show(bits(v)) ++ show_cands(t)

# a draw, after its top-k candidates when show is set (GEN_CANDS)
def draw.out(show: Bool, +k: Nat, +lg: List<&2, F32>, s: M.Cand) -> IO(M.AM):
  match show s:
    case True{} M.Cand{v, i}:
      do IO<M.AM>:
        IO.print("  cands" ++ show_cands(M.topk(k, lg)))
        return M.AM{i, v}
    case False{} M.Cand{v, i}:
      IO.pure(M.AM, M.AM{i, v})

def draw(show: Bool, +temp: F32, +k: Nat, +p: F32, +seed: U32, +lg: List<&2, F32>, +j: Nat) -> IO(M.AM):
  draw.out(show, k, lg, M.sample(lg, temp, k, p, M.uniform(seed, U32.from_nat(j))))

# the greedy path is the argmax alone, as before sampling existed (make bits checks it)
def choose(show: Bool, pk: Pick, lg: List<&2, F32>, +j: Nat) -> IO(M.AM):
  match pk:
    case Greedy{}:
      IO.pure(M.AM, M.argmax(lg))
    case Draw{+temp, +k, +p, +seed}:
      draw(show, temp, k, p, seed, lg, j)

# after <|im_end|> (id 2) nothing more is generated in chat mode
def left_chat(chat: Bool, +id: Nat, +left: Nat) -> Nat:
  match chat:
    case True{}:
      K.sel(Nat.is_eq(id, 2n), 0n, left_after(id, left))
    case False{}:
      left_after(id, left)

# how the tokens are chosen (pk), whether this is a chat (chat) and whether draws print their candidates (show)
type Opts is Data:
  Opts{pk: Pick, chat: Bool, show: Bool}

def Opts.pk(o: Opts) -> Pick:
  match o:
    case Opts{pk, chat, show}:
      pk

def Opts.chat(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show}:
      chat

def Opts.show(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show}:
      show

# tok runs at position t; queue: the prompt ids still to feed; left: tokens still to generate; gen: the generated
# ids, newest first, j of them. Returns the generated ids in order. The caches have room for cap positions, and fuel
# counts the steps left with t + fuel = cap (inv), which gives each step its proof that t < cap.
def loop(fuel: Nat, +c: M.Cfg, +o: Opts, +t0: Nat, -cap: Nat, +t: Nat, -inv: {Nat.add(t, fuel) == cap : Nat}, +tok: Nat, queue: List<&2, Nat>, left: Nat, +j: Nat, gen: List<&2, Nat>, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> IO(List<&2, Nat>):
  match fuel queue left:
    case 0n _ _:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))
    case 1n+p _ 0n:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))
    case 1n++p Con{+q, qt} 1n+lp:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- forward_pre returns lg and nf empty, and acts empty on more than one band not-expected-because: measured 2026-10-16: forward_pre took 5-16% off the time to the first token; one band keeps Act on purpose (NOTES)
      M.Out.then(IO(List<&2, Nat>), c, cap, t, M.forward_pre(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        loop(p, c, o, t0, cap, 1n+t, M.step_inv(t, p, cap, inv), q, qt, 1n+lp, j, gen, m2, cs2))
    case 1n++p Nil{} 1n+lp:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- nf is the logits' input, computed anyway; acts is empty on more than one band not-expected-because: forward_gen measured 19% faster per token on 2026-10-16; keeping Act on one band measured faster (NOTES)
      M.Out.then(IO(List<&2, Nat>), c, cap, t, M.forward_gen(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        do IO<List<&2, Nat>>:
          +am : M.AM <- choose(Opts.show(o), Opts.pk(o), lg, j)
          t1 : Nat <- IO.now()
          IO.print("  id " ++ Nat.show(M.AM.i(am)) ++ "  logit " ++ U32.show(bits(M.AM.v(am))) ++ " " ++ F32.show(M.AM.v(am)) ++ "  t=" ++ tsec(t1, t0))
          loop(p, c, o, t0, cap, 1n+t, M.step_inv(t, p, cap, inv), M.AM.i(am), Nil{}, left_chat(Opts.chat(o), M.AM.i(am), lp), 1n+j, M.AM.i(am) <> gen, m2, cs2))

# ---- main program ----

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def show_pick(pk: Pick) -> IO(Unit):
  match pk:
    case Greedy{}:
      IO.pure(Unit, Unit{})
    case Draw{+temp, k, +p, seed}:
      IO.print("sampling: temp " ++ U32.show(bits(temp)) ++ " " ++ F32.show(temp) ++ " top_k " ++ Nat.show(k) ++ " top_p " ++ U32.show(bits(p)) ++ " " ++ F32.show(p) ++ " seed " ++ U32.show(seed))

def start2(+c: M.Cfg, model: M.Model<c>, +o: Opts, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("empty prompt")
    case Con{+t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        show_pick(Opts.pk(o))
        IO.print("prompt ids: " ++ show_nats(ids))
        gen : List<&2, Nat> <- loop(Nat.add(List.length(&2, Nat, ids), n), c, o, t0ms, Nat.add(List.length(&2, Nat, ids), n), 0n, {==}, t0, qt, n, 0n, Nil{}, model, M.caches(c, M.Cfg.layers(c), Nat.add(List.length(&2, Nat, ids), n)))
        IO.print("text: " ++ K.decode_text(table, perm, List.append(&2, Nat, ids, gen)))

# the prompt's ids: its text, or in chat mode one user turn in SmolLM2-Instruct's template
def prompt_ids(chat: Bool, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, prompt: String) -> List<&2, Nat>:
  match chat:
    case True{}:
      K.chat_ids(table, perm, [K.bytes_of(String.to_list("user")), K.utf8_of(String.to_list(prompt))], True{})
    case False{}:
      K.encode_text(table, perm, K.utf8_of(String.to_list(prompt)))

def generate(+c: M.Cfg, model: M.Model<c>, +o: Opts, +ns: List<&2, Nat>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  +table = K.rules.go(ns, 0n)
  +perm = K.nums(pb)
  start2(c, model, o, table, perm, prompt_ids(Opts.chat(o), table, perm, prompt), n)

# refuse to run on a malformed merges table (checked before the weights are loaded)
def run3(ok: Bool, +dir: String, +o: Opts, +ns: List<&2, Nat>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +c : M.Cfg <- M.read_cfg(dir)
        model : M.Model<c> <- L.load_model(c, dir)
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ tsec(t1, t0))
        generate(c, model, o, ns, pb, prompt, n)
    case False{}:
      IO.print("invalid merges table")

def run2(+dir: String, +o: Opts, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  # the merges table is parsed once, for this check and for the rules, which are built once for encoding and decoding
  +ns = K.nums(mb)
  run3(K.ids_ok(ns, 0n), dir, o, ns, pb, prompt, n)

def usage() -> IO(Unit):
  IO.print("usage: gen <dir> [--chat] <prompt> <n_tokens> [<temp> <top_k> <top_p> <seed>] (temp >= 0, top_k >= 1, 0 < top_p <= 1)")

# GEN_CANDS=1: print the candidates of each draw
def cands_env(r: Result<&1, &1, U32 & String, String>) -> Bool:
  match r:
    case Done{v}:
      String.eq(v, "1")
    case Fail{e}:
      False{}

def run1(+dir: String, +chat: Bool, prompt: String, mn: Maybe<&2, Nat>, mp: Maybe<&2, Pick>) -> IO(Unit):
  match mn mp:
    case Some{n} Some{pk}:
      do IO<Unit>:
        r : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_CANDS")
        mb : List<&2, U32> <- K.read_bytes(dir ++ "/merges_num.txt")
        pb : List<&2, U32> <- K.read_bytes(dir ++ "/perm.txt")
        run2(dir, Opts{pk, chat, cands_env(r)}, mb, pb, prompt, n)
    case _ _:
      usage()

# temp 0 is greedy, as transformers' do_sample = False
def pick_sel(ok: Bool, zero: Bool, +temp: F32, +k: Nat, +p: F32, +seed: U32) -> Maybe<&2, Pick>:
  match ok zero:
    case True{} True{}:
      Some{Greedy{}}
    case True{} False{}:
      Some{Draw{temp, k, p, seed}}
    case False{} _:
      None{}

def pick_of(mt: Maybe<&2, F32>, mk: Maybe<&2, Nat>, mp: Maybe<&2, F32>, ms: Maybe<&2, U32>) -> Maybe<&2, Pick>:
  match mt mk mp ms:
    case Some{+temp} Some{+k} Some{+p} Some{+seed}:
      pick_sel(Bool.and(Bool.and(F32.is_ge(temp, 0.0), Nat.is_lt(0n, k)), Bool.and(F32.is_gt(p, 0.0), F32.is_le(p, 1.0))), F32.is_eq(temp, 0.0), temp, k, p, seed)
    case _ _ _ _:
      None{}

def run_chat(is_chat: Bool, +dir: String, prompt: String, mn: Maybe<&2, Nat>, mp: Maybe<&2, Pick>) -> IO(Unit):
  match is_chat:
    case True{}:
      run1(dir, True{}, prompt, mn, mp)
    case False{}:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{d, Con{prompt, Con{n, Nil{}}}}}:
      run1(d, False{}, prompt, Nat.read(n), Some{Greedy{}})
    case Con{prog, Con{d, Con{f, Con{prompt, Con{n, Nil{}}}}}}:
      run_chat(String.eq(f, "--chat"), d, prompt, Nat.read(n), Some{Greedy{}})
    case Con{prog, Con{d, Con{prompt, Con{n, Con{tp, Con{k, Con{p, Con{s, Nil{}}}}}}}}}:
      run1(d, False{}, prompt, Nat.read(n), pick_of(F32.read(tp), Nat.read(k), F32.read(p), U32.read(s)))
    case Con{prog, Con{d, Con{f, Con{prompt, Con{n, Con{tp, Con{k, Con{p, Con{s, Nil{}}}}}}}}}}:
      run_chat(String.eq(f, "--chat"), d, prompt, Nat.read(n), pick_of(F32.read(tp), Nat.read(k), F32.read(p), U32.read(s)))
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
