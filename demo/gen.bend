# Generation with a Llama model in Bend, on the package in llama/ (bend-ml-llama): the prompt is tokenized in Bend
# (tok.bend), runs through the forward pass position by position (main.bend, one key/value cache per layer), and then
# each new token is chosen from the logits and fed back in (gen.bend's loop): the argmax (greedy, the default), or a
# draw with temperature, top-k and top-p from a seed. Generation stops after n tokens, or after the end-of-text token
# (id 0), as transformers' generate does; in chat mode after <|im_end|> (id 2) too. This file is the command line: it
# loads the weights with llama/load.bend and prints each token as the loop hands it over, with its time.
# reference/test_gen.py compares greedy ids, chosen logits and text with transformers; reference/test_sample.py
# checks the draws.
#
#   gen <dir> "prompt" n                              greedy (the model in dir: cfg.txt, w/ and the tokenizer
#                                                     tables, from reference/prep.py)
#   gen <dir> "prompt" n <temp> <top_k> <top_p> <seed>   sampled; temp 0 is greedy, top_k >= 1, 0 < top_p <= 1
#   gen <dir> --chat "message" n [<temp> <top_k> <top_p> <seed>]
#                                                     the message as one user turn in SmolLM2-Instruct's chat
#                                                     template (llama/tok.bend's chat), the default system
#                                                     message first and the assistant's turn opened
#
# prints `prompt ids: ...`, one line per new token (`id <id> logit <F32 bits> <logit> t=<seconds>`) and
# `text: <prompt and continuation>` (special tokens left out); a sampled run first prints `sampling: ...` with the
# parameters as read (F32 bits). With GEN_CANDS=1 in the environment, each draw is preceded by `cands <id>:<logit
# bits> ...`, the top-k candidates largest first (for reference/test_sample.py).

import Base
import ../llama/main.bend as M
import ../llama/tok.bend as K
import ../llama/gen.bend as G
import ../llama/load.bend as L

# ---- printing the tokens ----

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

def show_cands(cs: List<&2, M.Cand>) -> String:
  match cs:
    case Nil{}:
      ""
    case Con{M.Cand{+v, i}, t}:
      " " ++ Nat.show(i) ++ ":" ++ U32.show(bits(v)) ++ show_cands(t)

# how the tokens are chosen (pk), whether this is a chat (chat) and whether draws print their candidates (show)
type Opts is Data:
  Opts{pk: G.Pick, chat: Bool, show: Bool}

def Opts.pk(o: Opts) -> G.Pick:
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

# the top-k candidates of a draw, when show is set (GEN_CANDS); a greedy token has none
def cands(show: Bool, pk: G.Pick, cs: List<&2, M.Cand>) -> IO(Unit):
  match show pk:
    case True{} G.Draw{temp, k, p, seed}:
      IO.print("  cands" ++ show_cands(cs))
    case _ _:
      IO.pure(Unit, Unit{})

# one line per new token, printed when llama/gen.bend's loop hands it over (s), with the time since t0; the next
# forward pass runs after the line is out (rest(m)). n: the tokens asked for; gen: the ids so far, newest first.
# Returns the generated ids in order.
def show_toks(n: Nat, -c: M.Cfg, +o: Opts, +t0: Nat, s: G.Toks<c>, gen: List<&2, Nat>) -> IO(List<&2, Nat>):
  match n s:
    case 1n+p G.Tok{+am, cs, m, rest}:
      do IO<List<&2, Nat>>:
        cands(Opts.show(o), Opts.pk(o), cs)
        t1 : Nat <- IO.now()
        IO.print("  id " ++ Nat.show(M.AM.i(am)) ++ "  logit " ++ U32.show(bits(M.AM.v(am))) ++ " " ++ F32.show(M.AM.v(am)) ++ "  t=" ++ tsec(t1, t0))
        show_toks(p, c, o, t0, rest(m), M.AM.i(am) <> gen)
    case _ _:
      IO.pure(List<&2, Nat>, List.reverse(&2, Nat, gen))

# ---- main program ----

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def show_pick(pk: G.Pick) -> IO(Unit):
  match pk:
    case G.Greedy{}:
      IO.pure(Unit, Unit{})
    case G.Draw{+temp, k, +p, seed}:
      IO.print("sampling: temp " ++ U32.show(bits(temp)) ++ " " ++ F32.show(temp) ++ " top_k " ++ Nat.show(k) ++ " top_p " ++ U32.show(bits(p)) ++ " " ++ F32.show(p) ++ " seed " ++ U32.show(seed))

def start2(+c: M.Cfg, model: M.Model<c>, +o: Opts, tb: K.Tables, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("empty prompt")
    case Con{t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        show_pick(Opts.pk(o))
        IO.print("prompt ids: " ++ show_nats(ids))
        gen : List<&2, Nat> <- show_toks(n, c, o, t0ms, G.tokens(c, model, Opts.pk(o), Opts.chat(o), ids, n), Nil{})
        IO.print("text: " ++ K.decode(tb, List.append(&2, Nat, ids, gen)))

# the prompt's ids: its text, or in chat mode one user turn in SmolLM2-Instruct's template
def prompt_ids(chat: Bool, tb: K.Tables, prompt: String) -> List<&2, Nat>:
  match chat:
    case True{}:
      K.chat(tb, ["user", prompt], True{})
    case False{}:
      K.encode(tb, prompt)

# refuse to run on a malformed merges table (checked before the weights are loaded)
def run2(mt: Maybe<&2, K.Tables>, +dir: String, +o: Opts, prompt: String, n: Nat) -> IO(Unit):
  match mt:
    case Some{+tb}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +c : M.Cfg <- M.read_cfg(dir)
        model : M.Model<c> <- L.load_model(c, dir)
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ tsec(t1, t0))
        start2(c, model, o, tb, prompt_ids(Opts.chat(o), tb, prompt), n)
    case None{}:
      IO.print("invalid merges table")

def usage() -> IO(Unit):
  IO.print("usage: gen <dir> [--chat] <prompt> <n_tokens> [<temp> <top_k> <top_p> <seed>] (temp >= 0, top_k >= 1, 0 < top_p <= 1)")

# GEN_CANDS=1: print the candidates of each draw
def cands_env(r: Result<&1, &1, U32 & String, String>) -> Bool:
  match r:
    case Done{v}:
      String.eq(v, "1")
    case Fail{e}:
      False{}

def run1(+dir: String, +chat: Bool, prompt: String, mn: Maybe<&2, Nat>, mp: Maybe<&2, G.Pick>) -> IO(Unit):
  match mn mp:
    case Some{n} Some{pk}:
      do IO<Unit>:
        r : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_CANDS")
        mt : Maybe<&2, K.Tables> <- K.read_tables(dir ++ "/merges_num.txt", dir ++ "/perm.txt")
        run2(mt, dir, Opts{pk, chat, cands_env(r)}, prompt, n)
    case _ _:
      usage()

# temp 0 is greedy, as transformers' do_sample = False
def pick_sel(ok: Bool, zero: Bool, +temp: F32, +k: Nat, +p: F32, +seed: U32) -> Maybe<&2, G.Pick>:
  match ok zero:
    case True{} True{}:
      Some{G.Greedy{}}
    case True{} False{}:
      Some{G.Draw{temp, k, p, seed}}
    case False{} _:
      None{}

def pick_of(mt: Maybe<&2, F32>, mk: Maybe<&2, Nat>, mp: Maybe<&2, F32>, ms: Maybe<&2, U32>) -> Maybe<&2, G.Pick>:
  match mt mk mp ms:
    case Some{+temp} Some{+k} Some{+p} Some{+seed}:
      pick_sel(Bool.and(Bool.and(F32.is_ge(temp, 0.0), Nat.is_lt(0n, k)), Bool.and(F32.is_gt(p, 0.0), F32.is_le(p, 1.0))), F32.is_eq(temp, 0.0), temp, k, p, seed)
    case _ _ _ _:
      None{}

def run_chat(is_chat: Bool, +dir: String, prompt: String, mn: Maybe<&2, Nat>, mp: Maybe<&2, G.Pick>) -> IO(Unit):
  match is_chat:
    case True{}:
      run1(dir, True{}, prompt, mn, mp)
    case False{}:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{d, Con{prompt, Con{n, Nil{}}}}}:
      run1(d, False{}, prompt, Nat.read(n), Some{G.Greedy{}})
    case Con{prog, Con{d, Con{f, Con{prompt, Con{n, Nil{}}}}}}:
      run_chat(String.eq(f, "--chat"), d, prompt, Nat.read(n), Some{G.Greedy{}})
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
