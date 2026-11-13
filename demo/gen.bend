# Generation with a Llama model in Bend, on the package in llama/ (bend-ml-llama): the prompt is tokenized in Bend
# (tok.bend), runs through the forward pass in one pass (prompt.bend; one key/value cache per layer), and then each
# new token runs through it at its position (main.bend), is chosen from the logits and is fed back in (gen.bend's
# loop): the argmax (greedy, the default), or a
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
#   gen <dir> --chat - n [<temp> <top_k> <top_p> <seed>]
#                                                     a conversation: every line of standard input is a user turn,
#                                                     answered with n tokens at most. The caches stay between the
#                                                     turns (llama/gen.bend's turn), so a turn runs its own ids and
#                                                     nothing before them. It ends with the input, or when the
#                                                     caches are full (GEN_CTX positions, 2048 by default)
#
# prints `prompt ids: ...`, one line per new token (`id <id> logit <F32 bits> <logit> t=<seconds>`) and
# `text: <prompt and continuation>` (special tokens left out); a sampled run first prints `sampling: ...` with the
# parameters as read (F32 bits). With GEN_CANDS=1 in the environment, each draw is preceded by `cands <id>:<logit
# bits> ...`, the top-k candidates largest first (for reference/test_sample.py). With GEN_SEQ=1 the prompt runs
# position by position, the path before prompt.bend, which gives the same output (reference/test_prompt.py). With
# GEN_IDS=1 the prompt (or the chat message) is a list of ids, run as they are.
#
# A conversation prints `> ` before it reads a turn, then the answer's text as its tokens come, and a line in
# parentheses when the answer did not end by itself. With GEN_TRACE=1 it prints what the tests read instead: `turn
# ids: ...` (the ids the turn adds after the last answer), the line of each new token as above, `text: <answer>` and
# `over: <why> room <positions left>` (reference/test_chat_turns.py).

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

# how the tokens are chosen (pk), whether this is a chat (chat), whether draws print their candidates (show),
# whether the prompt runs position by position (seq), whether the prompt is a list of ids (ids), whether a
# conversation prints the tests' lines (trace), and the positions a conversation's caches have room for (ctx)
type Opts is Data:
  Opts{pk: G.Pick, chat: Bool, show: Bool, seq: Bool, ids: Bool, trace: Bool, ctx: Nat}

def Opts.pk(o: Opts) -> G.Pick:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      pk

def Opts.chat(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      chat

def Opts.show(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      show

def Opts.seq(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      seq

def Opts.ids(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      ids

def Opts.trace(o: Opts) -> Bool:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      trace

def Opts.ctx(o: Opts) -> Nat:
  match o:
    case Opts{pk, chat, show, seq, ids, trace, ctx}:
      ctx

# the new tokens after the prompt: the prompt in one pass (llama/prompt.bend), or position by position when seq is
# set (GEN_SEQ=1), which reference/test_prompt.py compares with it
def toks(seq: Bool, +c: M.Cfg, model: M.Model<c>, +pk: G.Pick, +chat: Bool, +ids: List<&2, Nat>, +n: Nat) -> G.Toks<c>:
  match seq:
    case True{}:
      G.tokens.seq(c, model, pk, chat, ids, n)
    case False{}:
      G.tokens(c, model, pk, chat, ids, n)

# the top-k candidates of a draw, when show is set (GEN_CANDS); a greedy token has none
def cands(show: Bool, pk: G.Pick, cs: List<&2, M.Cand>) -> IO(Unit):
  match show pk:
    case True{} G.Draw{temp, k, p, seed}:
      IO.print("  cands" ++ show_cands(cs))
    case _ _:
      IO.pure(Unit, Unit{})

# a new token's line: its candidates when asked for, its id, its logit (bits and value) and the time since t0
def tok_line(+o: Opts, +t0: Nat, +am: M.AM, cs: List<&2, M.Cand>) -> IO(Unit):
  do IO<Unit>:
    cands(Opts.show(o), Opts.pk(o), cs)
    t1 : Nat <- IO.now()
    IO.print("  id " ++ Nat.show(M.AM.i(am)) ++ "  logit " ++ U32.show(bits(M.AM.v(am))) ++ " " ++ F32.show(M.AM.v(am)) ++ "  t=" ++ tsec(t1, t0))

# one line per new token, printed when llama/gen.bend's loop hands it over (s), with the time since t0; the next
# forward pass runs after the line is out (rest(m)). n: the tokens asked for; gen: the ids so far, newest first.
# Returns the generated ids in order.
def show_toks(n: Nat, -c: M.Cfg, +o: Opts, +t0: Nat, s: G.Toks<c>, gen: List<&2, Nat>) -> IO(List<&2, Nat>):
  match n s:
    case 1n+p G.Tok{+am, cs, m, rest}:
      do IO<List<&2, Nat>>:
        tok_line(o, t0, am, cs)
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
        gen : List<&2, Nat> <- show_toks(n, c, o, t0ms, toks(Opts.seq(o), c, model, Opts.pk(o), Opts.chat(o), ids, n), Nil{})
        IO.print("text: " ++ K.decode(tb, List.append(&2, Nat, ids, gen)))

# the prompt's ids: its text, or in chat mode one user turn in SmolLM2-Instruct's template; with ids set (GEN_IDS=1)
# the numbers written in it, as they are
def prompt_ids(ids: Bool, chat: Bool, tb: K.Tables, prompt: String) -> List<&2, Nat>:
  match ids chat:
    case True{} _:
      K.nums(K.utf8_of(String.to_list(prompt)))
    case False{} True{}:
      K.chat(tb, ["user", prompt], True{})
    case False{} False{}:
      K.encode(tb, prompt)

# ---- a conversation from standard input ----

# what a turn leaves: why the answer ended, the model, the positions left, the conversation, the answer's ids
type After<-c: M.Cfg, -cap: Nat> is Type:
  After{why: G.Why, m: M.Model<c>, room: Nat, chat: G.Chat<c, cap>, gen: List<&2, Nat>}

def After.then(-c: M.Cfg, -cap: Nat, a: After<c, cap>, k: G.Why -> M.Model<c> -> Nat -> G.Chat<c, cap> -> List<&2, Nat> -> IO(Unit)) -> IO(Unit):
  match a:
    case After{why, m, room, chat, gen}:
      k(why, m, room, chat, gen)

# hold: the text ends in U+FFFD, which is how an unfinished UTF-8 sequence decodes, so the next token may complete
# it and nothing is written yet. Returns the characters on the screen
def piece.out(hold: Bool, +text: String, +shown: Nat) -> IO(Nat):
  match hold:
    case True{}:
      IO.pure(Nat, shown)
    case False{}:
      do IO<Nat>:
        IO.write(String.drop(text, shown))
        # a sleep makes the runtime flush what was written before the next token is computed
        IO.sleep(0)
        return String.length(text)

# the answer's text so far, of which shown characters are on the screen: the rest is written now
def piece(+text: String, +shown: Nat) -> IO(Nat):
  piece.out(String.ends_with(text, "�"), text, shown)

# a new token on the screen: its line (trace), or what it adds to the answer's text. gen: the answer's ids with the
# new one, newest first
def tok_out(trace: Bool, +o: Opts, +tb: K.Tables, +t0: Nat, +am: M.AM, cs: List<&2, M.Cand>, gen: List<&2, Nat>, +shown: Nat) -> IO(Nat):
  match trace:
    case True{}:
      do IO<Nat>:
        tok_line(o, t0, am, cs)
        return shown
    case False{}:
      piece(K.decode(tb, List.reverse(&2, Nat, gen)), shown)

# the end of an answer on the screen: `text: <answer>` (trace), or what is left of its text and the end of the line
def end_out(trace: Bool, text: String, shown: Nat) -> IO(Unit):
  match trace:
    case True{}:
      IO.print("text: " ++ text)
    case False{}:
      IO.print(String.drop(text, shown))

# an answer, token by token as llama/gen.bend's turn hands it over (s); fuel: one more than the tokens asked for;
# gen: the ids so far, newest first; shown: the characters of the answer on the screen
def say(fuel: Nat, -c: M.Cfg, -cap: Nat, +o: Opts, +tb: K.Tables, +t0: Nat, s: G.Turn<c, cap>, +gen: List<&2, Nat>, +shown: Nat) -> IO(After<c, cap>):
  match fuel s:
    case 1n+p G.Said{+am, cs, m, rest}:
      do IO<After<c, cap>>:
        sh : Nat <- tok_out(Opts.trace(o), o, tb, t0, am, cs, M.AM.i(am) <> gen, shown)
        say(p, c, cap, o, tb, t0, rest(m), M.AM.i(am) <> gen, sh)
    case _ G.Over{why, m, room, chat}:
      do IO<After<c, cap>>:
        end_out(Opts.trace(o), K.decode(tb, List.reverse(&2, Nat, gen)), shown)
        return After{why, m, room, chat, List.reverse(&2, Nat, gen)}
    case 0n G.Said{am, cs, m, rest}:
      IO.die(After<c, cap>, 1, "gen: an answer of more tokens than asked for")

def why_name(why: G.Why) -> String:
  match why:
    case G.Stop{}:
      "stop"
    case G.Count{}:
      "count"
    case G.Full{}:
      "full"
    case G.Unfit{need}:
      "unfit " ++ Nat.show(need)
    case G.Idle{}:
      "idle"

# what a person reads when an answer did not end by itself; a full conversation ends the program
def why_text(why: G.Why, +n: Nat, +room: Nat, +cap: Nat) -> IO(Unit):
  match why:
    case G.Stop{}:
      IO.pure(Unit, Unit{})
    case G.Count{}:
      IO.print("(stopped at " ++ Nat.show(n) ++ " tokens)")
    case G.Full{}:
      do IO<Unit>:
        IO.print("(the conversation ends here: its " ++ Nat.show(cap) ++ " positions are full)")
        IO.exit(0)
    case G.Unfit{need}:
      IO.print("(no room for this turn: it needs " ++ Nat.show(need) ++ " positions and " ++ Nat.show(room) ++ " of " ++ Nat.show(cap) ++ " are left)")
    case G.Idle{}:
      IO.pure(Unit, Unit{})

def why_out(trace: Bool, why: G.Why, +n: Nat, +room: Nat, +cap: Nat) -> IO(Unit):
  match trace:
    case True{}:
      IO.print("over: " ++ why_name(why) ++ " room " ++ Nat.show(room))
    case False{}:
      why_text(why, n, room, cap)

# whether a turn left the conversation as it was: its ids did not fit, or it asked for nothing
def idle(why: G.Why) -> Bool:
  match why:
    case G.Unfit{need}:
      True{}
    case G.Idle{}:
      True{}
    case _:
      False{}

# before a turn is read: `> ` for a person
def ask(trace: Bool) -> IO(Unit):
  match trace:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      do IO<Unit>:
        IO.write("> ")
        IO.sleep(0)

# the ids a turn adds, for the tests
def ids_out(trace: Bool, ids: List<&2, Nat>) -> IO(Unit):
  match trace:
    case True{}:
      IO.print("turn ids: " ++ show_nats(ids))
    case False{}:
      IO.pure(Unit, Unit{})

# the ids of a turn: the first one opens the conversation in the chat template; the others continue it after an
# answer
def turn_ids(first: Bool, tb: K.Tables, msg: String) -> List<&2, Nat>:
  match first:
    case True{}:
      K.chat(tb, ["user", msg], True{})
    case False{}:
      K.chat_next(tb, msg)

# cs without its first character, when that is a line feed or a carriage return
def chomp1(cs: List<&2, Char>) -> List<&2, Char>:
  match cs:
    case Con{Chr{+c}, +t}:
      Bool.pick(List<&2, Char>, Bool.or(U32.is_eq(c, 10), U32.is_eq(c, 13)), t, Chr{c} <> t)
    case Nil{}:
      Nil{}

# a line without the end of line that File.read_line leaves on it (\n, or \r\n)
def chomp(s: String) -> String:
  String.from_list(List.reverse(&2, Char, chomp1(chomp1(List.reverse(&2, Char, String.to_list(s))))))

# empty: nothing was read, which is the end of the input
def line_end(empty: Bool, f: File, line: String, eof: File -> IO(Unit), k: File -> String -> IO(Unit)) -> IO(Unit):
  match empty:
    case True{}:
      eof(f)
    case False{}:
      k(f, chomp(line))

def line_res(f: File, res: Result<&1, &1, U32 & String, String>, eof: File -> IO(Unit), k: File -> String -> IO(Unit)) -> IO(Unit):
  match res:
    case Done{+line}:
      line_end(String.is_empty(line), f, line, eof, k)
    case Fail{e}:
      eof(f)

# continues with the file and the line read from it (k), or at the end of the input with the file (eof)
def line_then(r: File & Result<&1, &1, U32 & String, String>, eof: File -> IO(Unit), k: File -> String -> IO(Unit)) -> IO(Unit):
  match r:
    case Tuple{f, res}:
      line_res(f, res, eof, k)

# The conversation: a line of f is a user turn, answered with n tokens at most on top of the caches the turns before
# it left (ch); first: no turn has run yet, so the next one opens the conversation; turns bounds the number of turns
def talk(turns: Nat, +c: M.Cfg, +cap: Nat, +o: Opts, +tb: K.Tables, +n: Nat, +first: Bool, f: File, ch: G.Chat<c, cap>, m: M.Model<c>) -> IO(Unit):
  match turns:
    case 0n:
      File.close(f)
    case 1n+p:
      do IO<Unit>:
        ask(Opts.trace(o))
        r : File & Result<&1, &1, U32 & String, String> <- File.read_line(f, 1048576)
        line_then(r, f2 => File.close(f2), f2 => line =>
          do IO<Unit>:
            t0 : Nat <- IO.now()
            +ids : List<&2, Nat> = turn_ids(first, tb, line)
            ids_out(Opts.trace(o), ids)
            a : After<c, cap> <- say(1n+n, c, cap, o, tb, t0, G.turn(c, Opts.pk(o), cap, ch, m, ids, n), Nil{}, 0n)
            After.then(c, cap, a, +why => m2 => +room => ch2 => gen =>
              do IO<Unit>:
                why_out(Opts.trace(o), why, n, room, cap)
                talk(p, c, cap, o, tb, n, Bool.and(first, idle(why)), f2, ch2, m2)))

def talk_open(r: Result<&1, &1, U32 & String, File>, +c: M.Cfg, +o: Opts, +tb: K.Tables, +n: Nat, m: M.Model<c>) -> IO(Unit):
  match r:
    case Done{f}:
      talk(1000000n, c, Opts.ctx(o), o, tb, n, True{}, f, G.Chat.start(c, Opts.ctx(o)), m)
    case Fail{e}:
      IO.print("gen: cannot read the standard input")

# conv: a conversation from the standard input (the message is "-" in chat mode); otherwise one prompt
def start(conv: Bool, +c: M.Cfg, model: M.Model<c>, +o: Opts, +tb: K.Tables, prompt: String, +n: Nat) -> IO(Unit):
  match conv:
    case True{}:
      do IO<Unit>:
        show_pick(Opts.pk(o))
        r : Result<&1, &1, U32 & String, File> <- File.open("/dev/stdin", "r")
        talk_open(r, c, o, tb, n, model)
    case False{}:
      start2(c, model, o, tb, prompt_ids(Opts.ids(o), Opts.chat(o), tb, prompt), n)

# refuse to run on a malformed merges table (checked before the weights are loaded)
def run2(mt: Maybe<&2, K.Tables>, +dir: String, +o: Opts, +prompt: String, +n: Nat) -> IO(Unit):
  match mt:
    case Some{+tb}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +c : M.Cfg <- M.read_cfg(dir)
        model : M.Model<c> <- L.load_model(c, dir)
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ tsec(t1, t0))
        start(Bool.and(Opts.chat(o), String.eq(prompt, "-")), c, model, o, tb, prompt, n)
    case None{}:
      IO.print("invalid merges table")

def usage() -> IO(Unit):
  IO.print("usage: gen <dir> [--chat] <prompt> <n_tokens> [<temp> <top_k> <top_p> <seed>] (temp >= 0, top_k >= 1, 0 < top_p <= 1; --chat - reads the turns of a conversation from the standard input)")

# whether a variable of the environment is "1": GEN_CANDS=1 prints the candidates of each draw, GEN_SEQ=1 runs the
# prompt position by position, GEN_IDS=1 reads the prompt as ids, GEN_TRACE=1 prints a conversation's lines for the
# tests
def cands_env(r: Result<&1, &1, U32 & String, String>) -> Bool:
  match r:
    case Done{v}:
      String.eq(v, "1")
    case Fail{e}:
      False{}

# the positions a conversation's caches have room for: GEN_CTX when it is a number above 0, else 2048
def ctx_of(m: Maybe<&2, Nat>) -> Nat:
  match m:
    case Some{+n}:
      Bool.pick(Nat, Nat.is_eq(n, 0n), 2048n, n)
    case None{}:
      2048n

def ctx_env(r: Result<&1, &1, U32 & String, String>) -> Nat:
  match r:
    case Done{v}:
      ctx_of(Nat.read(v))
    case Fail{e}:
      2048n

def run1(+dir: String, +chat: Bool, prompt: String, mn: Maybe<&2, Nat>, mp: Maybe<&2, G.Pick>) -> IO(Unit):
  match mn mp:
    case Some{n} Some{pk}:
      do IO<Unit>:
        r : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_CANDS")
        q : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_SEQ")
        i : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_IDS")
        t : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_TRACE")
        x : Result<&1, &1, U32 & String, String> <- IO.get_env("GEN_CTX")
        mt : Maybe<&2, K.Tables> <- K.read_tables(dir ++ "/merges_num.txt", dir ++ "/perm.txt")
        run2(mt, dir, Opts{pk, chat, cands_env(r), cands_env(q), cands_env(i), cands_env(t), ctx_env(x)}, prompt, n)
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
