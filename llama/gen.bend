# bend-ml-llama, generation: the ids of a prompt run through main.bend's forward pass position by position (one
# key/value cache per layer), and then each new token is chosen from the logits and fed back in: the argmax (greedy),
# or a draw with temperature, top-k and top-p from a seed (main.bend's sampler). Generation stops after n tokens, or
# after the end-of-text token (id 0), as transformers' generate does; in a chat after <|im_end|> (id 2) too.
#
#   import ./main.bend as M
#   import ./gen.bend as G
#   G.generate(c, model, G.Greedy{}, False{}, ids, n)                  # G.Gen{toks, m}: the new tokens and the model
#   G.tokens(c, model, G.Draw{0.8, 50n, 0.95, 42}, False{}, ids, n)    # the same loop, one token at a time (G.Toks)
#   G.turn(c, G.Greedy{}, cap, G.Chat.start(c, cap), model, ids, n)    # a conversation's next answer (G.Turn); it
#                                                                      # hands the conversation back, caches and all
#
# The model is a parameter, so the caller chooses its loader. Nothing here prints or reads a clock: a program that
# shows each token as it comes walks G.Toks, as llama.bend's demo/gen.bend does.
#
# A conversation (the last section) keeps its caches between the turns: a turn's ids go through the layers in one
# pass on top of the positions already there, and the answer is generated after them, so no turn runs the ones before
# it again. Its answers are the tokens that `tokens` gives for the whole conversation up to each of them.

import Base
import bend-ml-tensor-array@0.1.20.0/main.bend as TA
import ./main.bend as M
import ./prompt.bend as P

# how each new token is chosen: the argmax (greedy, transformers' do_sample = False), or a draw (main.bend's sample)
# with temperature temp > 0, the k >= 1 largest logits, top-p p in (0, 1], and draw j of the seed for the j-th new
# token
type Pick is Data:
  Greedy{}
  Draw{temp: F32, k: Nat, p: F32, seed: U32}

# a chosen token: its id and logit (am), and the candidates it was drawn from, largest first (cs; none when greedy)
type Chosen is Data:
  Chosen{am: M.AM, cs: List<&2, M.Cand>}

def Chosen.am(x: Chosen) -> M.AM:
  match x:
    case Chosen{am, cs}:
      am

def Chosen.cs(x: Chosen) -> List<&2, M.Cand>:
  match x:
    case Chosen{am, cs}:
      cs

def drawn(cs: List<&2, M.Cand>, s: M.Cand) -> Chosen:
  match s:
    case M.Cand{v, i}:
      Chosen{M.AM{i, v}, cs}

# the j-th new token, from the logits lg. Greedy is the argmax alone. A draw is main.bend's sample in its two halves
# (topk, then sample.cs), so that the candidates are computed once and handed out with the token.
def choose(pk: Pick, lg: List<&2, F32>, +j: Nat) -> Chosen:
  match pk:
    case Greedy{}:
      Chosen{M.argmax(lg), Nil{}}
    case Draw{+temp, +k, +p, +seed}:
      +cs = M.topk(k, lg)
      drawn(cs, M.sample.cs(cs, temp, M.top_of(cs), p, M.uniform(seed, U32.from_nat(j))))

# the tokens left to generate after the token id: none after the end-of-text token (id 0)
def left_after(+id: Nat, +left: Nat) -> Nat:
  Bool.pick(Nat, Nat.is_eq(id, 0n), 0n, left)

# the same in a chat, where nothing more is generated after <|im_end|> (id 2) either
def left_chat(chat: Bool, +id: Nat, +left: Nat) -> Nat:
  match chat:
    case True{}:
      Bool.pick(Nat, Nat.is_eq(id, 2n), 0n, left_after(id, left))
    case False{}:
      left_after(id, left)

# The new tokens, one at a time, as a generator yields them. Tok: a new token (am: its id and logit; cs: the
# candidates of a draw), the model handed back, and the rest of the generation, which runs when it is given that model
# again (rest(m)): the next forward pass happens then, not before. End: no more tokens, and the model. A program may
# stop at any Tok and keep the model; the caches go with the rest it does not ask for. (A closure can be called once
# in Bend, so a callback per token is not an option: the caller walks the tokens instead.)
type Toks<-c: M.Cfg> is Type:
  End{m: M.Model<c>}
  Tok{am: M.AM, cs: List<&2, M.Cand>, m: M.Model<c>, rest: M.Model<c> -> Toks<c>}

# tok runs at position t; queue: the prompt ids still to feed; left: tokens still to generate, j generated so far.
# The caches have room for cap positions, and fuel counts the steps left with t + fuel = cap (inv), which gives each
# step its proof that t < cap. The forward pass is matched by Out.then, so the loop recurses inside its closure.
def loop(fuel: Nat, +c: M.Cfg, +pk: Pick, +chat: Bool, -cap: Nat, +t: Nat, -inv: {Nat.add(t, fuel) == cap : Nat}, +tok: Nat, queue: List<&2, Nat>, left: Nat, +j: Nat, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> Toks<c>:
  match fuel queue left:
    case 0n _ _:
      End{m}
    case 1n+p _ 0n:
      End{m}
    case 1n++p Con{+q, qt} 1n+lp:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- forward_pre returns lg and nf empty, and acts empty not-expected-because: measured 2026-10-16: forward_pre took 5-16% off the time to the first token; it leaves Act empty on one band too since 2026-11-12 (NOTES)
      M.Out.then(Toks<c>, c, cap, t, M.forward_pre(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        loop(p, c, pk, chat, cap, 1n+t, M.step_inv(t, p, cap, inv), q, qt, 1n+lp, j, m2, cs2))
    case 1n++p Nil{} 1n+lp:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- nf is the logits' input, computed anyway; acts is empty not-expected-because: forward_gen measured 19% faster per token on 16 threads on 2026-10-16, and leaves Act empty on one band too since 2026-11-12 (NOTES)
      M.Out.then(Toks<c>, c, cap, t, M.forward_gen(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        +x = choose(pk, lg, j)
        +id = M.AM.i(Chosen.am(x))
        Tok{Chosen.am(x), Chosen.cs(x), m2, m3 => loop(p, c, pk, chat, cap, 1n+t, M.step_inv(t, p, cap, inv), id, Nil{}, left_chat(chat, id, lp), 1n+j, m3, cs2)})

# the prompt's positions one after another (main.bend's forward_pre), then the new tokens: the path of a prompt of
# one id, where there is nothing to share
def tokens.seq(+c: M.Cfg, m: M.Model<c>, +pk: Pick, +chat: Bool, +ids: List<&2, Nat>, +n: Nat) -> Toks<c>:
  match ids:
    case Nil{}:
      End{m}
    case Con{+t0, qt}:
      loop(Nat.add(List.length(&2, Nat, ids), n), c, pk, chat, Nat.add(List.length(&2, Nat, ids), n), 0n, {==}, t0, qt, n, 0n, m, M.caches(c, M.Cfg.layers(c), Nat.add(List.length(&2, Nat, ids), n)))

# after a prompt of plen ids that ran in one pass (prompt.bend): its last position's logits choose the first new
# token, and the loop goes on at position plen with the caches the pass grew; 1 + lp tokens were asked for
def after(+c: M.Cfg, +pk: Pick, +chat: Bool, -cap: Nat, +plen: Nat, +lp: Nat, -inv: {Nat.add(plen, 1n+lp) == cap : Nat}, o: P.OutP<c, cap, plen>) -> Toks<c>:
  P.OutP.then(Toks<c>, c, cap, plen, o, lg => cs => m =>
    +x = choose(pk, lg, 0n)
    +id = M.AM.i(Chosen.am(x))
    Tok{Chosen.am(x), Chosen.cs(x), m, m2 => loop(1n+lp, c, pk, chat, cap, plen, inv, id, Nil{}, left_chat(chat, id, lp), 1n, m2, cs)})

# pp: the prompt's ids but one. One id runs position by position; two or more go through each layer together, with
# room for the prompt and the 1 + lp new tokens
def tokens.pick(pp: Nat, +lp: Nat, +c: M.Cfg, m: M.Model<c>, +pk: Pick, +chat: Bool, +ids: List<&2, Nat>) -> Toks<c>:
  match pp:
    case 0n:
      tokens.seq(c, m, pk, chat, ids, 1n+lp)
    case 1n++q:
      after(c, pk, chat, Nat.add(2n+q, 1n+lp), 2n+q, lp, {==}, P.forward_prompt(c, Nat.add(2n+q, 1n+lp), 0n, 2n+q, 2n+q, {==}, TA.leb_up(2n+q, 2n+q, 1n+lp, TA.leb_refl(2n+q)), m, M.caches(c, M.Cfg.layers(c), Nat.add(2n+q, 1n+lp)), ids))

# n: the tokens asked for; plen: the prompt's ids. No token asked for, or an empty prompt, gives none (a Llama needs
# one position to start from)
def tokens.by(n: Nat, plen: Nat, +c: M.Cfg, m: M.Model<c>, +pk: Pick, +chat: Bool, +ids: List<&2, Nat>) -> Toks<c>:
  match n plen:
    case 0n _:
      End{m}
    case 1n+lp 0n:
      End{m}
    case 1n+lp 1n+pp:
      tokens.pick(pp, lp, c, m, pk, chat, ids)

# the new tokens after the prompt ids, n at most, one at a time: the prompt runs first, then the first token is
# chosen
def tokens(+c: M.Cfg, m: M.Model<c>, +pk: Pick, +chat: Bool, +ids: List<&2, Nat>, +n: Nat) -> Toks<c>:
  tokens.by(n, List.length(&2, Nat, ids), c, m, pk, chat, ids)

# the new tokens (the id and logit of each, in order) and the model handed back
type Gen<-c: M.Cfg> is Type:
  Gen{toks: List<&2, M.AM>, m: M.Model<c>}

# continues with the tokens and the model
def Gen.then(-R: Type, -c: M.Cfg, g: Gen<c>, k: List<&2, M.AM> -> M.Model<c> -> R) -> R:
  match g:
    case Gen{toks, m}:
      k(toks, m)

# the tokens alone (the model is dropped)
def Gen.toks(-c: M.Cfg, g: Gen<c>) -> List<&2, M.AM>:
  match g:
    case Gen{toks, m}:
      toks

# acc: the tokens taken so far, newest first. The n-th token is taken without asking for the rest.
def take.go(n: Nat, -c: M.Cfg, s: Toks<c>, acc: List<&2, M.AM>) -> Gen<c>:
  match n s:
    case 1n Tok{am, cs, m, rest}:
      Gen{List.reverse(&2, M.AM, am <> acc), m}
    case 1n+p Tok{am, cs, m, rest}:
      take.go(p, c, rest(m), am <> acc)
    case 0n Tok{am, cs, m, rest}:
      Gen{List.reverse(&2, M.AM, acc), m}
    case _ End{m}:
      Gen{List.reverse(&2, M.AM, acc), m}

# the first n tokens of s, and the model: all of s if it ends by then, and no forward pass after the n-th
def take(n: Nat, -c: M.Cfg, s: Toks<c>) -> Gen<c>:
  take.go(n, c, s, Nil{})

# n new tokens after the prompt ids, all at once (fewer after a stop token)
def generate(+c: M.Cfg, m: M.Model<c>, +pk: Pick, +chat: Bool, +ids: List<&2, Nat>, +n: Nat) -> Gen<c>:
  take(n, c, tokens(c, m, pk, chat, ids, n))

# the ids of tokens
def ids_of(toks: List<&2, M.AM>) -> List<&2, Nat>:
  match toks:
    case Nil{}:
      Nil{}
    case Con{M.AM{i, v}, t}:
      i <> ids_of(t)

# =====================================================================================================
# A conversation that keeps its caches
# =====================================================================================================

# k positions out of room: what is left, with the equation that says so (room = k + rest), or Short when k is more
# than the room. The equation is what turns a count made at run time into the proofs the caches ask for
type Fit<-k: Nat, -room: Nat> is Type:
  Fits{rest: Nat, e: {Nat.add(k, rest) == room : Nat}}
  Short{}

# a = b gives 1 + a = 1 + b
def succ_eq(-a: Nat, -b: Nat, e: {a == b : Nat}) -> {1n+a == 1n+b : Nat}:
  %e : {1n+a == 1n+_ : Nat}
  {==}

# one more on both sides: when k out of room leave rest, 1 + k out of 1 + room leave the same rest
def fit.up(-p: Nat, -q: Nat, f: Fit<p, q>) -> Fit<1n+p, 1n+q>:
  match f:
    case Fits{rest, e}:
      Fits{rest, succ_eq(Nat.add(p, rest), q, e)}
    case Short{}:
      Short{}

# k and room are counted down together. When k ends first, what is left of the room is the rest, and 0 + rest = rest
# holds by computing; when the room ends first, k was more than the room. On the way back every step adds one to
# both sides of the equation (fit.up)
def fit(k: Nat, room: Nat) -> Fit<k, room>:
  match k room:
    case 0n _:
      Fits{room, {==}}
    case 1n+p 0n:
      Short{}
    case 1n+p 1n+q:
      fit.up(p, q, fit(p, q))

# t + room = cap and k + rest = room give (t + k) + rest = cap: after k more positions the room is rest. Proof: cap
# is t + room (inv) and room is k + rest (e), so the claim is (t + k) + rest = t + (k + rest), which is add_assoc
def room_inv(+t: Nat, -k: Nat, -rest: Nat, -room: Nat, -cap: Nat, e: {Nat.add(k, rest) == room : Nat}, inv: {Nat.add(t, room) == cap : Nat}) -> {Nat.add(Nat.add(t, k), rest) == cap : Nat}:
  %inv : {Nat.add(Nat.add(t, k), rest) == _ : Nat}
  %e : {Nat.add(Nat.add(t, k), rest) == Nat.add(t, _) : Nat}
  M.add_assoc(t, k, rest)

# and so the k positions fit: t + k <= cap. Proof: cap is (t + k) + rest (room_inv), and a number is at most itself
# plus another (tensor-array's leb_up over leb_refl)
def room_le(+t: Nat, +k: Nat, +rest: Nat, -room: Nat, -cap: Nat, e: {Nat.add(k, rest) == room : Nat}, inv: {Nat.add(t, room) == cap : Nat}) -> {True{} == TA.leb(Nat.add(t, k), cap) : Bool}:
  %room_inv(t, k, rest, room, cap, e, inv) : {True{} == TA.leb(Nat.add(t, k), _) : Bool}
  TA.leb_up(Nat.add(t, k), Nat.add(t, k), rest, TA.leb_refl(Nat.add(t, k)))

# A conversation between two answers: its t positions are in the caches (one per layer, with room for cap), and pend
# is the last answer's last token when it still has to go through the model: it does not after a stop token, and
# there is none before the first turn
type Chat<-c: M.Cfg, -cap: Nat> is Type:
  Chat{t: Nat, pend: Maybe<&2, Nat>, cs: List<&1, M.KV<cap, t>>}

# a conversation with no turn yet: empty caches with room for cap positions
def Chat.start(+c: M.Cfg, +cap: Nat) -> Chat<c, cap>:
  Chat{0n, None{}, M.caches(c, M.Cfg.layers(c), cap)}

# why an answer ended. Stop: after <|im_end|> (id 2) or the end-of-text token (id 0). Count: the tokens asked for
# were generated. Full: the caches are full, so the last token has no position to run at. Unfit: nothing ran, since
# the turn's ids (need of them, the pending token among them) are more than the room. Idle: nothing ran, since the
# turn had no id or no token was asked for
type Why is Data:
  Stop{}
  Count{}
  Full{}
  Unfit{need: Nat}
  Idle{}

# An answer, one token at a time as Toks hands them: Said is a new token, the model handed back and the rest of the
# answer, which runs when it is given that model again. Over: no more tokens, why, the model, the positions the
# caches still have room for, and the conversation for the next turn
type Turn<-c: M.Cfg, -cap: Nat> is Type:
  Said{am: M.AM, cs: List<&2, M.Cand>, m: M.Model<c>, rest: M.Model<c> -> Turn<c, cap>}
  Over{why: Why, m: M.Model<c>, room: Nat, chat: Chat<c, cap>}

def is_stop(+id: Nat) -> Bool:
  Bool.or(Nat.is_eq(id, 0n), Nat.is_eq(id, 2n))

# an answer's last token as the conversation keeps it: a stop token is not kept
def pend_of(+tok: Nat) -> Maybe<&2, Nat>:
  Bool.pick(Maybe<&2, Nat>, is_stop(tok), None{}, Some{tok})

# tok: the answer's last token so far, which runs at position t when another is asked for (left of them; j so far).
# room counts the positions left, with t + room = cap (inv), as loop's fuel does. The answer is over when no token
# is left to generate, or when no position is left to run tok at
def turn.go(room: Nat, +c: M.Cfg, +pk: Pick, -cap: Nat, +t: Nat, -inv: {Nat.add(t, room) == cap : Nat}, +tok: Nat, left: Nat, +j: Nat, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> Turn<c, cap>:
  match room left:
    case _ 0n:
      Over{Bool.pick(Why, is_stop(tok), Stop{}, Count{}), m, room, Chat{t, pend_of(tok), cs}}
    case 0n 1n+lp:
      Over{Full{}, m, 0n, Chat{t, pend_of(tok), cs}}
    case 1n++p 1n+lp:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- nf is the logits' input, computed anyway; acts is empty not-expected-because: as in loop above
      M.Out.then(Turn<c, cap>, c, cap, t, M.forward_gen(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        +x = choose(pk, lg, j)
        +id = M.AM.i(Chosen.am(x))
        Said{Chosen.am(x), Chosen.cs(x), m2, m3 => turn.go(p, c, pk, cap, 1n+t, M.step_inv(t, p, cap, inv), id, left_chat(True{}, id, lp), 1n+j, m3, cs2)})

# after the turn's ids ran in one pass: the last position's logits choose the answer's first token, and the answer
# goes on at position t with the caches the pass grew; 1 + lp tokens were asked for
def turn.first(+c: M.Cfg, +pk: Pick, -cap: Nat, +t: Nat, +room: Nat, -inv: {Nat.add(t, room) == cap : Nat}, +lp: Nat, o: P.OutP<c, cap, t>) -> Turn<c, cap>:
  P.OutP.then(Turn<c, cap>, c, cap, t, o, lg => cs => m =>
    +x = choose(pk, lg, 0n)
    +id = M.AM.i(Chosen.am(x))
    Said{Chosen.am(x), Chosen.cs(x), m, m2 => turn.go(room, c, pk, cap, t, inv, id, left_chat(True{}, id, lp), 1n, m2, cs)})

# ids: the k ids the turn feeds, at positions t .. t + k - 1. When they fit in the room (f), they go through each
# layer together (prompt.bend); when they do not, nothing runs and the conversation is handed back as it was
def turn.fit(+c: M.Cfg, +pk: Pick, -cap: Nat, +t: Nat, +k: Nat, +room: Nat, f: Fit<k, room>, -inv: {Nat.add(t, room) == cap : Nat}, +pend: Maybe<&2, Nat>, +ids: List<&2, Nat>, +lp: Nat, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> Turn<c, cap>:
  match f:
    case Short{}:
      Over{Unfit{k}, m, room, Chat{t, pend, cs}}
    case Fits{+rest, e}:
      turn.first(c, pk, cap, Nat.add(t, k), rest, room_inv(t, k, rest, room, cap, e, inv), lp, P.forward_prompt(c, cap, t, k, Nat.add(t, k), {==}, room_le(t, k, rest, room, cap, e, inv), m, cs, ids))

# n: the tokens asked for; k: the ids the turn feeds
def turn.by(n: Nat, k: Nat, +c: M.Cfg, +pk: Pick, -cap: Nat, +t: Nat, +room: Nat, -inv: {Nat.add(t, room) == cap : Nat}, +pend: Maybe<&2, Nat>, +ids: List<&2, Nat>, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> Turn<c, cap>:
  match n k:
    case 0n _:
      Over{Idle{}, m, room, Chat{t, pend, cs}}
    case 1n+lp 0n:
      Over{Idle{}, m, room, Chat{t, pend, cs}}
    case 1n+lp 1n++kp:
      turn.fit(c, pk, cap, t, 1n+kp, room, fit(1n+kp, room), inv, pend, ids, lp, m, cs)

# f: the room the caches have left after their t positions (cap = t + room). A conversation never holds more than
# cap positions, so Short does not happen; it is answered as a full conversation
def turn.room(+c: M.Cfg, +pk: Pick, -cap: Nat, +t: Nat, f: Fit<t, cap>, n: Nat, +pend: Maybe<&2, Nat>, +ids: List<&2, Nat>, m: M.Model<c>, cs: List<&1, M.KV<cap, t>>) -> Turn<c, cap>:
  match f:
    case Short{}:
      Over{Full{}, m, 0n, Chat{t, pend, cs}}
    case Fits{+room, inv}:
      turn.by(n, List.length(&2, Nat, ids), c, pk, cap, t, room, inv, pend, ids, m, cs)

# the ids a turn feeds: the last answer's pending token, when there is one, then the turn's own
def feed(pend: Maybe<&2, Nat>, ids: List<&2, Nat>) -> List<&2, Nat>:
  match pend:
    case Some{x}:
      x <> ids
    case None{}:
      ids

# The conversation's next answer, n tokens at most, one at a time. ids continue the conversation after the last
# answer: for the first turn the conversation's ids in the chat template (tok.bend's chat), and after an answer
# <|im_end|> and what follows it (tok.bend's chat_next); the answer's last token goes before them when it was not a
# stop token. They run through each layer in one pass, on top of the caches, and each new token then runs at its
# position, until a stop token, the count, or the end of the room (cap positions: Chat.start's). The draws of a Pick
# count from the answer's first token, so an answer is the tokens that `tokens` gives for the whole conversation up
# to it.
def turn(+c: M.Cfg, +pk: Pick, +cap: Nat, ch: Chat<c, cap>, m: M.Model<c>, +ids: List<&2, Nat>, n: Nat) -> Turn<c, cap>:
  match ch:
    case Chat{+t, +pend, cs}:
      turn.room(c, pk, cap, t, fit(t, cap), n, pend, feed(pend, ids), m, cs)
