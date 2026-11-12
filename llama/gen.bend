# bend-ml-llama, generation: the ids of a prompt run through main.bend's forward pass position by position (one
# key/value cache per layer), and then each new token is chosen from the logits and fed back in: the argmax (greedy),
# or a draw with temperature, top-k and top-p from a seed (main.bend's sampler). Generation stops after n tokens, or
# after the end-of-text token (id 0), as transformers' generate does; in a chat after <|im_end|> (id 2) too.
#
#   import ./main.bend as M
#   import ./gen.bend as G
#   G.generate(c, model, G.Greedy{}, False{}, ids, n)                  # G.Gen{toks, m}: the new tokens and the model
#   G.tokens(c, model, G.Draw{0.8, 50n, 0.95, 42}, False{}, ids, n)    # the same loop, one token at a time (G.Toks)
#
# The model is a parameter, so the caller chooses its loader. Nothing here prints or reads a clock: a program that
# shows each token as it comes walks G.Toks, as llama.bend's demo/gen.bend does.

import Base
import ./main.bend as M

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
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- forward_pre returns lg and nf empty, and acts empty on more than one band not-expected-because: measured 2026-10-16: forward_pre took 5-16% off the time to the first token; one band keeps Act on purpose (NOTES)
      M.Out.then(Toks<c>, c, cap, t, M.forward_pre(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        loop(p, c, pk, chat, cap, 1n+t, M.step_inv(t, p, cap, inv), q, qt, 1n+lp, j, m2, cs2))
    case 1n++p Nil{} 1n+lp:
      # bend-lint: disable-next bend-lint-rules/performance/unused_result -- nf is the logits' input, computed anyway; acts is empty not-expected-because: forward_gen measured 19% faster per token on 16 threads on 2026-10-16, and leaves Act empty on one band too since 2026-11-12 (NOTES)
      M.Out.then(Toks<c>, c, cap, t, M.forward_gen(c, cap, t, m, cs, M.step_room(t, p, cap, inv), tok), lg => nf => acts => cs2 => m2 =>
        +x = choose(pk, lg, j)
        +id = M.AM.i(Chosen.am(x))
        Tok{Chosen.am(x), Chosen.cs(x), m2, m3 => loop(p, c, pk, chat, cap, 1n+t, M.step_inv(t, p, cap, inv), id, Nil{}, left_chat(chat, id, lp), 1n+j, m3, cs2)})

# the new tokens after the prompt ids, n at most, one at a time: the prompt runs first, then the first token is
# chosen. An empty prompt gives no token (a Llama needs one position to start from).
def tokens(+c: M.Cfg, m: M.Model<c>, +pk: Pick, +chat: Bool, +ids: List<&2, Nat>, +n: Nat) -> Toks<c>:
  match ids:
    case Nil{}:
      End{m}
    case Con{+t0, qt}:
      loop(Nat.add(List.length(&2, Nat, ids), n), c, pk, chat, Nat.add(List.length(&2, Nat, ids), n), 0n, {==}, t0, qt, n, 0n, m, M.caches(c, M.Cfg.layers(c), Nat.add(List.length(&2, Nat, ids), n)))

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
