# The sampler of llama/main.bend (sample, uniform) on logits made from the same hash, for reference/test_sample.py,
# which repeats every draw in Python (reference/sample_ref.py) and must find the same ids and the same logit bits.
# Takes no arguments and refuses any (runtime options go between +RTS and -RTS).
#
# A case: a seed, a vocabulary of n logits, made with many ties (64 levels) or few, a temperature, k, p and a number
# of draws. Logit j of a case is hash32(seed + j): its top 6 bits / 4 - 4 with ties, its top 24 bits · 2^-20 - 8
# without. Prints `case <c> draw <d> id <id> logit <F32 bits>` for each draw d (uniform(seed, d)).

import Base
import ../llama/main.bend as M

type Case is Data:
  Case{seed: U32, n: Nat, ties: Bool, temp: F32, k: Nat, p: F32, draws: Nat}

def logit(ties: Bool, +h: U32) -> F32:
  match ties:
    case True{}:
      F32.sub(F32.mul(U32.to_f32(U32.shrn(h, 26n)), 0.25), 4.0)
    case False{}:
      F32.sub(F32.mul(U32.to_f32(U32.shrn(h, 8n)), 9.5367431640625e-7), 8.0)

# the logits from j on, n of them, newest first onto acc (a tail loop), then reversed
def logits.go(n: Nat, +ties: Bool, +seed: U32, +j: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      List.reverse(&2, F32, acc)
    case 1n+p:
      logits.go(p, ties, seed, U32.inc(j), logit(ties, M.hash32(U32.add(seed, j))) <> acc)

def draws(d: Nat, +c: Nat, +xs: List<&2, F32>, +seed: U32, +temp: F32, +k: Nat, +p: F32, +j: Nat) -> String:
  match d:
    case 0n:
      ""
    case 1n+q:
      +s = M.sample(xs, temp, k, p, M.uniform(seed, U32.from_nat(j)))
      "case " ++ Nat.show(c) ++ " draw " ++ Nat.show(j) ++ " id " ++ Nat.show(M.Cand.i(s)) ++ " logit " ++ U32.show(F32.bits(M.Cand.v(s))) ++ "\n" ++ draws(q, c, xs, seed, temp, k, p, 1n+j)

def run(cs: List<&2, Case>, +c: Nat) -> String:
  match cs:
    case Nil{}:
      ""
    case Con{Case{+seed, n, +ties, +temp, +k, +p, d}, t}:
      draws(d, c, logits.go(n, ties, seed, 0, Nil{}), seed, temp, k, p, 0n) ++ run(t, 1n+c)

# reference/test_sample.py holds the same list
def cases() -> List<&2, Case>:
  [Case{1, 1n, True{}, 1.0, 1n, 1.0, 4n},
   Case{2, 2n, True{}, 1.0, 2n, 1.0, 8n},
   Case{3, 7n, True{}, 0.5, 3n, 0.9, 8n},
   Case{4, 7n, False{}, 1.0, 9n, 1.0, 8n},
   Case{5, 100n, True{}, 1.0, 100n, 1.0, 16n},
   Case{6, 100n, True{}, 2.0, 10n, 0.5, 16n},
   Case{7, 1000n, False{}, 0.7, 40n, 0.9, 16n},
   Case{8, 1000n, True{}, 0.7, 40n, 0.9, 16n},
   Case{9, 1000n, True{}, 1.3, 1n, 1.0, 4n},
   Case{10, 1000n, False{}, 0.01, 50n, 1.0, 8n},
   Case{11, 1000n, False{}, 100.0, 50n, 0.3, 16n},
   Case{12, 1000n, True{}, 1.0, 200n, 0.05, 8n},
   Case{13, 1000n, True{}, 1.0, 1200n, 0.95, 16n},
   Case{14, 49152n, False{}, 0.8, 50n, 0.95, 12n},
   Case{15, 49152n, True{}, 1.0, 50n, 1.0, 12n},
   Case{4294967295, 49152n, False{}, 0.6, 20n, 0.8, 12n}]

def main.go(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Nil{}}:
      IO.write(run(cases(), 0n))
    case _:
      IO.print("usage: sample_test (no arguments; runtime options go between +RTS and -RTS)")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.go(args)
