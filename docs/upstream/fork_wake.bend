# A tiny parallel let between sequential stretches: 2,000 rounds, each a stretch of 400,000 steps on one thread and
# then two leaves of 2,000 steps. go passes pair 1n, the leaves in a parallel let; with 0n they run one after the
# other, which is the twin fork_wake.sh builds.
import Base

def spin(n: Nat, +x: U32) -> U32:
  match n:
    case 0n:
      x
    case 1n+p:
      spin(p, ((x * 1664525 : U32) + 1013904223 : U32))

def pair(fork: Nat, +s: U32) -> U32:
  match fork:
    case 0n:
      (spin(2000n, s) + spin(2000n, (s + 1 : U32)) : U32)
    case 1n+f:
      a b = spin(2000n, s) spin(2000n, (s + 1 : U32))
      (a + b : U32)

def go(k: Nat, +x: U32) -> U32:
  match k:
    case 0n:
      x
    case 1n+p:
      # bend-lint: disable-next bend-lint-rules/performance/gpu_fit -- the repro of a CPU issue: the let stays on the CPU's threads not-expected-because: a let here is two leaves of under a microsecond
      go(p, pair(1n, spin(400000n, x)))

def main() -> IO(Unit):
  IO.print(U32.show(go(2000n, 1)))
