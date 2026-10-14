import Base
import ../llama/main.bend as M
# MUST FAIL: a cache made for 2 positions already holds 2, and a third write asks for a proof that 2 < 2, which
# does not exist. Writing past the capacity does not compile.
def third(kv: M.KV<2n, 2n>, ks: List<&2, List<&2, F32>>, vs: List<&2, List<&2, F32>>) -> M.KV<2n, 3n>:
  M.KV.push(M.smollm2_135m(), 2n, 2n, kv, {==}, ks, vs)
def main() -> Nat:
  0n
