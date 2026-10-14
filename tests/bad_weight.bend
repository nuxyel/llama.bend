import Base
import ../llama/main.bend as M
# MUST FAIL: the query and key projections of the 135M swapped. q is 576 x 576 and k is 192 x 576 (3 key/value
# heads of 64), and each field of LW<c> has its shape from the config, so the swap does not compile.
def swap(lw: M.LW<M.smollm2_135m()>) -> M.LW<M.smollm2_135m()>:
  match lw:
    case M.LW{ln1, q, k, v, o, ln2, gate, up, down}:
      M.LW{ln1, k, q, v, o, ln2, gate, up, down}
def main() -> Nat:
  0n
