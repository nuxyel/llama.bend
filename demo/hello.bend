# The smallest program on the package (bend-ml-llama, llama/): it imports the package's files and nothing else, loads
# a model directory prepared by reference/prep.py, and prints n greedy tokens after a prompt, one id per line.
# reference/test_hello.py checks them against transformers' (bench/expected-<MODEL>.txt).
#
#   hello <dir> "prompt" n
#
# The weights come through main.bend's loader, in plain Bend (demo/gen.bend uses the faster llama/load.bend).

import Base
import ../llama/main.bend as M
import ../llama/tok.bend as K
import ../llama/gen.bend as G

def run(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{+dir, Con{prompt, Con{n, Nil{}}}}}:
      do IO<Unit>:
        tb : K.Tables <- K.load(dir)
        +c : M.Cfg <- M.read_cfg(dir)
        model : M.Model<c> <- M.load_model(c, dir)
        g : G.Gen<c> = G.generate(c, model, G.Greedy{}, False{}, K.encode(tb, prompt), Maybe.default(&2, Nat, Nat.read(n), 0n))
        List.for_each(~&2, ~Nat, ~(id => IO.print(Nat.show(id))), G.ids_of(G.Gen.toks(c, g)))
    case _:
      IO.print("usage: hello <dir> <prompt> <n_tokens>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    run(args)
