# llama.bend

SmolLM2-135M, a Llama-architecture language model, running in Bend 2: token for token against PyTorch,
with the shapes in the types and the structural guarantees (grouped-query head mapping, even RoPE head size,
causal KV cache, weight layout) as laws checked by Bend's kernel. Step 1 of the plans after bend-ml v1.2
("llama.bend" from Victor Taelin's wishlist, https://x.com/VictorTaelin/status/2104949395770712276). The
typed layers will become the BendHub package `bend-ml-llama`. **Private for now.**

## Where things are

- `docs/plan.md`: the milestones, each with its pass criterion, and the laws to prove.
- `NOTES.md`: every finding, measurement and decision, by date. Read it before changing anything.
- `llama/main.bend`: the typed package (future `bend-ml-llama`): `Cfg`, `LW<c>`, `Model<c>`, `KV<cap, t>`, `block`,
  `forward`, the loader and every law; `llama/README.md` explains the laws. `make laws` runs bend and `--verdict`.
- `demo/`: programs on the package (`forward.bend`, `gen.bend`, `layer_test.bend`, which take the model
  directory as their first argument) and the SmolLM2 tokenizer (`tok.bend`, `unicode.bend`).
- `reference/`: the Python side: `models.py` (the pinned models; `MODEL=` picks one), `prep.py` (download,
  bf16 → F32, `cfg.txt`, tokenizer tables, `--check`), the `transformers` oracles (`*_ref.py`), the tests
  (`test_*.py`) and `bits.py` (exact outputs, to show a refactor changes no bit: `make bits`).
- `data/<MODEL>/` (not committed): what `make prep MODEL=...` downloads and writes (default `smollm2-135m`).
- bend-ml (`~/Repos/Bend2/bend2-ml`, https://github.com/nuxyel/bend-ml) is the parent project: its packages are
  imported from BendHub at pinned versions (`bend-ml-tensor-array@0.1.7.0`, `bend-ml-bpe-tokenizer@0.1.2.0`),
  and its GPT-2 demo (`demos/gpt2/`) is the pattern for the loader, the tokenizer and the oracle tests. Its
  `NOTES.md` holds the performance experiments (exp. 1-14) this project builds on.
- Renan's planning notes are private and live outside this repository; never link them from it.

## Branches and versions

Work on `devel`; `main` only receives releases through a pull request `devel` → `main`. Version numbers follow
bend-ml's `docs/VERSIONING.md` (a major only for a change in what the project is).

## About Renan

Software developer at Sidia (Samsung R&D), where he trains NLP, ML and CV models, puts them into production and
builds agentic systems over LLMs (RAG, MCP, skill routing); writes Python and TypeScript; studies ML formally. No background
in type theory: explain each proof in plain language, in a comment and in chat. Machine: Core Ultra 7
155H (22 threads; P-cores are CPUs 0-11, pin one-thread timings with `taskset -c 3`), 32 GB RAM, RTX 4050 6 GB
(for these memory-bound kernels the GPU is slower than the CPU, bend-ml NOTES exp. 10).

## Rules

1. Bend is pinned to **2.0.37** (and Lean 4.34.0 for `--verdict`). Bend 2 changes fast and differs from
   Bend 1: before writing code, read `bend guide`, `bend base <name>` and the Base source
   (`~/.bend/bend2/base.bend`). A version bump goes in `NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second.
3. Every proof is real: no `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet is
   recorded in `NOTES.md` as debt and left out of the README.
4. Prove structure (shapes, sizes, composition); test numbers. `F32` is not a real number, so numerics
   are validated against PyTorch / `transformers` / `tokenizers` by the scripts in `reference/`.
5. Every proved law appears in the package README in plain language.
6. Publishing to BendHub is public and permanent: four-number versions (`0.1.0.0`), names of 12+
   characters, a `LICENSE` next to the entry file. Ask Renan before publishing.
7. Benchmarks are honest: hardware and versions written down, numbers measured on an idle machine,
   alternated runs, one thread pinned to a P-core.
8. Everything in the repository is in English. Commits are small, with clear messages and no
   attribution lines (no `Co-Authored-By`, no "Generated with").
9. Public text (READMEs, `NOTES.md`, `docs/`) is written in Renan's first person singular: "I measured",
   never "we"/"our". Where a name is needed: "Renan Vinícius (@nuxyel)". This file stays in the third person.

## Bend 2.0.37 pitfalls (paid for in bend-ml)

- `match` only inspects a parameter or a pattern variable; a computed value (including a destructuring
  `let` of a call) needs its own def or a parameter computed by the caller. Matching several parameters at
  once (`match b small:`) is the way to branch on a computed `Bool` and a value together.
- No mutual recursion. Termination reads arguments left to right: each must be passed unchanged until one
  shrinks. Erased (`-x`) parameters are skipped, so put a changing size after the shrinking argument or
  make it erased.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type counts as a use.
- `%e : P` rewrites: `_` marks where the right-hand side of `e` sits in the current goal, and the new goal
  has the left-hand side there; use `Equal.sym` to go the other way. `Nat.sub(1n+p, 1n)` does not reduce to
  `p` in the checker: pass `p` itself.
- Parallel lets (`a b = f(..) g(..)`) suffer from non-tail recursive helpers in the fork tree (use tail
  loops with an accumulator). Check scaling with `--threads 1` against `--threads 16`, and read the emitted
  C (`bend X.bend -o x.c`): the joins of each parallel let are there. (Up to 2.0.36 they silently became
  sequential when the def's last parameter was erased; fixed in 2.0.37 by bendlang/bend#1377.)
- `IO.fork` gives concurrency, not parallelism (by design, bendlang/bend#1415). Use parallel lets in pure
  code for several cores.
- Proofs follow the affine rules too: a lemma about values that hold an `Array` cannot mention them twice at
  run-time positions; pass the expression in an erased argument (`-s`) and the arrays once. A lemma
  quantified over a single-constructor value (`g: G`) can match it, which avoids computing the call that
  produced it.
- `Array` is a binary tree in Base (`ALeaf`/`ANode`) but one flat block in the compiled program: `get`/`set` are
  an index mask and a load or store (no descent), while matching an array copies each half into its own block
  (BendRT, "Arrays"), so never walk an array by `match`; `Array.clone` is O(n); `Array.fork`/`Array.join` are
  `@unsafe`. Indexes wrap around, so never write past a capacity. Corrected 2026-10-13 (bend-ml NOTES).
- `File.read_at` returns a `List<U32>` with one cell per byte; producing and walking it is the cost of
  loading (bend-ml exp. 13), and erasing a large unread list costs more than walking it.
- An `import ... as M` does not re-export, so a publishable package is one file. Relative imports
  (`import ./x.bend as X`) work for local copies.
- `Nat` is unary in the checker: `{==}` on a count of millions overflows its stack. State large counts in
  `U32` (the checker computes them at once; mind the 2^32 wrap) and keep `Nat` for sizes in types.
- The pair `A & B` has kind `Type`; a value that must be `Data` (reused, like a cache read once per head) needs its
  own datatype (`Snoc<A>`). A type can be computed by recursion (`def Cache(t: Nat) -> Data`).
- Scrutinees of a multi-`match` follow parameter order: `match fuel queue left`, never `match fuel left queue`
  when `queue` is declared first.
- A loop that must match a computed result (a forward pass at each position) recurses inside a closure: pass the
  result to a `then`-style def that matches it and calls a lambda (`Out.then(o, lg => ... => loop(p, ...))`). The
  termination checker accepts the recursive call inside the lambda, and there is no mutual recursion.
- A running F32 sum loses the small terms next to a large one (Llama's massive activations): sum long reductions
  as a balanced tree (`tree_sum` in `llama/main.bend`), and compare against a float64 run before blaming PyTorch.

## Agent skills

### Issue tracker

Issues live in GitHub Issues of `nuxyel/llama.bend`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root, alongside `NOTES.md` and `docs/plan.md`. See `docs/agents/domain.md`.
