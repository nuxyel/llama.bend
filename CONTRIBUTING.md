# Contributing to llama.bend

Thanks for looking at llama.bend. I'm Renan Vinícius (@nuxyel) and I maintain it alone, so I read every issue
and pull request myself. Bug reports, numbers from other machines, fixes and new laws are all welcome.

## Branches

- **Open pull requests against `devel`.** All work happens there; `main` only receives releases, through a pull
  request from `devel`, and the tags live on `main`. GitHub proposes `main` by default, so change the base branch
  when you open the PR. If you forget, I'll retarget it.
- Branch from `devel` in your fork, not from `main`.

## Before you start

- For anything beyond a small fix (a change to the package's types or laws, a new model, performance work, a new
  dependency), open an issue first so we can agree on the approach before you spend time on it.
- Read [`NOTES.md`](NOTES.md): it records what was tried, measured and decided, including the experiments that did
  not pay off. [`docs/plan.md`](docs/plan.md) has the milestones.
- Bend 2 is young and differs a lot from Bend 1. The "pitfalls" section of [`CLAUDE.md`](CLAUDE.md) lists the traps
  I hit in practice; read it before writing Bend.

## Setup and checks

```bash
make setup   # the pinned Bend 2.0.37, Lean 4.34.0 and the Python venv of reference/
make prep    # SmolLM2-135M into data/smollm2-135m/ (not committed)
make laws    # the proofs, by bend and again by --verdict; tests/bad_*.bend must be rejected
make check   # the laws, then the tokenizer, one layer, the forward pass and generation against transformers
make bits    # forward and generation give the same bits as before
make bench   # one line of ms per token, failing if the tokens differ from transformers'
```

A pull request should pass `make check`. If it touches `llama/` or `demo/`, also run `make bits`: a refactor must
not change a single output bit. If the bits move on purpose (a different summation order, say), explain why in the
PR and show that `make check` still passes on the 135M; the 360M and 1.7B (`MODEL=smollm2-360m`,
`MODEL=smollm2-1.7b`) are welcome when the change could depend on the model's shape.

## Rules for code

- **Bend stays pinned to 2.0.37.** A version bump is its own change, with an entry in `NOTES.md`.
- **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms in `llama/`. A law that cannot be proved yet goes
  into an issue, not into the package.
- **Prove structure, test numbers.** Shapes, sizes and composition belong in the types and laws; `F32` results are
  checked against PyTorch / `transformers` by the scripts in `reference/`.
- **Every proved law is explained in plain language** in [`llama/README.md`](llama/README.md).
- `llama/main.bend` is published to BendHub as one file (`bend-ml-llama`), so it imports nothing local.

## Benchmarks

Performance claims need numbers that someone else can reproduce: the hardware, the Bend and compiler versions, an
idle machine, runs alternated between the old and the new version, and one-thread timings pinned to a single core.
Report the losses and the noise too, not only the best run. `scripts/bench_compare.sh` alternates two builds for
you. A speedup that changes the generated tokens is a bug.

## Commits and pull requests

- Small commits, one logical change each, with clear messages in English.
- Keep a pull request to one topic. Several small PRs are easier to review than one large one.
- Don't commit model weights, `data/`, `.scratch/` or large logs.
- Using AI tools is fine. You are still the author: say in the PR what you ran yourself and what you did not, and
  keep the description short enough for a person to read.

## License

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE) of this repository.
