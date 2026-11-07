# Release checklist

The steps for a Bend bump and for a release of llama.bend, in order. A step that fails stops the list. Steps marked
**Renan** are his decision or his action: the agent does not publish to BendHub, tag a release, merge into `main` or
post. Each run of this list gets a dated `NOTES.md` entry with what the checks reported.

The first public version (Bend 2.0.37, `v1.0.0` and `bend-ml-llama@0.1.0.0`, 2026-10-13) went through an earlier,
longer version of this list, with the launch media and the move to a fresh repository; that history is in `NOTES.md`
and in this file's git history.

## 1. The pin (a Bend bump only)

1. Read the release notes (`gh release view v2.0.N -R bendlang/bend`): breaking changes in Base or the syntax, a
   change in how parallel lets are compiled or scheduled, and whether `--verdict` needs a new Lean.
2. `scripts/bump_bend.sh 2.0.N`: downloads the two Linux tarballs, computes their SHA256 and rewrites the pin in
   `scripts/setup.sh`, the CI step, the READMEs, CONTRIBUTING and CLAUDE.md. Check its diff; it touches pins only.
3. `make setup` installs it; `bend version` says 2.0.N.
4. CLAUDE.md, "pitfalls": read the release notes against each pitfall; one the release fixes becomes history, with
   the issue or pull request that fixed it.

## 2. No regression

1. `make laws`: the checker and `--verdict`, both `ALL PROOFS CHECK`, no `@unsafe`, no `?TODO`, both
   `tests/bad_*.bend` rejected with their own error, `main.bend` free of foreign code.
2. `make check` on the 135M: prep, the tokenizer against `tokenizers`, the layer test, the forward pass and the
   twelve generations against `transformers`, 0 failures, and the thread probe (1 thread on one core, 16 on several).
3. `make bits`: every output of the 135M identical to the saved bits. A bump or a refactor that keeps the order of
   every sum must not move a bit; if one moves, find out why before anything else, and only then `bits.py save`.
4. `BENDRTS="--threads 1" make bits`: the same check on one thread, which runs attention's sequential order.
5. `reference/test_gen.py .scratch/gen` with `MODEL=smollm2-360m` and with `MODEL=smollm2-1.7b`: the twelve prompts,
   0 failures. The 1.7B takes about half an hour and 7 GB; run it alone.
6. The emitted C (`bend demo/gen.bend -o gen.c`) still has `par_att`'s two joins (`..._par_att_j0` and `_j3` at
   v1.5.1 on 2.0.40; the numbers move with the code and the compiler).

## 3. Text

1. NOTES.md: the dated entry, with what the checks above reported.
2. `BENCHMARK.md` and the README's numbers stay dated until the next measurement on an idle machine; a number measured
   on a busy machine goes to NOTES at most.
3. A release that changes the package (`llama/`) bumps `bend-ml-llama`'s version where the READMEs write it, and
   NOTES says why.

## 4. bend-ml

On a Bend bump, the same bump in bend-ml, with its own checklist (bend-ml's `docs/` and `NOTES.md`). Its packages are
republished only if their source changes, and llama.bend moves to a new pin only after it is on BendHub.

## 5. The release (Renan)

1. A pull request `devel` → `main` (CI green), merged.
2. The tag `v1.X.Y` on `main` (bend-ml's `docs/VERSIONING.md`: a patch for fixes, docs and checks, a minor for a new
   capability, a major only for a change in what the project is), and a GitHub release with the notes: what changed,
   what the checks reported, and whether `bend-ml-llama` changed.
3. BendHub, only if `llama/` changed: `bend login` (GitHub), then from `llama/`:
   `bend main.bend --publish bend-ml-llama@0.X.Y.Z` (12+ characters, four numbers, `LICENSE` next to `main.bend`, the
   imports BendHub packages at pinned versions). Publishing is permanent; a signature change is a new version that
   breaks callers. Then check the published package from a scratch directory: `import
   bend-ml-llama@0.X.Y.Z/main.bend as M` and a forward pass on the 135M.
4. Posts, if any: drafts prepared, Renan posts (X in English, LinkedIn in Portuguese).
