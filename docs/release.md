# Release checklist: Bend 2.0.37 and the first public version

Status on 2026-10-13: steps 1 to 4b, 5.1 and 7.2 are done (NOTES, "2026-10-13"; the measurement ran before the README,
so its numbers are the new ones). What is left is Renan's: 6, then 5.2 and 5.3 (with `llama.bend.mp4`, rendered by
`scripts/make_media.sh --video`, attached to the v1.0.0 release), bend-ml's release PR and tag (v1.2.4), and 7.1.
Order: publish to BendHub (6) before the repository goes public (5.2), since both READMEs already import
`bend-ml-llama@0.1.0.0`.

The steps from a new Bend release to llama.bend public and `bend-ml-llama` on BendHub, in order. A step that fails
stops the list. Steps marked **Renan** are his decision or his action: the agent does not make the repository
public, publish to BendHub, tag a release or post.

## 1. The pin

1. Read the release notes (`gh release view v2.0.37 -R bendlang/bend`): breaking changes in Base or the syntax, and
   whether `--verdict` needs a new Lean.
2. `scripts/bump_bend.sh 2.0.37`: downloads the two Linux tarballs, computes their SHA256 and rewrites the pin in
   `scripts/setup.sh`, the CI step, the READMEs and CLAUDE.md. Check its diff; it touches pins only.
3. `make setup` installs it; `bend version` says 2.0.37.

## 2. No regression

1. `make laws`: the checker and `--verdict`, both `ALL PROOFS CHECK`, no `@unsafe`, no `?TODO`.
2. `make check` on the 135M: prep, the tokenizer against `tokenizers`, the layer test, the forward pass and the
   twelve generations against `transformers`, 0 failures.
3. `make bits`: every output of the 135M identical to the saved bits. #1377 and #1479 change how parallel lets are
   compiled and scheduled, not the order of any sum, so the bits should not move; if one does, find out why before
   anything else, and only then `bits.py save`.
4. The same bits check with `--threads 1` (wrappers that pass it), which runs attention's sequential order.
5. `MODEL=smollm2-360m` and `MODEL=smollm2-1.7b`: `reference/test_gen.py` on the twelve prompts, 0 failures. 
   1.7B takes about half an hour; run it alone.
6. The emitted C (`bend demo/gen.bend -o gen.c`) still has `par_att`'s two joins.

## 3. Text

1. CLAUDE.md, "pitfalls": the erased-last-parameter pitfall becomes history (fixed in 2.0.37 by #1377); keep the
   advice to check the joins in the C.
2. NOTES.md: a dated entry for the bump, with what the checks above reported.
3. `BENCHMARK.md` stays dated (2026-10-12, Bend 2.0.36) until the next measurement; say so in the README if the
   release goes out before it.

## 4. bend-ml

The same bump in bend-ml, with its own checklist (bend-ml `NOTES.md`, "Checklist for the bump to the release that
carries #1377"): `make check-full`, the C join check of `bmv`, the workaround comment retired. Its packages are
republished only if their source changes.

## 4b. The README and the launch video

In the identity of bend-ml's (its `docs/media/`: `shoot.mjs` for terminal screenshots of real runs, `charts.mjs` for
the banner and the diagram from measured numbers, `deck.mjs` with a `readme.mjs` for the result frames,
`render_video.mjs` for the video: a scene rendered frame by frame in headless Brave at 60 fps, blended to 30 with
motion blur, encoded with ffmpeg, plus a poster and a webp teaser), a `docs/media/` here and a `make media`:

1. The README rewritten around the results, like bend-ml's latest: a banner, badges, numbered sections, a terminal
   shot of a real generation, the laws in a few lines, the frames for the benchmark (from `BENCHMARK.md`) and the flat
   per-token cost, reproduce, limits.
2. A launch video for v1.0 (attached to the release, not committed): what llama.bend is, a model generating in the
   terminal, the types that reject a wrong weight, the laws, the numbers with their losses.
3. Every number in both comes from `BENCHMARK.md` or a captured run; nothing typed by hand.

## 5. Public (Renan)

1. Done on 2026-10-13: `nuxyel/legacy-llama.bend` (the old one, private), a new private `nuxyel/llama.bend` with `main`,
   `devel`, the tags and the v0.0.1 and v0.0.2 releases, the local clone's `origin` pointing at it (`legacy` at the old
   one). Read the README, the package README, `BENCHMARK.md` and `docs/` once more as a stranger would. Tracked files
   name no private repository and describe no measurement method, and since 2026-10-12 neither does the history
   (rewritten with git filter-repo; every tree on `devel` unchanged; the old -> new commit map is kept privately).
   GitHub still keeps the pull requests #1 and #2 (the v0.0.1 and v0.0.2 releases) pointing at the old commits, so
   this repository does not go public. Decided on 2026-10-12: rename it `legacy-llama.bend` (it stays private),
   create a new `nuxyel/llama.bend` (private until the launch), push `devel`, `main` and the tags (the rewritten
   history, so the commits cited in `BENCHMARK.md` and `NOTES.md` exist), recreate the v0.0.1 and v0.0.2 releases
   with the same notes, and point the local clone at it.
2. Make the repository public; bend-ml's README links it.
3. A pull request `devel` → `main`, then the tag `v1.0.0` on `main`: by bend-ml's `docs/VERSIONING.md`, 1.0 is the
   first stable, reproducible release that matches the reference (decided 2026-10-12). The BendHub package keeps
   its own number, `bend-ml-llama@0.1.0.0`.

## 6. BendHub (Renan)

1. The API: `cap` is in `KV`'s type since 2026-10-12 (NOTES, "the capacity in the cache's type"), so the first
   version publishes it. Publishing is permanent; a later signature change is a new version that breaks callers.
2. `bend login` (GitHub), then from `llama/`: `bend main.bend --publish bend-ml-llama@0.1.0.0`. The name has 12+
   characters, the version four numbers, `LICENSE` sits next to `main.bend`, and the imports are BendHub packages at
   pinned versions.
3. Check the published package from a scratch directory: `import bend-ml-llama@0.1.0.0/main.bend as M` and a
   forward pass on the 135M.

## 7. After

1. Posts (drafts prepared, Renan posts): X in English, LinkedIn in Portuguese.
2. The next measurement, on an idle machine on mains power: the current commit on 2.0.36 and on 2.0.37 in the same
   session, so the release's effect is measured on its own (one build per Bend version).
