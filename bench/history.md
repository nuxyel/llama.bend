# make bench across Bend releases

One line per Bend release, measured with `scripts/bench_compare.sh <previous bend> <new bend>` on the reference machine:
Intel Core Ultra 7 155H (22 threads), 32 GB, Linux, idle, on mains power, `performance` profile; 5 alternated,
shuffled rounds; "1 thread" pinned to a performance core, "16 threads" unpinned. SmolLM2-135M, 128 tokens from
"Once upon a time", ms per token by the program's own clock; every run produced transformers' tokens. A new release is
detected by `.github/workflows/bend-release.yml`, which opens an issue.

| date | Bend | llama.bend | 1 thread | 16 threads | previous release, 1 / 16 | rounds |
|---|---|---|---|---|---|---|
