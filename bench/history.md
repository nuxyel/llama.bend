# make bench across Bend releases

One line per Bend release, measured with `scripts/bench_compare.sh <previous bend> <new bend>` on the reference machine:
Intel Core Ultra 7 155H (22 threads), 32 GB, Linux, idle, on mains power, `performance` profile; 5 alternated,
shuffled rounds; "1 thread" pinned to a performance core, "16 threads" unpinned. SmolLM2-135M, 128 tokens from
"Once upon a time", ms per token by the program's own clock; every run produced transformers' tokens. A new release is
detected by `.github/workflows/bend-release.yml`, which opens an issue.

| date | Bend | llama.bend | 1 thread | 16 threads | previous release, 1 / 16 | rounds |
|---|---|---|---|---|---|---|
| 2026-10-13 | 2.0.37 | 8775c40 | 123.2 | 79.1 | 2.0.36: 123.3 / 79.0 | 5 |
| 2026-10-24 | 2.0.38 | e0de45a | 120.2 | 64.2 | 2.0.37: 121.5 / 62.1 | 5 |
| 2026-10-24 | 2.0.39 | e0de45a | 116.4 | 60.5 | 2.0.38: 119.7 / 64.1 | 5 |
| 2026-11-01 | 2.0.39 | 998a3e0 | 122.3 | 60.2 | 2.0.38: 123.7 / 63.9 | 5 |

The row of 2026-11-01 is no new release: 2.0.39 measured again at the commit of the `BENCHMARK.md` refresh of that day,
against 2.0.38 on the same commit. Against 2026-10-24 the one-thread number is 5% higher (116.4 to 122.3), within the
spread between sessions on this machine.
