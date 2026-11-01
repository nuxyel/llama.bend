# Benchmark

Measured on 2026-11-01 by Renan Vinícius (@nuxyel), 06:21 to 08:17, with the machine idle (load average under 1.0
before every block, nothing else running), on mains power in the `performance` profile: Intel Core Ultra 7 155H (6
performance cores with two threads, 8 efficiency cores, 2 low-power cores; 22 threads), 32 GB, Linux 7.2.8, clang
23.1.3. llama.bend at `998a3e0` (the code under `llama/` is what v1.4.1 published), built with Bend 2.0.39, its pin,
with `bend-ml-tensor-array@0.1.14.0`, `bend-ml-binary-read@0.2.0.0` and `bend-ml-bpe-tokenizer@0.1.3.0`. "1 thread" is
one process pinned to a performance core (`taskset -c 3`) with every thread setting at 1; "16 threads" is unpinned with
every setting at 16. The cases ran in alternated, shuffled rounds (5 for the 135M and 360M, 3 for the 1.7B, 2 for the
512-token runs); the numbers are medians.

The cost per token excludes loading and the prompt: it is the difference between a run that generates many tokens and
one that generates a single token. Prompt: "Once upon a time", greedy decoding. Every system produced the same tokens,
with one exception noted below.

## Milliseconds per generated token

**SmolLM2-135M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend | 118 | 52.9 |
| transformers 5.19 (PyTorch 2.14.1) | 27.1 | 21.0 |
| llama.cpp `23b0202` (F32 GGUF) | 23.8 | 13.9 |
| candle `c68b249` (F32) | 26.8 | 29.2 |

**SmolLM2-360M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend | 272 | 98.3 |
| transformers 5.19 (PyTorch 2.14.1) | 60.0 | 38.2 |
| llama.cpp `23b0202` (F32 GGUF) | 63.3 | 28.2 |
| candle `c68b249` (F32) | 71.5 | 77.3 |

**SmolLM2-1.7B, 128 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend | 1233 | 297 |
| transformers 5.19 (PyTorch 2.14.1) | 238 | 125 |
| llama.cpp `23b0202` (F32 GGUF) | 326 | 101 |
| candle `c68b249` (F32) | 298 | 308 |

On the 1.7B, llama.bend is 3.8 times slower than llama.cpp on one thread and 2.9 times on 16; on 16 threads it is
slightly faster than candle, whose time does not drop with more threads on this machine. On the 135M and 360M it is 4.3 to
5.0 times slower than llama.cpp on one thread and 3.5 to 3.8 times on 16. transformers at 16 threads varies a lot from
sample to sample (135M: 14.7 to 37.8 ms; 360M: 20.2 to 60.3 ms); its medians carry that spread. llama.bend's samples
stay within 5% of their median.

Loading the 1.7B now takes about 5 s, so it no longer inflates the 128-token totals, as it did on 2026-10-13 (66 s then):
llama.bend's own per-token timestamps give 296 ms at 16 threads and 1232 on one, within 1 ms of the black-box numbers.

## Longer generations: 512 tokens

| model | llama.bend, 1 / 16 threads | llama.cpp, 1 / 16 threads |
|---|---|---|
| SmolLM2-135M | 128 / 72.7 | 25.4 / 14.3 |
| SmolLM2-360M | 302 / 118 | 64.9 / 28.8 |
| SmolLM2-1.7B | 1239 / 294 | 332 / 104 |

llama.bend's cost per token grows little with the position: the key/value cache is in flat arrays. From its own
per-token timestamps, the first and last quarter of the 512 tokens, and the growth per cached position:

| model, threads | first quarter | last quarter | growth per position |
|---|---|---|---|
| SmolLM2-135M, 1 | 120 ms | 138 ms | 0.05 ms |
| SmolLM2-135M, 16 | 61 ms | 83 ms | 0.06 ms |
| SmolLM2-360M, 1 | 282 ms | 319 ms | 0.09 ms |
| SmolLM2-360M, 16 | 109 ms | 126 ms | 0.05 ms |
| SmolLM2-1.7B, 1 | 1236 ms | 1313 ms | 0.21 ms |
| SmolLM2-1.7B, 16 | 295 ms | 311 ms | 0.04 ms |

For the 1.7B the timestamps cover the last 496 of the 512 tokens (the harness keeps the tail of the output).

On the 360M over 512 tokens, llama.cpp's text leaves the other systems' after about 340 tokens, at the same place in
every sample, as on 2026-10-12 and 2026-10-13: a run of transformers over the same 513 tokens gives llama.bend's tokens
exactly, so the difference is llama.cpp's own order of F32 additions.

## Start-up and memory

Wall time of a run that loads the model and generates one token, and peak resident memory, at 16 threads:

| system | 135M | 360M | 1.7B |
|---|---|---|---|
| llama.bend | 0.5 s, 716 MB | 1.2 s, 2,255 MB | 5.1 s, 6,819 MB |
| transformers 5.19 (PyTorch 2.14.1) | 2.7 s, 1,135 MB | 2.8 s, 2,436 MB | 3.2 s, 10,157 MB |
| llama.cpp `23b0202` (F32 GGUF) | 0.2 s, 742 MB | 0.2 s, 1,753 MB | 0.8 s, 8,100 MB |
| candle `c68b249` (F32) | 0.2 s, 790 MB | 0.4 s, 2,107 MB | 3.0 s, 9,844 MB |

llama.bend loads the weight matrices from `model.safetensors` with `bend-ml-binary-read`, straight into their arrays.
It is still the slowest to start on the 1.7B, and it has the lowest peak memory on the 1.7B, also on one thread (6,800
MB; 8.4 GB on 2026-10-13 with the old loader). The peak over 512 tokens is 7.1 GB.

## Against the run of 2026-10-13

The run of 2026-10-13 measured `3ea435a` on Bend 2.0.37 with `bend-ml-tensor-array@0.1.7.0` and the old loader, with
the same protocol, the same baselines at the same versions and the same machine. Between the two, Bend went to 2.0.39,
tensor-array to 0.1.14.0, the loader to `bend-ml-binary-read`, and generation to `forward_gen`, which skips the
activations it never reads; this run does not separate those. ms per generated token, llama.bend:

| model | 1 thread, 10-13 → 11-01 | 16 threads, 10-13 → 11-01 |
|---|---|---|
| SmolLM2-135M, 32 tokens | 121 → 118 | 76.7 → 52.9 |
| SmolLM2-360M, 32 tokens | 283 → 272 | 133 → 98.3 |
| SmolLM2-1.7B, 128 tokens | 1203 → 1233 | 492 → 297 (own clock 426 → 296) |
| SmolLM2-135M, 512 tokens | 132 → 128 | 87.1 → 72.7 |
| SmolLM2-360M, 512 tokens | 310 → 302 | 154 → 118 |
| SmolLM2-1.7B, 512 tokens | 1165 → 1239 | 423 → 294 |

On 16 threads every model is faster: 17 to 31% per token, 40% on the 1.7B over 128 tokens counting the loading noise
of 2026-10-13 (31% by its own clock). On one thread little changed: the 135M and 360M are 2 to 4% faster, and the 1.7B is 2.5% slower over
128 tokens and 6% slower over 512 (1165 to 1239 ms). Gaps under about 15% between two sessions are within what I see from
session to session on this machine, so I do not count that as a regression, but it is no gain either. The baselines
moved by 0 to 6% between the two sessions (llama.cpp by 2% or less), which bounds how much of the change is the machine.

The 2.0.37 against 2.0.36 comparison that this file held until now is in its history and in [`NOTES.md`](NOTES.md)
(2026-10-13): no change in speed between those two releases. Bend 2.0.38 against 2.0.39 is in
[`bench/history.md`](bench/history.md).

## Earlier runs

On 2026-10-12, on Bend 2.0.36, I measured three builds: the key/value cache in lists (`b212ccd`), in flat arrays with
the heads in sequence (`aa60b5f`) and with the heads always in parallel (`bd912a7`). The arrays made the 1.7B 4x faster
over 128 tokens at 16 threads (1692 to 427 and 385 ms per token), and the cost per position went from 15-20 ms to
under 0.2 ms. Those numbers are in this file's history and in [`NOTES.md`](NOTES.md) ("milestone 9 closed").
