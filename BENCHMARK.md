# Benchmark

Measured on 2026-10-13 by Renan Vinícius (@nuxyel), 16:26 to 19:35, with the machine idle, on mains power in the
`performance` profile: Intel Core Ultra 7 155H (6 performance cores with two threads, 8 efficiency cores, 2 low-power
cores; 22 threads), 32 GB, Linux 7.2.5, clang 22.1.8. llama.bend at `3ea435a` with `bend-ml-tensor-array@0.1.7.0`,
built twice: with Bend 2.0.37 (its pin) and with Bend 2.0.36, to show what the new release changes on its own. "1
thread" is one process pinned to a performance core (`taskset -c 3`) with every thread setting at 1; "16 threads" is
unpinned with every setting at 16. The cases ran in alternated, shuffled rounds (5 for the 135M and 360M, 3 for the
1.7B, 2 for the 512-token runs); the numbers are medians.

The cost per token excludes loading and the prompt: it is the difference between a run that generates many tokens and
one that generates a single token. Prompt: "Once upon a time", greedy decoding. Every system produced the same tokens,
with one exception noted below.

At this commit llama.bend runs attention's key/value heads one after another with two threads or fewer, and in parallel
otherwise.

## Milliseconds per generated token

**SmolLM2-135M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend, Bend 2.0.37 | 121 | 76.7 |
| llama.bend, Bend 2.0.36 | 111 | 74.5 |
| transformers 5.19 (PyTorch 2.14.1) | 26.3 | 21.1 |
| llama.cpp `23b0202` (F32 GGUF) | 23.7 | 14.0 |
| candle `c68b249` (F32) | 26.6 | 28.8 |

**SmolLM2-360M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend, Bend 2.0.37 | 283 | 133 |
| llama.bend, Bend 2.0.36 | 292 | 135 |
| transformers 5.19 (PyTorch 2.14.1) | 57.1 | 40.0 |
| llama.cpp `23b0202` (F32 GGUF) | 63.1 | 28.3 |
| candle `c68b249` (F32) | 71.6 | 74.3 |

**SmolLM2-1.7B, 128 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend, Bend 2.0.37 | 1203 | 492 |
| llama.bend, Bend 2.0.36 | 1161 | 419 |
| transformers 5.19 (PyTorch 2.14.1) | 231 | 132 |
| llama.cpp `23b0202` (F32 GGUF) | 324 | 101 |
| candle `c68b249` (F32) | 292 | 295 |

The 1.7B's numbers for llama.bend carry its loading. Both runs of a sample load the model (66 to 73 s, varying by
several seconds from run to run), and over 128 tokens a few seconds of difference is tens of milliseconds per token.
At 16 threads the samples on 2.0.37 range from 432 to 510 ms, while the program's own per-token timestamps give 426 ms
on 2.0.37 and 423 on 2.0.36. The 512-token runs below are less exposed to it: 423 ms at 16 threads, 1165 on one.

## Bend 2.0.36 against 2.0.37

Bend 2.0.37 brings two changes to how parallel code compiles and runs: my fix for an erased last parameter that made a
parallel let sequential (bendlang/bend#1377), and forks of a slow grow handed to idle threads (#1479). On llama.bend
neither changes the speed: every difference above is within the spread of the samples, and the per-token timestamps
agree (on the 135M at one thread, 118.8 ms on both). No def here had the shape that #1377 fixes, and the parallel
heads gain nothing more from #1479 on this machine. The compiled program is less than half the size (0.57 MB against
1.09 MB), and its output is the same bit for bit.

## Longer generations: 512 tokens

| model | llama.bend, 1 / 16 threads | llama.cpp, 1 / 16 threads |
|---|---|---|
| SmolLM2-135M | 132 / 87.1 | 24.9 / 14.2 |
| SmolLM2-360M | 310 / 154 | 65.1 / 28.8 |
| SmolLM2-1.7B | 1165 / 423 | 329 / 104 |

llama.bend's cost per token does not grow with the position: the key/value cache is in flat arrays. From its own
per-token timestamps, the first and last quarter of the 512 tokens, and the growth per cached position:

| model, threads | first quarter | last quarter | growth per position |
|---|---|---|---|
| SmolLM2-135M, 1 | 125 ms | 145 ms | 0.06 ms |
| SmolLM2-135M, 16 | 79 ms | 95 ms | 0.04 ms |
| SmolLM2-360M, 1 | 293 ms | 327 ms | 0.09 ms |
| SmolLM2-360M, 16 | 148 ms | 164 ms | 0.05 ms |
| SmolLM2-1.7B, 1 | 1161 ms | 1231 ms | 0.18 ms |
| SmolLM2-1.7B, 16 | 427 ms | 452 ms | 0.07 ms |

On the 360M over 512 tokens, llama.cpp's text leaves the other systems' after about 340 tokens, in every sample (as on
2026-10-12): a run of transformers over the same 513 tokens gives llama.bend's tokens exactly, so the difference is
llama.cpp's own order of F32 additions.

## Start-up and memory

Wall time of a run that loads the model and generates one token, and peak resident memory, at 16 threads:

| system | 135M | 360M | 1.7B |
|---|---|---|---|
| llama.bend, Bend 2.0.37 | 4.4 s, 781 MB | 11.5 s, 2,332 MB | 66.4 s, 6,935 MB |
| llama.bend, Bend 2.0.36 | 4.4 s, 785 MB | 11.6 s, 2,331 MB | 73.0 s, 6,935 MB |
| transformers 5.19 (PyTorch 2.14.1) | 2.7 s, 1,168 MB | 2.7 s, 2,469 MB | 4.1 s, 10,189 MB |
| llama.cpp `23b0202` (F32 GGUF) | 0.2 s, 743 MB | 0.2 s, 1,752 MB | 0.5 s, 8,116 MB |
| candle `c68b249` (F32) | 0.2 s, 791 MB | 0.4 s, 2,107 MB | 2.5 s, 9,844 MB |

Loading is llama.bend's slowest part: reading a file in Bend gives one list cell per byte, and building and walking
those lists is the cost (bend-ml `NOTES.md`, exp. 13). The weight products are the rest of the distance to llama.cpp;
both are tracked in [`NOTES.md`](NOTES.md). On one thread the 1.7B peaks at 8.4 GB.

## Earlier runs

On 2026-10-12, on Bend 2.0.36, I measured three builds: the key/value cache in lists (`b212ccd`), in flat arrays with
the heads in sequence (`aa60b5f`) and with the heads always in parallel (`bd912a7`). The arrays made the 1.7B 4x faster
over 128 tokens at 16 threads (1692 to 427 and 385 ms per token), and the cost per position went from 15-20 ms to
under 0.2 ms. Those numbers are in this file's history and in [`NOTES.md`](NOTES.md) ("milestone 9 closed").
