# Benchmark

Measured on 2026-11-12 by Renan Vinícius (@nuxyel), 00:14 to 01:57, in an unattended run on an idle machine (started
after five quiet minutes: load 0.06, the one open agent session idle at 0.6% of a CPU, nothing else running), on
mains power in the `performance` profile: Intel Core Ultra 7 155H (6 performance cores with two threads, 8 efficiency
cores, 2 low-power cores; 22 threads), 32 GB, Linux 7.2.8, clang 23.1.3. llama.bend at `a76390e`, built with Bend
2.0.40, its pin, with `bend-ml-tensor-array@0.1.20.0`, `bend-ml-binary-read@0.2.0.1` and
`bend-ml-bpe-tokenizer@0.1.3.0`; the commits after it change neither one thread nor 16 in what these tables measure
(the band depth's rule and the prompt's pass, in their sections below). "1 thread" is one process pinned to a performance core (`taskset -c 3`) with every
thread setting at 1; "16 threads" is unpinned with every setting at 16. The cases ran in alternated, shuffled rounds
(5 for the 135M and 360M, 3 for the 1.7B, 2 for the 512-token runs); the numbers are medians.

The cost per token excludes loading and the prompt: it is the difference between a run that generates many tokens and
one that generates a single token. Prompt: "Once upon a time", greedy decoding. Every system produced the same tokens,
with one exception noted below.

## Milliseconds per generated token

**SmolLM2-135M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend | 76.0 | 53.8 |
| transformers 5.19 (PyTorch 2.14.1) | 27.1 | 21.1 |
| llama.cpp `23b0202` (F32 GGUF) | 23.7 | 13.8 |
| candle `c68b249` (F32) | 27.5 | 28.9 |

**SmolLM2-360M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend | 192 | 100 |
| transformers 5.19 (PyTorch 2.14.1) | 60.1 | 39.1 |
| llama.cpp `23b0202` (F32 GGUF) | 63.1 | 28.0 |
| candle `c68b249` (F32) | 72.2 | 76.3 |

**SmolLM2-1.7B, 128 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend | 869 | 294 |
| transformers 5.19 (PyTorch 2.14.1) | 241 | 130 |
| llama.cpp `23b0202` (F32 GGUF) | 329 | 101 |
| candle `c68b249` (F32) | 298 | 300 |

llama.bend is 3.2, 3.0 and 2.6 times slower than llama.cpp on one thread (135M, 360M, 1.7B) and 3.9, 3.6 and 2.9
times on 16. On the 1.7B at 16 threads it is as fast as candle, whose time does not drop with more threads on this
machine. transformers at 16 threads varies a lot from sample to sample (135M: 13.4 to 22.6 ms; 360M: 38.5 to 58.7;
1.7B: 125 to 148); its medians carry that spread. llama.bend's samples stay within 4% of their median.

llama.bend's own per-token timestamps agree with these: on the 1.7B, 868 ms on one thread and 293 on 16, within 1 ms
of the black-box numbers.

The one-thread numbers are the idle protocol's: nothing else running and the desktop still. With the desktop
redrawing every frame (a notification replaced about 30 times a second; the panel on or off), llama.bend's token on
one thread is 8 to 11% faster on this machine ([`NOTES.md`](NOTES.md), 2026-11-12). I did not measure the other
systems in that state, so the ratios above are for the still desktop only.

## Other thread counts

16 threads is not llama.bend's fastest count on the two small models. llama.bend alone, at `336819b`, in a second
idle round the same night (04:13 to 04:33): ms per token by the program's own clock, medians of 5 alternated rounds (3
for the 1.7B), and the cores the 135M's whole run used.

| threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 22 |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M, 40 tokens | 76.5 | 52.2 | 51.2 | 46.2 | 47.6 | 45.7 | 53.4 | 54.6 | 57.4 |
| SmolLM2-360M, 32 tokens | 192 | 113 | 106 | 89.5 | 95.1 | 90.6 | 95.7 | 99.6 | 105 |
| SmolLM2-1.7B, 16 tokens | 889 | 458 | 378 | 334 | 289 | 306 | 268 | 282 | 270 |
| cores used, 135M | 1.00 | 1.67 | 2.10 | 2.54 | 3.21 | 4.11 | 5.56 | 7.29 | 9.49 |

The 135M is fastest on 8 threads and the 360M on 4, and both get slower from there: on 16 threads the 135M uses 7.3
cores for a slower token than on 8 threads with 4.1. 22 threads is what the runtime takes when nothing is asked
(`+RTS --threads 8 -RTS` asks for 8). The 1.7B gains up to 12 threads; 16 and 22 are no faster.

`336819b` is the band depth by thread count and width: two bands on two threads, and fewer than 16 bands for a model
narrower than 1024 on 3 to 8 threads. Before it two threads ran on one core (76.3, 194 and 897 ms per token); with
it the two small models are 9 and 10% faster on 4 threads and 7 and 2% faster on 8. The comparison, cell by cell, is in
[`NOTES.md`](NOTES.md) (2026-11-12), with the raw rows in [`bench/2026-11-12-band-depth/`](bench/2026-11-12-band-depth/).
I did not measure the other systems on these counts.

Where the threads run matters too on this processor. In another idle round that day (20:46 to 20:52, `e0c785d`, a
56-token prompt and 33 new tokens, 5 alternated rounds), 8 threads kept on the performance cores' twelve
(`taskset -c 0-11`) took 42.7 ms per token on the 135M and 81.9 on the 360M, against 45.8 and 89.6 for 8 threads
left to the system and 57.9 and 109 for the 22 the runtime takes by default. The rest is in
[`NOTES.md`](NOTES.md) (2026-11-12), with the raw rows in
[`bench/2026-11-12-performance-cores/`](bench/2026-11-12-performance-cores/).

## The prompt: time to the first token

The tables above leave the prompt out. Since `4770462` a prompt runs through each layer in one pass
(`llama/prompt.bend`), with the bits of the position-by-position path before it (`250bc14`), and since `e0c785d` the
pass runs a position's attention heads one after another, which is what stopped it from losing time on many
threads. llama.bend alone, in two idle rounds on 2026-11-12 (12:42 to 13:05 and 18:42 to 18:56): seconds from the
start of the prompt to the first new token, medians of 5 alternated rounds (3 for the 1.7B).

| model, prompt | 1 thread: by position → one pass | 16 threads: by position → one pass → heads in sequence | 22 threads, now |
|---|---|---|---|
| SmolLM2-135M, 56 tokens | 3.51 → 1.12 | 2.71 → 1.03 → 0.72 | 0.66 |
| SmolLM2-135M, 77 tokens | 4.73 → 1.54 | 3.86 → 1.41 → 0.98 | 0.93 |
| SmolLM2-360M, 56 tokens | 9.17 → 2.91 | 5.11 → 1.79 → 1.47 | 1.39 |
| SmolLM2-360M, 77 tokens | 12.7 → 3.94 | 7.20 → 2.50 → 2.04 | 1.99 |
| SmolLM2-1.7B, 25 tokens | 21.1 → 6.79 | 6.27 → 2.41 → 2.26 | 2.25 |
| SmolLM2-1.7B, 77 tokens | 65.4 → 18.0 | 20.8 → 7.28 → 6.39 | 6.33 |

One pass is 3.1 to 3.6 times faster than position by position on one thread; with the heads in sequence it is 2.8
to 3.9 times faster on 16. The second round ran the pass with its heads in parallel and in sequence, and its
numbers for the parallel heads on 16 threads repeat the first round's within 2.1%. The heads in sequence change
nothing on one thread (35.11 and 35.09 G instructions for the 135M's 56 tokens) and take 15% to 38% off the 135M's
time on 4 to 22 threads, 7% to 26% off the 360M's, and 3% to 12% off the 1.7B's on 8 to 22. The two small models no
longer lose time past 4 threads, and gain none either: the 135M's 56 tokens take 0.66, 0.71, 0.72 and 0.66 s on 4,
8, 16 and 22 threads (0.78, 0.87, 1.04 and 1.05 before). I did not measure the other systems' time to the first
token. The rest is in [`NOTES.md`](NOTES.md) (2026-11-12), with the raw rows in
[`bench/2026-11-12-prompt-pass/`](bench/2026-11-12-prompt-pass/) and
[`bench/2026-11-12-prompt-threads/`](bench/2026-11-12-prompt-threads/).

## A conversation: the first token of a later turn

`gen --chat -` keeps a conversation's caches between its turns, so a turn runs its own tokens and nothing before
them. On the instruct checkpoints at `e0c785d`, in the idle round of 2026-11-12 (18:57 to 19:11): milliseconds from
a turn's line to the first token of its answer, with the caches kept → with the conversation so far run again as
one prompt; medians of 5 alternated rounds (3 for the 1.7B). The conversations are the first and the third of
`reference/chat_turns_ref.py`; the first one's later turns have 15 and 19 tokens of their own in conversations of 59
and 85, the third one's 17 and 13 in 93 and 130.

| model, threads | conversation 1: turn 2 | turn 3 | conversation 3: turn 2 | turn 3 |
|---|---|---|---|---|
| SmolLM2-135M-Instruct, 1 | 389 → 1169 | 487 → 1674 | 466 → 1836 | 391 → 2602 |
| SmolLM2-135M-Instruct, 16 | 318 → 777 | 407 → 1087 | 390 → 1203 | 331 → 1743 |
| SmolLM2-360M-Instruct, 1 | 1005 → 3004 | 1223 → 4273 | 1186 → 4654 | 1012 → 6500 |
| SmolLM2-360M-Instruct, 16 | 561 → 1579 | 739 → 2281 | 712 → 2481 | 624 → 3507 |
| SmolLM2-1.7B-Instruct, 16 | 1909 → 4763 | 2461 → 7224 | | |

A later turn's first token comes 2.4 to 6.7 times sooner with the caches kept. The kept turn's time includes
encoding its text (16 to 21 ms for these turns, timed alone on a busy machine); the one-prompt runs were given their
ids. A first turn has nothing to keep: it took 49 to 104 ms longer than the same tokens as one prompt, which is
that encoding. Every answer has the same tokens and logit bits both ways. I did not measure the other systems on a
conversation. Raw rows: [`bench/2026-11-12-chat-turns/`](bench/2026-11-12-chat-turns/).

## Longer generations: 512 tokens

| model | llama.bend, 1 / 16 threads | llama.cpp, 1 / 16 threads |
|---|---|---|
| SmolLM2-135M | 83.2 / 70.5 | 25.2 / 14.2 |
| SmolLM2-360M | 203 / 118 | 65.5 / 28.8 |
| SmolLM2-1.7B | 880 / 294 | 335 / 104 |

llama.bend's cost per token grows little with the position: the key/value cache is in flat arrays. From its own
per-token timestamps, the first and last quarter of the 512 tokens, and the growth per cached position:

| model, threads | first quarter | last quarter | growth per position |
|---|---|---|---|
| SmolLM2-135M, 1 | 77 ms | 89 ms | 0.03 ms |
| SmolLM2-135M, 16 | 60 ms | 80 ms | 0.05 ms |
| SmolLM2-360M, 1 | 197 ms | 213 ms | 0.04 ms |
| SmolLM2-360M, 16 | 109 ms | 125 ms | 0.04 ms |
| SmolLM2-1.7B, 1 | 882 ms | 937 ms | 0.15 ms |
| SmolLM2-1.7B, 16 | 293 ms | 311 ms | 0.05 ms |

For the 1.7B the timestamps cover the last 496 of the 512 tokens (the harness keeps the tail of the output).

On the 360M over 512 tokens, llama.cpp's text leaves the other systems' after about 340 tokens, at the same place in
every sample, as in every run since 2026-10-12: a run of transformers over the same 513 tokens gives llama.bend's
tokens exactly, so the difference is llama.cpp's own order of F32 additions.

## Start-up and memory

Wall time of a run that loads the model and generates one token, and peak resident memory, at 16 threads:

| system | 135M | 360M | 1.7B |
|---|---|---|---|
| llama.bend | 0.4 s, 718 MB | 0.8 s, 2,255 MB | 3.6 s, 6,817 MB |
| transformers 5.19 (PyTorch 2.14.1) | 2.7 s, 1,135 MB | 2.8 s, 2,437 MB | 3.5 s, 10,158 MB |
| llama.cpp `23b0202` (F32 GGUF) | 0.2 s, 743 MB | 0.2 s, 1,752 MB | 0.8 s, 8,116 MB |
| candle `c68b249` (F32) | 0.2 s, 790 MB | 0.5 s, 2,107 MB | 3.2 s, 9,845 MB |

llama.bend loads the weight matrices from `model.safetensors` with `bend-ml-binary-read`, straight into their arrays.
On the 1.7B it now starts about as fast as transformers and candle (5.1 s on 2026-11-01), and it has the lowest peak
memory there, also on one thread (6,799 MB). The peak over 512 tokens is 7.1 GB.

## v1.5.2's code the same night

The same rounds measured `bc50732`, which is v1.5.2's code. `a76390e` differs from it in one expression: generation
leaves every layer's activations empty on one band too, as it already did on more.

| model | 1 thread, v1.5.2 → `a76390e` | 16 threads, v1.5.2 → `a76390e` |
|---|---|---|
| SmolLM2-135M, 32 tokens | 77.7 → 76.0 (-2.2%, the ranges apart) | 54.5 → 53.8 |
| SmolLM2-360M, 32 tokens | 193 → 192 (-0.4%, the ranges overlapping) | 99.8 → 100 |
| SmolLM2-1.7B, 128 tokens | 940 → 869 (-7.6%) | 295 → 294 |

On 16 threads the two run the same code. One of v1.5.2's three one-thread samples on the 1.7B came out at 841 ms
because that round's one-token run was a cold load (19 s against 6.5 and 6.8); the other two are 940 and 946, and the
program's own clock gives 941. The A/B by the program's own clock, the builds with aligned loops, and how the gain
changes with what else the machine is doing are in [`NOTES.md`](NOTES.md) (2026-11-12).

## Against the run of 2026-11-01

Until this run, this file's tables were of 2026-11-01: `998a3e0` (v1.4.1's code) on Bend 2.0.39 with
`bend-ml-tensor-array@0.1.14.0`, the same protocol, the same baselines at the same versions and the same machine.
ms per generated token, llama.bend:

| model | 1 thread, 11-01 → 11-12 | 16 threads, 11-01 → 11-12 |
|---|---|---|
| SmolLM2-135M, 32 tokens | 118 → 76.0 | 52.9 → 53.8 |
| SmolLM2-360M, 32 tokens | 272 → 192 | 98.3 → 100 |
| SmolLM2-1.7B, 128 tokens | 1233 → 869 | 297 → 294 |
| SmolLM2-135M, 512 tokens | 128 → 83.2 | 72.7 → 70.5 |
| SmolLM2-360M, 512 tokens | 302 → 203 | 118 → 118 |
| SmolLM2-1.7B, 512 tokens | 1239 → 880 | 294 → 294 |

One thread is 29 to 36% faster and 16 threads are the same within 3%. Start-up at 16 threads went from 0.5, 1.2 and
5.1 s to 0.4, 0.8 and 3.6 s. Between the two runs: v1.5.0's dot product unrolled by four with the same bits
(`bend-ml-tensor-array@0.1.19.0`; measured alone on 2026-11-01, llama.bend only, by the program's own clock: one
thread 17 to 30% faster, 16 threads the same; raw rows in
[`bench/2026-11-01-tensor-array-0.1.19/`](bench/2026-11-01-tensor-array-0.1.19/)), a loader that builds no tree of
zeros first, Bend 2.0.40, and the activations left empty on one band. The baselines moved by about 4% or less
between the two sessions (llama.cpp by 1% or less), which bounds how much of the change is the machine.

## Earlier runs

- **2026-11-07**, the 135M with Bend 2.0.39 and 2.0.40 at `4ae89bd` (v1.5.1's code), the same protocol: 80.1 and 79.5
  ms on one thread, 51.7 and 53.1 on 16, every range overlapping; start-up 0.49 and 0.42 s on one thread, the ranges
  apart; 6.4% fewer instructions over a 33-token run with 2.0.40. On 2026-11-12 `scripts/bench_compare.sh` put the
  two releases 4.1% apart on one thread and 2.0% the other way on 16, the ranges apart
  ([`bench/history.md`](bench/history.md)).
- **2026-11-01**, the tables above's first column: v1.4.1's code on Bend 2.0.39.
- **2026-10-13**, `3ea435a` on Bend 2.0.37 with `bend-ml-tensor-array@0.1.7.0` and the old loader: 121, 283 and 1203
  ms on one thread and 76.7, 133 and 492 on 16 (the 1.7B's 16 threads inflated by a 66 s load; 426 by its own
  clock). 2.0.37 against 2.0.36 that day: no change in speed.
- **2026-10-12**, on Bend 2.0.36, three builds: the key/value cache in lists (`b212ccd`), in flat arrays with the
  heads in sequence (`aa60b5f`) and with the heads always in parallel (`bd912a7`). The arrays made the 1.7B 4x faster
  over 128 tokens at 16 threads (1692 to 427 and 385 ms per token), and the cost per position went from 15-20 ms to
  under 0.2 ms.

Each of those runs' full tables is in this file's history and in [`NOTES.md`](NOTES.md).
