<p align="center">
  <img src="docs/media/banner.svg" alt="llama.bend: Llama models in Bend 2. Shapes in the types, laws checked by the kernel, the same tokens as PyTorch." width="100%">
</p>

<p align="center">
  <a href="https://github.com/nuxyel/llama.bend/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nuxyel/llama.bend/ci.yml?branch=main&style=flat-square&label=ci&labelColor=4d4a44&color=7e9a5e" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-8b83b5?style=flat-square&labelColor=4d4a44" alt="MIT license"></a>
  <a href="https://bend-lang.com"><img src="https://img.shields.io/badge/Bend-2.0.39-8b83b5?style=flat-square&labelColor=4d4a44" alt="Bend 2.0.39"></a>
  <img src="https://img.shields.io/badge/laws%20proved-21-7e9a5e?style=flat-square&labelColor=4d4a44" alt="21 laws proved">
</p>

**llama.bend** runs Llama-architecture language models in [Bend 2](https://bend-lang.com): SmolLM2-135M, SmolLM2-360M
and SmolLM2-1.7B, on the CPU, with the same greedy tokens as PyTorch (`transformers`). One typed package,
[`llama/`](llama/), runs all three; only the config changes between them. Every weight matrix has its shape in its
type, computed from that config, so a weight in the wrong place does not compile, and the structural guarantees
(which key/value head each query head reads, that position t sees exactly positions 0..t, that the cache never writes
past its room, that the weights cover the checkpoint) are laws that Bend's kernel checks. The package is on BendHub as
`bend-ml-llama`; it builds on [bend-ml](https://github.com/nuxyel/bend-ml).

llama.bend is models and inference: loading the weights, the forward pass, the KV cache, the tokenizer and generation.
Training them is out of its scope: that is [BendTorch](https://github.com/nuxyel/bendtorch)'s, which fine-tunes
SmolLM2-135M and 360M against PyTorch in Bend 2.

<p align="center"><img src="docs/media/deck/frames/01-title.webp" alt="llama.bend v1.4: 21 laws proved, 3 models in one package, 0 different tokens against transformers" width="90%"></p>

## 01 · A weight in the wrong place is a type error

<p align="center"><img src="docs/media/shots/bad-weight.png" alt="bend rejects a layer whose q and k projections are swapped: expected Bands&lt;576n, 576n&gt;, observed Bands&lt;192n, 576n&gt;" width="85%"></p>

```python
# the weights of one layer, with their shapes from the config c (row = output, as PyTorch stores them)
type LW<-c: Cfg> is Type:
  LW{ln1: List<&2, F32>, q: TA.Bands<Cfg.dim(c), Cfg.dim(c)>, k: TA.Bands<Cfg.kvw(c), Cfg.dim(c)>, ...}
```

SmolLM2-135M has 9 query heads but only 3 key/value heads, so its `q` is 576 × 576 and its `k` is 192 × 576. Swapping
them ([`tests/bad_weight.bend`](tests/bad_weight.bend)) is rejected before anything runs. The same goes for the
key/value cache: its type `KV<cap, t>` carries its capacity and the number of positions written, and every write asks
for a proof that the position is below the capacity. A third write into a cache made for two positions needs a proof
that 2 < 2 ([`tests/bad_cap.bend`](tests/bad_cap.bend)):

<p align="center"><img src="docs/media/shots/bad-cap.png" alt="bend rejects a third write into a cache made for two positions: expected True, observed False" width="85%"></p>

The proofs are erased when the program is compiled: they cost nothing at run time. `make laws` checks that both
programs are rejected.

## 02 · Laws, not trust

<p align="center"><img src="docs/media/shots/law-gqa.png" alt="The law gqa_head as written in the source, and ALL PROOFS CHECK" width="85%"></p>

| Law | In plain language |
|---|---|
| `gqa_head` | with g query heads per key/value head, query head j·g + i reads key/value head j, the mapping `transformers` uses; proved over `Nat.div` itself |
| `causal`, `read_len` | at position t, attention scores exactly t + 1 keys, positions 0..t, one per position written |
| `kv_room` | a cache made for `cap` positions holds every position below `cap` inside its arrays, so a write never wraps around onto position 0 |
| `groups_len`, `kv_heads`, `arrays_len` | the query heads form exactly one group per key/value head, and an empty cache holds one array of keys and one of values per key/value head |
| `head_split` | a head splits into the two halves that RoPE rotates together and joins back unchanged; the config stores half a head, so a head is always even |
| `sample_below` | a token drawn with temperature, top-k and top-p is an id of the logits it was drawn from, whatever the settings and the draw |
| `layout_135m`, `layout_360m`, `layout_1_7b` | the shapes the types give to the weights add up to the number of values in each checkpoint (134,515,008, 361,821,120 and 1,711,376,384); the loader stops on any file whose size differs from its type |

Each one is explained at length in [`llama/README.md`](llama/README.md#the-laws-in-plain-language), next to the
arithmetic lemmas they use. `bend llama/main.bend --verdict` checks them again with Bend's kernel proved in Lean, with
no `@unsafe`, no `?TODO` and no axioms. The numbers are a different matter: `F32` is not a real number, so the numerics
are tested against PyTorch instead of proved.

## 03 · It runs, and it computes the same thing

<p align="center"><img src="docs/media/shots/gen.png" alt="SmolLM2-1.7B in Bend generates 'The capital of France is Paris.'" width="85%"></p>

All against `transformers` 5.19 in F32 with eager attention, by the scripts in [`reference/`](reference/):

| | SmolLM2-135M | SmolLM2-360M | SmolLM2-1.7B |
|---|---|---|---|
| config | 3 key/value heads × 3 query heads, 30 layers | 5 × 3, 32 layers | 32 × 1, 24 layers, RoPE base 130,000 |
| logits, worst error relative to the largest logit (11 prompts) | 2.1e-5 | 4.8e-5 | 8.3e-5 |
| argmax at each of the 88 positions | the same | the same | the same |
| 32 greedy tokens and decoded text (12 prompts) | identical | identical | identical |

The tokenizer is written in Bend too (SmolLM2's digit split, the GPT-2 regex and 48,900 merges). It gives the same ids
as `tokenizers` on 68 fixed texts and 600 random ones, and decodes like it on 1,500 random id sequences, most of them
invalid UTF-8. SmolLM2-Instruct's chat template, written in Bend on that tokenizer, gives the same ids as `transformers`'
`apply_chat_template` on 13 conversations, with and without the generation prompt. Sampling (temperature, top-k, top-p,
a seeded generator written in Bend) gives the same tokens and logit bits as a Python twin that repeats it operation
for operation: 180 draws on made-up logits (1 to 49,152 of them, with and without ties) and 92 draws of the 135M,
whose top-k candidates are also the top-k of `transformers`' logits for the same tokens.

## 04 · How it compares

<p align="center"><img src="docs/media/deck/frames/02-vs-references.webp" alt="SmolLM2-1.7B, ms per token over 128 tokens: llama.cpp 326 and 102, transformers 238 and 125, candle 298 and 308, llama.bend 1233 and 297, on 1 and 16 threads" width="90%"></p>

<p align="center"><img src="docs/media/deck/frames/03-three-models.webp" alt="ms per token for the three models on 1 and 16 threads, llama.bend next to transformers, llama.cpp and candle" width="90%"></p>

<p align="center"><img src="docs/media/deck/frames/04-flat-cost.webp" alt="Over 512 tokens a token costs about the same: growth per cached position between 0.04 and 0.21 ms" width="90%"></p>

<p align="center"><img src="docs/media/deck/frames/05-startup.webp" alt="SmolLM2-1.7B, load and generate one token: llama.bend 5.1 s, the others 0.8 to 3.2 s; peak memory 6.8 GB, the lowest" width="90%"></p>

Every system generates the same tokens. On the 1.7B llama.bend is 3.8 times slower than llama.cpp on one thread and
2.9 times on 16 (2026-11-01, Bend 2.0.39). Against the run of 2026-10-13 it is 17 to 31% faster per token on 16 threads,
and about the same on one (2 to 4% faster on the 135M and 360M, 2.5% slower on the 1.7B: gaps within the spread
between sessions). Loading the 1.7B takes 5 s (66 s
before the loader moved to `bend-ml-binary-read`). The machine, the versions and every number are in
[`BENCHMARK.md`](BENCHMARK.md).

Bend 2.0.39 is faster than 2.0.38 on the same commit. On the 135M, idle machine, 5 alternated rounds (2026-10-24), a
token takes 51.6 ms on 16 threads against 58.7, and 114 ms on one thread against 116, inside the spread. The program
runs 2.5% fewer instructions and 4% fewer cycles over a 33-token run (hardware counters, 3 rounds). Earlier, built with
2.0.36 and with 2.0.37 (which carries my fix for parallel code with an erased last parameter), the same commit ran at
the same speed: no def here had the shape that fix is about.

## 05 · Reproduce

Linux x86_64 (or WSL), clang, Python 3.14 (the version the pinned dependencies were tested with), `curl`. About 1.6 GB
of disk for the 135M: 1.2 GB for the Python venv with PyTorch, 0.4 GB for the model's files and the test data.

```bash
git clone https://github.com/nuxyel/llama.bend.git && cd llama.bend
make setup   # Bend 2.0.39 (SHA256-checked), Lean 4.34.0, the Python venv of reference/; no sudo
make prep    # SmolLM2-135M at a pinned revision into data/smollm2-135m/, checked
make check   # the laws (--verdict), the rejected programs, then every test above against tokenizers and transformers (~4 min)
bend demo/gen.bend -o gen && ./gen data/smollm2-135m "The capital of France is" 32
./gen data/smollm2-135m "Once upon a time" 32 0.8 50 0.95 42   # sampled: temperature, top-k, top-p, seed
./gen data/smollm2-135m --chat "What is the capital of France?" 32   # one user turn in the chat template
```

Greedy is the default and `temperature 0` is greedy too. A seed gives the same tokens on every run, on one thread as
on all of them (nothing is kept between draws). `--chat` wraps the prompt in SmolLM2-Instruct's template (default system message, then the user turn, then the
assistant's) and stops at `<|im_end|>`; it takes the sampling settings after the token count as well.

The other models are one variable away: `make prep check MODEL=smollm2-360m`, or `MODEL=smollm2-1.7b`.
Every loader reads the downloaded `model.safetensors` itself, so `make prep` writes no copy of the weights;
`make prep F32=1` also writes the F32 copies (`w/<name>.bin`, 6.4 GB for the 1.7B) that `bend-ml-llama@0.2.0.0`'s
loader reads.
`reference/models.py` pins each one to a revision. In your own Bend program:

```python
import bend-ml-llama@0.2.0.0/main.bend as M
```

`make bench` is a one-command benchmark for optimization loops: it builds the generator with the `bend` on your PATH,
generates 128 tokens on the 135M, fails if they differ from `transformers`' (`bench/`), and prints one line
(`ms_per_token=... load_s=... tokens=128 ok`). `MODEL=` and `THREADS=` pick the model and the thread count.

`make bench-gpu` runs the same check on `demo/gen_gpu.bend`, whose matrix · vector products run on the GPU through
bend-ml-tensor-gpu (`bend-ml-tensor-gpu@0.1.5.1` on BendHub). It needs CUDA (`CUDA_HOME`); `GPU=off` runs the
same program with its GPU calls on the CPU threads. It gives the same tokens and logit bits, and it is slower (section 06).

`make media` regenerates the figures on this page from real runs (needs node, ffmpeg and Brave). CI runs `make check`
on every push. `main` holds the released versions; work in progress lives on `devel`.
Contributions are welcome: see [`CONTRIBUTING.md`](CONTRIBUTING.md) (pull requests go to `devel`).

## 06 · Limits

- **Slower than the references** ([`BENCHMARK.md`](BENCHMARK.md), 2026-11-01, Bend 2.0.39). On the 1.7B over 128
  tokens at 16 threads, llama.bend takes 297 ms per generated token, `transformers` 125 and llama.cpp 101; on one
  thread, 1.23 s against 0.24 and 0.33, with no gain since 2026-10-13. The cost of a token barely grows with the
  position (0.2 ms per cached position at most). What remains is the weight products, and still loading: 5.1 s to load
  the 1.7B and generate one token, against 3.2 s for `transformers` and 0.8 s for llama.cpp. Peak memory 6.8 GB, on one
  thread as on 16, the lowest of the four.
- **The GPU path is slower than the CPU** (idle machine, 2026-11-01, Bend 2.0.39, [`NOTES.md`](NOTES.md)). With the products on an RTX
  4050 Laptop, over 128 tokens, the 135M takes 184 ms per token and the 360M 240, against 52.9 and 98.3 on 16 CPU threads
  ([`BENCHMARK.md`](BENCHMARK.md), 32 tokens): every product is a GPU call with a fixed cost of about 1 ms, and
  generation has one vector a token to amortize it over. The 1.7B does not fit in the card's 6 GB; with
  `+RTS --gpu 12GB -RTS` it runs oversubscribed at 1.56 s per token against 0.30 on 16 CPU threads, with 7.6 GB of host memory.
- Base models only. The chat template is in (`--chat`), but `reference/models.py` does not pin the instruct checkpoints
  yet, so a chat runs on a base model and answers like one. Sampling matches its Python twin, not `transformers`' own
  sampler (whose random numbers are PyTorch's); it runs on the CPU path only (`gen_gpu.bend` stays greedy), needs a
  top-k (k >= 1), and its cost grows with k (not timed yet). A special token written inside a chat message is encoded
  as text. The vectors between layers are lists whose length is not in their type, and the numerics are tested, never
  proved.

## Where things are

| Path | What |
|---|---|
| [`llama/`](llama/) | the package `bend-ml-llama`: config, layers, model, loader and laws (`main.bend`); the fast loader the demos use (`load.bend`) |
| [`llama/gpu.bend`](llama/gpu.bend) | generation with the products on the GPU (bend-ml-tensor-gpu), outside the package and its laws |
| [`demo/`](demo/) | `gen.bend` (generation: greedy, sampled, chat), `gen_gpu.bend` (greedy on the GPU), `forward.bend`, `layer_test.bend` and `sample_test.bend` (what the tests read), the tokenizer and chat template |
| [`tests/`](tests/) | programs that must not compile |
| [`reference/`](reference/) | download and preparation, the `transformers` and `tokenizers` oracles, the tests |
| [`BENCHMARK.md`](BENCHMARK.md) | the numbers in section 04, with the machine and the method |
| [`NOTES.md`](NOTES.md) | every finding, measurement and decision, by date |
| [`docs/plan.md`](docs/plan.md), [`docs/bendtorch.md`](docs/bendtorch.md) | the milestones; what a backward pass would need |

## Upstream

While building bend-ml I found that a parallel let became sequential when its def's last parameter was erased
([bendlang/bend#1374](https://github.com/bendlang/bend/issues/1374)). My fix,
[#1377](https://github.com/bendlang/bend/pull/1377), is in Bend 2.0.37 and later; this project runs on 2.0.39.

<p align="center"><sub>MIT · Renan Vinícius (@nuxyel) · the visual identity is bend-ml's, in Bend's own colours · the launch video (v1.0): <a href="https://github.com/nuxyel/llama.bend/releases/download/v1.0.0/llama.bend.mp4">llama.bend.mp4</a></sub></p>
