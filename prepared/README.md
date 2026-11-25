# prepared/

The small files the Bend programs read besides the weights, as `reference/prep.py` writes them, so a model runs
without Python: `scripts/fetch.sh` (`make fetch`) copies them into `data/<MODEL>/` and downloads `model.safetensors`
at the pinned revision, checked by its SHA256.

| file | what it is | from |
|---|---|---|
| `tokenizer/merges_num.txt` | one BPE rule per line, `a b`, in bend-ml's numbering (byte b is b, rule k is 256 + k) | `tokenizer.json` |
| `tokenizer/perm.txt` | the model id of each of the 256 bytes, 0 for the 21 bytes without a token | `tokenizer.json` |
| `<model>/cfg.txt` | the config `llama/main.bend` reads: kv, g, half, f, layers, vocab, then the F32 bits of `rms_norm_eps` and `rope_theta` | `config.json` |
| `<model>/st.txt` | where each tensor starts in `model.safetensors`, two numbers per tensor (`llama/load.bend`) | the safetensors header |

The six models share one `tokenizer.json` (SHA256 `9ca9acdd...`, `reference/models.py`), so the two tables are the
same for all of them. `make check` compares every file here with what `prep.py` writes for the model it checks, so
they cannot drift from it.

These files are derived from SmolLM2 by Hugging Face (https://huggingface.co/HuggingFaceTB), released under the
Apache License 2.0, a copy of which is in `LICENSE-Apache-2.0.txt`. They are a numeric rewrite of its configuration,
its tokenizer's merges and vocabulary, and its weight file's header; the weights themselves are not here.
