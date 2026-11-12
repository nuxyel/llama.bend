# llama.bend: setup and verification shortcuts. Run `make setup`, `make prep` once, then `make check`.
# MODEL picks the model (reference/models.py): `make prep check MODEL=smollm2-360m`.
export BEND_NO_TELEMETRY := 1
export MODEL ?= smollm2-135m
DIR := data/$(MODEL)
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
# the files of bend-ml-llama: the entry to publish (package.bend) and what it imports. load.bend (foreign code) and
# gpu.bend (an unsafe primitive of Base) stay outside
PKG := llama/main.bend llama/unicode.bend llama/tok.bend llama/prompt.bend llama/gen.bend llama/package.bend

.PHONY: setup prep laws check bits bench bench-gpu media lint

setup:            ## install the pinned Bend and Lean if missing, create the Python venv
	scripts/setup.sh

prep:             ## download MODEL (pinned revision) and write data/MODEL/ (135M: 270 MB down, under 1 MB written); F32=1: also the F32 copies that bend-ml-llama 0.2.0.0 reads (135M: 540 MB, 1.7B: 6.4 GB)
	$(PY) reference/prep.py

laws:             ## every file of the package checked by bend and again by the kernel proved in Lean (--verdict); tests/bad_*.bend rejected, each with its own error; the package free of foreign code and whole in its entry
	$(PY) scripts/check_rts.py --self-test
	$(PY) scripts/check_rts.py
	! grep -n "@unsafe\|?TODO" llama/*.bend demo/*.bend tests/*.bend
	! grep -n "binary-read\|^import .*load\.bend\|^import \.\./" $(PKG)
	! grep -n "tensor-gpu\|gpu.bend" $(PKG) llama/load.bend tests/*.bend
	for f in $(filter-out llama/package.bend,$(PKG)); do grep -q "^import \./$${f#llama/} as " llama/package.bend || { echo "llama/package.bend does not import $$f: --publish would leave it out"; exit 1; }; done
	for f in $(PKG); do echo "== $$f"; bend $$f && bend $$f --verdict | tee /dev/stderr | tail -1 | grep -qx "ALL PROOFS CHECK" || exit 1; done
	scripts/check_rejected.sh

check: laws       ## the laws, prep, the Bend tokenizer and chat template against `tokenizers`/`transformers`; one layer, the forward and generation against transformers; conversations of three turns, their caches kept, against transformers generating each turn from the whole conversation; the prompt in one pass against position by position; sampling against its Python twin; demo/hello.bend, built from a copy of the package's files alone, against transformers' tokens
	$(PY) reference/prep.py --check
	mkdir -p .scratch
	bend demo/tok.bend -o .scratch/tok
	$(PY) reference/test_tok.py .scratch/tok
	$(PY) reference/test_detok.py .scratch/tok
	$(PY) reference/test_chat.py .scratch/tok
	[ -f $(DIR)/test/inv_freq.bin ] || $(PY) reference/layer_ref.py
	bend demo/layer_test.bend -o .scratch/layer_test
	$(PY) reference/test_layer.py .scratch/layer_test
	[ -f $(DIR)/test/fwd10_logits.bin ] || $(PY) reference/forward_ref.py
	bend demo/forward.bend -o .scratch/forward
	$(PY) reference/test_forward.py .scratch/forward
	[ -f $(DIR)/test/gen.json ] || $(PY) reference/gen_ref.py
	bend demo/gen.bend -o .scratch/gen
	$(PY) reference/test_gen.py .scratch/gen
	[ -f $(DIR)/test/chat_gen.json ] || $(PY) reference/chat_gen_ref.py
	$(PY) reference/test_chat_gen.py .scratch/gen
	[ -f $(DIR)/test/chat_turns.json ] || $(PY) reference/chat_turns_ref.py
	$(PY) reference/test_chat_turns.py .scratch/gen
	$(PY) reference/test_prompt.py .scratch/gen
	bend demo/sample_test.bend -o .scratch/sample_test
	$(PY) reference/test_sample.py .scratch/sample_test .scratch/gen
	[ -f bench/expected-$(MODEL).txt ] || $(PY) reference/bench_ref.py
	rm -rf .scratch/pkg && mkdir -p .scratch/pkg/llama .scratch/pkg/demo && cp $(PKG) .scratch/pkg/llama && cp demo/hello.bend .scratch/pkg/demo
	bend .scratch/pkg/demo/hello.bend -o .scratch/hello
	$(PY) reference/test_hello.py .scratch/hello
	$(PY) scripts/rts_probe.py .scratch/gen $(DIR) "Once upon a time" 24

bits:             ## forward and gen give the same bits as the saved ones ($(DIR)/test/bits/; `bits.py save` writes them)
	mkdir -p .scratch
	bend demo/forward.bend -o .scratch/forward
	bend demo/gen.bend -o .scratch/gen
	$(PY) reference/bits.py check .scratch/forward .scratch/gen

bench:            ## one line: ms per token by the program's own clock, failing if the tokens differ from transformers'
	scripts/bench.sh

bench-gpu:        ## make bench with the products on the GPU (demo/gen_gpu.bend, needs CUDA_HOME); GPU=off: its bangs on the CPU
	GPU=$${GPU:-1} scripts/bench.sh

media:            ## regenerate the README images from docs/media/outputs/ and numbers.json (needs node, ffmpeg, Brave)
	scripts/make_media.sh

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate; measure a finding before taking it
	bunx bend-lint --jobs max
