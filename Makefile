# llama.bend: setup and verification shortcuts. Run `make setup`, `make prep` once, then `make check`.
# MODEL picks the model (reference/models.py): `make prep check MODEL=smollm2-360m`.
export BEND_NO_TELEMETRY := 1
export MODEL ?= smollm2-135m
DIR := data/$(MODEL)
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python

.PHONY: setup prep laws check bits bench media

setup:            ## install the pinned Bend and Lean if missing, create the Python venv
	scripts/setup.sh

prep:             ## download MODEL (pinned revision) and write data/MODEL/ (135M: 270 MB down, 540 MB written)
	$(PY) reference/prep.py

laws:             ## the package's proofs, checked by bend and again by the kernel proved in Lean (--verdict); tests/bad_*.bend rejected
	! grep -n "@unsafe\|?TODO" llama/*.bend demo/*.bend tests/*.bend
	bend llama/main.bend
	bend llama/main.bend --verdict | tee /dev/stderr | tail -1 | grep -qx "ALL PROOFS CHECK"
	for f in tests/bad_*.bend; do bend $$f 2>&1 | grep -q "^- expected" && echo "rejected as it must be: $$f" || { echo "COMPILED, but must not: $$f"; exit 1; }; done

check: laws       ## the laws, prep, the Bend tokenizer against `tokenizers`; one layer, the forward and generation against transformers
	$(PY) reference/prep.py --check
	mkdir -p .scratch
	bend demo/tok.bend -o .scratch/tok
	$(PY) reference/test_tok.py .scratch/tok
	$(PY) reference/test_detok.py .scratch/tok
	[ -f $(DIR)/test/inv_freq.bin ] || $(PY) reference/layer_ref.py
	bend demo/layer_test.bend -o .scratch/layer_test
	$(PY) reference/test_layer.py .scratch/layer_test
	[ -f $(DIR)/test/fwd10_logits.bin ] || $(PY) reference/forward_ref.py
	bend demo/forward.bend -o .scratch/forward
	$(PY) reference/test_forward.py .scratch/forward
	[ -f $(DIR)/test/gen.json ] || $(PY) reference/gen_ref.py
	bend demo/gen.bend -o .scratch/gen
	$(PY) reference/test_gen.py .scratch/gen

bits:             ## forward and gen give the same bits as the saved ones ($(DIR)/test/bits/; `bits.py save` writes them)
	mkdir -p .scratch
	bend demo/forward.bend -o .scratch/forward
	bend demo/gen.bend -o .scratch/gen
	$(PY) reference/bits.py check .scratch/forward .scratch/gen

bench:            ## one line: ms per token by the program's own clock, failing if the tokens differ from transformers'
	scripts/bench.sh

media:            ## regenerate the README images from docs/media/outputs/ and numbers.json (needs node, ffmpeg, Brave)
	scripts/make_media.sh
