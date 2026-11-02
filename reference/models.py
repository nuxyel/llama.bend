"""The models llama.bend runs, each pinned to a revision, and where their files live.

The scripts in reference/ work on the model named by the environment variable MODEL (default smollm2-135m), in
data/<MODEL>/: hf/ (as downloaded), w/ and cfg.txt (what Bend reads), the tokenizer tables, and test/ (what the
oracles write). The Bend programs take that directory as their first argument.
"""
import json, os

MODELS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),  # checked 2026-10-09
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),  # checked 2026-10-10
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"),  # checked 2026-10-10
}

# SmolLM2-Instruct's chat template, from tokenizer_config.json at these revisions (checked 2026-11-01; the same text in
# the three). Their tokenizer.json is the base models' file byte for byte (sha256 9ca9acdd...), so the template is
# applied with the base tokenizer in data/<MODEL>/hf and the instruct checkpoints are not downloaded.
INSTRUCT = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
}
CHAT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MODEL", "smollm2-135m")
if MODEL not in MODELS:
    raise SystemExit(f"MODEL={MODEL!r}: expected one of {', '.join(MODELS)}")
REPO, REV = MODELS[MODEL]
D = os.path.join(ROOT, "data", MODEL)
HF = os.path.join(D, "hf")
W = os.path.join(D, "w")
T_DIR = os.path.join(D, "test")


def config():
    return json.load(open(os.path.join(HF, "config.json")))


def sizes():
    """hidden size, query heads, key/value heads, head size, layers, vocabulary"""
    c = config()
    heads = c["num_attention_heads"]
    hd = c.get("head_dim") or c["hidden_size"] // heads
    return c["hidden_size"], heads, c["num_key_value_heads"], hd, c["num_hidden_layers"], c["vocab_size"]
