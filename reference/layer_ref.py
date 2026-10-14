"""Expected values for one layer (milestone 3), from `transformers` in F32 with eager attention, for the model
MODEL (models.py).

  reference/.venv/bin/python reference/layer_ref.py

Writes to data/<MODEL>/test/, as little-endian F32 arrays of T rows (widths for SmolLM2-135M: hidden 576, keys 192):
  <case>_in.bin                   the input of the layer (T x hidden)
  <case>_<name>.bin               every intermediate, from forward hooks on the layer:
    n1    input_layernorm output (hidden)     q, k, v   projections before RoPE (hidden, keys, keys)
    qr, kr  q and k after RoPE, from transformers' own apply_rotary_pos_emb (hidden, keys)
    attn  o_proj input (hidden)               a         o_proj output (hidden)
    n2    post_attention_layernorm (hidden)   m         mlp output (hidden)
    y     the layer output (hidden)
  inv_freq.bin                    rotary_emb.inv_freq (head size / 2)
Cases: A_L0 and A_L15 (the real hidden states of a prompt entering layers 0 and 15), B_L0 (40 positions of
Gaussian noise at the embedding scale, seed 0, entering layer 0).
"""
import os
import numpy as np
import torch

from models import HF, T_DIR as OUT, sizes

PROMPT = "The capital of France is the city of"


def capture(model, layer, embeds, case):
    """Runs the whole model on embeds and saves layer `layer`'s input, intermediates and output."""
    lay = model.model.layers[layer]
    got = {}

    def out_hook(name):
        return lambda mod, inp, out: got.__setitem__(name, (out[0] if isinstance(out, tuple) else out).detach())

    hooks = [
        lay.input_layernorm.register_forward_hook(out_hook("n1")),
        lay.self_attn.q_proj.register_forward_hook(out_hook("q")),
        lay.self_attn.k_proj.register_forward_hook(out_hook("k")),
        lay.self_attn.v_proj.register_forward_hook(out_hook("v")),
        lay.self_attn.o_proj.register_forward_pre_hook(lambda mod, inp: got.__setitem__("attn", inp[0].detach())),
        lay.self_attn.o_proj.register_forward_hook(out_hook("a")),
        lay.post_attention_layernorm.register_forward_hook(out_hook("n2")),
        lay.mlp.register_forward_hook(out_hook("m")),
    ]
    with torch.no_grad():
        hs = model.model(inputs_embeds=embeds, output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    t = embeds.shape[1]
    hidden, heads, kv, hd = sizes()[:4]
    cos, sin = model.model.rotary_emb(embeds, torch.arange(t)[None, :])
    q4 = got["q"].reshape(1, t, heads, hd).transpose(1, 2)
    k4 = got["k"].reshape(1, t, kv, hd).transpose(1, 2)
    qr, kr = apply_rotary_pos_emb(q4, k4, cos, sin)
    got["qr"] = qr.transpose(1, 2).reshape(1, t, heads * hd)
    got["kr"] = kr.transpose(1, 2).reshape(1, t, kv * hd)
    got["in"] = hs[layer]
    got["y"] = hs[layer + 1]
    for name, t in got.items():
        a = t.reshape(t.shape[-2], t.shape[-1]).float().numpy().astype("<f4")
        a.tofile(os.path.join(OUT, f"{case}_{name}.bin"))
    print(f"{case}: {embeds.shape[1]} positions, " + ", ".join(f"{k} {tuple(v.shape[-2:])}" for k, v in got.items()))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(HF)
    ids = torch.tensor([tok(PROMPT, add_special_tokens=False)["input_ids"]])
    emb = model.model.embed_tokens(ids).detach()
    capture(model, 0, emb, "A_L0")
    capture(model, 15, emb, "A_L15")
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(1, 40, sizes()[0], generator=g) * emb.std()
    capture(model, 0, noise, "B_L0")
    model.model.rotary_emb.inv_freq.float().numpy().astype("<f4").tofile(os.path.join(OUT, "inv_freq.bin"))
    print("ids", ids.tolist()[0])


if __name__ == "__main__":
    main()
