#!/usr/bin/env bash
# The model without Python: data/<MODEL>/ as the Bend programs read it, from the repository's prepared/ files and the
# weights downloaded at the pinned revision. Needs curl and sha256sum, no venv.
#
#   scripts/fetch.sh                      SmolLM2-135M (270 MB down)
#   MODEL=smollm2-360m scripts/fetch.sh   any model of reference/models.py
#
# prepared/ holds what reference/prep.py writes from the model's config.json and tokenizer.json (cfg.txt, the
# tokenizer tables merges_num.txt and perm.txt, the same for the six models, and w/st.txt, where each tensor starts in
# model.safetensors); `make check` compares it with prep.py's output. The weights are the Hub's model.safetensors,
# checked by the SHA256 its revision lists. What it gives is enough for demo/gen.bend, demo/hello.bend and `make
# bench`; `make check` still needs `make setup prep` (the oracles are Python).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-smollm2-135m}

# model: repository, revision (reference/models.py's) and model.safetensors's SHA256 at that revision
case "$MODEL" in
  smollm2-135m)          REPO=HuggingFaceTB/SmolLM2-135M;          REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2; SHA=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1 ;;
  smollm2-360m)          REPO=HuggingFaceTB/SmolLM2-360M;          REV=f8027fd0eaeea54caa13c31d31b9fdc459c38b49; SHA=7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f ;;
  smollm2-1.7b)          REPO=HuggingFaceTB/SmolLM2-1.7B;          REV=effd688a12921b4cc83e3312b6feb579f70f9c71; SHA=1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0 ;;
  smollm2-135m-instruct) REPO=HuggingFaceTB/SmolLM2-135M-Instruct; REV=12fd25f77366fa6b3b4b768ec3050bf629380bac; SHA=5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c ;;
  smollm2-360m-instruct) REPO=HuggingFaceTB/SmolLM2-360M-Instruct; REV=a10cc1512eabd3dde888204e902eca88bddb4951; SHA=e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e ;;
  smollm2-1.7b-instruct) REPO=HuggingFaceTB/SmolLM2-1.7B-Instruct; REV=31b70e2e869a7173562077fd711b654946d38674; SHA=f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8 ;;
  *) echo "MODEL=$MODEL: expected one of reference/models.py's"; exit 2 ;;
esac

D=data/$MODEL
mkdir -p "$D/hf" "$D/w"
cp prepared/"$MODEL"/cfg.txt "$D/cfg.txt"
cp prepared/"$MODEL"/st.txt "$D/w/st.txt"
cp prepared/tokenizer/merges_num.txt prepared/tokenizer/perm.txt "$D/"

ST=$D/hf/model.safetensors
if [ -f "$ST" ] && echo "$SHA  $ST" | sha256sum -c --status -; then
  echo "$ST: already there"
else
  echo "$MODEL: downloading model.safetensors at $REV"
  curl --proto '=https' --tlsv1.2 -fL --retry 3 --progress-bar -o "$ST.part" "https://huggingface.co/$REPO/resolve/$REV/model.safetensors"
  echo "$SHA  $ST.part" | sha256sum -c --status - || { echo "SHA256 mismatch for $REPO@$REV model.safetensors"; rm -f "$ST.part"; exit 1; }
  mv "$ST.part" "$ST"
fi
echo "$D ready: bend demo/gen.bend -o gen && ./gen $D \"Once upon a time\" 32"
