#!/usr/bin/env bash
# One-command benchmark for optimization loops: builds demo/gen.bend with the `bend` on PATH (or $BEND), generates
# N tokens greedily from "Once upon a time", fails if the ids differ from transformers' (bench/expected-<MODEL>.txt),
# and prints one line:
#
#   ms_per_token=56.3 load_s=4.5 tokens=128 ok
#
# ms_per_token is the program's own clock: (time of the last token - time of the first) / (tokens - 1), so neither
# loading nor the prompt is in it. Exit status: 0 ok, 1 wrong tokens, 2 setup error.
#
#   scripts/bench.sh                 SmolLM2-135M, 128 tokens, all threads (needs `make prep`)
#   MODEL=smollm2-360m THREADS=1 scripts/bench.sh
#   BEND=/path/to/bend scripts/bench.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-smollm2-135m}
BEND=${BEND:-bend}
command -v "$BEND" >/dev/null || BEND=$HOME/.bend/bin/bend
EXP=bench/expected-$MODEL.txt
DIR=data/$MODEL
[ -f "$EXP" ] || { echo "no $EXP (MODEL=$MODEL reference/.venv/bin/python reference/bench_ref.py)"; exit 2; }
[ -f "$DIR/cfg.txt" ] || { echo "no $DIR (make prep MODEL=$MODEL)"; exit 2; }
want=$(grep -v '^#' "$EXP")
n=$(echo "$want" | wc -w)
mkdir -p .scratch
export BEND_NO_TELEMETRY=1
"$BEND" demo/gen.bend -o .scratch/bench-gen > .scratch/bench-build.log 2>&1 || { cat .scratch/bench-build.log; echo "build failed"; exit 2; }
out=$(.scratch/bench-gen ${THREADS:+--threads $THREADS} "$DIR" "Once upon a time" "$n") || { echo "run failed"; exit 2; }
got=$(echo "$out" | awk '/^  id /{printf "%s ", $2}' | sed 's/ $//')
line=$(echo "$out" | awk -v n="$n" '
  /^weights loaded in / { sub(/s$/, "", $4); load = $4 }
  /^  id / { sub(/^t=/, "", $NF); sub(/s$/, "", $NF); t[k++] = $NF }
  END { if (k < 2) exit 1; printf "ms_per_token=%.1f load_s=%.1f tokens=%d", (t[k-1] - t[0]) * 1000 / (k - 1), load, k }')
if [ "$got" != "$want" ]; then
  echo "$line WRONG TOKENS"
  echo "want: $want"
  echo "got:  $got"
  exit 1
fi
echo "$line ok"
