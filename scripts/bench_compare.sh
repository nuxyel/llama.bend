#!/usr/bin/env bash
# Compares `make bench` under two Bend binaries on this machine, as the benchmarks here are measured: idle, on mains
# power, in the `performance` profile, alternated and shuffled rounds, "1 thread" pinned to a performance core
# (taskset -c 3) and "16 threads" unpinned. Prints the medians and a line for bench/history.md.
#
#   scripts/bench_compare.sh <old bend> <new bend> [rounds]      (default 5 rounds; MODEL= as in bench.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
OLD=${1:?usage: scripts/bench_compare.sh <old bend> <new bend> [rounds]}; NEW=${2:?}; R=${3:-5}
online=$(cat /sys/class/power_supply/*/online 2>/dev/null | grep -c 1)
prof=$(powerprofilesctl get 2>/dev/null)
if [ "$online" = 0 ] || [ "$prof" != performance ]; then
  [ "${FORCE:-0}" = 1 ] || { echo "refusing to measure: mains power ${online/0/off}, profile '$prof' (FORCE=1 overrides)"; exit 2; }
fi
vo=$("$OLD" version | awk '{print $NF}'); vn=$("$NEW" version | awk '{print $NF}')
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
one() { # bend threads
  local b=$1 t=$2 out
  if [ "$t" = 1 ]; then out=$(BEND="$b" THREADS=1 taskset -c 3 scripts/bench.sh); else out=$(BEND="$b" THREADS="$t" scripts/bench.sh); fi
  echo "$out" | grep -q " ok$" || { echo "FAILED with $b, $t threads: $out" >&2; exit 1; }
  echo "$out" | sed -n 's/.*ms_per_token=\([0-9.]*\).*/\1/p'
}
for r in $(seq 1 "$R"); do
  printf '%s\n' "o 1" "o 16" "n 1" "n 16" | shuf | while read -r w t; do
    b=$([ "$w" = o ] && echo "$OLD" || echo "$NEW")
    v=$(one "$b" "$t") || exit 1
    echo "$v" >> "$TMP/$w-$t"
    echo "round $r: Bend $([ "$w" = o ] && echo "$vo" || echo "$vn"), $t thread$([ "$t" = 1 ] || echo s): $v ms/token"
  done || exit 1
done
med() { sort -n "$1" | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2}'; }
o1=$(med "$TMP/o-1"); o16=$(med "$TMP/o-16"); n1=$(med "$TMP/n-1"); n16=$(med "$TMP/n-16")
echo
echo "medians over $R rounds, ms per token (${MODEL:-smollm2-135m}, 128 tokens):"
echo "  Bend $vo: $o1 (1 thread), $o16 (16 threads)"
echo "  Bend $vn: $n1 (1 thread), $n16 (16 threads)"
echo
echo "for bench/history.md:"
echo "| $(date +%F) | $vn | $(git rev-parse --short HEAD) | $n1 | $n16 | $vo: $o1 / $o16 | $R |"
