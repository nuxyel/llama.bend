#!/usr/bin/env bash
# Builds fork_wake.bend and its sequential twin (pair(0n, ...) in go) and runs both under /usr/bin/time: wall, user
# and system time and voluntary context switches. Usage: fork_wake.sh [threads] (default 16); needs bend and GNU time.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
sed 's/pair(1n, spin(400000n, x))/pair(0n, spin(400000n, x))/' fork_wake.bend > "$T/seq.bend"
grep -q 'pair(0n, spin(400000n, x))' "$T/seq.bend"
bend fork_wake.bend -o "$T/par" > /dev/null
bend "$T/seq.bend" -o "$T/seq" > /dev/null
for p in par seq; do
  /usr/bin/time -f "$p: %e s wall, %U s user, %S s system, %w voluntary context switches" \
    "$T/$p" +RTS --threads "${1:-16}" --gpu off -RTS > /dev/null
done
