#!/bin/sh
# Every tests/bad_*.bend must be rejected by the checker, and for the reason its header names. Each file carries
#   # expected: <text>    found in the error's "- expected :" line (required)
#   # observed: <text>    found in its "- observed :" line
#   # location: <def>     the error's "Location:"
# and the run must exit with an error and print "SOME PROOFS FAIL". A program rejected for another reason (a
# missing import, a syntax error, a different mismatch) fails this check like a program that compiles.
#
#   scripts/check_rejected.sh [bend]      (default: $BEND, else bend)
#
# Prints one line per program and exits 1 if any is not rejected as it must be. It reads bend's status itself,
# not through a pipe: under `set -o pipefail` (GitHub's `shell: bash`) `bend f | grep -q ...` takes bend's exit
# status, which is 1 for every rejected program, so a loop written that way reported rejected programs as compiled
# (issues #9 and #13).
set -u
cd "$(dirname "$0")/.."
B=${1:-${BEND:-bend}}
fail=0
for f in tests/bad_*.bend; do
  out=$("$B" "$f" 2>&1)
  st=$?
  why=
  [ "$st" -ne 0 ] || why="exit status 0"
  [ -n "$why" ] || printf '%s\n' "$out" | grep -qx "SOME PROOFS FAIL" || why="no SOME PROOFS FAIL"
  for key in expected observed location; do
    [ -n "$why" ] && break
    want=$(sed -n "s/^# $key: //p" "$f" | head -1)
    if [ -z "$want" ]; then
      [ "$key" = expected ] && why="no '# expected:' line in the file"
      continue
    fi
    case $key in
      location) got=$(printf '%s\n' "$out" | grep -m1 "^Location: ") ;;
      *) got=$(printf '%s\n' "$out" | grep -m1 "^- $key : ") ;;
    esac
    printf '%s\n' "$got" | grep -qF -- "$want" || why="$key: wanted '$want', got '${got:-nothing}'"
  done
  if [ -z "$why" ]; then
    echo "rejected as it must be: $f"
  else
    echo "NOT REJECTED as it must be: $f ($why)"
    fail=1
  fi
done
exit $fail
