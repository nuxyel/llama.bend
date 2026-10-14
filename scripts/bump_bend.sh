#!/usr/bin/env bash
# Moves the Bend pin to a new release: downloads the two Linux tarballs from the release, computes their SHA256
# (the release publishes no checksums file) and rewrites the pin where it is a pin: scripts/setup.sh (version and
# both sums), the CI step's name, the README badge and setup line, the package README and CLAUDE.md's rule 1 and
# pitfalls title. Dated text (NOTES.md, BENCHMARK.md) is history and stays. It changes no code and runs no check:
# the steps after it are in docs/release.md.
#
#   scripts/bump_bend.sh 2.0.37
set -euo pipefail
NEW=${1:?usage: scripts/bump_bend.sh <version>}
cd "$(dirname "$0")/.."
OLD=$(sed -n 's/^BEND_VERSION=//p' scripts/setup.sh)
[ "$OLD" != "$NEW" ] || { echo "already on $NEW"; exit 0; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
sum() { # arch
  local name="bend-$NEW-linux-$1.tar.gz"
  curl --proto '=https' --tlsv1.2 -fsSL -o "$TMP/$name" "https://github.com/bendlang/bend/releases/download/v$NEW/$name"
  sha256sum "$TMP/$name" | awk '{print $1}'
}
X64=$(sum x64); ARM=$(sum arm64)
OLDX64=$(sed -n 's/.*Linux-x86_64)  OS=linux; ARCH=x64;   SHA=\([0-9a-f]*\) ;;/\1/p' scripts/setup.sh)
OLDARM=$(sed -n 's/.*Linux-aarch64) OS=linux; ARCH=arm64; SHA=\([0-9a-f]*\) ;;/\1/p' scripts/setup.sh)
[ -n "$OLDX64" ] && [ -n "$OLDARM" ] || { echo "setup.sh: SHA lines not found"; exit 1; }
sed -i "s/$OLDX64/$X64/; s/$OLDARM/$ARM/; s/^BEND_VERSION=$OLD$/BEND_VERSION=$NEW/; s/Bend $OLD/Bend $NEW/g" scripts/setup.sh
sed -i "s/Set up (Bend $OLD,/Set up (Bend $NEW,/" .github/workflows/ci.yml
sed -i "s/\[!\[Bend $OLD\](https:\/\/img.shields.io\/badge\/Bend-$OLD-/[![Bend $NEW](https:\/\/img.shields.io\/badge\/Bend-$NEW-/; s/installs Bend $OLD (SHA256-checked)/installs Bend $NEW (SHA256-checked)/" README.md
sed -i "s/^- Bend: \*\*$OLD\*\*/- Bend: **$NEW**/" llama/README.md
sed -i "s/Bend is pinned to \*\*$OLD\*\*/Bend is pinned to **$NEW**/; s/^## Bend $OLD pitfalls/## Bend $NEW pitfalls/" CLAUDE.md
echo "Bend $OLD -> $NEW"
echo "  linux-x64   $X64"
echo "  linux-arm64 $ARM"
git --no-pager diff --stat
echo "next: docs/release.md, from step 2"
