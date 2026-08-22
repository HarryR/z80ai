#!/bin/sh
# Build every shipped example for every target into a directory.
#
# This is exactly what CI builds and releases, so running it locally reproduces
# the release artifacts byte for byte.
#
#   ./build-examples.sh [output-dir]     (default: dist)
#
# Verify the results with:
#
#   python verify_artifacts.py --dist dist

set -eu

OUT="${1:-dist}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT"

for example in guess:GUESS tinychat:CHAT; do
    model="examples/${example%%:*}/model.npz"
    name="${example##*:}"

    "$PYTHON" buildz80com.py     -m "$model" -o "$OUT/$name.COM"
    "$PYTHON" buildfastz80com.py -m "$model" -o "$OUT/$name-FAST.COM"
    "$PYTHON" buildz80tap.py     -m "$model" -o "$OUT/$name.TAP"
    "$PYTHON" buildez80.py       -m "$model" -o "$OUT/$name.bin"
done

echo
echo "Built into $OUT/:"
ls -l "$OUT"
