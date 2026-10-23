#!/usr/bin/env bash
# Renders every stage of a build directory into <work>/frames (resumable: existing frames are skipped
# by rendering only the missing tail of each stage).
set -euo pipefail
WORK=${1:-build/final}
cd "$(dirname "$0")/.."
mkdir -p "$WORK/frames"
for k in 0 1 2 3 4 5 6; do
  echo "== stage $k $(date +%H:%M:%S)"
  ./render/slicer "$WORK/stage_$k.bin" "$WORK/frames" 0 99999 --skip-existing
done
echo "== all done $(date +%H:%M:%S)"
