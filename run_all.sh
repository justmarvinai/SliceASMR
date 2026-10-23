#!/usr/bin/env bash
# Builds the whole video from scratch: physics -> renderer input -> frames -> sound -> mp4.
#
#   ./run_all.sh                 full quality (1080x1920, 30 fps)
#   PREVIEW=1 ./run_all.sh       quick low-resolution preview
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${PREVIEW:-0}" == "1" ]]; then
  WORK=build/preview; RES="360 640"; SPP="4 4"; OUT=output/slice_asmr_preview.mp4
else
  WORK=build/final; RES="1080 1920"; SPP="4 16"; OUT=output/slice_asmr_1_vs_1000_slices.mp4
fi
mkdir -p "$WORK/frames" output

( cd sim && [[ -d node_modules ]] || npm install --silent )
make -C render >/dev/null

python3 sim/build.py "$WORK" --res $RES --spp $SPP
for k in 0 1 2 3 4 5 6; do
  ./render/slicer "$WORK/stage_$k.bin" "$WORK/frames" 0 99999
done
python3 audio/synth.py "$WORK/timeline.json" "$WORK/audio.wav"
python3 tools/compose.py "$WORK/frames" "$WORK/timeline.json" "$WORK/audio.wav" "$OUT"
echo "done: $OUT"
