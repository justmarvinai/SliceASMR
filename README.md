# SliceASMR: 1 slice vs 10 vs 25 vs 100 vs 250 vs 500 vs 1000 slices

A 9:16 "oddly satisfying" video of a rainbow jelly cube that flies in, gets sliced by
laser combs and drops onto a glossy studio floor: first sliced once, then into 10, 25,
100, 250, 500 and finally 1000 pieces. It runs about 65 seconds.

No Blender, no game engine and no GPU are involved. Everything is code in this repository:

| Part | What it does | Where |
|---|---|---|
| Physics | Real rigid-body dynamics with [Rapier](https://rapier.rs) (friction, restitution, stacking, CCD) at 240 Hz | `sim/physics.mjs`, `sim/physics.py` |
| Jelly wobble | Every piece carries a damped strain oscillator. Contact forces from the simulation drive it, so a piece squashes and wobbles (keeping its volume) where it is hit | `sim/jelly.py` |
| Scene | Stages, the cube's fly-in, laser passes, release kicks, camera that follows the pile, light rig | `sim/stages.py`, `sim/build.py`, `sim/look.py` |
| Renderer | A purpose-built CPU path tracer (C++/OpenMP) | `render/slicer.cpp` |
| Sound | Procedural ASMR sound design: whooshes, laser sizzle and "shhk" cuts, and one jelly impact per physics collision, panned by position | `audio/synth.py` |
| Titles and encode | Animated titles, progress row and end card, then H.264 and AAC | `tools/compose.py` |

## Renderer highlights

* Every piece is a convex polyhedron under an affine map (the rigid transform plus the jelly
  strain). Ray entry and exit are therefore analytic, which makes these cheap:
  * refraction with exact Fresnel and total internal reflection,
  * Beer-Lambert absorption that varies with the 3D rainbow, so cut faces show the inside colours,
  * single scattering of the area lights, integrated in closed form along each segment
    (no noise), plus a multiple-scattering glow term.
* Area-light soft shadows with coloured transmittance through the jelly, a glossy floor
  (GGX with multiple importance sampling) and short-range AO.
* Shading-only rounded edges, like a bevel shader.
* Laser beams and glowing cut sheets. Until the drop, the sliced cube is rendered as one
  volume with "virtual" kerf planes (with total internal reflection in the thin air gaps).
* Motion blur (8 time slices per frame, with per-sample interpolation of the lasers), depth
  of field, adaptive sampling with shuffled, digitally shifted Sobol (0,2) points, bloom, AgX tone
  mapping, vignette and film grain.

## Run it

Requirements: Python 3 with `numpy`, `scipy`, `pillow`; Node 18+; g++ with OpenMP; libpng;
ffmpeg with libx264.

```bash
pip install numpy scipy pillow
PREVIEW=1 ./run_all.sh     # low-res preview, a few minutes
./run_all.sh               # 1080x1920 final (a few hours on 4 CPU cores)
```

The result is written to `output/`.
