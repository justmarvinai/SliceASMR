# SliceASMR: 1 slice vs 10 vs 25 vs 100 vs 250 vs 500 vs 1000 slices

A 9:16 "oddly satisfying" video, 64.6 seconds at 1080x1920 and 30 fps, with sound.
A rainbow jelly cube flies in, laser combs slice it, and it drops onto a glossy studio
floor. It is first sliced once, then into 10, 25, 100, 250, 500 and finally 1000 pieces.

**The video:** [`output/slice_asmr_1_vs_1000_slices.mp4`](output/slice_asmr_1_vs_1000_slices.mp4)

It was made without Blender, without a game engine and without a GPU. Everything is
code in this repository:

| Part | What it does | Where |
|---|---|---|
| Physics | Real rigid-body dynamics with [Rapier](https://rapier.rs) at 240 Hz: friction, restitution, stacking, CCD | `sim/physics.mjs`, `sim/physics.py` |
| Jelly wobble | Every piece carries a damped strain oscillator, driven by the contact forces from the simulation. A piece squashes, bulges (keeping its volume) and wobbles about the side it was hit on | `sim/jelly.py` |
| Scene | The seven stages, the cube's fly-in, the laser passes, how the pieces are released, a camera that follows the pile, the light rig | `sim/stages.py`, `sim/build.py`, `sim/look.py` |
| Renderer | A purpose-built CPU path tracer (C++/OpenMP) | `render/slicer.cpp` |
| Sound | Procedural ASMR sound design: fly-in whoosh, levitation hum, laser sizzle and wet "shhk" cuts, and one jelly impact for every collision in the physics simulation, panned by position, plus a small studio reverb | `audio/synth.py` |
| Titles and encode | Animated rainbow titles, a progress row and an end card, then H.264 with AAC | `tools/compose.py` |

## The stages

| Stage | Cuts | Pieces | What happens |
|---|---|---|---|
| 1 slice | 1 vertical cut | 2 halves | They fall open like a book |
| 10 slices | 9 cuts | 10 slabs | They fan out like a sliced loaf |
| 25 slices | 4 + 4 cuts | 5x5 columns | They bloom open like a flower |
| 100 slices | 9 + 9 cuts | 10x10 sticks | They burst into a starburst |
| 250 slices | 3 passes | 10x5x5 bricks | Dropped on an edge, they avalanche into a ridge |
| 500 slices | 3 passes | 10x5x10 bars | Dropped on a corner, they spread into a mound |
| 1000 slices | 3 passes | 10x10x10 dice | Dropped on a corner, they burst into a rainbow pile |

## Renderer highlights

* Every piece is a convex polyhedron under an affine map (the rigid transform plus the jelly
  strain), so ray entry and exit points are analytic. That makes the following cheap:
  * refraction with exact Fresnel and total internal reflection,
  * Beer-Lambert absorption that follows a 3D rainbow, so cut faces show the colours inside,
  * single scattering of the area lights, integrated in closed form along each segment
    (no noise), plus a multiple-scattering glow term.
* Area-light soft shadows with coloured transmittance through the jelly, a glossy floor
  (GGX with multiple importance sampling) and short-range AO.
* Shading-only rounded edges, like a bevel shader.
* Laser beams and glowing cut sheets. Until the drop, the sliced cube is rendered as one
  volume with "virtual" kerf planes, including total internal reflection in the thin air gaps.
* Motion blur (8 time slices per frame, with lasers interpolated per sample), depth of
  field, adaptive sampling (4 to 16 samples per pixel) with shuffled, digitally shifted
  Sobol (0,2) points, bloom, AgX tone mapping, vignette and film grain.

## Run it

Requirements: Python 3 with `numpy`, `scipy` and `pillow`; Node 18+; g++ with OpenMP;
libpng; ffmpeg with libx264.

```bash
pip install numpy scipy pillow
PREVIEW=1 ./run_all.sh     # low-res preview, about 15 minutes
./run_all.sh               # 1080x1920 final, about 2 hours on 4 CPU cores
```

The result is written to `output/`. `tools/sheet.py` makes contact sheets of rendered
frames, and `tools/lookdev.py` builds a single still for tuning materials.
