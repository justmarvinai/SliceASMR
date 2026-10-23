"""Static look-development scene: a hovering rainbow jelly cube, a laser and
some sliced pieces lying on the floor. Used to tune materials and lighting."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from sceneio import Geom, Palette, SceneWriter, piece_dtype, world_to_local  # noqa: E402
from look import PARAMS, lights_for, PALETTE_KW  # noqa: E402


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def main(out, W=540, H=960, spp=(8, 16)):
    K = 1
    S = 1.0  # half size of the cube
    geoms = [Geom([-S, -S, -S], [S, S, S], orig_mask=0x3F, bevel_orig=0.06, bevel_cut=0.004)]
    h = 0.1
    geoms.append(Geom([-h, -h, -h], [h, h, h], orig_mask=0, bevel_orig=0.02, bevel_cut=0.01))
    geoms.append(Geom([-0.1, -S, -S], [0.1, S, S], orig_mask=0b111100, bevel_orig=0.05, bevel_cut=0.006))
    pals = [Palette(**PALETTE_KW)]
    rng = np.random.default_rng(3)
    pieces = []
    # hovering cube
    R = rot([0, 1, 0], math.radians(35)) @ rot([1, 0, 0], math.radians(-12))
    pieces.append((0, 0, 0, 0, [0, 0, 0], world_to_local(R, [0.0, 2.4, 0.0])))
    # dice on the floor
    for i in range(40):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(1.6, 3.2)
        p = np.array([r * math.cos(a), h, r * math.sin(a) * 0.8 + 0.4])
        Rz = rot([0, 1, 0], rng.uniform(0, 6.28))
        u = rng.uniform(-S + h, S - h, 3)
        pieces.append((1, 0, 0, 0, u, world_to_local(Rz, p)))
    # a fallen slab
    Rs = rot([0, 0, 1], math.radians(90)) @ rot([1, 0, 0], math.radians(10))
    pieces.append((2, 0, 0, 0, [0.5, 0, 0], world_to_local(Rs, [-1.6, 0.1, -1.4])))
    arr = np.zeros(len(pieces), piece_dtype(K))
    for i, (g, p, st, fl, u, M) in enumerate(pieces):
        arr[i]["ids"] = [g, p, st, fl]
        arr[i]["uoff"] = u
        arr[i]["M"][0] = M

    cam_pos = np.array([0.0, 5.6, 9.6])
    target = np.array([0.0, 1.0, 0.0])
    fwd = target - cam_pos
    focus = float(np.linalg.norm(np.array([0, 2.0, 0]) - cam_pos))
    cams = [[*cam_pos, *fwd, 0, 1, 0, math.radians(38), 0.025, focus]]
    lights = lights_for(np.array([0.0, 0.0, 0.0]))
    # one laser through the cube
    lasers = [[[-2.6, 2.3, 0.9, 2.6, 2.9, -0.5, 1.0, 0.12, 0.25, 0.012, 30.0]]]
    glows = [[[0, 0, 1, 0, 0.1, 1, 0, 0, 0.2, 0.6, 0.015, 8.0, 0.0, 1.0, 0.3, 0.1]]]
    sw = SceneWriter(out, W, H, spp[0], spp[1], K, 1, PARAMS, geoms, pals)
    sw.write_frame(0, 0.0, cams, lights, lasers, glows, arr)
    sw.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lookdev.bin")
