"""Runs the Rapier simulation (physics.mjs) for one stage and loads the result."""
import hashlib
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_RATE = 240          # Hz, physics step and output sample rate
GRAVITY = -9.81

JELLY = dict(density=1100.0, friction=0.32, restitution=0.12)
FLOOR = dict(friction=0.80, restitution=0.10)


class PhysicsResult:
    def __init__(self, pos, quat, vel, angvel, dt, t0):
        self.pos, self.quat, self.vel, self.angvel = pos, quat, vel, angvel
        self.dt = dt
        self.t0 = t0    # stage-local time of sample 0 (the release)

    @property
    def nsamples(self):
        return self.pos.shape[0]


def run(stage, states, duration, work_dir):
    bodies = []
    for pc, (p, q, lv, av) in zip(stage.pieces, states):
        b = dict(pos=[float(x) for x in p], quat=[float(x) for x in q],
                 linvel=[float(x) for x in lv], angvel=[float(x) for x in av], **JELLY)
        if pc.shape == "box":
            b.update(shape="box", half=[float(x) for x in pc.half])
        else:
            b.update(shape="hull", verts=[[float(x) for x in v] for v in pc.verts])
        if pc.size < 0.35:
            b["ccd"] = True
        bodies.append(b)
    cfg = dict(dt=1.0 / SIM_RATE, steps_per_sample=1, nsamples=int(round(duration * SIM_RATE)) + 1,
               gravity=GRAVITY, solver_iterations=8, floor=FLOOR,
               floor_center=[float(x) for x in stage.arena], bodies=bodies)
    blob = json.dumps(cfg, sort_keys=True).encode()
    key = hashlib.sha1(blob + open(os.path.join(HERE, "physics.mjs"), "rb").read()).hexdigest()[:16]
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, f"phys_{stage.index}_{key}.bin")
    if not os.path.exists(out):
        inp = os.path.join(work_dir, f"phys_{stage.index}_{key}.json")
        with open(inp, "wb") as f:
            f.write(blob)
        print(f"[physics] stage {stage.index}: {len(bodies)} bodies, {duration:.1f}s", flush=True)
        subprocess.run(["node", os.path.join(HERE, "physics.mjs"), inp, out + ".tmp"], check=True, cwd=HERE)
        os.replace(out + ".tmp", out)
    with open(out, "rb") as f:
        ns, nb = np.frombuffer(f.read(8), np.int32)
        dt = float(np.frombuffer(f.read(4), np.float32)[0])
        data = np.frombuffer(f.read(), np.float32).reshape(ns, nb, 13).astype(np.float64)
    return PhysicsResult(data[..., 0:3], data[..., 3:7], data[..., 7:10], data[..., 10:13], dt, stage.release_t)
