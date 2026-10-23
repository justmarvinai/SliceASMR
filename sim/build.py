"""Builds everything the renderer and the sound design need.

usage: python3 sim/build.py <work_dir> [--fps 30] [--res W H] [--spp MIN MAX] [--stages 0,1,..]

Writes <work_dir>/stage_<k>.bin (renderer scene streams, one per stage),
<work_dir>/timeline.json (titles, cut/release times, impact events).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jelly  # noqa: E402
import physics  # noqa: E402
from look import PALETTE_KW, PARAMS, lights_for  # noqa: E402
from sceneio import Palette, SceneWriter, piece_dtype, world_to_local  # noqa: E402
from stages import build_stages, cube_motion, release_states  # noqa: E402
from xform import nlerp_arrays, quat_to_mat, smootherstep  # noqa: E402

K = 8                     # motion-blur time slices per frame
SHUTTER = 0.5             # fraction of the frame interval the shutter is open (180 degrees)
PHYS_TAIL = 1.4           # physics keeps running this long into the next stage (camera pan)
PAN_BEFORE, PAN_AFTER = 0.0, 1.05
LASER_RGB = np.array([1.0, 0.22, 0.42])
GLOW_RGB = np.array([1.0, 0.42, 0.55])
GLOW_FADE = 0.45          # seconds for a cut glow to cool down after its sweep


# ------------------------------------------------------------------ camera --

TRACKS = {}   # stage index -> (times, smoothed xz centroid of the pieces) after release


def _track(st, tau):
    """Camera target offset that follows the pile (blended in after the drop)."""
    if st.index not in TRACKS:
        return np.zeros(3)
    tt, xz = TRACKS[st.index]
    i = int(np.clip(np.searchsorted(tt, tau), 1, len(tt) - 1))
    f = float(np.clip((tau - tt[i - 1]) / (tt[i] - tt[i - 1]), 0, 1))
    p = xz[i - 1] * (1 - f) + xz[i] * f
    w = smootherstep((tau - st.release_t - 0.3) / 1.4)
    return np.array([p[0] - st.arena[0], 0.0, p[1] - st.arena[2]]) * w


def _stage_cam(st, tau):
    r = st.release_t
    d = st.cam_dist
    hy = st.hover_y - 0.15
    keys = [
        (0.0, hy, d, 22.0, -5.0),
        (r, hy - 0.1, d * 0.975, 23.5, -2.0),
        (r + 0.3, hy - 0.15, d * 0.98, 24.0, -1.5),
        (r + 1.8, st.pile_y, d * st.pile_dist, st.pile_elev, 1.5),
        (st.duration, st.pile_y - 0.2, d * (st.pile_dist - 0.04), st.pile_elev + 2.0, 5.0),
    ]
    tau = min(max(tau, keys[0][0]), keys[-1][0])
    for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
        if tau <= t1:
            x = smootherstep((tau - t0) / max(t1 - t0, 1e-6))
            vals = [u + (v - u) * x for u, v in zip(a, b)]
            break
    ty, dist, elev, azim = vals
    return np.array([st.arena[0], ty, st.arena[2]]) + _track(st, tau), dist, elev, azim


def camera_at(stages, t):
    k = 0
    for i, s in enumerate(stages):
        if t >= s.t_start:
            k = i
    st = stages[k]
    tgt, dist, elev, azim = _stage_cam(st, t - st.t_start)
    # pan into the next stage, or still finishing the pan from the previous one
    nxt = stages[k + 1] if k + 1 < len(stages) else None
    if nxt is not None and t >= nxt.t_start - PAN_BEFORE:
        a, b = st, nxt
    elif k > 0 and t < st.t_start + PAN_AFTER:
        a, b = stages[k - 1], st
    else:
        a = b = None
    if a is not None:
        x = smootherstep((t - (b.t_start - PAN_BEFORE)) / (PAN_BEFORE + PAN_AFTER))
        ta, da, ea, aa = _stage_cam(a, t - a.t_start)
        tb, db, eb, ab = _stage_cam(b, max(0.0, t - b.t_start))
        # lift the camera a little in the middle of the dolly
        lift = math.sin(math.pi * x) * 0.6
        tgt = ta + (tb - ta) * x + np.array([0, lift, 0])
        dist = da + (db - da) * x
        elev = ea + (eb - ea) * x
        azim = aa + (ab - aa) * x
    e, az = math.radians(elev), math.radians(azim)
    pos = tgt + dist * np.array([math.cos(e) * math.sin(az), math.sin(e), math.cos(e) * math.cos(az)])
    return pos, tgt


def cam_record(stages, t, fov=36.0, aperture=0.045):
    pos, tgt = camera_at(stages, t)
    fwd = tgt - pos
    focus = float(np.linalg.norm(fwd))
    return [*pos, *fwd, 0.0, 1.0, 0.0, math.radians(fov), aperture, focus]


# ------------------------------------------------------------- per stage ----

class StageData:
    """Simulation results of one stage, queried by stage-local time."""

    def __init__(self, st, work_dir, rng):
        self.st = st
        t0 = time.time()
        self.hover_t, self.hover_E = jelly.hover_strain(st, cube_motion)
        states = release_states(st, rng)
        dur = st.duration - st.release_t + PHYS_TAIL
        self.phys = physics.run(st, states, dur, work_dir)
        self.E, self.piv = jelly.simulate(self.phys, st.pieces)
        self.events = jelly.impact_events(self.phys, st.pieces)
        TRACKS[st.index] = self.centroid_track()
        print(f"[stage {st.index}] {st.title}: {len(st.pieces)} pieces, release {st.release_t:.2f}s, "
              f"{len(self.events)} impacts, {time.time() - t0:.1f}s", flush=True)

    def centroid_track(self, tau=0.55):
        """Volume-weighted xz centroid of the pieces, low-passed like a camera operator."""
        ph = self.phys
        vol = np.array([p.volume for p in self.st.pieces])
        c = np.einsum("snk,n->sk", ph.pos[:, :, [0, 2]], vol) / vol.sum()
        out = np.zeros_like(c)
        x = c[0].copy()
        v = np.zeros(2)
        w = 1.0 / tau
        for i in range(len(c)):     # critically damped follower
            a = w * w * (c[i] - x) - 2 * w * v
            v += a * ph.dt
            x += v * ph.dt
            out[i] = x
        times = self.st.release_t + np.arange(len(c)) * ph.dt
        return np.concatenate([[0.0], times]), np.concatenate([out[:1], out])

    def hover_E_at(self, t):
        x = np.clip(t / self.hover_t[1], 0, len(self.hover_t) - 1.001)
        i = int(x)
        f = x - i
        return self.hover_E[i] * (1 - f) + self.hover_E[i + 1] * f

    def transforms(self, t, set_idx):
        """World->local 3x4 maps for the pieces of piece set `set_idx` at stage time t."""
        st = self.st
        pieces = st.piece_sets[set_idx]
        out = np.zeros((len(pieces), 12))
        if t < st.release_t:
            pos, q = cube_motion(st, t)
            R = quat_to_mat(q)
            D = np.eye(3) + self.hover_E_at(t)
            for i, pc in enumerate(pieces):
                T = pos + R @ D @ pc.center
                out[i] = world_to_local(R, T, D, np.zeros(3))
            return out
        ph = self.phys
        u = (t - st.release_t) / ph.dt
        u = min(max(u, 0.0), ph.nsamples - 1.001)
        i0 = int(u)
        f = u - i0
        pos = ph.pos[i0] * (1 - f) + ph.pos[i0 + 1] * f
        quat = nlerp_arrays(ph.quat[i0], ph.quat[i0 + 1], f)
        E = self.E[i0] * (1 - f) + self.E[i0 + 1] * f
        piv = self.piv[i0] * (1 - f) + self.piv[i0 + 1] * f
        R = quat_to_mat(quat)
        D = np.eye(3)[None] + E
        Di = np.linalg.inv(D)
        A = np.einsum("nij,nkj->nik", Di, R)              # D^-1 R^T
        b = piv - np.einsum("nij,nj->ni", Di, piv + np.einsum("nji,nj->ni", R, pos))
        out = np.concatenate([A, b[:, :, None]], axis=2).reshape(-1, 12)
        return out

    def lasers(self, t):
        """Beam segments (cube motion applied) for all passes visible near t."""
        st = self.st
        res = []
        pos, q = cube_motion(st, min(t, st.release_t))
        R = quat_to_mat(q)
        D = np.eye(3) + self.hover_E_at(t)
        M = R @ D
        for p in st.passes:
            inten = p.beam_intensity(t)
            lam = p.front(t)
            for n, d in p.planes:
                c = n * d + p.sweep * lam
                a = pos + M @ (c - p.beam * p.beam_len)
                b = pos + M @ (c + p.beam * p.beam_len)
                res.append([*a, *b, *(LASER_RGB), 0.0075, 26.0 * inten])
        return res

    def glows(self, t):
        st = self.st
        res = []
        for p in st.passes:
            if t < p.t0 - 0.05:
                inten = 0.0
            elif t <= p.t1:
                inten = 1.0
            else:
                inten = math.exp(-(t - p.t1) / GLOW_FADE * 2.2)
            inten = inten * 6.0 if inten > 1e-3 else 0.0
            front = p.front(t) if t >= p.t0 else -10.0
            for n, d in p.planes:
                res.append([st.index, *n, d, *p.sweep, front, 0.55, 0.014, inten, 0.35, *GLOW_RGB])
        return res


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--res", type=int, nargs=2, default=[1080, 1920])
    ap.add_argument("--spp", type=int, nargs=2, default=[8, 16])
    ap.add_argument("--stages", default="")
    ap.add_argument("--every", type=int, default=1, help="write only every n-th frame (previews)")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    stages = build_stages()
    wanted = set(range(len(stages))) if not args.stages else {int(x) for x in args.stages.split(",")}
    need = set(wanted) | {k - 1 for k in wanted if k > 0}
    rng = np.random.default_rng(7)
    data = {}
    for st in stages:
        sub = np.random.default_rng(1000 + st.index)
        if st.index in need:
            data[st.index] = StageData(st, os.path.join(args.work_dir, "phys"), sub)
    del rng
    fps = args.fps
    total = sum(s.duration for s in stages)
    nframes = int(round(total * fps))
    frame_dt = 1.0 / fps
    slice_offsets = [((k + 0.5) / K - 0.5) * SHUTTER * frame_dt for k in range(K)]

    timeline = dict(fps=fps, total=total, nframes=nframes, stages=[])
    for st in stages:
        ev = data[st.index].events if st.index in data else []
        timeline["stages"].append(dict(
            index=st.index, title=st.title, count=st.count, t_start=st.t_start, duration=st.duration,
            release=st.t_start + st.release_t,
            passes=[dict(t0=st.t_start + p.t0, t1=st.t_start + p.t1, beams=len(p.planes)) for p in st.passes],
            impacts=[dict(e, t=e["t"] + st.t_start) for e in ev],
            arena_x=float(st.arena[0])))

    for st in stages:
        if st.index not in wanted:
            continue
        cur = data[st.index]
        prev = data.get(st.index - 1)
        f0 = int(math.ceil(st.t_start * fps - 1e-9))
        f1 = int(math.ceil((st.t_start + st.duration) * fps - 1e-9))
        frames = list(range(f0, f1, args.every))
        # geometry table: every piece set of this stage plus the previous stage's final pile
        geoms = []
        gid = {}
        for s_idx, ps in enumerate(st.piece_sets):
            for j, pc in enumerate(ps):
                gid[(st.index, s_idx, j)] = len(geoms)
                geoms.append(pc.geom)
        if prev is not None:
            for j, pc in enumerate(prev.st.pieces):
                gid[(prev.st.index, -1, j)] = len(geoms)
                geoms.append(pc.geom)
        pals = []
        for s in stages:
            kw = dict(PALETTE_KW)
            kw["offset"] = kw["offset"] + s.palette_offset
            pals.append(Palette(**kw))
        path = os.path.join(args.work_dir, f"stage_{st.index}.bin")
        sw = SceneWriter(path, args.res[0], args.res[1], args.spp[0], args.spp[1], K, len(frames), PARAMS,
                         geoms, pals)
        t_build = time.time()
        for fno in frames:
            t = fno * frame_dt
            tau = t - st.t_start
            tks = [t + o for o in slice_offsets]
            cams = [cam_record(stages, tk) for tk in tks]
            _, tgt = camera_at(stages, t)
            lights = lights_for(np.array([tgt[0], 0.0, st.arena[2]]))
            # Until the drop, the sliced cube is rendered as one piece with "virtual" cut
            # planes (much cheaper than hundreds of touching pieces); afterwards the real pieces.
            virtual = max(tks) - st.t_start < st.release_t
            set_idx = 0 if virtual else len(st.piece_sets) - 1
            recs = []
            pcs = st.piece_sets[set_idx]
            arr = np.zeros(len(pcs), piece_dtype(K))
            arr["ids"][:, 0] = [gid[(st.index, set_idx, j)] for j in range(len(pcs))]
            arr["ids"][:, 1] = st.index
            arr["ids"][:, 2] = st.index
            arr["ids"][:, 3] = 1 if virtual else 0
            arr["uoff"] = np.array([pc.center for pc in pcs])
            for k, tk in enumerate(tks):
                arr["M"][:, k, :] = cur.transforms(tk - st.t_start, set_idx)
            recs.append(arr)
            if prev is not None and tau < PAN_AFTER + 0.6:
                pp = prev.st.pieces
                arr2 = np.zeros(len(pp), piece_dtype(K))
                arr2["ids"][:, 0] = [gid[(prev.st.index, -1, j)] for j in range(len(pp))]
                arr2["ids"][:, 1] = prev.st.index
                arr2["ids"][:, 2] = prev.st.index
                arr2["uoff"] = np.array([pc.center for pc in pp])
                for k, tk in enumerate(tks):
                    arr2["M"][:, k, :] = prev.transforms(tk - prev.st.t_start, len(prev.st.piece_sets) - 1)
                recs.append(arr2)
            pieces = np.concatenate(recs)
            lasers = [cur.lasers(tk - st.t_start) for tk in tks]
            keep = [i for i in range(len(lasers[0])) if max(L[i][10] for L in lasers) > 0]
            lasers = [[L[i] for i in keep] for L in lasers]
            glows = [cur.glows(tk - st.t_start) for tk in tks]
            if virtual:   # every plane a beam has started to cut acts as a cut
                keep = [i for i in range(len(glows[0])) if max(G[i][8] for G in glows) > -5]
            else:
                keep = [i for i in range(len(glows[0])) if max(G[i][11] for G in glows) > 0]
            glows = [[G[i] for i in keep] for G in glows]
            sw.write_frame(fno, t, cams, lights, lasers, glows, pieces)
        sw.close()
        print(f"[export] stage {st.index}: frames {f0}..{f1 - 1} ({len(frames)}) -> {path} "
              f"in {time.time() - t_build:.1f}s", flush=True)
    with open(os.path.join(args.work_dir, "timeline.json"), "w") as f:
        json.dump(timeline, f)


if __name__ == "__main__":
    main()
