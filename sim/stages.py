"""The seven stages of the video: 1, 10, 25, 100, 250, 500 and 1000 slices.

Every stage happens in its own arena on an endless studio floor (arenas are
ARENA_DX apart along +x; the camera dollies from one to the next). A rainbow
jelly cube flies in, hovers, is sliced by sweeping laser combs and is dropped.
All times here are stage-local seconds.
"""
import math

import numpy as np

from geometry import CUBE_H, grid_pieces, plane_split, whole_cube
from xform import quat_axis_angle, quat_from_two, quat_mul, quat_to_mat

ARENA_DX = 16.0
HOVER_Y = 2.55
FLY_START = 0.15      # the camera dollies in from the previous arena meanwhile
FLY_TIME = 1.15
FIRST_CUT = FLY_START + 1.35
SWEEP = 0.40          # duration of one laser sweep through the cube
PASS_GAP = 0.20       # pause between consecutive passes
LASER_FADE_IN = 0.12
LASER_FADE_OUT = 0.18
RELEASE_DELAY = 0.45  # after the last sweep finished


class CutPass:
    """Parallel laser beams sweeping through the cube, each cutting one plane."""

    def __init__(self, planes, sweep, beam, t0, dur=SWEEP, lam=(-1.35, 1.35), beam_len=2.2):
        self.planes = [(np.asarray(n, float), float(d)) for n, d in planes]
        self.sweep = np.asarray(sweep, float) / np.linalg.norm(sweep)
        self.beam = np.asarray(beam, float) / np.linalg.norm(beam)
        self.t0 = t0
        self.dur = dur
        self.lam0, self.lam1 = lam
        self.beam_len = beam_len

    @property
    def t1(self):
        return self.t0 + self.dur

    def progress(self, t):
        """Eased 0..1 progress of the sweep."""
        x = np.clip((t - self.t0) / self.dur, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    def front(self, t):
        return self.lam0 + (self.lam1 - self.lam0) * self.progress(t)

    def beam_intensity(self, t):
        a = np.clip((t - (self.t0 - LASER_FADE_IN)) / LASER_FADE_IN, 0, 1)
        b = np.clip((self.t1 + LASER_FADE_OUT - t) / LASER_FADE_OUT, 0, 1)
        return float(min(a, b))


AXES = {"x": 0, "y": 1, "z": 2}
# laser layout per cut axis: (sweep direction, beam direction) in cube coordinates
PASS_LAYOUT = {
    "x": ((0, -1, 0), (0, 0, 1)),
    "z": ((0, -1, 0), (1, 0, 0)),
    "y": ((1, 0, 0), (0, 0, 1)),
}


class Stage:
    def __init__(self, index, title, count, duration, kind, *, grid=None, order="", plane=None,
                 yaw=30.0, orient="flat", release="drop", release_params=None, sweep=SWEEP,
                 hover_y=HOVER_Y, palette_offset=0.0, cam_dist=12.4, pile_dist=1.065, pile_elev=31.0,
                 pile_y=1.0):
        self.index = index
        self.title = title
        self.count = count
        self.duration = duration
        self.arena = np.array([index * ARENA_DX, 0.0, 0.0])
        self.release_style = release
        self.release_params = release_params or {}
        self.palette_offset = palette_offset
        self.cam_dist = cam_dist
        self.pile_dist = pile_dist
        self.pile_elev = pile_elev
        self.pile_y = pile_y
        # hover orientation
        q = quat_axis_angle([0, 1, 0], math.radians(yaw))
        if orient == "edge":
            q = quat_mul(q, quat_axis_angle([0, 0, 1], math.radians(45)))
        elif orient == "corner":
            q = quat_mul(q, quat_from_two([1, 1, 1], [0, 1, 0]))
        self.q_hover = q / np.linalg.norm(q)
        self.hover_y = hover_y
        self.passes = []
        self.piece_sets = [whole_cube()]
        t = FIRST_CUT
        if kind == "plane":
            n, d = plane
            n = np.asarray(n, float) / np.linalg.norm(n)
            s = np.array([0.0, -1.0, 0.0])
            s = s - n * np.dot(s, n)
            s /= np.linalg.norm(s)
            b = np.cross(n, s)
            self.passes.append(CutPass([(n, d)], s, b, t, dur=sweep))
            self.piece_sets.append(plane_split(n, d))
            t += sweep
        else:
            counts = [1, 1, 1]
            for ax in order:
                a = AXES[ax]
                n_cells = grid[a]
                counts[a] = n_cells
                edges = np.linspace(-CUBE_H, CUBE_H, n_cells + 1)[1:-1]
                normal = np.zeros(3)
                normal[a] = 1
                s, b = PASS_LAYOUT[ax]
                self.passes.append(CutPass([(normal, e) for e in edges], s, b, t, dur=sweep))
                self.piece_sets.append(grid_pieces(*counts))
                t += sweep + PASS_GAP
            t -= PASS_GAP
        self.release_t = t + RELEASE_DELAY

    @property
    def pieces(self):
        return self.piece_sets[-1]

    def piece_set_at(self, t):
        """Index into piece_sets for the geometry shown at stage time t."""
        k = 0
        for i, p in enumerate(self.passes):
            if t >= p.t1:
                k = i + 1
        return k


def spring01(t, freq=1.05, zeta=0.68):
    """Step response of a damped spring: 0 -> 1 with a gentle overshoot."""
    if t <= 0:
        return 0.0
    w = 2 * math.pi * freq
    wd = w * math.sqrt(1 - zeta * zeta)
    e = math.exp(-zeta * w * t)
    return 1 - e * (math.cos(wd * t) + zeta * w / wd * math.sin(wd * t))


def cube_motion(stage, t):
    """Kinematic cube transform before release: (position, quaternion)."""
    A = stage.arena
    p_h = A + np.array([0.0, stage.hover_y, 0.0])
    p0 = A + np.array([-0.4, 9.2, -1.5])
    t = t - FLY_START
    s = spring01(t / FLY_TIME * 1.0)
    pos = p_h + (p0 - p_h) * (1 - s)
    # gentle hover bob, fading in after the fly-in
    bob = 0.035 * math.sin(2 * math.pi * 0.45 * (t - FLY_TIME)) * min(1.0, max(0.0, (t - FLY_TIME) / 0.6))
    pos = pos + np.array([0.0, bob, 0.0])
    # spin-in: rotate about a tilted axis, settling onto the hover orientation
    k = min(1.0, max(0.0, t / (FLY_TIME * 1.05)))
    k = 1 - (1 - k) ** 3
    spin = quat_axis_angle([0.35, 1.0, 0.25], math.radians(-220) * (1 - k))
    q = quat_mul(spin, stage.q_hover)
    return pos, q / np.linalg.norm(q)


def cube_velocity(stage, t, h=1e-3):
    p0, q0 = cube_motion(stage, t - h)
    p1, q1 = cube_motion(stage, t + h)
    v = (p1 - p0) / (2 * h)
    dq = quat_mul(q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
    if dq[0] < 0:
        dq = -dq
    ang = 2 * math.acos(min(1.0, dq[0]))
    axis = dq[1:]
    nrm = np.linalg.norm(axis)
    w = axis / nrm * ang / (2 * h) if nrm > 1e-9 else np.zeros(3)
    return v, w


def build_stages():
    st = [
        Stage(0, "1 SLICE", 1, 7.6, "plane", plane=((1.0, 0.0, 0.0), 0.0), yaw=-32.0, cam_dist=10.6,
              sweep=0.62, hover_y=2.35, pile_dist=1.06, pile_elev=30.0,
              release="bloom_x", release_params=dict(k=0.0, kmin=0.95, jitter=0.03, pivot="spine")),
        Stage(1, "10 SLICES", 10, 8.4, "grid", grid=(10, 1, 1), order="x", yaw=34.0, cam_dist=11.0,
              pile_dist=1.12, pile_elev=33.0,
              release="bloom_x", release_params=dict(k=1.5, kmin=0.22, jitter=0.10)),
        Stage(2, "25 SLICES", 25, 8.6, "grid", grid=(5, 1, 5), order="xz", yaw=30.0, cam_dist=11.4,
              pile_dist=1.15, pile_elev=36.0,
              release="bloom_radial", release_params=dict(k=1.6, kmin=0.75, jitter=0.12)),
        Stage(3, "100 SLICES", 100, 9.0, "grid", grid=(10, 1, 10), order="xz", yaw=30.0, cam_dist=12.0,
              pile_dist=1.22, pile_elev=40.0, pile_y=0.6,
              release="bloom_radial", release_params=dict(k=1.1, kmin=0.0, jitter=0.2)),
        Stage(4, "250 SLICES", 250, 9.2, "grid", grid=(10, 5, 5), order="xzy", orient="edge", yaw=72.0,
              cam_dist=12.4, pile_dist=1.22, pile_elev=40.0, pile_y=0.5,
              release="pop", release_params=dict(pop=0.25, jitter=0.03)),
        Stage(5, "500 SLICES", 500, 9.4, "grid", grid=(10, 5, 10), order="xzy", orient="corner", yaw=20.0,
              cam_dist=12.4, pile_dist=1.25, pile_elev=42.0, pile_y=0.45,
              release="pop", release_params=dict(pop=0.3, jitter=0.03)),
        Stage(6, "1000 SLICES", 1000, 12.4, "grid", grid=(10, 10, 10), order="xzy", orient="corner", yaw=12.0,
              cam_dist=12.6, pile_dist=1.28, pile_elev=44.0, pile_y=0.45,
              release="pop", release_params=dict(pop=0.35, jitter=0.03)),
    ]
    t = 0.0
    for s in st:
        s.t_start = t
        t += s.duration
    return st


def release_states(stage, rng):
    """Initial rigid-body states of the final pieces at the release moment."""
    t = stage.release_t
    pos, q = cube_motion(stage, t)
    v, w = cube_velocity(stage, t)
    R = quat_to_mat(q)
    prm = stage.release_params
    out = []
    for pc in stage.pieces:
        c = pc.center
        wp = pos + R @ c
        lv = v + np.cross(w, R @ c)
        av = w.copy()
        style = stage.release_style
        half = (pc.geom.hi - pc.geom.lo) / 2
        if style in ("bloom_x", "bloom_radial"):
            # pieces tip away from the middle about their outer bottom edge, like a
            # loaf (or a flower) falling open; spinning about the edge rather than the
            # centre keeps neighbouring feet from colliding at the release
            if style == "bloom_x":
                out_dir = np.array([np.sign(c[0]), 0.0, 0.0])
                w_loc = np.array([0.0, 0.0, -prm["k"] * c[0] - prm.get("kmin", 0.0) * np.sign(c[0])])
            else:
                r = np.array([c[0], 0.0, c[2]])
                rn = np.linalg.norm(r)
                out_dir = r / rn if rn > 1e-6 else np.zeros(3)
                kick = prm["k"] * r + prm.get("kmin", 0.0) * out_dir
                w_loc = np.cross([0.0, 1.0, 0.0], kick)
            if prm.get("pivot", "outer") == "spine":   # hinge on the shared bottom edge, like a book
                pivot = np.array([0.0, -CUBE_H, c[2]])
            else:
                pivot = np.array([c[0] + out_dir[0] * half[0], -CUBE_H, c[2] + out_dir[2] * half[2]])
            w_kick = R @ w_loc
            av = av + w_kick
            lv = lv + np.cross(w_kick, R @ (c - pivot))
        elif style == "pop":
            lv = lv + R @ (prm["pop"] * c)
        j = prm.get("jitter", 0.0)
        av = av + rng.normal(0, j, 3)
        lv = lv + rng.normal(0, j * 0.3, 3)
        out.append((wp, q.copy(), lv, av))
    return out
