"""Piece geometry: dicing the unit jelly cube ([-1,1]^3 in cube coordinates)."""
import itertools

import numpy as np

from sceneio import Geom

CUBE_H = 1.0          # half size of the cube (it is 2 m wide)
GAP = 0.006           # laser kerf between neighbouring pieces
BEVEL_ORIG = 0.06     # rounded outer edges of the original cube
BEVEL_CUT = 0.004     # freshly cut edges are sharp


class Piece:
    """A convex jelly piece. `center` is its centroid in cube coordinates (also the
    colour lookup offset); geometry is stored relative to that centroid."""

    def __init__(self, center, geom, shape, half=None, verts=None, volume=None):
        self.center = np.asarray(center, float)
        self.geom = geom
        self.shape = shape            # 'box' or 'hull'
        self.half = None if half is None else np.asarray(half, float)
        self.verts = None if verts is None else np.asarray(verts, float)
        if volume is None:
            volume = float(np.prod(2 * self.half))
        self.volume = volume
        lo, hi = geom.lo, geom.hi
        self.extent = hi - lo
        self.size = float(volume ** (1 / 3))

    def support_center(self, n):
        """Centre of the AABB face whose outward normal best matches -n (local)."""
        a = int(np.argmax(np.abs(n)))
        p = np.zeros(3)
        p[a] = self.geom.lo[a] if n[a] > 0 else self.geom.hi[a]
        return p


def _orig_mask(lo, hi, eps=1e-6):
    m = 0
    for a in range(3):
        if lo[a] <= -CUBE_H + eps:
            m |= 1 << (2 * a)
        if hi[a] >= CUBE_H - eps:
            m |= 1 << (2 * a + 1)
    return m


def grid_pieces(nx, ny, nz, gap=GAP):
    """Dice the cube into nx*ny*nz boxes; inner faces are pulled back by gap/2."""
    edges = [np.linspace(-CUBE_H, CUBE_H, n + 1) for n in (nx, ny, nz)]
    pieces = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        lo = np.array([edges[0][i], edges[1][j], edges[2][k]])
        hi = np.array([edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]])
        mask = _orig_mask(lo, hi)
        lo_s = lo + np.where(lo > -CUBE_H + 1e-6, gap / 2, 0.0)
        hi_s = hi - np.where(hi < CUBE_H - 1e-6, gap / 2, 0.0)
        c = (lo_s + hi_s) / 2
        half = (hi_s - lo_s) / 2
        bev = min(BEVEL_ORIG, 0.25 * float(half.min()))
        g = Geom(-half, half, orig_mask=mask, bevel_orig=bev, bevel_cut=BEVEL_CUT)
        pieces.append(Piece(c, g, "box", half=half))
    return pieces


def _polytope(planes):
    """Vertices of the bounded convex polytope {x : n.x <= d for (n, d) in planes}."""
    verts = []
    for (n1, d1), (n2, d2), (n3, d3) in itertools.combinations(planes, 3):
        A = np.array([n1, n2, n3])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        x = np.linalg.solve(A, [d1, d2, d3])
        if all(np.dot(n, x) <= d + 1e-9 for n, d in planes):
            if not any(np.allclose(x, v, atol=1e-9) for v in verts):
                verts.append(x)
    return np.array(verts)


def _volume_centroid(planes, verts):
    """Exact volume and centroid of a convex polytope via a tetrahedral fan."""
    inner = verts.mean(axis=0)
    vol = 0.0
    cen = np.zeros(3)
    for n, d in planes:
        n = np.asarray(n, float)
        on = [v for v in verts if abs(np.dot(n, v) - d) < 1e-7]
        if len(on) < 3:
            continue
        on = np.array(on)
        fc = on.mean(axis=0)
        u = on[0] - fc
        u /= np.linalg.norm(u)
        w = np.cross(n / np.linalg.norm(n), u)
        ang = np.arctan2((on - fc) @ w, (on - fc) @ u)
        ring = on[np.argsort(ang)]
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            v = abs(np.dot(a - inner, np.cross(b - inner, fc - inner))) / 6.0
            vol += v
            cen += v * (inner + a + b + fc) / 4.0
    return vol, cen / vol


CUBE_PLANES = [(np.array(n, float), CUBE_H) for n in
               ([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1])]


def plane_split(n, d, gap=GAP):
    """Split the cube by the plane n.x = d into two convex pieces (with kerf)."""
    n = np.asarray(n, float)
    n /= np.linalg.norm(n)
    out = []
    for sign in (1.0, -1.0):
        cut = (sign * n, sign * d - gap / 2)
        planes = CUBE_PLANES + [cut]
        verts = _polytope(planes)
        vol, c = _volume_centroid(planes, verts)
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        mask = _orig_mask(lo, hi)
        g = Geom(lo - c, hi - c, planes=[(cut[0], cut[1] - np.dot(cut[0], c))], orig_mask=mask,
                 bevel_orig=BEVEL_ORIG, bevel_cut=BEVEL_CUT)
        out.append(Piece(c, g, "hull", verts=verts - c, volume=vol))
    return out


def whole_cube():
    h = CUBE_H
    g = Geom([-h, -h, -h], [h, h, h], orig_mask=0x3F, bevel_orig=BEVEL_ORIG, bevel_cut=BEVEL_CUT)
    return [Piece(np.zeros(3), g, "box", half=[h, h, h])]
