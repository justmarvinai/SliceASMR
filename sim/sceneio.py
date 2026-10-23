"""Binary scene-stream writer for the C++ renderer (render/slicer.cpp).

Layout (little endian):
  header : magic 'SLCRv001', int32 W,H,spp_min,spp_max,K,nframes,ngeoms,npals,
           float32 params[64]
  geoms  : 26 words each - lo[3] hi[3] int np, 4x(plane n[3], d), uint orig_mask,
           bevel_orig, bevel_cut
  pals   : 12 floats each - gdir[3] scale offset sat bright d0 sigma_s g pad pad
  offsets: int64[nframes]
  frames : see SceneWriter.write_frame
"""
import struct

import numpy as np

PARAMS = [
    "EXPOSURE", "BLOOM_STRENGTH", "BLOOM_THRESHOLD", "VIGNETTE", "GRAIN",
    "FLOOR_R", "FLOOR_G", "FLOOR_B", "FLOOR_ROUGH", "FLOOR_F0", "FLOOR_SPEC",
    "AMB_R", "AMB_G", "AMB_B",
    "ENV_TOP_R", "ENV_TOP_G", "ENV_TOP_B", "ENV_HOR_R", "ENV_HOR_G", "ENV_HOR_B",
    "FOG", "IOR", "AO_DIST",
    "AMBSCAT_R", "AMBSCAT_G", "AMBSCAT_B",
    "LASER_GEL", "LOOK_SAT", "LOOK_POWER",
    "ADAPT_THRESH", "FOG_R", "FOG_G", "FOG_B",
    "FLOOR_TEX", "FIREFLY", "MAX_DEPTH", "REFL_DEPTH", "LASER_FLOOR",
]
PARAM_INDEX = {n: i for i, n in enumerate(PARAMS)}


def pack_params(d):
    arr = np.zeros(64, np.float32)
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            base = k[:-4] if k.endswith("_RGB") else k
            for suffix, x in zip("RGB", v):
                arr[PARAM_INDEX[f"{base}_{suffix}"]] = x
        else:
            arr[PARAM_INDEX[k]] = v
    return arr.tobytes()


class Geom:
    """Convex piece in its local frame: AABB [lo,hi] clipped by planes n.x <= d."""

    def __init__(self, lo, hi, planes=(), orig_mask=0, bevel_orig=0.0, bevel_cut=0.0):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.planes = [(np.asarray(n, float), float(d)) for n, d in planes]
        assert len(self.planes) <= 4
        self.orig_mask = int(orig_mask)
        self.bevel_orig = bevel_orig
        self.bevel_cut = bevel_cut

    def pack(self):
        pl = []
        for i in range(4):
            if i < len(self.planes):
                n, d = self.planes[i]
                pl += [n[0], n[1], n[2], d]
            else:
                pl += [0, 0, 0, 0]
        return struct.pack("<6fi16fI2f", *self.lo, *self.hi, len(self.planes), *pl,
                           self.orig_mask, self.bevel_orig, self.bevel_cut)


class Palette:
    def __init__(self, gdir, scale, offset, sat=1.0, bright=1.0, d0=0.35, sigma_s=2.0, g=0.35):
        self.v = [*gdir, scale, offset, sat, bright, d0, sigma_s, g, 0.0, 0.0]

    def pack(self):
        return struct.pack("<12f", *self.v)


def piece_dtype(K):
    return np.dtype([("ids", "<i4", (4,)), ("uoff", "<f4", (3,)), ("M", "<f4", (K, 12))])


class SceneWriter:
    def __init__(self, path, W, H, spp_min, spp_max, K, nframes, params, geoms, pals):
        self.f = open(path, "wb")
        self.K = K
        self.nframes = nframes
        self.f.write(b"SLCRv001")
        self.f.write(struct.pack("<8i", W, H, spp_min, spp_max, K, nframes, len(geoms), len(pals)))
        self.f.write(pack_params(params))
        for g in geoms:
            self.f.write(g.pack())
        for p in pals:
            self.f.write(p.pack())
        self.off_pos = self.f.tell()
        self.offsets = np.zeros(nframes, np.int64)
        self.f.write(self.offsets.tobytes())
        self.count = 0

    def write_frame(self, frame_no, time, cams, lights, lasers, glows, pieces):
        """cams: K x 12 floats (pos, fwd, up, fovy, aperture, focus)
        lights: list of 16 floats (c, ex, ey, col, spot, pick, shadow, 0)
        lasers: K lists of 11 floats (a, b, col, width, intensity) - same length per slice
        glows : K lists of 16 floats (stage, n, d, s, front, decay, width, intensity, sheet, col)
        pieces: structured array of piece_dtype(K)
        """
        K = self.K
        self.offsets[self.count] = self.f.tell()
        self.count += 1
        w = self.f.write
        w(struct.pack("<if", frame_no, time))
        w(np.asarray(cams, np.float32).reshape(K, 12).tobytes())
        w(struct.pack("<i", len(lights)))
        if lights:
            w(np.asarray(lights, np.float32).reshape(len(lights), 16).tobytes())
        nl = len(lasers[0]) if lasers else 0
        w(struct.pack("<i", nl))
        for k in range(K):
            if nl:
                w(np.asarray(lasers[k], np.float32).reshape(nl, 11).tobytes())
        ng = len(glows[0]) if glows else 0
        w(struct.pack("<i", ng))
        for k in range(K):
            if ng:
                w(np.asarray(glows[k], np.float32).reshape(ng, 16).tobytes())
        w(struct.pack("<i", len(pieces)))
        if len(pieces):
            w(np.ascontiguousarray(pieces).tobytes())

    def close(self):
        assert self.count == self.nframes, (self.count, self.nframes)
        self.f.seek(self.off_pos)
        self.f.write(self.offsets.tobytes())
        self.f.close()


def world_to_local(R, T, D=None, pivot=None):
    """Affine world->local map (3x4) for x_world = T + R (p + D (x_local - p))."""
    R = np.asarray(R, float)
    T = np.asarray(T, float)
    if D is None:
        A = R.T
        b = -A @ T
    else:
        p = np.zeros(3) if pivot is None else np.asarray(pivot, float)
        Di = np.linalg.inv(D)
        A = Di @ R.T
        b = p - Di @ (p + R.T @ T)
    return np.concatenate([A, b[:, None]], axis=1).reshape(12)
