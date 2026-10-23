"""Jelly wobble: a damped linear strain oscillator per piece.

Every piece carries a small strain tensor E (in its local frame) that is
driven by the piece's *proper acceleration* (contact forces from the rigid
body simulation): a push along n squashes the jelly along n and bulges it
sideways (trace-free, i.e. volume preserving to first order). The strain is
applied about a pivot on the contact side, so a piece sitting on the floor
keeps its foot planted while its top wobbles.
"""
import math

import numpy as np

from xform import quat_to_mat

BETA = 0.024          # peak strain per m/s of contact velocity change
MAX_STRAIN = 0.22
PIVOT_MEMORY = 0.25   # seconds


def piece_params(pieces):
    size = np.array([p.size for p in pieces])
    freq = 6.0 * np.sqrt(0.2 / size)               # small dice ~6 Hz, the big cube ~1.9 Hz
    zeta = np.clip(0.10 + 0.08 * (0.2 / size), 0.10, 0.20)
    return 2 * math.pi * freq, zeta


def _squash(n):
    """Trace-free squash tensor 1.5 nn^T - 0.5 I for unit vectors n (..., 3)."""
    S = 1.5 * n[..., :, None] * n[..., None, :]
    S[..., 0, 0] -= 0.5
    S[..., 1, 1] -= 0.5
    S[..., 2, 2] -= 0.5
    return S


def _clamp(E):
    m = np.abs(E).reshape(E.shape[0], 9).max(axis=1)
    s = np.where(m > MAX_STRAIN, MAX_STRAIN / np.maximum(m, 1e-9), 1.0)
    return E * s[:, None, None]


def simulate(phys, pieces, gravity=-9.81):
    """Returns strain E (S,N,3,3), pivot (S,N,3) and impact events."""
    dt = phys.dt
    S, N, _ = phys.pos.shape
    R = quat_to_mat(phys.quat)
    f = np.zeros((S, N, 3))
    f[1:] = (phys.vel[1:] - phys.vel[:-1]) / dt
    f[1:, :, 1] -= gravity
    f_l = np.einsum("snji,snj->sni", R, f)
    w, zeta = piece_params(pieces)
    kappa = BETA * w
    lo = np.array([p.geom.lo for p in pieces])
    hi = np.array([p.geom.hi for p in pieces])
    E = np.zeros((N, 3, 3))
    Ed = np.zeros((N, 3, 3))
    P_acc = np.zeros((N, 3))
    W_acc = np.zeros(N)
    lam = math.exp(-dt / PIVOT_MEMORY)
    E_out = np.zeros((S, N, 3, 3), np.float32)
    piv_out = np.zeros((S, N, 3), np.float32)
    idx = np.arange(N)
    for s in range(1, S):
        a = f_l[s]
        mag = np.linalg.norm(a, axis=1)
        n = a / np.maximum(mag, 1e-9)[:, None]
        force = -(kappa * mag)[:, None, None] * _squash(n)
        Ed += (force - (w * w)[:, None, None] * E - (2 * zeta * w)[:, None, None] * Ed) * dt
        E = _clamp(E + Ed * dt)
        # pivot: the face of the piece the push came from
        ax = np.argmax(np.abs(n), axis=1)
        p_t = np.zeros((N, 3))
        p_t[idx, ax] = np.where(n[idx, ax] > 0, lo[idx, ax], hi[idx, ax])
        wgt = mag * dt
        P_acc = lam * P_acc + wgt[:, None] * p_t
        W_acc = lam * W_acc + wgt
        piv = P_acc / np.maximum(W_acc, 1e-9)[:, None]
        E_out[s] = E
        piv_out[s] = piv
    return E_out, piv_out


def impact_events(phys, pieces, gravity=-9.81, threshold=0.22, refractory=0.06):
    """Collision events (time, strength, position, size, on_floor) for the sound design."""
    dt = phys.dt
    S, N, _ = phys.pos.shape
    dv = phys.vel[1:] - phys.vel[:-1]
    dv[:, :, 1] -= gravity * dt
    mag = np.linalg.norm(dv, axis=2)
    R = quat_to_mat(phys.quat)
    half = np.array([(p.geom.hi - p.geom.lo) / 2 for p in pieces])
    last = np.full(N, -1e9)
    events = []
    for s in range(S - 1):
        m = mag[s]
        hits = np.nonzero(m > threshold)[0]
        if len(hits) == 0:
            continue
        t = phys.t0 + (s + 1) * dt
        for i in hits:
            if t - last[i] < refractory:
                continue
            last[i] = t
            p = phys.pos[s + 1, i]
            low = p[1] - np.sum(np.abs(R[s + 1, i, 1, :]) * half[i])
            events.append(dict(t=float(t), v=float(m[i]), x=float(p[0]), y=float(p[1]), z=float(p[2]),
                               size=float(pieces[i].size), floor=bool(low < 0.03)))
    return events


def hover_strain(stage, cube_motion, rate=240):
    """Global wobble of the whole cube while it is flown in and cut (pivot = centre)."""
    T = stage.release_t
    S = int(math.ceil(T * rate)) + 2
    dt = 1.0 / rate
    times = np.arange(S) * dt
    pos = np.zeros((S, 3))
    quats = np.zeros((S, 4))
    for i, t in enumerate(times):
        pos[i], quats[i] = cube_motion(stage, t)
    acc = np.zeros((S, 3))
    acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) / (dt * dt)
    R = quat_to_mat(quats)
    w = 2 * math.pi * 1.9
    zeta = 0.12
    kappa = 0.010 * w
    E = np.zeros((3, 3))
    Ed = np.zeros((3, 3))
    out = np.zeros((S, 3, 3))
    kicks = {int(round(p.t0 * rate)): p.sweep for p in stage.passes}
    for i in range(1, S):
        a = R[i].T @ acc[i]
        mag = np.linalg.norm(a)
        force = np.zeros((3, 3))
        if mag > 1e-6:
            force = -kappa * mag * _squash(a / mag)
        if i in kicks:   # each laser pass sends a little shiver through the jelly
            Ed += -0.18 * _squash(np.asarray(kicks[i], float))
        Ed += (force - w * w * E - 2 * zeta * w * Ed) * dt
        E = E + Ed * dt
        m = np.abs(E).max()
        if m > 0.15:
            E *= 0.15 / m
        out[i] = E
    return times, out
