"""Procedural ASMR sound design, synthesised from the timeline and physics events.

usage: python3 audio/synth.py <timeline.json> <out.wav>

Every sound is generated from scratch with numpy/scipy:
  * fly-in whoosh + a low jelly "bwomp" when the cube is caught mid-air,
  * laser passes: an electric whine (one voice per beam), sizzle and a wet
    "shhk" while the beams are inside the jelly,
  * a jelly impact (pitched thud + wet slap + squish + wobble tail) for every
    collision reported by the rigid-body simulation, panned by position,
  * soft UI ticks for the titles, camera-dolly whooshes, a quiet room tone.
"""
import json
import math
import sys

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.io import wavfile

SR = 48000
RNG = np.random.default_rng(1234)


def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x)


def noise(n):
    return RNG.standard_normal(n)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


class Mix:
    def __init__(self, dur):
        n = int(dur * SR) + SR
        self.L = np.zeros(n)
        self.R = np.zeros(n)

    def add(self, t, sig, pan=0.0, gain=1.0):
        i = int(round(t * SR))
        if i < 0:
            sig = sig[-i:]
            i = 0
        n = min(len(sig), len(self.L) - i)
        if n <= 0:
            return
        a = (np.clip(pan, -1, 1) + 1) * math.pi / 4
        self.L[i:i + n] += sig[:n] * gain * math.cos(a) * math.sqrt(2)
        self.R[i:i + n] += sig[:n] * gain * math.sin(a) * math.sqrt(2)

    def add_stereo(self, t, l, r, gain=1.0):
        i = int(round(t * SR))
        n = min(len(l), len(self.L) - i)
        self.L[i:i + n] += l[:n] * gain
        self.R[i:i + n] += r[:n] * gain


# ------------------------------------------------------------------ sounds --

def jelly_impact(v, size, floor):
    """Gummy hitting the floor (floor=True) or another gummy: a soft pitched thud,
    a wet slap, a squish and, for small pieces, a crisp little tick."""
    dur = 0.32 + 0.25 * min(1.0, size)
    t = tvec(dur)
    det = RNG.uniform(0.9, 1.1)
    f0 = 185.0 * (0.2 / size) ** 0.33 * det * (0.85 if floor else 1.3)
    f = f0 * (1 + 0.6 * np.exp(-t / 0.010)) * (1 - 0.18 * (1 - np.exp(-t / 0.07)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    tb = 0.038 * (size / 0.2) ** 0.35
    body = np.sin(ph) * np.exp(-t / tb) + 0.35 * np.sin(2.43 * ph) * np.exp(-t / (tb * 0.5))
    hi = min(7000.0, 2400.0 * (0.2 / size) ** 0.22)
    slap = filt(noise(len(t)), "bandpass", [hi * 0.45, hi * 1.9]) * np.exp(-t / 0.007)
    tq = 0.010 + 0.012 * min(1.0, size)
    squish = filt(noise(len(t)), "bandpass", [450, 2300]) * (t / tq) * np.exp(1 - t / tq)
    squish *= 0.7 + 0.3 * np.sin(2 * np.pi * RNG.uniform(25, 45) * t)
    tick = filt(noise(len(t)), "bandpass", [3000, 9000]) * np.exp(-t / 0.0018)
    fw = 5.5 * math.sqrt(0.2 / size)
    tail = np.sin(2 * np.pi * f0 * 0.62 * t) * np.exp(-t / (2.2 * tb)) * (0.55 + 0.45 * np.sin(2 * np.pi * fw * t))
    small = float(np.clip((0.5 - size) / 0.3, 0, 1))
    s = (0.55 * body + (1.25 if floor else 0.9) * slap + 0.65 * squish + 0.18 * tail
         + (0.25 if floor else 0.6) * small * tick)
    s *= np.minimum(1.0, t / 0.0006)            # no click at the onset
    return s


def laser_pass(beams, t_fade_in, t_sweep, t_fade_out, contact):
    """Electric whine for the beams plus the sound of the jelly being cut.
    contact: (start, end) seconds within the sound where beams are inside the cube."""
    dur = t_fade_in + t_sweep + t_fade_out + 0.25
    t = tvec(dur)
    n = len(t)
    env = np.clip(t / t_fade_in, 0, 1) * np.clip((t_fade_in + t_sweep + t_fade_out - t) / t_fade_out, 0, 1)
    env = env ** 1.5
    voices = min(beams, 6)
    whine = np.zeros(n)
    for j in range(voices):
        fj = 740.0 * (1.5 if j % 2 else 1.0) * (1 + 0.009 * (j - (voices - 1) / 2)) * (1 + 0.004 * np.sin(2 * np.pi * (5.5 + j) * t))
        glide = 1 - 0.06 * np.clip((t - t_fade_in - t_sweep) / t_fade_out, 0, 1)
        whine += np.sin(2 * np.pi * np.cumsum(fj * glide) / SR + j)
    whine /= math.sqrt(voices)
    hum = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 7)) * 0.25
    c0, c1 = contact
    cenv = np.clip((t - c0) / 0.02, 0, 1) * np.clip((c1 - t) / 0.03, 0, 1)
    sizzle_src = filt(noise(n), "highpass", 4500.0)
    crackle = (RNG.random(n) < 0.004 * (1 + beams / 3)).astype(float)
    crackle = signal.lfilter([1.0], [1.0, -0.985], crackle)
    sizzle = sizzle_src * (0.35 + 0.65 * np.clip(crackle, 0, 1.5))
    # the wet "shhhk": band-passed noise whose centre falls during the cut
    shh = np.zeros(n)
    bands = [(5200, 9000), (2600, 5600), (1200, 3200)]
    for k, (lo, hi) in enumerate(bands):
        w = np.clip(1 - np.abs((t - c0) / max(c1 - c0, 1e-3) - (k + 0.5) / 3) * 2.2, 0, 1)
        shh += filt(noise(n), "bandpass", [lo, hi]) * w
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 13 * t + RNG.uniform(0, 6))
    squelch = filt(noise(n), "bandpass", [420, 1300]) * lfo
    thick = 1 + 0.08 * min(beams, 9)
    s = (0.10 * whine + 0.04 * hum) * env + (0.20 * sizzle + 0.60 * shh * thick + 0.30 * squelch * thick) * cenv
    # crisp tick when the beams leave the jelly
    k0 = int(c1 * SR)
    tick = filt(noise(int(0.03 * SR)), "highpass", 2500) * np.exp(-tvec(0.03) / 0.004)
    s[k0:k0 + len(tick)] += 0.6 * tick[: max(0, n - k0)]
    return s


def whoosh(dur, f_hi=3200.0, f_lo=380.0, peak=0.75):
    t = tvec(dur)
    n = len(t)
    out = np.zeros(n)
    centers = np.geomspace(f_hi, f_lo, 6)
    for k, fc in enumerate(centers):
        pos = k / (len(centers) - 1)
        w = np.exp(-((t / dur - pos * 0.9) ** 2) / 0.03)
        out += filt(noise(n), "bandpass", [fc * 0.6, fc * 1.5]) * w
    env = np.where(t < peak * dur, (t / (peak * dur)) ** 2, np.exp(-(t - peak * dur) / (0.08 * dur)))
    return out * env


def bwomp(f0=72.0, dur=0.9, wob=1.9):
    t = tvec(dur)
    f = f0 * (1 + 0.35 * np.exp(-t / 0.05))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    s *= 0.6 + 0.4 * np.cos(2 * np.pi * wob * t)
    s += 0.25 * filt(noise(len(t)), "bandpass", [180, 700]) * np.exp(-t / 0.04)
    return s * np.minimum(1, t / 0.004)


def levitation_hum(dur, f0=55.0):
    """Soft magnetic hum while the cube floats; fades in, then drops in pitch at the release."""
    t = tvec(dur)
    drop = np.clip((t - (dur - 0.35)) / 0.35, 0, 1)
    f = f0 * (1 - 0.45 * drop ** 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph + 0.5)
    s *= 1 + 0.15 * np.sin(2 * np.pi * 3.1 * t)
    env = np.clip(t / 0.5, 0, 1) * (1 - drop) ** 1.5
    return s * env


def sub_thump(weight=1.0):
    t = tvec(0.8)
    f = 48.0 * (1 + 0.8 * np.exp(-t / 0.03))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.16 + 0.08 * weight))
    s += 0.5 * filt(noise(len(t)), "lowpass", 300.0) * np.exp(-t / 0.05)
    return s * np.minimum(1, t / 0.003)


def tick(f=950.0):
    t = tvec(0.12)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.018) + 0.4 * np.sin(2 * np.pi * 2.01 * f * t) * np.exp(-t / 0.008)
    return s * np.minimum(1, t / 0.0005)


def release_blip():
    t = tvec(0.25)
    f = 520 * np.exp(-t / 0.08) + 180
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05) * np.minimum(1, t / 0.002)


def room_tone(dur):
    n = int(dur * SR)
    x = filt(noise(n), "lowpass", 260.0, order=2)
    return x / (np.abs(x).max() + 1e-9)


def pad(dur, t_fade=3.0):
    t = tvec(dur)
    notes = [110.0, 164.81, 220.0, 277.18, 329.63]
    s = np.zeros(len(t))
    for i, f in enumerate(notes):
        lfo = 1 + 0.002 * np.sin(2 * np.pi * (0.07 + 0.03 * i) * t + i)
        s += np.sin(2 * np.pi * np.cumsum(f * lfo) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t + i * 1.3))
    s = filt(s, "lowpass", 900.0)
    env = np.clip(t / t_fade, 0, 1) * np.clip((dur - t) / t_fade, 0, 1)
    return s * env / len(notes)


# -------------------------------------------------------------------- main --

def main(timeline_path, out_path):
    tl = json.load(open(timeline_path))
    total = tl["total"]
    mix = Mix(total + 1.0)
    from_stages = tl["stages"]
    for st in from_stages:
        t0 = st["t_start"]
        ax = st["arena_x"]
        # camera dolly from the previous arena (objects slide right -> left)
        if st["index"] > 0:
            w = whoosh(1.05, 2400, 300, peak=0.5)
            ramp = np.linspace(0, 1, len(w))
            mix.add_stereo(t0, w * (1 - 0.6 * ramp), w * (0.4 + 0.6 * ramp), gain=0.10)
        # title tick
        mix.add(t0 + 0.25, tick(1180.0), 0.0, 0.10)
        mix.add(t0 + 0.31, tick(1580.0), 0.0, 0.06)
        # the cube flies in from above and is caught by the levitation field
        catch = st.get("catch", t0 + 0.85)
        mix.add(catch - 0.62, whoosh(0.75, 3600, 420, peak=0.8), 0.0, 0.32)
        mix.add(catch - 0.03, bwomp(), 0.0, 0.36)
        mix.add(catch, levitation_hum(st["release"] - catch + 0.05), 0.0, 0.016)
        # laser passes
        for p in st["passes"]:
            sweep = p["t1"] - p["t0"]
            fi, fo = 0.12, 0.18
            # sweep progress is smoothstep-eased; beams are inside the cube while |lambda| < 1
            lam0, lam1 = -1.35, 1.35
            xs = np.linspace(0, 1, 400)
            lam = lam0 + (lam1 - lam0) * (xs * xs * (3 - 2 * xs))
            inside = np.nonzero(np.abs(lam) < 1.0)[0]
            c0 = fi + xs[inside[0]] * sweep
            c1 = fi + xs[inside[-1]] * sweep
            s = laser_pass(p["beams"], fi, sweep, fo, (c0, c1))
            mix.add(p["t0"] - fi, s, 0.0, 0.55)
        mix.add(st["release"], release_blip(), 0.0, 0.10)
        # impacts from the physics simulation
        # contacts in the first instants after the drop are pieces pushing apart, not impacts
        ev = [e for e in st["impacts"] if e["t"] > st["release"] + 0.06]
        floor_hits = [e["t"] for e in ev if e["floor"]]
        if floor_hits:   # the weight of the whole cube meeting the floor
            mix.add(min(floor_hits) - 0.004, sub_thump(), 0.0, 0.26)
        if ev:
            times = np.array([e["t"] for e in ev])
            order = np.argsort(times)
            times = times[order]
            for i in order:
                e = ev[i]
                lo = np.searchsorted(times, e["t"] - 0.04)
                hi = np.searchsorted(times, e["t"] + 0.04)
                dens = hi - lo
                v = e["v"]
                amp = min(2.2, (v / 3.0) ** 1.15) * (e["size"] / 0.2) ** 0.45
                amp *= 1.0 if e["floor"] else 0.55
                amp /= max(1.0, dens / 4.0) ** 0.35
                if amp < 0.004:
                    continue
                pan = float(np.clip((e["x"] - ax) / 3.2, -1, 1)) * 0.75
                mix.add(e["t"], jelly_impact(v, e["size"], e["floor"]), pan, 0.30 * amp)
    # beds
    mix.add(0.0, room_tone(total + 1.0), 0.0, 0.006)
    p = pad(total + 0.8)
    mix.add(0.0, p, -0.2, 0.030)
    mix.add(0.0, np.roll(p, 1200), 0.2, 0.030)
    # master: small-studio reverb, high-pass, loudness normalisation, look-ahead limiter
    dry = np.stack([mix.L, mix.R])
    ir_t = tvec(0.9)
    wet = np.zeros_like(dry)
    for ch in range(2):
        ir = noise(len(ir_t)) * np.exp(-ir_t / 0.16)
        ir[: int(0.012 * SR)] = 0.0                       # pre-delay
        ir = filt(ir, "lowpass", 5500.0)
        ir /= np.sqrt(np.sum(ir ** 2))
        wet[ch] = signal.fftconvolve(dry[ch], ir)[: dry.shape[1]]
    st_mix = dry + 0.22 * wet
    st_mix = signal.sosfilt(sos("highpass", 28.0), st_mix, axis=1)
    mono = st_mix.mean(axis=0)
    win = int(0.4 * SR)
    cs = np.concatenate([[0.0], np.cumsum(mono ** 2)])
    rms = np.sqrt((cs[win::win // 2] - cs[: -win: win // 2][: len(cs[win::win // 2])]) / win + 1e-12)
    ref = np.percentile(rms, 80)
    st_mix *= 10 ** (-17.0 / 20) / ref                    # loud parts around -17 dBFS RMS
    ceiling = 10 ** (-1.0 / 20)
    peak = np.abs(st_mix).max(axis=0)
    g = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    la = int(0.004 * SR)
    g = minimum_filter1d(g, size=2 * la + 1)
    g = uniform_filter1d(g, size=la + 1)
    st_mix *= g
    st_mix = np.clip(st_mix, -ceiling, ceiling)
    n = int(round(total * SR))
    st_mix = st_mix[:, :n]
    fade = int(0.6 * SR)
    st_mix[:, -fade:] *= np.linspace(1, 0, fade) ** 2
    wavfile.write(out_path, SR, (st_mix.T * 32767).astype(np.int16))
    print(f"wrote {out_path}: {n / SR:.2f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
