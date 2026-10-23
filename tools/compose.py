"""Adds the titles to the rendered frames and encodes the final video with sound.

usage: python3 tools/compose.py <frames_dir> <timeline.json> <audio.wav> <out.mp4> [--scale S] [--crf N]

Overlays (all drawn with Pillow):
  * a progress row "1 . 10 . 25 . 100 . 250 . 500 . 1000" with the current stage lit,
  * the stage title: the slice count in a rainbow gradient + "SLICES" underneath,
    popping in when the stage starts and fading out when the next one begins,
  * an end card asking which one was the most satisfying.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = "/usr/share/fonts/opentype/inter"
F_NUM = os.path.join(FONT_DIR, "InterDisplay-Black.otf")
F_WORD = os.path.join(FONT_DIR, "Inter-SemiBold.otf")
F_ROW = os.path.join(FONT_DIR, "Inter-Bold.otf")
F_END = os.path.join(FONT_DIR, "InterDisplay-ExtraBold.otf")

RAINBOW = [(255, 64, 72), (255, 150, 40), (255, 220, 60), (70, 220, 110), (40, 200, 240), (90, 120, 255),
           (190, 90, 255), (255, 80, 170)]


def rainbow_strip(w, h, phase=0.0):
    x = (np.arange(w) / max(w - 1, 1) + phase) % 1.0
    k = x * (len(RAINBOW) - 1)
    i = np.clip(k.astype(int), 0, len(RAINBOW) - 2)
    f = (k - i)[:, None]
    cols = np.array(RAINBOW, float)
    row = cols[i] * (1 - f) + cols[i + 1] * f
    return np.repeat(row[None, :, :], h, axis=0)


def text_layer(text, font, fill, tracking=0):
    """RGBA image of a single text line (tracking in px between letters)."""
    asc, desc = font.getmetrics()
    widths = [font.getlength(ch) for ch in text]
    w = int(sum(widths) + tracking * (len(text) - 1) + 8)
    h = asc + desc + 8
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    x = 4
    for ch, cw in zip(text, widths):
        d.text((x, 4), ch, font=font, fill=255)
        x += cw + tracking
    if isinstance(fill, np.ndarray):
        rgb = fill[:h, :w] if fill.shape[0] >= h and fill.shape[1] >= w else np.resize(fill, (h, w, 3))
    else:
        rgb = np.zeros((h, w, 3)) + np.array(fill, float)
    a = np.asarray(im, float)
    return np.dstack([rgb, a]).astype(np.uint8)


def with_shadow(rgba, blur=14, offset=(0, 8), opacity=0.55, pad=40):
    h, w = rgba.shape[:2]
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), float)
    a = Image.fromarray(rgba[..., 3])
    sh = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    sh.paste(a, (pad + offset[0], pad + offset[1]))
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(blur)), float) * opacity
    out[..., 3] = sh
    fg = rgba.astype(float)
    region = out[pad:pad + h, pad:pad + w]
    fa = fg[..., 3:4] / 255.0
    region[..., :3] = fg[..., :3] * fa + region[..., :3] * (1 - fa)
    region[..., 3] = fg[..., 3] + region[..., 3] * (1 - fa[..., 0])
    return out.clip(0, 255).astype(np.uint8)


def stack_center(layers, gap):
    w = max(l.shape[1] for l in layers)
    h = sum(l.shape[0] for l in layers) + gap * (len(layers) - 1)
    out = np.zeros((h, w, 4), np.uint8)
    y = 0
    for l in layers:
        x = (w - l.shape[1]) // 2
        out[y:y + l.shape[0], x:x + l.shape[1]] = l
        y += l.shape[0] + gap
    return out


def title_image(count, word, W):
    num = str(count)
    f_num = ImageFont.truetype(F_NUM, int(W * 0.205))
    f_word = ImageFont.truetype(F_WORD, int(W * 0.050))
    nw = int(f_num.getlength(num)) + 40
    grad = rainbow_strip(max(nw, 10), 400)
    num_l = text_layer(num, f_num, grad, tracking=-4)
    word_l = text_layer(word, f_word, (255, 255, 255), tracking=int(W * 0.016))
    return with_shadow(stack_center([num_l, word_l], gap=-int(W * 0.012)))


def end_image(W):
    f1 = ImageFont.truetype(F_END, int(W * 0.064))
    l1 = text_layer("WHICH ONE WAS THE", f1, (255, 255, 255), tracking=2)
    l2 = text_layer("MOST SATISFYING?", f1, rainbow_strip(int(W * 0.9), 300), tracking=2)
    return with_shadow(stack_center([l1, l2], gap=4))


def row_images(counts, W):
    f = ImageFont.truetype(F_ROW, int(W * 0.030))
    return [text_layer(str(c), f, (255, 255, 255)) for c in counts]


def blit(frame, rgba, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.003:
        return
    im = rgba
    if abs(scale - 1.0) > 1e-3:
        h, w = rgba.shape[:2]
        im = np.asarray(Image.fromarray(rgba).resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                                      Image.BICUBIC))
    h, w = im.shape[:2]
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    H, Wf = frame.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(Wf, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    src = im[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = src[..., 3:4] / 255.0 * alpha
    dst = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (src[..., :3] * a + dst * (1 - a)).astype(np.uint8)


def ease_out_back(x, s=1.7):
    x = min(max(x, 0.0), 1.0) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("timeline")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--size", type=int, nargs=2, default=None, help="output size (frames are resized)")
    args = ap.parse_args()
    tl = json.load(open(args.timeline))
    fps = tl["fps"]
    n = tl["nframes"]
    first = Image.open(os.path.join(args.frames, "00000.png"))
    W, H = args.size if args.size else first.size
    stages = tl["stages"]
    titles = [title_image(s["count"], "SLICE" if s["count"] == 1 else "SLICES", W) for s in stages]
    rows = row_images([s["count"] for s in stages], W)
    endc = end_image(W)
    total = tl["total"]
    row_gap = int(W * 0.032)
    row_w = sum(r.shape[1] for r in rows) + row_gap * (len(rows) - 1)
    dot = np.zeros((10, 10, 4), np.uint8)
    yy, xx = np.mgrid[0:10, 0:10]
    dot[..., :3] = 255
    dot[..., 3] = (np.clip(4.2 - np.hypot(xx - 4.5, yy - 4.5), 0, 1) * 255).astype(np.uint8)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-i", args.audio, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-tune", "film", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-movflags", "+faststart", "-shortest", args.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    y_row = int(H * 0.085)
    y_title = int(H * 0.168)
    for i in range(n):
        t = i / fps
        im = Image.open(os.path.join(args.frames, f"{i:05d}.png")).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        frame = np.array(im)
        k = max(j for j, s in enumerate(stages) if t >= s["t_start"] - 1e-9)
        st = stages[k]
        tau = t - st["t_start"]
        # progress row
        intro = min(1.0, max(0.0, t / 0.6))
        x = W / 2 - row_w / 2
        for j, r in enumerate(rows):
            lit = 1.0 if j == k else (0.55 if j < k else 0.28)
            if j == k:
                lit = 0.28 + 0.72 * min(1.0, tau / 0.35)
            if j == k - 1:
                lit = 0.55 + 0.45 * max(0.0, 1 - tau / 0.35)
            blit(frame, r, x + r.shape[1] / 2, y_row, alpha=lit * intro)
            if j < len(rows) - 1:
                blit(frame, dot, x + r.shape[1] + row_gap / 2, y_row + 2, alpha=0.35 * intro)
            x += r.shape[1] + row_gap
        # stage title: pop in, fade out as the next stage starts
        t_in = tau - 0.12
        a = min(1.0, max(0.0, t_in / 0.18))
        sc = 0.78 + 0.22 * ease_out_back(t_in / 0.42)
        end_fade = 1.0
        if k == len(stages) - 1:
            end_fade = 1 - min(1.0, max(0.0, (t - (total - 3.2)) / 0.4))
        blit(frame, titles[k], W / 2, y_title + titles[k].shape[0] / 2 * 0.0, alpha=a * end_fade, scale=sc)
        if k > 0 and tau < 0.3:   # previous title leaving
            blit(frame, titles[k - 1], W / 2, y_title - tau * 120, alpha=1 - tau / 0.3, scale=1 - tau * 0.3)
        # end card
        if k == len(stages) - 1:
            ea = min(1.0, max(0.0, (t - (total - 2.9)) / 0.5))
            if ea > 0:
                blit(frame, endc, W / 2, y_title, alpha=ea, scale=0.92 + 0.08 * ease_out_back((t - (total - 2.9)) / 0.6))
        ff.stdin.write(frame.tobytes())
        if i % 100 == 0:
            print(f"compose {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
