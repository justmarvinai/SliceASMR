"""Contact sheet of rendered frames: sheet.py out.png frame_dir first last step [cols] [thumb_w]"""
import glob
import os
import sys

from PIL import Image

out, d, first, last, step = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
cols = int(sys.argv[6]) if len(sys.argv) > 6 else 8
tw = int(sys.argv[7]) if len(sys.argv) > 7 else 180
files = [os.path.join(d, f"{i:05d}.png") for i in range(first, last + 1, step)]
files = [f for f in files if os.path.exists(f)]
ims = [Image.open(f) for f in files]
th = int(tw * ims[0].height / ims[0].width)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * th))
for i, im in enumerate(ims):
    sheet.paste(im.resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
sheet.save(out)
print(len(ims), "frames")
