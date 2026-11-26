"""Fig. 1: one DTD query under the eight shift types (severity 1 and 3) and KTH-TIPS2-b real capture variation."""
import os, sys, io, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "figs")
os.makedirs(out_dir, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
qbase = np.load(os.path.join(RUNS, "cache", "dtd_query_base256.npy"), mmap_mode="r")
lab = json.load(open(os.path.join(ROOT, "results", "dtd_labels.json")))
k = 5  # a fixed query image
img = Image.fromarray(np.array(qbase[k]))
names = {"low_light": "Low light", "colour_temp": "Colour temp.", "rotation": "Rotation", "perspective": "Perspective",
         "partial_crop": "Partial view", "blur": "Blur", "jpeg": "JPEG", "downscale": "Down-sampling"}
kth = load_kth()
def kimg(sc, im, smp="cork_a"):
    r = [x for x in kth if x["sample"] == smp and x["scale"] == sc and x["im"] == im][0]
    return Image.open(r["path"]).convert("RGB")
fig = plt.figure(figsize=(7.16, 2.55))
gs = fig.add_gridspec(3, 10, left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.05, hspace=0.32)
ax = fig.add_subplot(gs[0:2, 0]); ax.imshow(img); ax.set_title("Clean query", fontsize=7); ax.axis("off")
for j, t in enumerate(TYPES):
    for i, l in enumerate((1, 3)):
        a = fig.add_subplot(gs[i, j + 1]); a.imshow(perturb(img, t, l, f"dtd_q{lab['query_index'][k]}")); a.axis("off")
        if i == 0: a.set_title(names[t], fontsize=7)
        if j == 0: a.text(-0.08, 0.5, f"s{l}", transform=a.transAxes, ha="right", va="center", fontsize=7)
kcond = [(6, 1, "Reference"), (6, 3, "Illum. (side)"), (6, 10, "Illum. (ambient)"), (6, 4, "Pose (right)"), (6, 7, "Pose (left)"),
         (2, 1, "Scale 2"), (4, 1, "Scale 4"), (8, 1, "Scale 8"), (10, 1, "Scale 10")]
ax = fig.add_subplot(gs[2, 0]); ax.axis("off"); ax.text(0.5, 0.5, "KTH-TIPS2-b\n(real capture)", ha="center", va="center", fontsize=7)
for j, (sc, im, nm) in enumerate(kcond):
    a = fig.add_subplot(gs[2, j + 1]); a.imshow(kimg(sc, im)); a.axis("off"); a.set_title(nm, fontsize=6.5, pad=2)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out_dir, f"fig1_examples.{ext}"), dpi=600)
print("saved", out_dir)
