"""Paper figures (editable code). Fig. 1: query-side shifts (DTD) and real capture variation (KTH-TIPS2-b).
Fig. 2: accuracy-latency trade-off of all efficiency configurations (severity-2 mean shifted mAP vs median CPU latency).
Usage: python make_figures.py <out_dir>"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, RUNS, TYPES, SEV, perturb, load_kth

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

def fig1():
    qb = np.load(os.path.join(RUNS, "cache", "dtd_query_base256.npy"), mmap_mode="r")
    lab = json.load(open(os.path.join(ROOT, "results", "dtd_labels.json")))
    k = 5; img = Image.fromarray(np.array(qb[k]))
    names = {"low_light": "Low light", "colour_temp": "Color temp.", "rotation": "Rotation", "perspective": "Perspective",
             "partial_crop": "Partial view", "blur": "Blur", "jpeg": "JPEG", "downscale": "Down-sampling"}
    fmt = {"low_light": lambda v: f"\u00d7{v}", "colour_temp": lambda v: f"a={v}", "rotation": lambda v: f"{v}\u00b0", "perspective": lambda v: f"{int(v*100)}%",
           "partial_crop": lambda v: f"{int(v*100)}% area kept" if v == 0.1 else f"{int(v*100)}%", "blur": lambda v: f"\u03c3={v:g}", "jpeg": lambda v: f"q={v}", "downscale": lambda v: f"1/{v}"}
    kth = load_kth()
    def kimg(sc, im, smp="cork_a"):
        r = [x for x in kth if x["sample"] == smp and x["scale"] == sc and x["im"] == im][0]
        return Image.open(r["path"]).convert("RGB")
    fig = plt.figure(figsize=(7.16, 2.15))
    gs = fig.add_gridspec(4, 9, left=0.01, right=0.995, top=0.995, bottom=0.045, wspace=0.07, hspace=0.34,
                          height_ratios=[0.20, 1, 0.20, 1])
    def title_row(r, txt):
        a = fig.add_subplot(gs[r, :]); a.axis("off"); a.text(0.0, 0.0, txt, fontsize=8.5, fontweight="bold", va="bottom", ha="left", transform=a.transAxes)
    title_row(0, "(a) DTD: query perturbations at severity s3 (gallery images unchanged); labels give s1 / s3")
    ax = fig.add_subplot(gs[1, 0]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_linewidth(1.2); s_.set_color("#1f4e79")
    ax.set_xlabel("Clean query", fontsize=7, labelpad=1.5)
    for j, t in enumerate(TYPES):
        for i, l in enumerate((3,)):
            a = fig.add_subplot(gs[1 + i, j + 1]); a.imshow(perturb(img, t, l, f"dtd_q{lab['query_index'][k]}"))
            a.set_xticks([]); a.set_yticks([])
            for s_ in a.spines.values(): s_.set_linewidth(0.4); s_.set_color("#888")
            a.set_xlabel(f"{fmt[t](SEV[t][0])} / {fmt[t](SEV[t][2])}", fontsize=6.3, labelpad=1.2)
            if i == 0: a.set_title(names[t], fontsize=7.2, pad=2.5)
    title_row(2, "(b) KTH-TIPS2-b: real capture variation of one physical sample (cork, sample a)")
    kcond = [(6, 1, "Reference"), (6, 3, "Illum. 45\u00b0 side"), (6, 2, "Illum. 45\u00b0 top"), (6, 10, "Illum. ambient"), (6, 4, "Pose 22.5\u00b0 right"),
             (2, 1, "Scale 2"), (4, 1, "Scale 4"), (8, 1, "Scale 8"), (10, 1, "Scale 10")]
    for j, (sc, im, nm) in enumerate(kcond):
        a = fig.add_subplot(gs[3, j]); a.imshow(kimg(sc, im)); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values(): s_.set_linewidth(1.2 if j == 0 else 0.4); s_.set_color("#1f4e79" if j == 0 else "#888")
        a.set_xlabel(nm, fontsize=6.3, labelpad=1.2)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig1_shifts.{ext}"), dpi=600)
    plt.close(fig)

def fig2():
    r = json.load(open(os.path.join(ROOT, "results", "r2_posthoc.json")))["E3_pareto_sev2"]
    col = {"CLIP-B/32": "#8fb3d9", "CLIP-B/16": "#3f6fa8", "SigLIP-B/16": "#6a51a3", "DINOv2-S/14": "#e6863b", "DINOv2-B/14": "#b3501b",
           "ResNet-50": "#4d9a6a", "MobileNetV2": "#9cc98a"}
    mk = {"fp32-224": "o", "int8-224": "^", "fp32-160": "s", "fp32-112": "D"}
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.axvline(100, color="#999", lw=0.7, ls="--", zorder=0); ax.text(96, 0.105, "100 ms", fontsize=6.2, color="#666", rotation=90, va="bottom", ha="right")
    for p in r:
        ax.scatter(p["latency_ms_sessions_median"], p["shifted_mAP_sev2"], s=26 if p["variant"] == "fp32-224" else 22, marker=mk[p["variant"]],
                   facecolor=col[p["model"]], edgecolor="k" if p["pareto_sessions"] else "none", linewidth=0.7, zorder=3)
    front = sorted([p for p in r if p["pareto_sessions"]], key=lambda p: p["latency_ms_sessions_median"])
    ax.step([p["latency_ms_sessions_median"] for p in front], [p["shifted_mAP_sev2"] for p in front], where="post", color="k", lw=0.8, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(5, 200); ax.set_ylim(0.06, 0.53)
    ax.set_xticks([5, 10, 20, 50, 100, 200]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Median CPU latency, 4 threads, batch 1 (ms)", fontsize=7.5); ax.set_ylabel("Shifted mAP (severity 2, mean)", fontsize=7.5)
    ax.tick_params(labelsize=7, width=0.6, length=2.5); ax.grid(True, which="major", lw=0.3, color="#ddd")
    h1 = [Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor="none", markersize=5, label=m) for m, c in col.items()]
    h2 = [Line2D([], [], marker=v, ls="", markerfacecolor="#bbb", markeredgecolor="#444", markersize=4.5, label=l)
          for l, v in [("FP32 224", "o"), ("INT8 224", "^"), ("FP32 160", "s"), ("FP32 112", "D")]]
    l1 = ax.legend(handles=h1, loc="upper left", fontsize=5.6, frameon=True, framealpha=0.95, handletextpad=0.1, borderpad=0.3, labelspacing=0.2, ncol=4, columnspacing=0.5)
    ax.add_artist(l1)
    ax.legend(handles=h2 + [Line2D([], [], color="k", lw=0.8, label="Pareto front")], loc="lower right", fontsize=5.8, frameon=True, framealpha=0.9,
              handletextpad=0.2, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig2_pareto.{ext}"), dpi=600)
    plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); print("saved", OUT)








