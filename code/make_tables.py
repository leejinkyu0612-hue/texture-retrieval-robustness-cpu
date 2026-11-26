"""Generate LaTeX table bodies and a numbers sheet from results/*.json (no hand-typed numbers)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, TYPES
R = os.path.join(ROOT, "results")
dtd = json.load(open(os.path.join(R, "dtd_main.json")))
kth = json.load(open(os.path.join(R, "kth_main.json")))["reference_frontal"]
lat = {}
lp = os.path.join(ROOT, "logs", "latency.jsonl")
if os.path.exists(lp):
    for l in open(lp, encoding="utf-8"):
        d = json.loads(l); lat[(d["model"], d["variant"], d["resolution"], d["threads"])] = d  # last record wins
ORDER = ["CLIP-B/32", "CLIP-B/16", "SigLIP-B/16", "DINOv2-S/14", "DINOv2-B/14", "ResNet-50", "MobileNetV2", "LBP+colour"]
SHORT = {"low_light": "Low", "colour_temp": "Col", "rotation": "Rot", "perspective": "Per", "partial_crop": "Crop", "blur": "Blur", "jpeg": "JPEG", "downscale": "Down"}
f3 = lambda x: ("%.3f" % x)[1:] if 0 <= x < 1 else "%.3f" % x
out = []
# Table: clean, mean over 24 shifts, severity-3 per type, KTH
rows = []
for m in ORDER:
    if m not in dtd: continue
    D = dtd[m]; c = D["clean"]["mAP"]
    allp = np.mean([D[f"{t}_s{l}"]["mAP"] for t in TYPES for l in (1, 2, 3)])
    s3 = [D[f"{t}_s3"]["mAP"] for t in TYPES]
    k = kth.get(m, {})
    row = [m, f3(c), f3(allp), "%.2f" % (allp / c)] + [f3(x) for x in s3] + ([f3(k["cat_mAP"]), "%.1f" % (100 * k["inst_R1"])] if k else ["--", "--"])
    rows.append(" & ".join(row) + r" \\")
out.append("% TABLE main\n" + "\n".join(rows))
# Table: efficiency
rows = []
for m in ORDER[:-1]:
    base = lat.get((m, "fp32", 224, 4)); q = lat.get((m, "int8", 224, 4)); r112 = lat.get((m, "fp32", 112, 4))
    if not base: continue
    rows.append(" & ".join([m, "%.1f" % base["params_M"], "%.0f" % base["latency_ms_median"], ("%.0f" % q["latency_ms_median"]) if q else "--",
                            ("%.0f" % r112["latency_ms_median"]) if r112 else "--", "%.0f" % base["throughput_img_s_b32"], "%d" % base["emb_dim"]]) + r" \\")
out.append("% TABLE efficiency\n" + "\n".join(rows))
open(os.path.join(R, "tables_autogen.tex"), "w", encoding="utf-8").write("\n\n".join(out) + "\n")
print("\n\n".join(out))
