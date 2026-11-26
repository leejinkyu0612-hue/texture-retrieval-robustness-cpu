"""Generate LaTeX table rows from results/*.json and logs/latency.jsonl (no hand-typed numbers)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, TYPES
R = os.path.join(ROOT, "results")
dtd = json.load(open(os.path.join(R, "dtd_main.json")))
kth = json.load(open(os.path.join(R, "kth_main.json")))["reference_frontal"]
rq3 = json.load(open(os.path.join(R, "rq3_variants.json")))
ex = json.load(open(os.path.join(R, "rq3_decisions_and_exploratory.json")))["exploratory"]["e1_relative_robustness_sev2"]
lat = {}
for l in open(os.path.join(ROOT, "logs", "latency.jsonl"), encoding="utf-8"):
    d = json.loads(l); lat[(d["model"], d["variant"], d["resolution"], d["threads"])] = d
ORDER = ["CLIP-B/32", "CLIP-B/16", "SigLIP-B/16", "DINOv2-S/14", "DINOv2-B/14", "ResNet-50", "MobileNetV2", "LBP+colour"]
f3 = lambda x: "%.3f" % x
sg = lambda x: ("$+$%.3f" % x) if x >= 0.0005 else (("$-$%.3f" % -x) if x <= -0.0005 else "0.000")
rows = []
for m in ORDER:
    D = dtd[m]; c = D["clean"]["mAP"]
    allp = np.mean([D[f"{t}_s{l}"]["mAP"] for t in TYPES for l in (1, 2, 3)])
    s3 = [D[f"{t}_s3"]["mAP"] for t in TYPES]
    k = kth[m]
    rows.append(" & ".join([m, f3(c), f3(allp), "%.2f" % (allp / c)] + [f3(x) for x in s3] + [f3(k["cat_mAP"]), f3(k["inst_R1"])]) + r" \\")
main = "\n".join(rows)
rows = []
for m in ORDER[:-1]:
    b = lat[(m, "fp32", 224, 4)]; q = lat.get((m, "int8", 224, 4)); r1 = lat[(m, "fp32", 112, 4)]; V = rq3[m]
    rows.append(" & ".join([m, "%.1f" % b["params_M"], "%.0f" % b["latency_ms_median"], ("%.0f" % q["latency_ms_median"]) if q else "--",
        "%.0f" % r1["latency_ms_median"],
        sg(V["int8_224"]["delta_clean"]) if "int8_224" in V else "--",
        ("%.2f/%.2f" % (ex[m]["RR_fp32_224"], ex[m]["RR_int8_224"])) if "RR_int8_224" in ex[m] else "--",
        sg(V["fp32_112"]["delta_clean"]), sg(V["int8emb"]["delta_clean"]), sg(V["bin"]["delta_clean"])]) + r" \\")
eff = "\n".join(rows)
open(os.path.join(R, "table_main_rows.tex"), "w", encoding="utf-8").write(main + "\n")
open(os.path.join(R, "table_eff_rows.tex"), "w", encoding="utf-8").write(eff + "\n")
print(main); print(); print(eff)
