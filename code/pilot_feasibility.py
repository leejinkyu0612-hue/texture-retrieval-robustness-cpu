"""Feasibility pilot (before the pre-registered plan): checks preprocessing parity with HF processors,
that every model/variant runs, and throughput for budgeting. NO retrieval accuracy is computed here."""
import io, time, json, sys, os
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from transformers import AutoImageProcessor

g_bytes, g_lab, q_bytes, q_lab = load_dtd()
print("DTD gallery", len(g_bytes), "query", len(q_bytes), "classes", len(set(g_lab)))
recs = load_kth(); print("KTH images", len(recs))
imgs = [centre_base(Image.open(io.BytesIO(b))) for b in q_bytes[:64]]
# perturbation sanity
for t in TYPES:
    for l in (1, 2, 3):
        p = perturb(imgs[0], t, l, "q0"); assert p.size == (256, 256) and p.mode == "RGB", (t, l)
print("perturbations ok")
u8 = to_uint8_tensor(imgs)
out = {}
for name, (hf, fam, emb) in MODELS.items():
    proc = AutoImageProcessor.from_pretrained(hf)
    ref = proc(images=imgs[:4], return_tensors="pt")["pixel_values"]
    mine = preprocess(u8[:4], name, 224)
    diff = float((ref - mine).abs().mean())
    m = load_model(name)
    res = {"prep_mean_abs_diff_vs_hf": round(diff, 4)}
    for tgt in (224, 160, 112):
        x = preprocess(u8, name, tgt)
        embed(m, name, x[:8], tgt)
        t0 = time.perf_counter(); e = embed(m, name, x, tgt); dt = time.perf_counter() - t0
        res[f"fp32_{tgt}_img_per_s"] = round(64 / dt, 1); res["dim"] = e.shape[1]
    if not name.startswith(("ResNet", "MobileNet")):
        mq = load_model(name, int8=True)
        x = preprocess(u8, name, 224)
        embed(mq, name, x[:8], 224)
        t0 = time.perf_counter(); e2 = embed(mq, name, x, 224); dt = time.perf_counter() - t0
        res["int8_224_img_per_s"] = round(64 / dt, 1)
    out[name] = res; print(name, res, flush=True)
t0 = time.perf_counter(); d = lbp_colour_descriptor(u8); print("LBP img/s", round(64 / (time.perf_counter() - t0), 1), d.shape)
json.dump(out, open(os.path.join(ROOT, "results", "pilot_feasibility.json"), "w"), indent=1)

