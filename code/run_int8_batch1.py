"""POST HOC (review round 3): INT8 queries encoded one image at a time (online setting) against the batch-32 INT8 gallery.
DINOv2-S/14 and DINOv2-B/14, clean + eight severity-2 shifts on the 940 DTD queries."""
import os, sys, json, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import analyze as A
qb = np.load(os.path.join(RUNS, "cache", "dtd_query_base256.npy"), mmap_mode="r")
qi = json.load(open(os.path.join(ROOT, "results", "dtd_labels.json")))["query_index"]
conds = [("clean", None, 0)] + [(f"{t}_s2", t, 2) for t in TYPES]
out = {}
for m in sys.argv[1:] or ["DINOv2-S/14", "DINOv2-B/14"]:
    model = load_model(m, int8=True); tag = A.tag(m, "int8", 224); d = os.path.join(RUNS, "dtd_b1", tag); os.makedirs(d, exist_ok=True)
    G = A.load(tag, "gallery"); R = {}
    for c, t, l in conds:
        p = os.path.join(d, c + ".npy")
        if not os.path.exists(p):
            t0 = time.time(); E = []
            for k in range(len(qi)):
                im = Image.fromarray(np.array(qb[k])); im = im if t is None else perturb(im, t, l, f"dtd_q{int(qi[k])}")
                E.append(embed(model, m, preprocess(to_uint8_tensor([im]), m, 224)).numpy())
            E = np.concatenate(E); np.save(p, E)
            log_jsonl("runs.jsonl", dict(tag=tag + "__batch1query", model=m, variant="int8", resolution=224, dataset="DTD", condition=c, n_images=len(E), wall_s=round(time.time() - t0, 1), batch=1, **env_info()))
        Q = np.load(p); ap, _ = ap_per_query(Q, G, A.q_lab, A.g_lab)
        Q32 = A.load(tag, c); ap32, _ = ap_per_query(Q32, G, A.q_lab, A.g_lab)
        R[c] = {"mAP_b1": float(ap.mean()), "mAP_b32": float(ap32.mean()), "cos_b1_b32_median": float(np.median((Q * Q32).sum(1)))}
        print(m, c, R[c], flush=True)
    cl = R["clean"]["mAP_b1"]; sh = np.mean([R[c]["mAP_b1"] for c, _, _ in conds[1:]])
    R["summary"] = {"clean_b1": cl, "shifted_sev2_b1": float(sh), "RR_b1": float(sh / cl),
                    "clean_b32": R["clean"]["mAP_b32"], "shifted_sev2_b32": float(np.mean([R[c]["mAP_b32"] for c, _, _ in conds[1:]]))}
    out[m] = R; del model
prev = json.load(open(os.path.join(ROOT, "results", "r3_int8_batch1.json"))) if os.path.exists(os.path.join(ROOT, "results", "r3_int8_batch1.json")) else {}
prev.update(out); json.dump(prev, open(os.path.join(ROOT, "results", "r3_int8_batch1.json"), "w"), indent=1)
print(json.dumps({m: v["summary"] for m, v in out.items()}, indent=1))
