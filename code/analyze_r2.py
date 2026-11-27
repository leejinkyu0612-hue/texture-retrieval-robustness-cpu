"""POST-HOC analyses requested in review round 2 (labelled post hoc in the paper).
E1/E2/E6: INT8 reproducibility, quantisation-engine sensitivity and batch-size sensitivity on 94 DTD query images (2 per class).
E3: unified Pareto at severity 2 (clean + 8 severity-2 shifts) for all FP32/INT8/160/112 variants.
E7: rank-correlation sensitivity (neural only, leave-one-model-out, clean DTD vs KTH).
E8: binary-code diversity and score ties for 1-bit codes.  R: random-ranking mAP baseline on DTD."""
import os, sys, json, io
import numpy as np, torch
from scipy.stats import kendalltau
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import analyze as A
from common import TYPES, load_model, embed, preprocess, to_uint8_tensor, ap_per_query, RUNS
OUT = A.OUT; res = {}
lat = {}
for l in open(os.path.join(A.ROOT, "logs", "latency.jsonl"), encoding="utf-8"):
    d = json.loads(l); lat[(d["model"], d["variant"], d["resolution"], d["threads"])] = d["latency_ms_median"]
# E3 unified Pareto
conds = [f"{t}_s2" for t in TYPES]; pts = []
for m in A.MODELS[:-1]:
    for v, r in (("fp32", 224), ("int8", 224), ("fp32", 160), ("fp32", 112)):
        t = A.tag(m, v, r)
        if A.load(t, "clean") is None or (m, v, r, 4) not in lat: continue
        sh = float(np.mean([A.per_query(t, c)[0].mean() for c in conds])); cl = float(A.per_query(t, "clean")[0].mean())
        pts.append({"model": m, "variant": f"{v}-{r}", "latency_ms": lat[(m, v, r, 4)], "clean_mAP": cl, "shifted_mAP_sev2": sh})
for p in pts:
    p["pareto"] = not any(q["latency_ms"] <= p["latency_ms"] and q["shifted_mAP_sev2"] >= p["shifted_mAP_sev2"] and q is not p and
                          (q["latency_ms"] < p["latency_ms"] or q["shifted_mAP_sev2"] > p["shifted_mAP_sev2"]) for q in pts)
res["E3_pareto_sev2"] = pts
# E7 rank correlation sensitivity
r2 = json.load(open(os.path.join(OUT, "rq2.json"))); ms = r2["models"]
dp, km = r2["dtd_mean_perturbed_mAP"], r2["kth_cat_mAP"]
dclean = [A.per_query(A.tag(m), "clean")[0].mean() for m in ms]
nn = [i for i, m in enumerate(ms) if m != "LBP+colour"]
res["E7"] = {"tau_neural_only": float(kendalltau([dp[i] for i in nn], [km[i] for i in nn]).statistic),
             "tau_clean_dtd_vs_kth": float(kendalltau(dclean, km).statistic),
             "leave_one_out": {ms[j]: float(kendalltau([dp[i] for i in range(len(ms)) if i != j], [km[i] for i in range(len(ms)) if i != j]).statistic) for j in range(len(ms))}}
# E8 binary codes
e8 = {}
for m in A.MODELS[:-1]:
    G = A.load(A.tag(m), "gallery"); Q = A.load(A.tag(m), "clean")
    bg = (G > 0); uniq = len({r.tobytes() for r in np.packbits(bg, axis=1)}) / len(bg)
    S = np.sign(Q) @ np.sign(G).T
    ties = float(np.mean([(np.sort(s)[::-1][:10][:, None] == s[None, :]).sum() / 10 for s in S[:100]]))
    e8[m] = {"unique_code_share": float(uniq), "mean_share_positive_bits": float(bg.mean()), "mean_items_tied_with_top10_score": ties}
res["E8_binary"] = e8
# random baseline
rng = np.random.default_rng(2027); Rq = rng.standard_normal((len(A.q_lab), 64)); Rg = rng.standard_normal((len(A.g_lab), 64))
Rq /= np.linalg.norm(Rq, axis=1, keepdims=True); Rg /= np.linalg.norm(Rg, axis=1, keepdims=True)
res["random_ranking_mAP"] = float(ap_per_query(Rq, Rg, A.q_lab, A.g_lab)[0].mean())
# E1/E2/E6 on 94 images
qb = np.load(os.path.join(RUNS, "cache", "dtd_query_base256.npy"), mmap_mode="r")
sel = []
for c in sorted(set(A.q_lab.tolist())): sel.extend(np.where(A.q_lab == c)[0][:2].tolist())
u8 = to_uint8_tensor([Image.fromarray(np.array(qb[i])) for i in sel])
e = {}
for m in ["DINOv2-B/14", "DINOv2-S/14", "CLIP-B/16"]:
    x = preprocess(u8, m, 224)
    f1 = embed(load_model(m), m, x).numpy(); f2 = embed(load_model(m), m, x).numpy()
    out = {"fp32_reload_cos_min": float((f1 * f2).sum(1).min())}
    for eng in [x for x in torch.backends.quantized.supported_engines if x in ("onednn", "x86", "fbgemm")]:
        torch.backends.quantized.engine = eng
        try:
            qa = load_model(m, int8=True); qb2 = load_model(m, int8=True)
            a = embed(qa, m, x).numpy(); b = embed(qb2, m, x).numpy()
            b1 = np.concatenate([embed(qa, m, x[i:i + 1]).numpy() for i in range(len(sel))])
            b8 = np.concatenate([embed(qa, m, x[i:i + 8]).numpy() for i in range(0, len(sel), 8)])
            out[eng] = {"cos_fp32_int8_median": float(np.median((f1 * a).sum(1))), "int8_requantise_cos_min": float((a * b).sum(1).min()),
                        "batch1_vs_batch94_cos_median": float(np.median((b1 * a).sum(1))), "batch8_vs_batch94_cos_median": float(np.median((b8 * a).sum(1))),
                        "nan_or_zero": bool(np.isnan(a).any() or (np.linalg.norm(a, axis=1) == 0).any())}
        except Exception as ex:
            out[eng] = {"error": str(ex)[:200]}
    torch.backends.quantized.engine = "onednn"
    e[m] = out
res["E1_E2_E6_int8_checks_94imgs"] = e
json.dump(res, open(os.path.join(OUT, "r2_posthoc.json"), "w"), indent=1)
for p in pts: print(p["model"], p["variant"], p["latency_ms"], round(p["shifted_mAP_sev2"], 3), "PARETO" if p["pareto"] else "")
print("E7", res["E7"]); print("E8", {k: {a: round(b, 3) for a, b in v.items()} for k, v in e8.items()}); print("random", res["random_ranking_mAP"])
print(json.dumps(e, indent=0))
