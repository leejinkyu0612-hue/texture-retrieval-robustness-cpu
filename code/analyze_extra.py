"""Pre-specified H3/H4/H5 decisions from rq3_variants.json, plus EXPLORATORY analyses added after seeing results:
(e1) relative-robustness (RR) version of the interaction, (e2) mean-centred sign binarisation, (e3) threshold sensitivity."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, TYPES
import analyze as A

R = json.load(open(os.path.join(A.OUT, "rq3_variants.json")))
out = {"pre_specified": {}, "exploratory": {}}
tr = [m for m in R if "int8_224" in R[m]]
# H3: |delta_clean| < 0.01 with CI inside +-0.01 (and sensitivity 0.005 / 0.02)
for thr in (0.005, 0.01, 0.02):
    out["pre_specified"][f"H3_equivalence_thr_{thr}"] = {m: bool(-thr < R[m]["int8_224"]["delta_clean_ci"][0] and R[m]["int8_224"]["delta_clean_ci"][1] < thr) for m in tr}
out["pre_specified"]["H3_int_DD"] = {m: {"DD": R[m]["int8_224"]["DD_mean"], "ci": R[m]["int8_224"]["DD_ci"], "p": R[m]["int8_224"]["p_wilcoxon_DD"]} for m in tr}
# H4: res 112 DD for blur and downscale (severity 2), averaged over models
h4 = {}
for t in ("blur_s2", "downscale_s2"):
    vals = [R[m]["fp32_112"]["per_type"][t]["DD"] for m in R if "fp32_112" in R[m]]
    h4[t] = {"mean_DD_over_models": float(np.mean(vals)), "per_model": {m: R[m]["fp32_112"]["per_type"][t]["DD"] for m in R}}
out["pre_specified"]["H4"] = h4
# H5
out["pre_specified"]["H5_int8emb_equiv_0.01"] = {m: bool(-0.01 < R[m]["int8emb"]["delta_clean_ci"][0] and R[m]["int8emb"]["delta_clean_ci"][1] < 0.01) for m in R}
out["pre_specified"]["H5_bin_reduces_more_than_0.01"] = {m: bool(R[m]["bin"]["delta_clean_ci"][1] < -0.01) for m in R}
# e1: RR-based interaction (exploratory): mean RR over 8 severity-2 shifts, variant vs base
conds = [f"{t}_s2" for t in TYPES]
e1 = {}
for m in R:
    base = {c: A.per_query(A.tag(m), c)[0] for c in ["clean"] + conds}
    rr_b = np.mean([base[c].mean() / base["clean"].mean() for c in conds])
    e1[m] = {"RR_fp32_224": float(rr_b)}
    for key, (v, r, comp) in {"int8_224": ("int8", 224, None), "fp32_160": ("fp32", 160, None), "fp32_112": ("fp32", 112, None), "bin": ("fp32", 224, "bin")}.items():
        pv = {c: A.per_query(A.tag(m, v, r), c, comp) for c in ["clean"] + conds}
        if any(x is None for x in pv.values()): continue
        e1[m][f"RR_{key}"] = float(np.mean([pv[c][0].mean() / pv["clean"][0].mean() for c in conds]))
out["exploratory"]["e1_relative_robustness_sev2"] = e1
# e2: mean-centred sign binarisation (centre = gallery mean)
def binc(G, Q):
    mu = G.mean(0); return np.sign(G - mu) / np.sqrt(G.shape[1]), np.sign(Q - mu) / np.sqrt(Q.shape[1])
e2 = {}
for m in R:
    t = A.tag(m); G = A.load(t, "gallery"); Q = A.load(t, "clean"); Gb, Qb = binc(G, Q)
    ap, _ = A.ap_per_query(Qb, Gb, A.q_lab, A.g_lab); apc = A.per_query(t, "clean")[0]; d = ap - apc
    e2[m] = {"clean_mAP_centred_bin": float(ap.mean()), "delta": float(d.mean()), "delta_ci": A.ci(d[A.BOOT].mean(1))}
out["exploratory"]["e2_centred_binarisation"] = e2
json.dump(out, open(os.path.join(A.OUT, "rq3_decisions_and_exploratory.json"), "w"), indent=1)
print(json.dumps(out["pre_specified"]["H4"], indent=0)[:600])
for m in R: print(m, {k: round(v, 3) for k, v in e1[m].items()}, "centred-bin", round(e2[m]["clean_mAP_centred_bin"], 3), round(e2[m]["delta"], 3))
print("H3 thr0.01", out["pre_specified"]["H3_equivalence_thr_0.01"])
