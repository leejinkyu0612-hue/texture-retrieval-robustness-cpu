"""POST-HOC analyses requested in review round 1 (labelled exploratory/sensitivity in the paper).
a) INT8 sanity: FP32-vs-INT8 cosine of the same images, mixed index (FP32 gallery + INT8 query), quantised layer count/backend.
b) Cluster bootstrap for KTH-TIPS2-b (resample physical samples within material) and category-stratified bootstrap for DTD.
c) KTH same-instance exclusion: material mAP when the query's own physical sample is removed from the catalogue.
d) Patch-grid check for reduced resolutions."""
import os, sys, json
import numpy as np, torch
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__))
import analyze as A
from common import MODELS, TYPES, load_model, ap_per_query
OUT = A.OUT; rng = np.random.default_rng(2027); res = {}
TR = ["CLIP-B/32", "CLIP-B/16", "SigLIP-B/16", "DINOv2-S/14", "DINOv2-B/14"]
# a) INT8 sanity
a = {}
for m in TR:
    tf, tq = A.tag(m), A.tag(m, "int8", 224)
    Gf, Qf, Gq, Qq = A.load(tf, "gallery"), A.load(tf, "clean"), A.load(tq, "gallery"), A.load(tq, "clean")
    cos = (Gf * Gq).sum(1)
    ap_mixed, _ = ap_per_query(Qq, Gf, A.q_lab, A.g_lab)
    S1 = Qf @ Gf.T; S2 = Qq @ Gq.T
    top1 = np.argsort(-S1, 1)[:, :10]; top2 = np.argsort(-S2, 1)[:, :10]
    overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(top1, top2)])
    mdl = load_model(m, int8=True)
    nq = sum(1 for mod in mdl.modules() if "quantized" in type(mod).__module__ and "Linear" in type(mod).__name__)
    a[m] = {"cos_fp32_int8_gallery_median": float(np.median(cos)), "cos_p5": float(np.percentile(cos, 5)),
            "mAP_fp32_fp32": float(A.per_query(tf, "clean")[0].mean()), "mAP_int8_int8": float(A.per_query(tq, "clean")[0].mean()),
            "mAP_fp32gallery_int8query": float(ap_mixed.mean()), "top10_overlap": float(overlap), "n_quantized_linear": int(nq),
            "backend": torch.backends.quantized.engine}
    del mdl
res["a_int8_sanity"] = a
# b) cluster bootstrap for KTH, category-stratified for DTD
meta = json.load(open(os.path.join(OUT, "kth_meta.json")))
sc = np.array([r["scale"] for r in meta]); po = np.array([r["pose"] for r in meta]); il = np.array([r["illum"] for r in meta])
mat = np.array([r["material"] for r in meta]); smp = np.array([r["sample"] for r in meta])
ref = (sc == 6) & (po == "frontal") & (il == "frontal"); gi = np.where(ref)[0]; qi = np.where(~ref)[0]
samples = sorted(set(smp)); by_mat = {}
for s in samples: by_mat.setdefault(s.rsplit("_", 1)[0], []).append(s)
q_smp = smp[qi]; idx_by_s = {s: np.where(q_smp == s)[0] for s in samples}
kth_ap = {}; kth_r1 = {}; kth_ex = {}
for m in A.MODELS:
    E = A.load(A.tag(m), "all", "kth"); G, Q = E[gi], E[qi]
    ap, _ = ap_per_query(Q, G, mat[qi], mat[gi]); S = Q @ G.T
    r1 = (smp[gi][S.argmax(1)] == smp[qi]).astype(float)
    excl = (smp[gi][None, :] == smp[qi][:, None])  # c) same physical sample removed
    ap_ex, r1_ex = ap_per_query(Q, G, mat[qi], mat[gi], exclude=excl)
    kth_ap[m], kth_r1[m], kth_ex[m] = ap, r1, (ap_ex, r1_ex)
B = 1000; boots = []
for b in range(B):
    pick = []
    for mt, ss in by_mat.items():
        for s in rng.choice(ss, size=len(ss), replace=True): pick.append(idx_by_s[s])
    boots.append(np.concatenate(pick))
cb = {}
for m in A.MODELS:
    vals = np.array([kth_ap[m][bi].mean() for bi in boots]); vr = np.array([kth_r1[m][bi].mean() for bi in boots])
    vx = np.array([kth_ex[m][0][bi].mean() for bi in boots])
    cb[m] = {"cat_mAP": float(kth_ap[m].mean()), "cat_mAP_cluster_ci": A.ci(vals), "inst_R1": float(kth_r1[m].mean()), "inst_R1_cluster_ci": A.ci(vr),
             "cat_mAP_excl_same_sample": float(kth_ex[m][0].mean()), "cat_mAP_excl_cluster_ci": A.ci(vx), "R1_material_excl_same_sample": float(kth_ex[m][1].mean())}
res["b_kth_cluster_bootstrap"] = cb
# DTD category-stratified bootstrap for clean mAP and Kendall tau with cluster bootstrap on KTH
ql = A.q_lab; cls = {c: np.where(ql == c)[0] for c in sorted(set(ql.tolist()))}
conds = [f"{t}_s{l}" for t in TYPES for l in (1, 2, 3)]
apd = {m: {c: A.per_query(A.tag(m), c)[0] for c in ["clean"] + conds} for m in A.MODELS}
strat = {}; taus = []; taus_ex = []
for b in range(B):
    bi = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in cls.values()])
    d_pert = [np.mean([apd[m][c][bi].mean() for c in conds]) for m in A.MODELS]
    k = [kth_ap[m][boots[b]].mean() for m in A.MODELS]; kx = [kth_ex[m][0][boots[b]].mean() for m in A.MODELS]
    taus.append(kendalltau(d_pert, k).statistic); taus_ex.append(kendalltau(d_pert, kx).statistic)
    for m in A.MODELS: strat.setdefault(m, []).append(apd[m]["clean"][bi].mean())
res["b_dtd_stratified_clean_ci"] = {m: A.ci(np.array(v)) for m, v in strat.items()}
d_pert_full = [np.mean([apd[m][c].mean() for c in conds]) for m in A.MODELS]
res["b_tau_primary_cluster_ci"] = A.ci(np.array(taus))
res["c_tau_excl_same_sample"] = {"tau": float(kendalltau(d_pert_full, [kth_ex[m][0].mean() for m in A.MODELS]).statistic), "ci": A.ci(np.array(taus_ex))}
# d) patch grids at reduced resolution
res["d_patch_grid"] = {m: {r: (r // p, r % p) for r in (224, 160, 112)} for m, p in [("CLIP-B/32", 32), ("CLIP-B/16", 16), ("SigLIP-B/16", 16), ("DINOv2-S/14", 14), ("DINOv2-B/14", 14)]}
json.dump(res, open(os.path.join(OUT, "r1_posthoc.json"), "w"), indent=1)
for k, v in res.items(): print(k, json.dumps(v)[:900])
