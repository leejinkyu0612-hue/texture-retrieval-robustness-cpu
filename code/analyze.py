"""Analysis following PLAN_pre-specified_analysis.md. Reads runs/*.npy, writes results/*.json."""
import os, sys, json, glob, itertools
import numpy as np
from scipy.stats import wilcoxon, kendalltau
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, RUNS, TYPES, ap_per_query

B = 2000; RNG = np.random.default_rng(2027)
OUT = os.path.join(ROOT, "results")
lab = json.load(open(os.path.join(OUT, "dtd_labels.json")))
g_lab = np.array(lab["gallery_labels"]); q_lab = np.array(lab["query_labels"])
NQ = len(q_lab)
BOOT = RNG.integers(0, NQ, size=(B, NQ))  # same resamples for every paired comparison
MODELS = ["CLIP-B/32", "CLIP-B/16", "SigLIP-B/16", "DINOv2-S/14", "DINOv2-B/14", "ResNet-50", "MobileNetV2", "LBP+colour"]

def tag(m, v="fp32", r=224):
    return "LBP+colour__none__256" if m == "LBP+colour" else f"{m.replace('/', '-')}__{v}__{r}"

def load(t, cond, ds="dtd"):
    p = os.path.join(RUNS, ds, t, cond + ".npy")
    return np.load(p) if os.path.exists(p) else None

def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

def boot_mean(ap):
    return ci(ap[BOOT].mean(1))

def holm(ps):
    ps = np.asarray(ps, float); n = len(ps); order = np.argsort(ps); adj = np.empty(n); run = 0
    for k, i in enumerate(order):
        run = max(run, (n - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()

def int8_quant(G, Q):
    lo, hi = G.min(0), G.max(0); sc = (hi - lo) / 255.0 + 1e-12
    qg = np.round((G - lo) / sc).clip(0, 255) * sc + lo
    qq = np.round((Q - lo) / sc).clip(0, 255) * sc + lo
    n = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    return n(qg), n(qq)

def binarise(G, Q):
    return np.sign(G) / np.sqrt(G.shape[1]), np.sign(Q) / np.sqrt(Q.shape[1])

def per_query(t, cond, comp=None):
    G = load(t, "gallery"); Q = load(t, cond)
    if G is None or Q is None: return None
    if comp == "int8emb": G, Q = int8_quant(G, Q)
    if comp == "bin": G, Q = binarise(G, Q)
    ap, r1 = ap_per_query(Q, G, q_lab, g_lab)
    return ap, r1

def summary_dtd():
    res = {}
    conds = ["clean"] + [f"{t}_s{l}" for t in TYPES for l in (1, 2, 3)]
    for m in MODELS:
        t = tag(m); clean = per_query(t, "clean")
        if clean is None: continue
        apc, r1c = clean; R = {"clean": {"mAP": float(apc.mean()), "mAP_ci": boot_mean(apc), "R1": float(r1c.mean())}}
        pvals = []; keys = []
        for c in conds[1:]:
            pq = per_query(t, c)
            if pq is None: continue
            ap, r1 = pq; d = apc - ap
            try: p = float(wilcoxon(apc, ap).pvalue)
            except ValueError: p = 1.0
            R[c] = {"mAP": float(ap.mean()), "mAP_ci": boot_mean(ap), "R1": float(r1.mean()), "drop": float(d.mean()),
                    "drop_ci": ci(d[BOOT].mean(1)), "RR": float(ap.mean() / apc.mean()), "p_wilcoxon": p}
            if c.endswith("_s3"): pvals.append(p); keys.append(c)
        for k, a in zip(keys, holm(pvals)): R[k]["p_holm_s3"] = a
        res[m] = R
    json.dump(res, open(os.path.join(OUT, "dtd_main.json"), "w"), indent=1)
    return res

def hypotheses_rq1(res):
    out = {}
    # H1a: every type at severity 3 reduces mAP (CI of drop excludes 0) for every model
    h1a = {m: {t: (R.get(f"{t}_s3", {}).get("drop_ci", [0, 0])[0] > 0) for t in TYPES} for m, R in res.items()}
    out["H1a_all_true"] = all(all(v.values()) for v in h1a.values()); out["H1a_detail"] = h1a
    # H1b: monotone non-decreasing drop with severity (allowing CI overlap)
    mono = []; det = {}
    for m, R in res.items():
        for t in TYPES:
            d = [R.get(f"{t}_s{l}") for l in (1, 2, 3)]
            if None in d: continue
            ok = all(d[i]["drop"] <= d[i + 1]["drop"] or d[i]["drop_ci"][0] <= d[i + 1]["drop_ci"][1] for i in (0, 1))
            mono.append(ok); det[f"{m}|{t}"] = [round(x["drop"], 4) for x in d]
    out["H1b_share_monotone"] = float(np.mean(mono)) if mono else None; out["H1b_detail"] = det
    # H1c: colour-temp s3 drop smaller than blur, downscale, partial_crop s3 (mean over models)
    mean_drop = {t: float(np.mean([R[f"{t}_s3"]["drop"] for R in res.values() if f"{t}_s3" in R])) for t in TYPES}
    out["mean_drop_s3_over_models"] = mean_drop
    out["H1c"] = all(mean_drop["colour_temp"] < mean_drop[x] for x in ("blur", "downscale", "partial_crop"))
    json.dump(out, open(os.path.join(OUT, "rq1_hypotheses.json"), "w"), indent=1)
    return out

KTH_REF = (6, "frontal", "frontal")
def kth_eval(ref_illum="frontal"):
    meta = json.load(open(os.path.join(OUT, "kth_meta.json")))
    sc = np.array([r["scale"] for r in meta]); po = np.array([r["pose"] for r in meta]); il = np.array([r["illum"] for r in meta])
    mat = np.array([r["material"] for r in meta]); smp = np.array([r["sample"] for r in meta])
    ref = (sc == 6) & (po == "frontal") & (il == ref_illum)
    gi = np.where(ref)[0]; qi = np.where(~ref)[0]
    grp = np.full(len(meta), "mixed", dtype=object)
    grp[(sc == 6) & (po == "frontal") & (il != ref_illum)] = "illumination_only"
    grp[(sc == 6) & (il == ref_illum) & (po != "frontal")] = "pose_only"
    grp[(po == "frontal") & (il == ref_illum) & (sc != 6)] = "scale_only"
    res = {}
    for m in MODELS:
        E = load(tag(m), "all", "kth")
        if E is None: continue
        G, Q = E[gi], E[qi]
        ap, _ = ap_per_query(Q, G, mat[qi], mat[gi])
        S = Q @ G.T; inst_r1 = (smp[gi][S.argmax(1)] == smp[qi]).astype(float)
        R = {"n_gallery": int(len(gi)), "n_query": int(len(qi)), "cat_mAP": float(ap.mean()), "inst_R1": float(inst_r1.mean())}
        bq = RNG.integers(0, len(qi), size=(B, len(qi)))
        R["cat_mAP_ci"] = ci(ap[bq].mean(1)); R["inst_R1_ci"] = ci(inst_r1[bq].mean(1))
        for gname in ("illumination_only", "pose_only", "scale_only", "mixed"):
            msk = grp[qi] == gname
            R[gname] = {"n": int(msk.sum()), "cat_mAP": float(ap[msk].mean()), "inst_R1": float(inst_r1[msk].mean())}
        for dsc in range(0, 5):
            msk = np.abs(sc[qi] - 6) == dsc
            R[f"scale_gap_{dsc}"] = {"n": int(msk.sum()), "cat_mAP": float(ap[msk].mean()), "inst_R1": float(inst_r1[msk].mean())}
        near = np.abs(sc[qi] - 6) == 0; far = np.abs(sc[qi] - 6) >= 2
        R["RR_scale"] = float(ap[far].mean() / ap[near].mean())
        R["_ap"] = ap.tolist(); R["_r1"] = inst_r1.tolist()
        res[m] = R
    return res, qi

def rq2(dtd, kth):
    ms = [m for m in MODELS if m in dtd and m in kth]
    conds = [f"{t}_s{l}" for t in TYPES for l in (1, 2, 3)]
    dtd_pert = [np.mean([dtd[m][c]["mAP"] for c in conds]) for m in ms]
    dtd_rr = [np.mean([dtd[m][c]["RR"] for c in conds]) for m in ms]
    dtd_clean = [dtd[m]["clean"]["mAP"] for m in ms]
    kth_map = [kth[m]["cat_mAP"] for m in ms]; kth_rr = [kth[m]["RR_scale"] for m in ms]
    out = {"models": ms, "dtd_mean_perturbed_mAP": dtd_pert, "kth_cat_mAP": kth_map, "dtd_mean_RR": dtd_rr, "kth_RR_scale": kth_rr,
           "tau_primary": float(kendalltau(dtd_pert, kth_map).statistic), "tau_secondary_RR": float(kendalltau(dtd_rr, kth_rr).statistic),
           "tau_clean_vs_perturbed_dtd": float(kendalltau(dtd_clean, dtd_pert).statistic)}
    # bootstrap over queries in both datasets
    apd = {m: {c: per_query(tag(m), c)[0] for c in conds} for m in ms}
    kap = {m: np.array(kth[m]["_ap"]) for m in ms}
    nk = len(next(iter(kap.values())))
    taus = []; taus2 = []
    for b in range(500):
        bi = RNG.integers(0, NQ, NQ); bk = RNG.integers(0, nk, nk)
        a = [np.mean([apd[m][c][bi].mean() for c in conds]) for m in ms]
        k = [kap[m][bk].mean() for m in ms]
        taus.append(kendalltau(a, k).statistic)
    out["tau_primary_ci"] = ci(np.array(taus)); out["n_boot_tau"] = 500
    # secondary tau (RR) CI: added after the plan (exploratory uncertainty estimate)
    meta = json.load(open(os.path.join(OUT, "kth_meta.json"))); sc = np.array([r["scale"] for r in meta])
    po = np.array([r["pose"] for r in meta]); il = np.array([r["illum"] for r in meta])
    qi = np.where(~((sc == 6) & (po == "frontal") & (il == "frontal")))[0]; gap = np.abs(sc[qi] - 6)
    clean = {m: per_query(tag(m), "clean")[0] for m in ms}
    for b in range(500):
        bi = RNG.integers(0, NQ, NQ); bk = RNG.integers(0, nk, nk)
        a = [np.mean([apd[m][c][bi].mean() / clean[m][bi].mean() for c in conds]) for m in ms]
        g = gap[bk]; k = [kap[m][bk][g >= 2].mean() / kap[m][bk][g == 0].mean() for m in ms]
        taus2.append(kendalltau(a, k).statistic)
    out["tau_secondary_RR_ci_exploratory"] = ci(np.array(taus2))
    json.dump(out, open(os.path.join(OUT, "rq2.json"), "w"), indent=1)
    return out

def rq3(dtd):
    out = {}
    conds = ["clean"] + [f"{t}_s2" for t in TYPES]
    for m in MODELS[:-1]:
        base = {c: per_query(tag(m), c) for c in conds}
        if any(v is None for v in base.values()): continue
        R = {}
        variants = [("int8", 224, None), ("fp32", 160, None), ("fp32", 112, None), ("fp32", 224, "int8emb"), ("fp32", 224, "bin")]
        for v, r, comp in variants:
            t = tag(m, v, r)
            pv = {c: per_query(t, c, comp) for c in conds}
            if any(x is None for x in pv.values()): continue
            key = (f"{v}_{r}" if comp is None else comp)
            dclean = pv["clean"][0] - base["clean"][0]
            drops_base = np.mean([base["clean"][0] - base[c][0] for c in conds[1:]], axis=0)
            drops_var = np.mean([pv["clean"][0] - pv[c][0] for c in conds[1:]], axis=0)
            dd = drops_var - drops_base
            per_type = {}
            for c in conds[1:]:
                ddt = (pv["clean"][0] - pv[c][0]) - (base["clean"][0] - base[c][0])
                per_type[c] = {"DD": float(ddt.mean()), "DD_ci": ci(ddt[BOOT].mean(1)), "mAP_var": float(pv[c][0].mean()), "mAP_base": float(base[c][0].mean())}
            try: p_dd = float(wilcoxon(drops_var, drops_base).pvalue)
            except ValueError: p_dd = 1.0
            R[key] = {"clean_mAP": float(pv["clean"][0].mean()), "delta_clean": float(dclean.mean()), "delta_clean_ci": ci(dclean[BOOT].mean(1)),
                      "mean_drop_var": float(drops_var.mean()), "mean_drop_base": float(drops_base.mean()),
                      "DD_mean": float(dd.mean()), "DD_ci": ci(dd[BOOT].mean(1)), "p_wilcoxon_DD": p_dd, "per_type": per_type}
        out[m] = R
    json.dump(out, open(os.path.join(OUT, "rq3_variants.json"), "w"), indent=1)
    return out

if __name__ == "__main__":
    what = sys.argv[1:] or ["dtd", "kth", "rq2", "rq3"]
    dtd = summary_dtd(); h = hypotheses_rq1(dtd)
    print("RQ1 H1a", h["H1a_all_true"], "H1b", h["H1b_share_monotone"], "H1c", h["H1c"])
    for m, R in dtd.items(): print(m, "clean mAP %.3f" % R["clean"]["mAP"], "n_conds", len(R) - 1)
    if "kth" in what or "rq2" in what:
        kth, _ = kth_eval(); kth_alt, _ = kth_eval("ambient")
        strip = lambda d: {m: {k: v for k, v in R.items() if not k.startswith("_")} for m, R in d.items()}
        json.dump({"reference_frontal": strip(kth), "reference_ambient": strip(kth_alt)}, open(os.path.join(OUT, "kth_main.json"), "w"), indent=1)
        if kth and "rq2" in what: print("RQ2", {k: v for k, v in rq2(dtd, kth).items() if k.startswith("tau")})
    if "rq3" in what:
        r3 = rq3(dtd); print("RQ3 models", list(r3))

