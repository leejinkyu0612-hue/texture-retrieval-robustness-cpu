import sys, json, numpy as np
sys.path.insert(0,"code"); import analyze as A
from scipy.stats import kendalltau
r2=json.load(open("results/rq2.json")); ms=r2["models"]; krr=r2["kth_RR_scale"]
out={}
for t in ("downscale","partial_crop","blur"):
    rr=[np.mean([A.per_query(A.tag(m),f"{t}_s{l}")[0].mean() for l in (1,2,3)])/A.per_query(A.tag(m),"clean")[0].mean() for m in ms]
    out[f"tau_{t}RR_vs_KTHscaleRR"]=float(kendalltau(rr,krr).statistic)
conds=[f"{t}_s2" for t in A.TYPES]; rd={}
for m in ["CLIP-B/32","CLIP-B/16","SigLIP-B/16","DINOv2-S/14","DINOv2-B/14"]:
    b={c:A.per_query(A.tag(m),c)[0] for c in ["clean"]+conds}; v={c:A.per_query(A.tag(m,"int8",224),c)[0] for c in ["clean"]+conds}
    def rr(d,idx): return np.mean([d[c][idx].mean() for c in conds])/d["clean"][idx].mean()
    full=np.arange(len(A.q_lab)); diff=rr(v,full)-rr(b,full)
    bs=np.array([rr(v,i)-rr(b,i) for i in A.BOOT[:1000]])
    rd[m]={"RR_diff_int8_minus_fp32":float(diff),"ci":A.ci(bs)}
out["RR_diff_int8_ci_exploratory"]=rd
json.dump(out,open("results/r3_extra.json","w"),indent=1); print(json.dumps(out,indent=1))
