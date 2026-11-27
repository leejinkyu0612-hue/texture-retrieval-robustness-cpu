import json
r=json.load(open("results/r2_posthoc.json")); ls=json.load(open("results/latency_sessions.json"))
pts=r["E3_pareto_sev2"]
for p in pts:
    v,res=p["variant"].split("-"); p["latency_ms_sessions_median"]=ls[f"{p['model']}|{v}|{res}|4"]["median_of_sessions"]
for p in pts:
    p["pareto_sessions"]=not any(q is not p and q["latency_ms_sessions_median"]<=p["latency_ms_sessions_median"] and q["shifted_mAP_sev2"]>=p["shifted_mAP_sev2"] and (q["latency_ms_sessions_median"]<p["latency_ms_sessions_median"] or q["shifted_mAP_sev2"]>p["shifted_mAP_sev2"]) for q in pts)
r["E3_pareto_sev2"]=pts; json.dump(r,open("results/r2_posthoc.json","w"),indent=1)
for p in sorted(pts,key=lambda x:x["latency_ms_sessions_median"]): print(p["model"],p["variant"],round(p["latency_ms_sessions_median"],1),round(p["shifted_mAP_sev2"],3),"P" if p["pareto_sessions"] else "")
