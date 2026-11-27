import json, numpy as np, collections
L=[json.loads(l) for l in open("logs/latency.jsonl",encoding="utf-8")]
S=collections.defaultdict(dict); S0={}
for d in L:
    k=(d["model"],d["variant"],d["resolution"],d["threads"])
    if "session" in d: S[k][d["session"]]=d
    else: S0[k]=d
out={}
for k,v in S.items():
    med=[v[s]["latency_ms_median"] for s in sorted(v)]
    out["|".join(map(str,k))]={"n_sessions":len(v),"session_medians":med,"median_of_sessions":float(np.median(med)),"range":[min(med),max(med)],
       "p90_median":float(np.median([v[s]["latency_ms_p90"] for s in v])),"throughput_b32_median":float(np.median([v[s]["throughput_img_s_b32"] for s in v])),
       "state_dict_MB":v[sorted(v)[0]]["state_dict_MB"],"original_single_session":S0.get(k,{}).get("latency_ms_median"),"bg_load":[v[s]["cpu_load_before_pct"] for s in sorted(v)]}
json.dump(out,open("results/latency_sessions.json","w"),indent=1)
for k,v in sorted(out.items()): print(k.ljust(26), v["n_sessions"], [round(x,1) for x in v["session_medians"]], "MED", round(v["median_of_sessions"],1), "orig", v["original_single_session"], "thr", v["throughput_b32_median"], "sd", v["state_dict_MB"])
