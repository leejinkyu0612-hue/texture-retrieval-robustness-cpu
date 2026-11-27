import subprocess, sys, os, json, random
e = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M7 = ["CLIP-B/32","CLIP-B/16","SigLIP-B/16","DINOv2-S/14","DINOv2-B/14","ResNet-50","MobileNetV2"]
done = {(d.get("session"), d["model"], d["resolution"]) for d in map(json.loads, open(os.path.join(e, "logs", "latency.jsonl"), encoding="utf-8")) if "session" in d}
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH="", PYTHONHOME="", HF_HUB_OFFLINE="1")
for s in ("1", "2", "3"):
    order = [(m, r) for m in M7 for r in (160, 112)]; random.Random(200 + int(s)).shuffle(order)
    for m, r in order:
        if (s, m, r) in done: continue
        env["LAT_SESSION"] = s
        subprocess.run([sys.executable, "-W", "ignore", os.path.join(e, "code", "run_latency.py"), m, "fp32", str(r), "4"], env=env, capture_output=True)
        print(s, m, r, flush=True)
print("DONE", flush=True)
