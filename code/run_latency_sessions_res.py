"""Repeated latency sessions (post hoc, review round 2): 3 sessions, randomised order, 4 threads, FP32-224 x7 and INT8-224 x5."""
import subprocess, sys, os, random, json
e = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg0 = [(m, "fp32", 224, 4) for m in ["CLIP-B/32","CLIP-B/16","SigLIP-B/16","DINOv2-S/14","DINOv2-B/14","ResNet-50","MobileNetV2"]] + \
      [(m, "int8", 224, 4) for m in ["CLIP-B/32","CLIP-B/16","SigLIP-B/16","DINOv2-S/14","DINOv2-B/14"]]
M7 = ["CLIP-B/32","CLIP-B/16","SigLIP-B/16","DINOv2-S/14","DINOv2-B/14","ResNet-50","MobileNetV2"]
cfg = cfg0 if os.environ.get("LAT_SET", "main") == "main" else [(m, "fp32", r, 4) for m in M7 for r in (160, 112)]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH="", PYTHONHOME="", HF_HUB_OFFLINE="1")
for s in range(1, 4):
    order = cfg[:]; random.Random(100 + s).shuffle(order)
    for c in order:
        env["LAT_SESSION"] = str(s)
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(e, "code", "run_latency.py"), c[0], c[1], str(c[2]), str(c[3])], env=env, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        print(s, c, [l for l in r.stdout.splitlines() if l.startswith("{")][-1][:120] if "{" in r.stdout else r.stderr[-200:], flush=True)

