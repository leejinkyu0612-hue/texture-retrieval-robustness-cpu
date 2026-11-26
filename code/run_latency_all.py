import subprocess, sys, os
cfg = []
M = ["CLIP-B/32","CLIP-B/16","SigLIP-B/16","DINOv2-S/14","DINOv2-B/14","ResNet-50","MobileNetV2"]
for m in M: cfg.append((m,"fp32",224,4))
for m in M[:5]: cfg.append((m,"int8",224,4))
for m in M: cfg += [(m,"fp32",160,4),(m,"fp32",112,4),(m,"fp32",224,14)]
if len(sys.argv) > 1: cfg = [tuple(x.split("@")[:2]) + (int(x.split("@")[2]), int(x.split("@")[3])) for x in sys.argv[1:]]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH="", PYTHONHOME="", HF_HUB_OFFLINE="1")
for c in cfg:
    r = subprocess.run([sys.executable, "-W", "ignore", "code/run_latency.py", c[0], c[1], str(c[2]), str(c[3])], env=env, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    print(c, line[-1][:200] if line else r.stderr[-300:], flush=True)


