import subprocess, sys, os
e = sys.argv[1]
jobs = ["LBP+colour@none@256@full","CLIP-B/32@fp32@224@full","MobileNetV2@fp32@224@full","DINOv2-S/14@fp32@224@full","ResNet-50@fp32@224@full","CLIP-B/16@fp32@224@full","SigLIP-B/16@fp32@224@full","DINOv2-B/14@fp32@224@full","LBP+colour@none@256@kth","CLIP-B/32@fp32@224@kth","MobileNetV2@fp32@224@kth","DINOv2-S/14@fp32@224@kth","ResNet-50@fp32@224@kth","CLIP-B/16@fp32@224@kth","SigLIP-B/16@fp32@224@kth","DINOv2-B/14@fp32@224@kth"]
jobs = sys.argv[3:] or jobs
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH="", PYTHONHOME="", HF_HUB_OFFLINE="1")
log = open(os.path.join(e, "logs", sys.argv[2]), "ab")
flags = 0x00000008 | 0x01000000 | 0x00000200  # DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB | CREATE_NEW_PROCESS_GROUP
p = subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(e, "code", "run_extract.py")] + jobs, cwd=e, env=env, stdout=log, stderr=log, creationflags=flags)
print("pid", p.pid)
