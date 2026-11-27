import subprocess, sys, os
e = sys.argv[1]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH="", PYTHONHOME="", HF_HUB_OFFLINE="1")
log = open(os.path.join(e, "logs", "latency_sessions.out"), "ab")
p = subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(e, "code", "run_latency_sessions.py")], cwd=e, env=env, stdout=log, stderr=log, creationflags=0x00000008 | 0x01000000 | 0x00000200)
print("pid", p.pid)
