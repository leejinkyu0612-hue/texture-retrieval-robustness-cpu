"""CPU efficiency: batch-1 latency (median, p90), batch-32 throughput, memory, parameters. One model/variant per process.
Usage: python run_latency.py <model> <variant fp32|int8> <res> <threads>"""
import os, sys, time, json, statistics
import numpy as np, torch, psutil
sys.path.insert(0, os.path.dirname(__file__))
from common import *

name, variant, res, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
from transformers import CLIPVisionModelWithProjection, SiglipVisionModel
def load_vision(name, int8=False):
    """Vision tower only (the text tower is never used for image retrieval)."""
    hf = MODELS[name][0]
    if name.startswith("CLIP"): m = CLIPVisionModelWithProjection.from_pretrained(hf)
    elif name.startswith("SigLIP"): m = SiglipVisionModel.from_pretrained(hf)
    else: return load_model(name, int8)
    m.eval()
    if int8: m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    return m
@torch.inference_mode()
def embed_v(model, name, x, target):
    kw = {"interpolate_pos_encoding": True} if target != 224 else {}
    if name.startswith("CLIP"): return torch.nn.functional.normalize(model(pixel_values=x, **kw).image_embeds, dim=-1)
    if name.startswith("SigLIP"): return torch.nn.functional.normalize(model(pixel_values=x, **kw).pooler_output, dim=-1)
    return embed(model, name, x, target)
torch.set_num_threads(threads)
proc = psutil.Process()
rss0 = proc.memory_info().rss
import io as _io
params = sum(p.numel() for p in load_vision(name).parameters())
import gc; gc.collect(); rss0 = proc.memory_info().rss
model = load_vision(name, int8=(variant == "int8"))
rss_loaded = proc.memory_info().rss
_b = _io.BytesIO(); torch.save(model.state_dict(), _b); size_mb = _b.tell() / 2**20
rng = np.random.default_rng(0)
u8 = torch.from_numpy(rng.integers(0, 256, size=(32, 3, 256, 256), dtype=np.uint8))
load_cpu = psutil.cpu_percent(interval=1.0)
x1 = preprocess(u8[:1], name, res)
for _ in range(20): embed_v(model, name, x1, res)
lat = []
for _ in range(100):
    t0 = time.perf_counter(); embed_v(model, name, preprocess(u8[:1], name, res), res); lat.append((time.perf_counter() - t0) * 1000)
xb = preprocess(u8, name, res)
for _ in range(2): embed_v(model, name, xb, res)
tb = []
for _ in range(5):
    t0 = time.perf_counter(); e = embed_v(model, name, xb, res); tb.append(time.perf_counter() - t0)
peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
rec = {"model": name, "variant": variant, "resolution": res, "threads": threads, "params_M": round(params / 1e6, 2),
       "state_dict_MB": round(size_mb, 1), "latency_ms_median": round(statistics.median(lat), 2),
       "latency_ms_p90": round(float(np.percentile(lat, 90)), 2), "throughput_img_s_b32": round(32 / statistics.median(tb), 1),
       "rss_loaded_MB": round((rss_loaded - rss0) / 2**20, 1), "peak_wset_MB": round(peak / 2**20, 1),
       "emb_dim": int(e.shape[1]), "emb_bytes_fp32": int(e.shape[1] * 4), "cpu_load_before_pct": load_cpu, "vision_tower_only": True, "warmup": 20, "timed_runs_b1": 100, "timed_runs_b32": 5}
rec.update(env_info()); log_jsonl("latency.jsonl", rec); print(json.dumps(rec))



