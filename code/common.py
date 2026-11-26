"""Shared utilities: data loading, perturbations, preprocessing, models, retrieval metrics.
Robustness-efficiency evaluation of frozen image embeddings for texture retrieval on CPU.
"""
import io, os, re, glob, json, time, hashlib, platform
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_raw")
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")
BASE = 256  # every image is first normalised to a 256x256 centre crop

MODELS = {  # name: (hf id, family, embedding)
    "CLIP-B/32": ("openai/clip-vit-base-patch32", "language-supervised", "projected image embedding"),
    "CLIP-B/16": ("openai/clip-vit-base-patch16", "language-supervised", "projected image embedding"),
    "SigLIP-B/16": ("google/siglip-base-patch16-224", "language-supervised", "pooled image embedding"),
    "DINOv2-S/14": ("facebook/dinov2-small", "self-supervised", "CLS (pooler) token"),
    "DINOv2-B/14": ("facebook/dinov2-base", "self-supervised", "CLS (pooler) token"),
    "ResNet-50": ("microsoft/resnet-50", "supervised CNN", "global average pool"),
    "MobileNetV2": ("google/mobilenet_v2_1.0_224", "supervised CNN", "global average pool"),
}
# preprocessing replicating each official HF processor at 224 and scaled to other targets
# (resize-shorter-side ratio, centre crop?, interpolation, mean, std)
PRE = {
    "CLIP-B/32": (1.0, True, "bicubic", (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "CLIP-B/16": (1.0, True, "bicubic", (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "SigLIP-B/16": (None, False, "bicubic", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "DINOv2-S/14": (256 / 224, True, "bicubic", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "DINOv2-B/14": (256 / 224, True, "bicubic", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "ResNet-50": (1 / 0.875, True, "bicubic", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "MobileNetV2": (256 / 224, True, "bilinear", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

# ---------------- data ----------------
def centre_base(img, size=BASE):
    img = img.convert("RGB")
    w, h = img.size; s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size; l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))

def load_dtd():
    """DTD split 1 from the HF mirror jxie/dtd (40/40/40 images per class). gallery = train+val, query = test."""
    import pyarrow.parquet as pq
    d = os.path.join(DATA, "dtd_hf_jxie", "data")
    out = {}
    for split in ["train", "validation", "test"]:
        f = glob.glob(os.path.join(d, f"{split}-*.parquet"))[0]
        t = pq.read_table(f).to_pydict()
        out[split] = ([x["bytes"] for x in t["image"]], np.array(t["label"]))
    g_bytes = out["train"][0] + out["validation"][0]
    g_lab = np.concatenate([out["train"][1], out["validation"][1]])
    q_bytes, q_lab = out["test"]
    return g_bytes, g_lab, q_bytes, q_lab

KTH_POSE = {1: "frontal", 2: "frontal", 3: "frontal", 10: "frontal", 4: "right", 5: "right", 6: "right", 11: "right", 7: "left", 8: "left", 9: "left", 12: "left"}
KTH_ILLUM = {1: "frontal", 4: "frontal", 7: "frontal", 2: "top", 5: "top", 8: "top", 3: "side", 6: "side", 9: "side", 10: "ambient", 11: "ambient", 12: "ambient"}

def load_kth():
    root = os.path.join(DATA, "KTH-TIPS2-b")
    recs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "sample_*", "*.png"))):
        mat = os.path.basename(os.path.dirname(os.path.dirname(p)))
        smp = os.path.basename(os.path.dirname(p))[-1]
        m = re.search(r"scale_(\d+)_im_(\d+)_", os.path.basename(p))
        sc, im = int(m.group(1)), int(m.group(2))
        recs.append({"path": p, "material": mat, "sample": f"{mat}_{smp}", "scale": sc, "im": im,
                     "pose": KTH_POSE[im], "illum": KTH_ILLUM[im]})
    return recs

# ---------------- perturbations (query side only) ----------------
SEV = {
    "low_light": [0.5, 0.3, 0.15],          # multiplicative exposure factor
    "colour_temp": [0.15, 0.30, 0.45],      # R gain 1+a, B gain 1-a
    "rotation": [15, 30, 45],               # degrees, then largest inscribed square
    "perspective": [0.08, 0.16, 0.24],      # keystone: top corners moved inward by fraction
    "partial_crop": [0.5, 0.25, 0.10],      # kept area fraction, seeded location
    "blur": [1.5, 3.0, 6.0],                # Gaussian sigma (px at 256)
    "jpeg": [30, 15, 5],                    # JPEG quality
    "downscale": [2, 4, 8],                 # down/up-sampling factor
}
TYPES = list(SEV.keys())

def seed_for(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

def perturb(img, ptype, level, key):
    """img: 256x256 RGB PIL. level: 1..3. key: stable image id for seeding."""
    v = SEV[ptype][level - 1]
    if ptype == "low_light":
        a = np.asarray(img).astype(np.float32) * v
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if ptype == "colour_temp":
        a = np.asarray(img).astype(np.float32)
        a[..., 0] *= (1 + v); a[..., 2] *= (1 - v)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if ptype == "rotation":
        r = img.rotate(v, resample=Image.BICUBIC, expand=False)
        th = np.deg2rad(v); side = BASE / (abs(np.cos(th)) + abs(np.sin(th)))
        o = (BASE - side) / 2
        return r.crop((o, o, BASE - o, BASE - o)).resize((BASE, BASE), Image.BICUBIC)
    if ptype == "perspective":
        d = v * BASE
        # map output square to a source quad whose top edge is narrower (camera tilt)
        src = [(d, 0), (BASE - d, 0), (BASE, BASE), (0, BASE)]
        dst = [(0, 0), (BASE, 0), (BASE, BASE), (0, BASE)]
        A = []; B = []
        for (x, y), (u, w) in zip(dst, src):
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
            A.append([0, 0, 0, x, y, 1, -w * x, -w * y]); B.append(w)
        coeffs = np.linalg.solve(np.array(A, float), np.array(B, float))
        return img.transform((BASE, BASE), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)
    if ptype == "partial_crop":
        rng = np.random.default_rng(seed_for(key + ptype + str(level)))
        side = int(round(BASE * np.sqrt(v)))
        x = int(rng.integers(0, BASE - side + 1)); y = int(rng.integers(0, BASE - side + 1))
        return img.crop((x, y, x + side, y + side)).resize((BASE, BASE), Image.BICUBIC)
    if ptype == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=v))
    if ptype == "jpeg":
        b = io.BytesIO(); img.save(b, format="JPEG", quality=int(v)); b.seek(0)
        return Image.open(b).convert("RGB")
    if ptype == "downscale":
        s = BASE // int(v)
        return img.resize((s, s), Image.BILINEAR).resize((BASE, BASE), Image.BILINEAR)
    raise ValueError(ptype)

# ---------------- preprocessing (vectorised) ----------------
def to_uint8_tensor(imgs):
    return torch.from_numpy(np.stack([np.asarray(i) for i in imgs])).permute(0, 3, 1, 2).contiguous()

def preprocess(batch_u8, model_name, target=224):
    ratio, crop, interp, mean, std = PRE[model_name]
    x = batch_u8.float() / 255.0
    if ratio is None:  # direct resize (SigLIP)
        x = F.interpolate(x, size=(target, target), mode=interp, align_corners=False, antialias=True)
    else:
        r = int(round(target * ratio))
        x = F.interpolate(x, size=(r, r), mode=interp, align_corners=False, antialias=True)  # inputs are square
        if crop:
            o = (r - target) // 2
            x = x[:, :, o:o + target, o:o + target]
    x = x.clamp(0, 1)
    m = torch.tensor(mean).view(1, 3, 1, 1); s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s

# ---------------- models ----------------
def load_model(name, int8=False):
    from transformers import AutoModel, CLIPModel, SiglipModel
    hf = MODELS[name][0]
    if name.startswith("CLIP"):
        m = CLIPModel.from_pretrained(hf)
    elif name.startswith("SigLIP"):
        m = SiglipModel.from_pretrained(hf)
    else:
        m = AutoModel.from_pretrained(hf)
    m.eval()
    if int8:
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    return m

@torch.inference_mode()
def embed(model, name, x, target=224):
    if name.startswith("CLIP") or name.startswith("SigLIP"):
        kw = {"interpolate_pos_encoding": True} if target != 224 else {}
        out = model.get_image_features(pixel_values=x, **kw)
        if not torch.is_tensor(out):
            out = out.pooler_output if hasattr(out, "pooler_output") else out[0]
    elif name.startswith("DINOv2"):
        out = model(pixel_values=x).pooler_output
    else:
        out = model(pixel_values=x).pooler_output.flatten(1)
    return F.normalize(out.float(), dim=-1)

def lbp_colour_descriptor(batch_u8):
    """Training-free texture baseline: 256-bin LBP(8,1) histogram + 4x4x4 RGB histogram (sqrt, L2 each)."""
    x = batch_u8.float()
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])
    c = g[:, 1:-1, 1:-1]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = torch.zeros_like(c, dtype=torch.int64)
    H, W = g.shape[1], g.shape[2]
    for k, (dy, dx) in enumerate(offs):
        n = g[:, 1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
        code |= ((n >= c).long() << k)
    lbp = torch.stack([torch.bincount(cc.flatten(), minlength=256) for cc in code]).float()
    q = (x / 64).long().clamp(0, 3)
    idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    col = torch.stack([torch.bincount(i.flatten(), minlength=64) for i in idx]).float()
    lbp = F.normalize(lbp.sqrt(), dim=-1); col = F.normalize(col.sqrt(), dim=-1)
    return F.normalize(torch.cat([lbp, col], dim=-1), dim=-1)

# ---------------- metrics ----------------
def ap_per_query(Q, G, q_lab, g_lab, exclude=None):
    """Average precision per query with cosine similarity (inputs L2-normalised). exclude: bool mask (nq, ng) of gallery items to drop."""
    S = Q @ G.T
    if exclude is not None:
        S = np.where(exclude, -np.inf, S)
    order = np.argsort(-S, axis=1, kind="stable")
    rel = (g_lab[order] == q_lab[:, None])
    if exclude is not None:
        valid = ~np.take_along_axis(exclude, order, axis=1)
        rel = rel & valid
    cum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)[None, :]
    nrel = rel.sum(1)
    ap = (rel * (cum / ranks)).sum(1) / np.maximum(nrel, 1)
    r1 = rel[:, 0].astype(float)
    return ap, r1

def env_info():
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "cpu": platform.processor(), "machine": platform.machine(), "os": platform.platform(),
            "torch_threads": torch.get_num_threads()}

def log_jsonl(name, rec):
    os.makedirs(LOGS, exist_ok=True)
    rec = dict(rec); rec["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(LOGS, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
