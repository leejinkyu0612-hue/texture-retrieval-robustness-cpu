"""Main extraction: embeddings for DTD (clean + perturbed queries) and KTH-TIPS2-b, per model/variant/resolution.
Usage: python run_extract.py <job> [<job> ...]   job = model@variant@res@scope  (scope: full = clean+24 perturbations, sev2 = clean+8 at severity 2, kth)
Each (job, condition) writes runs/<dataset>/<tag>/<condition>.npy and one JSON line in logs/runs.jsonl. Existing files are skipped (resume)."""
import io, os, sys, time, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import *

torch.manual_seed(2027)
BS = 32

def dtd_queries(q_lab):
    rng = np.random.default_rng(2027)
    idx = []
    for c in sorted(set(q_lab.tolist())):
        ids = np.where(q_lab == c)[0]
        idx.extend(sorted(rng.choice(ids, size=20, replace=False).tolist()))
    return np.array(idx)

def run_embed(fn_imgs, n, model, name, res, out_path, meta):
    if os.path.exists(out_path):
        return
    t0 = time.time(); embs = []
    for s in range(0, n, BS):
        imgs = fn_imgs(s, min(n, s + BS))
        u8 = to_uint8_tensor(imgs)
        if name == "LBP+colour":
            e = lbp_colour_descriptor(u8)
        else:
            e = embed(model, name, preprocess(u8, name, res), res)
        embs.append(e.numpy().astype(np.float32))
    E = np.concatenate(embs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, E)
    rec = dict(meta); rec.update({"n_images": int(n), "dim": int(E.shape[1]), "wall_s": round(time.time() - t0, 2),
                                  "img_per_s": round(n / (time.time() - t0), 2), "out": os.path.relpath(out_path, ROOT).replace("\\", "/")})
    rec.update(env_info()); log_jsonl("runs.jsonl", rec)
    print(json.dumps({k: rec[k] for k in ("tag", "condition", "n_images", "img_per_s")}), flush=True)

def main(jobs):
    g_bytes, g_lab, q_bytes, q_lab = load_dtd()
    qi = dtd_queries(q_lab)
    np.save(os.path.join(ROOT, "results", "dtd_query_index.npy"), qi)
    json.dump({"gallery_labels": g_lab.tolist(), "query_index": qi.tolist(), "query_labels": q_lab[qi].tolist()},
              open(os.path.join(ROOT, "results", "dtd_labels.json"), "w"))
    q_sel = [q_bytes[i] for i in qi]
    cdir = os.path.join(RUNS, "cache"); os.makedirs(cdir, exist_ok=True)
    def cached(fn, items):
        p = os.path.join(cdir, fn)
        if not os.path.exists(p):
            np.save(p, np.stack([np.asarray(centre_base(Image.open(io.BytesIO(b)))) for b in items]))
        return np.load(p, mmap_mode="r")
    gbase = cached("dtd_gallery_base256.npy", g_bytes); qbase = cached("dtd_query_base256.npy", q_sel)
    kth = load_kth()
    json.dump(kth, open(os.path.join(ROOT, "results", "kth_meta.json"), "w"))
    for job in jobs:
        name, variant, res, scope = job.split("@"); res = int(res)
        tag = f"{name.replace('/', '-')}__{variant}__{res}"
        model = None if name == "LBP+colour" else load_model(name, int8=(variant == "int8"))
        meta = {"tag": tag, "model": name, "hf_id": MODELS.get(name, ("training-free",))[0], "variant": variant, "resolution": res, "scope": scope}
        if scope in ("full", "sev2"):
            d = os.path.join(RUNS, "dtd", tag)
            gfn = lambda s, e: [Image.fromarray(np.array(gbase[k])) for k in range(s, e)]
            run_embed(gfn, len(g_bytes), model, name, res, os.path.join(d, "gallery.npy"), dict(meta, dataset="DTD", condition="gallery"))
            conds = [("clean", None, 0)] + [(f"{t}_s{l}", t, l) for t in TYPES for l in ((1, 2, 3) if scope == "full" else (2,))]
            for cname, t, l in conds:
                def qfn(s, e, t=t, l=l):
                    out = []
                    for k in range(s, e):
                        im = Image.fromarray(np.array(qbase[k]))
                        out.append(im if t is None else perturb(im, t, l, f"dtd_q{int(qi[k])}"))
                    return out
                run_embed(qfn, len(q_sel), model, name, res, os.path.join(d, f"{cname}.npy"), dict(meta, dataset="DTD", condition=cname))
        if scope == "kth":
            d = os.path.join(RUNS, "kth", tag)
            kfn = lambda s, e: [centre_base(Image.open(r["path"])) for r in kth[s:e]]
            run_embed(kfn, len(kth), model, name, res, os.path.join(d, "all.npy"), dict(meta, dataset="KTH-TIPS2-b", condition="all"))
        del model

if __name__ == "__main__":
    main(sys.argv[1:])


