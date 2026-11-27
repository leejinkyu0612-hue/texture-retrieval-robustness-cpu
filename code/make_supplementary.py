"""Generate Supplementary material S1-S7 (Markdown) from logged results. All numbers are read from results/*.json and logs/*.jsonl."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, TYPES, SEV, MODELS, PRE
R = os.path.join(ROOT, "results"); S = os.path.join(ROOT, "supplementary"); os.makedirs(S, exist_ok=True)
J = lambda f: json.load(open(os.path.join(R, f), encoding="utf-8"))
dtd, kth, rq2, rq3, dec, r1, r2, r3b, r3x, ls = J("dtd_main.json"), J("kth_main.json"), J("rq2.json"), J("rq3_variants.json"), J("rq3_decisions_and_exploratory.json"), J("r1_posthoc.json"), J("r2_posthoc.json"), J("r3_int8_batch1.json"), J("r3_extra.json"), J("latency_sessions.json")
ORDER = ["CLIP-B/32", "CLIP-B/16", "SigLIP-B/16", "DINOv2-S/14", "DINOv2-B/14", "ResNet-50", "MobileNetV2", "LBP+colour"]
f3 = lambda x: f"{x:.3f}"
def w(name, txt): open(os.path.join(S, name), "w", encoding="utf-8").write(txt); print("wrote", name)

# S1
t = ["# S1. Query perturbation operators\n", "Every image is first resized (shorter side 256, bicubic) and center-cropped to 256x256. Perturbations act on the query base image only; the gallery stays clean. Implementation: `code/common.py::perturb` (PIL/NumPy, sRGB 8-bit values, clipping to [0,255]). Random elements (partial-view crop location) are seeded by the image identifier.\n",
     "| Shift | s1 | s2 | s3 | Operation |", "|---|---|---|---|---|"]
ops = {"low_light": "multiply sRGB pixel values by factor, clip", "colour_temp": "R x (1+a), B x (1-a), clip", "rotation": "rotate (bicubic) then crop largest inscribed square, resize to 256",
       "perspective": "keystone: top corners moved inward by fraction of width (PIL perspective, bicubic)", "partial_crop": "seeded random square crop keeping the area fraction, resize to 256 (bicubic)",
       "blur": "PIL GaussianBlur radius = sigma", "jpeg": "PIL JPEG encode/decode at quality q", "downscale": "bilinear down by factor, bilinear up to 256"}
for k in TYPES: t.append(f"| {k} | {SEV[k][0]} | {SEV[k][1]} | {SEV[k][2]} | {ops[k]} |")
t.append("\nModel input preprocessing replicates each official Hugging Face processor (resize ratio, center crop, interpolation, mean/std):\n")
t += ["| Model | HF id | resize ratio | center crop | interpolation |", "|---|---|---|---|---|"] + [f"| {m} | {MODELS[m][0]} | {PRE[m][0]} | {PRE[m][1]} | {PRE[m][2]} |" for m in MODELS]
t.append("\nExample images: Fig. 1 of the paper (`results/figs`).")
w("S1_operators.md", "\n".join(t) + "\n")
# S2
t = ["# S2. Extended retrieval results\n", "## DTD mAP (95% query-level bootstrap CI) and R@1 for every condition\n"]
conds = ["clean"] + [f"{a}_s{l}" for a in TYPES for l in (1, 2, 3)]
t += ["| Condition | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
for c in conds: t.append(f"| {c} | " + " | ".join(f"{dtd[m][c]['mAP']:.3f} [{dtd[m][c]['mAP_ci'][0]:.3f}, {dtd[m][c]['mAP_ci'][1]:.3f}]" for m in ORDER) + " |")
t += ["\n## DTD Recall@1\n", "| Condition | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
for c in conds: t.append(f"| {c} | " + " | ".join(f"{dtd[m][c]['R1']:.3f}" for m in ORDER) + " |")
for ref in ("reference_frontal", "reference_ambient"):
    K = kth[ref]
    t += [f"\n## KTH-TIPS2-b ({ref.replace('_', ' ')}): material mAP / instance R@1 by query group\n", "| Model | all | illumination only | pose only | scale only | mixed | scale gap 0..4 (instance R@1) |", "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        k = K[m]; t.append(f"| {m} | {k['cat_mAP']:.3f} / {k['inst_R1']:.3f} | " + " | ".join(f"{k[g]['cat_mAP']:.3f} / {k[g]['inst_R1']:.3f}" for g in ("illumination_only", "pose_only", "scale_only", "mixed")) + " | " + ", ".join(f"{k[f'scale_gap_{i}']['inst_R1']:.2f}" for i in range(5)) + " |")
cb = r1["b_kth_cluster_bootstrap"]
t += ["\n## KTH-TIPS2-b sample-cluster bootstrap (post hoc) and same-sample exclusion (post hoc)\n", "| Model | material mAP [cluster CI] | instance R@1 [cluster CI] | material mAP, own sample excluded |", "|---|---|---|---|"]
for m in ORDER: k = cb[m]; t.append(f"| {m} | {k['cat_mAP']:.3f} [{k['cat_mAP_cluster_ci'][0]:.3f}, {k['cat_mAP_cluster_ci'][1]:.3f}] | {k['inst_R1']:.3f} [{k['inst_R1_cluster_ci'][0]:.3f}, {k['inst_R1_cluster_ci'][1]:.3f}] | {k['cat_mAP_excl_same_sample']:.3f} |")
t.append(f"\nKendall tau (DTD mean shifted mAP vs KTH material mAP) = {rq2['tau_primary']:.2f}; query-level CI {rq2['tau_primary_ci']}; cluster CI {r1['b_tau_primary_cluster_ci']}; own-sample excluded tau = {r1['c_tau_excl_same_sample']['tau']:.2f} {r1['c_tau_excl_same_sample']['ci']}. Relative-robustness tau = {rq2['tau_secondary_RR']:.2f}, exploratory CI {rq2['tau_secondary_RR_ci_exploratory']}; per-type exploratory taus: {json.dumps({k: round(v, 2) for k, v in r3x.items() if k.startswith('tau')})}.")
w("S2_extended_results.md", "\n".join(t) + "\n")
# S3
t = ["# S3. Efficiency checks\n", "## INT8 dynamic quantization (torch.ao.quantization.quantize_dynamic, nn.Linear, qint8, oneDNN engine — the only engine available in this build)\n",
     "| Model | FP32-INT8 cosine median (5th pct.) | mAP FP32/FP32 | INT8/INT8 | FP32 gallery + INT8 query | top-10 overlap |", "|---|---|---|---|---|---|"]
for m, a in r1["a_int8_sanity"].items(): t.append(f"| {m} | {a['cos_fp32_int8_gallery_median']:.3f} ({a['cos_p5']:.3f}) | {a['mAP_fp32_fp32']:.3f} | {a['mAP_int8_int8']:.3f} | {a['mAP_fp32gallery_int8query']:.3f} | {a['top10_overlap']:.2f} |")
t.append("\nReproducibility and batch-size checks on 94 DTD queries: " + json.dumps(r2["E1_E2_E6_int8_checks_94imgs"]))
t.append("\nBatch-1 INT8 queries vs batch-32 INT8 gallery (940 queries, clean + 8 severity-2 shifts): " + json.dumps({m: v["summary"] for m, v in r3b.items()}))
t += ["\n## Pre-specified variant effects (clean mAP change, interaction DD = drop_variant - drop_fp32 over the eight severity-2 shifts)\n", "| Model | variant | clean mAP | delta clean [CI] | DD [CI] |", "|---|---|---|---|---|"]
for m, V in rq3.items():
    for k, v in V.items(): t.append(f"| {m} | {k} | {v['clean_mAP']:.3f} | {v['delta_clean']:+.3f} [{v['delta_clean_ci'][0]:+.3f}, {v['delta_clean_ci'][1]:+.3f}] | {v['DD_mean']:+.3f} [{v['DD_ci'][0]:+.3f}, {v['DD_ci'][1]:+.3f}] |")
t.append("\nVariant keys: int8_224 = INT8 dynamic; fp32_160 / fp32_112 = input resolution; int8emb = per-dimension int8 scalar quantization of stored embeddings (min/max from the gallery, applied to gallery and queries, dequantized and re-normalized before cosine ranking); bin = zero-threshold sign codes (ranking by inner product of sign vectors, equivalent to Hamming).")
t.append("\nRelative robustness (exploratory): " + json.dumps(dec["exploratory"]["e1_relative_robustness_sev2"]))
t.append("\nRR change under INT8 with paired bootstrap CI (exploratory): " + json.dumps(r3x["RR_diff_int8_ci_exploratory"]))
t.append("\nMean-centered 1-bit codes (exploratory): " + json.dumps(dec["exploratory"]["e2_centred_binarisation"]))
t.append("\n1-bit code diversity: " + json.dumps(r2["E8_binary"]) + f"; random-ranking mAP on DTD = {r2['random_ranking_mAP']:.3f}")
t.append("\nUnified Pareto set (26 configurations, severity-2 mean shifted mAP vs three-session median latency): " + json.dumps([{k: p[k] for k in ("model", "variant", "latency_ms_sessions_median", "shifted_mAP_sev2", "pareto_sessions")} for p in r2["E3_pareto_sev2"]]))
w("S3_efficiency_checks.md", "\n".join(t) + "\n")
# S4
t = ["# S4. CPU benchmark\n", "Intel Core i5-13600K (hybrid P/E cores), 32 GB, Windows; PyTorch CPU; vision encoder only; batch 1: 20 warm-up + 100 timed runs (median, p90); batch 32: 5 runs; 4 intra-op threads unless noted; no core pinning; no other experiment running. The pre-specified plan measured one session; three randomized-order sessions were added post hoc and the paper reports the median of the three session medians.\n",
     "| Config (model, variant, px, threads) | session medians (ms) | median | p90 (median) | img/s batch 32 | state dict MiB | original single session (ms) | background load % |", "|---|---|---|---|---|---|---|---|"]
for k, v in sorted(ls.items()): t.append(f"| {k} | {', '.join(f'{x:.1f}' for x in v['session_medians'])} | {v['median_of_sessions']:.1f} | {v['p90_median']:.1f} | {v['throughput_b32_median']:.1f} | {v['state_dict_MB']} | {v['original_single_session']} | {v['bg_load']} |")
t.append("\nProcess-memory (RSS) readings were inconsistent because of allocator reuse and are not reported. Raw records: `logs/latency.jsonl`, `logs/latency_session1_original.jsonl`, `logs/latency_superseded_fullmodel.jsonl` (first attempt that loaded the text tower as well; superseded).")
w("S4_cpu_benchmark.md", "\n".join(t) + "\n")
# S5
h = J("rq1_hypotheses.json")
t = ["# S5. Statistics\n", "Query-level bootstrap (2,000 resamples, seed 2027) for DTD CIs; paired comparisons reuse the same resamples. Wilcoxon signed-rank tests with Holm correction over the eight severity-3 tests per system (H1a).\n",
     "## Hypothesis decisions\n", f"- H1a (all severity-3 drops > 0 for every system): {h['H1a_all_true']}", f"- H1b (share of monotone model-type pairs): {h['H1b_share_monotone']}", f"- H1c: {h['H1c']}; mean severity-3 drop by type: {json.dumps({k: round(v, 3) for k, v in h['mean_drop_s3_over_models'].items()})}",
     f"- H2: tau = {rq2['tau_primary']:.2f} {rq2['tau_primary_ci']}", f"- H3 equivalence (whole 95% CI within margin): {json.dumps({k: v for k, v in dec['pre_specified'].items() if k.startswith('H3_equivalence')})}",
     f"- H3 interaction: {json.dumps(dec['pre_specified']['H3_int_DD'])}", f"- H4: {json.dumps({k: round(v['mean_DD_over_models'], 3) for k, v in dec['pre_specified']['H4'].items()})}",
     f"- H5: int8 embedding equivalence {json.dumps(dec['pre_specified']['H5_int8emb_equiv_0.01'])}; 1-bit reduces > 0.01 {json.dumps(dec['pre_specified']['H5_bin_reduces_more_than_0.01'])}",
     "\n## Holm-adjusted p-values (severity 3)\n", "| Model | " + " | ".join(TYPES) + " |", "|---" * (len(TYPES) + 1) + "|"]
for m in ORDER: t.append(f"| {m} | " + " | ".join(f"{dtd[m][f'{a}_s3']['p_holm_s3']:.1e}" for a in TYPES) + " |")
t.append("\nDTD category-stratified bootstrap CIs of clean mAP (post hoc): " + json.dumps(r1["b_dtd_stratified_clean_ci"]))
t.append("\nKTH query-level (pre-specified) CIs: " + json.dumps({m: {"cat_mAP_ci": kth["reference_frontal"][m]["cat_mAP_ci"], "inst_R1_ci": kth["reference_frontal"][m]["inst_R1_ci"]} for m in ORDER}))
w("S5_statistics.md", "\n".join(t) + "\n")
# S6
env = [json.loads(l) for l in open(os.path.join(ROOT, "logs", "runs.jsonl"), encoding="utf-8")][0]
revs = {}
for m, (hf, _, _) in MODELS.items():
    d = glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + hf.replace("/", "--"), "refs", "main"))
    revs[m] = open(d[0]).read().strip() if d else "n/a"
t = ["# S6. Versions and checkpoints\n", f"Environment (from logs/runs.jsonl): python {env['python']}, torch {env['torch']}, transformers {env['transformers']}, numpy {env['numpy']}, {env['os']}.\n",
     "| Model | Hugging Face id | revision (commit) | embedding |", "|---|---|---|---|"] + [f"| {m} | {MODELS[m][0]} | {revs[m]} | {MODELS[m][2]} |" for m in MODELS]
t.append("\nSeeds: query subsample `numpy.random.default_rng(2027)`; bootstrap seed 2027; perturbation seeds from MD5 of the image identifier. DTD split 1 via the Hugging Face mirror `jxie/dtd` (47 classes, 40/40/40); KTH-TIPS2-b colour 200x200 from KTH. Datasets are not redistributed.")
w("S6_versions.md", "\n".join(t) + "\n")
# S7
t = ["# S7. Analysis plan and deviations\n", "The pre-specified plan is `PLAN_pre-specified_analysis.md` (commit 73f8d9c, 2026-11-27 00:25 KST, before the first main-experiment log entry at 00:27 KST).\n",
     "| # | Deviation / addition | When | Reason | Label in paper |", "|---|---|---|---|---|",
     "| 1 | KTH-TIPS2-b archive re-downloaded (first copy had 4,743 of 4,752 images) | before KTH extraction | corrupted download | none (data integrity) |",
     "| 2 | Latency: text tower excluded (vision encoder only); first attempt superseded | before analysis | the text tower is unused in image retrieval | Methods |",
     "| 3 | Latency: three randomized-order sessions; paper reports median of session medians | review round 2 | single-session values varied up to ~1.5x on the hybrid CPU | post hoc |",
     "| 4 | KTH sample-cluster bootstrap; DTD category-stratified bootstrap | review round 1 | repeated views of 44 physical samples | post hoc |",
     "| 5 | Same-sample exclusion on KTH | review round 1 | separate same-instance matches from material retrieval | post hoc |",
     "| 6 | INT8 sanity checks (cosine, mixed precision, determinism, batch size, batch-1 queries) | review rounds 1-3 | verify the large INT8 losses | post hoc |",
     "| 7 | Unified Pareto analysis over 26 configurations | review round 2 | compare efficiency configurations on one condition set | post hoc |",
     "| 8 | Relative-robustness CIs, RR-based interaction, per-type tau, mean-centered 1-bit codes, random baseline | after results | interpretation aids | exploratory |",
     "| 9 | 1-bit codes, int8 embedding quantization | pre-specified (H5) | — | pre-specified |"]
w("S7_plan_and_deviations.md", "\n".join(t) + "\n")
