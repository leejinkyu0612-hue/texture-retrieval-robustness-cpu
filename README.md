# Texture image retrieval: robustness and CPU efficiency of off-the-shelf image embeddings

Code, logs and Supplementary material for the paper *Robustness and CPU Efficiency of Off-the-Shelf Image Embeddings for Texture Retrieval under Query Distribution Shifts* (ICAIIC 2027 submission).

* `PLAN_pre-specified_analysis.md` - analysis plan committed before the main experiment (commit 73f8d9c, 2026-11-27 00:25 KST).
* `supplementary/S1`-`S7` - perturbation operators, extended results, efficiency checks, CPU benchmark, statistics, versions/checkpoints, plan deviations.
* `code/` - `common.py` (data, perturbations, preprocessing, models, metrics), `run_extract.py` (embedding extraction), `run_latency*.py` (CPU timing), `analyze*.py` (pre-specified and post-hoc analyses), `make_tables.py`, `make_figures*.py`, `make_supplementary.py`.
* `results/` - JSON outputs used for every number in the paper; `logs/` - one JSON line per extraction run (`runs.jsonl`) and per latency measurement (`latency.jsonl`).

Reproduce: obtain DTD (split 1) and KTH-TIPS2-b from the original sources (see `code/README_data.md`), then run `code/run_extract.py` jobs, `code/analyze.py`, `code/analyze_extra.py`, `code/analyze_r1.py`, `code/analyze_r2.py`, `code/run_int8_batch1.py`, `code/make_supplementary.py`. CPU only; no fine-tuning.

Datasets are not redistributed. DTD: released for research purposes by its authors. KTH-TIPS2-b: publicly downloadable from KTH.

AI assistance: ChatGPT (OpenAI) and Claude (Anthropic) assisted with drafting, language editing, code generation and internal review; all numbers come from the logged outputs in this repository.

License: code MIT (see LICENSE).
