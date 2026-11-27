# S7. Analysis plan and deviations

The pre-specified plan is `PLAN_pre-specified_analysis.md` (commit 73f8d9c, 2026-11-27 00:25 KST, before the first main-experiment log entry at 00:27 KST).

| # | Deviation / addition | When | Reason | Label in paper |
|---|---|---|---|---|
| 1 | KTH-TIPS2-b archive re-downloaded (first copy had 4,743 of 4,752 images) | before KTH extraction | corrupted download | none (data integrity) |
| 2 | Latency: text tower excluded (vision encoder only); first attempt superseded | before analysis | the text tower is unused in image retrieval | Methods |
| 3 | Latency: three randomized-order sessions; paper reports median of session medians | review round 2 | single-session values varied up to ~1.5x on the hybrid CPU | post hoc |
| 4 | KTH sample-cluster bootstrap; DTD category-stratified bootstrap | review round 1 | repeated views of 44 physical samples | post hoc |
| 5 | Same-sample exclusion on KTH | review round 1 | separate same-instance matches from material retrieval | post hoc |
| 6 | INT8 sanity checks (cosine, mixed precision, determinism, batch size, batch-1 queries) | review rounds 1-3 | verify the large INT8 losses | post hoc |
| 7 | Unified Pareto analysis over 26 configurations | review round 2 | compare efficiency configurations on one condition set | post hoc |
| 8 | Relative-robustness CIs, RR-based interaction, per-type tau, mean-centered 1-bit codes, random baseline | after results | interpretation aids | exploratory |
| 9 | 1-bit codes, int8 embedding quantization | pre-specified (H5) | — | pre-specified |
