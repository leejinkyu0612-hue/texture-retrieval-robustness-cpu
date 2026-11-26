# Pre-specified analysis plan (사전계획) — fixed before any retrieval accuracy is computed

Study: Robustness–efficiency evaluation of frozen image embeddings for texture retrieval on CPU
Author: Jinkyu Lee (HappyAI). Plan written 2026-11-27 00:25 KST, committed to a private GitHub repository before the main experiment.
Status of data at commit time: only a feasibility pilot was run (EXP_실험/results/pilot_feasibility.json): preprocessing parity with the official Hugging Face processors, that all models/variants execute, and throughput in images/s. No retrieval accuracy (AP, mAP, Recall@1) was computed before this commit. Any analysis added after this commit is labelled exploratory.
한국어 요약: 이 문서는 정확도 실험 전에 연구 질문·가설·지표·기준값·분석·제외 기준을 고정한 것이다. 이후 추가 분석은 '탐색적'으로 표시한다.

## 1. Research questions
- RQ1 (synthetic capture/transmission shifts): How much does texture-category retrieval quality of frozen embeddings drop under eight controlled query-side perturbations at three severities?
- RQ2 (synthetic vs real capture variation): Does the model ranking obtained under synthetic perturbations (DTD) agree with the ranking under real capture variation in illumination, pose and scale (KTH-TIPS2-b)?
- RQ3 (CPU efficiency): What are the CPU latency, throughput and memory of each model, and do common CPU efficiency measures (INT8 dynamic quantisation, lower input resolution, embedding compression) change clean accuracy and the robustness drop?

## 2. Data
- DTD (Cimpoi et al., CVPR 2014), split 1 as distributed in the Hugging Face mirror `jxie/dtd` (verified: 47 classes, 40/40/40 images per class in train/validation/test). Gallery = train+validation (3,760 clean images). Queries = 20 test images per class (940 queries), selected once with `numpy.random.default_rng(2027)` per class before any accuracy is computed. Reason for 940 instead of 1,880: CPU budget measured in the pilot (about 8 images/s for ViT-B/16 under concurrent load); 20 queries per class keeps the class balance.
- KTH-TIPS2-b (Caputo et al., ICCV 2005; documentation by Mallikarjuna et al. 2006): 11 materials × 4 physical samples × 9 scales × 3 poses × 4 illuminations (108 images per sample, 200×200 crops). Image index → pose/illumination mapping from Table 5 of the database documentation.
- Exclusions: images that fail to decode, or missing files in the KTH archive, are excluded and counted; no other exclusions. No image is excluded based on model output.

## 3. Common pipeline
Every image is resized (shorter side 256, bicubic) and centre-cropped to 256×256 (the base image). Perturbations act on the base image of queries only; the gallery remains clean (catalogue-versus-field-photo setting). Model input preprocessing replicates each official processor (resize ratio, centre crop, interpolation, mean/std; checked against the HF processor in the pilot). Embeddings are L2-normalised; retrieval is exact cosine nearest-neighbour search over the full gallery. No training or fine-tuning.

## 4. Models (independent variable: backbone family and size)
CLIP ViT-B/32 and ViT-B/16 (language-supervised), SigLIP B/16-224 (language-supervised), DINOv2 ViT-S/14 and ViT-B/14 (self-supervised), ResNet-50 and MobileNetV2 (ImageNet-supervised CNNs). Training-free baseline: LBP(8,1) 256-bin histogram + 4×4×4 RGB histogram (square-root, L2). Embeddings: CLIP/SigLIP image features; DINOv2 pooled CLS token; CNN global average pool.

## 5. Perturbations (RQ1), applied to queries; severity 1/2/3
low-light exposure factor 0.5/0.3/0.15; colour-temperature shift (R×(1+a), B×(1−a)) a = 0.15/0.30/0.45; in-plane rotation 15°/30°/45° followed by the largest inscribed square; perspective (keystone) top-corner inset 8%/16%/24%; partial view (seeded random crop keeping 50%/25%/10% of the area, resized back); Gaussian blur σ = 1.5/3/6 px; JPEG quality 30/15/5; down-up-sampling factor 2/4/8. Seeds are derived from the image identifier, so every model sees identical perturbed queries.

## 6. Real capture variation (RQ2)
Catalogue gallery = one image per physical sample at the reference condition (scale 6 = relative scale 1.0, frontal pose, frontal illumination; 44 gallery images). Queries = all other KTH-TIPS2-b images. Groups: illumination-only change (scale 6, frontal pose, other illumination), pose-only change (scale 6, frontal illumination, other pose), scale-only change (frontal pose and illumination, other scale), and mixed (all remaining). Metrics: instance Recall@1 (same physical sample ranked first) and category mAP (same material). Sensitivity: alternative reference condition (scale 6, frontal pose, ambient illumination).

## 7. Efficiency (RQ3)
- Measurements on the same CPU (Intel Core i5-13600K, 14 cores, 32 GB, Windows, PyTorch CPU): batch-1 latency (20 warm-up, 100 timed runs; median and 90th percentile) with 4 threads (primary, edge-like budget) and with all threads; batch-32 throughput; resident memory after loading and peak during inference; parameter count; embedding storage (bytes/image). Background load is recorded at measurement time.
- Efficiency variants: (a) INT8 dynamic quantisation of all Linear layers (`torch.ao.quantization.quantize_dynamic`) for the five transformer models (not applicable to the CNN feature extractors, whose Linear layers are only in the unused classifier); (b) input resolution 160 and 112 (instead of 224) for all models, with position-embedding interpolation for CLIP/SigLIP; (c) embedding compression after extraction: per-dimension int8 scalar quantisation (range from the gallery) and 1-bit sign binarisation (Hamming ranking).
- Variants are evaluated on clean queries and on all eight perturbations at severity 2 (middle level) to bound CPU time.

## 8. Metrics
Primary: mAP over the full gallery. Secondary: Recall@1. Robustness: drop Δ = mAP_clean − mAP_perturbed (absolute, mAP points) and relative robustness RR = mAP_perturbed / mAP_clean. Efficiency-robustness interaction: ΔΔ = Δ_variant − Δ_fp32-224 (paired over the same queries). Efficiency: median latency (ms), throughput (images/s), memory (MB), storage (bytes/image).

## 9. Hypotheses (confirmatory)
- H1a: At severity 3, every perturbation type reduces mAP relative to clean for every model (paired 95% bootstrap CI of Δ excludes 0). H1b: The drop is non-decreasing with severity (Δ1 ≤ Δ2 ≤ Δ3 within CI overlap) in at least 80% of model × type pairs. H1c: Averaged over models, colour-temperature shift at severity 3 produces a smaller drop than each of blur, down-sampling and partial view at severity 3.
- H2: Model rankings by mean perturbed mAP on DTD (24 conditions) and by KTH category mAP over non-reference queries are positively associated (Kendall τ > 0; 95% bootstrap CI reported). Secondary: τ between DTD mean RR and KTH RR (KTH RR = mAP of scale-shifted queries with |scale−6| ≥ 2 divided by mAP of scale-6 queries). Given Taori et al. (2020), a weak association is a plausible outcome and will be reported as such.
- H3: INT8 dynamic quantisation changes clean mAP by less than the smallest effect of interest (|Δclean| < 0.01, 95% CI within ±0.01) for each transformer model. H3-int: whether ΔΔ (mean over the eight severity-2 perturbations) differs from 0 is tested two-sided (no direction pre-committed).
- H4: Reducing resolution to 112 increases the robustness drop (ΔΔ > 0) for blur and down-sampling at severity 2, averaged over models.
- H5: Int8 embedding quantisation changes clean mAP by less than 0.01; 1-bit binarisation reduces clean mAP by more than 0.01 for every model.

## 10. Thresholds and their justification
- Smallest effect of interest 0.01 mAP (one percentage point): an operational choice for deciding whether an efficiency step is "practically neutral" in a product-search deployment; it is not a universal standard. Sensitivity analysis with 0.005 and 0.02.
- Latency budget 100 ms per query image at 4 threads: based on the commonly cited ~0.1 s limit for a response to feel immediate (Card, Robertson & Mackinlay, CHI 1991; to be verified), used only to mark configurations; sensitivity with 50 ms and 200 ms.
- Severity levels: chosen to span mild to strong changes in the style of Hendrycks & Dietterich (2019); they are not calibrated to a specific camera.

## 11. Statistical analysis
- Query-level bootstrap (2,000 resamples, seed 2027) for 95% CIs of mAP, Δ, RR and ΔΔ (paired: the same resampled queries for both conditions).
- Paired Wilcoxon signed-rank tests on per-query AP differences, Holm correction within each family (H1a: 8 types × 7 models + baseline; H3: 5 models; H4/H5 per model). Effects are interpreted with CIs and the 0.01 threshold, not p-values alone (large n makes tiny differences significant).
- Kendall τ-b with bootstrap CI over queries (resampling queries in both datasets) for H2.
- Interpretation is associational under controlled manipulation; no causal claims beyond "changing condition X on the same queries changed mAP by Y in this setup".

## 12. Exploratory analyses (labelled as such if run)
Per-class failure analysis, CLS vs patch-mean pooling, instance-level relevance on DTD near-duplicates, alternative KTH reference conditions beyond the one listed in Section 6.

## 13. Reproducibility and logging
Code in `EXP_실험/code`, one JSON line per extraction run in `EXP_실험/logs/runs.jsonl` (model, variant, resolution, dataset, condition, number of images, wall time, library versions). Hugging Face model identifiers fixed; images are not redistributed (download script only).
