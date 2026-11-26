# Texture retrieval robustness and CPU efficiency (frozen image embeddings)

Code and logs for a study of how frozen image embeddings (CLIP, SigLIP, DINOv2, ResNet-50, MobileNetV2) behave in texture-category image retrieval under controlled capture/transmission shifts (DTD) and real capture variation (KTH-TIPS2-b), and how CPU efficiency measures (INT8 dynamic quantisation, lower input resolution, embedding compression) change accuracy and robustness.

* `PLAN_pre-specified_analysis.md`: analysis plan fixed before the main experiment (see commit history for the timestamp).
* `code/`: experiment code. Datasets are not redistributed; see `code/README_data.md`.
* `results/`, `logs/`: outputs of the logged runs.

Datasets: DTD (Cimpoi et al., CVPR 2014; released for research purposes) and KTH-TIPS2-b (Caputo et al., ICCV 2005; publicly downloadable from KTH). Please obtain them from the original sources.
