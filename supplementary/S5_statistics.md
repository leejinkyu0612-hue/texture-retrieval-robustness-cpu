# S5. Statistics

Query-level bootstrap (2,000 resamples, seed 2027) for DTD CIs; paired comparisons reuse the same resamples. Wilcoxon signed-rank tests with Holm correction over the eight severity-3 tests per system (H1a).

## Hypothesis decisions

- H1a (all severity-3 drops > 0 for every system): True
- H1b (share of monotone model-type pairs): 1.0
- H1c: True; mean severity-3 drop by type: {"low_light": 0.087, "colour_temp": 0.029, "rotation": 0.031, "perspective": 0.02, "partial_crop": 0.096, "blur": 0.2, "jpeg": 0.148, "downscale": 0.178}
- H2: tau = 1.00 [0.8571428571428571, 0.9999999999999998]
- H3 equivalence (whole 95% CI within margin): {"H3_equivalence_thr_0.005": {"CLIP-B/32": false, "CLIP-B/16": false, "SigLIP-B/16": false, "DINOv2-S/14": false, "DINOv2-B/14": false}, "H3_equivalence_thr_0.01": {"CLIP-B/32": false, "CLIP-B/16": false, "SigLIP-B/16": false, "DINOv2-S/14": true, "DINOv2-B/14": false}, "H3_equivalence_thr_0.02": {"CLIP-B/32": false, "CLIP-B/16": false, "SigLIP-B/16": false, "DINOv2-S/14": true, "DINOv2-B/14": false}}
- H3 interaction: {"CLIP-B/32": {"DD": -0.0019572257937025533, "ci": [-0.0063214884356018256, 0.0021187608519541382], "p": 0.9173529343620875}, "CLIP-B/16": {"DD": -0.022059761677443338, "ci": [-0.029225388366199583, -0.015016455961459311], "p": 1.5716322871672401e-09}, "SigLIP-B/16": {"DD": 0.017925894520640574, "ci": [0.013276021045010423, 0.022759028134144033], "p": 2.533791657529387e-14}, "DINOv2-S/14": {"DD": 0.006481886430327571, "ci": [0.004337739973272349, 0.008597755687353164], "p": 1.0998083151417293e-12}, "DINOv2-B/14": {"DD": 0.020751819759253016, "ci": [0.015844268451524878, 0.025610064663255053], "p": 2.3399198093436973e-19}}
- H4: {"blur_s2": -0.071, "downscale_s2": -0.045}
- H5: int8 embedding equivalence {"CLIP-B/32": true, "CLIP-B/16": true, "SigLIP-B/16": true, "DINOv2-S/14": true, "DINOv2-B/14": true, "ResNet-50": true, "MobileNetV2": true}; 1-bit reduces > 0.01 {"CLIP-B/32": false, "CLIP-B/16": false, "SigLIP-B/16": false, "DINOv2-S/14": true, "DINOv2-B/14": true, "ResNet-50": true, "MobileNetV2": true}

## Holm-adjusted p-values (severity 3)

| Model | low_light | colour_temp | rotation | perspective | partial_crop | blur | jpeg | downscale |
|---|---|---|---|---|---|---|---|---|
| CLIP-B/32 | 1.3e-134 | 8.5e-55 | 9.1e-42 | 2.4e-31 | 2.5e-100 | 3.2e-148 | 2.2e-126 | 1.1e-144 |
| CLIP-B/16 | 6.0e-137 | 5.4e-56 | 6.2e-41 | 4.4e-42 | 4.5e-101 | 8.3e-147 | 6.8e-131 | 9.1e-142 |
| SigLIP-B/16 | 4.2e-107 | 2.2e-29 | 2.2e-33 | 5.9e-18 | 1.4e-97 | 9.6e-145 | 7.2e-129 | 2.3e-143 |
| DINOv2-S/14 | 2.3e-32 | 6.6e-10 | 1.8e-11 | 1.9e-04 | 1.6e-26 | 2.1e-108 | 8.9e-80 | 3.2e-98 |
| DINOv2-B/14 | 1.5e-25 | 4.0e-05 | 1.1e-06 | 2.0e-06 | 2.0e-12 | 2.8e-105 | 1.8e-53 | 2.3e-85 |
| ResNet-50 | 1.8e-03 | 1.1e-13 | 3.7e-13 | 6.3e-03 | 6.7e-66 | 1.8e-123 | 1.5e-87 | 5.6e-115 |
| MobileNetV2 | 2.1e-82 | 1.4e-10 | 1.1e-38 | 3.6e-16 | 5.1e-97 | 1.0e-138 | 1.1e-120 | 3.5e-130 |
| LBP+colour | 8.5e-75 | 1.6e-44 | 3.6e-18 | 1.7e-11 | 6.1e-15 | 1.6e-37 | 1.1e-34 | 1.7e-32 |

DTD category-stratified bootstrap CIs of clean mAP (post hoc): {"CLIP-B/32": [0.28185791877783456, 0.3019705924986558], "CLIP-B/16": [0.30089388606260564, 0.32155005780964646], "SigLIP-B/16": [0.39898628443445394, 0.42034241716235193], "DINOv2-S/14": [0.4132511959317034, 0.43502867737347445], "DINOv2-B/14": [0.43953024629924065, 0.4617392473099282], "ResNet-50": [0.2803735344987165, 0.30107540678040245], "MobileNetV2": [0.21743330885712914, 0.23443819986233538], "LBP+colour": [0.048661172496351206, 0.053277804815938946]}

KTH query-level (pre-specified) CIs: {"CLIP-B/32": {"cat_mAP_ci": [0.7307381422787442, 0.7431925183541702], "inst_R1_ci": [0.6074713254035684, 0.6353016142735769]}, "CLIP-B/16": {"cat_mAP_ci": [0.764500449605879, 0.777132148243457], "inst_R1_ci": [0.6395497026338148, 0.6665250637213254]}, "SigLIP-B/16": {"cat_mAP_ci": [0.8312022580328159, 0.8411242295040932], "inst_R1_ci": [0.6873300764655905, 0.7130469413763806]}, "DINOv2-S/14": {"cat_mAP_ci": [0.841028435902922, 0.8507534418561042], "inst_R1_ci": [0.6671622769753611, 0.6941376380628717]}, "DINOv2-B/14": {"cat_mAP_ci": [0.8704126817814133, 0.8797136522893506], "inst_R1_ci": [0.6985981308411215, 0.7251486830926084]}, "ResNet-50": {"cat_mAP_ci": [0.7652496242955453, 0.7782514519199949], "inst_R1_ci": [0.5686066270178419, 0.596644010195412]}, "MobileNetV2": {"cat_mAP_ci": [0.72866533825381, 0.7416400533700599], "inst_R1_ci": [0.5875053101104503, 0.6146983857264231]}, "LBP+colour": {"cat_mAP_ci": [0.505710980691168, 0.5218568697831445], "inst_R1_ci": [0.5142257858963466, 0.5424808836023789]}}
