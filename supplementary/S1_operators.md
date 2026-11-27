# S1. Query perturbation operators

Every image is first resized (shorter side 256, bicubic) and center-cropped to 256x256. Perturbations act on the query base image only; the gallery stays clean. Implementation: `code/common.py::perturb` (PIL/NumPy, sRGB 8-bit values, clipping to [0,255]). Random elements (partial-view crop location) are seeded by the image identifier.

| Shift | s1 | s2 | s3 | Operation |
|---|---|---|---|---|
| low_light | 0.5 | 0.3 | 0.15 | multiply sRGB pixel values by factor, clip |
| colour_temp | 0.15 | 0.3 | 0.45 | R x (1+a), B x (1-a), clip |
| rotation | 15 | 30 | 45 | rotate (bicubic) then crop largest inscribed square, resize to 256 |
| perspective | 0.08 | 0.16 | 0.24 | keystone: top corners moved inward by fraction of width (PIL perspective, bicubic) |
| partial_crop | 0.5 | 0.25 | 0.1 | seeded random square crop keeping the area fraction, resize to 256 (bicubic) |
| blur | 1.5 | 3.0 | 6.0 | PIL GaussianBlur radius = sigma |
| jpeg | 30 | 15 | 5 | PIL JPEG encode/decode at quality q |
| downscale | 2 | 4 | 8 | bilinear down by factor, bilinear up to 256 |

Model input preprocessing replicates each official Hugging Face processor (resize ratio, center crop, interpolation, mean/std):

| Model | HF id | resize ratio | center crop | interpolation |
|---|---|---|---|---|
| CLIP-B/32 | openai/clip-vit-base-patch32 | 1.0 | True | bicubic |
| CLIP-B/16 | openai/clip-vit-base-patch16 | 1.0 | True | bicubic |
| SigLIP-B/16 | google/siglip-base-patch16-224 | None | False | bicubic |
| DINOv2-S/14 | facebook/dinov2-small | 1.1428571428571428 | True | bicubic |
| DINOv2-B/14 | facebook/dinov2-base | 1.1428571428571428 | True | bicubic |
| ResNet-50 | microsoft/resnet-50 | 1.1428571428571428 | True | bicubic |
| MobileNetV2 | google/mobilenet_v2_1.0_224 | 1.1428571428571428 | True | bilinear |

Example images: Fig. 1 of the paper (`results/figs`).
