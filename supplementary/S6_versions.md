# S6. Versions and checkpoints

Environment (from logs/runs.jsonl): python 3.13.12, torch 2.15.0+cpu, transformers 5.24.0, numpy 2.5.5, Windows-11-10.0.26300-SP0.

| Model | Hugging Face id | revision (commit) | embedding |
|---|---|---|---|
| CLIP-B/32 | openai/clip-vit-base-patch32 | 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 | projected image embedding |
| CLIP-B/16 | openai/clip-vit-base-patch16 | 57c216476eefef5ab752ec549e440a49ae4ae5f3 | projected image embedding |
| SigLIP-B/16 | google/siglip-base-patch16-224 | 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed | pooled image embedding |
| DINOv2-S/14 | facebook/dinov2-small | ed25f3a31f01632728cabb09d1542f84ab7b0056 | CLS (pooler) token |
| DINOv2-B/14 | facebook/dinov2-base | f9e44c814b77203eaa57a6bdbbd535f21ede1415 | CLS (pooler) token |
| ResNet-50 | microsoft/resnet-50 | 34c2154c194f829b11125337b98c8f5f9965ff19 | global average pool |
| MobileNetV2 | google/mobilenet_v2_1.0_224 | 75e607b00aeae1297cc89d026a118bce012f5c5a | global average pool |

Seeds: query subsample `numpy.random.default_rng(2027)`; bootstrap seed 2027; perturbation seeds from MD5 of the image identifier. DTD split 1 via the Hugging Face mirror `jxie/dtd` (47 classes, 40/40/40); KTH-TIPS2-b colour 200x200 from KTH. Datasets are not redistributed.
