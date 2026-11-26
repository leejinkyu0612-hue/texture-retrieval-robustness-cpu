@echo off
set PYTHONIOENCODING=utf-8
set PYTHONPATH=
set PYTHONHOME=
set HF_HUB_OFFLINE=1
cd /d "C:\Users\leeji\Desktop\전체서류\03_연구논문\논문\(진행)ICAIIC2027_자동투고\EXP_실험"
"C:\Users\leeji\.aside\runtime\python\runtime\python.exe" -W ignore code\run_extract.py "LBP+colour@none@256@full" "CLIP-B/32@fp32@224@full" "MobileNetV2@fp32@224@full" "DINOv2-S/14@fp32@224@full" "ResNet-50@fp32@224@full" "CLIP-B/16@fp32@224@full" "SigLIP-B/16@fp32@224@full" "DINOv2-B/14@fp32@224@full" "LBP+colour@none@256@kth" "CLIP-B/32@fp32@224@kth" "MobileNetV2@fp32@224@kth" "DINOv2-S/14@fp32@224@kth" "ResNet-50@fp32@224@kth" "CLIP-B/16@fp32@224@kth" "SigLIP-B/16@fp32@224@kth" "DINOv2-B/14@fp32@224@kth" >> logs\extract_main.out 2>&1
