#!/bin/bash
# A-b standalone: DINO unfrozen (action-anchored), libero_all 4-suite, 60k+60k, eval on libero_10.
set -eo pipefail
cd "$(dirname "$0")/.."
CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
OUT=results/dino_mamba_A_b_liberoall
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd)
b="--framework VLA_DINO_Mamba_Diff --dino_backbone dinov2_vitb14 --qwen_lora --lora_r 16 --lora_alpha 32 --train_dino \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix libero_all \
  --max_steps 60000 --warmup_steps 500 --log_every 20 --save_every 5000 --num_workers 12 --cuda 0"
mkdir -p ${OUT}/pred ${OUT}/stage2
echo "### A-b Stage1 predictor (60k, libero_all) ###"
python -u scripts/train_mamba_wm.py --stage predictor ${b} --batch_size 16 --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
echo "### A-b Stage2 (60k, resume) ###"
python -u scripts/train_mamba_wm.py --stage stage2 ${b} --batch_size 16 --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
echo "### A-b eval on libero_10 (50 trials) ###"
bash scripts/eval_libero_dino.sh libero_10 50 18021 ${OUT}/stage2/checkpoints/mamba_wm_final.pt A_b_50 0
echo "A-b:"; grep -i "Total success rate" results/eval/libero_10_A_b_50/eval.log | tail -1
