#!/bin/bash
# V2 + Qwen-LoRA two-stage chain: predictor -> stage2 (auto if stage1 succeeds).
# Both stages bs16 (tested: predictor 28.9GB, stage2 35.8GB of 48GB).
# Usage: bash scripts/run_2stage_dino_lora.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_10
OUT=results/dino_mamba_diff_lora_${MIX}
STEPS=15000

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

common="--framework VLA_DINO_Mamba_Diff --dino_backbone dinov2_vitb14 --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --max_steps ${STEPS} --warmup_steps 300 --log_every 10 --save_every 3000 --num_workers 12 --cuda 0"

mkdir -p ${OUT}/pred ${OUT}/stage2

echo "=== Stage 1: predictor (+LoRA), bs8 ==="
python -u scripts/train_mamba_wm.py --stage predictor ${common} \
  --batch_size 16 --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log

echo "=== Stage 2: stage2 (+LoRA+head), bs8, resume from Stage1 ==="
python -u scripts/train_mamba_wm.py --stage stage2 ${common} \
  --batch_size 16 --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
  --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
