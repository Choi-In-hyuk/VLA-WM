#!/bin/bash
# V3-a: temporal-buffer world model (past 7 consecutive latents -> Mamba -> endpoint s_{t+7}).
# Encoder = FROZEN DINO (best V2+LoRA recipe). Both stages +Qwen-LoRA. Target = beat 88.4%.
#   Stage1 (predictor): L_pred only (Mamba temporal predictor).
#   Stage2 (stage2)   : predictor fine-tune + diffusion head, L_pred+L_action.
# Usage: bash scripts/run_2stage_dino_temporal.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_10
OUT=results/dino_mamba_temporal_${MIX}
STEPS=15000

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

common="--framework VLA_DINO_Mamba_Temporal --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --max_steps ${STEPS} --warmup_steps 300 --log_every 10 --save_every 3000 --num_workers 12 --cuda 0"

mkdir -p ${OUT}/pred ${OUT}/stage2

echo "=== Stage 1: predictor (temporal buffer + LoRA), bs8 ==="
python -u scripts/train_mamba_wm.py --stage predictor ${common} \
  --batch_size 8 --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log

echo "=== Stage 2: stage2 (temporal + head + LoRA), bs8, resume from Stage1 ==="
python -u scripts/train_mamba_wm.py --stage stage2 ${common} \
  --batch_size 8 --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
  --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
