#!/bin/bash
# #2: unfreeze DINO + V-JEPA-style encoder learning, then V2 stage-2 action head.
# Built on the best V2 recipe (+Qwen-LoRA, 88.4%); target = beat 88.4%.
#   Stage1 (jepa) : online DINO + EMA target + context masking + Qwen-LoRA, L_pred only. lr 5e-5 (DINO-safe).
#   Stage2 (stage2): FREEZE learned DINO, V2 stage-2 (predictor fine-tune + diffusion head) + Qwen-LoRA, L_pred+L_action.
# Usage: bash scripts/run_2stage_dino_jepa.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_10
OUT=results/dino_mamba_jepa_${MIX}
S1_STEPS=30000   # jepa: learns DINO encoder too -> longer
S2_STEPS=25000   # stage2: predictor fine-tune + diffusion head

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

common="--framework VLA_DINO_Mamba_JEPA --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps 500 --log_every 10 --save_every 3000 --num_workers 12 --cuda 0"

mkdir -p ${OUT}/jepa ${OUT}/stage2

echo "=== Stage 1: jepa (online DINO + EMA + masking + LoRA), bs16, lr 5e-5, ${S1_STEPS} steps ==="
python -u scripts/train_mamba_wm.py --stage jepa ${common} --max_steps ${S1_STEPS} \
  --batch_size 16 --lr 5e-5 --mask_ratio 0.5 --ema_momentum 0.996 \
  --output_dir ${OUT}/jepa 2>&1 | tee ${OUT}/jepa/train.log

echo "=== Stage 2: stage2 (freeze learned DINO + head + LoRA), bs16, ${S2_STEPS} steps, resume from Stage1 ==="
python -u scripts/train_mamba_wm.py --stage stage2 ${common} --max_steps ${S2_STEPS} \
  --batch_size 16 --lr 1e-4 --resume_ckpt ${OUT}/jepa/checkpoints/mamba_wm_final.pt \
  --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
