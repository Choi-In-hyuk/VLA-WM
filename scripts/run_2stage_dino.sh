#!/bin/bash
# Two-stage curriculum for VLA_DINO_Mamba: predictor (Stage 1) -> id (Stage 2).
# Stage 2 auto-starts only if Stage 1 succeeds (&&), resuming from its final ckpt.
#
# Usage: bash scripts/run_2stage_dino.sh
set -eo pipefail   # pipefail so a python failure (not tee) aborts the chain
cd "$(dirname "$0")/.."

CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_10
CACHE=results/cache/qwen_libero10
OUT=results/dino_mamba_${MIX}
BS=8
STEPS=15000

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

common="--framework VLA_DINO_Mamba --dino_backbone dinov2_vitb14 --qwen_cache ${CACHE} \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --batch_size ${BS} --max_steps ${STEPS} --warmup_steps 300 \
  --log_every 10 --save_every 3000 --num_workers 12 --cuda 0"

mkdir -p ${OUT}/pred ${OUT}/id

echo "=== Stage 1: predictor ==="
python -u scripts/train_mamba_wm.py --stage predictor ${common} \
  --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log

echo "=== Stage 2: id (resume from Stage 1) ==="
python -u scripts/train_mamba_wm.py --stage id ${common} \
  --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
  --output_dir ${OUT}/id 2>&1 | tee ${OUT}/id/train.log

echo "=== done. final (eval-ready) ckpt: ${OUT}/id/checkpoints/mamba_wm_final.pt ==="
