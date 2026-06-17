#!/bin/bash
# Direction A (MambaVLA insight: long training matters). Two variants, each train->eval:
#   A-a: V2 + Qwen-LoRA, DINO FROZEN, 40k+40k  (isolates training-length effect vs 88.4%)
#   A-b: V2 + Qwen-LoRA, DINO UNFROZEN (train encoder too), 40k+40k (MambaVLA-style enc learning)
# Runs sequentially on one GPU: A-a train -> A-a eval -> A-b train -> A-b eval.
# Usage: bash scripts/run_A_longtrain.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_10
STEPS=40000

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

base="--framework VLA_DINO_Mamba_Diff --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --max_steps ${STEPS} --warmup_steps 500 --log_every 20 --save_every 5000 --num_workers 12 --cuda 0"

run_variant () {   # $1=tag  $2=OUT  $3=extra train flag  $4=eval_port  $5=train_mix  $6=steps
  local TAG=$1 OUT=$2 EXTRA=$3 PORT=$4 TMIX=$5 ST=$6
  local b="--framework VLA_DINO_Mamba_Diff --dino_backbone dinov2_vitb14 \
    --qwen_lora --lora_r 16 --lora_alpha 32 \
    --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${TMIX} \
    --max_steps ${ST} --warmup_steps 500 --log_every 20 --save_every 5000 --num_workers 12 --cuda 0"
  mkdir -p ${OUT}/pred ${OUT}/stage2
  echo "############ A-${TAG}: Stage1 predictor (${ST}, ${TMIX}) ${EXTRA} ############"
  python -u scripts/train_mamba_wm.py --stage predictor ${b} ${EXTRA} \
    --batch_size 16 --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
  echo "############ A-${TAG}: Stage2 (${ST}, resume) ############"
  python -u scripts/train_mamba_wm.py --stage stage2 ${b} ${EXTRA} \
    --batch_size 16 --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
  echo "############ A-${TAG}: eval on libero_10 (50 trials) ############"
  bash scripts/eval_libero_dino.sh libero_10 50 ${PORT} \
    ${OUT}/stage2/checkpoints/mamba_wm_final.pt A_${TAG}_50 0
  echo "A-${TAG} success rate:"; grep -i "Total success rate" results/eval/libero_10_A_${TAG}_50/eval.log | tail -1
}

# A-a: DINO frozen, libero_10, 40k (isolate training-length).
# A-b: DINO unfrozen (action-anchored), libero_all 4-suite, 60k (encoder learning needs data diversity).
run_variant a results/dino_mamba_A_a_${MIX}     ""             18020 libero_10  40000

echo "############ A DONE ############"
echo "A-a (libero_10, 40k, DINO frozen):"; grep -i "Total success rate" results/eval/libero_10_A_a_50/eval.log | tail -1
echo "A-b (libero_all 4-suite, 60k, DINO trained):"; grep -i "Total success rate" results/eval/libero_10_A_b_50/eval.log | tail -1
echo "(baseline V2+LoRA = 88.4%)"
