#!/bin/bash
# Stable LIBERO-10 comparison: non-LoRA vs LoRA, 50 trials/task each (sequential).
# Usage: bash scripts/eval_compare_50.sh
cd "$(dirname "$0")/.."
NONLORA=results/dino_mamba_diff_libero10/stage2/checkpoints/mamba_wm_final.pt
LORA=results/dino_mamba_diff_lora_libero_10/stage2/checkpoints/mamba_wm_final.pt

echo "############ non-LoRA: 50 trials ############"
bash scripts/eval_libero_dino.sh libero_10 50 18010 ${NONLORA} diff50 0

echo "############ LoRA: 50 trials ############"
bash scripts/eval_libero_dino.sh libero_10 50 18011 ${LORA} diff_lora50 0

echo "############ DONE ############"
echo "non-LoRA:" ; grep -i "Total success rate" results/eval/libero_10_diff50/eval.log | tail -1
echo "LoRA:" ; grep -i "Total success rate" results/eval/libero_10_diff_lora50/eval.log | tail -1
