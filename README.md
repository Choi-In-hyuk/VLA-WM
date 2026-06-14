<h3 align="center" style="font-size:44px; font-weight:bold; color:#9C276A; margin: 0;">
  VLA-WM: A Lightweight Mamba Latent World Model<br/>for Inference-Time Action Generation in VLA Models
</h3>

<div align="center">
<p>
  <img src="https://img.shields.io/badge/Task-Vision--Language--Action-blue.svg" alt="VLA">
  <img src="https://img.shields.io/badge/World%20Model-Mamba%20(SSM)-purple.svg" alt="Mamba">
  <img src="https://img.shields.io/badge/Benchmark-LIBERO--10-green.svg" alt="LIBERO">
</p>
<p align="center">
  ⭐ If this project helps you, please give it a star!
</p>
</div>

---

> **TL;DR.** VLA-JEPA trains a V-JEPA world model only as an auxiliary loss and
> **drops it at inference**. We *revive the world model at inference time*: a
> lightweight **Mamba (SSM)** latent world model — conditioned on **Qwen-VL's
> language+visual intent** — predicts the future latent state, from which a
> flow-matching head generates the action chunk. On LIBERO-10 this reaches
> **88.4%** (V2 + Qwen-LoRA), near base-VLA-JEPA quality but with a *lightweight,
> inference-time* world model. We also explore **unfreezing the DINO encoder and
> learning it the V-JEPA way** (online + EMA + masking); see Results for an honest
> account of when that helps and when it does not.

This repository builds on [VLA-JEPA](https://arxiv.org/abs/2602.10098) and is a
research extension, not the original release.

## Key ideas

1. **World model at inference (not just an auxiliary loss).** VLA-JEPA's V-JEPA
   world model is dropped at test time; here the world model *is* the action
   generator's conditioning at inference.
2. **Lightweight Mamba instead of V-JEPA.** The heavy V-JEPA video encoder / ViT
   predictor is replaced by light Mamba blocks operating on per-frame DINO
   latents — fast enough for closed-loop control.
3. **VLM-intent conditioning.** Qwen-VL's action tokens (its language+visual
   understanding of *how the scene should change*) condition the future-latent
   prediction.
4. **Endpoint prediction + diffusion.** We predict the chunk-**endpoint** latent
   `s_end` (a large, learnable change) rather than near-identical per-frame
   latents, and a flow-matching head decomposes `(s_0 -> s_end)` into the action
   chunk.
5. **JEPA-trained encoder.** DINO is unfrozen and trained with the V-JEPA recipe
   (online encoder + EMA target + context masking, latent prediction only), with
   BYOL-style anti-collapse — yielding a task-specific latent space.

## Method

```
                 ┌──────────── Qwen-VL (intent) ────────────┐
 instruction ───►│ language + current-frame understanding   │── action tokens ─┐
                 └──────────────────────────────────────────┘                  │
                                                                                ▼
 current frame ──► DINO encoder ──► s_0 ───────────────────────►  Mamba predictor ──► s_end
                   (JEPA-trained)                                  (latent world model)   │
                                                                                          ▼
 robot state ───────────────────────────────►  Flow-matching head (s_0, s_end, state) ──► action chunk (H=7)
```

**Training (two stages).**

| Stage | Trains | Loss | Notes |
|---|---|---|---|
| `jepa` | online DINO + Mamba predictor (+ Qwen-LoRA) | `L_pred` only | EMA target encoder, context masking (~50%), BYOL-style anti-collapse (`s0_std` monitored) |
| `stage2` | predictor (fine-tune) + flow-matching head (+ Qwen-LoRA) | `L_pred + L_action` | DINO frozen (the JEPA-learned one); head conditioned on the *predicted* `s_end` (inference-consistent) |

**Inference.** `current frame → DINO → s_0`; `(s_0, Qwen tokens) → Mamba → s_end`;
`(s_0, s_end, state) → flow-matching head → action chunk`. The world model is run
every control cycle.

## Results (LIBERO-10, 50 trials/task = 500 episodes)

LIBERO-10 is the hardest suite (long-horizon, multi-step).

| Model | Encoder | Inference WM | Success |
|---|---|---|---|
| V1 — per-frame + inverse dynamics | DINO (frozen) | ✅ | 20% |
| V2 — endpoint + diffusion | DINO (frozen) | ✅ | 84.8% |
| **V2 + Qwen-LoRA** *(best)* | DINO (frozen) | ✅ | **88.4%** |
| VLA-WM — V2 + JEPA-DINO + LoRA | DINO (JEPA-trained) | ✅ | 84.8% |
| *(ref)* base VLA-JEPA | V-JEPA | ❌ (dropped) | ~90%+ |

*Single-seed; LIBERO action sampling is non-deterministic (≈±2% at 500 episodes).*

**Takeaways.**
- **Endpoint + diffusion is the key jump** (V1 20% → V2 84.8%): predicting the
  chunk-endpoint latent (a large change) instead of near-identical per-frame
  latents makes the transition learnable.
- **Qwen-LoRA helps** (84.8% → 88.4%, +3.6%, z≈1.7): re-optimizing the action
  tokens for the Mamba/DINO space, vs. their original V-JEPA-tube target.
- **JEPA-training the encoder did *not* help here** (84.8%). The encoder fits the
  prediction task well (`pred_cos` ≈ 0.98, `s0_std` stable ~1.4 → no collapse),
  but the learned latent space did not translate into better actions for this
  single seed — a case where a better *representation* objective does not improve
  the *policy*. Whether this is a real regression or seed noise needs multi-seed
  evaluation. The best configuration remains **V2 + Qwen-LoRA (88.4%)**, reaching
  near base-VLA-JEPA quality **with a lightweight, inference-time world model**.

## Code map

| Path | What |
|---|---|
| [`starVLA/model/framework/VLA_DINO_Mamba_JEPA.py`](starVLA/model/framework/VLA_DINO_Mamba_JEPA.py) | **VLA-WM**: online DINO + EMA target + masking (JEPA stage) |
| [`starVLA/model/framework/VLA_DINO_Mamba_Diff.py`](starVLA/model/framework/VLA_DINO_Mamba_Diff.py) | V2: endpoint predictor + flow-matching head |
| [`starVLA/model/framework/VLA_DINO_Mamba.py`](starVLA/model/framework/VLA_DINO_Mamba.py) | V1: per-frame + inverse dynamics baseline |
| [`starVLA/model/modules/world_model/mamba_world_model.py`](starVLA/model/modules/world_model/mamba_world_model.py) | Mamba state encoder / predictor / inverse-dynamics head |
| [`scripts/train_mamba_wm.py`](scripts/train_mamba_wm.py) | Trainer (`--stage jepa\|predictor\|stage2`, `--qwen_lora`) |
| [`scripts/run_2stage_dino_jepa.sh`](scripts/run_2stage_dino_jepa.sh) | VLA-WM two-stage chain (jepa → stage2) |
| [`scripts/eval_libero_dino.sh`](scripts/eval_libero_dino.sh) | LIBERO eval (server + rollout) |
| [`RESEARCH.md`](RESEARCH.md) | Full design notes, decisions, failure analysis |

## Quick start

```bash
# Train VLA-WM (JEPA-DINO stage -> action stage), both with Qwen-LoRA
bash scripts/run_2stage_dino_jepa.sh

# Evaluate on LIBERO-10 (50 trials/task)
bash scripts/eval_libero_dino.sh libero_10 50 18012 \
  results/dino_mamba_jepa_libero_10/stage2/checkpoints/mamba_wm_final.pt jepa50 0
```

Backbone checkpoint, datasets, and LIBERO setup follow the base VLA-JEPA repo;
paths are configurable in the scripts.

## Acknowledgement

Built on [VLA-JEPA](https://arxiv.org/abs/2602.10098). World model uses
[Mamba](https://github.com/state-spaces/mamba); encoder is
[DINOv2](https://github.com/facebookresearch/dinov2); VLM is Qwen3-VL.
