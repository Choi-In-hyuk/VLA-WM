#!/usr/bin/env python
# Copyright 2026 VLA-JEPA research. MIT License.
"""
Lean single-GPU trainer for the Mamba latent world model (VLA_JEPA_Mamba).

Only the three new Mamba modules are trained; the Qwen / V-JEPA / flow-matching
backbone is loaded frozen from a pretrained VLA_JEPA checkpoint. The output dir
mirrors the eval-harness layout (config.yaml + dataset_statistics.json +
checkpoints/*.pt) so the existing LIBERO server can evaluate it directly via
`predict_action` (which uses the WM+ID inference path, V-JEPA dropped).

Example:
  python scripts/train_mamba_wm.py \
    --backbone_ckpt /home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \
    --data_root /home/choi/data/datasets/LIBERO --data_mix libero_10 \
    --output_dir results/mamba_wm_libero10 --batch_size 8 --max_steps 20000
"""
import argparse
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from starVLA.model.tools import read_mode_config
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


def build_cfg(args):
    model_config, _ = read_mode_config(Path(args.backbone_ckpt))
    cfg = dict_to_namespace(model_config)
    cfg.framework.name = args.framework
    cfg.trainer.pretrained_checkpoint = None
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.data_mix = args.data_mix
    # mamba world-model hyperparameters
    from omegaconf import OmegaConf
    cfg.framework.mamba_wm = OmegaConf.create({
        "encoder_depth": args.encoder_depth,
        "predictor_depth": args.predictor_depth,
        "idm_hidden": args.idm_hidden,
        "consist_weight": args.consist_weight,
        "dino_backbone": args.dino_backbone,
        "mask_ratio": args.mask_ratio,            # jepa: fraction of s_0 tokens hidden
        "ema_momentum": args.ema_momentum,        # jepa: EMA target-encoder momentum (->1.0)
    })
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--framework", default="VLA_DINO_Mamba",
                   help="framework name (VLA_DINO_Mamba: DINO encoder; VLA_JEPA_Mamba: V-JEPA distill)")
    p.add_argument("--dino_backbone", default="dinov2_vitb14",
                   help="DINOv2 variant for VLA_DINO_Mamba encoder")
    p.add_argument("--qwen_cache", default=None,
                   help="dir with precomputed Qwen action tokens (skips per-step Qwen forward)")
    p.add_argument("--qwen_lora", action="store_true",
                   help="fine-tune Qwen with LoRA (disables cache; Qwen runs live with grad)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--stage", choices=["encoder", "predictor", "id", "joint", "stage2", "jepa"], default="predictor",
                   help="predictor: train Mamba predictor (L_pred); id/stage2: train action head "
                        "(stage2 = VLA_DINO_Mamba_Diff: predictor fine-tune + diffusion head); "
                        "jepa: VLA_DINO_Mamba_JEPA stage-1 (online DINO + EMA target + masking, L_pred); joint: all")
    p.add_argument("--mask_ratio", type=float, default=0.5, help="jepa: fraction of s_0 tokens hidden")
    p.add_argument("--ema_momentum", type=float, default=0.996, help="jepa: EMA target momentum (cosine ->1.0)")
    p.add_argument("--resume_ckpt", default=None,
                   help="full VLA_JEPA_Mamba state_dict to continue from (e.g. stage-1 output for stage 2)")
    p.add_argument("--data_root", default="/home/choi/data/datasets/LIBERO")
    p.add_argument("--data_mix", default="libero_10")
    p.add_argument("--output_dir", default="results/mamba_wm_libero10")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--predictor_depth", type=int, default=8)
    p.add_argument("--idm_hidden", type=int, default=1024)
    p.add_argument("--consist_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}")
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)

    # ---- model: frozen backbone + trainable mamba modules
    model = build_framework(cfg)
    model.load_backbone(args.backbone_ckpt)
    if args.qwen_lora:                              # apply LoRA AFTER backbone load (key match)
        model.apply_qwen_lora(r=args.lora_r, alpha=args.lora_alpha)
        # record in cfg (AFTER build) so eval/from_pretrained re-applies LoRA before loading
        cfg.framework.mamba_wm.qwen_lora = True
        cfg.framework.mamba_wm.lora_r = args.lora_r
        cfg.framework.mamba_wm.lora_alpha = args.lora_alpha
    if args.qwen_cache and not args.qwen_lora:
        model.load_qwen_cache(args.qwen_cache)
    elif args.qwen_cache and args.qwen_lora:
        print("[trainer] --qwen_lora set: ignoring --qwen_cache (Qwen trains live)")
    if args.resume_ckpt:                           # continue from a prior stage (loads trained mamba weights)
        missing, unexpected = model.load_state_dict(torch.load(args.resume_ckpt, map_location="cpu"), strict=False)
        print(f"[trainer] resumed mamba weights from {args.resume_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)
    model.set_stage(args.stage)                    # sets requires_grad + loss weights for the stage

    model.eval()                                   # frozen backbone stays in eval
    # set train() on any top-level submodule that has trainable params (framework-agnostic)
    for _, mod in model.named_children():
        if any(pm.requires_grad for pm in mod.parameters()):
            mod.train()

    trainable = [pm for pm in model.parameters() if pm.requires_grad]
    n_tr = sum(pm.numel() for pm in trainable)
    print(f"[trainer] stage={args.stage}, trainable params: {n_tr/1e6:.1f}M")

    # ---- persist config + norm stats for the eval harness
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, out / "config.yaml")
    src_dir = Path(args.backbone_ckpt).parent.parent
    if (src_dir / "dataset_statistics.json").exists():
        shutil.copy(src_dir / "dataset_statistics.json", out / "dataset_statistics.json")

    # ---- data
    dataset = get_vla_dataset(
        # only obs_0..obs_H are used (H+1 frames); decoding fewer frames ~halves dataloader cost
        data_cfg=cfg.datasets.vla_data, action_horizon=model.horizon, video_horizon=model.horizon + 1,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=False,
    )
    print(f"[trainer] dataset {args.data_mix}: {len(dataset)} samples, {len(loader)} steps/epoch")

    # ---- optim + cosine schedule with linear warmup
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    # framework may override which params the lean intermediate save keeps (e.g. JEPA also
    # trains the online DINO encoder, so it must be saved to resume the next stage)
    mamba_prefixes = tuple(getattr(model, "save_prefixes", ("mamba_encoder", "mamba_predictor", "idm")))

    def save(tag, full=False):
        """full=True -> entire model (eval-ready, ~8.7GB). Else only the Mamba
        modules (~1.9GB), enough to resume the next curriculum stage."""
        path = out / "checkpoints" / f"mamba_wm_{tag}.pt"
        sd = model.state_dict()
        if not full:
            sd = {k: v for k, v in sd.items() if k.startswith(mamba_prefixes)}
        torch.save(sd, path)
        print(f"[trainer] saved {path} ({'full' if full else 'mamba-only'})", flush=True)

    steps_per_epoch = max(1, len(loader))
    # framework-agnostic: show the active (nonzero-weight) component losses + any
    # cosine-similarity metrics the framework reports (keys ending in "_cos").
    active = [k for k in model.loss_weights if model.loss_weights[k] > 0]
    show = [f"{k}_loss" for k in active]   # *_cos keys appended dynamically at log time

    step, t0 = 0, time.time()
    opt.zero_grad(set_to_none=True)
    running = {}
    while step < args.max_steps:
        for batch in loader:
            losses = model(batch)
            (losses["loss"] / args.grad_accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_at(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if hasattr(model, "post_step"):       # e.g. JEPA EMA target-encoder update
                    model.post_step(step, args.max_steps)

            step += 1
            if step % args.log_every == 0:
                n = args.log_every
                dt = time.time() - t0
                sps = n / dt
                pct = 100.0 * step / args.max_steps
                ep = step / steps_per_epoch
                eta_h = (args.max_steps - step) / max(sps, 1e-6) / 3600
                extra = sorted(k for k in running if k.endswith("_cos") or k.endswith("_std"))
                msg = " ".join(f"{m.replace('_loss','')}={running.get(m, 0.0)/n:.4f}" for m in show + extra)
                print(f"[{args.stage}] {step}/{args.max_steps} ({pct:4.1f}%) ep{int(ep)} "
                      f"| {msg} | lr={args.lr*lr_at(step):.1e} | {sps:.2f} it/s | ETA {eta_h:.1f}h",
                      flush=True)
                running, t0 = {}, time.time()
            if step % args.save_every == 0:
                save(f"step{step}")
            if step >= args.max_steps:
                break

    save("final", full=True)
    print("[trainer] done.", flush=True)


if __name__ == "__main__":
    main()
