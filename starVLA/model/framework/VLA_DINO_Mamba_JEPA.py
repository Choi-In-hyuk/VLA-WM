# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_DINO_Mamba_JEPA (#2) — V2 with the DINO encoder UNFROZEN and trained the
V-JEPA way (online encoder + EMA target + context masking + latent prediction).

Motivation
----------
V2 (VLA_DINO_Mamba_Diff) keeps DINO frozen, so the predictor distills a fixed
latent space. Here we LEARN a task-specific latent space with the JEPA recipe:
the encoder must produce a representation of the *present* from which the
predictor can forecast the *future* — a representation shaped by what actually
matters for the action, not generic DINO features.

JEPA stage (`set_stage("jepa")`, L_pred only — no action anchor, like the
VLA-JEPA world-model pretraining):

    frame0 --online DINO(trained)----------> s_0            [B, N, D]
    (random ~mask_ratio of s_0 tokens hidden)               context masking
    frameH --EMA DINO(stop-grad)-----------> s_end_target   [B, N, D] (LayerNorm'd)
    (visible s_0 + qwen action tokens) --Mamba--> s_end_pred [B, N, D] (all N)
    L_pred = smooth_l1(s_end_pred, s_end_target.detach())

Anti-collapse = BYOL-style: EMA target encoder + predictor asymmetry + stop-grad,
with context masking making the prediction non-trivial. The EMA encoder is updated
after every optimizer step (`post_step`) with a cosine momentum schedule 0.996->1.0.
A `s0_std` collapse monitor is logged.

Stage 2 (`set_stage("stage2")`): FREEZE the now-learned online DINO and fall back
to V2's exact stage-2 (predictor fine-tune + diffusion head, L_pred + L_action,
full context). Inference is identical to V2 (full context). Only the encoder
weights differ — they are the JEPA-trained ones.
"""
import copy
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLA_DINO_Mamba_Diff import VLA_DINO_Mamba_Diff
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_Mamba_JEPA")
class VLA_DINO_Mamba_JEPA(VLA_DINO_Mamba_Diff):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)
        self.mask_ratio = float(get("mask_ratio", 0.5))      # fraction of s_0 tokens hidden (jepa)
        self.ema_base = float(get("ema_momentum", 0.996))    # EMA start; -> 1.0 over training
        self.dino_ema = None                                 # built lazily on set_stage("jepa")
        self._stage = "stage2"
        # intermediate trainer saves must keep the (trained) online DINO weights too
        self.save_prefixes = ("mamba_encoder", "mamba_predictor", "idm", "cond_proj", "dino.")

    # ------------------------------------------------------------------ EMA target encoder
    def _build_ema(self):
        self.dino_ema = copy.deepcopy(self.dino)
        for p in self.dino_ema.parameters():
            p.requires_grad_(False)
        self.dino_ema.eval()
        logger.info("[jepa] built EMA target DINO (frozen copy of online DINO)")

    @torch.no_grad()
    def post_step(self, step: int, max_steps: int):
        """Called by the trainer after each optimizer step. EMA-update the target
        encoder (jepa stage only) with a cosine momentum schedule base->1.0."""
        if self._stage != "jepa" or self.dino_ema is None:
            return
        prog = min(1.0, step / max(1, max_steps))
        m = 1.0 - (1.0 - self.ema_base) * 0.5 * (1 + math.cos(math.pi * prog))  # ema_base -> 1.0
        for pe, po in zip(self.dino_ema.parameters(), self.dino.parameters()):
            pe.mul_(m).add_(po.detach(), alpha=1.0 - m)
        for be, bo in zip(self.dino_ema.buffers(), self.dino.buffers()):
            be.copy_(bo)

    # ------------------------------------------------------------------ stages
    def set_stage(self, stage: str):
        self._stage = stage
        if stage == "jepa":
            for p in self.parameters():
                p.requires_grad_(False)
            for m in (self.dino, self.mamba_predictor):       # train encoder + predictor
                for p in m.parameters():
                    p.requires_grad_(True)
            self.dino.train()
            if self.dino_ema is None:
                self._build_ema()
            self.loss_weights = {"pred": 1.0, "action": 0.0}
            self._set_qwen_lora_trainable(True)
            logger.info(f"[set_stage] stage=jepa (online DINO + predictor, L_pred only) "
                        f"mask_ratio={self.mask_ratio} ema_base={self.ema_base}")
            return self
        # stage2 / predictor: freeze the (learned) DINO, defer to V2's stage logic
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)
        return super().set_stage(stage)

    # ------------------------------------------------------------------ grad-enabled encoder
    def _dino_latents_grad(self, frames):
        """Same as the frozen _dino_latents but WITHOUT no_grad (online encoder)."""
        B, V, T, H, W, _ = frames.shape
        x = frames.float() / 255.0
        x = x.permute(0, 1, 2, 5, 3, 4).reshape(B * V * T, 3, H, W)
        x = F.interpolate(x, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dino(x)
        tok, dim = feats.shape[1], feats.shape[2]
        return feats.reshape(B, V, T, tok, dim).permute(0, 2, 1, 3, 4).reshape(B, T, V * tok, dim)

    @torch.no_grad()
    def _dino_ema_latents(self, frames):
        """EMA target encoder on frameH -> [B, N, D] (single frame in)."""
        B, V, T, H, W, _ = frames.shape
        x = frames.float() / 255.0
        x = x.permute(0, 1, 2, 5, 3, 4).reshape(B * V * T, 3, H, W)
        x = F.interpolate(x, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dino_ema(x)
        tok, dim = feats.shape[1], feats.shape[2]
        return feats.reshape(B, V, T, tok, dim).permute(0, 2, 1, 3, 4).reshape(B, T, V * tok, dim)

    # ------------------------------------------------------------------ forward
    def forward(self, examples: List[dict] = None, **kwargs):
        if self._stage != "jepa":
            return super().forward(examples=examples, **kwargs)   # V2 path (frozen learned DINO)

        device = self.cond_proj.weight.device
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)  # [B,V,T,256,256,3]
        frame0 = videos[:, :, [0]]                              # [B,V,1,...]
        frameH = videos[:, :, [self.endpoint]]

        with torch.autocast("cuda", dtype=torch.float32):
            s_0 = self._dino_latents_grad(frame0)[:, 0]        # [B,N,D]  online, grad
            s_end_tgt = self._dino_ema_latents(frameH)[:, 0]   # [B,N,D]  EMA, no grad
        B, N, D = s_0.shape

        # context masking: keep a random Nv-subset of s_0 tokens visible to the predictor
        Nv = max(1, int(round(N * (1.0 - self.mask_ratio))))
        vis_idx = torch.argsort(torch.rand(B, N, device=device), dim=1)[:, :Nv]  # [B,Nv]

        if self._qwen_cache is not None and "cache_key" in examples[0]:
            action_tokens = self._cached_action_tokens(examples, device)
        else:
            action_tokens = self._qwen_action_tokens(batch_images, instructions)

        s_end_pred = self.mamba_predictor(s_0, action_tokens, vis_idx=vis_idx)[:, 0]  # [B,N,D]

        # predict LayerNorm'd target (V-JEPA practice): scale-stable, aids anti-collapse
        tgt = F.layer_norm(s_end_tgt, (D,)).detach()
        L_pred = F.smooth_l1_loss(s_end_pred, tgt)

        with torch.no_grad():
            s0_std = s_0.float().std(dim=0).mean()             # collapse monitor (->0 = collapsing)
        out = {
            "pred_loss": L_pred,
            "loss": L_pred,
            "pred_cos": F.cosine_similarity(s_end_pred, tgt, dim=-1).mean(),
            "s0_std": s0_std,
        }
        return out
