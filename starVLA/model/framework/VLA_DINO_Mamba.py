# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_DINO_Mamba — fast latent world model for inference-time action generation.

Design (consolidated after design discussion + lit positioning):
  - Encoder  : frozen DINOv2 (per-frame, in-distribution, fast) -> per-frame latents
  - Condition: Qwen-VL action tokens (intent/goal from language+current frame),
               reused frozen from the author's VLA-JEPA checkpoint. Conditioning on
               INTENT (not the real action) keeps inference non-circular.
  - Predictor: Mamba, action-token-conditioned latent rollout (V-JEPA-2-AC-style
               latent prediction objective), s0 -> s1..sH in DINO latent space.
  - Action   : learned inverse-dynamics head, ID(s_k, s_{k+1}) + robot state -> a_k.
               Replaces the slow diffusion head / MPC for real-time action chunks.

Trainable : Mamba predictor, inverse-dynamics head.
Frozen    : DINOv2, Qwen-VL (and the inherited V-JEPA / diffusion head, kept only
            as an A/B baseline on the same backbone).

Two-stage curriculum:
  predictor : train Mamba predictor (L_pred: predicted latents ~= DINO(future)).
  id        : freeze predictor, train ID head (L_idm on real latents + L_consist
              on predicted latents).
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.world_model.mamba_world_model import (
    MambaStatePredictor, InverseDynamicsHead,
)
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_Mamba")
class VLA_DINO_Mamba(VLA_JEPA):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)
        self.dino_name = get("dino_backbone", "dinov2_vitb14")

        self.dino = get_dino_model(self.dino_name)          # frozen per-frame encoder
        dino_dim = self.dino.num_channels                   # vitb14 -> 768
        num_views = 2
        self.dino_size = 224                                # patch14 -> 16x16=256 tokens
        tokens_per_view = (self.dino_size // 14) ** 2       # 256
        self.tokens_per_frame = num_views * tokens_per_view  # 512 (token-axis concat)
        self.num_views = num_views

        qwen_dim = self.qwen_vl_interface.model.config.hidden_size  # 2048
        self.horizon = self.future_action_window_size + 1   # 7

        self.mamba_predictor = MambaStatePredictor(
            state_dim=dino_dim, action_token_dim=qwen_dim,
            tokens_per_frame=self.tokens_per_frame, horizon=self.horizon,
            depth=get("predictor_depth", 8),
        )
        self.idm = InverseDynamicsHead(
            latent_dim=dino_dim,
            robot_state_dim=config.framework.action_model.state_dim,
            action_dim=config.framework.action_model.action_dim,
            hidden_dim=get("idm_hidden", 1024),
        )
        self.consist_weight = get("consist_weight", 1.0)
        self.loss_weights = {"pred": 1.0, "idm": 1.0, "consist": self.consist_weight}

        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self._qwen_cache = None   # set via load_qwen_cache() to skip per-step Qwen forward

        self.freeze_backbone()

    def load_qwen_cache(self, cache_dir: str):
        """Load precomputed Qwen action tokens (see scripts/precompute_qwen_cache.py)."""
        import json
        from pathlib import Path
        cache_dir = Path(cache_dir)
        meta = json.load(open(cache_dir / "qwen_meta.json"))
        mem = np.memmap(cache_dir / "qwen.dat", dtype=np.float16, mode="r",
                        shape=(meta["N"], meta["Na"], meta["H"]))
        self._qwen_cache = {"key_to_row": meta["key_to_row"], "mem": mem}
        logger.info(f"[qwen_cache] loaded {meta['N']} cached windows from {cache_dir}")
        return self

    def _cached_action_tokens(self, examples, device):
        c = self._qwen_cache
        rows = [c["key_to_row"][e["cache_key"]] for e in examples]
        toks = torch.from_numpy(np.ascontiguousarray(c["mem"][rows])).to(device, dtype=torch.float32)
        return toks

    # ------------------------------------------------------------------ utils
    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for m in (self.mamba_predictor, self.idm):
            for p in m.parameters():
                p.requires_grad_(True)

    def set_stage(self, stage: str):
        """predictor: train Mamba predictor (L_pred). id: freeze predictor, train ID."""
        for mod in (self.mamba_predictor, self.idm):
            for p in mod.parameters():
                p.requires_grad_(False)
        if stage == "predictor":
            train, self.loss_weights = (self.mamba_predictor,), {"pred": 1.0, "idm": 0.0, "consist": 0.0}
        elif stage == "id":
            train, self.loss_weights = (self.idm,), {"pred": 0.0, "idm": 1.0, "consist": self.consist_weight}
        elif stage == "joint":
            train, self.loss_weights = (self.mamba_predictor, self.idm), {"pred": 1.0, "idm": 1.0, "consist": self.consist_weight}
        else:
            raise ValueError(f"unknown stage: {stage}")
        for mod in train:
            for p in mod.parameters():
                p.requires_grad_(True)
        logger.info(f"[set_stage] stage={stage} loss_weights={self.loss_weights}")
        return self

    def load_backbone(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        new_prefixes = ("mamba_predictor", "idm", "dino")
        leaked = [k for k in missing if not k.startswith(new_prefixes)]
        if leaked:
            logger.warning(f"[load_backbone] non-new missing keys: {leaked[:8]} ...")
        if unexpected:
            logger.warning(f"[load_backbone] unexpected keys: {list(unexpected)[:8]} ...")
        logger.info(f"[load_backbone] loaded backbone; {len(missing)} new params fresh")
        self.freeze_backbone()
        return self

    @torch.no_grad()
    def _dino_frame_latents(self, videos: torch.Tensor) -> torch.Tensor:
        """videos: [B, V, T, H, W, 3] uint8 -> per-frame DINO latents [B, T, V*tok, dim]."""
        B, V, T, H, W, _ = videos.shape
        x = videos.float() / 255.0
        x = x.permute(0, 1, 2, 5, 3, 4).reshape(B * V * T, 3, H, W)
        x = F.interpolate(x, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dino(x)                                # [B*V*T, tok, dim]
        tok, dim = feats.shape[1], feats.shape[2]
        feats = feats.reshape(B, V, T, tok, dim).permute(0, 2, 1, 3, 4).reshape(B, T, V * tok, dim)
        return feats                                        # [B, T, V*tok, dim]

    @torch.no_grad()
    def _qwen_action_tokens(self, batch_images, instructions) -> torch.Tensor:
        replace = {"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt}
        template = self.config.datasets.vla_data.get("CoT_prompt", "")
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
            prompt_replace_dict=replace, prompt_template=template)
        ids = qwen_inputs["input_ids"]
        mask = torch.isin(ids, torch.tensor(self.action_token_ids, device=ids.device))
        idx = mask.nonzero(as_tuple=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1]
        B, _, Hd = last.shape
        return last[idx[0], idx[1], :].view(B, -1, Hd).float()

    # ---------------------------------------------------------------- forward
    def forward(self, examples: List[dict] = None, **kwargs):
        device = self.mamba_predictor.cur_pos.device
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)  # [B,V,16,256,256,3]
        Hh = self.horizon
        frames = videos[:, :, : Hh + 1]                     # obs_0..obs_H

        s_gt = self._dino_frame_latents(frames).float()     # [B,H+1,N,D]
        actions = torch.tensor(np.array([e["action"] for e in examples]), device=device, dtype=torch.float32)
        state = None
        if "state" in examples[0]:
            state = torch.tensor(np.array([e["state"] for e in examples]), device=device, dtype=torch.float32)
            state = state.reshape(state.shape[0], -1)

        w = self.loss_weights
        zero = torch.zeros((), device=device)
        out = {}

        need_pred = w["pred"] > 0 or w["consist"] > 0
        if need_pred:
            if self._qwen_cache is not None and "cache_key" in examples[0]:
                action_tokens = self._cached_action_tokens(examples, device)      # [B,Na,2048]
            else:
                action_tokens = self._qwen_action_tokens(batch_images, instructions)
            pred = self.mamba_predictor(s_gt[:, 0], action_tokens)                # [B,H,N,D]
            L_pred = F.l1_loss(pred, s_gt[:, 1:])
            out["pred_cos"] = F.cosine_similarity(pred, s_gt[:, 1:], dim=-1).mean()
        else:
            L_pred, pred = zero, None

        L_idm = F.mse_loss(self.idm(s_gt, state), actions) if w["idm"] > 0 else zero
        if w["consist"] > 0:
            traj = torch.cat([s_gt[:, :1], pred], dim=1)    # [B,H+1,N,D]
            L_consist = F.mse_loss(self.idm(traj, state), actions)
        else:
            L_consist = zero

        comps = {"pred": L_pred, "idm": L_idm, "consist": L_consist}
        out["loss"] = sum(w[k] * comps[k] for k in comps)
        out.update({f"{k}_loss": v for k, v in comps.items()})
        return out

    # -------------------------------------------------------------- inference
    @torch.inference_mode()
    def predict_action(self, batch_images: List[List[Image.Image]], instructions: List[str],
                       state: Optional[np.ndarray] = None, **kwargs) -> dict:
        device = self.mamba_predictor.cur_pos.device
        action_tokens = self._qwen_action_tokens(batch_images, instructions)

        views = []
        for sample in batch_images:
            vs = [torch.from_numpy(np.asarray(img.convert("RGB"))) for img in sample]
            views.append(torch.stack(vs))                   # [V,H,W,3]
        frame0 = torch.stack(views).unsqueeze(2).to(device)  # [B,V,1,H,W,3]
        s0 = self._dino_frame_latents(frame0)[:, 0]          # [B,N,D]

        st = None
        if state is not None:
            st = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32).reshape(len(batch_images), -1)

        future = self.mamba_predictor(s0, action_tokens)     # [B,H,N,D]
        traj = torch.cat([s0.unsqueeze(1), future], dim=1)   # [B,H+1,N,D]
        actions = self.idm(traj, st)                         # [B,H,7]
        return {"normalized_actions": actions.float().cpu().numpy()}
