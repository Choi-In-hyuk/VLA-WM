# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_JEPA_Mamba — inference-time world-model action generation.

Extends VLA_JEPA by adding a Mamba latent world model that is actually used at
inference to produce actions (the original world model is only a training-time
aux loss). The heavy V-JEPA video encoder / ViT predictor are used ONLY as a
frozen teacher during training and dropped at inference; a fast Mamba encoder +
predictor reproduce the per-frame latent space, and an inverse-dynamics head
decodes latent transitions into the action chunk.

Trainable : MambaStateEncoder, MambaStatePredictor, InverseDynamicsHead
Frozen    : Qwen-VL backbone, V-JEPA encoder (teacher), V-JEPA ViT predictor,
            flow-matching action head (kept as A/B baseline on the same backbone)

Training losses (per-frame V-JEPA latents s_k = VJEPA(frame_k), k=0..H):
    L_distill : mamba_enc(frame_0)            ~= s_0
    L_pred    : mamba_pred(s_0, action_tokens) ~= [s_1..s_H]
    L_id      : idm([s_0..s_H])                ~= GT actions
    L_consist : idm([enc(frame_0), pred...])   ~= GT actions
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.VLA_JEPA import VLA_JEPA, IGNORE_INDEX
from starVLA.model.modules.world_model.mamba_world_model import (
    MambaStateEncoder, MambaStatePredictor, InverseDynamicsHead,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_JEPA_Mamba")
class VLA_JEPA_Mamba(VLA_JEPA):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        vj_dim = self.vj_encoder.config.hidden_size           # 1024
        num_views = 2
        state_dim = vj_dim * num_views                        # 2048
        img_size = self.vj_encoder.config.image_size          # 256
        tokens_per_frame = (img_size // 16) ** 2              # 256
        qwen_dim = self.qwen_vl_interface.model.config.hidden_size  # 2048
        self.horizon = self.future_action_window_size + 1    # 7

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)

        self.mamba_encoder = MambaStateEncoder(
            img_size=img_size, patch_size=16, num_views=num_views,
            dim_per_view=vj_dim, depth=get("encoder_depth", 6),
        )
        self.mamba_predictor = MambaStatePredictor(
            state_dim=state_dim, action_token_dim=qwen_dim,
            tokens_per_frame=tokens_per_frame, horizon=self.horizon,
            depth=get("predictor_depth", 8),
        )
        self.idm = InverseDynamicsHead(
            latent_dim=state_dim,
            robot_state_dim=config.framework.action_model.state_dim,
            action_dim=config.framework.action_model.action_dim,
            hidden_dim=get("idm_hidden", 1024),
        )
        self.consist_weight = get("consist_weight", 1.0)
        # default: joint objective; staged training overrides via set_stage()
        self.loss_weights = {"distill": 1.0, "pred": 1.0, "id": 1.0, "consist": self.consist_weight}

        # V-JEPA pixel normalization (ImageNet); applied manually for batched per-frame encoding
        self.register_buffer("vj_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("vj_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.freeze_backbone()

    # ------------------------------------------------------------------ utils
    def freeze_backbone(self):
        """Freeze everything except the three new Mamba modules."""
        trainable = (self.mamba_encoder, self.mamba_predictor, self.idm)
        for p in self.parameters():
            p.requires_grad_(False)
        for m in trainable:
            for p in m.parameters():
                p.requires_grad_(True)

    def set_stage(self, stage: str):
        """Configure staged (curriculum) training.

        encoder   : train ONLY the Mamba encoder (pixel -> V-JEPA latent) with the
                    distill loss. Grounds perception into the latent space first.
        predictor : freeze the encoder, train ONLY the Mamba predictor on the
                    encoder's OUTPUT latents -> future latents (pred loss). Trains
                    on the same latent distribution seen at inference.
        id        : freeze encoder+predictor, train ONLY the inverse-dynamics head
                    (id + consist losses) on top of the stable latents.
        joint     : train all three modules with all four losses.
        """
        for mod in (self.mamba_encoder, self.mamba_predictor, self.idm):
            for p in mod.parameters():
                p.requires_grad_(False)
        if stage == "encoder":
            train, self.loss_weights = (self.mamba_encoder,), \
                {"distill": 1.0, "pred": 0.0, "id": 0.0, "consist": 0.0}
        elif stage == "predictor":
            train, self.loss_weights = (self.mamba_predictor,), \
                {"distill": 0.0, "pred": 1.0, "id": 0.0, "consist": 0.0}
        elif stage == "id":
            train, self.loss_weights = (self.idm,), \
                {"distill": 0.0, "pred": 0.0, "id": 1.0, "consist": self.consist_weight}
        elif stage == "joint":
            train, self.loss_weights = (self.mamba_encoder, self.mamba_predictor, self.idm), \
                {"distill": 1.0, "pred": 1.0, "id": 1.0, "consist": self.consist_weight}
        else:
            raise ValueError(f"unknown stage: {stage}")
        for mod in train:
            for p in mod.parameters():
                p.requires_grad_(True)
        logger.info(f"[set_stage] stage={stage} loss_weights={self.loss_weights}")
        return self

    def load_backbone(self, ckpt_path: str):
        """Load a pretrained VLA_JEPA checkpoint into the inherited submodules.

        strict=False: missing keys are exactly the new Mamba modules; there
        should be no unexpected keys.
        """
        sd = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        mamba_prefixes = ("mamba_encoder", "mamba_predictor", "idm")
        leaked = [k for k in missing if not k.startswith(mamba_prefixes)]
        if leaked:
            logger.warning(f"[load_backbone] non-mamba missing keys: {leaked[:8]} ...")
        if unexpected:
            logger.warning(f"[load_backbone] unexpected keys: {list(unexpected)[:8]} ...")
        logger.info(f"[load_backbone] loaded backbone, {len(missing)} mamba params freshly initialized")
        self.freeze_backbone()
        return self

    def _normalize_frame(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """frames_uint8: [..., H, W, 3] uint8/float -> [..., 3, H, W] normalized."""
        x = frames_uint8.float() / 255.0
        x = x.permute(*range(x.dim() - 3), x.dim() - 1, x.dim() - 3, x.dim() - 2)  # HWC->CHW
        shape = x.shape
        x = x.reshape(-1, 3, shape[-2], shape[-1])
        x = (x - self.vj_mean) / self.vj_std
        return x.reshape(*shape)

    @torch.no_grad()
    def _vjepa_frame_latents(self, videos: torch.Tensor) -> torch.Tensor:
        """Per-frame static V-JEPA latents (frozen teacher).

        videos: [B, V, T, H, W, 3] uint8 -> states_gt: [B, T, tokens, V*dim].
        Each frame is encoded as a minimal 2-frame static clip and batched.
        """
        B, V, T, H, W, _ = videos.shape
        x = self._normalize_frame(videos)                     # [B,V,T,3,H,W]
        x = x.reshape(B * V * T, 1, 3, H, W).repeat(1, 2, 1, 1, 1)  # 2-frame static clip
        feats = self.vj_encoder.get_vision_features(pixel_values_videos=x)  # [B*V*T, tokens, dim]
        tok, dim = feats.shape[1], feats.shape[2]
        feats = feats.reshape(B, V, T, tok, dim).permute(0, 2, 3, 1, 4).reshape(B, T, tok, V * dim)
        return feats

    @torch.no_grad()
    def _qwen_action_tokens(self, batch_images, instructions, with_action: bool) -> torch.Tensor:
        """Run frozen Qwen-VL and gather the `<|action_i|>` hidden states. -> [B, Na, H]."""
        replace = {"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt}
        template = (self.config.datasets.vla_data.get("CoT_prompt", "") if with_action
                    else self.config.datasets.video_data.get("CoT_prompt", ""))
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
            prompt_replace_dict=replace, prompt_template=template)
        ids = qwen_inputs["input_ids"]
        action_mask = torch.isin(ids, torch.tensor(self.action_token_ids, device=ids.device))
        idx = action_mask.nonzero(as_tuple=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1]
        B, _, Hd = last.shape
        return last[idx[0], idx[1], :].view(B, -1, Hd).float()

    # ---------------------------------------------------------------- forward
    def forward(self, examples: List[dict] = None, **kwargs):
        device = self.mamba_encoder.pos_embed.device
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        videos = np.stack([e["video"] for e in examples])     # [B,V,16,256,256,3]
        videos = torch.from_numpy(videos).to(device)
        Hh = self.horizon

        w = self.loss_weights
        need_pred = w["pred"] > 0 or w["consist"] > 0
        need_future_gt = need_pred or w["id"] > 0             # future-frame targets needed?
        zero = torch.zeros((), device=device)
        out = {}

        # Only encode as many V-JEPA frames as the active losses require
        # (encoder stage needs just obs_0 -> big V-JEPA / Qwen savings).
        n_frames = (Hh + 1) if need_future_gt else 1
        frames = videos[:, :, :n_frames]                      # [B,V,n_frames,256,256,3]
        states_gt = self._vjepa_frame_latents(frames).float()  # [B,n_frames,256,2048]

        # --- encoder: pixel -> latent (distill); cos-sim is the encoding-quality metric
        cur = self._normalize_frame(frames[:, :, 0])          # [B,V,3,256,256]
        s0_pred = self.mamba_encoder(cur)                     # [B,256,2048]
        L_distill = F.l1_loss(s0_pred, states_gt[:, 0])
        out["enc_cos"] = F.cosine_similarity(s0_pred, states_gt[:, 0], dim=-1).mean()

        actions = torch.tensor(np.array([e["action"] for e in examples]), device=device, dtype=torch.float32)  # [B,H,7]
        state = None
        if "state" in examples[0]:
            state = torch.tensor(np.array([e["state"] for e in examples]), device=device, dtype=torch.float32)
            state = state.reshape(state.shape[0], -1)         # [B, state_dim]

        # --- predictor: encoder-output latent -> future latents (matches inference)
        if need_pred:
            action_tokens = self._qwen_action_tokens(batch_images, instructions, with_action=True)  # [B,Na,2048]
            pred_future = self.mamba_predictor(s0_pred, action_tokens)  # [B,H,256,2048]
            L_pred = F.l1_loss(pred_future, states_gt[:, 1:])
            out["pred_cos"] = F.cosine_similarity(pred_future, states_gt[:, 1:], dim=-1).mean()
        else:
            L_pred, pred_future = zero, None

        # --- inverse dynamics
        L_id = F.mse_loss(self.idm(states_gt, state), actions) if w["id"] > 0 else zero
        if w["consist"] > 0:
            traj_pred = torch.cat([s0_pred.unsqueeze(1), pred_future], dim=1)  # [B,H+1,256,2048]
            L_consist = F.mse_loss(self.idm(traj_pred, state), actions)
        else:
            L_consist = zero

        comps = {"distill": L_distill, "pred": L_pred, "id": L_id, "consist": L_consist}
        total = sum(w[k] * comps[k] for k in comps)
        out["loss"] = total
        out.update({f"{k}_loss": v for k, v in comps.items()})
        return out

    # -------------------------------------------------------------- inference
    @torch.inference_mode()
    def predict_action(self, batch_images: List[List[Image.Image]], instructions: List[str],
                       state: Optional[np.ndarray] = None, **kwargs) -> dict:
        """WM+ID action generation (V-JEPA NOT used)."""
        device = self.mamba_encoder.pos_embed.device
        action_tokens = self._qwen_action_tokens(batch_images, instructions, with_action=True)

        # build current-frame tensor (resize obs views to V-JEPA resolution)
        size = self.vj_encoder.config.image_size
        views = []
        for sample in batch_images:
            vs = [torch.from_numpy(np.asarray(img.convert("RGB").resize((size, size)))) for img in sample]
            views.append(torch.stack(vs))                     # [V,256,256,3]
        frame0 = torch.stack(views).to(device)                # [B,V,256,256,3]
        cur = self._normalize_frame(frame0)                   # [B,V,3,256,256]

        st = None
        if state is not None:
            st = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32).reshape(len(batch_images), -1)

        s0 = self.mamba_encoder(cur)
        future = self.mamba_predictor(s0, action_tokens)
        traj = torch.cat([s0.unsqueeze(1), future], dim=1)    # [B,H+1,256,2048]
        actions = self.idm(traj, st)                          # [B,H,7]
        return {"normalized_actions": actions.float().cpu().numpy()}
