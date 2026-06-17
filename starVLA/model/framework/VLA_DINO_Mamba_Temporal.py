# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_DINO_Mamba_Temporal (V3) — endpoint world model over a TEMPORAL BUFFER.

V2 feeds the Mamba predictor a single current frame, so Mamba acts only as a
spatial token mixer and the time axis is unused. V3 feeds a length-K buffer of
PAST per-frame latents [s_{t-K+1}, ..., s_t] as a time-then-space sequence, so
Mamba's causal recurrence integrates motion/trajectory across frames, and still
predicts the chunk-ENDPOINT latent s_{t+H} (V2's large-change target -> avoids
V1's per-frame small-change failure).

    frames t-6..t  --DINO--> buffer [B, 7, N, D]
    (buffer, qwen action tokens) --MambaTemporalPredictor--> s_end (= s_{t+7})
    (s_t, s_end, robot state) --FlowmatchingActionHead--> action chunk

First-action / cold-start: the buffer is left-padded by replicating the first
frame (velocity 0), matching inference where past frames are unavailable.

Encoder variants (config.framework.mamba_wm.dino_source):
    "frozen" : DINO frozen (best V2+LoRA recipe)            -> V3-a
    "jepa"   : load JEPA-trained DINO weights, kept frozen  -> V3-b
Both train the temporal predictor + flow head (+ optional Qwen-LoRA).
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLA_DINO_Mamba_Diff import VLA_DINO_Mamba_Diff
from starVLA.model.modules.world_model.mamba_world_model import MambaTemporalPredictor
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_Mamba_Temporal")
class VLA_DINO_Mamba_Temporal(VLA_DINO_Mamba_Diff):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)   # builds DINO, Qwen, cond_proj, flow head
        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)

        self.context_len = int(get("context_len", self.horizon))   # past buffer length (=7)
        dino_dim = self.dino.num_channels
        qwen_dim = self.qwen_vl_interface.model.config.hidden_size

        # replace the single-frame predictor with the temporal-buffer one
        self.mamba_predictor = MambaTemporalPredictor(
            state_dim=dino_dim, action_token_dim=qwen_dim,
            tokens_per_frame=self.tokens_per_frame, context_len=self.context_len,
            depth=get("predictor_depth", 8),
        )
        # at inference the predictor outputs s_end directly (not [B,H,N,D]); inherited
        # predict_action calls mamba_predictor(...)[:, 0], so it would break -> overridden below.
        self._inference_buffer = None   # rolling deque of past s_t latents (eval)
        # observation delta indices the trainer should request: past buffer [-(K-1)..0] + endpoint [horizon]
        self.obs_indices = list(range(-(self.context_len - 1), 1)) + [self.horizon]
        logger.info(f"[temporal] context_len={self.context_len} obs_indices={self.obs_indices} "
                    f"dino_source={get('dino_source','frozen')}")

    def load_jepa_dino(self, ckpt_path: str):
        """V3-b: load JEPA-trained online DINO weights from a JEPA checkpoint."""
        sd = torch.load(ckpt_path, map_location="cpu")
        dino_sd = {k[len("dino."):]: v for k, v in sd.items() if k.startswith("dino.")}
        missing, unexpected = self.dino.load_state_dict(dino_sd, strict=False)
        logger.info(f"[temporal] loaded JEPA DINO ({len(dino_sd)} tensors, "
                    f"missing={len(missing)}, unexpected={len(unexpected)})")
        return self

    # ------------------------------------------------------------------ forward (training)
    def forward(self, examples: List[dict] = None, **kwargs):
        device = self.cond_proj.weight.device
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)  # [B,V,T,256,256,3]
        K = self.context_len
        # obs_indices = [-(K-1)..0, horizon]: video index 0..K-1 = buffer (deltas -(K-1)..0,
        # oldest..current), index K = endpoint (delta=horizon). At episode start the negative
        # deltas are auto first-frame-padded by the dataloader (natural cold-start, matches eval).
        buf_frames = videos[:, :, :K]                         # [B,V,K,...]  past buffer
        end_frame = videos[:, :, [K]]                         # [B,V,1,...]  endpoint (delta=horizon)

        with torch.autocast("cuda", dtype=torch.float32):
            buf = self._dino_latents(buf_frames).float()      # [B,K,N,D]
            s_end_gt = self._dino_latents(end_frame).float()[:, 0]   # [B,N,D]
        s_t = buf[:, -1]                                       # current = delta-0 frame

        if self._qwen_cache is not None and "cache_key" in examples[0]:
            action_tokens = self._cached_action_tokens(examples, device)
        else:
            action_tokens = self._qwen_action_tokens(batch_images, instructions)

        s_end_pred = self.mamba_predictor(buf, action_tokens)  # [B,N,D]
        L_pred = F.l1_loss(s_end_pred, s_end_gt)
        out = {"pred_cos": F.cosine_similarity(s_end_pred, s_end_gt, dim=-1).mean()}

        w = self.loss_weights
        L_action = torch.zeros((), device=device)
        if w["action"] > 0:
            actions = torch.tensor(np.array([e["action"] for e in examples]), device=device, dtype=torch.float32)
            state = None
            if "state" in examples[0]:
                state = torch.tensor(np.array([e["state"] for e in examples]), device=device, dtype=torch.float32)
            rep = self.config.trainer.get("repeated_diffusion_steps", 4)
            cond = self._cond(s_t, s_end_pred)                 # condition on (current, predicted endpoint)
            with torch.autocast("cuda", dtype=torch.float32):
                L_action = self.action_model(cond.repeat(rep, 1, 1),
                                             actions.repeat(rep, 1, 1),
                                             state.repeat(rep, 1, 1) if state is not None else None)

        out["pred_loss"], out["action_loss"] = L_pred, L_action
        out["loss"] = w["pred"] * L_pred + w["action"] * L_action
        return out

    # ------------------------------------------------------------------ inference
    def _frames_to_tensor(self, samples, device):
        """samples: list (B) of list (V) of PIL -> [B,V,256,256,3] uint8 tensor."""
        views = []
        for sample in samples:
            vs = [torch.from_numpy(np.asarray(img.convert("RGB").resize((256, 256)))) for img in sample]
            views.append(torch.stack(vs))
        return torch.stack(views).to(device)                   # [B,V,256,256,3]

    @torch.inference_mode()
    def predict_action(self, batch_images, instructions, state=None, image_buffer=None, **kwargs):
        """image_buffer: optional list (length K, oldest..current) of batch_images-style
        frame lists, maintained per-step client-side -> a CONSECUTIVE past buffer (matches
        training). If absent, the current frame is replicated K times (cold-start)."""
        device = self.cond_proj.weight.device
        action_tokens = self._qwen_action_tokens(batch_images, instructions)
        cur = self._frames_to_tensor(batch_images, device)     # [B,V,256,256,3]

        if image_buffer is not None:
            frames = torch.stack([self._frames_to_tensor(f, device) for f in image_buffer], dim=2)  # [B,V,K,...]
        else:
            frames = cur.unsqueeze(2).expand(-1, -1, self.context_len, -1, -1, -1)  # replicate -> [B,V,K,...]
        buf = self._dino_latents(frames)                       # [B,K,N,D]
        s_t = buf[:, -1]                                        # current = last buffer frame

        s_end = self.mamba_predictor(buf, action_tokens)       # [B,N,D]
        st = None
        if state is not None:
            st = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32)
        cond = self._cond(s_t, s_end)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(cond, st)
        return {"normalized_actions": actions.float().cpu().numpy()}
