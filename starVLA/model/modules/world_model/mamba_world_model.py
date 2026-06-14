# Copyright 2026 VLA-JEPA research. MIT License.
"""
Mamba-based latent world model for VLA-JEPA inference-time action generation.

Motivation
----------
In the original VLA-JEPA, the V-JEPA encoder + ViT predictor act only as a
training-time auxiliary loss; at inference actions come from the flow-matching
head and the world model is never run. This module makes the world model usable
*at inference* and fast enough for real-time control by replacing the heavy
V-JEPA video encoder / ViT predictor with light-weight Mamba (SSM) modules.

Design (single-observation, no tubelet)
---------------------------------------
At deployment only the *current* observation exists, so these modules never
consume video / tubelet stacks. V-JEPA's temporal axis is dropped; we distill
only its per-frame latent space and let Mamba own the time evolution:

    img_t                      --MambaStateEncoder-->  s_t        [B, 256, 2048]
    (s_t, qwen_action_tokens)  --MambaStatePredictor-> s_{t+1..H} [B, H, 256, 2048]
    (s_t, s_{t+1})             --InverseDynamicsHead-> a_t        [B, 7]

Targets during training (frozen V-JEPA as teacher):
    MambaStateEncoder   : s_t        ~= VJEPA_enc(frame_t)          (per-frame distill)
    MambaStatePredictor : s_{t+k}    ~= VJEPA_enc(frame_{t+k})      (latent forward model)
    InverseDynamicsHead : a_t        ~= ground-truth action          (inverse dynamics)

Measured interface (LIBERO ckpt): per-frame latent = [256 spatial tokens, 2048]
(V-JEPA vitl hidden 1024, 2 views concat = 2048); Qwen3-VL-2B hidden = 2048;
action chunk H = 7; action_dim = 7.
"""
from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual. Optionally bidirectional.

    Bidirectional mode runs a second Mamba over the time-reversed sequence and
    sums the two passes -- appropriate for spatial (non-causal) token sets such
    as a single frame's patch tokens. Unidirectional (causal) mode is used for
    temporal roll-out where future must not leak into the past.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.bidirectional = bidirectional
        self.fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) \
            if bidirectional else None
        self.mlp = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, expand * dim), nn.GELU(), nn.Linear(expand * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.fwd(h)
        if self.bwd is not None:
            out = out + self.bwd(h.flip(1)).flip(1)
        x = x + out
        x = x + self.mlp(x)
        return x


class MambaStateEncoder(nn.Module):
    """Encode a single (multi-view) observation into a V-JEPA-style latent state.

    Replaces the frozen V-JEPA video encoder. Each view is patch-embedded
    (16x16) into `tokens_per_frame` tokens of `dim_per_view`, processed by
    bidirectional Mamba, then views are concatenated on the feature axis to
    match the V-JEPA multi-view latent [tokens_per_frame, num_views*dim_per_view].
    """

    def __init__(self, img_size: int = 256, patch_size: int = 16, num_views: int = 2,
                 dim_per_view: int = 1024, depth: int = 6, **mamba_kwargs):
        super().__init__()
        self.num_views = num_views
        self.dim_per_view = dim_per_view
        self.tokens_per_frame = (img_size // patch_size) ** 2
        self.state_dim = num_views * dim_per_view

        self.patch_embed = nn.Conv2d(3, dim_per_view, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_frame, dim_per_view))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [MambaBlock(dim_per_view, bidirectional=True, **mamba_kwargs) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim_per_view)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, V, 3, H, W] -> state: [B, tokens_per_frame, V*dim_per_view]."""
        B, V, C, H, W = images.shape
        assert V == self.num_views, f"expected {self.num_views} views, got {V}"
        x = images.reshape(B * V, C, H, W)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B*V, tokens, dim]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.reshape(B, V, self.tokens_per_frame, self.dim_per_view)
        x = x.permute(0, 2, 1, 3).reshape(B, self.tokens_per_frame, V * self.dim_per_view)
        return x  # [B, 256, 2048]


class MambaStatePredictor(nn.Module):
    """Action-conditioned latent forward model (replaces V-JEPA ViT predictor).

    Given the current latent state and Qwen action tokens, autoregressively
    rolls out H future latent states in a single causal Mamba pass. Construct a
    sequence [action_tokens | current_state | query_1 .. query_H] and read out
    the per-step query blocks; Mamba's causal structure ensures query_k only
    attends to action_tokens, the current state, and earlier queries.
    """

    def __init__(self, state_dim: int = 2048, action_token_dim: int = 2048,
                 tokens_per_frame: int = 256, horizon: int = 7, depth: int = 8,
                 **mamba_kwargs):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.horizon = horizon
        self.state_dim = state_dim

        self.action_proj = nn.Linear(action_token_dim, state_dim)
        self.cur_pos = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        # shared spatial query + per-step temporal embedding
        self.query = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        self.step_embed = nn.Parameter(torch.zeros(1, horizon, 1, state_dim))
        for p in (self.cur_pos, self.query, self.step_embed):
            nn.init.trunc_normal_(p, std=0.02)

        self.blocks = nn.ModuleList(
            [MambaBlock(state_dim, bidirectional=False, **mamba_kwargs) for _ in range(depth)]
        )
        self.norm = RMSNorm(state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)

    def forward(self, state: torch.Tensor, action_tokens: torch.Tensor,
                vis_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """state: [B, N, D], action_tokens: [B, Na, Da] -> [B, H, N, D].

        vis_idx: optional [B, Nv] long indices selecting which of the N current-state
        tokens are VISIBLE to the predictor (I-JEPA context masking). When given,
        `state` is the full [B, N, D] latent; only the Nv selected tokens (with their
        matching positional embeddings) enter the Mamba sequence, but all H*N future
        queries are still predicted. None -> full context (inference / stage2).
        """
        B, N, D = state.shape
        a = self.action_proj(action_tokens)                       # [B, Na, D]
        cur = state + self.cur_pos                                 # [B, N, D]
        if vis_idx is not None:                                    # keep only visible context tokens
            cur = torch.gather(cur, 1, vis_idx.unsqueeze(-1).expand(-1, -1, D))
        Nc = cur.shape[1]
        q = self.query + self.step_embed                          # [1, H, N, D]
        q = q.expand(B, -1, -1, -1).reshape(B, self.horizon * N, D)
        seq = torch.cat([a, cur, q], dim=1)                       # [B, Na+Nc+H*N, D]
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        q_out = seq[:, a.shape[1] + Nc:, :]                       # [B, H*N, D]
        q_out = self.out_proj(q_out).reshape(B, self.horizon, N, D)
        return q_out


class InverseDynamicsHead(nn.Module):
    """Recover the action that drives one latent transition s_t -> s_{t+1}.

    Spatially pools each latent and feeds [pool(s_t), pool(s_{t+1}), their diff]
    plus an embedding of the current robot proprioceptive state to an MLP. The
    (single, current) robot state is broadcast across the whole action chunk —
    at inference only the current state is known; per-step change is carried by
    the latent transition.
    """

    def __init__(self, latent_dim: int = 2048, robot_state_dim: int = 8,
                 action_dim: int = 7, hidden_dim: int = 1024, state_embed_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.robot_state_dim = robot_state_dim
        self.state_encoder = (
            nn.Sequential(nn.Linear(robot_state_dim, state_embed_dim), nn.GELU())
            if robot_state_dim else None
        )
        in_dim = 3 * latent_dim + (state_embed_dim if robot_state_dim else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _encode_state(self, ref: torch.Tensor, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """ref: tensor whose dim0 is batch. state None -> zeros (eval without state)."""
        if self.state_encoder is None:
            return None
        if state is None:
            state = ref.new_zeros(ref.shape[0], self.robot_state_dim)
        return self.state_encoder(state)                          # [B, state_embed_dim]

    def _pair(self, s_t: torch.Tensor, s_tp1: torch.Tensor,
              state_emb: Optional[torch.Tensor]) -> torch.Tensor:
        p0, p1 = s_t.mean(dim=-2), s_tp1.mean(dim=-2)             # [..., D]
        feat = torch.cat([p0, p1, p1 - p0], dim=-1)
        if state_emb is not None:
            feat = torch.cat([feat, state_emb], dim=-1)
        return self.net(feat)

    def forward(self, states: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """states: [B, H+1, N, D], state: [B, robot_state_dim] -> [B, H, action_dim]."""
        s_t, s_tp1 = states[:, :-1], states[:, 1:]
        emb = self._encode_state(s_t, state)
        if emb is not None:
            emb = emb.unsqueeze(1).expand(-1, s_t.shape[1], -1)  # broadcast over chunk
        return self._pair(s_t, s_tp1, emb)

    def single(self, s_t: torch.Tensor, s_tp1: torch.Tensor,
               state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """s_t, s_tp1: [B, N, D] -> [B, action_dim]."""
        return self._pair(s_t, s_tp1, self._encode_state(s_t, state))


if __name__ == "__main__":
    dev = "cuda"
    B, V, N, D, H = 2, 2, 256, 2048, 7
    enc = MambaStateEncoder().to(dev)
    pred = MambaStatePredictor().to(dev)
    idm = InverseDynamicsHead().to(dev)

    imgs = torch.randn(B, V, 3, 256, 256, device=dev)
    action_tokens = torch.randn(B, 24, 2048, device=dev)
    robot_state = torch.randn(B, 8, device=dev)

    s_t = enc(imgs)
    print("encoder out:", tuple(s_t.shape))                       # [2, 256, 2048]
    future = pred(s_t, action_tokens)
    print("predictor out:", tuple(future.shape))                 # [2, 7, 256, 2048]
    traj = torch.cat([s_t.unsqueeze(1), future], dim=1)          # [2, 8, 256, 2048]
    actions = idm(traj, robot_state)
    print("inverse-dynamics out:", tuple(actions.shape))         # [2, 7, 7]
    print("idm without state (zeros):", tuple(idm(traj).shape))

    n = sum(p.numel() for p in list(enc.parameters()) + list(pred.parameters()) + list(idm.parameters()))
    print(f"total new params: {n/1e6:.1f}M")
