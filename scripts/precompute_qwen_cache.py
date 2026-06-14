#!/usr/bin/env python
# Copyright 2026 VLA-JEPA research. MIT License.
"""
Precompute & cache Qwen action tokens (the per-step inference bottleneck).

Qwen is frozen, so its action tokens for a given (current-frame image, language)
are deterministic. We enumerate every training window deterministically over the
mixture's component datasets, keyed by (dataset_name, trajectory_id, base_index),
run Qwen once per window, and store the tokens in a fp16 memmap + key->row index.

Training then looks up tokens by key instead of running Qwen each step (see
VLA_DINO_Mamba.load_qwen_cache + the cache path in its forward).

Usage:
  PYTHONPATH=$(pwd) python scripts/precompute_qwen_cache.py \
    --backbone_ckpt .../VLA-JEPA-LIBERO.pt --data_mix libero_10 \
    --out results/cache/qwen_libero10 --batch_size 16
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from starVLA.model.tools import read_mode_config
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def window_image_lang(mixture, dataset, traj_id, base_index):
    """Replicate the mixture's per-step (image, lang) extraction deterministically."""
    data = dataset.transforms(dataset.get_step_data(traj_id, base_index))
    images = []
    for vk in dataset.modality_keys["video"]:
        video = data[vk]                                   # (T,H,W,C)
        video = mixture.resize_video_opencv(video, mixture.video_resolution_size)
        images.append(Image.fromarray(video[0]).resize((mixture.resolution_size, mixture.resolution_size)))
    if len(dataset.modality_keys["video"]) == 1:
        images = [images[0], images[0]]
    lang = data[dataset.modality_keys["language"][0]][0]
    return images, lang


class _EnumDataset(torch.utils.data.Dataset):
    """Parallel (DataLoader-worker) video-decode of every window -> (row, key, images, lang).
    Decode is the bottleneck; workers parallelize it while the main process batches Qwen."""

    def __init__(self, mixture, entries):
        self.mixture = mixture
        self.entries = entries          # list of (name, traj_id, base_index, dataset)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, row):
        name, traj_id, base_index, ds = self.entries[row]
        images, lang = window_image_lang(self.mixture, ds, traj_id, base_index)
        return row, f"{name}|{traj_id}|{base_index}", images, lang


def _collate(batch):
    return batch                        # keep PIL lists intact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--data_root", default="/home/choi/data/datasets/LIBERO")
    p.add_argument("--data_mix", default="libero_10")
    p.add_argument("--out", default="results/cache/qwen_libero10")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8, help="parallel video-decode workers")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="cap #windows (0=all); for quick verification")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # build model (frozen) just to reuse Qwen + the action-token machinery
    mc, _ = read_mode_config(Path(args.backbone_ckpt)); cfg = dict_to_namespace(mc)
    cfg.framework.name = "VLA_DINO_Mamba"; cfg.trainer.pretrained_checkpoint = None
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.data_mix = args.data_mix
    from omegaconf import OmegaConf
    cfg.framework.mamba_wm = OmegaConf.create({"dino_backbone": "dinov2_vitb14"})
    model = build_framework(cfg); model.load_backbone(args.backbone_ckpt); model = model.to(device).eval()

    mixture = get_vla_dataset(data_cfg=cfg.datasets.vla_data, action_horizon=model.horizon, video_horizon=16)
    components = mixture.datasets
    # deterministic enumeration of every window across components
    entries = []  # (dataset_name, traj_id, base_index)
    for ds in components:
        name = os.path.basename(str(getattr(ds, "dataset_path", ds)))
        for traj_id, base_index in ds.all_steps:
            entries.append((name, int(traj_id), int(base_index), ds))
    if args.limit:
        entries = entries[: args.limit]
    N = len(entries)
    print(f"[precompute] {N} windows across {len(components)} component(s)")

    # probe token shape on the first window
    imgs, lang = window_image_lang(mixture, entries[0][3], entries[0][1], entries[0][2])
    with torch.no_grad():
        tok0 = model._qwen_action_tokens([imgs], [lang])
    Na, H = tok0.shape[1], tok0.shape[2]
    print(f"[precompute] action token shape per window: [{Na}, {H}] -> {N*Na*H*2/1e9:.1f} GB fp16")

    mem = np.memmap(out / "qwen.dat", dtype=np.float16, mode="w+", shape=(N, Na, H))
    key_to_row = {}

    from torch.utils.data import DataLoader
    loader = DataLoader(_EnumDataset(mixture, entries), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, collate_fn=_collate)
    for batch in tqdm(loader, desc="qwen-cache"):
        rows = [b[0] for b in batch]
        imgs = [b[2] for b in batch]
        langs = [b[3] for b in batch]
        with torch.no_grad():
            toks = model._qwen_action_tokens(imgs, langs)          # [b, Na, H]
        mem[rows] = toks.to(torch.float16).cpu().numpy()
        for b in batch:
            key_to_row[b[1]] = b[0]
    mem.flush()

    meta = {"N": N, "Na": Na, "H": H, "data_mix": args.data_mix, "key_to_row": key_to_row}
    with open(out / "qwen_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[precompute] saved {out/'qwen.dat'} and meta ({N} keys)")


if __name__ == "__main__":
    main()
