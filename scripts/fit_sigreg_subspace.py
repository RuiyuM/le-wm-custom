#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils import build_dataset_and_splits, load_world_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a fixed PCA subspace for rank-SIGReg from a warmup checkpoint."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("weights", "object"),
        default="weights",
        help="Checkpoint type. Use 'weights' for lewm_weights.ckpt, 'object' for *_object.ckpt.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--subset-seed", type=int, default=None)
    parser.add_argument("--subset-indices-file", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run encoding on. Defaults to cuda if available, else cpu.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)

    with open_dict(cfg):
        if args.subset_indices_file is not None:
            cfg.subset = cfg.get("subset") or {}
            cfg.subset.indices_file = str(args.subset_indices_file.expanduser().resolve())
            cfg.subset.fraction = None
        elif args.subset_fraction is not None:
            cfg.subset = cfg.get("subset") or {}
            cfg.subset.fraction = float(args.subset_fraction)
            cfg.subset.seed = (
                int(args.subset_seed) if args.subset_seed is not None else int(cfg.seed)
            )
            cfg.subset.indices_file = None
        elif args.subset_seed is not None and cfg.get("subset"):
            cfg.subset.seed = int(args.subset_seed)

    _, train_set, _, selected_episodes, _ = build_dataset_and_splits(cfg)

    loader_cfg = dict(cfg.loader)
    if args.batch_size is not None:
        loader_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        loader_cfg["num_workers"] = int(args.num_workers)
    if int(loader_cfg.get("num_workers", 0)) == 0:
        loader_cfg.pop("prefetch_factor", None)
        loader_cfg["persistent_workers"] = False

    loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )

    device = choose_device(args.device)
    model = load_world_model_from_checkpoint(
        cfg,
        checkpoint_path=args.checkpoint_path,
        checkpoint_kind=args.checkpoint_kind,
    ).to(device)
    model.eval()

    latents = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            output = model.encode(batch)
            emb = output["emb"].reshape(-1, output["emb"].shape[-1]).detach().cpu()
            latents.append(emb)

    if not latents:
        raise RuntimeError("No latents were collected. Check the subset and dataloader settings.")

    latents = torch.cat(latents, dim=0).float()
    mean = latents.mean(dim=0)
    centered = latents - mean
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    basis = evecs[:, order].contiguous()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "mean": mean.contiguous(),
        "basis": basis,
        "rank": int(basis.shape[1]),
        "latent_dim": int(basis.shape[0]),
        "subset_fraction": (
            None
            if cfg.get("subset") is None
            else cfg.subset.get("fraction")
        ),
        "subset_seed": (
            None
            if cfg.get("subset") is None
            else cfg.subset.get("seed")
        ),
        "subset_indices_file": (
            None
            if cfg.get("subset") is None
            else cfg.subset.get("indices_file")
        ),
        "selected_episode_count": (
            None if selected_episodes is None else int(len(selected_episodes))
        ),
        "source_checkpoint": str(args.checkpoint_path.expanduser().resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "num_latents_used": int(latents.shape[0]),
        "num_batches_used": min(len(loader), args.max_batches),
        "explained_variance": evals.contiguous(),
        "pca_centered": True,
        "pca_whitened": False,
    }
    torch.save(artifact, args.output_path)

    print(f"saved={args.output_path.resolve()}")
    print(f"latent_dim={artifact['latent_dim']}")
    print(f"num_latents_used={artifact['num_latents_used']}")
    print(f"selected_episode_count={artifact['selected_episode_count']}")


if __name__ == "__main__":
    main()
