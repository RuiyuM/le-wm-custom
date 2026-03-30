#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a random orthonormal subspace artifact compatible with fixed-subspace SIGReg."
    )
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_random_orthonormal_basis(latent_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(latent_dim, latent_dim, generator=generator)
    q, r = torch.linalg.qr(gaussian, mode="reduced")

    # Fix the sign ambiguity in QR so the artifact is reproducible for a seed.
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.contiguous()


def main() -> None:
    args = parse_args()
    reference = torch.load(
        args.reference_artifact.expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )

    if "mean" not in reference:
        raise KeyError(
            f"Reference artifact at {args.reference_artifact} must contain 'mean'."
        )

    mean = torch.as_tensor(reference["mean"], dtype=torch.float32).reshape(-1).contiguous()
    latent_dim = int(reference.get("latent_dim", mean.numel()))
    basis = make_random_orthonormal_basis(latent_dim=latent_dim, seed=args.seed)

    artifact = {
        "mean": mean,
        "basis": basis,
        "rank": int(basis.shape[1]),
        "latent_dim": latent_dim,
        "subset_fraction": reference.get("subset_fraction"),
        "subset_seed": reference.get("subset_seed"),
        "subset_indices_file": reference.get("subset_indices_file"),
        "selected_episode_count": reference.get("selected_episode_count"),
        "source_checkpoint": reference.get("source_checkpoint"),
        "checkpoint_kind": reference.get("checkpoint_kind"),
        "num_latents_used": reference.get("num_latents_used"),
        "num_batches_used": reference.get("num_batches_used"),
        "pca_centered": bool(reference.get("pca_centered", True)),
        "pca_whitened": False,
        "basis_kind": "random_orthonormal",
        "random_seed": int(args.seed),
        "reference_artifact": str(args.reference_artifact.expanduser().resolve()),
    }

    args.output_path = args.output_path.expanduser().resolve()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_path)
    print(f"saved={args.output_path}")
    print(f"latent_dim={latent_dim}")
    print(f"random_seed={args.seed}")


if __name__ == "__main__":
    main()
