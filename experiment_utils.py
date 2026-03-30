from __future__ import annotations

from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import (
    filter_dataset_by_episodes,
    get_column_normalizer,
    get_img_preprocessor,
)


def resolve_selected_episodes(cfg, dataset) -> np.ndarray | None:
    subset_cfg = cfg.get("subset")
    if not subset_cfg:
        return None

    if subset_cfg.get("indices_file"):
        indices_path = Path(subset_cfg.indices_file).expanduser().resolve()
        if indices_path.suffix == ".npy":
            return np.asarray(np.load(indices_path), dtype=np.int64)
        return np.asarray(np.loadtxt(indices_path, dtype=np.int64), dtype=np.int64)

    if subset_cfg.get("fraction") is None:
        return None

    total_episodes = len(dataset.lengths)
    keep_count = max(1, int(round(total_episodes * float(subset_cfg.fraction))))
    rng = np.random.default_rng(int(subset_cfg.get("seed", cfg.seed)))
    return np.sort(
        rng.choice(np.arange(total_episodes), size=keep_count, replace=False)
    ).astype(np.int64)


def build_dataset_and_splits(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    selected_episodes = resolve_selected_episodes(cfg, dataset)
    if selected_episodes is not None:
        dataset = filter_dataset_by_episodes(dataset, selected_episodes)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    return dataset, train_set, val_set, selected_episodes, rnd_gen


def build_world_model(cfg) -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )


def load_world_model_from_checkpoint(cfg, checkpoint_path, checkpoint_kind="weights"):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if checkpoint_kind == "object":
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(model, JEPA):
            raise TypeError(
                f"Expected a serialized JEPA object at {checkpoint_path}, got {type(model)}"
            )
        return model.eval()

    if checkpoint_kind != "weights":
        raise ValueError(
            f"Unsupported checkpoint kind '{checkpoint_kind}'. Expected 'weights' or 'object'."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain a 'state_dict' entry."
        )

    model = build_world_model(cfg)
    state_dict = checkpoint["state_dict"]
    model_state = {}
    prefix = "model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            model_state[key[len(prefix) :]] = value

    missing, unexpected = model.load_state_dict(model_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch for {checkpoint_path}. missing={missing}, unexpected={unexpected}"
        )

    return model.eval()
