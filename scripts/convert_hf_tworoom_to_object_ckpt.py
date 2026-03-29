#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import stable_pretraining as spt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def build_model(config: dict) -> JEPA:
    encoder_cfg = config["encoder"]
    predictor_cfg = config["predictor"]
    action_encoder_cfg = config["action_encoder"]
    projector_cfg = config["projector"]
    pred_proj_cfg = config["pred_proj"]

    encoder = spt.backbone.utils.vit_hf(
        encoder_cfg["scale"],
        patch_size=encoder_cfg["patch_size"],
        image_size=encoder_cfg["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    predictor = ARPredictor(**predictor_cfg)
    action_encoder = Embedder(**action_encoder_cfg)
    projector = MLP(norm_fn=torch.nn.BatchNorm1d, **projector_cfg)
    pred_proj = MLP(norm_fn=torch.nn.BatchNorm1d, **pred_proj_cfg)

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official HF LeWM TwoRoom weights to a local *_object.ckpt."
    )
    parser.add_argument(
        "--hf-model-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and weights.pt from quentinll/lewm-tworooms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output run directory. eval.py can point policy to this directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="lewm_hf",
        help="Base filename for the saved object checkpoint.",
    )
    args = parser.parse_args()

    hf_model_dir = args.hf_model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = hf_model_dir / "config.json"
    weights_path = hf_model_dir / "weights.pt"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected config.json and weights.pt under {hf_model_dir}"
        )

    with config_path.open() as f:
        config = json.load(f)

    model = build_model(config)
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch. missing={missing}, unexpected={unexpected}"
        )

    model = model.eval()
    object_ckpt = output_dir / f"{args.output_name}_object.ckpt"
    torch.save(model, object_ckpt)

    with (output_dir / "hf_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(object_ckpt)


if __name__ == "__main__":
    main()
