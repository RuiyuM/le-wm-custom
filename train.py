from functools import partial
from pathlib import Path
import shutil

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from experiment_utils import build_dataset_and_splits, build_world_model
from module import SIGReg, SupportSubspaceProjector
from utils import ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg, sigreg_projector=None):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    emb_reg = emb.transpose(0, 1)
    if sigreg_projector is not None:
        emb_reg = sigreg_projector(emb_reg)
    output["sigreg_loss"]= self.sigreg(emb_reg)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset, train_set, val_set, selected_episodes, rnd_gen = build_dataset_and_splits(cfg)

    if selected_episodes is not None:
        print(
            f"Subset active: kept {len(selected_episodes)} episodes, "
            f"{len(dataset)} clips."
        )

    loader_cfg = dict(cfg.loader)
    if int(loader_cfg.get("num_workers", 0)) == 0:
        loader_cfg.pop("prefetch_factor", None)
        loader_cfg["persistent_workers"] = False

    train = torch.utils.data.DataLoader(
        train_set,
        **loader_cfg,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **loader_cfg,
        shuffle=False,
        drop_last=False,
    )
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = build_world_model(cfg)

    sigreg_projector = None
    subspace_cfg = cfg.get("subspace")
    if subspace_cfg and subspace_cfg.get("enabled", False):
        basis_path = subspace_cfg.get("basis_path")
        if not basis_path:
            raise ValueError("subspace.enabled=True requires subspace.basis_path")
        sigreg_projector = SupportSubspaceProjector.from_artifact(
            Path(basis_path).expanduser().resolve(),
            rank=subspace_cfg.get("rank"),
            center=subspace_cfg.get("center", True),
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg, sigreg_projector=sigreg_projector),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    if selected_episodes is not None:
        np.save(run_dir / "subset_episode_indices.npy", selected_episodes)

    target_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    resume_ckpt_path = cfg.get("resume_ckpt_path")
    if resume_ckpt_path:
        resume_ckpt_path = Path(resume_ckpt_path).expanduser().resolve()
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"resume_ckpt_path does not exist: {resume_ckpt_path}")
        if target_ckpt_path.exists():
            print(
                f"Target checkpoint already exists at {target_ckpt_path}; "
                f"ignoring resume_ckpt_path={resume_ckpt_path}."
            )
        elif resume_ckpt_path != target_ckpt_path:
            shutil.copy2(resume_ckpt_path, target_ckpt_path)
            print(f"Copied warmup checkpoint to {target_ckpt_path} for resume.")

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=target_ckpt_path,
    )

    manager()
    return


if __name__ == "__main__":
    run()
