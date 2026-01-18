from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

from cds_repository.model import MotifCNNModule
from cds_repository.data import get_dataloaders
import hydra
import os


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wandb_enabled() -> bool:
    return "WANDB_MODE" not in os.environ or os.environ.get("WANDB_MODE") != "offline"


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-2
    save_dir: str = "models"
    save_name: str = "motifcnn.pt"
    patience: int = 5
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    # model hyperparameters
    channels: tuple[int, int, int] = (64, 128, 256)
    kernel_sizes: tuple[int, int, int] = (7, 5, 3)
    dropout: float = 0.2
    # logging
    log_dir: str = "logs"
    tensorboard: bool = True


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg) -> None:
    # setup logging to write to Hydra outputs folder
    log_file = Path("train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training with config: {cfg}")

    # seed
    seed = getattr(cfg, "seed", 42)
    pl.seed_everything(seed)
    logger.info(f"Set random seed to {seed}")

    # load dataloaders using config values
    batch_size = cfg.hyperparameters.batch_size if "hyperparameters" in cfg else getattr(cfg, "batch_size", 32)
    num_workers = getattr(cfg, "num_workers", 0)
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)
    logger.info(f"Loaded dataloaders with batch_size={batch_size}, num_workers={num_workers}")

    # pass the Hydra config (DictConfig) directly to run_training;

    model = MotifCNNModule(
        in_channels=cfg.in_channels,
        lr=cfg.lr,
        channels=cfg.channels,
        kernel_sizes=cfg.kernel_sizes,
        dropout=cfg.dropout,
        weight_decay=cfg.weight_decay,
        use_scheduler=cfg.use_scheduler,
        scheduler_factor=cfg.scheduler_factor,
        scheduler_patience=cfg.scheduler_patience,
    )

    # Wandb Logger if enabled
    if wandb_enabled():
        wandb_logger = pl.loggersWandbLogger(
            project="cds_predictor",
            log_model="all",
        )
    else:
        wandb_logger = None

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to {cfg.save_dir}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.save_dir,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=wandb_logger,  # none if offline
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto",
        # profiler="simple",
    )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = checkpoint_callback.best_model_path
    logger.info(f"Best checkpoint path: {best_ckpt}")

    if wandb_enabled() and wandb_logger is not None and best_ckpt:
        artifact = wandb.Artifact(
            name="motifcnn",
            type="model",
            description="Best MotifCNN checkpoint",
            metadata=dict(cfg),
        )
        artifact.add_file(best_ckpt)
        run = wandb_logger.experiment
        logged_artifact = run.log_artifact(artifact)
        logged_artifact.wait()

        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
