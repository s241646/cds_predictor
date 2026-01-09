"""Script to run a wandb hyperparameter optimization sweep."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from cds_repository.data import get_dataloaders
from cds_repository.train import run_training, TrainConfig


def run_sweep_trial() -> None:
    """Run a single trial in a wandb hyperparameter sweep.

    Expects wandb.config to be populated by the sweep controller with
    hyperparameters: lr, dropout, weight_decay, channels, epochs, scheduler_factor.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Get hyperparameters from wandb sweep config
    config = wandb.config
    logger.info(f"Running sweep trial with config: {dict(config)}")

    # Load data
    batch_size = 64
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, num_workers=0)
    logger.info(f"Loaded dataloaders with batch_size={batch_size}")

    # Create training config from sweep parameters
    channels = tuple(config.get("channels", [64, 128, 256]))
    cfg = TrainConfig(
        epochs=int(config.get("epochs", 30)),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
        dropout=float(config.get("dropout", 0.2)),
        channels=channels,
        scheduler_factor=float(config.get("scheduler_factor", 0.5)),
        use_scheduler=True,
        use_wandb=True,
        wandb_project="cds_predictor",
    )

    logger.info(f"Training with config: {cfg}")

    # Run training
    try:
        run_training(train_loader, val_loader, cfg=cfg, logger=logger)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_sweep_trial()
