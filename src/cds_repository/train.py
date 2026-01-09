from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cds_repository.model import build_model
from cds_repository.evaluate import evaluate


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def train_one_epoch(model: torch.nn.Module, loader, optimizer, device: torch.device) -> float:
    model.train()
    total = 0
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        n = y.numel()
        total += n
        total_loss += loss.item() * n

    return total_loss / total if total > 0 else float("nan")


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def run_training(
    train_loader,
    val_loader,
    cfg: TrainConfig | None = None,
    experiment_name: str | None = None,
) -> Path:
    cfg = cfg or TrainConfig()
    device = get_device()

    # Initialize TensorBoard writer
    writer = None
    if cfg.tensorboard:
        log_dir = Path(cfg.log_dir)
        if experiment_name:
            log_dir = log_dir / experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = log_dir / f"run_{timestamp}"
        writer = SummaryWriter(str(log_dir))

    model = build_model(
        in_channels=4,
        channels=cfg.channels,
        kernel_sizes=cfg.kernel_sizes,
        dropout=cfg.dropout,
        pool="max",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if getattr(cfg, "use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
        )
    best_val_loss = float("inf")
    out_path = Path(cfg.save_dir) / cfg.save_name
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics.loss, epoch)
            writer.add_scalar("Accuracy/val", val_metrics.acc, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | val_acc={val_metrics.acc:.4f}"
        )

        # Save best / early stopping
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            epochs_without_improvement = 0
            save_checkpoint(model, out_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Step LR scheduler on validation loss (if configured)
        if scheduler is not None:
            scheduler.step(val_metrics.loss)

    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {log_dir}")

    print(f"Saved best checkpoint to: {out_path}")
    return out_path


if __name__ == "__main__":
    from cds_repository.data import get_dataloaders

    train_loader, val_loader = get_dataloaders()

    run_training(train_loader, val_loader)
