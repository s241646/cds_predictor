from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl


class MotifCNN(nn.Module):
    """
    Strong, simple baseline for fixed-length DNA binary classification.

    Expected input x:
      - (B, 4, L)  channels-first OR
      - (B, L, 4)  channels-last (we transpose)
    Output:
      - logits of shape (B,)  (use BCEWithLogitsLoss)
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: tuple[int, int, int] = (64, 128, 256),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
        dropout: float = 0.2,
        pool: str = "max",  # "max" or "avg"
    ) -> None:
        super().__init__()
        if pool not in {"max", "avg"}:
            raise ValueError("pool must be 'max' or 'avg'")

        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes

        def conv_block(cin: int, cout: int, k: int) -> nn.Sequential:
            pad = k // 2  # "same" padding for odd kernels
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=pad),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(in_channels, c1, k1),
            nn.Dropout(dropout),
            conv_block(c1, c2, k2),
            nn.Dropout(dropout),
            conv_block(c2, c3, k3),
        )

        self.global_pool: nn.Module
        if pool == "max":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, C, 1) -> (B, C)
            nn.Dropout(dropout),
            nn.Linear(c3, 1),  # -> (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have 3 dims (B,*,*). Got {tuple(x.shape)}")

        # (B, L, 4) -> (B, 4, L)
        if x.shape[-1] == 4:
            x = x.transpose(1, 2)

        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 channels at dim=1, got {tuple(x.shape)}")

        x = self.features(x)  # (B, C, L)
        x = self.global_pool(x)  # (B, C, 1)
        logits = self.classifier(x)  # (B, 1)
        return logits.squeeze(1)  # (B,)


def build_model(**kwargs) -> MotifCNN:
    """Convenience factory (useful for configs/tests)."""
    return MotifCNN(**kwargs)


class MotifCNNModule(pl.LightningModule):
    """PyTorch Lightning wrapper for training/evaluating MotifCNN."""

    def __init__(
        self,
        lr: float = 1e-3,
        # Model hyperparameters
        in_channels: int = 4,
        channels: tuple[int, int, int] = (64, 128, 256),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
        dropout: float = 0.2,
        pool: str = "max",
        criterion: nn.Module | None = nn.BCEWithLogitsLoss(),
        weight_decay: float = 1e-2,
        use_scheduler: bool = True,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
    ) -> None:
        super().__init__()
        self.model = build_model(
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            pool=pool,
        )
        self.lr = lr
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
        preds = (torch.sigmoid(logits) >= 0.5).float()
        return (preds == targets).float().mean()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self._acc(logits, y), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self._acc(logits, y), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        x = x.float()
        y = y.float()

        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self._acc(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=self.scheduler_factor, patience=self.scheduler_patience
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        if not self.use_scheduler:
            return opt  # no lr scheduler
