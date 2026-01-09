from __future__ import annotations

import torch
import torch.nn as nn


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
            nn.Flatten(),      # (B, C, 1) -> (B, C)
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

        x = self.features(x)         # (B, C, L)
        x = self.global_pool(x)      # (B, C, 1)
        logits = self.classifier(x)  # (B, 1)
        return logits.squeeze(1)     # (B,)


def build_model(**kwargs) -> MotifCNN:
    """Convenience factory (useful for configs/tests)."""
    return MotifCNN(**kwargs)
