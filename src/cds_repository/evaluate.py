from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class EvalMetrics:
    loss: float
    acc: float


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> EvalMetrics:
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        n = y.numel()
        total += n
        correct += (preds == y).sum().item()
        total_loss += loss.item() * n

    if total == 0:
        return EvalMetrics(loss=float("nan"), acc=float("nan"))

    return EvalMetrics(
        loss=total_loss / total,
        acc=correct / total,
    )
