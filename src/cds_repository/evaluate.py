from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer


@dataclass
class EvalMetrics:
    loss: float
    acc: float


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> EvalMetrics:
    model.eval()

    trainer = Trainer(accelerator="auto", devices=device)
    test_results = trainer.test(model, dataloaders=loader)

    return EvalMetrics(
        loss=test_results['loss'],
        acc=test_results['acc'],
    )
