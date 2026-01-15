import pytest
import torch
import torch.nn as nn

from cds_repository.model import MotifCNN, build_model, MotifCNNModule


# ------------------------
# MotifCNN (nn.Module)
# ------------------------

def test_motifcnn_forward_channels_first():
    """Model should accept (B, 4, L) input."""
    model = MotifCNN()
    x = torch.randn(8, 4, 100)

    logits = model(x)

    assert logits.shape == (8,), "Output logits shape mismatch, should be (B,)"
    assert logits.dtype == torch.float32, "Output logits dtype should be float32"


def test_motifcnn_forward_channels_last():
    """Model should accept (B, L, 4) input and transpose internally."""
    model = MotifCNN()
    x = torch.randn(8, 100, 4)

    logits = model(x)

    assert logits.shape == (8,), "Output logits shape mismatch, should be (B,)"


def test_motifcnn_invalid_input_dims():
    """Invalid input dimensionality should raise ValueError."""
    model = MotifCNN()
    x = torch.randn(4, 100)  # missing batch or channel dim

    with pytest.raises(ValueError):
        model(x)


def test_motifcnn_invalid_channel_size():
    """Wrong channel size should raise ValueError."""
    model = MotifCNN()
    x = torch.randn(4, 5, 100)  # 5 channels instead of 4

    with pytest.raises(ValueError):
        model(x)


def test_build_model_returns_motifcnn():
    """build_model should return a MotifCNN instance."""
    model = build_model()
    assert isinstance(model, MotifCNN), "build_model did not return a MotifCNN instance"


# ------------------------
# MotifCNNModule (Lightning)
# ------------------------

def test_lightning_module_forward():
    """Lightning module forward should delegate to model."""
    module = MotifCNNModule()
    x = torch.randn(4, 4, 80)

    logits = module(x)

    assert logits.shape == (4,), "Output logits shape mismatch, should be (B,)"


def test_training_step_runs_and_returns_loss():
    """training_step should return a scalar loss."""
    module = MotifCNNModule()
    x = torch.randn(4, 4, 60)
    y = torch.randint(0, 2, (4,)).float()

    loss = module.training_step((x, y), batch_idx=0)

    assert isinstance(loss, torch.Tensor), "training_step did not return a tensor"
    assert loss.ndim == 0, "training_step loss is not a scalar tensor"


def test_accuracy_computation():
    """_acc should return value in [0, 1]."""
    logits = torch.tensor([10.0, -10.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    acc = MotifCNNModule._acc(logits, targets)

    assert 0.0 <= acc <= 1.0, "Accuracy not in [0, 1] range"


def test_configure_optimizers_returns_expected():
    """Optimizer config should include optimizer and scheduler."""
    module = MotifCNNModule(use_scheduler=True)
    config = module.configure_optimizers()

    assert "optimizer" in config, "Optimizer config missing 'optimizer' key"
    assert "lr_scheduler" in config, "Optimizer config missing 'lr_scheduler' key"
    assert isinstance(config["optimizer"], torch.optim.Optimizer), "'optimizer' is not an instance of torch.optim.Optimizer"
