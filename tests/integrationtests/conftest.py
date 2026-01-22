import pytest
from unittest.mock import patch, MagicMock
import torch


@pytest.fixture(autouse=True)
def mock_inference_pipeline():
    """Mocks GCS, model loading, and the Lightning Trainer for all tests."""
    with (
        patch("cds_repository.api.get_latest_checkpoint") as mock_ckpt,
        patch("cds_repository.api._load_model") as mock_load,
        patch("pytorch_lightning.Trainer.predict") as mock_predict,
    ):
        mock_ckpt.return_value = "mock_bucket/model.ckpt"

        mock_model = MagicMock()
        mock_meta = {"ckpt_path": "mock", "device": "cpu", "in_channels": 4}
        mock_load.return_value = (mock_model, mock_meta)

        mock_predict.return_value = [
            {"preds": torch.tensor([1]), "logits": torch.tensor([2.5]), "probs": torch.tensor([0.92])}
        ]

        yield
