import os
import pandas as pd
import numpy as np
import torch
import typer
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import model
import sys
from cds_repository.model import MotifCNNModule, MotifCNN

# Data drift detection and model robustness check.
# This tests model robustness to biological covariate shift (different translation tables).


# Resolve paths relative to project structure
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent  # up from src/cds_repository/
sys.path.insert(0, str(project_root / "src"))


def get_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get state dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Check if keys have "model." prefix (Lightning wrapper) or not (raw MotifCNN)
    sample_key = next(iter(state_dict.keys()))

    if sample_key.startswith("model."):
        # Lightning checkpoint - load into MotifCNNModule
        hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
        model = MotifCNNModule(**hparams) if hparams else MotifCNNModule()
        model.load_state_dict(state_dict)
    else:
        # Raw MotifCNN checkpoint - load directly into MotifCNN
        # Model was trained with channels=(128, 256, 512)
        model = MotifCNN(channels=(128, 256, 512))
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model


def predict_batch(model, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Run predictions on a batch of one-hot encoded sequences."""
    model.eval()
    predictions = []

    # Data is (n_samples, n_features) where features are flattened one-hot
    # Model expects (batch, 4, 300) - need to reshape
    n_samples = X.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X[i : i + batch_size]
            # Reshape from (batch, 1200) to (batch, 4, 300)
            batch = batch.reshape(-1, 4, 300)
            x = torch.from_numpy(batch.astype(np.float32)).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs.flatten())

    return np.array(predictions)


def evaluate_model(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """Calculate classification metrics."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def extract_embeddings(model, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Extract internal representations from the model (after global pooling, before classifier)."""
    model.eval()
    embeddings = []

    # Get the underlying MotifCNN if wrapped in Lightning module
    if hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model

    n_samples = X.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X[i : i + batch_size]
            # Reshape from (batch, 1200) to (batch, 4, 300)
            batch = batch.reshape(-1, 4, 300)
            x = torch.from_numpy(batch.astype(np.float32)).to(device)

            # Pass through features and global pool (but not classifier)
            x = base_model.features(x)  # (B, C, L)
            x = base_model.global_pool(x)  # (B, C, 1)
            x = x.squeeze(-1)  # (B, C)

            embeddings.append(x.cpu().numpy())

    return np.vstack(embeddings)


def main(
    new_file: Path = typer.Option(
        default=None, help="Path to the drift data file (e.g., data/processed/drift_check/tt4_genome.csv.gz)"
    ),
    dataset_name: str = typer.Option(default="new_data_dfset", help="Name of the dataset to use for drift detection"),
) -> None:
    # Setup device
    device = get_device()

    # Paths
    train_path = project_root / "data" / "processed" / "training"
    if new_file is None:
        new_file = project_root / "data" / "processed" / "drift_check" / "tt4_genome.csv.gz"

    # Model path: use MODEL_CKPT_PATH env var if set, otherwise default
    model_path_env = os.getenv("MODEL_CKPT_PATH", "").strip()
    if model_path_env:
        model_path = Path(model_path_env)
    else:
        model_path = project_root / "models" / "motifcnn.ckpt"

    # Load data
    # train_data_df = pd.read_csv(train_path / "train.csv.gz", compression='gzip').sample(n=10000, random_state=42)
    # new_data_df = pd.read_csv(new_file, compression='gzip').sample(n=10000, random_state=42)
    train_data_df = pd.read_csv(train_path / "train.csv.gz", compression="gzip")
    new_data_df = pd.read_csv(new_file, compression="gzip")

    # Separate features and labels
    label_col = train_data_df.columns[-1]
    feature_cols = [c for c in train_data_df.columns if c != label_col]

    X_train = train_data_df[feature_cols].values
    X_drift = new_data_df[feature_cols].values

    report_emb = Report([DataDriftPreset()])
    result_emb = report_emb.run(train_data_df, new_data_df)
    emb_report_path = project_root / "reports" / "drift_check" / f"input_feature_drift_report_{dataset_name}.html"
    result_emb.save_html(str(emb_report_path))
    print(f"Input feature drift report saved to: {emb_report_path}.")

    # Load model
    model = load_model(model_path, device)

    # Extract Embeddings
    train_embeddings = extract_embeddings(model, X_train, device)
    new_embeddings = extract_embeddings(model, X_drift, device)

    # Create DataFrames for embedding drift detection
    embedding_cols = [f"emb_{i}" for i in range(train_embeddings.shape[1])]
    train_emb_df = pd.DataFrame(train_embeddings, columns=embedding_cols)
    new_emb_df = pd.DataFrame(new_embeddings, columns=embedding_cols)

    # Data drift detection on embeddings
    report_emb = Report([DataDriftPreset()])
    result_emb = report_emb.run(train_emb_df, new_emb_df)
    emb_report_path = project_root / "reports" / "drift_check" / f"embedding_drift_report_{dataset_name}.html"
    result_emb.save_html(str(emb_report_path))
    print(f"Embedding drift report saved to: {emb_report_path}.")


if __name__ == "__main__":
    typer.run(main)
