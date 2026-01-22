import pandas as pd
import numpy as np
import torch
import typer
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional
import fsspec
import time
from typing import Tuple, Dict, Any
from cds_repository.model import MotifCNNModule
import sys

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


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    """
    Get latest checkpoint from a directory.
    save_dir MUST include scheme (e.g. gs://).
    """
    if "://" not in save_dir:
        raise ValueError(f"save_dir must include scheme (e.g. gs://), got: {save_dir}")

    fs, path = fsspec.core.url_to_fs(save_dir)

    print(f"[DEBUG] Using filesystem: {type(fs)}")
    print(f"[DEBUG] Resolved path: {path}")

    files = fs.ls(path, detail=True)

    ckpts = [f for f in files if f["name"].endswith(".ckpt") or f["name"].endswith(".pt")]

    if not ckpts:
        return None

    ckpts = sorted(ckpts, key=lambda f: f.get("mtime", 0))

    latest = ckpts[-1]["name"]

    # Re-attach scheme explicitly
    if not latest.startswith("gs://"):
        latest = f"gs://{latest}"

    return latest


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[MotifCNNModule, Dict[str, Any]]:
    """
    Load MotifCNNModule from a local or GCS checkpoint.
    """
    print(f"Loading checkpoint: {ckpt_path}")

    # Open via fsspec (works for gs:// and local)
    with fsspec.open(ckpt_path, "rb") as f:
        model = MotifCNNModule.load_from_checkpoint(
            f,
            map_location=device,
            weights_only=False,
        )

    model.eval()
    model.to(device)

    # Extract metadata
    hparams: Dict[str, Any] = {}
    in_channels = None
    try:
        hparams = dict(model.hparams) if hasattr(model, "hparams") else {}
        if "in_channels" in hparams:
            in_channels = int(hparams["in_channels"])
    except Exception:
        pass

    meta = {
        "ckpt_path": ckpt_path,
        "device": str(device),
        "in_channels": in_channels,
        "hparams": hparams,
        "loaded_at_unix": time.time(),
    }

    return model, meta


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


def read_csv_any(path: str | Path, **kwargs) -> pd.DataFrame:
    path = str(path)
    if path.startswith("gs://"):
        return pd.read_csv(path, **kwargs)
    return pd.read_csv(Path(path), **kwargs)


def position_encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw nt_seq dataframe into position-encoded format
    identical to data/processed/training.
    """
    if df.columns[0].startswith("pos_"):
        # Already encoded
        return df

    if "nt_seq" not in df.columns:
        raise ValueError("Expected 'nt_seq' column for raw sequence input")

    sequences = []
    for seq in df["nt_seq"]:
        encoded_seq = []
        for nucleotide in seq:
            if nucleotide == "A":
                encoded_seq.extend([1, 0, 0, 0])
            elif nucleotide == "T":
                encoded_seq.extend([0, 1, 0, 0])
            elif nucleotide == "C":
                encoded_seq.extend([0, 0, 1, 0])
            elif nucleotide == "G":
                encoded_seq.extend([0, 0, 0, 1])
            else:  # N or unknown
                encoded_seq.extend([0, 0, 0, 0])
        sequences.append(encoded_seq)

    # Build column names exactly like training
    max_length = len(sequences[0])
    nucleotides = ["A", "T", "C", "G"]
    columns = [f"pos_{i // 4}_{nucleotides[i % 4]}" for i in range(max_length)]

    encoded_df = pd.DataFrame(sequences, columns=columns)

    if "label" in df.columns:
        encoded_df["label"] = df["label"].values

    return encoded_df


def main(
    new_file: str = typer.Option(
        default=None,
        help="Path to the drift data file (local or gs://bucket/path.csv.gz)",
    ),
    dataset_name: str = typer.Option(default="new_data_dfset", help="Name of the dataset to use for drift detection"),
) -> None:
    # Setup device
    device = get_device()

    # Find latest checkpoint in container
    ckpt_path = get_latest_checkpoint("gs://cds-predictor/models")
    print(f"Latest checkpoint: {ckpt_path}")

    if ckpt_path is None:
        raise RuntimeError("No checkpoint found in gs://cds-predictor/models")

    # Load model
    model, model_meta = load_model_from_checkpoint(ckpt_path, device)
    print(f"Loaded model metadata: {model_meta}")

    # Paths
    train_path = project_root / "data" / "processed" / "training"
    if new_file is None:
        new_file = project_root / "data" / "processed" / "drift_check" / "tt4_genome.csv.gz"

    # Load data
    # train_data_df = pd.read_csv(train_path / "train.csv.gz", compression='gzip').sample(n=10000, random_state=42)
    # new_data_df = pd.read_csv(new_file, compression='gzip').sample(n=10000, random_state=42)
    train_data_df = read_csv_any(
        train_path / "train.csv.gz",
        compression="gzip",
    )

    # Load training data (already encoded)
    train_data_df = read_csv_any(
        train_path / "train.csv.gz",
        compression="gzip",
    )

    if new_file is None:
        new_file = project_root / "data" / "processed" / "drift_check" / "tt4_genome.csv.gz"

    # Load new data (raw OR encoded)
    raw_new_df = read_csv_any(
        new_file,
        compression="gzip",
    )

    # Ensure same encoding as training
    new_data_df = position_encode_df(raw_new_df)

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
