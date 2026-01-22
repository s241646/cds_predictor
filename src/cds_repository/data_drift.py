import sys
import tempfile
from pathlib import Path
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import torch
import typer
from evidently import Report
from evidently.presets import DataDriftPreset
from google.cloud import storage

from cds_repository.model import MotifCNNModule, MotifCNN


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    fs, path = fsspec.core.url_to_fs(save_dir)
    files = fs.ls(path)
    ckpts = [f for f in files if f.endswith(".ckpt") or f.endswith(".pt")]

    if not ckpts:
        return None

    ckpts = sorted(ckpts, key=lambda f: fs.info(f)["mtime"])
    return ckpts[-1]


def download_from_gcs(path: str, suffix: str = ".ckpt") -> Path:
    """
    Download a file from GCS to a temporary local path and return the local Path.
    Accepts:
      - gs://bucket/blob
      - bucket/blob
    """
    if path.startswith("gs://"):
        bucket_name, blob_path = path.replace("gs://", "").split("/", 1)
    else:
        bucket_name, blob_path = path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {bucket_name}/{blob_path}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    blob.download_to_filename(tmp.name)
    return Path(tmp.name)


def load_csv_from_gcs(gcs_uri: str) -> pd.DataFrame:
    assert gcs_uri.startswith("gs://")

    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with tempfile.NamedTemporaryFile(suffix=".csv.gz") as tmp:
        blob.download_to_filename(tmp.name)
        return pd.read_csv(tmp.name, compression="gzip")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    sample_key = next(iter(state_dict.keys()))

    if sample_key.startswith("model."):
        hparams = checkpoint.get("hyper_parameters", {})
        model = MotifCNNModule(**hparams) if hparams else MotifCNNModule()
        model.load_state_dict(state_dict)
    else:
        model = MotifCNN(channels=(128, 256, 512))
        model.load_state_dict(state_dict)

    model.eval().to(device)
    return model


# ---------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------
def extract_embeddings(model, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    embeddings = []

    base_model = model.model if hasattr(model, "model") else model
    n_samples = X.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X[i : i + batch_size].reshape(-1, 4, 300)
            x = torch.from_numpy(batch.astype(np.float32)).to(device)

            x = base_model.features(x)
            x = base_model.global_pool(x)
            x = x.squeeze(-1)

            embeddings.append(x.cpu().numpy())

    return np.vstack(embeddings)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(
    input_gcs_uri: str = typer.Option(..., help="gs://bucket/path/to/file.csv.gz"),
    dataset_name: str = typer.Option(default="latest"),
):
    device = get_device()

    report_dir = project_root / "reports" / "drift_check"

    train_path = project_root / "data" / "raw" / "training"
    ckpt_path = get_latest_checkpoint("gs://cds-predictor/models")
    print("Using checkpoint:", ckpt_path)

    model_path = (
        download_from_gcs(ckpt_path, suffix=".ckpt") if ckpt_path else project_root / "models" / "motifcnn.ckpt"
    )

    train_df = pd.read_csv(train_path / "train.csv.gz", compression="gzip")
    new_df = load_csv_from_gcs(input_gcs_uri)

    missing_cols = set(train_df.columns) - set(new_df.columns)
    if missing_cols:
        raise ValueError(f"Drift data is missing columns: {missing_cols}")

    label_col = train_df.columns[-1]
    feature_cols = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_cols].values
    X_new = new_df[feature_cols].values

    # -----------------------------------------------------------------
    # Input feature drift
    # -----------------------------------------------------------------
    # Remove labels for drift
    drift_ref = train_df.drop(columns=[label_col])
    drift_cur = new_df.drop(columns=[label_col])

    report = Report([DataDriftPreset()])
    report.run(
        reference_data=drift_ref,
        current_data=drift_cur,
    )

    feature_report_path = report_dir / f"input_feature_drift_report_{dataset_name}.html"

    html = report.as_html()
    feature_report_path.parent.mkdir(parents=True, exist_ok=True)
    feature_report_path.write_text(html, encoding="utf-8")

    print(f"Input feature drift report saved to: {feature_report_path}")

    # -----------------------------------------------------------------
    # Embedding drift
    # -----------------------------------------------------------------
    model = load_model(model_path, device)

    train_emb = extract_embeddings(model, X_train, device)
    new_emb = extract_embeddings(model, X_new, device)

    emb_cols = [f"emb_{i}" for i in range(train_emb.shape[1])]
    train_emb_df = pd.DataFrame(train_emb, columns=emb_cols)
    new_emb_df = pd.DataFrame(new_emb, columns=emb_cols)

    emb_report = Report([DataDriftPreset()])
    emb_report.run(
        reference_data=train_emb_df,
        current_data=new_emb_df,
    )

    emb_report_path = report_dir / f"embedding_drift_report_{dataset_name}.html"

    html = emb_report.as_html()
    emb_report_path.parent.mkdir(parents=True, exist_ok=True)
    emb_report_path.write_text(html, encoding="utf-8")

    print(f"Embedding drift report saved to: {emb_report_path}")


if __name__ == "__main__":
    typer.run(main)
