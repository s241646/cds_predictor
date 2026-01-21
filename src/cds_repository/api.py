import io
import os
import time
from pathlib import Path
from uuid import uuid4
import fsspec
from typing import Any, Dict, List, Optional, Tuple
from Bio import SeqIO

import torch
from pytorch_lightning import Trainer

from fastapi import FastAPI, HTTPException
from fastapi.logger import logger
from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field

from cds_repository.model import MotifCNNModule
from cds_repository.data import load_dataloader
from cds_repository.preprocess_fasta import preprocess_fasta

# ----------------------------
# Configuration
# ----------------------------
MODEL_CKPT_PATH_ENV = "MODEL_CKPT_PATH"
DEFAULT_THRESHOLD_ENV = "DEFAULT_THRESHOLD"  # optional; defaults to 0.5
MAX_BATCH_ENV = "MAX_BATCH"  # optional; defaults to 256
MIN_LENGTH_ENV = "MIN_LENGTH"  # optional; defaults to 1
MAX_LENGTH_ENV = "MAX_LENGTH"  # optional; defaults to 10_000
ALLOW_2D_SINGLE_ENV = "ALLOW_2D_SINGLE"  # optional; defaults to "true"


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return default if not v else float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return default if not v else int(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


DEFAULT_THRESHOLD = _env_float(DEFAULT_THRESHOLD_ENV, 0.5)
MAX_BATCH = _env_int(MAX_BATCH_ENV, 256)
MIN_LENGTH = _env_int(MIN_LENGTH_ENV, 1)
MAX_LENGTH = _env_int(MAX_LENGTH_ENV, 10_000)
ALLOW_2D_SINGLE = _env_bool(ALLOW_2D_SINGLE_ENV, True)
DEFAULT_BATCH_SIZE = _env_int("DEFAULT_BATCH_SIZE", 32)


# ----------------------------
# Device selection
# ----------------------------
def get_device() -> torch.device:
    # Prefer MPS on Apple silicon, then CUDA, then CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Checkpoint selection
# ----------------------------


def _find_latest_ckpt(checkpoint_dir: str) -> Optional[str]:
    p = Path(checkpoint_dir)
    if not p.exists() or not p.is_dir():
        return None
    ckpt_files = list(p.glob("*.ckpt"))
    if not ckpt_files:
        return None
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    return str(latest_ckpt.resolve())


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    fs, path = fsspec.core.url_to_fs(save_dir)

    files = fs.ls(path)
    ckpts = [f for f in files if f.endswith(".ckpt") or f.endswith(".pt")]

    if not ckpts:
        return None

    # sort by modified time
    ckpts = sorted(ckpts, key=lambda f: fs.info(f)["mtime"])
    return ckpts[-1]


# ----------------------------
# Request/Response schemas
# ----------------------------


class PredictResponseItem(BaseModel):
    id: str
    sequence: str
    pred: int
    prob: Optional[float] = None
    logit: Optional[float] = None


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


class ModelInfo(BaseModel):
    ckpt_path: str
    device: str
    in_channels: Optional[int] = None
    hparams: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="CDS Predictor Inference API", version="0.1.0")


# ----------------------------
# Core model load / validation
# ----------------------------
# def _resolve_ckpt_path(requested: Optional[str] = None) -> str:
#     ckpt_path = (requested or os.getenv(MODEL_CKPT_PATH_ENV, "")).strip()
#     if not ckpt_path:
#         raise RuntimeError(f"{MODEL_CKPT_PATH_ENV} is not set and no ckpt_path was provided.")
#     p = Path(ckpt_path)
#     if not p.exists():
#         raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
#     return str(p.resolve())


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if not device_str or device_str == "auto":
        return get_device()
    device_str = device_str.lower()
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    raise RuntimeError(f"Unknown device '{device_str}'. Use cpu|cuda|mps|auto.")


def _load_model(ckpt_path: str, device: torch.device) -> Tuple[MotifCNNModule, Dict[str, Any]]:
    logger.info(f"Loading checkpoint: {ckpt_path}")
    with fsspec.open("gs://" + ckpt_path, "rb") as f:
        # 2. Pass the open file handle directly to Lightning
        model = MotifCNNModule.load_from_checkpoint(f, map_location=device, weights_only=False)
    # model = MotifCNNModule.load_from_checkpoint(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # Try to extract expected channels from saved hparams
    in_channels = None
    hparams: Dict[str, Any] = {}
    try:
        hparams = dict(model.hparams) if hasattr(model, "hparams") else {}
        if "in_channels" in hparams:
            in_channels = int(hparams["in_channels"])
    except Exception:
        hparams = {}

    meta = {
        "ckpt_path": ckpt_path,
        "device": str(device),
        "in_channels": in_channels,
        "hparams": hparams,
        "loaded_at_unix": time.time(),
    }
    return model, meta


# ----------------------------
# Startup: load model once
# ----------------------------
@app.on_event("startup")
def startup() -> None:
    ckpt_path = get_latest_checkpoint("gs://cds-predictor/models")
    print(ckpt_path)

    if ckpt_path is None:
        logger.info("No checkpoint found in specified save_dir.")

    device = _resolve_device("auto")
    model, meta = _load_model(ckpt_path, device)

    app.state.model = model
    app.state.model_meta = meta
    app.state.trainer = Trainer(accelerator="auto")


# TODO: cleanup this code
# TODO: look at the report & my tasks


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Minimal liveness + readiness check.
    """
    loaded = hasattr(app.state, "model") and app.state.model is not None
    meta = getattr(app.state, "model_meta", {}) if loaded else {}
    return {
        "status": "ok" if loaded else "not_ready",
        "model_loaded": loaded,
        "device": meta.get("device"),
    }


@app.get("/info", response_model=ModelInfo)
def info() -> ModelInfo:
    """
    Returns model metadata useful for debugging and clients (expected channels, checkpoint path, device).
    """
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    meta = app.state.model_meta
    return ModelInfo(
        ckpt_path=meta["ckpt_path"],
        device=meta["device"],
        in_channels=meta.get("in_channels"),
        hparams=meta.get("hparams", {}),
    )


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
async def predict_fasta(
    fasta: UploadFile = File(...),
    return_logits: bool = Form(False),
    return_probs: bool = Form(False),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
) -> PredictResponse:
    content = await fasta.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="FASTA must be UTF-8 encoded text.")

    parsed = list(SeqIO.parse(io.StringIO(text), "fasta"))
    if not parsed:
        raise HTTPException(status_code=400, detail="No sequences found in FASTA.")

    # Store original data for mapping
    fasta_records = [{"id": x.id, "seq": str(x.seq)} for x in parsed]

    # Create unique keys for preprocessor
    records = {f"{x.id}_{i}": str(x.seq) for i, x in enumerate(parsed)}
    preprocess_fasta(seq_dict=records)

    effective_batch = min(batch_size, MAX_BATCH)
    test_loader = load_dataloader("data/tmp/", split="test", batch_size=effective_batch)

    # Run inference
    predictions = app.state.trainer.predict(app.state.model, dataloaders=test_loader)
    if not predictions:
        raise HTTPException(status_code=500, detail="Model returned no predictions.")

    all_preds = torch.cat([batch["preds"] for batch in predictions]).cpu().tolist()

    # Only flatten these if the flags are true
    all_logits = torch.cat([batch["logits"] for batch in predictions]).cpu().tolist() if return_logits else None
    all_probs = torch.cat([batch["probs"] for batch in predictions]).cpu().tolist() if return_probs else None

    results = []
    # Use the length of all_preds to prevent IndexError
    for i in range(len(all_preds)):
        # Truncate sequence for UI visibility
        orig_seq = fasta_records[i]["seq"]
        display_seq = orig_seq[:50] + "..." if len(orig_seq) > 50 else orig_seq

        item = PredictResponseItem(id=str(fasta_records[i]["id"]), sequence=display_seq, pred=int(all_preds[i]))

        if return_probs and all_probs is not None:
            item.prob = float(all_probs[i])
        if return_logits and all_logits is not None:
            item.logit = float(all_logits[i])

        results.append(item)

    return PredictResponse(results=results)


@app.get("/config")
def runtime_config() -> Dict[str, Any]:
    """
    Exposes runtime limits/settings (useful for clients).
    Does not expose secrets.
    """
    return {
        "default_threshold": DEFAULT_THRESHOLD,
        "max_batch": MAX_BATCH,
        "min_length": MIN_LENGTH,
        "max_length": MAX_LENGTH,
        "allow_2d_single": ALLOW_2D_SINGLE,
    }


@app.post("/upload_input_fasta")
async def upload_input_fasta(
    fasta: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Endpoint to upload an input FASTA file and save it to a temporary location.
    """
    content = await fasta.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="FASTA must be UTF-8 encoded text.")

    parsed = list(SeqIO.parse(io.StringIO(text), "fasta"))
    if not parsed:
        raise HTTPException(status_code=400, detail="No sequences found in FASTA.")

    unique_id = uuid4()
    local_tmp_path = f"/tmp/uploaded_{unique_id}.fasta"
    gcs_dest_path = f"gs://cds-predictor/uploaded_data/input_{unique_id}.fasta"

    # Create unique keys for preprocessor
    records = {f"{x.id}_{i}": str(x.seq) for i, x in enumerate(parsed)}
    preprocess_fasta(seq_dict=records, filepath=local_tmp_path)

    fs = fsspec.filesystem("gs")
    fs.put(local_tmp_path, gcs_dest_path)

    # 4. Clean up the local temp file to save memory/disk
    os.remove(local_tmp_path)

    return {"message": "FASTA file uploaded successfully.", "gcs_path": gcs_dest_path}
