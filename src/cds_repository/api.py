import io
import os
import time
import threading
from uuid import uuid4
from pathlib import Path
import fsspec
import json
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
from Bio import SeqIO

import torch
from pytorch_lightning import Trainer

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.logger import logger
from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field

from cds_repository.model import MotifCNNModule
from cds_repository.data import load_dataloader
from cds_repository.preprocess_fasta import preprocess_fasta

try:
    import resource
except ImportError:
    resource = None

# ----------------------------
# Configuration
# ----------------------------
MODEL_CKPT_PATH_ENV = "MODEL_CKPT_PATH"
DEFAULT_THRESHOLD_ENV = "DEFAULT_THRESHOLD"  # optional; defaults to 0.5
MAX_BATCH_ENV = "MAX_BATCH"  # optional; defaults to 256
MIN_LENGTH_ENV = "MIN_LENGTH"  # optional; defaults to 1
MAX_LENGTH_ENV = "MAX_LENGTH"  # optional; defaults to 10_000
ALLOW_2D_SINGLE_ENV = "ALLOW_2D_SINGLE"  # optional; defaults to "true"


TMP_DIR = Path("/tmp/data")
TMP_DIR.mkdir(parents=True, exist_ok=True)


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

_METRICS_LOCK = threading.Lock()
_METRICS: Dict[str, Any] = {
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0,
    "total_latency_ms": 0.0,
    "last_latency_ms": 0.0,
}

# ----------------------------
# Core model load / validation
# ----------------------------


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
# Output:Save results to GC
# ----------------------------


def save_predictions(prediction_data: PredictResponse, gcs_dest_path: str):
    """
    Serializes PredictResponse to JSON and saves directly to GCS.
    """

    fs = fsspec.filesystem("gs")
    results_dict = prediction_data.model_dump()

    with fs.open(gcs_dest_path, "w") as f:
        json.dump(results_dict, f)


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
    app.state.trainer = Trainer(accelerator="cpu", devices=1)


@app.middleware("http")
async def collect_metrics(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Collects basic request metrics for the API.
    """
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        with _METRICS_LOCK:
            _METRICS["request_count"] += 1
            _METRICS["error_count"] += 1
            _METRICS["total_latency_ms"] += latency_ms
            _METRICS["last_latency_ms"] = latency_ms
        raise

    latency_ms = (time.perf_counter() - start) * 1000.0
    with _METRICS_LOCK:
        _METRICS["request_count"] += 1
        if response.status_code >= 500:
            _METRICS["error_count"] += 1
        _METRICS["total_latency_ms"] += latency_ms
        _METRICS["last_latency_ms"] = latency_ms
    return response


# ----------------------------
# Endpoints
# ----------------------------


@app.get("/")
def read_root():
    return {"message": "CDS Predictor API is live", "docs": "/docs", "health": "/health"}


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

    unique_id = uuid4()
    local_tmp_path = f"{TMP_DIR}/uploaded_{unique_id}.csv.gz"
    gcs_dest_path = f"gs://cds-predictor/uploaded_data/input_{unique_id}.csv.gz"
    gcs_output_path = f"gs://cds-predictor/predictions/preds_{unique_id}.json"

    # Store original data for mapping
    fasta_records = [{"id": x.id, "seq": str(x.seq)} for x in parsed]

    # Create unique keys for preprocessor
    records = {f"{x.id}_{i}": str(x.seq) for i, x in enumerate(parsed)}
    preprocess_fasta(seq_dict=records, filepath=local_tmp_path)

    effective_batch = min(batch_size, MAX_BATCH)
    test_loader = load_dataloader(TMP_DIR, split=f"uploaded_{unique_id}", batch_size=effective_batch)

    # Run inference
    predictions = app.state.trainer.predict(app.state.model, dataloaders=test_loader)
    if not predictions:
        raise HTTPException(status_code=500, detail="Model returned no predictions.")

    all_preds = torch.cat([batch["preds"] for batch in predictions]).cpu().tolist()

    # Only flatten these if the flags are true
    all_logits = torch.cat([batch["logits"] for batch in predictions]).cpu().tolist() if return_logits else None
    all_probs = torch.cat([batch["probs"] for batch in predictions]).cpu().tolist() if return_probs else None

    results = []

    num_to_process = min(len(all_preds), len(fasta_records))
    # Use the length of all_preds to prevent IndexError
    for i in range(num_to_process):
        # Truncate sequence for UI visibility
        orig_seq = fasta_records[i]["seq"]
        display_seq = orig_seq[:50] + "..." if len(orig_seq) > 50 else orig_seq

        item = PredictResponseItem(id=str(fasta_records[i]["id"]), sequence=display_seq, pred=int(all_preds[i]))

        if return_probs and all_probs is not None:
            item.prob = float(all_probs[i])
        if return_logits and all_logits is not None:
            item.logit = float(all_logits[i])

        results.append(item)

    fs = fsspec.filesystem("gs")
    fs.put(local_tmp_path, gcs_dest_path)

    # Clean up the local temp file to save memory/disk
    os.remove(local_tmp_path)
    save_predictions(PredictResponse(results=results), gcs_output_path)

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


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """
    Returns basic system and request metrics for the API.
    """
    load_avg = None
    if hasattr(os, "getloadavg"):
        load_avg = os.getloadavg()[0]

    max_rss = None
    if resource is not None:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with _METRICS_LOCK:
        request_count = int(_METRICS["request_count"])
        error_count = int(_METRICS["error_count"])
        total_latency_ms = float(_METRICS["total_latency_ms"])
        last_latency_ms = float(_METRICS["last_latency_ms"])
        uptime_seconds = time.time() - float(_METRICS["start_time"])

    avg_latency_ms = total_latency_ms / request_count if request_count else 0.0
    return {
        "uptime_seconds": uptime_seconds,
        "request_count": request_count,
        "error_count": error_count,
        "avg_latency_ms": avg_latency_ms,
        "last_latency_ms": last_latency_ms,
        "load_avg_1m": load_avg,
        "max_rss": max_rss,
    }


@app.post("/upload_input_fasta")
async def upload_input_fasta(
    fasta: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Endpoint to upload an input FASTA file and save it to a gcloud location.
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

    # Clean up the local temp file to save memory/disk
    os.remove(local_tmp_path)

    return {"message": "FASTA file uploaded successfully.", "gcs_path": gcs_dest_path}
