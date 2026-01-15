from fastapi import FastAPI
import torch
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from cds_repository.model import build_model

app = FastAPI(title="CDS Predictor API")

model = None
device = None


class SequenceInput(BaseModel):
    """Input model for prediction request."""

    sequence: str


@app.on_event("startup")
def load_model():
    global model, device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_path = Path("models/motifcnn.pt")
    if checkpoint_path.exists():
        model = build_model(channels=(128, 256, 512), dropout=0.2)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
        model.to(device)
        model.eval()


def encode_sequence(seq: str) -> np.ndarray:
    """One-hot encode a DNA sequence."""
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
    encoded = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            encoded[mapping[nucleotide], i] = 1.0
    return encoded


@app.get("/")
def read_root():
    return {"message": "CDS Predictor API", "status": "ready" if model is not None else "model not loaded"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: SequenceInput):
    """Predict CDS probability for a DNA sequence."""
    if model is None:
        return {"error": "Model not loaded"}

    # Encode sequence
    encoded = encode_sequence(data.sequence)
    x = torch.from_numpy(encoded).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probability = torch.sigmoid(logits).item()

    return {
        "sequence": data.sequence,
        "length": len(data.sequence),
        "logits": logits.item(),
        "probability": probability,
    }
