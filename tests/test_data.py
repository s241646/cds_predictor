import pandas as pd
import pytest
import torch

from cds_repository.data import CDSDataset, load_dataset
from tests import _PATH_DATA
from pathlib import Path

RAW_PATH = Path(_PATH_DATA+"/raw")
PROCESSED_PATH = Path(_PATH_DATA+"/processed")

# ------------------------
# RAW DATA TESTS
# ------------------------

@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not found")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_raw_dataset_length_matches_csv(split):
    """Raw dataset length should match CSV rows."""
    dataset = load_dataset(RAW_PATH, split=split)
    df = pd.read_csv(RAW_PATH / f"{split}.csv.gz")

    assert len(dataset) == len(df), f"Dataset split {split} length does not match CSV rows."


@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not found")
def test_raw_dataset_encodes_to_one_hot():
    """
    Raw nt_seq strings must be encoded to (4, L).
    """
    dataset = load_dataset(RAW_PATH, split="train")
    df = pd.read_csv(RAW_PATH / "train.csv.gz")

    x, y = dataset[0]
    nt_seq = df.iloc[0]["nt_seq"]

    assert isinstance(x, torch.Tensor), "Input x is not a torch.Tensor"
    assert x.shape == (4, len(nt_seq)), f"Input x shape {x.shape} does not match expected (4, {len(nt_seq)})"
    assert y.ndim == 0, "Label y is not a scalar tensor (ndim != 0)"

@pytest.mark.skipif(not RAW_PATH.exists(), reason="Raw data not found")
def test_raw_nt_seq_characters_valid():
    """Ensure nt_seq contains only expected DNA bases."""
    df = pd.read_csv(RAW_PATH / "train.csv.gz")

    valid = set("ATCG")
    for seq in df["nt_seq"].head(100):  # sample for speed
        assert set(seq).issubset(valid), f"Invalid characters found in sequence: {seq}, should be A,T,C,G"


# ------------------------
# PROCESSED DATA TESTS
# ------------------------

@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="Processed data not found")
def test_processed_dataset_shape():
    """
    Processed data should already be one-hot encoded
    and reshaped to (4, L).
    """
    dataset = CDSDataset(PROCESSED_PATH / "train.csv.gz")

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor), "Input x is not a torch.Tensor"
    assert x.ndim == 2, f"Input x ndim {x.ndim} is not 2"
    assert x.shape[0] == 4, f"Input x shape[0] {x.shape[0]} is not 4"
    assert x.shape[1] > 0, f"Input x shape[1] {x.shape[1]} is not positive"
    assert y.ndim == 0, "Label y is not a scalar tensor (ndim != 0)"


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="Processed data not found")
def test_processed_one_hot_validity():
    """Processed data must be strictly one-hot."""
    dataset = CDSDataset(PROCESSED_PATH / "train.csv.gz")
    x, _ = dataset[0]

    column_sums = x.sum(dim=0)
    assert torch.all(column_sums == 1), "One-hot encoding invalid: columns do not sum to 1"


# ------------------------
# SHARED / SAFETY TESTS
# ------------------------

def test_invalid_split_raises():
    """Invalid split name should raise ValueError."""
    with pytest.raises(ValueError):
        load_dataset(_PATH_DATA, split="invalid")
