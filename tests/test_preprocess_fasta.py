from pathlib import Path
import pandas as pd
import pytest

from cds_repository.preprocess_fasta import preprocess_fasta


@pytest.fixture
def setup_output_dir():
    """Create output directory before tests."""
    output_path = Path("../../data/tmp")
    output_path.mkdir(parents=True, exist_ok=True)
    yield output_path
    # Cleanup after test
    output_file = output_path / "preprocessed.csv.gz"
    if output_file.exists():
        output_file.unlink()


def test_basic_one_hot_encoding(setup_output_dir):
    """Test that nucleotides are correctly one-hot encoded."""
    seq_dict = {"seq1": "ATCG"}
    preprocess_fasta(seq_dict, max_length=4)

    # Read the output file
    df = pd.read_csv("../../data/tmp/preprocessed.csv.gz")

    # Verify A, T, C, G are correctly encoded at each position
    assert df.loc[0, "pos_0_A"] == 1 and df.loc[0, "pos_0_T"] == 0
    assert df.loc[0, "pos_1_T"] == 1 and df.loc[0, "pos_1_A"] == 0
    assert df.loc[0, "pos_2_C"] == 1 and df.loc[0, "pos_2_G"] == 0
    assert df.loc[0, "pos_3_G"] == 1 and df.loc[0, "pos_3_A"] == 0


def test_padding_short_sequences(setup_output_dir):
    """Test that short sequences are padded with zeros."""
    seq_dict = {"seq1": "AT"}
    preprocess_fasta(seq_dict, max_length=4)

    df = pd.read_csv("../../data/tmp/preprocessed.csv.gz")

    # First two positions encoded, last two should be all zeros
    assert df.loc[0, "pos_0_A"] == 1 and df.loc[0, "pos_1_T"] == 1
    for pos in [2, 3]:
        for nuc in ["A", "T", "C", "G"]:
            assert df.loc[0, f"pos_{pos}_{nuc}"] == 0


def test_truncating_long_sequences(setup_output_dir):
    """Test that sequences longer than max_length are truncated."""
    seq_dict = {"seq1": "ATCGATCG"}
    preprocess_fasta(seq_dict, max_length=4)

    df = pd.read_csv("../../data/tmp/preprocessed.csv.gz")

    assert df.shape[1] == 4 * 4 + 1
    # Only first 4 nucleotides should be encoded
    assert df.loc[0, "pos_0_A"] == 1 and df.loc[0, "pos_3_G"] == 1


def test_auto_detect_max_length(setup_output_dir):
    """Test automatic detection of max_length from sequences."""
    seq_dict = {"seq1": "AT", "seq2": "ATCGATCG"}
    preprocess_fasta(seq_dict, max_length=None)

    df = pd.read_csv("../../data/tmp/preprocessed.csv.gz")

    assert df.shape[1] == 8 * 4 + 1


def test_unknown_nucleotide_encoded_as_zeros(setup_output_dir):
    """Test that unknown nucleotides are encoded as all zeros."""
    seq_dict = {"seq1": "ATNC"}
    preprocess_fasta(seq_dict, max_length=4)

    df = pd.read_csv("../../data/tmp/preprocessed.csv.gz")

    # Position 2 has 'N' (unknown) - should be all zeros
    for nuc in ["A", "T", "C", "G"]:
        assert df.loc[0, f"pos_2_{nuc}"] == 0


def test_empty_sequence_dict():
    """Test that empty sequence dict raises ValueError."""
    with pytest.raises(ValueError):
        preprocess_fasta({}, max_length=None)
