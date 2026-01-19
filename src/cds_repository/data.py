from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader, Dataset


class CDSDataset(Dataset):
    """Dataset for loading CSV.GZ files."""

    def __init__(self, data_path: Path) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to the CSV.GZ file.
        """
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data: extract sequences and labels."""
        # Check if data is already one-hot encoded (position_encoded format)
        if "pos_" in self.df.columns[0]:
            # Already encoded data - extract one-hot encoded features
            label_col = "label" if "label" in self.df.columns else self.df.columns[-1]
            self.labels = self.df[label_col].values.astype(np.float32)

            # Get all feature columns (everything except label)
            feature_cols = [col for col in self.df.columns if col != label_col]
            self.sequences = self.df[feature_cols].values.astype(np.float32)
        else:
            # Raw sequence data - one-hot encode it
            self.sequences = []
            self.labels = self.df["label"].values.astype(np.float32)

            for seq in self.df["nt_seq"]:
                encoded = self._encode_sequence(seq)
                self.sequences.append(encoded)
            self.sequences = np.array(self.sequences, dtype=np.float32)

    def _encode_sequence(self, seq: str) -> np.ndarray:
        """One-hot encode a DNA sequence.

        Args:
            seq: DNA sequence string.

        Returns:
            One-hot encoded sequence as (4, L) array.
        """
        mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
        encoded = np.zeros((4, len(seq)), dtype=np.float32)
        for i, nucleotide in enumerate(seq):
            if nucleotide in mapping:
                encoded[mapping[nucleotide], i] = 1.0
        return encoded

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a given sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (x, y) where x is the sequence tensor (4, L) and y is the label.
        """
        seq = self.sequences[index]
        label = self.labels[index]

        # Reshape position-encoded data from flat (N*4,) to (L, 4) then (4, L)
        if seq.ndim == 1:
            # Reshape from position-encoded format (pos_0_A, pos_0_T, pos_0_C, pos_0_G, pos_1_A, ...)
            seq = seq.reshape(-1, 4).T  # (4, L)
        elif seq.ndim == 2 and seq.shape[0] != 4:
            # Reshape from (L, 4) to (4, L)
            seq = seq.T

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.float32)

    def save(self, output_folder: Path) -> None:
        """Save the processed data with position-encoded nucleotides to the output folder.

        Args:
            output_folder: Path to save preprocessed data.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Use all sequences
        df_processed = self.df.copy()

        # One-hot encode each nucleotide at each position
        sequence_data = []
        for seq in df_processed["nt_seq"]:
            encoded_seq = []
            for nucleotide in seq:
                # One-hot encode: A=[1,0,0,0], T=[0,1,0,0], C=[0,0,1,0], G=[0,0,0,1]
                if nucleotide == "A":
                    encoded_seq.extend([1, 0, 0, 0])
                elif nucleotide == "T":
                    encoded_seq.extend([0, 1, 0, 0])
                elif nucleotide == "C":
                    encoded_seq.extend([0, 0, 1, 0])
                elif nucleotide == "G":
                    encoded_seq.extend([0, 0, 0, 1])
                elif nucleotide == "N":
                    encoded_seq.extend([0, 0, 0, 0])  # Unknown nucleotide / padding
            sequence_data.append(encoded_seq)

        # Create dataframe with encoded positions
        max_length = len(sequence_data[0])
        nucleotides = ["A", "T", "C", "G"]
        columns = [f"pos_{i // 4}_{nucleotides[i % 4]}" for i in range(max_length)]
        df_encoded = pd.DataFrame(sequence_data, columns=columns)
        df_encoded["label"] = df_processed["label"].values

        output_file = output_folder / self.data_path.name
        df_encoded.to_csv(output_file, index=False)
        print(f"Saved all sequences with position-encoded nucleotides (shape {df_encoded.shape}) to {output_file}")


def load_dataset(data_path: Path, split: str = "train") -> CDSDataset:
    """Load a dataset from a CSV.GZ file.

    Args:
        data_path: Path to the data directory containing CSV.GZ files.
        split: Data split to load ('train', 'val', or 'test').

    Returns:
        CSVDataset instance for the specified split.

    Raises:
        ValueError: If split is not one of 'train', 'val', or 'test'.
    """
    if split not in ["train", "val", "test"]:
        msg = f"split must be 'train', 'val', or 'test', got '{split}'"
        raise ValueError(msg)

    file_path = Path(data_path) / f"{split}.csv.gz"
    print(file_path)
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    return CDSDataset(file_path)


def load_dataloader(
    data_path: Path,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Load a DataLoader for a specific split.

    Args:
        data_path: Path to the data directory containing CSV.GZ files.
        split: Data split to load ('train', 'val', or 'test').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.
        num_workers: Number of workers for data loading.

    Returns:
        DataLoader instance for the specified split.
    """
    dataset = load_dataset(data_path, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
    )


def get_dataloaders(
    data_path: str | Path = "data/processed/training",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Load train and validation dataloaders.

    Args:
        data_path: Path to the data directory containing CSV.GZ files.
        batch_size: Number of samples per batch.
        num_workers: Number of workers for data loading.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_path = Path(data_path)
    train_loader = load_dataloader(
        data_path, split="train", batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = load_dataloader(data_path, split="val", batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess data from all splits.

    Args:
        data_path: Path to the data directory.
        output_folder: Path to save preprocessed data.
    """
    print("Preprocessing data...")
    for split in ["train", "val", "test"]:
        dataset = load_dataset(data_path, split)
        print(f"Processing {split} split...")
        dataset.save(output_folder)
        print(f"Finished processing {split} split.")


if __name__ == "__main__":
    typer.run(preprocess)
