from pathlib import Path

import pandas as pd
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

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        """Return a given sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionary containing the sample data.
        """
        row = self.df.iloc[index]
        return row.to_dict()

    def save(self, output_folder: Path) -> None:
        """Save the processed data with position-encoded nucleotides to the output folder.

        Args:
            output_folder: Path to save preprocessed data.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Use first 5000 sequences
        df_processed = self.df.head(5000).copy()

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
            sequence_data.append(encoded_seq)

        # Create dataframe with encoded positions
        max_length = len(sequence_data[0])
        nucleotides = ["A", "T", "C", "G"]
        columns = [f"pos_{i//4}_{nucleotides[i%4]}" for i in range(max_length)]
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
