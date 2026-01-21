import pandas as pd
import typer


def preprocess_fasta(seq_dict: dict, max_length: int | None = 300, filepath: str = "data/tmp/test.csv.gz") -> None:
    """Preprocess a FASTA file into CSV format for prediction.

    Args:
        seq_dict: Dictionary of sequence id, sequences key-value pairs).
        max_length: Maximum sequence length. If None, no length constraint is applied.
    """
    # Check for empty sequence dictionary
    if not seq_dict:
        raise ValueError("seq_dict cannot be empty")

    # Create list to store rows
    data = []

    for seq_id, nt_seq in seq_dict.items():
        # Create row with required columns
        if max_length and len(nt_seq) > max_length:
            nt_seq = nt_seq[:max_length]
        elif len(nt_seq) < max_length:
            nt_seq = nt_seq.ljust(max_length, "N")
        row = {"id": seq_id, "nt_seq": nt_seq, "label": -1, "seq_desc": -1}
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filepath, index=False, compression="gzip")


def main() -> None:
    """Preprocess FASTA file for CDS prediction.

    Converts DNA sequences from FASTA format to CSV format
    compatible with the CDS prediction model.
    """
    preprocess_fasta(seq_dict={})


if __name__ == "__main__":
    typer.run(main)
