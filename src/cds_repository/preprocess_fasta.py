import pandas as pd
import typer


def preprocess_fasta(seq_dict: dict, max_length: int | None = None) -> None:
    """Preprocess a FASTA file into position-encoded CSV format for prediction.

    Args:
        seq_dict: Dictionary of sequence id, sequences key-value pairs).
        max_length: Maximum sequence length. If None, uses the longest sequence.
                   Sequences shorter than this will be padded with zeros.
    """

    # Determine max_length if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in seq_dict.values())
        print(f"Using max sequence length: {max_length}")
    else:
        print(f"Using specified max length: {max_length}")

    # One-hot encode each nucleotide at each position
    # This follows the same logic as CDSDataset.save()
    sequence_data = []
    for seq in seq_dict.values():
        # Truncate or pad sequence to max_length
        if len(seq) > max_length:
            seq = seq[:max_length]
        elif len(seq) < max_length:
            # Pad with spaces (will be encoded as zeros)
            seq = seq + " " * (max_length - len(seq))

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
            else:
                # Unknown nucleotides or padding encoded as zeros
                encoded_seq.extend([0, 0, 0, 0])
        sequence_data.append(encoded_seq)

    # Create dataframe with encoded positions
    # Column naming follows the same pattern as CDSDataset.save()
    nucleotides = ["A", "T", "C", "G"]
    columns = [f"pos_{i // 4}_{nucleotides[i % 4]}" for i in range(max_length * 4)]
    df_encoded = pd.DataFrame(sequence_data, columns=columns)

    # Add sequence IDs as first column (optional, useful for tracking predictions)
    df_encoded.insert(0, "seq_id", list(seq_dict.keys()))

    # Save to CSV (matching the format from CDSDataset.save())
    df_encoded.to_csv("../../data/tmp/preprocessed.csv.gz", index=False, compression="gzip")


def main() -> None:
    """Preprocess FASTA file for CDS prediction.

    Converts DNA sequences from FASTA format to position-encoded CSV format
    compatible with the CDS prediction model.
    """
    preprocess_fasta(seq_dict={})


if __name__ == "__main__":
    typer.run(main)
