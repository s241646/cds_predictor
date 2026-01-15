import random
import pandas as pd
from pathlib import Path


def make_position_encoded_csv(
    path: Path,
    n: int = 200,
    seq_len: int = 50,
    seed: int = 0,
):
    import random
    import pandas as pd

    random.seed(seed)
    nucleotides = ["A", "T", "C", "G"]

    rows = []
    for _ in range(n):
        row = {}
        count_a = 0

        for i in range(seq_len):
            nt = random.choice(nucleotides)
            if nt == "A":
                count_a += 1

            # 👇 FIX: enforce A,T,C,G order
            for n in ["A", "T", "C", "G"]:
                row[f"pos_{i}_{n}"] = 1.0 if nt == n else 0.0

        row["label"] = 1 if count_a > seq_len // 2 else 0
        rows.append(row)

    # 👇 FIX: explicitly set column order
    feature_cols = [
        f"pos_{i}_{n}"
        for i in range(seq_len)
        for n in ["A", "T", "C", "G"]
    ]

    df = pd.DataFrame(rows)
    df = df[feature_cols + ["label"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
