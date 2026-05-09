"""
Create train, validation, and test splits by SVG file.

This script randomly shuffles the filtered trainable SVG dataset and splits it
into 98% train, 1% validation, and 1% test by SVG file. Splitting by file avoids
leakage that could happen if token positions from the same SVG appeared in
multiple splits.

Input:
    data/processed/svg_combined_trainable_1024.parquet

Outputs:
    data/processed/splits/train.parquet
    data/processed/splits/val.parquet
    data/processed/splits/test.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd


IN_PATH = Path("data/processed/svg_combined_trainable_1024.parquet")
OUT_DIR = Path("data/processed/splits")

SEED = 42

TRAIN_FRAC = 0.98
VAL_FRAC = 0.01
TEST_FRAC = 0.01

# summarize basic statistics for each split
def summarize(name, df):
    print(f"\n{name}")
    print(f"Rows: {len(df):,}")
    print(f"Tokens: {df['token_count'].sum():,}")
    print(f"Mean tokens: {df['token_count'].mean():.1f}")
    print(f"Median tokens: {df['token_count'].median():.1f}")
    print(f"Max tokens: {df['token_count'].max():,}")


def main():
    df = pd.read_parquet(IN_PATH)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(df)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUT_DIR / "val.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)

    summarize("Train", train_df)
    summarize("Validation", val_df)
    summarize("Test", test_df)

    print(f"\nSaved splits to: {OUT_DIR}")


if __name__ == "__main__":
    main()