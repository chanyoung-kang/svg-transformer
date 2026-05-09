"""
Filter SVGs by maximum token sequence length.

This script loads the cleaned SVG dataset with token counts and keeps only SVGs
whose token_count is less than or equal to the chosen maximum sequence length.
This creates the final trainable SVG dataset used for train/validation/test
splitting.

Input:
    data/processed/svg_combined_clean_with_tokens.parquet

Output:
    data/processed/svg_combined_trainable_1024.parquet
"""

from pathlib import Path
import pandas as pd

IN_PATH = Path("data/processed/svg_combined_clean_with_tokens.parquet")
OUT_PATH = Path("data/processed/svg_combined_trainable_1024.parquet")

MAX_TOKENS = 1024


def main():
    df = pd.read_parquet(IN_PATH)

    before_rows = len(df)
    before_tokens = df["token_count"].sum()

    filtered = df[df["token_count"] <= MAX_TOKENS].copy()

    after_rows = len(filtered)
    after_tokens = filtered["token_count"].sum()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUT_PATH, index=False)

    print(f"Input rows: {before_rows:,}")
    print(f"Input tokens: {before_tokens:,}")
    print(f"Kept rows: {after_rows:,}")
    print(f"Kept tokens: {after_tokens:,}")
    print(f"Removed rows: {before_rows - after_rows:,}")
    print(f"Removed tokens: {before_tokens - after_tokens:,}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()