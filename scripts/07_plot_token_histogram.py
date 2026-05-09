"""
Generate a sequence length histogram for the report.

This script plots the distribution of SVG sequence lengths, where sequence
length is measured as the number of SentencePiece tokens per SVG. It also shows
the 1024-token cutoff used for the trainable dataset.

Inputs:
    data/processed/svg_combined_clean_with_tokens.parquet
    data/processed/svg_combined_trainable_1024.parquet

Output:
    outputs/plots/token_length_histogram.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# full cleaned dataset path
IN_PATH = Path("data/processed/svg_combined_clean_with_tokens.parquet")
# dataset after removing SVGs over 1024 tokens
FILTERED_PATH = Path("data/processed/svg_combined_trainable_1024.parquet")

OUT_DIR = Path("outputs/plots")
OUT_PATH = OUT_DIR / "token_length_histogram.png"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_parquet(IN_PATH)
    df_filtered = pd.read_parquet(FILTERED_PATH)

    plt.figure(figsize=(10, 6))

    bins = range(0, 1601, 25)

    plt.hist(
        df_all["token_count"],
        bins=bins,
        alpha=0.6,
        label="All cleaned SVGs",
        color="blue",
    )

    plt.axvline(
        1024,
        linestyle="--",
        label="1024-token cutoff",
    )

    plt.xlabel("Token count per SVG")
    plt.ylabel("Number of SVG files")
    plt.title("SVG Sequence Length Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_PATH, dpi=200)

    print(f"All cleaned SVGs: {len(df_all):,}")
    print(f"Filtered SVGs: {len(df_filtered):,}")
    print(f"Saved histogram to: {OUT_PATH}")


if __name__ == "__main__":
    main()