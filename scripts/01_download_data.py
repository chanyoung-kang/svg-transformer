"""
Download and combine SVG datasets from Hugging Face.

This script loads the primary SVG icon dataset and supplementary SVG datasets,
standardizes them into a common schema, and saves the combined raw dataset as
a Parquet file.

Output:
    data/raw/svg_combined_raw.parquet

Columns:
    source   - dataset source name
    filename - original SVG/file identifier
    svg      - raw SVG XML string
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset # imports Hugging Face’s dataset-loading function
from tqdm import tqdm # progress bar


# Create data/raw
OUT_DIR = Path("data/raw")
OUT_PATH = OUT_DIR / "svg_combined_raw.parquet"

# Main dataset
PRIMARY_DATASET = "starvector/svg-icons-simple"

# Supplementary datasets
EMOJI_DATASET = "starvector/svg-emoji-simple"
FONTS_DATASET = "starvector/svg-fonts-simple"

# Start with a subset. We can increase this later if token count is still below 100M.
MAX_FONT_ROWS = 500_000


def load_starvector_dataset(dataset_name: str, source_name: str) -> pd.DataFrame:
    """
    Load a StarVector SVG dataset with columns:
    Filename, Svg
    """
    print(f"\nLoading full dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)
    df = dataset["train"].to_pandas()

    out = pd.DataFrame({
        "source": [source_name] * len(df),
        "filename": df["Filename"].astype(str),
        "svg": df["Svg"].astype(str),
    })

    print(f"Loaded {len(out):,} rows from {dataset_name}")
    return out


def load_starvector_streaming_subset(
    dataset_name: str,
    source_name: str,
    max_rows: int,
) -> pd.DataFrame:
    """
    Stream only the first max_rows rows from a large StarVector SVG dataset.
    This avoids downloading/loading the entire dataset into memory.
    """
    print(f"\nStreaming first {max_rows:,} rows from: {dataset_name}")

    stream = load_dataset(dataset_name, split="train", streaming=True)

    rows = []

    for i, row in enumerate(tqdm(stream, total=max_rows)):
        if i >= max_rows:
            break

        rows.append(
            {
                "source": source_name,
                "filename": str(row["Filename"]),
                "svg": str(row["Svg"]),
            }
        )

    out = pd.DataFrame(rows)

    print(f"Loaded {len(out):,} rows from {dataset_name}")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    icons_df = load_starvector_dataset(
        PRIMARY_DATASET,
        "svg-icons-simple",
    )

    emoji_df = load_starvector_dataset(
        EMOJI_DATASET,
        "svg-emoji-simple",
    )

    fonts_df = load_starvector_streaming_subset(
        FONTS_DATASET,
        "svg-fonts-simple",
        MAX_FONT_ROWS,
    )

    combined = pd.concat(
        [icons_df, emoji_df, fonts_df],
        ignore_index=True,
    )

    combined.to_parquet(OUT_PATH, index=False)

    print("\nCombined dataset summary:")
    print(combined["source"].value_counts())

    print(f"\nTotal rows: {len(combined):,}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()