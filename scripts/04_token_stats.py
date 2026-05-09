"""
Compute token-count statistics for each cleaned SVG.

This script loads the trained SentencePiece tokenizer and applies it to every
cleaned SVG. It records the number of tokens per SVG, prints corpus-level token
statistics, and saves a new Parquet file with a token_count column.

Inputs:
    data/processed/svg_combined_clean.parquet
    data/tokenizer/svg_bpe.model

Output:
    data/processed/svg_combined_clean_with_tokens.parquet

Output columns:
    source, filename, svg, char_count, token_count
"""

from pathlib import Path

import numpy as np
import pandas as pd
# from tokenizers import ByteLevelBPETokenizer
import sentencepiece as spm


DATA_PATH = Path("data/processed/svg_combined_clean.parquet")
# TOKENIZER_DIR = Path("data/tokenizer")
# VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
# MERGES_PATH = TOKENIZER_DIR / "merges.txt"
 
TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")
OUT_PATH = Path("data/processed/svg_combined_clean_with_tokens.parquet")


def main():
    # Load cleaned SVG dataset
    df = pd.read_parquet(DATA_PATH)

    # tokenizer = ByteLevelBPETokenizer(
    #     str(VOCAB_PATH),
    #     str(MERGES_PATH),
    # )

    # Load trained SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))

    token_counts = []

    # Count how many tokens each SVG becomes
    for svg in df["svg"]:
        # ids = tokenizer.encode(svg).ids
        ids = sp.encode(svg, out_type=int)
        token_counts.append(len(ids))

    token_counts = np.array(token_counts)

    print(f"Number of SVGs: {len(token_counts):,}")
    print(f"Total tokens: {token_counts.sum():,}")
    print(f"Mean tokens per SVG: {token_counts.mean():.1f}")
    print(f"Median tokens per SVG: {np.median(token_counts):.1f}")
    print(f"Min tokens: {token_counts.min():,}")
    print(f"Max tokens: {token_counts.max():,}")
    print(f"SVGs over 512 tokens: {(token_counts > 512).sum():,}")
    print(f"SVGs over 1024 tokens: {(token_counts > 1024).sum():,}")
    print(f"SVGs over 2048 tokens: {(token_counts > 2048).sum():,}")

    # Save stats back into the cleaned dataframe
    df["token_count"] = token_counts
    # out_path = Path("data/processed/svg_icons_clean_with_tokens.parquet")
    # df.to_parquet(out_path, index=False)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved token counts to: {OUT_PATH}")


if __name__ == "__main__":
    main()