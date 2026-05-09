"""
Encode train, validation, and test SVG splits into binary token-ID files.

This script loads the train/validation/test Parquet splits, tokenizes each SVG
using the trained SentencePiece tokenizer, adds BOS/EOS boundary tokens around
each SVG, and saves the resulting token IDs as compact binary files for
Transformer training.

Inputs:
    data/processed/splits/train.parquet
    data/processed/splits/val.parquet
    data/processed/splits/test.parquet
    data/tokenizer/svg_bpe.model

Outputs:
    data/processed/encoded/train.bin
    data/processed/encoded/val.bin
    data/processed/encoded/test.bin
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm


SPLIT_DIR = Path("data/processed/splits")
TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")
OUT_DIR = Path("data/processed/encoded")

# special token IDs from our SentencePiece setup - add these so the model can learn SVG boundaries
BOS_ID = 1
EOS_ID = 2


# encodes one split: train, val, or test
def encode_split(split_name: str, sp: spm.SentencePieceProcessor):
    # .parquet file = readable table with SVG text and metadata
    # .bin file = compact binary array of token IDs
    in_path = SPLIT_DIR / f"{split_name}.parquet"
    out_path = OUT_DIR / f"{split_name}.bin"

    df = pd.read_parquet(in_path)

    """
    creates an empty list to store all token IDs for this split.
    It will become one long list like:
        [1, ..., 2, 1, ..., 2, 1, ..., 2, ...]
    Each SVG is wrapped with <bos> and <eos>.
    """
    all_ids = []

    for svg in df["svg"]:
        ids = sp.encode(svg, out_type=int)

        # Add BOS/EOS so the model learns SVG boundaries
        # <bos> SVG tokens <eos>
        all_ids.extend([BOS_ID])
        all_ids.extend(ids)
        all_ids.extend([EOS_ID])

    # convert the Python list into a NumPy array
    # dtype=np.uint16 means each token ID is stored as a 16-bit unsigned integer
    # vocab size is around 2,498 and uint16 can store values up to 65,535
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_path)

    print(f"{split_name}")
    print(f"  SVGs: {len(df):,}")
    print(f"  Token IDs saved: {len(arr):,}")
    print(f"  Saved to: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))

    for split_name in ["train", "val", "test"]:
        encode_split(split_name, sp)


if __name__ == "__main__":
    main()