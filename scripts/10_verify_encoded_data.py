"""
Verify encoded binary token-ID files.

This script reads the train/validation/test .bin files, checks that they can be
loaded correctly, confirms that token IDs are within the tokenizer vocabulary
range, and verifies that BOS/EOS boundary tokens are present.

Inputs:
    data/processed/encoded/train.bin
    data/processed/encoded/val.bin
    data/processed/encoded/test.bin
    data/tokenizer/svg_bpe.model
"""

from pathlib import Path

import numpy as np
import sentencepiece as spm


ENCODED_DIR = Path("data/processed/encoded")
TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")

BOS_ID = 1
EOS_ID = 2


def check_split(name, vocab_size):
    path = ENCODED_DIR / f"{name}.bin"
    arr = np.fromfile(path, dtype=np.uint16)

    print(f"\n{name}")
    print(f"Path: {path}")
    print(f"Number of token IDs: {len(arr):,}")
    print(f"Min token ID: {arr.min()}")
    print(f"Max token ID: {arr.max()}")
    print(f"First 20 token IDs: {arr[:20].tolist()}")

    if arr.max() >= vocab_size:
        print(f"WARNING: max token ID is >= vocab size ({vocab_size})")
    else:
        print(f"Token ID range OK: max ID < vocab size ({vocab_size})")

    print(f"Starts with BOS_ID ({BOS_ID}): {arr[0] == BOS_ID}")
    print(f"Contains EOS_ID ({EOS_ID}): {(arr == EOS_ID).any()}")


def main():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))
    vocab_size = sp.get_piece_size()

    print(f"Tokenizer vocab size: {vocab_size:,}")

    for name in ["train", "val", "test"]:
        check_split(name, vocab_size)


if __name__ == "__main__":
    main()