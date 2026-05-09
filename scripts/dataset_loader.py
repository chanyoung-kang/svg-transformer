"""
Load encoded token-ID data and create next-token prediction batches.

This script reads the binary token-ID files created by 09_encode_splits.py and
provides helper functions for sampling x/y batches for Transformer training.

Input files:
    data/processed/encoded/train.bin
    data/processed/encoded/val.bin
    data/processed/encoded/test.bin

Each .bin file is a long sequence of uint16 token IDs:
    <bos> SVG_1_tokens <eos> <bos> SVG_2_tokens <eos> <bos> SVG_3_tokens <eos> ...
    Using IDs: 1 = <bos> and 2 = <eos>, each SVG boundary is marked by: 1 ... 2

Because of this, each training example in a batch is not guaranteed to be one
complete SVG. get_batch() randomly cuts fixed-length windows from this long
stream. A window may contain the middle of one SVG, the end of one SVG plus the
start of another, or a complete short SVG.

For next-token prediction:
    x = tokens[t : t + block_size]
    y = tokens[t + 1 : t + block_size + 1]
"""

from pathlib import Path

import numpy as np
import torch


ENCODED_DIR = Path("data/processed/encoded")

TRAIN_PATH = ENCODED_DIR / "train.bin"
VAL_PATH = ENCODED_DIR / "val.bin"
TEST_PATH = ENCODED_DIR / "test.bin"


def load_tokens(path: Path) -> np.ndarray:
    """
    Load a binary token-ID file as a NumPy array.

    The .bin files were saved as uint16 in 09_encode_splits.py, so we must read
    them back using the same dtype.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find token file: {path}")

    tokens = np.fromfile(path, dtype=np.uint16)

    if len(tokens) == 0:
        raise ValueError(f"Token file is empty: {path}")

    return tokens


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of input/target sequences.

    x contains block_size tokens.
    y contains the same sequence shifted one position to the right.

    Example:
        tokens = [10, 20, 30, 40, 50]

        if x starts at index 0 and block_size = 3:
            x = [10, 20, 30]
            y = [20, 30, 40]

    The model learns to predict each next token in y from the tokens in x.
    """
    max_start = len(tokens) - block_size - 1

    if max_start <= 0:
        raise ValueError(
            f"Token array is too short for block_size={block_size}. "
            f"Number of tokens: {len(tokens):,}"
        )

    # This randomly chooses batch_size starting positions
    starts = np.random.randint(0, max_start, size=batch_size)

    x_batch = np.stack([
        tokens[start : start + block_size]
        for start in starts
    ])

    y_batch = np.stack([
        tokens[start + 1 : start + block_size + 1]
        for start in starts
    ])

    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)

    return x, y


def load_all_splits() -> dict[str, np.ndarray]:
    """
    Load train, validation, and test token arrays.
    """
    return {
        "train": load_tokens(TRAIN_PATH),
        "val": load_tokens(VAL_PATH),
        "test": load_tokens(TEST_PATH),
    }


def main():
    """
    Small local sanity check.

    This does not train a model. It only confirms that we can load token files
    and create x/y batches.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_all_splits()

    print(f"Device: {device}")

    for split_name, tokens in data.items():
        print(f"{split_name}: {len(tokens):,} token IDs")

    batch_size = 4 # 4 examples in one batch
    block_size = 128 # 128 tokens per example

    x, y = get_batch(
        tokens=data["train"],
        batch_size=batch_size,
        block_size=block_size,
        device=device,
    )

    print("\nSample batch")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"x first row, first 20 tokens: {x[0, :20].tolist()}")
    print(f"y first row, first 20 tokens: {y[0, :20].tolist()}")


if __name__ == "__main__":
    main()