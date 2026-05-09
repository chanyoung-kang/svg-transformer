"""
Small μP training smoke test.

This script checks that the μP model can train on the real encoded SVG token
data before launching the expensive μP learning-rate sweep.

It:
    1. Loads encoded train/val/test token streams.
    2. Creates a small μP GPT-style model.
    3. Applies μP base shapes.
    4. Uses MuAdam optimizer.
    5. Runs a few training steps.
    6. Estimates train/validation loss.
    7. Saves a small checkpoint.

This is only a sanity check, not the final Part 3 experiment.
"""

from pathlib import Path
import math
import time

import sentencepiece as spm
import torch
from mup import MuAdam

from dataset_loader import load_all_splits, get_batch
from mup_model import (
    GPTConfig,
    GPT,
    count_parameters,
    apply_mup_base_shapes,
)


TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")
CHECKPOINT_DIR = Path("outputs/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "mup_smoke_test_model.pt"


# Tiny local/Colab sanity settings
BATCH_SIZE = 4
BLOCK_SIZE = 128
MAX_ITERS = 20
EVAL_INTERVAL = 5
EVAL_ITERS = 2

MAX_LR = 1e-3
WARMUP_FRACTION = 0.1
MIN_LR_FRACTION = 0.1

N_LAYER = 2
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1

SEED = 42


def get_lr(step: int, max_iters: int, warmup_iters: int) -> float:
    """
    Warmup + cosine decay learning-rate schedule.
    """
    min_lr = MAX_LR * MIN_LR_FRACTION

    if step < warmup_iters:
        return MAX_LR * (step + 1) / max(1, warmup_iters)

    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    progress = min(1.0, max(0.0, progress))

    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_coeff * (MAX_LR - min_lr)


@torch.no_grad()
def estimate_loss(model, data, device):
    """
    Estimate train and validation loss without updating weights.
    """
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(EVAL_ITERS):
            x, y = get_batch(
                tokens=data[split],
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE,
                device=device,
            )

            _, loss = model(x, y)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))
    vocab_size = sp.get_piece_size()

    print(f"Tokenizer vocab size: {vocab_size:,}")

    print("Loading encoded token data...")
    data = load_all_splits()

    print(f"Train token IDs: {len(data['train']):,}")
    print(f"Val token IDs: {len(data['val']):,}")
    print(f"Test token IDs: {len(data['test']):,}")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )

    model = GPT(config)

    # μP requirement:
    # apply base shapes before creating the μP optimizer.
    model = apply_mup_base_shapes(model, config)
    model = model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    has_infshape = any(hasattr(p, "infshape") for p in model.parameters())
    print(f"Has μP infshape metadata: {has_infshape}")

    optimizer = MuAdam(
        model.parameters(),
        lr=MAX_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    warmup_iters = int(WARMUP_FRACTION * MAX_ITERS)
    start_time = time.time()

    for step in range(1, MAX_ITERS + 1):
        lr = get_lr(
            step=step,
            max_iters=MAX_ITERS,
            warmup_iters=warmup_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(
            tokens=data["train"],
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
        )

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, data, device)

            elapsed = time.time() - start_time

            print(
                f"step {step:4d}/{MAX_ITERS} | "
                f"lr {lr:.2e} | "
                f"batch loss {loss.item():.4f} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"elapsed {elapsed:.1f} sec"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "vocab_size": vocab_size,
        "train_settings": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "max_iters": MAX_ITERS,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "max_lr": MAX_LR,
            "warmup_fraction": WARMUP_FRACTION,
            "min_lr_fraction": MIN_LR_FRACTION,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "dropout": DROPOUT,
            "optimizer": "MuAdam",
        },
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nSaved checkpoint to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()