"""
Learning-rate sweep for the smallest SVG Transformer model.

This script trains the same small decoder-only Transformer multiple times,
each time with a different peak learning rate.

Each run uses:
    warmup -> cosine decay

The goal is to choose the best peak learning rate based on validation loss.
The selected learning rate will then be reused for the full scaling study.

Important:
    This script trains the smallest model only.
    It does NOT train different model sizes yet.

Inputs:
    data/processed/encoded/train.bin
    data/processed/encoded/val.bin
    data/tokenizer/svg_bpe.model
    scripts/dataset_loader.py
    scripts/model.py

Outputs:
    outputs/lr_sweep_results.csv
    outputs/checkpoints/lr_sweep_<learning_rate>.pt
"""

from pathlib import Path
import csv
import math
import time

import sentencepiece as spm
import torch

from dataset_loader import load_all_splits, get_batch
from model import GPTConfig, GPT, count_parameters


TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_PATH = OUTPUT_DIR / "lr_sweep_results.csv"


# Candidate peak learning rates.
# These are roughly log-spaced.
LEARNING_RATES = [
    6e-4,
    1e-3,
    1.5e-3,
    2e-3,
    3e-3,
    4e-3,
    6e-3,
]

# Training setup
BATCH_SIZE = 32
BLOCK_SIZE = 256

# Warmup is a fraction of one epoch.
WARMUP_FRACTION = 0.05

EVAL_INTERVAL = 1000
EVAL_ITERS = 20

MIN_LR_FRACTION = 0.10


# Smallest model for LR sweep
N_LAYER = 2
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1

SEED = 42


def get_lr(step: int, max_lr: float, max_iters: int, warmup_iters: int) -> float:
    """
    Warmup + cosine decay learning-rate schedule.

    During warmup:
        learning rate increases linearly from 0 to max_lr.

    After warmup:
        learning rate decays smoothly from max_lr to min_lr.
    """
    min_lr = max_lr * MIN_LR_FRACTION

    if step < warmup_iters:
        return max_lr * (step + 1) / max(1, warmup_iters)

    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    progress = min(1.0, max(0.0, progress))

    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + cosine_coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data, device):
    """
    Estimate train and validation loss using random batches.

    This does not update model weights.
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


def save_results(results):
    """
    Save LR sweep results to CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "max_lr",
        "param_count",
        "max_iters",
        "warmup_iters",
        "final_train_loss",
        "final_val_loss",
        "elapsed_sec",
        "tokens_seen",
        "tokens_per_sec",
        "gpu_memory_mb",
        "checkpoint_path",
    ]

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved LR sweep results to: {RESULTS_PATH}")


def train_one_lr(
    max_lr: float,
    data,
    vocab_size: int,
    device: str,
    max_iters: int,
    warmup_iters: int,
) -> dict:
    """
    Train the smallest model using one candidate peak learning rate.
    """
    torch.manual_seed(SEED)

    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )

    model = GPT(config).to(device)
    param_count = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    print("\n" + "=" * 80)
    print(f"LR sweep run: max_lr={max_lr:.1e}")
    print(f"Model parameters: {param_count:,}")
    print(f"Max iters / one epoch: {max_iters:,}")
    print(f"Warmup iters: {warmup_iters:,}")
    print("=" * 80)

    start_time = time.time()
    last_train_loss = None
    last_val_loss = None

    for step in range(1, max_iters + 1):
        lr = get_lr(
            step=step,
            max_lr=max_lr,
            max_iters=max_iters,
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

        if step == 1 or step % EVAL_INTERVAL == 0 or step == max_iters:
            losses = estimate_loss(model, data, device)

            elapsed = time.time() - start_time
            tokens_seen = step * BATCH_SIZE * BLOCK_SIZE
            tokens_per_sec = tokens_seen / max(1e-8, elapsed)

            last_train_loss = losses["train"]
            last_val_loss = losses["val"]

            print(
                f"step {step:6d}/{max_iters:,} | "
                f"lr {lr:.2e} | "
                f"batch loss {loss.item():.4f} | "
                f"train loss {last_train_loss:.4f} | "
                f"val loss {last_val_loss:.4f} | "
                f"tokens/sec {tokens_per_sec:,.0f} | "
                f"elapsed {elapsed / 60:.1f} min"
            )

    elapsed = time.time() - start_time
    tokens_seen = max_iters * BATCH_SIZE * BLOCK_SIZE
    tokens_per_sec = tokens_seen / max(1e-8, elapsed)

    gpu_memory_mb = None
    if device == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    checkpoint_path = CHECKPOINT_DIR / f"lr_sweep_{max_lr:.0e}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size,
        "max_lr": max_lr,
        "train_settings": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "max_iters": max_iters,
            "warmup_iters": warmup_iters,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "min_lr_fraction": MIN_LR_FRACTION,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "dropout": DROPOUT,
        },
    }

    torch.save(checkpoint, checkpoint_path)

    result = {
        "max_lr": max_lr,
        "param_count": param_count,
        "max_iters": max_iters,
        "warmup_iters": warmup_iters,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "elapsed_sec": elapsed,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": tokens_per_sec,
        "gpu_memory_mb": gpu_memory_mb,
        "checkpoint_path": str(checkpoint_path),
    }

    del model
    del optimizer

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))
    vocab_size = sp.get_piece_size()

    print(f"Tokenizer vocab size: {vocab_size:,}")

    print("Loading encoded token data...")
    data = load_all_splits()

    print(f"Train token IDs: {len(data['train']):,}")
    print(f"Val token IDs: {len(data['val']):,}")
    print(f"Test token IDs: {len(data['test']):,}")

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    max_iters = len(data["train"]) // tokens_per_step
    warmup_iters = int(WARMUP_FRACTION * max_iters)

    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Calculated one-epoch steps: {max_iters:,}")
    print(f"Warmup fraction: {WARMUP_FRACTION}")
    print(f"Warmup steps: {warmup_iters:,}")

    results = []

    for max_lr in LEARNING_RATES:
        result = train_one_lr(
            max_lr=max_lr,
            data=data,
            vocab_size=vocab_size,
            device=device,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
        )

        results.append(result)
        save_results(results)

    best = min(results, key=lambda row: row["final_val_loss"])

    print("\n" + "=" * 80)
    print("Best learning rate based on final validation loss")
    print("=" * 80)
    print(f"max_lr: {best['max_lr']:.1e}")
    print(f"final val loss: {best['final_val_loss']:.4f}")
    print(f"checkpoint: {best['checkpoint_path']}")


if __name__ == "__main__":
    main()