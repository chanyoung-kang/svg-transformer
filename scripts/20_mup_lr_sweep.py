"""
μP learning-rate sweep for the smallest SVG Transformer model.

This script trains the same smallest μP decoder-only Transformer multiple times,
each time with a different peak learning rate.

Each run uses:
    linear warmup -> cosine decay

Important μP detail:
    We do NOT overwrite each optimizer parameter group's learning rate with
    the same absolute value. MuAdam creates μP-adjusted parameter groups with
    different base learning rates. The scheduler should multiply those existing
    group learning rates by a relative schedule factor.

Inputs:
    data/processed/encoded/train.bin
    data/processed/encoded/val.bin
    data/tokenizer/svg_bpe.model
    scripts/dataset_loader.py
    scripts/mup_model.py

Outputs:
    outputs/mup_lr_sweep_results_fixed.csv
    outputs/checkpoints/mup_lr_sweep_fixed_<learning_rate>.pt
"""

from pathlib import Path
import csv
import math
import time
import traceback

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

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_PATH = OUTPUT_DIR / "mup_lr_sweep_results_fixed.csv"


# μP learning-rate candidates.
# Since previous broken sweep suggested best around 1.5e-3 to 2e-3,
# this range tests around that area while still covering lower/higher values.
LEARNING_RATES = [
    5e-4,
    8e-4,
    1e-3,
    1.5e-3,
    2e-3,
    3e-3,
    4e-3,
]


# Same batch/token setup as Part 2.
BATCH_SIZE = 32
BLOCK_SIZE = 256

# One epoch is calculated from the train token count.
WARMUP_FRACTION = 0.05
MIN_LR_FRACTION = 0.10

EVAL_INTERVAL = 2000
EVAL_ITERS = 20

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
USE_AMP = True

# Smallest μP model for LR sweep.
# This matches the tiny model size from Part 2.
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1

SEED = 42


def get_lr_multiplier(step: int, max_iters: int, warmup_iters: int) -> float:
    """
    Warmup + cosine decay as a relative multiplier.

    This returns a factor, not an absolute learning rate.

    During warmup:
        multiplier increases from near 0 to 1.

    After warmup:
        multiplier decreases from 1 to MIN_LR_FRACTION.

    For MuAdam, this is important because MuAdam creates μP-adjusted
    parameter group learning rates. We preserve those by multiplying each
    group's original learning rate by this multiplier.
    """
    if step < warmup_iters:
        return (step + 1) / max(1, warmup_iters)

    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    progress = min(1.0, max(0.0, progress))

    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))

    return MIN_LR_FRACTION + cosine_coeff * (1.0 - MIN_LR_FRACTION)


@torch.no_grad()
def estimate_loss(model, data, device, use_amp: bool) -> dict:
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

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(x, y)

            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def save_results(results: list[dict]) -> None:
    """
    Save μP LR sweep results collected so far.
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
        "status",
        "error_message",
    ]

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved μP LR sweep results to: {RESULTS_PATH}")


def safe_lr_name(max_lr: float) -> str:
    """
    Create a safe filename component for a learning rate.

    Example:
        8e-3 -> 8p0em03
    """
    return f"{max_lr:.1e}".replace(".", "p").replace("-", "m")


def train_one_lr(
    max_lr: float,
    data,
    vocab_size: int,
    device: str,
    max_iters: int,
    warmup_iters: int,
) -> dict:
    """
    Train the smallest μP model with one candidate peak learning rate.
    """
    torch.manual_seed(SEED)

    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    use_amp = device == "cuda" and USE_AMP

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
    # Apply base shapes before optimizer creation.
    model = apply_mup_base_shapes(model, config)
    model = model.to(device)

    param_count = count_parameters(model)
    has_infshape = any(hasattr(p, "infshape") for p in model.parameters())

    optimizer = MuAdam(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    # IMPORTANT:
    # Store the MuAdam-created group LRs AFTER optimizer creation.
    # These already include μP-specific scaling. The scheduler should multiply
    # these values, not overwrite them with one absolute LR.
    base_group_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("\n" + "=" * 80)
    print(f"μP LR sweep run: max_lr={max_lr:.1e}")
    print(f"Model parameters: {param_count:,}")
    print(f"Has μP infshape metadata: {has_infshape}")
    print(f"Number of optimizer param groups: {len(optimizer.param_groups)}")
    print(f"First few MuAdam base group LRs: {base_group_lrs[:5]}")
    print(f"Max iters / one epoch: {max_iters:,}")
    print(f"Warmup iters: {warmup_iters:,}")
    print(f"AMP enabled: {use_amp}")
    print("=" * 80)

    start_time = time.time()
    last_train_loss = None
    last_val_loss = None

    for step in range(1, max_iters + 1):
        lr_multiplier = get_lr_multiplier(
            step=step,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
        )

        # μP-correct scheduling:
        # Each group keeps its MuAdam base LR, scaled by the same schedule factor.
        for param_group, base_lr in zip(optimizer.param_groups, base_group_lrs):
            param_group["lr"] = base_lr * lr_multiplier

        # This is only for logging the global schedule level.
        displayed_lr = max_lr * lr_multiplier

        x, y = get_batch(
            tokens=data["train"],
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == max_iters:
            losses = estimate_loss(model, data, device, use_amp)

            elapsed = time.time() - start_time
            tokens_seen_so_far = step * BATCH_SIZE * BLOCK_SIZE
            tokens_per_sec_so_far = tokens_seen_so_far / max(1e-8, elapsed)

            last_train_loss = losses["train"]
            last_val_loss = losses["val"]

            print(
                f"step {step:6d}/{max_iters:,} | "
                f"lr_mult {lr_multiplier:.3f} | "
                f"display_lr {displayed_lr:.2e} | "
                f"batch loss {loss.item():.4f} | "
                f"train loss {last_train_loss:.4f} | "
                f"val loss {last_val_loss:.4f} | "
                f"tokens/sec {tokens_per_sec_so_far:,.0f} | "
                f"elapsed {elapsed / 60:.1f} min"
            )

    elapsed = time.time() - start_time
    tokens_seen = max_iters * BATCH_SIZE * BLOCK_SIZE
    tokens_per_sec = tokens_seen / max(1e-8, elapsed)

    gpu_memory_mb = None
    if device == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    lr_name = safe_lr_name(max_lr)
    checkpoint_path = CHECKPOINT_DIR / f"mup_lr_sweep_fixed_{lr_name}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "vocab_size": vocab_size,
        "max_lr": max_lr,
        "param_count": param_count,
        "train_settings": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "max_iters": max_iters,
            "warmup_iters": warmup_iters,
            "warmup_fraction": WARMUP_FRACTION,
            "min_lr_fraction": MIN_LR_FRACTION,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "use_amp": use_amp,
            "optimizer": "MuAdam",
            "scheduler": "relative warmup + cosine multiplier",
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "dropout": DROPOUT,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

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
        "status": "completed",
        "error_message": "",
    }

    del model
    del optimizer
    del scaler

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def failed_result(max_lr: float, error: Exception) -> dict:
    """
    Create a result row if an LR run fails.
    """
    return {
        "max_lr": max_lr,
        "param_count": None,
        "max_iters": None,
        "warmup_iters": None,
        "final_train_loss": None,
        "final_val_loss": None,
        "elapsed_sec": None,
        "tokens_seen": None,
        "tokens_per_sec": None,
        "gpu_memory_mb": None,
        "checkpoint_path": "",
        "status": "failed",
        "error_message": repr(error),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
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

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    max_iters = len(data["train"]) // tokens_per_step
    warmup_iters = int(WARMUP_FRACTION * max_iters)

    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Calculated one-epoch steps: {max_iters:,}")
    print(f"Warmup steps: {warmup_iters:,}")

    results = []

    for max_lr in LEARNING_RATES:
        try:
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

        except Exception as e:
            print("\n" + "!" * 80)
            print(f"μP LR run failed: max_lr={max_lr:.1e}")
            print(repr(e))
            traceback.print_exc()
            print("!" * 80)

            results.append(failed_result(max_lr, e))
            save_results(results)

            if device == "cuda":
                torch.cuda.empty_cache()

            continue

    completed = [row for row in results if row["status"] == "completed"]

    if completed:
        best = min(completed, key=lambda row: row["final_val_loss"])

        print("\n" + "=" * 80)
        print("Best μP learning rate based on final validation loss")
        print("=" * 80)
        print(f"max_lr: {best['max_lr']:.1e}")
        print(f"final val loss: {best['final_val_loss']:.4f}")
        print(f"checkpoint: {best['checkpoint_path']}")
    else:
        print("No μP LR sweep runs completed successfully.")


if __name__ == "__main__":
    main()