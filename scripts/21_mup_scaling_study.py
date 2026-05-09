"""
μP scaling study for SVG decoder-only Transformer models.

This script trains multiple μP GPT-style model sizes on the same SVG token
dataset. Each model is trained for approximately 1 token-equivalent epoch using
the best μP peak learning rate selected from the corrected μP LR sweep.

For each model, the script saves:
    1. final model checkpoint (.pt)
    2. final metrics row in outputs/mup_scaling_results_fixed.csv
    3. intermediate training/validation losses in outputs/mup_training_curves_fixed.csv

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
    outputs/mup_scaling_results_fixed.csv
    outputs/mup_training_curves_fixed.csv
    outputs/checkpoints/mup_scaling_fixed_<model_name>.pt
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
RESULTS_PATH = OUTPUT_DIR / "mup_scaling_results_fixed.csv"
CURVES_PATH = OUTPUT_DIR / "mup_training_curves_fixed.csv"


# Same data/batch setup as Part 2
BATCH_SIZE = 32
BLOCK_SIZE = 256

# Best μP peak LR from corrected μP LR sweep
MAX_LR = 2e-3

# LR schedule
WARMUP_FRACTION = 0.05
MIN_LR_FRACTION = 0.10

# Evaluation settings
EVAL_INTERVAL = 2000
EVAL_ITERS = 20

# Training stability
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
USE_AMP = True

SEED = 42


MODEL_CONFIGS = [
    {
        "model_name": "tiny",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.1,
    },
    {
        "model_name": "small",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "dropout": 0.1,
    },
    {
        "model_name": "medium",
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.1,
    },
    {
        "model_name": "large",
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.1,
    },
    {
        "model_name": "xlarge",
        "n_layer": 10,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.1,
    },
]


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
    Save final per-model μP metrics collected so far.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "status",
        "error_message",
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "param_count",
        "batch_size",
        "block_size",
        "tokens_per_step",
        "max_iters",
        "warmup_iters",
        "max_lr",
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

    print(f"\nSaved current fixed μP scaling results to: {RESULTS_PATH}")


def save_curves(curve_rows: list[dict]) -> None:
    """
    Save μP training/validation loss curves collected so far.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "step",
        "lr_multiplier",
        "displayed_lr",
        "batch_loss",
        "train_loss",
        "val_loss",
        "elapsed_sec",
        "tokens_seen",
        "tokens_per_sec",
    ]

    with open(CURVES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in curve_rows:
            writer.writerow(row)

    print(f"Saved current fixed μP training curves to: {CURVES_PATH}")


def train_one_model(
    model_cfg: dict,
    data,
    vocab_size: int,
    device: str,
    max_iters: int,
    warmup_iters: int,
    curve_rows: list[dict],
) -> dict:
    """
    Train one μP model configuration for approximately 1 epoch.
    """
    model_name = model_cfg["model_name"]

    torch.manual_seed(SEED)

    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    use_amp = device == "cuda" and USE_AMP

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
    )

    model = GPT(config)

    # μP requirement:
    # Apply base shapes before creating the MuAdam optimizer.
    model = apply_mup_base_shapes(model, config)
    model = model.to(device)

    param_count = count_parameters(model)
    has_infshape = any(hasattr(p, "infshape") for p in model.parameters())

    optimizer = MuAdam(
        model.parameters(),
        lr=MAX_LR,
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
    print(f"Training fixed μP model: {model_name}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding dim: {config.n_embd}")
    print(f"Parameters: {param_count:,}")
    print(f"Has μP infshape metadata: {has_infshape}")
    print(f"Number of optimizer param groups: {len(optimizer.param_groups)}")
    print(f"First few MuAdam base group LRs: {base_group_lrs[:5]}")
    print(f"Max iters / one epoch: {max_iters:,}")
    print(f"Warmup iters: {warmup_iters:,}")
    print(f"Max LR: {MAX_LR:.1e}")
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
        displayed_lr = MAX_LR * lr_multiplier

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

            curve_rows.append(
                {
                    "model_name": model_name,
                    "step": step,
                    "lr_multiplier": lr_multiplier,
                    "displayed_lr": displayed_lr,
                    "batch_loss": loss.item(),
                    "train_loss": last_train_loss,
                    "val_loss": last_val_loss,
                    "elapsed_sec": elapsed,
                    "tokens_seen": tokens_seen_so_far,
                    "tokens_per_sec": tokens_per_sec_so_far,
                }
            )

            save_curves(curve_rows)

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

    checkpoint_path = CHECKPOINT_DIR / f"mup_scaling_fixed_{model_name}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "vocab_size": vocab_size,
        "model_name": model_name,
        "param_count": param_count,
        "train_settings": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "max_iters": max_iters,
            "warmup_iters": warmup_iters,
            "max_lr": MAX_LR,
            "min_lr_fraction": MIN_LR_FRACTION,
            "warmup_fraction": WARMUP_FRACTION,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "use_amp": use_amp,
            "optimizer": "MuAdam",
            "scheduler": "relative warmup + cosine multiplier",
        },
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    result = {
        "model_name": model_name,
        "status": "completed",
        "error_message": "",
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "param_count": param_count,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "tokens_per_step": BATCH_SIZE * BLOCK_SIZE,
        "max_iters": max_iters,
        "warmup_iters": warmup_iters,
        "max_lr": MAX_LR,
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
    del scaler

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def failed_result(model_cfg: dict, error: Exception) -> dict:
    """
    Create a CSV row for a failed μP model.
    """
    return {
        "model_name": model_cfg["model_name"],
        "status": "failed",
        "error_message": repr(error),
        "n_layer": model_cfg["n_layer"],
        "n_head": model_cfg["n_head"],
        "n_embd": model_cfg["n_embd"],
        "dropout": model_cfg["dropout"],
        "param_count": None,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "tokens_per_step": BATCH_SIZE * BLOCK_SIZE,
        "max_iters": None,
        "warmup_iters": None,
        "max_lr": MAX_LR,
        "final_train_loss": None,
        "final_val_loss": None,
        "elapsed_sec": None,
        "tokens_seen": None,
        "tokens_per_sec": None,
        "gpu_memory_mb": None,
        "checkpoint_path": "",
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
    curve_rows = []

    for model_cfg in MODEL_CONFIGS:
        try:
            result = train_one_model(
                model_cfg=model_cfg,
                data=data,
                vocab_size=vocab_size,
                device=device,
                max_iters=max_iters,
                warmup_iters=warmup_iters,
                curve_rows=curve_rows,
            )

            results.append(result)
            save_results(results)

        except Exception as e:
            print("\n" + "!" * 80)
            print(f"Fixed μP model failed: {model_cfg['model_name']}")
            print(repr(e))
            traceback.print_exc()
            print("!" * 80)

            results.append(failed_result(model_cfg, e))
            save_results(results)
            save_curves(curve_rows)

            if device == "cuda":
                torch.cuda.empty_cache()

            continue

    print("\nFixed μP scaling study complete.")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Training curves saved to: {CURVES_PATH}")


if __name__ == "__main__":
    main()