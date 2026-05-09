"""
Part 4: Train the best standard Transformer model from scratch.

Use this version when you DO NOT have the old Part 2 checkpoint:
    outputs/checkpoints/scaling_xlarge.pt

This script:
    1. Loads the SentencePiece tokenizer.
    2. Loads encoded train/val/test token files.
    3. Builds the standard xlarge Transformer from scratch.
    4. Trains for 5 epochs.
    5. Saves best and final checkpoints.
    6. Saves training curves.
    7. Computes final test loss and test perplexity.

Outputs:
    outputs/checkpoints/best_standard_xlarge_part4.pt
    outputs/checkpoints/final_standard_xlarge_part4.pt
    outputs/csvs/part4_best_model_training_curves.csv
    outputs/csvs/part4_best_model_metrics.csv
"""

from pathlib import Path
import csv
import math
import time

import sentencepiece as spm
import torch

from dataset_loader import load_all_splits, get_batch
from model import GPTConfig, GPT, count_parameters


# -----------------------------
# Paths
# -----------------------------

TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")

OUTPUT_DIR = Path("outputs")
CSV_DIR = OUTPUT_DIR / "csvs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_standard_xlarge_part4.pt"
FINAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "final_standard_xlarge_part4.pt"

CURVES_PATH = CSV_DIR / "part4_best_model_training_curves.csv"
METRICS_PATH = CSV_DIR / "part4_best_model_metrics.csv"


# -----------------------------
# Training setup
# -----------------------------

BATCH_SIZE = 32
BLOCK_SIZE = 256

# Train from scratch for 5 epochs
EPOCHS = 10

# Best standard LR from your earlier sweep
MAX_LR = 8e-3

WARMUP_FRACTION = 0.03
MIN_LR_FRACTION = 0.10

# Evaluate every 2000 steps and at the end
EVAL_INTERVAL = 1000
EVAL_ITERS = 30
TEST_EVAL_ITERS = 100

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
USE_AMP = True

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
def estimate_loss(model, data, device, use_amp: bool, eval_iters: int) -> dict:
    """
    Estimate train and validation loss using random batches.
    """
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(eval_iters):
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


@torch.no_grad()
def estimate_test_loss(model, data, device, use_amp: bool, eval_iters: int) -> float:
    """
    Estimate test loss using random test batches.
    """
    model.eval()
    test_losses = []

    for _ in range(eval_iters):
        x, y = get_batch(
            tokens=data["test"],
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(x, y)

        test_losses.append(loss.item())

    model.train()
    return sum(test_losses) / len(test_losses)


def save_curves(curve_rows: list[dict]) -> None:
    """
    Save training curve rows to CSV.
    """
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "epoch_fraction",
        "lr",
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

    print(f"Saved training curves to: {CURVES_PATH}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and USE_AMP

    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not TOKENIZER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find tokenizer model: {TOKENIZER_MODEL_PATH}"
        )

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))
    vocab_size = sp.get_piece_size()

    print(f"Tokenizer vocab size: {vocab_size:,}")

    print("Loading encoded token data...")
    data = load_all_splits()

    print(f"Train token IDs: {len(data['train']):,}")
    print(f"Val token IDs: {len(data['val']):,}")
    print(f"Test token IDs: {len(data['test']):,}")

    # Standard xlarge config from your earlier Part 2 setup
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=10,
        n_head=8,
        n_embd=512,
        dropout=0.1,
    )

    print("\nModel config:")
    print(config)

    model = GPT(config).to(device)
    param_count = count_parameters(model)

    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    steps_per_epoch = len(data["train"]) // tokens_per_step
    max_iters = steps_per_epoch * EPOCHS
    warmup_iters = int(WARMUP_FRACTION * max_iters)

    print("\nTraining setup:")
    print(f"Epochs: {EPOCHS}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total steps: {max_iters:,}")
    print(f"Warmup steps: {warmup_iters:,}")
    print(f"Max LR: {MAX_LR:.2e}")
    print(f"AMP enabled: {use_amp}")

    best_val_loss = float("inf")
    best_step = None
    curve_rows = []

    start_time = time.time()

    for step in range(1, max_iters + 1):
        lr = get_lr(
            step=step,
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

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == max_iters:
            losses = estimate_loss(
                model=model,
                data=data,
                device=device,
                use_amp=use_amp,
                eval_iters=EVAL_ITERS,
            )

            elapsed = time.time() - start_time
            tokens_seen = step * tokens_per_step
            tokens_per_sec = tokens_seen / max(1e-8, elapsed)
            epoch_fraction = step / steps_per_epoch

            train_loss = losses["train"]
            val_loss = losses["val"]

            curve_rows.append(
                {
                    "step": step,
                    "epoch_fraction": epoch_fraction,
                    "lr": lr,
                    "batch_loss": loss.item(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "elapsed_sec": elapsed,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_per_sec,
                }
            )

            save_curves(curve_rows)

            print(
                f"step {step:6d}/{max_iters:,} | "
                f"epoch {epoch_fraction:.3f}/{EPOCHS} | "
                f"lr {lr:.2e} | "
                f"batch loss {loss.item():.4f} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"tokens/sec {tokens_per_sec:,.0f} | "
                f"elapsed {elapsed / 60:.1f} min"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step

                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "vocab_size": vocab_size,
                    "model_name": "standard_xlarge_part4_best_from_scratch",
                    "param_count": param_count,
                    "best_step": best_step,
                    "best_val_loss": best_val_loss,
                    "train_settings": {
                        "batch_size": BATCH_SIZE,
                        "block_size": BLOCK_SIZE,
                        "epochs": EPOCHS,
                        "steps_per_epoch": steps_per_epoch,
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
                        "optimizer": "AdamW",
                    },
                }

                torch.save(best_checkpoint, BEST_CHECKPOINT_PATH)
                print(f"Saved new best checkpoint to: {BEST_CHECKPOINT_PATH}")

    elapsed = time.time() - start_time

    print("\nLoading best checkpoint for final test evaluation...")
    best_checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss = estimate_test_loss(
        model=model,
        data=data,
        device=device,
        use_amp=use_amp,
        eval_iters=TEST_EVAL_ITERS,
    )

    test_perplexity = math.exp(test_loss)

    gpu_memory_mb = None
    if device == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "vocab_size": vocab_size,
        "model_name": "standard_xlarge_part4_final_from_scratch",
        "param_count": param_count,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "train_settings": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "epochs": EPOCHS,
            "steps_per_epoch": steps_per_epoch,
            "max_iters": max_iters,
            "warmup_iters": warmup_iters,
            "max_lr": MAX_LR,
            "min_lr_fraction": MIN_LR_FRACTION,
            "warmup_fraction": WARMUP_FRACTION,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "test_eval_iters": TEST_EVAL_ITERS,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "use_amp": use_amp,
            "optimizer": "AdamW",
        },
    }

    torch.save(final_checkpoint, FINAL_CHECKPOINT_PATH)

    metrics = {
        "model_name": "standard_xlarge_part4_from_scratch",
        "best_checkpoint": str(BEST_CHECKPOINT_PATH),
        "final_checkpoint": str(FINAL_CHECKPOINT_PATH),
        "param_count": param_count,
        "epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "max_iters": max_iters,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "elapsed_sec": elapsed,
        "gpu_memory_mb": gpu_memory_mb,
    }

    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("\nPart 4 training complete.")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_perplexity:.4f}")
    print(f"Saved best checkpoint to: {BEST_CHECKPOINT_PATH}")
    print(f"Saved final checkpoint to: {FINAL_CHECKPOINT_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved curves to: {CURVES_PATH}")


if __name__ == "__main__":
    main()