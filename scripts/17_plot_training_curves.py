"""
Plot training and validation loss curves for the scaling study.

This script reads the intermediate loss values saved by 15_scaling_study.py and
creates loss-over-time plots for the Part 2 report.

Input:
    outputs/csvs/training_curves.csv

Outputs:
    outputs/plots/training_loss_curves.png
    outputs/plots/validation_loss_curves.png
    outputs/plots/train_val_loss_curves.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CURVES_PATH = Path("outputs/csvs/training_curves.csv")

OUT_DIR = Path("outputs/plots")
TRAIN_OUT_PATH = OUT_DIR / "training_loss_curves.png"
VAL_OUT_PATH = OUT_DIR / "validation_loss_curves.png"
BOTH_OUT_PATH = OUT_DIR / "train_val_loss_curves.png"


MODEL_ORDER = ["tiny", "small", "medium", "large", "xlarge"]


def load_curves() -> pd.DataFrame:
    """Load and sort training-curve data."""
    df = pd.read_csv(CURVES_PATH)

    df["model_name"] = pd.Categorical(
        df["model_name"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = df.sort_values(["model_name", "step"]).reset_index(drop=True)

    return df


def plot_single_metric(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot one loss metric over training steps for each model."""
    plt.figure(figsize=(9, 5.5))

    for model_name in MODEL_ORDER:
        model_df = df[df["model_name"] == model_name]

        if len(model_df) == 0:
            continue

        plt.plot(
            model_df["step"],
            model_df[metric_col],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {out_path}")


def plot_train_and_val(df: pd.DataFrame) -> None:
    """
    Plot training and validation loss together.

    Solid lines are training loss.
    Dashed lines are validation loss.
    """
    plt.figure(figsize=(10, 6))

    for model_name in MODEL_ORDER:
        model_df = df[df["model_name"] == model_name]

        if len(model_df) == 0:
            continue

        plt.plot(
            model_df["step"],
            model_df["train_loss"],
            linewidth=2,
            label=f"{model_name} train",
        )

        plt.plot(
            model_df["step"],
            model_df["val_loss"],
            linestyle="--",
            linewidth=2,
            label=f"{model_name} val",
        )

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.savefig(BOTH_OUT_PATH, dpi=200)
    plt.close()

    print(f"Saved plot to: {BOTH_OUT_PATH}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_curves()

    print("\nTraining curve rows:")
    print(df.head())

    print("\nRows per model:")
    print(df.groupby("model_name", observed=True).size())

    plot_single_metric(
        df=df,
        metric_col="train_loss",
        ylabel="Training loss",
        title="Training Loss Curves",
        out_path=TRAIN_OUT_PATH,
    )

    plot_single_metric(
        df=df,
        metric_col="val_loss",
        ylabel="Validation loss",
        title="Validation Loss Curves",
        out_path=VAL_OUT_PATH,
    )

    plot_train_and_val(df)


if __name__ == "__main__":
    main()