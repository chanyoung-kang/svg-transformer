"""
Scaling-law extrapolation for Part 3.

This script uses the better scaling law from Part 2 vs Part 3 to predict
validation loss for a model with 10x more parameters than the largest trained
model.

In our results, the standard parameterization performed much better than μP, so
the extrapolation uses the standard scaling law.

Inputs:
    outputs/csvs/scaling_results.csv
    outputs/csvs/standard_vs_mup_powerlaw_fit.csv

Outputs:
    outputs/extrapolation_prediction.csv
    outputs/plots/scaling_extrapolation_10x.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_DIR = Path("outputs/csvs")
OUT_DIR = Path("outputs/plots")

STANDARD_SCALING_PATH = CSV_DIR / "scaling_results.csv"
POWERLAW_FIT_PATH = CSV_DIR / "standard_vs_mup_powerlaw_fit.csv"

EXTRAPOLATION_OUT_PATH = Path("outputs/extrapolation_prediction.csv")
EXTRAPOLATION_PLOT_PATH = OUT_DIR / "scaling_extrapolation_10x.png"


def power_law(n_params: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """
    Power-law function:

        L = a * N^(-alpha) + c
    """
    return a * (n_params ** (-alpha)) + c


def load_standard_results() -> pd.DataFrame:
    """
    Load standard Part 2 scaling results.
    """
    df = pd.read_csv(STANDARD_SCALING_PATH)

    if "status" in df.columns:
        df = df[df["status"] == "completed"].copy()

    df = df.sort_values("param_count").reset_index(drop=True)

    return df


def load_standard_powerlaw_fit() -> dict:
    """
    Load standard power-law fit parameters from script 22.
    """
    fit_df = pd.read_csv(POWERLAW_FIT_PATH)

    standard_row = fit_df[fit_df["parameterization"] == "standard"].iloc[0]

    return {
        "a": float(standard_row["a"]),
        "alpha": float(standard_row["alpha"]),
        "c": float(standard_row["c"]),
        "fit_method": standard_row["fit_method"],
    }


def bootstrap_prediction_interval(
    df: pd.DataFrame,
    target_params: float,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict:
    """
    Estimate uncertainty using bootstrap resampling.

    Because we only have 5 model sizes, this uncertainty estimate is rough.
    It repeatedly samples the observed model-size/loss pairs with replacement,
    fits a simple log-log model with c = 0, and predicts the loss at the target
    parameter count.

    This is not a perfect confidence interval, but it gives a reasonable
    uncertainty range for the report.
    """
    rng = np.random.default_rng(random_seed)

    n_params = df["param_count"].to_numpy(dtype=float)
    val_loss = df["final_val_loss"].to_numpy(dtype=float)

    predictions = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(df), size=len(df))

        sampled_n = n_params[idx]
        sampled_l = val_loss[idx]

        # Avoid degenerate samples with too few unique x values.
        if len(np.unique(sampled_n)) < 2:
            continue

        try:
            log_n = np.log(sampled_n)
            log_l = np.log(sampled_l)

            slope, intercept = np.polyfit(log_n, log_l, deg=1)

            pred_log_l = intercept + slope * np.log(target_params)
            pred_l = float(np.exp(pred_log_l))

            if np.isfinite(pred_l):
                predictions.append(pred_l)

        except Exception:
            continue

    predictions = np.array(predictions)

    return {
        "bootstrap_n": int(len(predictions)),
        "bootstrap_mean": float(np.mean(predictions)),
        "bootstrap_lower_95": float(np.percentile(predictions, 2.5)),
        "bootstrap_upper_95": float(np.percentile(predictions, 97.5)),
    }


def make_extrapolation_plot(
    df: pd.DataFrame,
    fit: dict,
    target_params: float,
    predicted_loss: float,
) -> None:
    """
    Plot observed scaling results, fitted standard power law, and 10x
    extrapolated prediction.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_min = df["param_count"].min()
    n_max = target_params

    n_smooth = np.logspace(
        np.log10(n_min),
        np.log10(n_max),
        300,
    )

    fitted_loss = power_law(
        n_smooth,
        fit["a"],
        fit["alpha"],
        fit["c"],
    )

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        df["param_count"],
        df["final_val_loss"],
        marker="o",
        linewidth=2,
        label="Observed standard models",
    )

    plt.plot(
        n_smooth,
        fitted_loss,
        linestyle="--",
        linewidth=2,
        label=fr"Power-law fit ($\alpha={fit['alpha']:.3f}$)",
    )

    plt.scatter(
        [target_params],
        [predicted_loss],
        marker="*",
        s=180,
        label="10x extrapolated prediction",
    )

    plt.annotate(
        f"10x prediction\nloss={predicted_loss:.4f}",
        (target_params, predicted_loss),
        textcoords="offset points",
        xytext=(-90, 20),
        fontsize=9,
    )

    for _, row in df.iterrows():
        plt.annotate(
            str(row["model_name"]),
            (row["param_count"], row["final_val_loss"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    plt.xscale("log")
    plt.xlabel("Number of parameters (log scale)")
    plt.ylabel("Validation loss")
    plt.title("Scaling-Law Extrapolation to 10x Larger Model")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(EXTRAPOLATION_PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved extrapolation plot to: {EXTRAPOLATION_PLOT_PATH}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_standard_results()
    fit = load_standard_powerlaw_fit()

    largest_row = df.loc[df["param_count"].idxmax()]
    largest_params = float(largest_row["param_count"])
    target_params = largest_params * 10.0

    predicted_loss = float(
        power_law(
            np.array([target_params]),
            fit["a"],
            fit["alpha"],
            fit["c"],
        )[0]
    )

    uncertainty = bootstrap_prediction_interval(
        df=df,
        target_params=target_params,
    )

    result = {
        "chosen_parameterization": "standard",
        "reason_chosen": "Standard parameterization had much lower validation loss than μP in the observed runs.",
        "largest_model_name": largest_row["model_name"],
        "largest_param_count": largest_params,
        "target_param_count_10x": target_params,
        "powerlaw_a": fit["a"],
        "powerlaw_alpha": fit["alpha"],
        "powerlaw_c": fit["c"],
        "fit_method": fit["fit_method"],
        "predicted_val_loss_10x": predicted_loss,
        "bootstrap_n": uncertainty["bootstrap_n"],
        "bootstrap_mean": uncertainty["bootstrap_mean"],
        "bootstrap_lower_95": uncertainty["bootstrap_lower_95"],
        "bootstrap_upper_95": uncertainty["bootstrap_upper_95"],
        "uncertainty_note": (
            "Bootstrap interval is approximate because only five model sizes were available."
        ),
    }

    out_df = pd.DataFrame([result])
    out_df.to_csv(EXTRAPOLATION_OUT_PATH, index=False)

    print("\nScaling-law extrapolation result:")
    print(out_df.T)

    print(f"\nSaved extrapolation CSV to: {EXTRAPOLATION_OUT_PATH}")

    make_extrapolation_plot(
        df=df,
        fit=fit,
        target_params=target_params,
        predicted_loss=predicted_loss,
    )


if __name__ == "__main__":
    main()