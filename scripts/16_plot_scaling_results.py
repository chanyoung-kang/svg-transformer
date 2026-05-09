"""
Plot Transformer scaling-study results and fit a power law.

This script reads the scaling study CSV created by 15_scaling_study.py and
creates a validation-loss-vs-parameter-count plot for the Part 2 report.

It also fits the required power law:

    L = a * N^(-alpha) + c

where:
    L = validation loss
    N = number of model parameters
    alpha = scaling exponent

Input:
    outputs/scaling_results.csv

Outputs:
    outputs/plots/scaling_validation_loss_powerlaw.png
    outputs/scaling_powerlaw_fit.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = Path("outputs/csvs/scaling_results.csv")

OUT_DIR = Path("outputs/plots")
OUT_PATH = OUT_DIR / "scaling_validation_loss_powerlaw.png"

FIT_PATH = Path("outputs/scaling_powerlaw_fit.csv")


# Manual label offsets so text does not overlap borders or points.
LABEL_OFFSETS = {
    "tiny": (8, -12),
    "small": (8, 6),
    "medium": (8, 6),
    "large": (8, 6),
    "xlarge": (8, 8),
}


def power_law(n_params: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """
    Power-law function:

        L = a * N^(-alpha) + c

    n_params:
        Number of model parameters.
    """
    return a * (n_params ** (-alpha)) + c


def fit_power_law(n_params: np.ndarray, val_loss: np.ndarray) -> dict:
    """
    Fit the power law L = a * N^(-alpha) + c.

    We try scipy first. If scipy is unavailable or fitting fails, we fall back
    to a simpler log-log fit without the offset c.
    """
    try:
        from scipy.optimize import curve_fit

        # Initial guesses:
        # c should be slightly below the best observed validation loss.
        c0 = val_loss.min() - 0.05
        alpha0 = 0.1
        a0 = (val_loss.max() - c0) * (n_params.min() ** alpha0)

        # Bounds keep the fit reasonable:
        # a > 0, alpha > 0, c >= 0
        params, _ = curve_fit(
            power_law,
            n_params,
            val_loss,
            p0=[a0, alpha0, c0],
            bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, val_loss.min()]),
            maxfev=10000,
        )

        a, alpha, c = params

        return {
            "fit_method": "nonlinear_curve_fit_with_offset",
            "a": float(a),
            "alpha": float(alpha),
            "c": float(c),
        }

    except Exception as e:
        print(f"scipy curve_fit failed or unavailable: {repr(e)}")
        print("Falling back to log-log fit without offset c.")

        # Fallback: assume c = 0 and fit log(L) = log(a) - alpha * log(N)
        log_n = np.log(n_params)
        log_l = np.log(val_loss)

        slope, intercept = np.polyfit(log_n, log_l, deg=1)

        alpha = -slope
        a = np.exp(intercept)
        c = 0.0

        return {
            "fit_method": "log_log_fit_no_offset",
            "a": float(a),
            "alpha": float(alpha),
            "c": float(c),
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    # Keep only successfully completed model runs.
    df = df[df["status"] == "completed"].copy()

    # Sort by parameter count so the line connects models from small to large.
    df = df.sort_values("param_count").reset_index(drop=True)

    print("\nScaling results:")
    print(df[["model_name", "param_count", "final_train_loss", "final_val_loss"]])

    n_params = df["param_count"].to_numpy(dtype=float)
    val_loss = df["final_val_loss"].to_numpy(dtype=float)

    fit = fit_power_law(n_params, val_loss)

    a = fit["a"]
    alpha = fit["alpha"]
    c = fit["c"]

    print("\nPower-law fit:")
    print(f"Fit method: {fit['fit_method']}")
    print(f"a: {a:.6g}")
    print(f"alpha: {alpha:.6g}")
    print(f"c: {c:.6g}")

    # Save fit parameters for the report.
    fit_df = pd.DataFrame(
        [
            {
                "fit_method": fit["fit_method"],
                "a": a,
                "alpha": alpha,
                "c": c,
                "equation": "L = a * N^(-alpha) + c",
            }
        ]
    )
    fit_df.to_csv(FIT_PATH, index=False)
    print(f"\nSaved power-law fit parameters to: {FIT_PATH}")

    # Smooth x values for fitted curve.
    n_smooth = np.logspace(
        np.log10(n_params.min()),
        np.log10(n_params.max()),
        200,
    )
    fitted_loss = power_law(n_smooth, a, alpha, c)

    plt.figure(figsize=(9, 5.5))

    # Actual validation losses.
    plt.plot(
        df["param_count"],
        df["final_val_loss"],
        marker="o",
        linewidth=2,
        label="Validation loss",
    )

    # Fitted power-law curve.
    plt.plot(
        n_smooth,
        fitted_loss,
        linestyle="--",
        linewidth=2,
        label=fr"Power-law fit ($\alpha={alpha:.3f}$)",
    )

    # Labels for each point.
    for _, row in df.iterrows():
        model_name = row["model_name"]
        x_offset, y_offset = LABEL_OFFSETS.get(model_name, (8, 6))

        plt.annotate(
            model_name,
            (row["param_count"], row["final_val_loss"]),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            ha="left" if x_offset >= 0 else "right",
            fontsize=9,
        )

    plt.xscale("log")
    plt.xlabel("Number of parameters (log scale)")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("SVG Transformer Scaling Study")

    y_min = min(df["final_val_loss"].min(), fitted_loss.min())
    y_max = max(df["final_val_loss"].max(), fitted_loss.max())
    y_padding = (y_max - y_min) * 0.12
    plt.ylim(y_min - y_padding, y_max + y_padding)

    x_min = df["param_count"].min()
    x_max = df["param_count"].max()
    plt.xlim(x_min * 0.8, x_max * 1.25)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_PATH, dpi=200)
    plt.close()

    print(f"Saved scaling plot with power-law fit to: {OUT_PATH}")


if __name__ == "__main__":
    main()