"""
Compare standard parameterization vs μP scaling results.

This script reads:
    - standard scaling results from Part 2
    - μP scaling results from Part 3
    - standard LR sweep results
    - μP LR sweep results

It creates:
    1. standard vs μP scaling plot
    2. standard vs μP power-law fit CSV
    3. standard vs μP LR sweep comparison plot

Inputs:
    outputs/csvs/scaling_results.csv
    outputs/csvs/mup_scaling_results_fixed.csv
    outputs/csvs/lr_sweep_results.csv
    outputs/csvs/lr_sweep_results2.csv
    outputs/csvs/lr_sweep_results3.csv
    outputs/csvs/mup_lr_sweep_results_fixed.csv

Outputs:
    outputs/plots/standard_vs_mup_scaling.png
    outputs/plots/lr_sweep_standard_vs_mup.png
    outputs/standard_vs_mup_powerlaw_fit.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_DIR = Path("outputs/csvs")
OUT_DIR = Path("outputs/plots")

STANDARD_SCALING_PATH = CSV_DIR / "scaling_results.csv"
MUP_SCALING_PATH = CSV_DIR / "mup_scaling_results_fixed.csv"

STANDARD_LR_SWEEP_PATHS = [
    CSV_DIR / "lr_sweep_results.csv",
    CSV_DIR / "lr_sweep_results2.csv",
    CSV_DIR / "lr_sweep_results3.csv",
]

MUP_LR_SWEEP_PATH = CSV_DIR / "mup_lr_sweep_results_fixed.csv"

SCALING_PLOT_PATH = OUT_DIR / "standard_vs_mup_scaling.png"
LR_SWEEP_PLOT_PATH = OUT_DIR / "lr_sweep_standard_vs_mup.png"
POWERLAW_FIT_PATH = Path("outputs/standard_vs_mup_powerlaw_fit.csv")


MODEL_ORDER = ["tiny", "small", "medium", "large", "xlarge"]


def power_law(n_params: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """
    Power-law function:

        L = a * N^(-alpha) + c

    where:
        L = validation loss
        N = number of parameters
        alpha = scaling exponent
        c = irreducible loss floor / offset
    """
    return a * (n_params ** (-alpha)) + c


def fit_power_law(n_params: np.ndarray, val_loss: np.ndarray) -> dict:
    """
    Fit L = a * N^(-alpha) + c.

    Uses scipy curve_fit if available. Falls back to a log-log fit with c = 0.
    """
    try:
        from scipy.optimize import curve_fit

        c0 = max(0.0, val_loss.min() - 0.05)
        alpha0 = 0.1
        a0 = (val_loss.max() - c0) * (n_params.min() ** alpha0)

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
        print(f"curve_fit failed or unavailable: {repr(e)}")
        print("Falling back to log-log fit with c = 0.")

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


def load_scaling_results(path: Path, parameterization: str) -> pd.DataFrame:
    """
    Load scaling results and keep completed runs only.
    """
    df = pd.read_csv(path)

    if "status" in df.columns:
        df = df[df["status"] == "completed"].copy()

    df["parameterization"] = parameterization

    df["model_name"] = pd.Categorical(
        df["model_name"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = df.sort_values(["model_name"]).reset_index(drop=True)

    return df


def plot_standard_vs_mup_scaling(
    standard_df: pd.DataFrame,
    mup_df: pd.DataFrame,
    standard_fit: dict,
    mup_fit: dict,
) -> None:
    """
    Plot standard vs μP validation loss scaling curves and fitted power laws.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    # Observed points
    plt.plot(
        standard_df["param_count"],
        standard_df["final_val_loss"],
        marker="o",
        linewidth=2,
        label="Standard observed",
    )

    plt.plot(
        mup_df["param_count"],
        mup_df["final_val_loss"],
        marker="o",
        linewidth=2,
        label="μP observed",
    )

    # Fitted curves
    all_params = pd.concat(
        [standard_df["param_count"], mup_df["param_count"]],
        ignore_index=True,
    )

    n_smooth = np.logspace(
        np.log10(all_params.min()),
        np.log10(all_params.max()),
        200,
    )

    standard_curve = power_law(
        n_smooth,
        standard_fit["a"],
        standard_fit["alpha"],
        standard_fit["c"],
    )

    mup_curve = power_law(
        n_smooth,
        mup_fit["a"],
        mup_fit["alpha"],
        mup_fit["c"],
    )

    plt.plot(
        n_smooth,
        standard_curve,
        linestyle="--",
        linewidth=2,
        label=fr"Standard fit ($\alpha={standard_fit['alpha']:.3f}$)",
    )

    plt.plot(
        n_smooth,
        mup_curve,
        linestyle="--",
        linewidth=2,
        label=fr"μP fit ($\alpha={mup_fit['alpha']:.3f}$)",
    )

    # Label observed points
    for _, row in standard_df.iterrows():
        plt.annotate(
            str(row["model_name"]),
            (row["param_count"], row["final_val_loss"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    for _, row in mup_df.iterrows():
        plt.annotate(
            str(row["model_name"]),
            (row["param_count"], row["final_val_loss"]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=8,
        )

    plt.xscale("log")
    plt.xlabel("Number of parameters (log scale)")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Standard vs μP Transformer Scaling")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(SCALING_PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved scaling comparison plot to: {SCALING_PLOT_PATH}")


def load_standard_lr_sweeps(paths: list[Path]) -> pd.DataFrame:
    """
    Load and combine all standard LR sweep CSV files.
    """
    frames = []

    for path in paths:
        if not path.exists():
            print(f"Skipping missing standard LR sweep file: {path}")
            continue

        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No standard LR sweep files found.")

    df = pd.concat(frames, ignore_index=True)

    if "status" in df.columns:
        df = df[(df["status"].isna()) | (df["status"] == "completed")].copy()

    df["parameterization"] = "standard"

    return df


def load_mup_lr_sweep(path: Path) -> pd.DataFrame:
    """
    Load μP LR sweep CSV.
    """
    df = pd.read_csv(path)

    if "status" in df.columns:
        df = df[df["status"] == "completed"].copy()

    df["parameterization"] = "μP"

    return df


def plot_lr_sweeps(standard_lr_df: pd.DataFrame, mup_lr_df: pd.DataFrame) -> None:
    """
    Plot LR sweep validation loss for standard vs μP.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    standard_lr_df = standard_lr_df.sort_values("max_lr")
    mup_lr_df = mup_lr_df.sort_values("max_lr")

    plt.plot(
        standard_lr_df["max_lr"],
        standard_lr_df["final_val_loss"],
        marker="o",
        linewidth=2,
        label="Standard",
    )

    plt.plot(
        mup_lr_df["max_lr"],
        mup_lr_df["final_val_loss"],
        marker="o",
        linewidth=2,
        label="μP",
    )

    # Mark best points
    standard_best = standard_lr_df.loc[standard_lr_df["final_val_loss"].idxmin()]
    mup_best = mup_lr_df.loc[mup_lr_df["final_val_loss"].idxmin()]

    plt.scatter(
        [standard_best["max_lr"]],
        [standard_best["final_val_loss"]],
        s=100,
        marker="*",
        label=f"Standard best LR={standard_best['max_lr']:.1e}",
    )

    plt.scatter(
        [mup_best["max_lr"]],
        [mup_best["final_val_loss"]],
        s=100,
        marker="*",
        label=f"μP best LR={mup_best['max_lr']:.1e}",
    )

    plt.xscale("log")
    plt.xlabel("Peak learning rate (log scale)")
    plt.ylabel("Final validation loss")
    plt.title("Learning Rate Sweep: Standard vs μP")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(LR_SWEEP_PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved LR sweep comparison plot to: {LR_SWEEP_PLOT_PATH}")


def save_powerlaw_fits(standard_fit: dict, mup_fit: dict) -> None:
    """
    Save standard and μP power-law fit parameters to CSV.
    """
    rows = [
        {
            "parameterization": "standard",
            "fit_method": standard_fit["fit_method"],
            "a": standard_fit["a"],
            "alpha": standard_fit["alpha"],
            "c": standard_fit["c"],
            "equation": "L = a * N^(-alpha) + c",
        },
        {
            "parameterization": "mup",
            "fit_method": mup_fit["fit_method"],
            "a": mup_fit["a"],
            "alpha": mup_fit["alpha"],
            "c": mup_fit["c"],
            "equation": "L = a * N^(-alpha) + c",
        },
    ]

    fit_df = pd.DataFrame(rows)
    fit_df.to_csv(POWERLAW_FIT_PATH, index=False)

    print(f"Saved power-law comparison fit parameters to: {POWERLAW_FIT_PATH}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    standard_df = load_scaling_results(
        STANDARD_SCALING_PATH,
        parameterization="standard",
    )

    mup_df = load_scaling_results(
        MUP_SCALING_PATH,
        parameterization="μP",
    )

    print("\nStandard scaling results:")
    print(standard_df[["model_name", "param_count", "final_train_loss", "final_val_loss"]])

    print("\nμP scaling results:")
    print(mup_df[["model_name", "param_count", "final_train_loss", "final_val_loss"]])

    standard_fit = fit_power_law(
        standard_df["param_count"].to_numpy(dtype=float),
        standard_df["final_val_loss"].to_numpy(dtype=float),
    )

    mup_fit = fit_power_law(
        mup_df["param_count"].to_numpy(dtype=float),
        mup_df["final_val_loss"].to_numpy(dtype=float),
    )

    print("\nStandard power-law fit:")
    print(standard_fit)

    print("\nμP power-law fit:")
    print(mup_fit)

    save_powerlaw_fits(standard_fit, mup_fit)

    plot_standard_vs_mup_scaling(
        standard_df=standard_df,
        mup_df=mup_df,
        standard_fit=standard_fit,
        mup_fit=mup_fit,
    )

    standard_lr_df = load_standard_lr_sweeps(STANDARD_LR_SWEEP_PATHS)
    mup_lr_df = load_mup_lr_sweep(MUP_LR_SWEEP_PATH)

    print("\nBest standard LR:")
    print(standard_lr_df.loc[standard_lr_df["final_val_loss"].idxmin()])

    print("\nBest μP LR:")
    print(mup_lr_df.loc[mup_lr_df["final_val_loss"].idxmin()])

    plot_lr_sweeps(standard_lr_df, mup_lr_df)

    print("\nDone.")


if __name__ == "__main__":
    main()