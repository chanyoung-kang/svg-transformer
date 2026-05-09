"""
Script 27: Evaluate Part 4 generated SVG outputs.

This script summarizes:
1. Final test loss and perplexity from Part 4 training
2. XML validity rate for unconditional samples
3. XML validity rate by temperature
4. XML validity rate for prefix-conditioned samples
5. XML validity rate by prefix prompt
6. Optional SVG render rate using CairoSVG, if installed

Inputs:
    outputs/csvs/part4_best_model_metrics.csv
    outputs/generated_svg_samples/generation_summary.csv
    outputs/prefix_completion_samples/prefix_completion_summary.csv

Outputs:
    outputs/csvs/part4_generation_evaluation_summary.csv
"""

from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET


# -----------------------------
# Paths
# -----------------------------

METRICS_PATH = Path("outputs/csvs/part4_best_model_metrics.csv")

UNCONDITIONAL_SUMMARY_PATH = Path(
    "outputs/generated_svg_samples/generation_summary.csv"
)

PREFIX_SUMMARY_PATH = Path(
    "outputs/prefix_completion_samples/prefix_completion_summary.csv"
)

UNCONDITIONAL_SVG_DIR = Path("outputs/generated_svg_samples/svg_files")
PREFIX_SVG_DIR = Path("outputs/prefix_completion_samples/svg_files")

OUTPUT_PATH = Path("outputs/csvs/part4_generation_evaluation_summary.csv")


# -----------------------------
# Optional CairoSVG render check
# -----------------------------

def try_import_cairosvg():
    try:
        import cairosvg
        return cairosvg
    except Exception:
        return None


def check_xml_valid(svg_path: Path) -> bool:
    try:
        text = svg_path.read_text(encoding="utf-8", errors="ignore")
        ET.fromstring(text)
        return True
    except Exception:
        return False


def check_renderable(svg_path: Path, cairosvg_module) -> bool:
    """
    Returns True if the SVG can render to PNG using CairoSVG.
    This does not save the PNG permanently.
    """
    if cairosvg_module is None:
        return False

    try:
        cairosvg_module.svg2png(url=str(svg_path), write_to=None)
        return True
    except Exception:
        return False


def add_render_checks(df: pd.DataFrame, svg_dir: Path, cairosvg_module) -> pd.DataFrame:
    """
    Adds renderable column if SVG files exist.
    Only attempts rendering for XML-valid rows.
    """
    df = df.copy()

    renderable_values = []

    for _, row in df.iterrows():
        svg_filename = row.get("svg_filename")

        if pd.isna(svg_filename):
            renderable_values.append(False)
            continue

        svg_path = svg_dir / str(svg_filename)

        if not svg_path.exists():
            renderable_values.append(False)
            continue

        if not bool(row.get("valid_xml", False)):
            renderable_values.append(False)
            continue

        renderable_values.append(check_renderable(svg_path, cairosvg_module))

    df["renderable"] = renderable_values
    return df


def summarize_boolean_rate(df: pd.DataFrame, col: str) -> dict:
    total = len(df)

    if total == 0 or col not in df.columns:
        return {
            "count": total,
            "valid_count": 0,
            "rate": None,
        }

    valid_count = int(df[col].sum())
    rate = float(df[col].mean())

    return {
        "count": total,
        "valid_count": valid_count,
        "rate": rate,
    }


def main():
    print("Part 4 generation evaluation")
    print("=" * 80)

    rows = []

    # -----------------------------
    # 1. Training metrics
    # -----------------------------

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        print(f"\nLoaded training metrics from: {METRICS_PATH}")
        print(metrics)

        for col in metrics.columns:
            value = metrics[col].iloc[0]
            rows.append(
                {
                    "section": "training",
                    "metric": col,
                    "group": "overall",
                    "count": None,
                    "valid_count": None,
                    "rate": value,
                }
            )
    else:
        print(f"\nMissing training metrics file: {METRICS_PATH}")

    # -----------------------------
    # 2. Load generation summaries
    # -----------------------------

    if not UNCONDITIONAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {UNCONDITIONAL_SUMMARY_PATH}")

    if not PREFIX_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PREFIX_SUMMARY_PATH}")

    uncond = pd.read_csv(UNCONDITIONAL_SUMMARY_PATH)
    prefix = pd.read_csv(PREFIX_SUMMARY_PATH)

    print(f"\nLoaded unconditional summary: {UNCONDITIONAL_SUMMARY_PATH}")
    print(f"Rows: {len(uncond)}")

    print(f"\nLoaded prefix summary: {PREFIX_SUMMARY_PATH}")
    print(f"Rows: {len(prefix)}")

    # Convert valid_xml if it was read as string.
    for df in [uncond, prefix]:
        if "valid_xml" in df.columns:
            df["valid_xml"] = df["valid_xml"].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(df["valid_xml"])

    # -----------------------------
    # 3. XML validity rates
    # -----------------------------

    uncond_xml = summarize_boolean_rate(uncond, "valid_xml")
    prefix_xml = summarize_boolean_rate(prefix, "valid_xml")

    print("\nXML validity")
    print("-" * 80)
    print(
        f"Unconditional: {uncond_xml['valid_count']}/{uncond_xml['count']} "
        f"= {uncond_xml['rate']:.4f}"
    )
    print(
        f"Prefix-conditioned: {prefix_xml['valid_count']}/{prefix_xml['count']} "
        f"= {prefix_xml['rate']:.4f}"
    )

    rows.append(
        {
            "section": "unconditional",
            "metric": "xml_validity_rate",
            "group": "overall",
            "count": uncond_xml["count"],
            "valid_count": uncond_xml["valid_count"],
            "rate": uncond_xml["rate"],
        }
    )

    rows.append(
        {
            "section": "prefix_conditioned",
            "metric": "xml_validity_rate",
            "group": "overall",
            "count": prefix_xml["count"],
            "valid_count": prefix_xml["valid_count"],
            "rate": prefix_xml["rate"],
        }
    )

    # -----------------------------
    # 4. Validity by temperature
    # -----------------------------

    if "temperature" in uncond.columns:
        print("\nUnconditional XML validity by temperature")
        print("-" * 80)

        by_temp = uncond.groupby("temperature")["valid_xml"].agg(
            count="count",
            valid_count="sum",
            rate="mean",
        )

        print(by_temp)

        for temp, row in by_temp.iterrows():
            rows.append(
                {
                    "section": "unconditional",
                    "metric": "xml_validity_rate",
                    "group": f"temperature={temp}",
                    "count": int(row["count"]),
                    "valid_count": int(row["valid_count"]),
                    "rate": float(row["rate"]),
                }
            )

    # -----------------------------
    # 5. Validity by prefix prompt
    # -----------------------------

    if "prompt_name" in prefix.columns:
        print("\nPrefix-conditioned XML validity by prompt")
        print("-" * 80)

        by_prompt = prefix.groupby("prompt_name")["valid_xml"].agg(
            count="count",
            valid_count="sum",
            rate="mean",
        ).sort_values("rate", ascending=False)

        print(by_prompt)

        for prompt_name, row in by_prompt.iterrows():
            rows.append(
                {
                    "section": "prefix_conditioned",
                    "metric": "xml_validity_rate",
                    "group": str(prompt_name),
                    "count": int(row["count"]),
                    "valid_count": int(row["valid_count"]),
                    "rate": float(row["rate"]),
                }
            )

    # -----------------------------
    # 6. Optional render rate
    # -----------------------------

    cairosvg = try_import_cairosvg()

    print("\nSVG renderability")
    print("-" * 80)

    if cairosvg is None:
        print("CairoSVG is not installed. Skipping render-rate check.")
        print("To enable it, run: !pip install cairosvg")
    else:
        print("CairoSVG found. Checking renderability...")

        uncond_render = add_render_checks(uncond, UNCONDITIONAL_SVG_DIR, cairosvg)
        prefix_render = add_render_checks(prefix, PREFIX_SVG_DIR, cairosvg)

        uncond_render_rate = summarize_boolean_rate(uncond_render, "renderable")
        prefix_render_rate = summarize_boolean_rate(prefix_render, "renderable")

        print(
            f"Unconditional render rate: "
            f"{uncond_render_rate['valid_count']}/{uncond_render_rate['count']} "
            f"= {uncond_render_rate['rate']:.4f}"
        )

        print(
            f"Prefix-conditioned render rate: "
            f"{prefix_render_rate['valid_count']}/{prefix_render_rate['count']} "
            f"= {prefix_render_rate['rate']:.4f}"
        )

        rows.append(
            {
                "section": "unconditional",
                "metric": "svg_render_rate",
                "group": "overall",
                "count": uncond_render_rate["count"],
                "valid_count": uncond_render_rate["valid_count"],
                "rate": uncond_render_rate["rate"],
            }
        )

        rows.append(
            {
                "section": "prefix_conditioned",
                "metric": "svg_render_rate",
                "group": "overall",
                "count": prefix_render_rate["count"],
                "valid_count": prefix_render_rate["valid_count"],
                "rate": prefix_render_rate["rate"],
            }
        )

    # -----------------------------
    # 7. Save summary
    # -----------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved evaluation summary to:")
    print(OUTPUT_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()