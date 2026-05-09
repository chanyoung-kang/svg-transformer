"""
Save rendered SVG examples at different complexity levels.

This script selects SVG examples from the trainable dataset based on token-count
ranges. It saves candidate SVG files and creates an HTML preview page so the
examples can be visually inspected and used in the report.

Input:
    data/processed/svg_combined_trainable_1024.parquet

Outputs:
    outputs/sample_svgs/*.svg
    outputs/sample_svgs/index.html
"""

from pathlib import Path
import html

import pandas as pd


IN_PATH = Path("data/processed/svg_combined_trainable_1024.parquet")
OUT_DIR = Path("outputs/sample_svgs")

SAMPLE_GROUPS = {
    "low_complexity": (58, 100),
    "medium_complexity": (400, 500),
    "high_complexity": (800, 1024),
}

N_SAMPLES_PER_GROUP = 10
SEED = 42


def safe_filename(value: str) -> str:
    """Make a string safe to use as part of a file name."""
    value = str(value)
    keep_chars = []

    for ch in value:
        if ch.isalnum() or ch in ["-", "_"]:
            keep_chars.append(ch)
        else:
            keep_chars.append("_")

    return "".join(keep_chars)[:80]


def sample_rows_by_token_range(
    df: pd.DataFrame,
    min_tokens: int,
    max_tokens: int,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Sample SVG rows whose token_count falls within a target range."""
    candidates = df[
        (df["token_count"] >= min_tokens)
        & (df["token_count"] <= max_tokens)
    ].copy()

    if len(candidates) == 0:
        raise ValueError(
            f"No SVGs found in token range {min_tokens}-{max_tokens}."
        )

    actual_n = min(n_samples, len(candidates))

    return candidates.sample(n=actual_n, random_state=seed)


def write_html_preview(sample_records):
    """Create an HTML file that displays all saved SVG candidates."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>SVG Sample Candidates</title>",
        """
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 32px;
                background: #fafafa;
            }
            h1 {
                margin-bottom: 8px;
            }
            h2 {
                margin-top: 40px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 8px;
            }
            .note {
                color: #555;
                margin-bottom: 24px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                gap: 20px;
            }
            .card {
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 14px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
            }
            .meta {
                font-size: 13px;
                line-height: 1.35;
                color: #333;
            }
            .svg-box {
                width: 220px;
                height: 220px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px solid #eee;
                background: white;
                margin-top: 12px;
                overflow: hidden;
            }
            svg {
                max-width: 200px;
                max-height: 200px;
            }
            code {
                font-size: 12px;
                word-break: break-all;
            }
            a {
                color: #4b2e83;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
        """,
        "</head>",
        "<body>",
        "<h1>SVG Sample Candidates by Complexity</h1>",
        (
            '<p class="note">'
            "Choose one visually representative SVG from each group for the report. "
            "Complexity is approximated using SentencePiece token count."
            "</p>"
        ),
    ]

    grouped = {}
    for record in sample_records:
        grouped.setdefault(record["group"], []).append(record)

    for group, records in grouped.items():
        title = group.replace("_", " ").title()
        html_parts.append(f"<h2>{title}</h2>")
        html_parts.append('<div class="grid">')

        for record in records:
            row = record["row"]
            svg_filename = record["svg_filename"]

            html_parts.extend(
                [
                    '<div class="card">',
                    f"<h3>{html.escape(record['display_name'])}</h3>",
                    '<div class="meta">',
                    f"<p><strong>Source:</strong> {html.escape(str(row['source']))}</p>",
                    f"<p><strong>Filename:</strong> <code>{html.escape(str(row['filename']))}</code></p>",
                    f"<p><strong>Token count:</strong> {int(row['token_count']):,}</p>",
                    f'<p><a href="{html.escape(svg_filename)}" target="_blank">Open SVG file</a></p>',
                    "</div>",
                    '<div class="svg-box">',
                    row["svg"],
                    "</div>",
                    "</div>",
                ]
            )

        html_parts.append("</div>")

    html_parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    html_path = OUT_DIR / "index.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")

    return html_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH)

    sample_records = []

    for group_name, (min_tokens, max_tokens) in SAMPLE_GROUPS.items():
        sampled = sample_rows_by_token_range(
            df=df,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            n_samples=N_SAMPLES_PER_GROUP,
            seed=SEED,
        )

        print(f"\n{group_name}")
        print(f"  Token range: {min_tokens}-{max_tokens}")
        print(f"  Candidates available: {len(df[(df['token_count'] >= min_tokens) & (df['token_count'] <= max_tokens)]):,}")
        print(f"  Saved candidates: {len(sampled):,}")

        for i, (_, row) in enumerate(sampled.iterrows(), start=1):
            display_name = f"{group_name}_{i:02d}"
            source = safe_filename(row["source"])
            filename = safe_filename(row["filename"])

            svg_filename = f"{display_name}_{source}_{filename}.svg"
            svg_path = OUT_DIR / svg_filename

            svg_path.write_text(row["svg"], encoding="utf-8")

            sample_records.append(
                {
                    "group": group_name,
                    "display_name": display_name,
                    "svg_filename": svg_filename,
                    "row": row,
                }
            )

            print(
                f"  {display_name}: "
                f"tokens={int(row['token_count']):,}, "
                f"source={row['source']}, "
                f"saved={svg_path}"
            )

    html_path = write_html_preview(sample_records)

    print(f"\nSaved HTML preview to: {html_path}")
    print("Open this file in your browser to view rendered SVG candidates.")


if __name__ == "__main__":
    main()