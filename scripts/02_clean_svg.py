"""
Clean and validate raw SVG strings.

This script loads the combined raw SVG dataset, removes comments and metadata,
normalizes long numeric coordinates, removes unnecessary whitespace, filters out
SVGs that are too short or too long, and validates that each remaining SVG parses
as valid XML.

Input:
    data/raw/svg_combined_raw.parquet

Output:
    data/processed/svg_combined_clean.parquet

Output columns:
    source, filename, svg, char_count
"""

import re # regular expression library - use it to find and remove patterns in text, like XML comments or long decimal numbers
from pathlib import Path

import pandas as pd
from lxml import etree # checks whether the SVG is valid XML
from tqdm import tqdm # for progress bars
# import cairosvg

# input path
RAW_PATH = Path("data/raw/svg_combined_raw.parquet")
# output path
OUT_PATH = Path("data/processed/svg_combined_clean.parquet")

SVG_COL = "svg"

MIN_CHARS = 50 # remove SVG strings shorter than 50 characters
MAX_CHARS = 5000 # remove SVG strings longer than 5000 characters


def clean_svg(svg: str) -> str:
    """Basic SVG text cleanup."""

    # If the SVG value is not a string like None, return an empty string.
    if not isinstance(svg, str):
        return ""

    # Remove XML comments
    # re.DOTALL means the pattern can match across multiple lines.
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.DOTALL)

    # Remove metadata blocks if present
    svg = re.sub(r"<metadata.*?</metadata>", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<title.*?</title>", "", svg, flags=re.DOTALL | re.IGNORECASE)
    svg = re.sub(r"<desc.*?</desc>", "", svg, flags=re.DOTALL | re.IGNORECASE)

    # Normalize numeric precision: round long decimals to 1 decimal place
    """
    SVG path coordinates are the numbers that tell the browser where to draw lines and curves.
    For example: <path d="M10.059374809265137 7.275000095367432 L14.58566665649414 7.627137660980225" />
    The important part is the d="..." attribute.
    Inside d, commands like M, L, and C tell the SVG renderer what to do:
        M = move to this point
        L = draw a line to this point
        C = draw a curve using control points
    SVG path coordinates can have very long decimals. 
    The model learns from tokens. Without rounding, it sees tons of unique numbers:
        10.059374809265137
        10.060112953186035
        10.06180477142334
        10.077319145202637
    These are technically different strings, so the tokenizer/model has to deal with many rare numeric patterns.
    After rounding, many of them collapse into simpler repeated values:
        10.1
        10.1
        10.1
        10.1
    That reduces the number of weird one-off tokens and makes the SVG “language” easier to learn.
    Rounding makes the SVG code shorter and simpler while usually preserving the rough visual shape.
    """
    def round_decimal(match):
        return f"{float(match.group()):.1f}"

    svg = re.sub(r"-?\d+\.\d{2,}", round_decimal, svg)

    # Remove extra whitespace/newlines
    svg = re.sub(r"\s+", " ", svg).strip()

    return svg

# Check XML validity using lxml.etree
def is_valid_xml(svg: str) -> bool:
    """Check whether SVG parses as XML."""
    try:
        etree.fromstring(svg.encode("utf-8"))
        return True
    except Exception:
        return False

# # Check whether SVG can be rendered into PNG using CairoSVG
# def can_render(svg):
#     try:
#         cairosvg.svg2png(bytestring=svg.encode("utf-8"))
#         return True
#     except Exception:
#         return False

def main():
    df = pd.read_parquet(RAW_PATH)

    cleaned_rows = []
    invalid_count = 0
    # render_invalid_count = 0
    too_short_count = 0
    too_long_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source = row["source"]
        filename = row["filename"]
        svg = clean_svg(row[SVG_COL])

        if len(svg) < MIN_CHARS:
            too_short_count += 1
            continue

        if len(svg) > MAX_CHARS:
            too_long_count += 1
            continue

        if not is_valid_xml(svg):
            invalid_count += 1
            continue

        # if not can_render_svg(svg):
        #     render_invalid_count += 1
        #     continue

        cleaned_rows.append(
            {
                "source": source,
                "filename": filename,
                "svg": svg,
                "char_count": len(svg),
            }
        )

    out = pd.DataFrame(cleaned_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(f"Original rows: {len(df):,}")
    print(f"Clean rows: {len(out):,}")
    print(f"Too short: {too_short_count:,}")
    print(f"Too long: {too_long_count:,}")
    print(f"Invalid XML: {invalid_count:,}")
    # print(f"Render invalid: {render_invalid_count:,}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()