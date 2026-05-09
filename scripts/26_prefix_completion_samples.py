"""
Script 26: Prefix-conditioned SVG completion samples.

This script:
    1. Loads the best Part 4 checkpoint.
    2. Uses 10 manually designed partial SVG prefixes.
    3. Generates multiple completions per prefix.
    4. Saves raw generated text and final SVG files.
    5. Checks XML validity.
    6. Builds an HTML gallery showing:
        - left: prefix SVG input
        - right: model-completed SVG output

Inputs:
    data/tokenizer/svg_bpe.model
    outputs/checkpoints/best_standard_xlarge_part4.pt

Outputs:
    outputs/prefix_completion_samples/
"""

from pathlib import Path
import csv
import html
import random
import xml.etree.ElementTree as ET

import sentencepiece as spm
import torch
import torch.nn.functional as F

from model import GPTConfig, GPT


# -----------------------------
# Paths
# -----------------------------

TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")
CHECKPOINT_PATH = Path("outputs/checkpoints/best_standard_xlarge_part4.pt")

OUTPUT_DIR = Path("outputs/prefix_completion_samples")
RAW_DIR = OUTPUT_DIR / "raw_text"
SVG_DIR = OUTPUT_DIR / "svg_files"

SUMMARY_CSV_PATH = OUTPUT_DIR / "prefix_completion_summary.csv"
GALLERY_HTML_PATH = OUTPUT_DIR / "gallery.html"


# -----------------------------
# Generation settings
# -----------------------------

SEED = 42

NUM_SAMPLES_PER_PROMPT = 2
TEMPERATURE = 0.7
TOP_K = 30
MAX_NEW_TOKENS = 100
APPEND_CLOSING_SVG_TAG = True


# -----------------------------
# Prefix prompts
# -----------------------------
# These are the SVG prefixes the model starts from.
# The model then generates the continuation.

PREFIX_PROMPTS = {

    # Block-style T: give top bar and most of vertical stem, leave lower stem / cleanup for model
    "block_letter_T": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".5" d="M3.0 4.0 L21.0 4.0 L21.0 7.0 L14.0 7.0 L14.0 18.0"></path>',

    # Curved J: give vertical stem and beginning of curve
    "curved_letter_J": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".5" d="M15.5 4.0 L15.5 15.0 C15.5 18.0 13.5 20.0 10.5 20.0 L5.5 20.0"></path>',

    # Champagne glasses: give first glass and start of second glass
    "champagne_glasses": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".45" d="M5.0 8.0 C5.0 5.5 7.0 4.0 9.0 4.0 C11.0 4.0 13.0 5.5 13.0 8.0 C13.0 11.0 11.0 13.0 9.0 13.0 C7.0 13.0 5.0 11.0 5.0 8.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M9.0 13.0 L8.0 19.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M6.0 20.0 L10.0 20.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M12.0 7.0 C13.0 5.2 15.0 4.5 17.0 5.0 C19.0 5.5 20.0 7.5 19.5 9.5"></path>',

    # Grapes: give a few circles/ovals and stem, see if model adds more grape-like circles
    "grapes_cluster": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".45" d="M15.0 3.0 C16.5 4.5 17.5 5.5 19.0 6.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M12.0 6.0 C10.8 6.0 10.0 6.8 10.0 8.0 C10.0 9.2 10.8 10.0 12.0 10.0 C13.2 10.0 14.0 9.2 14.0 8.0 C14.0 6.8 13.2 6.0 12.0 6.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M8.0 9.0 C6.8 9.0 6.0 9.8 6.0 11.0 C6.0 12.2 6.8 13.0 8.0 13.0 C9.2 13.0 10.0 12.2 10.0 11.0 C10.0 9.8 9.2 9.0 8.0 9.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M16.0 9.0 C14.8 9.0 14.0 9.8 14.0 11.0 C14.0 12.2 14.8 13.0 16.0 13.0 C17.2 13.0 18.0 12.2 18.0 11.0 C18.0 9.8 17.2 9.0 16.0 9.0"></path>',

    # Easier version: repeated grape circles with more obvious pattern
    "grapes_more_complete": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".45" d="M15.0 3.0 C16.0 4.0 17.0 5.0 18.0 6.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M12.0 6.0 C10.8 6.0 10.0 6.8 10.0 8.0 C10.0 9.2 10.8 10.0 12.0 10.0 C13.2 10.0 14.0 9.2 14.0 8.0 C14.0 6.8 13.2 6.0 12.0 6.0"></path><path fill="none" stroke="black" stroke-width=".45" d="M8.0 9.5 C6.8 9.5 6.0 10.3 6.0 11.5 C6.0 12.7 6.8 13.5 8.0 13.5 C9.2 13.5 10.0 12.7 10.0 11.5 C10.0 10.3 9.2 9.5 8.0 9.5"></path><path fill="none" stroke="black" stroke-width=".45" d="M16.0 9.5 C14.8 9.5 14.0 10.3 14.0 11.5 C14.0 12.7 14.8 13.5 16.0 13.5 C17.2 13.5 18.0 12.7 18.0 11.5 C18.0 10.3 17.2 9.5 16.0 9.5"></path><path fill="none" stroke="black" stroke-width=".45" d="M12.0 13.0 C10.8 13.0 10.0 13.8 10.0 15.0 C10.0 16.2 10.8 17.0 12.0 17.0 C13.2 17.0 14.0 16.2 14.0 15.0 C14.0 13.8 13.2 13.0 12.0 13.0"></path>',

    # Easier font-like I/T hybrid, very similar to the screenshot
    "block_letter_I": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".5" d="M4.0 4.0 L20.0 4.0 L20.0 7.0 L14.0 7.0 L14.0 17.0 L20.0 17.0"></path>',

    # "three_sides_square": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M6.0 6.0 L18.0 6.0 L18.0 18.0 L6.0 18.0"></path>',

    # "two_sides_triangle": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M12.0 5.0 L19.0 18.0 L5.0 18.0"></path>',

    # "parallel_lines": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M6.0 8.0 L18.0 8.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M6.0 12.0 L18.0 12.0"></path>',

    # "vertical_bars": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M8.0 6.0 L8.0 18.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M12.0 6.0 L12.0 18.0"></path>',

    # "simple_grid": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M6.0 6.0 L18.0 6.0 L18.0 18.0 L6.0 18.0 L6.0 6.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M12.0 6.0 L12.0 18.0"></path>',

    # "staircase": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M5.0 18.0 L9.0 18.0 L9.0 14.0 L13.0 14.0 L13.0 10.0"></path>',

    # "zigzag": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M5.0 12.0 L8.0 8.0 L11.0 12.0 L14.0 8.0"></path>',

    # "letter_E": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M8.0 6.0 L8.0 18.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M8.0 6.0 L16.0 6.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M8.0 12.0 L14.0 12.0"></path>',

    # "letter_H": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M8.0 6.0 L8.0 18.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M16.0 6.0 L16.0 18.0"></path>',

    # "letter_T": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M6.0 6.0 L18.0 6.0"></path>',

    # # Face: rounder / larger eyes, missing mouth
    # "face_missing_mouth": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M6.0 12.0 C6.0 8.0 8.5 5.5 12.0 5.5 C15.5 5.5 18.0 8.0 18.0 12.0 C18.0 16.0 15.5 18.5 12.0 18.5 C8.5 18.5 6.0 16.0 6.0 12.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M8.7 10.0 C8.7 9.5 9.1 9.1 9.6 9.1 C10.1 9.1 10.5 9.5 10.5 10.0 C10.5 10.5 10.1 10.9 9.6 10.9 C9.1 10.9 8.7 10.5 8.7 10.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M13.5 10.0 C13.5 9.5 13.9 9.1 14.4 9.1 C14.9 9.1 15.3 9.5 15.3 10.0 C15.3 10.5 14.9 10.9 14.4 10.9 C13.9 10.9 13.5 10.5 13.5 10.0"></path>',

    # # Heart: about 70% complete, missing final right/bottom closure
    # "partial_heart": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M12.0 20.0 C10.0 18.0 7.0 16.0 6.0 12.5 C5.0 9.0 7.0 6.0 10.0 6.0 C11.2 6.0 12.0 6.8 12.0 8.0 C12.0 6.8 12.8 6.0 14.0 6.0 C16.2 6.0 18.0 7.6 18.3 10.0"></path>',

    # # Star: about 70% complete, missing last few edges
    # "partial_star": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M12.0 4.0 L14.0 9.0 L19.5 9.0 L15.0 12.5 L16.5 18.0 L12.0 14.6 L8.0 18.0"></path>',

    # # Flower: previous style, moved upward to leave room for possible stem
    # "partial_flower": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M12.0 6.8 C11.0 5.3 10.0 4.5 8.5 4.5 C7.2 4.5 6.2 5.5 6.2 6.8 C6.2 8.3 7.4 9.2 9.0 9.5 C7.4 9.8 6.2 10.7 6.2 12.2 C6.2 13.5 7.2 14.5 8.5 14.5 C10.0 14.5 11.0 13.5 12.0 12.0 C13.0 13.5 14.0 14.5 15.5 14.5 C16.8 14.5 17.8 13.5 17.8 12.2"></path><path fill="none" stroke="black" stroke-width=".3" d="M11.3 9.5 C11.3 9.1 11.6 8.8 12.0 8.8 C12.4 8.8 12.7 9.1 12.7 9.5 C12.7 9.9 12.4 10.2 12.0 10.2 C11.6 10.2 11.3 9.9 11.3 9.5"></path>',

    # # Arrow: keep simple partial arrow
    # "partial_arrow": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M5.0 12.0 L17.0 12.0 L13.5 8.5"></path>',

    # # House: roof + left wall + right wall, missing bottom/details
    # "partial_house": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M6.0 11.0 L12.0 6.0 L18.0 11.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M7.5 11.0 L7.5 18.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M16.5 11.0 L16.5 18.0"></path>',

    # # A: about 70% complete, missing right/down finishing behavior
    # "partial_A": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M8.0 18.0 L12.0 6.0 L15.0 15.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M9.7 13.0 L14.3 13.0"></path>',

    # # B: vertical + upper loop + start of lower loop
    # "partial_B": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M8.0 6.0 L8.0 18.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M8.0 6.0 C12.8 6.0 15.5 7.0 15.5 10.0 C15.5 12.5 12.8 13.0 8.0 13.0"></path><path fill="none" stroke="black" stroke-width=".3" d="M8.0 13.0 C12.5 13.0 15.0 14.0 15.2 16.2"></path>',

    # # S: about 70% complete, missing lower tail
    # "partial_S": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M16.0 7.5 C15.0 6.5 13.8 6.0 12.0 6.0 C9.5 6.0 8.0 7.0 8.0 8.8 C8.0 10.5 9.5 11.2 12.0 11.8 C14.5 12.4 16.0 13.1 16.0 15.0"></path>',

    # # M: about 70% complete, missing final right vertical extension / cleanup
    # "partial_M": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 24.0 24.0" height="200px" width="200px"><path fill="none" stroke="black" stroke-width=".3" d="M7.0 18.0 L7.0 6.0 L12.0 12.0 L17.0 6.0 L17.0 14.0"></path>',
}


def build_config_from_checkpoint(checkpoint: dict) -> GPTConfig:
    cfg = checkpoint["config"]
    return GPTConfig(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0),
    )


def get_start_ids(sp, prompt: str) -> list[int]:
    ids = sp.encode(prompt, out_type=int)
    if len(ids) == 0:
        bos_id = sp.bos_id()
        if bos_id >= 0:
            ids = [bos_id]
        else:
            ids = [0]
    return ids


@torch.no_grad()
def generate_ids(
    model,
    start_ids: list[int],
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> list[int]:
    model.eval()

    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond, None)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = values[:, [-1]]
                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(logits, -float("inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

    return idx[0].tolist()


def extract_svg(text: str) -> str:
    start = text.find("<svg")

    if start == -1:
        text = "<svg " + text
        start = 0

    text = text[start:]

    end = text.find("</svg>")
    if end != -1:
        text = text[: end + len("</svg>")]
    elif APPEND_CLOSING_SVG_TAG:
        text = text.rstrip() + "\n</svg>"

    return text.strip()


def is_valid_xml(svg_text: str) -> bool:
    try:
        ET.fromstring(svg_text)
        return True
    except Exception:
        return False


def make_displayable_prefix_svg(prompt_text: str) -> tuple[str, bool]:
    """
    Convert the prefix prompt into a displayable SVG snippet.

    The prompt normally starts with <svg> and contains the initial shapes.
    It may not include a closing </svg>, so we append it only for display.
    This does not affect model generation.
    """
    prefix_svg = prompt_text.strip()

    if "</svg>" not in prefix_svg:
        prefix_svg = prefix_svg + "\n</svg>"

    try:
        ET.fromstring(prefix_svg)
        return prefix_svg, True
    except Exception:
        return """
        <div class="invalid-placeholder">
            Prefix not renderable<br>
            Text shown below
        </div>
        """, False


def save_gallery(rows: list[dict]) -> None:
    cards = []

    for row in rows:
        svg_filename = row["svg_filename"]
        raw_filename = row["raw_filename"]
        prompt_name = row["prompt_name"]
        prompt_text = row["prompt_text"]
        valid_xml = row["valid_xml"]
        sample_id = row["sample_id"]

        svg_path = SVG_DIR / svg_filename

        if svg_path.exists() and valid_xml:
            svg_content = svg_path.read_text(encoding="utf-8", errors="ignore")
        else:
            svg_content = """
            <div class="invalid-placeholder">
                Invalid XML<br>
                SVG not displayed
            </div>
            """

        prefix_svg_content, prefix_valid_xml = make_displayable_prefix_svg(prompt_text)

        card = f"""
        <div class="card">
            <h3>Sample {sample_id} | {html.escape(prompt_name)} | valid_xml={valid_xml}</h3>

            <div class="comparison-row">
                <div class="comparison-col">
                    <div class="label">Prefix input</div>
                    <div class="svgbox">
                        {prefix_svg_content}
                    </div>
                    <p><strong>Prefix renderable:</strong> {prefix_valid_xml}</p>
                </div>

                <div class="comparison-col">
                    <div class="label">Model completion</div>
                    <div class="svgbox">
                        {svg_content}
                    </div>
                    <p><strong>Output valid XML:</strong> {valid_xml}</p>
                </div>
            </div>

            <p><strong>SVG:</strong> {html.escape(svg_filename)}</p>
            <p><strong>Raw:</strong> {html.escape(raw_filename)}</p>

            <details>
                <summary><strong>Show prefix prompt text</strong></summary>
                <pre>{html.escape(prompt_text)}</pre>
            </details>
        </div>
        """
        cards.append(card)

    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Prefix Completion SVG Samples</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 24px;
                background: #f7f7f7;
            }}
            h1 {{
                margin-bottom: 16px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
                gap: 16px;
            }}
            .card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                overflow: hidden;
            }}
            .comparison-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                align-items: start;
            }}
            .comparison-col {{
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .label {{
                font-size: 13px;
                font-weight: bold;
                margin-bottom: 6px;
                color: #333;
            }}
            .svgbox {{
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 8px;
                width: 220px;
                height: 220px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                background: white;
            }}
            .svgbox svg {{
                width: 200px !important;
                height: 200px !important;
                max-width: 200px !important;
                max-height: 200px !important;
                display: block;
                overflow: hidden;
            }}
            .invalid-placeholder {{
                width: 200px;
                height: 200px;
                border: 1px dashed #aaa;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #777;
                font-size: 13px;
                background: #fafafa;
            }}
            p {{
                font-size: 13px;
                color: #444;
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-word;
                background: #f3f3f3;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
            }}
            details {{
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Prefix-Conditioned SVG Completion Samples</h1>
        <p>Checkpoint: {html.escape(str(CHECKPOINT_PATH))}</p>
        <p>Temperature: {TEMPERATURE} | Top-k: {TOP_K} | Max new tokens: {MAX_NEW_TOKENS}</p>
        <p>Each card shows the prefix SVG input on the left and the model-completed output on the right.</p>
        <div class="grid">
            {''.join(cards)}
        </div>
    </body>
    </html>
    """

    GALLERY_HTML_PATH.write_text(page, encoding="utf-8")
    print(f"Saved gallery to: {GALLERY_HTML_PATH}")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not TOKENIZER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing tokenizer: {TOKENIZER_MODEL_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    print(f"Loading tokenizer: {TOKENIZER_MODEL_PATH}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL_PATH))

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    config = build_config_from_checkpoint(checkpoint)

    print("\nModel config:")
    print(config)

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nRunning prefix-conditioned generation...")
    print(f"Number of prompt types: {len(PREFIX_PROMPTS)}")
    print(f"Samples per prompt: {NUM_SAMPLES_PER_PROMPT}")
    print(f"Seed: {SEED}")

    rows = []
    sample_counter = 0

    for prompt_name, prompt_text in PREFIX_PROMPTS.items():
        print(f"\nPrompt: {prompt_name}")

        start_ids = get_start_ids(sp, prompt_text)

        for _ in range(NUM_SAMPLES_PER_PROMPT):
            sample_counter += 1

            generated_ids = generate_ids(
                model=model,
                start_ids=start_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                block_size=config.block_size,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                device=device,
            )

            generated_text = sp.decode(generated_ids)

            raw_filename = f"sample_{sample_counter:03d}_{prompt_name}.txt"
            svg_filename = f"sample_{sample_counter:03d}_{prompt_name}.svg"

            raw_path = RAW_DIR / raw_filename
            svg_path = SVG_DIR / svg_filename

            raw_path.write_text(generated_text, encoding="utf-8")

            svg_text = extract_svg(generated_text)
            valid_xml = is_valid_xml(svg_text)

            svg_path.write_text(svg_text, encoding="utf-8")

            row = {
                "sample_id": sample_counter,
                "prompt_name": prompt_name,
                "prompt_text": prompt_text,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS,
                "num_tokens_generated": len(generated_ids),
                "num_chars_raw": len(generated_text),
                "num_chars_svg": len(svg_text),
                "valid_xml": valid_xml,
                "raw_filename": raw_filename,
                "svg_filename": svg_filename,
            }
            rows.append(row)

            print(
                f"  Sample {sample_counter:03d} | "
                f"valid_xml={valid_xml} | "
                f"saved {svg_filename}"
            )

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary CSV to: {SUMMARY_CSV_PATH}")

    save_gallery(rows)

    valid_count = sum(1 for row in rows if row["valid_xml"])
    total_count = len(rows)

    print("\nPrefix-conditioned generation complete.")
    print(f"Total samples: {total_count}")
    print(f"Valid XML SVG samples: {valid_count}/{total_count}")
    print(f"Raw text directory: {RAW_DIR}")
    print(f"SVG directory: {SVG_DIR}")
    print(f"Gallery HTML: {GALLERY_HTML_PATH}")


if __name__ == "__main__":
    main()