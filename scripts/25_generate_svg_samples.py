"""
Part 4 / Script 25: Generate SVG samples from the trained best xlarge Transformer.

This script:
    1. Loads the best Part 4 checkpoint.
    2. Loads the SentencePiece tokenizer.
    3. Generates SVG-like text samples.
    4. Saves raw generated text.
    5. Extracts / fixes SVG snippets when possible.
    6. Saves .svg files.
    7. Saves a CSV summary.
    8. Saves a simple HTML gallery for quick viewing.

Inputs:
    data/tokenizer/svg_bpe.model
    outputs/checkpoints/best_standard_xlarge_part4.pt

Outputs:
    outputs/generated_svg_samples/
"""

from pathlib import Path
import csv
import html
import math
import random
import re
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

OUTPUT_DIR = Path("outputs/generated_svg_samples")
RAW_DIR = OUTPUT_DIR / "raw_text"
SVG_DIR = OUTPUT_DIR / "svg_files"

SUMMARY_CSV_PATH = OUTPUT_DIR / "generation_summary.csv"
GALLERY_HTML_PATH = OUTPUT_DIR / "gallery.html"


# -----------------------------
# Generation settings
# -----------------------------

SEED = 42

NUM_SAMPLES_PER_TEMPERATURE = 10

TEMPERATURES = [0.7, 0.9, 1.0]

TOP_K = 50

MAX_NEW_TOKENS = 800

PROMPT = "<svg"

# If generated text does not close the SVG tag, we can optionally append it.
APPEND_CLOSING_SVG_TAG = True


def build_config_from_checkpoint(checkpoint: dict) -> GPTConfig:
    """
    Rebuild GPTConfig from checkpoint config.
    """
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
    """
    Encode prompt into token IDs.
    """
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
    """
    Autoregressive next-token generation.
    """
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

        # Optional early stop if EOS is generated.
        eos_id = -1
        try:
            eos_id = model.config.eos_id
        except Exception:
            pass

        if eos_id >= 0 and next_id.item() == eos_id:
            break

    return idx[0].tolist()


def extract_svg(text: str) -> str:
    """
    Try to extract a clean SVG snippet from generated text.
    """
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
    """
    Check whether generated SVG is valid XML.
    """
    try:
        ET.fromstring(svg_text)
        return True
    except Exception:
        return False


def save_gallery(rows: list[dict]) -> None:
    """
    Save a simple HTML page that displays generated SVG files.
    """
    cards = []

    for row in rows:
        svg_filename = row["svg_filename"]
        raw_filename = row["raw_filename"]
        valid_xml = row["valid_xml"]
        temperature = row["temperature"]
        sample_id = row["sample_id"]

        svg_path = SVG_DIR / svg_filename

        svg_content = ""
        if svg_path.exists():
            svg_content = svg_path.read_text(encoding="utf-8", errors="ignore")

        card = f"""
        <div class="card">
            <h3>Sample {sample_id} | temp={temperature} | valid_xml={valid_xml}</h3>
            <div class="svgbox">
                {svg_content}
            </div>
            <p><strong>SVG:</strong> {html.escape(svg_filename)}</p>
            <p><strong>Raw:</strong> {html.escape(raw_filename)}</p>
        </div>
        """
        cards.append(card)

    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Generated SVG Samples</title>
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
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 16px;
            }}
            .card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            .svgbox {{
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 8px;
                min-height: 180px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: auto;
                background: white;
            }}
            svg {{
                max-width: 220px;
                max-height: 220px;
            }}
            p {{
                font-size: 13px;
                color: #444;
            }}
        </style>
    </head>
    <body>
        <h1>Generated SVG Samples</h1>
        <p>Checkpoint: {html.escape(str(CHECKPOINT_PATH))}</p>
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

    start_ids = get_start_ids(sp, PROMPT)

    print(f"\nPrompt: {PROMPT}")
    print(f"Prompt token IDs: {start_ids}")
    print(f"Generating {NUM_SAMPLES_PER_TEMPERATURE} samples per temperature...")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Top-k: {TOP_K}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

    rows = []
    sample_counter = 0

    for temperature in TEMPERATURES:
        for i in range(NUM_SAMPLES_PER_TEMPERATURE):
            sample_counter += 1

            generated_ids = generate_ids(
                model=model,
                start_ids=start_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                block_size=config.block_size,
                temperature=temperature,
                top_k=TOP_K,
                device=device,
            )

            generated_text = sp.decode(generated_ids)

            raw_filename = f"sample_{sample_counter:03d}_temp_{temperature}.txt"
            svg_filename = f"sample_{sample_counter:03d}_temp_{temperature}.svg"

            raw_path = RAW_DIR / raw_filename
            svg_path = SVG_DIR / svg_filename

            raw_path.write_text(generated_text, encoding="utf-8")

            svg_text = extract_svg(generated_text)
            valid_xml = is_valid_xml(svg_text)

            svg_path.write_text(svg_text, encoding="utf-8")

            row = {
                "sample_id": sample_counter,
                "temperature": temperature,
                "top_k": TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS,
                "prompt": PROMPT,
                "num_tokens_generated": len(generated_ids),
                "num_chars_raw": len(generated_text),
                "num_chars_svg": len(svg_text),
                "valid_xml": valid_xml,
                "raw_filename": raw_filename,
                "svg_filename": svg_filename,
            }

            rows.append(row)

            print(
                f"Sample {sample_counter:03d} | "
                f"temp={temperature} | "
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

    print("\nGeneration complete.")
    print(f"Total samples: {total_count}")
    print(f"Valid XML SVG samples: {valid_count}/{total_count}")
    print(f"Raw text directory: {RAW_DIR}")
    print(f"SVG directory: {SVG_DIR}")
    print(f"Gallery HTML: {GALLERY_HTML_PATH}")


if __name__ == "__main__":
    main()