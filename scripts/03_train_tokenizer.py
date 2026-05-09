"""
Train a SentencePiece BPE tokenizer on cleaned SVG text.

This script samples cleaned SVGs, writes them to a plain-text corpus file, and
trains a SentencePiece BPE tokenizer. The tokenizer is later used to convert SVG
strings into integer token IDs.

Input:
    data/processed/svg_combined_clean.parquet

Outputs:
    data/tokenizer/svg_corpus.txt
    data/tokenizer/svg_bpe.model
    data/tokenizer/svg_bpe.vocab

Special token IDs:
    <unk> = 0
    <bos> = 1
    <eos> = 2
    <pad> = 3
"""

from pathlib import Path

import pandas as pd
# from tokenizers import ByteLevelBPETokenizer 
import sentencepiece as spm


# input path
DATA_PATH = Path("data/processed/svg_combined_clean.parquet")
# where we save tokenizer files
TOKENIZER_DIR = Path("data/tokenizer")
# temporary text file
CORPUS_PATH = TOKENIZER_DIR / "svg_corpus.txt"

# SentencePiece API works differently and needs these paths
MODEL_PREFIX = TOKENIZER_DIR / "svg_bpe"
MODEL_PATH = TOKENIZER_DIR / "svg_bpe.model"
VOCAB_PATH = TOKENIZER_DIR / "svg_bpe.vocab"

# Target vocabulary size
VOCAB_SIZE = 4096

MAX_TOKENIZER_TRAIN_ROWS = 200_000

def main():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Loads the cleaned Parquet file into a DataFrame and take the svg column and convert it into a list
    df = pd.read_parquet(DATA_PATH)

    if len(df) > MAX_TOKENIZER_TRAIN_ROWS:
        train_df = df.sample(n=MAX_TOKENIZER_TRAIN_ROWS, random_state=42)
    else:
        train_df = df
        
    texts = train_df["svg"].tolist()

    # create tokenizer training corpus - write all SVGs into a plain text file
    # Save one SVG per line as tokenizer training corpus (newline acts like a separator)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for svg in texts:
            f.write(svg.replace("\n", " ") + "\n")

    print(f"Cleaned SVG rows available: {len(df):,}")
    print(f"Tokenizer training rows used: {len(texts):,}")

    # # This creates an untrained byte-level BPE tokenizer.
    # # It does not know the SVG corpus yet. At this point, it is just an empty tokenizer object.
    # tokenizer = ByteLevelBPETokenizer()

    # # train the tokenizer
    # """
    # Special tokens are tokens that do not come from the original SVG text naturally, but we add them so the model/tokenizer can handle special situations.
    #     <pad> = padding token, used if sequences need to be padded
    #     <unk> = unknown token
    #     <bos> = beginning of SVG sequence
    #     <eos> = end of SVG sequence
    # """
    # tokenizer.train(
    #     files=[str(CORPUS_PATH)],
    #     vocab_size=VOCAB_SIZE, # Try to learn up to 4096 unique token pieces
    #     min_frequency=1, # create a new merged token if pattern appears at least 2 times in the training corpus
    #     special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    # )

    # tokenizer.save_model(str(TOKENIZER_DIR))


    # train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_PATH),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0, # Keep coverage for 100% of characters seen in the training text (For English text, people sometimes use slightly lower values. But for SVG/code, we want to preserve all characters because symbols matter)
        hard_vocab_limit=False, # Try to make the vocabulary size 4096, but do not crash if it cannot hit that number exactly.

        # Reserve standard special token IDs.
        # these names are special keywords/parameters defined by SentencePiece.
        # unk_id, bos_id, eos_id, pad_id, unk_piece, bos_piece, eos_piece, and pad_piece are SentencePiece configuration parameter names.
        # we are predefining the IDs for the first four special tokens in the SentencePiece vocabulary. Then the rest of the vocabulary starts after those.
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,

        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        pad_piece="<pad>",
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))

    # print(f"Trained tokenizer on {len(texts):,} SVGs")
    print(f"Trained SentencePiece BPE tokenizer on {len(texts):,} SVGs")
    # print(f"Saved vocab/merges to: {TOKENIZER_DIR}")

    print(f"Target vocab size: {VOCAB_SIZE:,}")
    # print(f"Actual vocab size: {tokenizer.get_vocab_size():,}")
    print(f"Actual vocab size: {sp.get_piece_size():,}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vocab to: {VOCAB_PATH}")

    # sanity check
    sample = texts[0]
    # encoded = tokenizer.encode(sample) # SVG converted into token IDs
    # decoded = tokenizer.decode(encoded.ids) # converts the token IDs back into text
    encoded_ids = sp.encode(sample, out_type=int)  # SVG converted into token IDs
    decoded = sp.decode(encoded_ids)               # converts token IDs back into text


    print("\nSample SVG preview:")
    print(sample[:300])

    print("\nFirst 30 token IDs:")
    print(encoded_ids[:30])

    print("\nDecoded preview:")
    print(decoded[:300])

    print("\nOriginal starts with:")
    print(sample[:100])

    print("\nDecoded starts with:")
    print(decoded[:100])


if __name__ == "__main__":
    main()