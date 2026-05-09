# SVG Transformer Scaling Study

This repository contains the code for a CS-GY 6923 optional project on SVG language modeling. The project trains GPT-style decoder-only Transformer models to generate SVG code using next-token prediction, studies how performance changes with model size, compares standard parameterization with μP, and evaluates SVG generation quality.

## Project Overview

SVG files are both images and text. Visually, they represent icons, emojis, symbols, and glyphs. Computationally, they are XML-like sequences made of tags, attributes, coordinates, and path commands. This project treats SVG generation as a language modeling task, where a Transformer predicts the next SVG token from previous tokens.

The project investigates three main questions:

1. Whether Transformer scaling trends appear on SVG language modeling
2. Whether μP improves learning-rate transfer from small to larger models
3. Whether a next-token SVG model can learn both SVG syntax and meaningful visual structure

## Dataset

The SVG corpus was built from three StarVector datasets from Hugging Face:

- `svg-icons-simple`
- `svg-emoji-simple`
- `svg-fonts-simple`

After cleaning, tokenization, and length filtering, the final trainable dataset contained:

- 572,701 SVG files
- 135,296,594 trainable tokens
- 132.6M training tokens
- 1.37M validation tokens
- 1.32M test tokens

The final cleaned corpus was dominated by the fonts subset, which influenced many generated outputs to resemble letters, numbers, and glyph-like shapes.

## Preprocessing

The preprocessing pipeline:

- Removes XML comments, metadata, title, and description tags
- Normalizes whitespace and newlines
- Rounds long decimal coordinates to one decimal place
- Filters very short and very long SVG files
- Validates XML syntax using `lxml`
- Splits the final dataset into train, validation, and test sets

SVGs longer than 1,024 tokens were removed to keep training computationally manageable.

## Tokenization

A SentencePiece BPE tokenizer was trained on 200,000 cleaned SVGs. SentencePiece was chosen because it produced more reusable SVG and code-like token patterns than the earlier ByteLevel BPE tokenizer.

Tokenizer details:

- Target vocabulary size: 4,096
- Actual learned vocabulary size: 2,498
- Special tokens: `<unk>`, `<bos>`, `<eos>`, `<pad>`

## Model Architecture

The models use a GPT-style decoder-only Transformer architecture with:

- Token embeddings
- Positional embeddings
- Causal self-attention blocks
- Feedforward layers
- Final vocabulary projection layer

The final architecture configurations are documented in:

```text
configs/model_configs.json
```

The standard model family used in the scaling study was:

| Model | Layers | Heads | Embedding Dim | Parameters |
|---|---:|---:|---:|---:|
| tiny | 4 | 4 | 128 | 1.47M |
| small | 4 | 4 | 256 | 4.51M |
| medium | 6 | 6 | 384 | 12.67M |
| large | 8 | 8 | 512 | 27.91M |
| xlarge | 10 | 8 | 512 | 34.22M |

The xlarge model was a scaled-down version of the suggested full XL configuration because of available GPU memory and runtime limits.

## Training Setup

All standard scaling models used the same training setup:

- Optimizer: AdamW
- Batch size: 32
- Block size: 256 tokens
- Tokens per step: 8,192
- Training budget: one token-equivalent epoch
- Learning-rate schedule: linear warmup followed by cosine decay
- Warmup fraction: 5%
- Minimum learning rate: 10% of peak learning rate

A learning-rate sweep was first performed on the smallest standard model. The selected peak learning rate was then reused for the larger standard models.

## μP Experiment

The project also compared standard Transformer parameterization with μP, or Maximal Update Parameterization. The μP implementation used:

- `MuReadout`
- `set_base_shapes`
- `MuAdam`
- Adjusted attention scaling
- A corrected learning-rate scheduler for MuAdam parameter groups

The μP learning-rate sweep selected a different peak learning rate from the standard setup, but the μP models still produced much higher validation losses than the standard models in this implementation.

## Results Summary

The standard Transformer scaling experiment showed that validation loss generally decreased as model size increased. The fitted scaling exponent was approximately:

```text
α = 0.075
```

The strongest gains appeared between the smaller models. The high-end scaling behavior should be interpreted cautiously because the xlarge model was only modestly larger than the large model.

The standard scaling law was also used to extrapolate to a hypothetical model with 10× more parameters than the largest trained model. This model was not actually trained. The predicted validation loss was approximately:

```text
0.9784
```

For final generation, the selected standard xlarge model was trained for 10 epochs. It achieved:

```text
Test loss: 1.0677
Test perplexity: 2.9088
```

Unconditional SVG generation produced:

```text
61 / 80 valid XML samples
XML validity rate: 76.25%
```

Prefix-conditioned generation was more difficult. Across two prefix-conditioned generation runs:

```text
40 / 60 valid XML completions
XML validity rate: 66.7%
```

Although many outputs were valid XML, visual coherence varied. The model learned SVG syntax more reliably than high-level visual completion.

## Repository Structure

```text
.
├── configs/
│   └── model_configs.json        # Model architecture configurations
│
├── scripts/                      # Preprocessing, model definitions, training, scaling, and generation scripts
│   ├── 12_model.py               # Standard GPT-style Transformer model definition
│   ├── 18_mup_model.py           # μP Transformer model definition
│   └── ...
│
├── report/
│   └── CS-GY_6923_Optional_Project.pdf
│
├── requirements.txt              # Python dependencies
└── README.md
```

Generated artifacts such as raw datasets, processed token files, model checkpoints, CSV summaries, plots, SVG galleries, virtual environments, and zipped Colab packages are not committed to this repository. They are generated locally or in Colab when the scripts are run.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the project scripts in order. The exact script names may vary depending on the final repository version, but the workflow is:

```bash
# 1. Prepare and clean SVG data
python scripts/<data_preprocessing_script>.py

# 2. Train SentencePiece tokenizer
python scripts/<tokenizer_training_script>.py

# 3. Tokenize dataset
python scripts/<tokenization_script>.py

# 4. Run standard scaling study
python scripts/<standard_scaling_script>.py

# 5. Run μP comparison
python scripts/<mup_scaling_script>.py

# 6. Train the final selected model
python scripts/<final_training_script>.py

# 7. Generate and evaluate SVG samples
python scripts/<generation_script>.py
python scripts/<prefix_completion_script>.py
```

Check the `scripts/` directory for the exact filenames.

## Requirements

The main dependencies are listed in `requirements.txt`. The project uses PyTorch for model training, SentencePiece for tokenization, `datasets` and `huggingface_hub` for data access, and standard Python data tools such as pandas, NumPy, matplotlib, and tqdm.

## Limitations

The main limitations are:

- The largest model was a scaled-down xlarge model rather than the full suggested XL configuration
- The final dataset was heavily dominated by font SVGs
- Full render-rate validation was limited by local dependency issues
- Generation quality was evaluated mainly with XML validity and visual inspection
- The μP implementation did not achieve reliable learning-rate transfer in this setup

Future work could train larger models for longer, use a more balanced SVG corpus, improve the μP implementation, and add stronger visual evaluation metrics such as render rate, structural validity checks, and image-based similarity measures.