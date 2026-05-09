"""
Local smoke-test training script for one small decoder-only Transformer.

This script checks that the full training pipeline works locally:
    1. Load encoded train/validation/test token streams.
    2. Create next-token prediction batches.
    3. Create a small GPT-style model.
    4. Run a few training iterations.
    5. Estimate train/validation loss.
    6. Save a small checkpoint.

This is only a local sanity check. The real Part 2 training should run on
Colab/GPU with larger settings and longer training.
"""

from pathlib import Path
import time

import sentencepiece as spm # loads tokenizer only to get vocab size
import torch

# These imports work because dataset_loader.py and model.py are in scripts/
# and this file is also being run from scripts/.
from dataset_loader import load_all_splits, get_batch
from model import GPTConfig, GPT, count_parameters


TOKENIZER_MODEL_PATH = Path("data/tokenizer/svg_bpe.model")
CHECKPOINT_DIR = Path("outputs/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "local_smoke_test_model.pt"


# Local CPU-friendly training settings (how we train)
BATCH_SIZE = 4              # each batch contains 4 examples/chunks (train on 4 chunks at once)

BLOCK_SIZE = 128            # each chunk has 128 tokens

MAX_ITERS = 5               # train for only 5 update steps (update model weights only 5 times)
                            # each iteration:
                            #     sample one batch
                            #     run model
                            #     compute train loss on that batch
                            #     backpropagate
                            #     update weights once
                            # iteration 1: sample batch #1 with 4 chunks → update weights once
                            # iteration 2: sample batch #2 with 4 chunks → update weights once
                            # iteration 3: sample batch #3 with 4 chunks → update weights once
                            # iteration 4: sample batch #4 with 4 chunks → update weights once
                            # iteration 5: sample batch #5 with 4 chunks → update weights once

EVAL_INTERVAL = 1           # evaluate every step
EVAL_ITERS = 2              # use 2 random batches to estimate train/val loss 
LEARNING_RATE = 3e-4        # optimizer step size


# Small local smoke-test model (what model we train)
N_LAYER = 2         # stack 2 Transformer blocks
N_HEAD = 4          # each attention layer has 4 heads
N_EMBD = 128        # each token is represented by a 128-dimensional vector
DROPOUT = 0.1       # randomly drop 10% of values during training for regularization


@torch.no_grad() # Do not calculate gradients. Do not update weights. Only evaluate.
def estimate_loss(model, data, device):
    """
    Estimate train and validation loss using a few random batches.

    This does not update the model. It only measures current next-token
    prediction performance.
    """
    model.eval() # turns off training-specific behavior

    losses = {}

    for split in ["train", "val"]:      # train loss = how well the model fits examples from the training data
                                        # val loss   = how well the model performs on held-out examples it was not trained on
        split_losses = []

        # sample a few batches
        for _ in range(EVAL_ITERS):
            x, y = get_batch(
                tokens=data[split],
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE,
                device=device,
            )
            
            # Compute loss without updating weights
            # because of @torch.no_grad(), this does not train the model
            _, loss = model(x, y)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses) # Average the losses

    # After evaluation, we switch the model back to training mode so dropout is active again.
    model.train()
    return losses


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer only to get vocab size.
    sp = spm.SentencePieceProcessor() 
    sp.load(str(TOKENIZER_MODEL_PATH)) # Important: we do not use the tokenizer to tokenize data anymore. The data is already encoded in .bin files.
    vocab_size = sp.get_piece_size()

    print(f"Tokenizer vocab size: {vocab_size:,}")

    # Load encoded token streams.
    print("Loading encoded token data...")
    data = load_all_splits()

    print(f"Train token IDs: {len(data['train']):,}")
    print(f"Val token IDs: {len(data['val']):,}")
    print(f"Test token IDs: {len(data['test']):,}")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )

    model = GPT(config).to(device) # creates the actual Transformer model and moves it to CPU/GPU

    print(f"Model parameters: {count_parameters(model):,}") # prints how many trainable weights the model has
    """
    Token embedding = vocab_size x n_embd
        - vocab_size: number of unique token types tokenizer vocabulary has
        - n_embd: embedding dimension (vector size for each token)
        - 2498 x 128 = 319,744

    Position embedding = block_size x n_embd
        - block_size: number of tokens each chunk has
        - n_embd: embedding dimension (vector size for each token)
        - 128 x 128 = 16,384

    Transformer blocks: 198,272 per block (two blocks = 198272 * 2)

        QKV projection = n_embd x n_embd x 3 
            - n_embd: embedding dimension (vector size for each token)
            - Wq, Wk, Wv each have shape (n_embd, n_embd)
            - 128 x 128 x 3 + 128 x 3 (bias) = 49,536

        Attention output = n_embd x n_embd
            - n_embd: embedding dimension (vector size for each token)
            - 128 x 128 + 128= 16,512

        Feedforward layer 1 = n_embd x 4 * n_embd
            - n_embd: embedding dimension (vector size for each token)
            - each arrow from one input node to one output node is one weight parameter
            - 128 x 512 + 512 (bias) = 66,048

        Feedforward layer 2 = 4 * n_embd x n_embd
            - n_embd: embedding dimension (vector size for each token)
            - 512 x 128 + 128 = 65,664

        LayerNorms: 1 x n_embd x 2 x 2 
            - n_embd: embedding dimension (vector size for each token)
            - each LayerNorm has n_embd values for weight (gamma) and bias (bias)
            - there are 2 LayerNorms vectors
            - (128 + 128) * 2 = 512

    Final LayerNorm: 1 x n_embd x 2
        - n_embd: embedding dimension (vector size for each token)
        - 128 + 128 = 256
    
    Output head: n_embd x vocab_size
        - n_embd: embedding dimension (vector size for each token)
        - vocab_size: number of unique token types tokenizer vocabulary has
        - final layer that turns the model's internal vector into next-token scores
            - turns each 128-dimensional token vector into 2,498 next-token scores
        - 128 x 2498 = 319,744

    Total: 1,055,170
    """

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    start_time = time.time()

    # training loop - trains the model one batch at a time
    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(
            tokens=data["train"],
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
        )
        
        # forward pass 
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True) # clear old gradients
        loss.backward() # backpropagation
        optimizer.step() # update weights

        if step == 1 or step % EVAL_INTERVAL == 0: # Evaluate periodically 
            losses = estimate_loss(model, data, device)

            elapsed = time.time() - start_time

            print(
                f"step {step:5d} | "
                f"batch loss {loss.item():.4f} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"elapsed {elapsed:.1f} sec"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size,
        "train_settings": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "max_iters": MAX_ITERS,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "learning_rate": LEARNING_RATE,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "dropout": DROPOUT,
        },
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nSaved checkpoint to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()