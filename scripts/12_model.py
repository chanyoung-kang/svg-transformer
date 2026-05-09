"""
Define a decoder-only Transformer for SVG next-token prediction.

Create separate classes for each major building block.

This model takes token ID sequences as input and predicts the next token at each
position. It is similar to a small GPT-style language model:

    token IDs
        -> token embeddings
        -> positional embeddings
        -> stacked causal Transformer blocks
        -> vocabulary logits

Input shape:
    idx: [batch_size, block_size]

Output shape:
    logits: [batch_size, block_size, vocab_size]

The causal attention mask ensures that each position can only attend to previous
tokens and itself, not future tokens.
"""

from dataclasses import dataclass # module for making a simple class that mainly stores values

import torch
import torch.nn as nn               # neural network layers
import torch.nn.functional as F     # for functional operations like softmax and cross_entropy


# class for storing the model hyperparameters
@dataclass
class GPTConfig:
    vocab_size: int        # how many token types the model can predict
    block_size: int        # how many previous tokens the model can look at
    n_layer: int           # how many Transformer blocks to stack
    n_head: int            # how many attention heads per block
    n_embd: int            # embedding size (vector size for each token)
    dropout: float = 0.1   # regularization rate

# causal attention = masked self-attention
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0 # checks that the embedding dimension can split evenly into attention heads

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd) # creates the Query, Key, and Value vectors
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower triangular causal mask:
        # position i can attend only to positions <= i.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Shape:
        # [batch_size, seq_len, n_embd]
        # -> [batch_size, n_head, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Keep only the causal part of the attention matrix.
        att = att.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v

        # Shape:
        # [batch_size, n_head, seq_len, head_dim]
        # -> [batch_size, seq_len, n_embd]
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    """Position-wise MLP used inside each Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# combines the two core parts: attention + feedforward
class TransformerBlock(nn.Module):
    """
    One decoder-only Transformer block. 
    Each block contains two major parts:
        1. CausalSelfAttention
        2. FeedForward
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)     
        self.attn = CausalSelfAttention(config)     # look back mechanism

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)             # small neural network (FeedForward MLP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-layer normalization Transformer block
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

# full model
class GPT(nn.Module): # nn.Module means this is a PyTorch neural network model. Any layer inside it can have trainable weights.
    """Small GPT-style decoder-only Transformer."""

    # __init__ is the method that runs automatically when you create an object from a class
    # for GPT class __init__ defines the model layers
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config # config stores settings

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)       # embedding layer maps each token ID to a vector
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)    # add position information

        self.dropout = nn.Dropout(config.dropout) # Dropout randomly zeroes out some values during training to prevent overfitting

        # create n Transformer blocks - each block contains:
            # causal self-attention
            # feedforward MLP
            # layer norms ("Norm" part in “Add & Norm")
            # residual connections (“Add” part in “Add & Norm")
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd) # final layer normalization
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size) # lm_head converts each final token vector into scores over the vocabulary.

    # forward() defines how input goes through the model
    # In PyTorch, calling model(x, y) automatically runs model.forward(x, y).
    # So this method is the main computation path used for both:
    #   1. training:   logits, loss = model(x, y)
    #   2. generation: logits, _ = model(idx)
    def forward(
        self,
        idx: torch.Tensor, # input token IDs
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        batch_size, seq_len = idx.shape # read input shape

        if seq_len > self.config.block_size: # check sequence length
            raise ValueError(                # raise error if sequence length is bigger than the maximum context lenth model was created with
                f"Sequence length {seq_len} exceeds block_size "
                f"{self.config.block_size}"
            )

        token_emb = self.token_embedding(idx) # embedding: convert token IDs to token vectors
                                              # input shape: [batch_size, seq_len]
                                              # output shape: [batch_size, seq_len, n_embd]

        # create position embeddings
        positions = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb # add token + position embeddings
        x = self.dropout(x)     # apply dropout

        # pass through all transformer blocks 
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x) # final normalization 
        logits = self.lm_head(x) # convert vectors to vocabulary scores 
                                 # shape change: [batch_size, seq_len, n_embd] → [batch_size, seq_len, vocab_size] 

        loss = None

        if targets is not None: # targets is passed only during training
            loss = F.cross_entropy(     # cross entropy loss
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        
        # output
        # During training: logits, loss = model(x, y)
        # During generation: logits, _ = model(idx_cond)
        return logits, loss

    @torch.no_grad() # do not track gradients because we are not training here (only inference/generation)
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        idx:
            Starting token IDs with shape [batch_size, current_length].
        """
        self.eval() # setting eval mode turns off dropout 
                    # During generation, we want stable behavior, not random dropout.

        for _ in range(max_new_tokens): # Repeat generation loop
            
            idx_cond = idx[:, -self.config.block_size:] # Keep only the last block_size tokens as context

            logits, _ = self(idx_cond) # run model 
            logits = logits[:, -1, :] / temperature # take only the last position (generation only needs the next token after the current sequence)

            # Optional top-k filtering
            # If top_k=50, the model only samples from the top 50 most likely tokens.
            if top_k is not None: 
                values, _ = torch.topk(logits, top_k)
                cutoff = values[:, [-1]]
                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1) # Convert logits to probabilities
            next_id = torch.multinomial(probs, num_samples=1) # Sample next token according to the probabilities

            idx = torch.cat([idx, next_id], dim=1) # Append token

        return idx # Return generated sequence


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Small sanity check.

    This confirms that:
        1. The model can be created.
        2. The forward pass works.
        3. The loss can be computed.
        4. Generation runs without crashing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=2498,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )

    model = GPT(config).to(device)

    batch_size = 4

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.block_size),
        device=device,
    )

    y = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.block_size),
        device=device,
    )

    logits, loss = model(x, y)

    print(f"Device: {device}")
    print(f"Number of parameters: {count_parameters(model):,}")
    print(f"x shape: {x.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.4f}")

    start = torch.tensor([[1]], dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=20)

    print(f"Generated shape: {generated.shape}")
    print(f"Generated token IDs: {generated[0].tolist()}")


if __name__ == "__main__":
    main()