"""
μP-compatible GPT-style decoder-only Transformer model.

This file defines a μP version of the SVG Transformer used in Part 2.

Main differences from the standard model:
    1. Uses mup.MuReadout for the final vocabulary projection.
    2. Uses MuReadout(..., readout_zero_init=True).
    3. Uses μP Transformer attention scaling: 8 / head_dim instead of 1 / sqrt(head_dim).
    4. Applies mup.set_base_shapes before optimizer creation.
    5. Zero-initializes the query projection after set_base_shapes.

This file only defines and sanity-checks the μP model.
It does not train on the real SVG dataset yet.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mup import MuReadout, set_base_shapes


@dataclass
class GPTConfig:
    """Class for storing model hyperparameters."""

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer with μP-compatible attention scaling."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Combined QKV projection.
        # Output dimension is 3 * n_embd:
        #   first n_embd rows  -> Q
        #   second n_embd rows -> K
        #   third n_embd rows  -> V
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Standard Transformer scaling:
        #     attention_scores / sqrt(head_dim)
        #
        # μP Transformer scaling:
        #     attention_scores * 8 / head_dim
        #
        # This is proportional to 1 / d and follows the mup README's
        # Transformer recommendation.
        att = (q @ k.transpose(-2, -1)) * (8.0 / self.head_dim)

        att = att.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.n_embd,
        )

        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class FeedForward(nn.Module):
    """Feedforward layer used inside each Transformer block."""

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


class TransformerBlock(nn.Module):
    """One decoder-only Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))

        return x


class GPT(nn.Module):
    """μP-compatible GPT-style decoder-only Transformer."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        # μP change:
        # Use MuReadout instead of nn.Linear for the final output projection.
        #
        # Transformer-specific μP initialization:
        # readout_zero_init=True zero-initializes the readout weights.
        self.lm_head = MuReadout(
            config.n_embd,
            config.vocab_size,
            bias=False,
            readout_zero_init=True,
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size "
                f"{self.config.block_size}"
            )

        token_emb = self.token_embedding(idx)

        positions = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                cutoff = values[:, [-1]]
                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

        return idx


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_base_delta_configs(target_config: GPTConfig) -> tuple[GPTConfig, GPTConfig]:
    """
    Create base and delta configs for mup.set_base_shapes.

    μP needs:
        base model: smaller width
        delta model: wider than base model

    We keep vocab size, block size, layer count, and head count the same as the
    target model. We only change n_embd.

    The base/delta embedding sizes are chosen to be divisible by n_head.
    """
    base_n_embd = target_config.n_head * 32
    delta_n_embd = target_config.n_head * 64

    base_config = GPTConfig(
        vocab_size=target_config.vocab_size,
        block_size=target_config.block_size,
        n_layer=target_config.n_layer,
        n_head=target_config.n_head,
        n_embd=base_n_embd,
        dropout=target_config.dropout,
    )

    delta_config = GPTConfig(
        vocab_size=target_config.vocab_size,
        block_size=target_config.block_size,
        n_layer=target_config.n_layer,
        n_head=target_config.n_head,
        n_embd=delta_n_embd,
        dropout=target_config.dropout,
    )

    return base_config, delta_config


def apply_mup_transformer_initialization(model: GPT) -> GPT:
    """
    Apply Transformer-specific μP initialization recommended in the mup README.

    The README recommends:
        1. zero-initializing MuReadout weights via readout_zero_init=True
        2. zero-initializing the query matrix manually

    Since this model uses one combined qkv projection, the first n_embd output
    rows correspond to the query projection.
    """
    with torch.no_grad():
        for block in model.blocks:
            qkv_proj = block.attn.qkv_proj
            n_embd = block.attn.n_embd

            # Zero-initialize query projection weights.
            qkv_proj.weight[:n_embd].zero_()

            # Zero-initialize query projection bias if present.
            if qkv_proj.bias is not None:
                qkv_proj.bias[:n_embd].zero_()

    return model


def apply_mup_base_shapes(model: GPT, target_config: GPTConfig) -> GPT:
    """
    Apply μP base shapes and Transformer-specific μP initialization.

    This should be called before creating the μP optimizer.

    Important order:
        1. Create target model.
        2. Apply set_base_shapes.
        3. Apply Transformer-specific μP initialization.
        4. Create MuAdam optimizer.
    """
    base_config, delta_config = make_base_delta_configs(target_config)

    base_model = GPT(base_config)
    delta_model = GPT(delta_config)

    set_base_shapes(
        model,
        base_model,
        delta=delta_model,
    )

    model = apply_mup_transformer_initialization(model)

    return model


def main():
    """
    Small sanity check.

    This confirms that:
        1. μP model can be created.
        2. μP base shapes can be applied.
        3. Transformer-specific μP initialization is applied.
        4. Forward pass works.
        5. Loss can be computed.
        6. Generation runs without crashing.
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

    model = GPT(config)
    model = apply_mup_base_shapes(model, config)
    model = model.to(device)

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

    # Check that μP infshape metadata exists on at least some parameters.
    has_infshape = any(hasattr(p, "infshape") for p in model.parameters())
    print(f"Has μP infshape metadata: {has_infshape}")

    # Check that query projection is zero-initialized.
    first_block = model.blocks[0]
    q_weight = first_block.attn.qkv_proj.weight[: config.n_embd]
    q_abs_sum = q_weight.abs().sum().item()
    print(f"Query projection abs sum after init: {q_abs_sum:.6f}")

    # Check that readout is zero-initialized.
    readout_abs_sum = model.lm_head.weight.abs().sum().item()
    print(f"Readout abs sum after init: {readout_abs_sum:.6f}")


if __name__ == "__main__":
    main()