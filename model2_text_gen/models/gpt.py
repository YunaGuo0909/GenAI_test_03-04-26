"""
Minimal GPT (Generative Pre-trained Transformer) for character-level text generation.

A from-scratch implementation of a decoder-only Transformer following
Radford et al. (2018/2019) and Karpathy's nanoGPT, designed for generating
structured species descriptions for the Bestiary of the Unreal project.

Architecture: Token Emb + Pos Emb -> N x [LayerNorm -> CausalSelfAttn -> LayerNorm -> FFN] -> LayerNorm -> Linear
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: [B, n_head, T, head_dim]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward with GELU activation."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Character-level GPT model.

    Takes a sequence of character indices and predicts the next character
    at each position. Uses learned token and position embeddings fed through
    a stack of Transformer decoder blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        print(f"GPT model: {sum(p.numel() for p in self.parameters()):,} parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: [B, T] integer token indices

        Returns:
            logits: [B, T, vocab_size] unnormalized predictions
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        pos = torch.arange(T, device=idx.device)
        tok = self.tok_emb(idx)
        x = self.drop(tok + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Auto-regressive generation.

        Args:
            idx: [B, T] conditioning token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: keep only top-k logits before sampling

        Returns:
            idx: [B, T + max_new_tokens] extended sequence
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)

        return idx
