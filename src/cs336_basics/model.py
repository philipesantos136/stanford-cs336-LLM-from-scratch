"""
Transformer Model Architecture implementation for Stanford CS336.
Includes RMSNorm, SwiGLU, Rotary Positional Embedding (RoPE), Causal Self-Attention,
Transformer Block, and TransformerLM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., d_model)
        Returns:
            Normalized tensor of same shape.
        """
        # Calculate RMS: sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.eps)
        return x * rsqrt * self.weight


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit Feed-Forward Network.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., d_model)
        Returns:
            Tensor of shape (..., d_model)
        """
        # SiLU(x * W_gate) * (x * W_up) -> then project down
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) using adjacent pair rotation.
    """

    def __init__(self, d_key: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        if d_key % 2 != 0:
            raise ValueError(f"d_key must be even for RoPE, got {d_key}")

        self.d_key = d_key
        self.max_seq_len = max_seq_len
        self.theta = theta

        # theta_i = theta^(-2i / d_key)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_key, 2).float() / d_key))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_adjacent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates adjacent pairs of input x: [x0, x1] -> [-x1, x0].
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., seq_len, d_key)
            token_positions: Optional position IDs tensor of shape (..., seq_len) or seq_len int
        Returns:
            RoPE-transformed tensor of same shape.
        """
        seq_len = x.size(-2)
        if token_positions is None or isinstance(token_positions, int):
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        else:
            freqs = torch.einsum("...i,j->...ij", token_positions.float(), self.inv_freq)

        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)


        # Match dimensions with x if x has extra leading/middle dims (e.g. heads)
        while cos.ndim < x.ndim:
            if cos.ndim == x.ndim - 1:
                # Insert head dimension if x is (batch, num_heads, seq_len, d_key)
                cos = cos.unsqueeze(-3)
                sin = sin.unsqueeze(-3)
            else:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)

        return (x * cos) + (self._rotate_adjacent(x) * sin)



class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with optional RoPE.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute causal mask
        causal_mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape of Q, K, V: (batch, num_heads, seq_len, head_dim)

        # Apply RoPE if present
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)


        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Reshape and project out
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    """
    Single Transformer Block with Pre-LN, Causal Self-Attention, and SwiGLU FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope=rope,
        )
        self.ln_2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """
    Complete Decoder-Only Transformer Language Model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Shared RoPE positional embedding
        self.rope = RotaryPositionalEmbedding(
            d_key=d_model // num_heads,
            max_seq_len=max_seq_len,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                rope=self.rope,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch, seq_len) containing token indices.
        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")

        x = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits
