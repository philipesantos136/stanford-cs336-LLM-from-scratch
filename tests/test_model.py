"""
Unit tests for Transformer Model components (Stanford CS336 Assignment 1).
"""

import pytest
import torch

from cs336_basics.model import (
    RMSNorm,
    SwiGLU,
    RotaryPositionalEmbedding,
    CausalSelfAttention,
    TransformerBlock,
    TransformerLM,
)


def test_rmsnorm_shape_and_value():
    batch, seq_len, d_model = 2, 8, 64
    x = torch.randn(batch, seq_len, d_model)
    norm = RMSNorm(d_model)
    out = norm(x)

    assert out.shape == x.shape
    # Check RMS of normalized tensor before gain weight scaling (weight is ones initially)
    rms = torch.sqrt(out.pow(2).mean(-1) + 1e-5)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_swiglu_shape():
    batch, seq_len, d_model, d_ff = 2, 8, 64, 256
    x = torch.randn(batch, seq_len, d_model)
    ffn = SwiGLU(d_model, d_ff)
    out = ffn(x)
    assert out.shape == (batch, seq_len, d_model)


def test_rope_shape_and_rotation():
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
    x = torch.randn(batch, num_heads, seq_len, head_dim)
    rope = RotaryPositionalEmbedding(d_key=head_dim, max_seq_len=64)
    out = rope(x, seq_len)

    assert out.shape == x.shape
    # Vector magnitude across head_dim should be preserved under rotation
    norm_x = torch.norm(x, dim=-1)
    norm_out = torch.norm(out, dim=-1)
    assert torch.allclose(norm_x, norm_out, atol=1e-4)


def test_causal_self_attention_masking():
    batch, seq_len, d_model, num_heads = 1, 4, 32, 2
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=16)

    out = attn(x)
    assert out.shape == (batch, seq_len, d_model)

    # Check causal property: output at position 1 should NOT depend on input at position 2 or 3
    loss = out[0, 1].sum()
    loss.backward()

    # Gradient with respect to x[0, 2] and x[0, 3] should be 0
    assert torch.allclose(x.grad[0, 2], torch.zeros_like(x.grad[0, 2]), atol=1e-7)
    assert torch.allclose(x.grad[0, 3], torch.zeros_like(x.grad[0, 3]), atol=1e-7)


def test_transformer_lm_forward_pass():
    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128
    batch_size = 2
    seq_len = 16

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=64,
        tie_weights=True,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert model.lm_head.weight is model.token_embeddings.weight
