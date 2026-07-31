"""
Unit tests for Dataset batching, Training loop, Checkpointing, and Perplexity (Stanford CS336 Assignment 1).
"""

import os
import pytest
import torch

from cs336_basics.dataset import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.train import (
    evaluate_perplexity,
    load_checkpoint,
    save_checkpoint,
    train_lm,
)


def test_get_batch_shapes():
    data = torch.arange(100)
    batch_size = 4
    context_length = 10

    x, y = get_batch(data, batch_size=batch_size, context_length=context_length)
    assert x.shape == (batch_size, context_length)
    assert y.shape == (batch_size, context_length)

    # Next-token target relation: y[i, j] should be x[i, j+1] if sequential
    # Or in general y is shifted by 1 relative to input in original data
    for i in range(batch_size):
        first_elem = x[i, 0].item()
        assert y[i, 0].item() == first_elem + 1


def test_checkpoint_save_and_load(tmp_path):
    model = TransformerLM(
        vocab_size=20,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_seq_len=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "model.pt"

    save_checkpoint(model, optimizer, scheduler=None, step=42, filepath=ckpt_path)
    assert os.path.exists(ckpt_path)

    new_model = TransformerLM(
        vocab_size=20,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_seq_len=32,
    )
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    saved_step = load_checkpoint(ckpt_path, new_model, new_optimizer)
    assert saved_step == 42

    # Model parameters should match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_mini_train_loop(tmp_path):
    vocab_size = 50
    data = torch.randint(0, vocab_size, (200,))

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_seq_len=32,
    )

    history = train_lm(
        model=model,
        train_data=data,
        val_data=data,
        batch_size=2,
        context_length=8,
        max_steps=10,
        eval_interval=5,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history["step"]) == 2
    assert "val_ppl" in history
    assert history["val_ppl"][-1] > 0.0
