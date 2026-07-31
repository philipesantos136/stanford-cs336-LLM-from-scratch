"""
Unit tests for Cross-Entropy Loss (Stanford CS336 Assignment 1).
"""

import pytest
import torch
import torch.nn.functional as F
from cs336_basics.loss import cross_entropy_loss


def test_cross_entropy_matches_pytorch():
    batch, seq_len, vocab_size = 4, 16, 50
    logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch, seq_len))

    custom_loss = cross_entropy_loss(logits, targets)
    torch_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    assert torch.allclose(custom_loss, torch_loss, atol=1e-5)


def test_cross_entropy_ignore_index():
    batch, vocab_size = 2, 10
    logits = torch.randn(batch, vocab_size)
    targets = torch.tensor([3, -100])

    custom_loss = cross_entropy_loss(logits, targets, ignore_index=-100)
    torch_loss = F.cross_entropy(logits, targets, ignore_index=-100)

    assert torch.allclose(custom_loss, torch_loss, atol=1e-5)
