"""
Dataset and Batching utilities for language model training (Stanford CS336).
"""

from __future__ import annotations

import torch


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of inputs (x) and next-token targets (y) from a 1D token ID tensor.

    Args:
        data: 1D Tensor of token IDs.
        batch_size: Number of sequences in batch.
        context_length: Sequence context window size.
        device: Target device ("cpu" or "cuda").

    Returns:
        Tuple of (x, y) tensors of shape (batch_size, context_length).
    """
    if data.ndim != 1:
        raise ValueError(f"data tensor must be 1D, got shape {data.shape}")
    if len(data) <= context_length:
        raise ValueError(f"Dataset length ({len(data)}) must be > context_length ({context_length})")

    max_idx = len(data) - context_length - 1
    ix = torch.randint(0, max_idx + 1, (batch_size,))

    x = torch.stack([data[i : i + context_length] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix]).to(device)

    return x, y
