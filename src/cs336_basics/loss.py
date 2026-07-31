"""
Cross-Entropy Loss implementation for Stanford CS336.
Uses numerically stable log-sum-exp trick.
"""

from __future__ import annotations

import torch


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross entropy loss using stable log-sum-exp.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size) or (N, vocab_size).
        targets: Tensor of shape (batch, seq_len) or (N,) containing class indices.
        ignore_index: Index to exclude from loss computation.

    Returns:
        Scalar loss tensor.
    """
    if logits.ndim == 3:
        logits = logits.view(-1, logits.size(-1))
    if targets.ndim == 2:
        targets = targets.view(-1)

    mask = targets != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    masked_logits = logits[mask]
    masked_targets = targets[mask]

    # Log-sum-exp trick for numerical stability
    max_logits = torch.max(masked_logits, dim=-1, keepdim=True).values
    stable_logits = masked_logits - max_logits
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(stable_logits), dim=-1))

    # Log probability of true target tokens
    target_logits = torch.gather(masked_logits, dim=-1, index=masked_targets.unsqueeze(-1)).squeeze(-1)
    log_probs = target_logits - log_sum_exp

    return -torch.mean(log_probs)
