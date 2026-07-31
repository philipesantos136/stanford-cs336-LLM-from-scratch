"""
Unit tests for AdamW optimizer, LR scheduler, and gradient clipping (Stanford CS336 Assignment 1).
"""

import math
import pytest
import torch

from cs336_basics.optimizer import AdamW, CosineWarmupLRScheduler, clip_grad_norm_


def test_adamw_optimization():
    # Optimize quadratic function L(x) = (x - 3.0)^2
    x = torch.tensor([0.0], requires_grad=True)
    optimizer = AdamW([x], lr=1e-1, weight_decay=0.0)

    for _ in range(150):
        optimizer.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        optimizer.step()

    assert torch.allclose(x, torch.tensor([3.0]), atol=1e-2)



def test_gradient_clipping():
    p1 = torch.tensor([10.0], requires_grad=True)
    p2 = torch.tensor([20.0], requires_grad=True)

    p1.grad = torch.tensor([3.0])
    p2.grad = torch.tensor([4.0])
    # Norm = sqrt(9 + 16) = 5.0

    total_norm = clip_grad_norm_([p1, p2], max_norm=2.5)

    assert torch.allclose(total_norm, torch.tensor(5.0), atol=1e-4)
    # Scaled grad norm should now be 2.5 (grad elements halved)
    assert torch.allclose(p1.grad, torch.tensor([1.5]), atol=1e-4)
    assert torch.allclose(p2.grad, torch.tensor([2.0]), atol=1e-4)


def test_cosine_warmup_scheduler():
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = AdamW([x], lr=1.0)
    scheduler = CosineWarmupLRScheduler(optimizer, warmup_steps=10, total_steps=100, min_lr=0.1)

    # Step 0 (start of warmup)
    scheduler.step(0)
    assert optimizer.param_groups[0]["lr"] == 0.0

    # Step 5 (halfway warmup)
    scheduler.step(5)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.5, rel_tol=1e-5)

    # Step 10 (end of warmup)
    scheduler.step(10)
    assert math.isclose(optimizer.param_groups[0]["lr"], 1.0, rel_tol=1e-5)

    # Step 100 (end of total steps)
    scheduler.step(100)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.1, rel_tol=1e-5)
