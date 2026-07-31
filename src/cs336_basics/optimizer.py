"""
Custom AdamW Optimizer, Learning Rate Scheduler, and Gradient Clipping for Stanford CS336.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Tuple, Union

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Decoupled Weight Decay AdamW Optimizer.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                # 1. Apply decoupled weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 2. Update biased first & second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Compute bias-corrected moment estimates
                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 4. Update parameter tensor
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clips gradient norm of an iterable of parameters.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return torch.tensor(0.0)

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == 2.0:
        total_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
            norm_type,
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)

    return total_norm


class CosineWarmupLRScheduler:
    """
    Cosine Learning Rate Scheduler with Linear Warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self, current_step: int | None = None) -> None:
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        step = self.current_step

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (step / max(1, self.warmup_steps))
            elif step >= self.total_steps:
                lr = self.min_lr
            else:
                # Cosine decay
                decay_ratio = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                lr = self.min_lr + coeff * (base_lr - self.min_lr)

            group["lr"] = lr
