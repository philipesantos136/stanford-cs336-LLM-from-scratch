"""
Training, Checkpointing, and Perplexity Evaluation pipeline for Stanford CS336.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cs336_basics.dataset import get_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, CosineWarmupLRScheduler, clip_grad_norm_


def evaluate_perplexity(
    model: nn.Module,
    val_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_iters: int = 10,
    device: str = "cpu",
) -> float:
    """
    Evaluate validation loss and calculate perplexity = exp(loss).
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(val_data, batch_size=batch_size, context_length=context_length, device=device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            total_loss += loss.item()

    model.train()
    mean_loss = total_loss / max(1, eval_iters)
    return math.exp(mean_loss)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    step: int,
    filepath: str | os.PathLike,
) -> None:
    """
    Save training checkpoint.
    """
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step": getattr(scheduler, "current_step", step) if scheduler else step,
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | os.PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> int:
    """
    Load training checkpoint and return saved step count.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_step" in checkpoint:
        scheduler.step(checkpoint["scheduler_step"])
    return checkpoint.get("step", 0)


def train_lm(
    model: TransformerLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    max_steps: int,
    lr: float = 5e-4,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    eval_interval: int = 50,
    checkpoint_dir: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run full language model training loop.
    """
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineWarmupLRScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_steps,
        min_lr=lr * 0.1,
    )

    history = {"step": [], "train_loss": [], "val_ppl": []}

    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, batch_size=batch_size, context_length=context_length, device=device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()

        if max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step(step)

        if step % eval_interval == 0 or step == max_steps:
            val_ppl = evaluate_perplexity(
                model=model,
                val_data=val_data,
                batch_size=batch_size,
                context_length=context_length,
                eval_iters=5,
                device=device,
            )
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_ppl"].append(val_ppl)

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(model, optimizer, scheduler, step, ckpt_path)

    return history
