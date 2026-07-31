"""
Adapter functions connecting Stanford CS336 official test suite to cs336_basics implementation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.dataset import get_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.model import (
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
)
from cs336_basics.optimizer import AdamW, clip_grad_norm_
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train import load_checkpoint, save_checkpoint


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    return F.linear(in_features, weights)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    return F.embedding(token_ids, weights)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # SwiGLU: (silu(x @ W1^T) * (x @ W3^T)) @ W2^T
    gate = F.silu(F.linear(in_features, w1_weight))
    up = F.linear(in_features, w3_weight)
    return F.linear(gate * up, w2_weight)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    batch_shape = in_features.shape[:-2]
    seq_len = in_features.size(-2)
    head_dim = d_model // num_heads

    # Flatten leading batch dimensions if any
    x = in_features.view(-1, seq_len, d_model)
    batch_size = x.size(0)

    q = F.linear(x, q_proj_weight).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = F.linear(x, k_proj_weight).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = F.linear(x, v_proj_weight).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)

    out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    res = F.linear(out, o_proj_weight)

    return res.view(*batch_shape, seq_len, d_model)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    batch_shape = in_features.shape[:-2]
    seq_len = in_features.size(-2)
    head_dim = d_model // num_heads

    x = in_features.view(-1, seq_len, d_model)
    batch_size = x.size(0)

    q = F.linear(x, q_proj_weight).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = F.linear(x, k_proj_weight).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = F.linear(x, v_proj_weight).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    rope = RotaryPositionalEmbedding(d_key=head_dim, max_seq_len=max_seq_len, theta=theta)
    q = rope(q, seq_len)
    k = rope(k, seq_len)

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)

    out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    res = F.linear(out, o_proj_weight)

    return res.view(*batch_shape, seq_len, d_model)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = RotaryPositionalEmbedding(d_key=d_k, max_seq_len=max_seq_len, theta=theta)
    seq_len = in_query_or_key.size(-2)
    
    # Handle optional extra dimensions for heads / batches
    if in_query_or_key.ndim == 4:
        return rope(in_query_or_key, seq_len)
    elif in_query_or_key.ndim == 3:
        # Add head dim: (batch, seq_len, d_k) -> (batch, 1, seq_len, d_k)
        x_4d = in_query_or_key.unsqueeze(1)
        res = rope(x_4d, seq_len)
        return res.squeeze(1)
    else:
        return rope(in_query_or_key.unsqueeze(0).unsqueeze(0), seq_len).squeeze(0).squeeze(0)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    norm = RMSNorm(d_model, eps=eps)
    norm.weight.data = weights
    return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return F.silu(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    rope = RotaryPositionalEmbedding(d_key=d_model // num_heads, max_seq_len=max_seq_len, theta=theta)
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, rope=rope)

    # Load weights
    block.ln_1.weight.data = weights["ln1.weight"]
    block.attn.q_proj.weight.data = weights["attn.q_proj.weight"]
    block.attn.k_proj.weight.data = weights["attn.k_proj.weight"]
    block.attn.v_proj.weight.data = weights["attn.v_proj.weight"]
    block.attn.out_proj.weight.data = weights["attn.output_proj.weight"]
    block.ln_2.weight.data = weights["ln2.weight"]
    block.ffn.w_gate.weight.data = weights["ffn.w1.weight"]
    block.ffn.w_down.weight.data = weights["ffn.w2.weight"]
    block.ffn.w_up.weight.data = weights["ffn.w3.weight"]

    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
    )
    model.rope = RotaryPositionalEmbedding(d_key=d_model // num_heads, max_seq_len=context_length, theta=rope_theta)

    # Load state dict mapping keys
    model.token_embeddings.weight.data = weights["token_embeddings.weight"]
    model.ln_final.weight.data = weights["ln_final.weight"]
    model.lm_head.weight.data = weights["lm_head.weight"]

    for i, block in enumerate(model.blocks):
        block.rope = model.rope
        block.ln_1.weight.data = weights[f"layers.{i}.ln1.weight"]
        block.attn.q_proj.weight.data = weights[f"layers.{i}.attn.q_proj.weight"]
        block.attn.k_proj.weight.data = weights[f"layers.{i}.attn.k_proj.weight"]
        block.attn.v_proj.weight.data = weights[f"layers.{i}.attn.v_proj.weight"]
        block.attn.out_proj.weight.data = weights[f"layers.{i}.attn.output_proj.weight"]
        block.ln_2.weight.data = weights[f"layers.{i}.ln2.weight"]
        block.ffn.w_gate.weight.data = weights[f"layers.{i}.ffn.w1.weight"]
        block.ffn.w_down.weight.data = weights[f"layers.{i}.ffn.w2.weight"]
        block.ffn.w_up.weight.data = weights[f"layers.{i}.ffn.w3.weight"]

    return model(in_indices)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data_tensor = torch.tensor(dataset, dtype=torch.long)
    return get_batch(data_tensor, batch_size=batch_size, context_length=context_length, device=device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return F.softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    clip_grad_norm_(parameters, max_norm=max_l2_norm)


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * (it / max(1, warmup_iters))
    elif it >= cosine_cycle_iters:
        return min_learning_rate
    else:
        decay_ratio = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, scheduler=None, step=iteration, filepath=out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    special_token_map = {}
    bytes_to_id = {v: k for k, v in vocab.items()}
    if special_tokens:
        next_id = max(vocab.keys()) + 1 if vocab else 256
        for tok in special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes in bytes_to_id:
                special_token_map[tok] = bytes_to_id[tok_bytes]
            else:
                special_token_map[tok] = next_id
                next_id += 1
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_token_map)



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    tokenizer = BPETokenizer.train(text=text, vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer.vocab, tokenizer.merges
