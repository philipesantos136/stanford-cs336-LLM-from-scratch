import time
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Tuple, Optional

def benchmark_latency_and_throughput(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 5,
    num_steps: int = 20,
    forward_only: bool = False
) -> Dict[str, float]:
    """
    Benchmarks model latency (ms) and throughput (tokens/sec).
    """
    model.eval() if forward_only else model.train()
    
    # Warmup steps
    for _ in range(num_warmup):
        out = model(input_tensor)
        if not forward_only:
            loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
            loss.backward()

    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_steps):
        out = model(input_tensor)
        if not forward_only:
            loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
            loss.backward()

    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    
    total_time_sec = end_time - start_time
    avg_latency_ms = (total_time_sec / num_steps) * 1000.0
    
    # Tokens per batch = batch_size * sequence_length (if 2D or 3D tensor)
    tokens_per_step = input_tensor.numel() // input_tensor.shape[-1] * input_tensor.shape[-1] if input_tensor.ndim >= 2 else input_tensor.numel()
    throughput_tokens_sec = (tokens_per_step * num_steps) / total_time_sec

    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_tokens_per_sec": throughput_tokens_sec,
        "total_time_sec": total_time_sec,
        "num_steps": num_steps,
    }


def profile_execution(
    model: nn.Module,
    input_tensor: torch.Tensor,
    export_trace_path: Optional[str] = None
) -> str:
    """
    Profiles model execution using PyTorch Profiler.
    Returns string summary table.
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ] + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        out = model(input_tensor)
        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
        loss.backward()

    if export_trace_path:
        prof.export_chrome_trace(export_trace_path)

    return prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
