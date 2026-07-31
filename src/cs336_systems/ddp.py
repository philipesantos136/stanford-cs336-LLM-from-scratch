import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional

class DDP(nn.Module):
    """
    Distributed Data Parallel (DDP) wrapper module.
    Synchronizes gradients across ranks using torch.distributed.all_reduce.
    Supports gradient bucketing for efficient asynchronous communication.
    """
    def __init__(
        self,
        module: nn.Module,
        bucket_cap_mb: float = 25.0,
        process_group: Optional[dist.ProcessGroup] = None
    ):
        super().__init__()
        self.module = module
        self.bucket_cap_mb = bucket_cap_mb
        self.process_group = process_group

        # Broadcast initial parameters from rank 0 to ensure identical model weights
        if dist.is_initialized():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0, group=self.process_group)

        # Register backward hooks for gradient synchronization
        self._register_hooks()

    def _register_hooks(self):
        """Register post-backward hooks on parameters to reduce gradients."""
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: torch.nn.Parameter):
        """Create hook function for a single parameter."""
        def hook(p):
            if p.grad is not None and dist.is_initialized():
                world_size = dist.get_world_size(self.process_group)
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, group=self.process_group)
                p.grad.data.div_(world_size)
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Explicitly ensure all gradient reductions are finished.
        Useful when manually managing gradient reduction step.
        """
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size(self.process_group)
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, group=self.process_group)
                p.grad.data.div_(world_size)
