import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from typing import Iterable, Dict, Any, List, Optional

class ShardedOptimizer(optim.Optimizer):
    """
    Sharded Optimizer (ZeRO-1 style / ZeroRedundancyOptimizer).
    Shards optimizer state (e.g. AdamW momentum & variance tensors) across
    distributed ranks, reducing optimizer state memory overhead by factor of W (world_size).
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls=optim.AdamW,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs
    ):
        params_list = list(params)
        self.process_group = process_group
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size(self.process_group)
            self.rank = dist.get_rank(self.process_group)
        else:
            self.world_size = 1
            self.rank = 0

        # Partition parameters among ranks in round-robin fashion
        self.all_params = params_list
        self.sharded_params = [
            p for idx, p in enumerate(self.all_params)
            if idx % self.world_size == self.rank
        ]

        # Initialize underlying standard optimizer for assigned slice of parameters
        defaults = kwargs
        super().__init__(self.sharded_params, defaults)
        self.optim_cls = optimizer_cls
        self.inner_optimizer = optimizer_cls(self.sharded_params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Perform local optimizer step on assigned sharded parameters
        self.inner_optimizer.step()

        # Synchronize updated parameters across all ranks via all_gather
        if dist.is_initialized() and self.world_size > 1:
            for p in self.all_params:
                dist.broadcast(p.data, src=p_rank(p, self.all_params, self.world_size), group=self.process_group)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


def p_rank(param: torch.nn.Parameter, all_params: List[torch.nn.Parameter], world_size: int) -> int:
    idx = all_params.index(param)
    return idx % world_size
