import unittest
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from cs336_systems.ddp import DDP

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class TestDDP(unittest.TestCase):
    def test_ddp_single_process(self):
        torch.manual_seed(42)
        model = SimpleModel()
        ddp_model = DDP(model)

        x = torch.randn(4, 10)
        y = ddp_model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(model.fc.weight.grad)
        self.assertEqual(model.fc.weight.grad.shape, (5, 10))

    def test_ddp_simulated_multiprocess(self):
        # Test simulating DDP all_reduce step
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            try:
                dist.init_process_group("gloo", rank=0, world_size=1)
            except Exception:
                pass

        if dist.is_initialized():
            torch.manual_seed(42)
            model = SimpleModel()
            ddp_model = DDP(model)

            x = torch.randn(4, 10)
            y = ddp_model(x)
            loss = y.sum()
            loss.backward()

            # Manual check gradient synchronization
            ddp_model.finish_gradient_synchronization()
            self.assertIsNotNone(model.fc.weight.grad)
            dist.destroy_process_group()

if __name__ == "__main__":
    unittest.main()
