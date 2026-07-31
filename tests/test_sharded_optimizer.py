import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from cs336_systems.sharded_optimizer import ShardedOptimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class TestShardedOptimizer(unittest.TestCase):
    def test_sharded_optimizer_step_matches_adamw(self):
        torch.manual_seed(42)
        model_ref = SimpleModel()
        model_sharded = SimpleModel()

        # Ensure exact same initial parameters
        model_sharded.load_state_dict(model_ref.state_dict())

        opt_ref = optim.AdamW(model_ref.parameters(), lr=1e-3, weight_decay=1e-2)
        opt_sharded = ShardedOptimizer(model_sharded.parameters(), optimizer_cls=optim.AdamW, lr=1e-3, weight_decay=1e-2)

        x = torch.randn(4, 8)

        # Forward + backward pass 1
        y_ref = model_ref(x).sum()
        y_sharded = model_sharded(x).sum()

        y_ref.backward()
        y_sharded.backward()

        opt_ref.step()
        opt_sharded.step()

        # Verify updated weights are identical
        for p_ref, p_sharded in zip(model_ref.parameters(), model_sharded.parameters()):
            self.assertTrue(
                torch.allclose(p_ref, p_sharded, atol=1e-6),
                f"Parameters mismatch after step! Max diff: {(p_ref - p_sharded).abs().max()}"
            )

if __name__ == "__main__":
    unittest.main()
