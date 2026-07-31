import unittest
import torch
import torch.nn as nn
from cs336_systems.benchmark import benchmark_latency_and_throughput, profile_execution

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return self.fc(x)

class TestBenchmark(unittest.TestCase):
    def test_benchmark_latency(self):
        model = DummyModel()
        x = torch.randn(4, 32)

        results = benchmark_latency_and_throughput(model, x, num_warmup=2, num_steps=5)
        self.assertIn("avg_latency_ms", results)
        self.assertIn("throughput_tokens_per_sec", results)
        self.assertGreater(results["avg_latency_ms"], 0.0)

    def test_profile_execution(self):
        model = DummyModel()
        x = torch.randn(4, 32)

        table_str = profile_execution(model, x)
        self.assertIsInstance(table_str, str)
        self.assertIn("CPU time total", table_str)

if __name__ == "__main__":
    unittest.main()
