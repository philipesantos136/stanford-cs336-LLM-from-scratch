import unittest
import torch
import torch.nn as nn
from cs336_systems.rmsnorm import RMSNorm, RMSNormFunction

class ReferenceRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class TestRMSNorm(unittest.TestCase):
    def test_forward_and_backward(self):
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 4, 16, 64
        eps = 1e-5

        x_ref = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        x_custom = x_ref.clone().detach().requires_grad_(True)

        ref_norm = ReferenceRMSNorm(d_model, eps=eps)
        custom_norm = RMSNorm(d_model, eps=eps)

        # Ensure same initial weights
        custom_norm.weight.data.copy_(ref_norm.weight.data)

        # Forward pass
        y_ref = ref_norm(x_ref)
        y_custom = custom_norm(x_custom)

        self.assertTrue(
            torch.allclose(y_ref, y_custom, atol=1e-5),
            f"Forward outputs differ! Max diff: {(y_ref - y_custom).abs().max()}"
        )

        # Backward pass
        loss_ref = (y_ref ** 2).sum()
        loss_custom = (y_custom ** 2).sum()

        loss_ref.backward()
        loss_custom.backward()

        self.assertTrue(
            torch.allclose(x_ref.grad, x_custom.grad, atol=1e-5),
            f"Input gradients differ! Max diff: {(x_ref.grad - x_custom.grad).abs().max()}"
        )

        self.assertTrue(
            torch.allclose(ref_norm.weight.grad, custom_norm.weight.grad, atol=1e-5),
            f"Weight gradients differ! Max diff: {(ref_norm.weight.grad - custom_norm.weight.grad).abs().max()}"
        )

if __name__ == "__main__":
    unittest.main()
