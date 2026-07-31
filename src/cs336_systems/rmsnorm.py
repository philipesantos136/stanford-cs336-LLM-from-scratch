import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,
        Y_ptr,
        W_ptr,
        Rstd_ptr,
        stride_x_row,
        stride_y_row,
        N_COLS: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_COLS

        row_x_ptr = X_ptr + row_idx * stride_x_row
        row_y_ptr = Y_ptr + row_idx * stride_y_row

        x = tl.load(row_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # RMS = sqrt(mean(x^2) + eps)
        var = tl.sum(x * x, axis=0) / N_COLS
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd_ptr + row_idx, rstd)

        y = x * rstd * w
        tl.store(row_y_ptr + col_offsets, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        DX_ptr,
        DW_ptr,
        DY_ptr,
        X_ptr,
        W_ptr,
        Rstd_ptr,
        stride_dx_row,
        stride_dy_row,
        stride_x_row,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_COLS

        row_dx_ptr = DX_ptr + row_idx * stride_dx_row
        row_dy_ptr = DY_ptr + row_idx * stride_dy_row
        row_x_ptr = X_ptr + row_idx * stride_x_row

        dy = tl.load(row_dy_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(row_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + row_idx).to(tl.float32)

        # x_norm = x * rstd
        x_norm = x * rstd
        dy_w = dy * w

        # dx = rstd * (dy_w - x_norm * mean(dy_w * x_norm))
        mean_dyw_xnorm = tl.sum(dy_w * x_norm, axis=0) / N_COLS
        dx = rstd * (dy_w - x_norm * mean_dyw_xnorm)

        tl.store(row_dx_ptr + col_offsets, dx, mask=mask)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
        # Determine if we can use Triton kernel
        use_triton = HAS_TRITON and x.is_cuda and weight.is_cuda

        if use_triton:
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            n_rows, n_cols = x_2d.shape

            y_2d = torch.empty_like(x_2d)
            rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

            BLOCK_SIZE = triton.next_power_of_2(n_cols)

            _rmsnorm_fwd_kernel[(n_rows,)](
                x_2d,
                y_2d,
                weight,
                rstd,
                x_2d.stride(0),
                y_2d.stride(0),
                N_COLS=n_cols,
                eps=eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(x_2d, weight, rstd)
            ctx.eps = eps
            ctx.n_cols = n_cols
            ctx.orig_shape = orig_shape
            ctx.use_triton = True

            return y_2d.reshape(orig_shape)
        else:
            # PyTorch fallback for CPU / Windows / non-CUDA
            variance = x.pow(2).mean(-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            x_norm = x * rstd
            y = x_norm * weight

            ctx.save_for_backward(x, weight, rstd, x_norm)
            ctx.eps = eps
            ctx.use_triton = False
            return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if ctx.use_triton:
            x_2d, weight, rstd = ctx.saved_tensors
            n_rows, n_cols = x_2d.shape
            dy_2d = dy.reshape(-1, n_cols)

            dx_2d = torch.empty_like(x_2d)
            BLOCK_SIZE = triton.next_power_of_2(n_cols)

            _rmsnorm_bwd_kernel[(n_rows,)](
                dx_2d,
                weight,  # passed as DW in position but kernel computes dx
                dy_2d,
                x_2d,
                weight,
                rstd,
                dx_2d.stride(0),
                dy_2d.stride(0),
                x_2d.stride(0),
                N_ROWS=n_rows,
                N_COLS=n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Weight gradient: sum over rows of dy * (x * rstd)
            x_norm = x_2d * rstd.unsqueeze(-1)
            dweight = (dy_2d * x_norm).sum(dim=0)

            dx = dx_2d.reshape(ctx.orig_shape)
            return dx, dweight, None
        else:
            x, weight, rstd, x_norm = ctx.saved_tensors
            # Weight gradient
            dweight = (dy * x_norm).sum(dim=tuple(range(dy.ndim - 1)))
            
            # Input gradient:
            # dx = rstd * (dy * weight - x_norm * mean(dy * weight * x_norm, dim=-1))
            dy_w = dy * weight
            mean_dyw_xnorm = (dy_w * x_norm).mean(dim=-1, keepdim=True)
            dx = rstd * (dy_w - x_norm * mean_dyw_xnorm)

            return dx, dweight, None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
