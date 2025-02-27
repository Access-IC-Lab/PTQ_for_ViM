import torch
import torch.nn as nn

def rms_norm(x, weight, bias, residual=None, eps=1e-6):
    dtype = x.dtype
    weight = weight.float().cuda()
    bias = bias.float() if bias is not None else None
    x = x.float()
    residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(self.hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)

    def forward(self, x, residual=None):
        return rms_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
        )
