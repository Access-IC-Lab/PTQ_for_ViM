import torch
import torch.nn as nn

# def rms_norm(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False, upcast=False):
#     dtype = x.dtype
#     if upcast:
#         weight = weight.float()
#         bias = bias.float() if bias is not None else None
#     if upcast:
#         x = x.float()
#         residual = residual.float() if residual is not None else residual
#     if residual is not None:
#         x = (x + residual).to(x.dtype)
#     rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
#     out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
#     out = out.to(dtype)
#     return out if not prenorm else (out, x)


# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.eps = eps
#         self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
#         self.register_parameter("bias", None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.ones_(self.weight)

#     def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
#         return rms_norm(
#             x,
#             self.weight,
#             self.bias,
#             residual=residual,
#             eps=self.eps,
#             prenorm=prenorm,
#             residual_in_fp32=residual_in_fp32,
#             upcast=True
#         )

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
    # return out if not prenorm else (out, x)
    return out


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(self.hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.ones_(self.weight)

    # def forward(self, x, residual=None, prenorm=False):
    def forward(self, x, residual=None):
        return rms_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            # prenorm=prenorm,
        )
