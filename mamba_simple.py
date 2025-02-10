# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat, einsum
# import matplotlib.pyplot as plt
from plot import *
from ssm import SequentialSSM

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)


        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()
        # self.act = nn.ReLU()
        # self.act = nn.GELU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.softplus = Softplus()
        self.softplus_b = Softplus()

        self.hp = HadamardProduct()

        self.ssm = SequentialSSM(self.layer_idx, "f")
        self.ssm_b = SequentialSSM(self.layer_idx, "b")
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


        # # Initialize special dt projection to preserve variance at initialization
        # dt_init_std = self.dt_rank**-0.5 * dt_scale
        # if dt_init == "constant":
        #     nn.init.constant_(self.dt_proj.weight, dt_init_std)
        # elif dt_init == "random":
        #     nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # else:
        #     raise NotImplementedError

        # # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # dt = torch.exp(
        #     torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        #     + math.log(dt_min)
        # ).clamp(min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # inv_dt = dt + torch.log(-torch.expm1(-dt))
        # with torch.no_grad():
        #     self.dt_proj.bias.copy_(inv_dt)
        # # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True


    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape


        hidden_states = rearrange(hidden_states, "b l d -> (b l) d")
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "(b l) d -> b d l", l=seqlen)


        # Forward pass in bidirectional Mamba
        x, z = xz.chunk(2, dim=1) # x: B,d_inner,L = B,384,197, z: B,d_inner,L = B,384,197

        A = -torch.exp(self.A_log.float()) # d_inner,d_state = 384,16

        x = self.conv1d(x)[..., :seqlen] # B,d_inner,L = B,384,197
        x = self.act(x) # B,d_inner,L = B,384,197

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d")) 
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt: BL,dt_rank = 197*B,12
        # B: BL,d_state = 197*B,16
        # C: BL,d_state = 197*B,16

        dt = self.dt_proj(dt) # BL,d_inner = 197,384
        dt = rearrange(dt, "(b l) d -> b d l", l=seqlen) # B,d_inner,L = B,384,197
        # dt = F.softplus(dt)
        dt = self.softplus(dt)

        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # B,d_state,L = B,16,197
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # B,d_state,L = B,16,197
        D = self.D.float() # d_inner = 384

        y = self.selective_scan(x, dt, A, B, C, D, direction='f') # B,d_inner,L = B,384,197
        if z is not None:
            y = y * F.silu(z) # B,d_inner,L = B,384,197
        y = rearrange(y, "b d l -> b l d") # B,L,d_inner = B,197,384



        # Backward pass in bidirectional Mamba
        x_b, z_b = xz.flip([-1]).chunk(2, dim=1)

        A_b = -torch.exp(self.A_b_log.float())

        x_b = self.conv1d_b(x_b)[..., :seqlen]
        x_b = self.act(x_b)
        x_b_dbl = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))
        dt_b, B_b, C_b = torch.split(x_b_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_b = self.dt_proj_b(dt_b)
        dt_b = rearrange(dt_b, "(b l) d -> b d l", l=seqlen)
        # dt_b = F.softplus(dt_b)
        dt_b = self.softplus_b(dt_b)

        B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        D_b = self.D_b.float()

        y_b = self.selective_scan(x_b, dt_b, A_b, B_b, C_b, D_b, direction='b')
        if z_b is not None:
            y_b = y_b * F.silu(z_b)
        y_b = rearrange(y_b, "b d l -> b l d")


        # Output
        out = (y + y_b.flip([1])) / 2 # B,L,d_inner = B,197,384
        out = self.out_proj(out) # B,L,D = B,197,192


        # if self.layer_idx == 19:
        #     plt.figure(figsize=(6, 3))
        #     # plt.figure()
        #     plt.plot(out.amax((0, 2)).cpu().detach().numpy())
        #     plt.plot(out.amin((0, 2)).cpu().detach().numpy())
        #     plt.xlabel("Token Index")
        #     plt.ylabel("Value")
        #     plt.legend(["Max Value", "Min Value"])
        #     plt.tight_layout()
        #     plt.savefig("figures/out_proj.png")
        #     plt.close()

        #     exit()


        return out
    
    def selective_scan(self, x, delta, A, B, C, D=None, direction='f'):
        # k = 16
        # q = 2 ** (k - 1)

        seqlen = x.shape[2] # L = 197

        delta_expanded = delta.unsqueeze(-1) # B,d_inner,L,1 = B,384,197,1
        A_expanded = A.unsqueeze(1) # d_inner,1,d_state = 384,1,16

        # deltaA = delta_expanded * A_expanded # B,d_inner,L,d_state = B,384,197,16
        deltaA = self.hp(delta_expanded, A_expanded)
        deltaA = torch.exp(deltaA)

        # deltaB_x = torch.einsum('bdl,bnl,bdl->bdln', delta, B, x) # B,d_inner,L,d_state = B,384,197,16
        B_expanded = B.unsqueeze(1).permute(0, 1, 3, 2) # B,1,L,d_state = B,1,197,16
        x_expanded = x.unsqueeze(-1)
        deltaB = delta_expanded * B_expanded # B,d_inner,L,d_state = B,384,197,16

        D = D.unsqueeze(0)

        if direction == 'f':
            out = self.ssm(x, deltaA, deltaB, C, D)
        elif direction == 'b':
            out = self.ssm_b(x, deltaA, deltaB, C, D)

        # deltaB_x = deltaB * x_expanded

        # ys = []
        # for i in range(seqlen):
        #     if i == 0:
        #         h = deltaB_x[:, :, i] # B,d_inner,d_state = B,384,16
        #     else:
        #         h = deltaA[:, :, i] * h + deltaB_x[:, :, i] # B,d_inner,d_state = B,384,16

            
        #     # h_sim = (h / h.abs().max() * (q - 1)).round().clamp(-q, (q - 1)) * h.abs().max() / (q - 1)
        #     # h_sim = (h / h.abs().amax((0, 1)) * (q - 1)).round_().clamp_(-q, (q - 1)) * h.abs().amax((0, 1)) / (q - 1)
        #     # h = h_sim

        #     # y = torch.einsum('bdn,bn->bd', h, C[:, :, i]) # B,d_inner = B,384
        #     y = torch.matmul(h, C[:, :, i].unsqueeze(-1)).squeeze(-1) # B,d_inner = B,384
        #     ys.append(y) # B,d_inner = B,384

        # y = torch.stack(ys, dim=2) # B,d_inner,L = B,384,197

        # # y_sim = (y / y.abs().max() * (q - 1)).round_().clamp_(-q, (q - 1)) * y.abs().max() / (q - 1) 
        # # y = y_sim

        # out = y + x * rearrange(D, "d -> d 1")

        return out


class Softplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softplus(x)
    
class HadamardProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return A * B