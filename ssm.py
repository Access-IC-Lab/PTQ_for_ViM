import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

class SequentialSSM(nn.Module):
    def __init__(
        self,
        layer_idx,
        direction,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.direction = direction

    def ssm_step(self, h, deltaA, deltaB_x, C, step):
        h = deltaB_x[:, :, step] + deltaA[:, :, step] * h
        y = torch.matmul(h, C[:, :, step].unsqueeze(-1)).squeeze(-1) # B, d_inner
        
        return h, y

    def ssm(self, x, deltaA, deltaB, C, D):
        """
        x     : B, d_inner, L          = B, 384, 197
        deltaA: B, d_inner, L, d_state = B, 384, 197, 16
        deltaB: B, d_inner, L, d_state = B, 384, 197, 16
        C     : B, d_state, L          = B, 16, 197
        D     : d_inner                = 384
        """

        L = x.shape[2]

        deltaB_x = deltaB * x.unsqueeze(-1) # B, d_inner, L, d_state
        h = 0
        h_sim = 0

        ys = []
        for i in range(L):

            h, y = self.ssm_step(h, deltaA, deltaB_x, C, i)
            s = h.abs().max() / 127 + 1e-15
            h_sim = (h_sim / s).round_().clamp_(-128, 127).mul_(s)

            ys.append(y)
        


        y = torch.stack(ys, dim=-1) # B, d_inner, L

        out = y + x * D.unsqueeze(-1) # B, d_inner, L

        return out

    def forward(self, x, deltaA, deltaB, C, D):
        return self.ssm(x, deltaA, deltaB, C, D)