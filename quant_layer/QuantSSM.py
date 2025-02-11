import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from quant_config.QuantConfig import *
from ssm import SequentialSSM
# from plot import *
# from quantization_error import *
from tensor_decomposition import *
from kmeans import *


# class SmoothScaleSelector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=0)
#         self.s_C_selector = nn.Sequential(
#             # nn.Linear(32 * 384 * 197 * 16, 16),
#             # nn.Linear(16, 384),
#             nn.Linear(384 * 197, 384),
#         )
#         self.s_D_selector = nn.Sequential(
#             # nn.Linear(32 * 384 * 197 * 16, 16),
#             nn.Linear(16 * 197, 16),
#         )

#     def forward(self, deltaB_x):
#         # deltaB_x = self.flatten(deltaB_x.cpu())
#         # s_C = self.s_C_selector(deltaB_x)
#         # s_D = self.s_D_selector(deltaB_x)
#         s_C = self.s_C_selector(self.flatten(deltaB_x.abs().amax((0, 3)).cpu())).abs() ** 0.5
#         s_D = self.s_D_selector(self.flatten(deltaB_x.abs().amax((0, 1)).transpose(0, 1).cpu())).abs() ** 0.5
#         return s_C, s_D
    
# class CLDSelector(nn.Module):
#     def __init__(self, h_record):
#         super().__init__()
#         self.s_C = nn.parameter.Parameter(h_record.abs().mean((0, 2, 3)) ** 0.6)
#         self.s_L = nn.parameter.Parameter(h_record.abs().mean((0, 1, 3)) ** 0.2)
#         self.s_D = nn.parameter.Parameter(h_record.abs().mean((0, 1, 2)) ** 0.6)

#     def forward(self):
#         # print("========================")
#         # print("s_C: ", self.s_C)
#         # print("s_L: ", self.s_L)
#         # print("s_D: ", self.s_D)
#         # print("========================")
#         return self.s_C + 1e-15, self.s_L + 1e-15, self.s_D + 1e-15


class QuantSSM(SequentialSSM):
    """
    The state space model (L steps sequentially)
    """

    def __init__(
        self,
        layer_idx,
        direction,
        mode = "raw",
        
        search_round = 1,
        smooth_quant = 0,
        represent = "mean",
        trend = "std",
        power = "normalized",
        factor = "original",

        w_config = QuantConfig(),
        i_config = QuantConfig(),
        h_config = QuantConfig(),
        o_config = QuantConfig(),
    ):
        

        super().__init__(layer_idx, direction)
        self.mode = mode

        self.search_round = search_round
        self.smooth_quant = smooth_quant
        self.represent = represent
        self.trend = trend
        self.power = power
        self.factor = factor

        self.w_config = w_config
        self.i_config = i_config
        self.h_config = h_config
        self.o_config = o_config
        
        self.raw_input = None
        self.raw_out = None
        self.raw_grad = None

        self.s = None
        self.s_C = None
        self.s_D = None

        # self.metric = "hessian"

        # self.ssm = SSM()
        # self.ssm_step = SSMStep()

    
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

        ys = []
        for i in range(L):
            h, y = self.ssm_step(h, deltaA, deltaB_x, C, i)

            ys.append(y)
        y = torch.stack(ys, dim=-1) # B, d_inner, L
        # out = y + x * D.unsqueeze(-1) # B, d_inner, L # --
        out = y # ++
        return out


    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw,dim=-1,keepdim=True), tensor_sim-torch.mean(tensor_sim,dim=-1,keepdim=True), dim=-1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                raw_grad = self.raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=-1)
        return similarity
    
    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.size(0)  # 第一維度的大小
            m = tensor.size(1)  # 第二維度的大小
            
            # 如果第二個維度太大，將它分段處理
            if m >= 16777216:
                num_segments = m // 16777216  # 將第二個維度分成 num_segments 段，每段大小為 16777216
                segments = [torch.quantile(tensor[:, i*16777216:(i+1)*16777216], quantile, dim=1) 
                            for i in range(num_segments)]
                
                # 如果有剩餘的列，單獨處理
                if m % 16777216 != 0:
                    remaining = torch.quantile(tensor[:, num_segments*16777216:], quantile, dim=1)
                    segments.append(remaining)
                
                # 對每個段的結果取平均值
                return torch.stack(segments, dim=1).mean(dim=1)
            else:
                # 第二個維度不太大，直接計算分位數
                return torch.quantile(tensor, quantile, dim=1)
        else:
            # 張量總元素數量小於 16777216，直接計算分位數
            return torch.quantile(tensor, quantile, dim=1)


    def _initialize_intervals(self, x, deltaA, deltaB, C, D):
        self.w_config.interval = (D.abs().max() / (self.w_config.qmax - 0.5)).detach()
        self.i_config.interval = [0, 0, 0, 0] # For x, deltaA, deltaB, C
        self.i_config.interval[0] = (x.abs().max() / (self.i_config.qmax - 0.5)).detach()
        self.i_config.interval[1] = (deltaA.abs().max() / (self.i_config.qmax - 0.5)).detach()
        self.i_config.interval[2] = (deltaB.abs().max() / (self.i_config.qmax - 0.5)).detach()
        self.i_config.interval[3] = (C.abs().max() / (self.i_config.qmax - 0.5)).detach()
        d_inner = x.shape[1]
        L = x.shape[2]

        self.s = torch.zeros(L, d_inner)

        if self.h_config.k_scaled_config.k_scaled:
            self.h_config.interval = torch.zeros(L, self.h_config.k_scaled_config.k)
        else:
            self.h_config.interval = torch.zeros(L)
        deltaB_x = deltaB * x.unsqueeze(-1)
        h = 0
        ys = []

        # if self.smooth_quant == 2:
        #     self.s = (deltaB_x.abs().amax((0, 2, 3)) + 1e-15).detach()
        #     deltaB_x = deltaB_x / self.s.unsqueeze(-1).unsqueeze(-1).to(deltaB_x.device)
        # if self.smooth_quant == 3:
        #     self.s = (deltaB_x.abs().amax((0, 2)) + 1e-15).detach()
        # if self.smooth_quant == 4:
        #     # self.s_C = (deltaB_x.abs().amax((0, 2, 3)) ** 0.5 + 1e-15).detach() # 384
        #     # self.s_D = (deltaB_x.abs().amax((0, 1, 2)) ** 0.5 + 1e-15).detach() # 16
        #     self.s_C = (deltaB_x.abs().mean((0, 2, 3)) ** 0.5 + 1e-15).detach() # 384
        #     self.s_D = (deltaB_x.abs().mean((0, 1, 2)) ** 0.5 + 1e-15).detach() # 16
        #     # self.s_C = (deltaB_x.abs().transpose(1, 3).flatten(start_dim=0, end_dim=2).quantile(0.999, dim=0) ** 0.5 + 1e-15).detach()
        #     # self.s_D = (deltaB_x.abs().flatten(start_dim=0, end_dim=2).quantile(0.999, dim=0) ** 0.5 + 1e-15).detach()
        #     self.s = self.s_C.unsqueeze(-1) * self.s_D # 384, 16

        #     deltaB_x = deltaB_x / self.s.unsqueeze(1).to(deltaB_x.device)
        #     C = C * self.s_D.unsqueeze(-1).to(C.device)
        # if self.smooth_quant == 5:
        #     self.s_C = (deltaB_x.abs().mean((0, 2, 3)) ** 0.5 + 1e-15).detach() # 384
        #     self.s_L = (deltaB_x.abs().mean((0, 1, 3)) ** 0.2 + 1e-15).detach() # 197
        #     self.s_D = (deltaB_x.abs().mean((0, 1, 2)) ** 0.5 + 1e-15).detach() # 16
        #     self.sA = torch.cat([torch.ones(1).to(self.s_L.device), self.s_L[0:-1]]) / self.s_L # 197
        #     self.sB = 1 / (self.s_C.unsqueeze(-1).unsqueeze(-1) * self.s_L.unsqueeze(-1) * self.s_D) # 384, 197, 16
        #     self.sC = self.s_D.unsqueeze(-1) * self.s_L # 16, 197
        #     self.sY = self.s_C # 384

        #     deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
        #     deltaB_x = deltaB_x * self.sB.to(deltaB_x.device)
        #     C = C * self.sC.to(C.device)

        if self.smooth_quant == 6:
            h_record = torch.zeros(deltaB_x.shape)
            for i in range(L):
                h, _ = self.ssm_step(h, deltaA, deltaB_x, C, i)
                h_record[:, :, i] = h

            h_record = h_record.abs().mean(0)

            if self.represent == "max":
                C_represent = h_record.amax((1, 2))
                L_represent = h_record.amax((0, 2))
                D_represent = h_record.amax((0, 1))
            if self.represent == "mean":
                C_represent = h_record.mean((1, 2))
                L_represent = h_record.mean((0, 2))
                D_represent = h_record.mean((0, 1))
            if self.represent == "median":
                C_represent = h_record.permute(0, 1, 2).reshape(h_record.shape[0], -1).median(dim=1).values
                L_represent = h_record.permute(1, 0, 2).reshape(h_record.shape[1], -1).median(dim=1).values
                D_represent = h_record.permute(2, 1, 0).reshape(h_record.shape[2], -1).median(dim=1).values
            if self.represent == "percentile_p99":
                C_represent = h_record.amax((1, 2)) * 0.99
                L_represent = h_record.amax((0, 2)) * 0.99
                D_represent = h_record.amax((0, 1)) * 0.99
            if self.represent == "percentile_p5":
                C_represent = h_record.amax((1, 2)) * 0.5
                L_represent = h_record.amax((0, 2)) * 0.5
                D_represent = h_record.amax((0, 1)) * 0.5
            if self.represent == "quantile_q99":
                C_represent = self._quantile(h_record.permute(0, 1, 2).reshape(h_record.shape[0], -1), 0.99)
                L_represent = self._quantile(h_record.permute(1, 0, 2).reshape(h_record.shape[1], -1), 0.99)
                D_represent = self._quantile(h_record.permute(2, 1, 0).reshape(h_record.shape[2], -1), 0.99)
                # C_represent = h_record.abs().permute(1, 0, 2, 3).reshape(h_record.shape[1], -1).quantile(q=0.99, dim=1).values
                # L_represent = h_record.abs().permute(2, 1, 0, 3).reshape(h_record.shape[2], -1).quantile(q=0.99, dim=1).values
                # D_represent = h_record.abs().permute(3, 1, 2, 0).reshape(h_record.shape[3], -1).quantile(q=0.99, dim=1).values
            if self.represent == "HOSVD":
                C_represent, L_represent, D_represent = hosvd_decomposition(h_record.abs())
            if self.represent == "CP":
                C_represent, L_represent, D_represent = cp_decomposition(h_record.abs())
            # if self.represent == "ML":
            #     C_represent, L_represent, D_represent = ml_decomposition(h_record.abs())

            if self.trend == "same":
                C_trend = 1
                L_trend = 1
                D_trend = 1
            if self.trend == "std":
                C_trend = C_represent.std()
                L_trend = L_represent.std()
                D_trend = D_represent.std()
            if self.trend == "var":
                C_trend = C_represent.var()
                L_trend = L_represent.var()
                D_trend = D_represent.var()
            if self.trend == "range":
                C_trend = C_represent.amax() - C_represent.amin()
                L_trend = L_represent.amax() - L_represent.amin()
                D_trend = D_represent.amax() - D_represent.amin()

            # C_std = h_record.abs().std((0, 2, 3))
            # L_std = h_record.abs().std((0, 1, 3))
            # D_std = h_record.abs().std((0, 1, 2))
            # C_std = h_record.abs().mean((0, 2, 3)).std()
            # L_std = h_record.abs().mean((0, 1, 3)).std()
            # D_std = h_record.abs().mean((0, 1, 2)).std()
            # C_std = h_record.abs().permute(1, 0, 2, 3).reshape(h_record.shape[1], -1).median(dim=1).values.std()
            # L_std = h_record.abs().permute(2, 1, 0, 3).reshape(h_record.shape[2], -1).median(dim=1).values.std()
            # D_std = h_record.abs().permute(3, 1, 2, 0).reshape(h_record.shape[3], -1).median(dim=1).values.std()

            if self.power == "original":
                C_power = C_trend
                L_power = L_trend
                D_power = D_trend
            if self.power == "normalized":
                C_power = C_trend / (C_trend + L_trend + D_trend)
                L_power = L_trend / (C_trend + L_trend + D_trend)
                D_power = D_trend / (C_trend + L_trend + D_trend)
            if self.power == "ML":
                C_power, L_power, D_power = ml_power(h_record.abs(), C_represent, L_represent, D_represent)

            if self.factor == "original":
                self.s_C = (C_represent ** C_power + 1e-15).detach()
                self.s_L = (L_represent ** L_power + 1e-15).detach()
                self.s_D = (D_represent ** D_power + 1e-15).detach()
            if self.factor == "ML":
                self.s_C, self.s_L, self.s_D = ml_decomposition(h_record.abs(), (C_represent ** C_power + 1e-15), (L_represent ** L_power + 1e-15), (D_represent ** D_power + 1e-15))


            # np.savetxt(f'./s_C/{self.layer_idx}_{self.direction}.txt', self.s_C, fmt='%10.10f')
            # np.savetxt(f'./s_L/{self.layer_idx}_{self.direction}.txt', self.s_L, fmt='%10.10f')
            # np.savetxt(f'./s_D/{self.layer_idx}_{self.direction}.txt', self.s_D, fmt='%10.10f')


            # s_C = np.loadtxt(f'./s_C/{self.layer_idx}_{self.direction}.txt', dtype=np.float32)
            # s_L = np.loadtxt(f'./s_L/{self.layer_idx}_{self.direction}.txt', dtype=np.float32)
            # s_D = np.loadtxt(f'./s_D/{self.layer_idx}_{self.direction}.txt', dtype=np.float32)
            # self.s_C = torch.from_numpy(s_C)
            # self.s_L = torch.from_numpy(s_L)
            # self.s_D = torch.from_numpy(s_D)
            # self.s_D = torch.ones(16)

            self.sA = torch.cat([torch.ones(1).to(self.s_L.device), self.s_L[0:-1]]) / self.s_L
            self.sB = 1 / (self.s_C.unsqueeze(-1).unsqueeze(-1) * self.s_L.unsqueeze(-1) * self.s_D)
            self.sC = self.s_D.unsqueeze(-1) * self.s_L
            self.sY = self.s_C

            deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
            deltaB_x = deltaB_x * self.sB.to(deltaB_x.device)
            C = C * self.sC.to(C.device)
            h = 0

        # if self.smooth_quant == 7:
        #     model = SmoothScaleSelector()
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        #     # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        #     deltaB_x_max_C = torch.argmax(deltaB_x.abs().amax((0, 3)).cpu(), dim=1) + 197 * torch.arange(0, 384)
        #     deltaB_x_max_D = torch.argmax(deltaB_x.abs().amax((0, 1)).transpose(0, 1).cpu(), dim=1) + 197 * torch.arange(0, 16)
        #     s_C_selector_init = torch.nn.functional.one_hot(deltaB_x_max_C, num_classes = 384 * 197) + 1e-15
        #     s_D_selector_init = torch.nn.functional.one_hot(deltaB_x_max_D, num_classes = 16 * 197) + 1e-15
        #     for layer in model.modules():
        #         if isinstance(layer, nn.Linear):
        #             nn.init.zeros_(layer.bias)
        #             if layer.weight.data.shape[0] == 384:
        #                 layer.weight.data = s_C_selector_init
        #             elif layer.weight.data.shape[0] == 16:
        #                 layer.weight.data = s_D_selector_init

        #     model.train()
        #     loss_min = 1e9
        #     for epoch in range(100):
        #         loss = self._train(model, optimizer, epoch, x, deltaA, deltaB, C)
        #         if loss < loss_min:
        #             loss_min = loss
        #             torch.save(model.state_dict(), "./checkpoints/selector_model_0.5.pth")

        #     model.load_state_dict(torch.load("./checkpoints/selector_model_0.5.pth"))
        #     model.eval()
        #     self.s_C, self.s_D = model(deltaB_x)
        #     self.s = self.s_C.unsqueeze(-1) * self.s_D
        #     deltaB_x = deltaB_x / self.s.unsqueeze(1).to(deltaB_x.device)
        #     C = C * self.s_D.unsqueeze(-1).to(C.device)
        #     self.smooth_quant = 4

        # if self.smooth_quant == 8:
        #     h_record = torch.zeros(deltaB_x.shape)
        #     for i in range(L):
        #         h, _ = self.ssm_step(h, deltaA, deltaB_x, C, i)
        #         h_record[:, :, i] = h

        #     model = CLDSelector(h_record)

        #     # if self.layer_idx == 0:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        #     model.train()
        #     loss_min = 1e9
        #     for epoch in range(1000):
        #         loss = self._train_CLD_selector(model, optimizer, epoch, h_record)
        #         if loss < loss_min:
        #             loss_min = loss
        #     #         torch.save(model.state_dict(), "./checkpoints/CLD_selector_model_1.pth")

        #     # model.load_state_dict(torch.load("./checkpoints/CLD_selector_model_1.pth"))

        #     model.eval()
        #     self.s_C, self.s_L, self.s_D = model()

        #     self.sA = torch.cat([torch.ones(1).to(self.s_L.device), self.s_L[0:-1]]) / self.s_L
        #     self.sB = 1 / (self.s_C.unsqueeze(-1).unsqueeze(-1) * self.s_L.unsqueeze(-1) * self.s_D)
        #     self.sC = self.s_D.unsqueeze(-1) * self.s_L
        #     self.sY = self.s_C

        #     deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
        #     deltaB_x = deltaB_x * self.sB.to(deltaB_x.device)
        #     C = C * self.sC.to(C.device)

        # if self.smooth_quant == 9:
        #     h_record = torch.zeros(deltaB_x.shape)
        #     for i in range(L):
        #         h, _ = self.ssm_step(h, deltaA, deltaB_x, C, i)
        #         h_record[:, :, i] = h

        #     h_record = h_record.mean(0)
        #     # print("mean: ", h_record.abs().mean())
        #     h_norm = torch.norm(h_record)
        #     # print("h_norm: ", h_norm)
        #     # h_record /= h_norm
        #     k = 1

        #     self.s_C, self.s_L, self.s_D = cp_decomposition(h_record + k, max_iter=100, tol=1e-6)
        #     self.s_C *= h_norm
            
        #     self.sA = torch.cat([torch.ones(1).to(self.s_L.device), self.s_L[0:-1]]) / self.s_L
        #     self.sB = 1 / (self.s_C.unsqueeze(-1).unsqueeze(-1) * self.s_L.unsqueeze(-1) * self.s_D)
        #     self.sC = self.s_D.unsqueeze(-1) * self.s_L
        #     self.sY = self.s_C

        #     deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
        #     deltaB_x = deltaB_x * self.sB.to(deltaB_x.device)
        #     C = C * self.sC.to(C.device)
        #     h = 0


        for i in range(L):
            h, y = self.ssm_step(h, deltaA, deltaB_x, C, i) # B, d_inner, d_state; B, d_inner

            if self.smooth_quant == 1:
                self.s[i] = (h.abs().amax((0, 2)) + 1e-15).detach()
                h = h / self.s[i].unsqueeze(-1).to(h.device)
            if self.smooth_quant == 3:
                h = h / self.s.to(h.device)

            if self.h_config.k_scaled_config.k_scaled:
                if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                    h = h.transpose(-1, -2) # hidden-state-wise
                # self.h_config.k_scaled_config.k_scaled_clusters = self._kmeans(h, self.h_config.k_scaled_config.k, num_iters=100, dim=-1)
                self.h_config.k_scaled_config.k_scaled_clusters = kmeans(h, self.h_config.k_scaled_config.k, num_iters=100, dim=-1)
                for cluster_index in range(self.h_config.k_scaled_config.k):
                    self.h_config.interval[i][cluster_index] = ((h * torch.eq(self.h_config.k_scaled_config.k_scaled_clusters, cluster_index)).abs().max() / (self.h_config.qmax - 0.5)).detach()
                if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                    h = h.transpose(-1, -2) # hidden-state-wise
            else:
                self.h_config.interval[i] = (h.abs().max() / (self.h_config.qmax - 0.5)).detach()

            if self.smooth_quant == 1:
                h = h * self.s[i].unsqueeze(-1).to(h.device)
                # y = y * self.s[i].to(y.device)
            if self.smooth_quant == 3:
                h = h * self.s.to(h.device)

            ys.append(y)
        y = torch.stack(ys, dim=-1)

        # --
        # if self.smooth_quant == 2:
        #     y = y * self.s.unsqueeze(-1).to(y.device)
        # if self.smooth_quant == 4:
        #     y = y * self.s_C.unsqueeze(-1).to(y.device)
        # if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
        #     y = y * self.sY.unsqueeze(-1).to(y.device)

        # D = D[0]
        # out = y + x * D.unsqueeze(-1) # --
        out = y # ++
        if self.o_config.k_scaled_config.k_scaled:
            if self.o_config.k_scaled_config.k_scaled_mode == "channel_wise":
                pass
            elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                out = out.transpose(-2, -1)
            # self.o_config.k_scaled_config.k_scaled_clusters = self._kmeans(out, self.o_config.k_scaled_config.k, num_iters=100, dim=1)
            self.o_config.k_scaled_config.k_scaled_clusters = kmeans(out, self.o_config.k_scaled_config.k, num_iters=100, dim=1)
            self.o_config.interval = torch.zeros(self.o_config.k_scaled_config.k)
            for cluster_index in range(self.o_config.k_scaled_config.k):
                self.o_config.interval[cluster_index] = ((out.transpose(1, 2) * torch.eq(self.o_config.k_scaled_config.k_scaled_clusters, cluster_index)).abs().max() / (self.o_config.qmax - 0.5)).detach()
        else:
            self.o_config.interval = (out.abs().max() / (self.o_config.qmax - 0.5)).detach()


    def _train(self, model, optimizer, epoch, x, deltaA, deltaB, C):
        torch.set_grad_enabled(True)
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        deltaB_x = deltaB * x.unsqueeze(-1)
        s_C, s_D = model(deltaB_x)

        s = s_C.unsqueeze(-1) * s_D
        deltaB_x = deltaB_x / s.unsqueeze(1).to(deltaB_x.device)
        C = C * s_D.unsqueeze(-1).to(C.device)
        
        L = x.shape[2]
        h = 0
        h_sim = 0
        ys = torch.tensor([])
        ys_sim = torch.tensor([])
        interval = torch.tensor([])
        # interval = torch.zeros(L, dtype=torch.float32)
        for i in range(L):
            h, y = self.ssm_step(h, deltaA, deltaB_x, C, i)
            h_sim, y_sim = self.ssm_step(h_sim, deltaA, deltaB_x, C, i)
            # interval[i] = (h.abs().max() / (self.h_config.qmax - 0.5))
            interval = torch.cat((interval, (h.abs().max() / (self.h_config.qmax - 0.5)).unsqueeze(-1).to(interval.device)), -1)
            # h_sim = (h_sim / (interval[i] + 1e-15)).round_().clamp_(-self.h_config.qmax, self.h_config.qmax - 1).mul_(interval[i])
            h_sim_1 = h_sim / (interval[i] + 1e-15)
            h_sim_2 = h_sim_1.round_()
            h_sim_3 = h_sim_2.clamp_(-self.h_config.qmax, self.h_config.qmax - 1)
            h_sim_4 = h_sim_3 * interval[i]
            h_sim = h_sim_4
            ys = torch.cat((ys, y.unsqueeze(-1).to(ys.device)), -1)
            ys_sim = torch.cat((ys_sim, y_sim.unsqueeze(-1).to(ys_sim.device)), -1)
        
        y = ys * s_C.unsqueeze(-1).to(ys.device)
        y_sim = ys_sim * s_C.unsqueeze(-1).to(ys_sim.device)

        loss_fn = nn.MSELoss()
        loss = loss_fn(y, y_sim)
        
        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch, "Loss: ", loss.item())

        return loss
    
    def _train_CLD_selector(self, model, optimizer, epoch, h_record):
        torch.set_grad_enabled(True)
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        s_C, s_L, s_D = model()
        h_sim = s_C.unsqueeze(-1).unsqueeze(-1) * s_L.unsqueeze(-1) * s_D
        loss_fn = nn.MSELoss()
        loss = loss_fn(h_record, h_sim)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: ", epoch, "Loss: ", loss.item())
        return loss


    def _search_best_h_interval(self, x, deltaA, deltaB, C, hidden_interval_candidates):
        L = x.shape[2]
        # quantize input
        x_sim, deltaA_sim, deltaB_sim, C_sim = self.quant_input(x, deltaA, deltaB, C)
        # forward
        deltaB_x = deltaB * x.unsqueeze(-1)
        deltaB_x_sim = deltaB_sim * x_sim.unsqueeze(-1)
        h = 0
        h_sim = 0

        if self.smooth_quant == 4:
            deltaB_x = deltaB_x / self.s.unsqueeze(1).to(deltaB_x.device)
            deltaB_x_sim = deltaB_x_sim / self.s.unsqueeze(1).to(deltaB_x_sim.device)
            C = C * self.s_D.unsqueeze(-1).to(C.device)
            C_sim = C_sim * self.s_D.unsqueeze(-1).to(C_sim.device)
        if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
            deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
            deltaA_sim = deltaA_sim * self.sA.unsqueeze(-1).to(deltaA_sim.device)
            deltaB_x = deltaB_x * self.sB.to(deltaB_x.device)
            deltaB_x_sim = deltaB_x_sim * self.sB.to(deltaB_x_sim.device)
            C = C * self.sC.to(C.device)
            C_sim = C_sim * self.sC.to(C_sim.device)

        if self.h_config.k_scaled_config.k_scaled:
            for i in range(L):
                h, y = self.ssm_step(h, deltaA, deltaB_x, C, i) # B, d_inner, d_state; B, d_inner
                for cluster_index in range(self.h_config.k_scaled_config.k):
                    similarities = []
                    if i == 0:
                        h_sim, _ = self.ssm_step(h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)
                    else:
                        for p_st in range(0, self.h_config.similarity_config.eq_n, self.h_config.similarity_config.parallel_eq_n):
                            p_ed = min(self.h_config.similarity_config.eq_n, p_st + self.h_config.similarity_config.parallel_eq_n)
                            cur_h_interval = torch.where(self.h_config.k_scaled_config.k_scaled_clusters == cluster_index, hidden_interval_candidates[i-1][cluster_index][p_st:p_ed].view(-1, 1, 1, 1), self.h_config.interval.cuda()[i-1][self.h_config.k_scaled_config.k_scaled_clusters])
                            # quantize hidden
                            cur_h_sim = h_sim.unsqueeze(0)
                            if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                                cur_h_sim = cur_h_sim.transpose(-1, -2) # hidden-state-wise
                            cur_h_sim = (cur_h_sim / cur_h_interval).round_().clamp_(-self.h_config.qmax, self.h_config.qmax - 1).mul_(cur_h_interval)
                            if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                                cur_h_sim = cur_h_sim.transpose(-1, -2) # hidden-state-wise
                            # forward
                            _, y_sim = self.ssm_step(cur_h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)
                            # calculate similarity and store them
                            similarity = self._get_similarity(y, y_sim, self.h_config.similarity_config.metric)
                            similarity = torch.mean(similarity, dim=-1)
                            similarities.append(similarity)
                        # store best hidden interval and store in h_interval
                        similarities = torch.cat(similarities, dim=0)
                        h_best_index = similarities.argmax(dim=0)

                        self.h_config.interval[i-1][cluster_index] = hidden_interval_candidates[i-1][cluster_index][h_best_index]
                        if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                            h_sim = h_sim.transpose(-1, -2) # hidden-state-wise
                        h_sim = (h_sim / self.h_config.interval.cuda()[i-1][self.h_config.k_scaled_config.k_scaled_clusters]).round_().clamp_(-self.h_config.qmax, self.h_config.qmax - 1).mul_(self.h_config.interval.cuda()[i-1][self.h_config.k_scaled_config.k_scaled_clusters])
                        if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                            h_sim = h_sim.transpose(-1, -2) # hidden-state-wise
                        h_sim, _ = self.ssm_step(h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)
        else:
            for i in range(L):
                h, y = self.ssm_step(h, deltaA, deltaB_x, C, i) # B, d_inner, d_state; B, d_inner
                similarities = []
                if i == 0:
                    h_sim, _ = self.ssm_step(h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)
                else:
                    for p_st in range(0, self.h_config.similarity_config.eq_n, self.h_config.similarity_config.parallel_eq_n):
                        p_ed = min(self.h_config.similarity_config.eq_n, p_st + self.h_config.similarity_config.parallel_eq_n)
                        cur_h_interval = hidden_interval_candidates[i-1][p_st:p_ed].view(-1, 1, 1, 1) # parallel_eq_n, 1, 1, 1
                        # quantize hidden
                        cur_h_sim = h_sim.unsqueeze(0) # 1, B, d_inner, d_state
                        cur_h_sim = (cur_h_sim / cur_h_interval).round_().clamp_(-self.h_config.qmax, self.h_config.qmax - 1).mul_(cur_h_interval) # parallel_eq_n, B, d_inner, d_state
                        # forward
                        _, y_sim = self.ssm_step(cur_h_sim, deltaA_sim, deltaB_x_sim, C_sim, i) # parallel_eq_n, B, d_inner, d_state; parallel_eq_n, B, d_inner
                        # calculate similarity and store them
                        similarity = self._get_similarity(y, y_sim, self.h_config.similarity_config.metric) # parallel_eq_n, B
                        similarity = torch.mean(similarity, dim=-1) # parallel_eq_n
                        similarities.append(similarity)
                    # store best hidden interval and store in h_interval
                    similarities = torch.cat(similarities, dim=0) # eq_n
                    h_best_index = similarities.argmax(dim=0)
                    
                    self.h_config.interval[i-1] = hidden_interval_candidates[i-1][h_best_index]
                    h_sim = (h_sim / self.h_config.interval[i-1]).round_().clamp_(-self.h_config.qmax, self.h_config.qmax - 1).mul_(self.h_config.interval[i-1])
                    h_sim, _ = self.ssm_step(h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)

    def _search_best_o_interval(self, x, deltaA, deltaB, C, D, output_interval_candidates):
        L = x.shape[2]

        if self.smooth_quant == 4:
            deltaB = deltaB / self.s.unsqueeze(1).to(deltaB.device)
            # deltaB_x_sim = deltaB_x_sim / self.s.unsqueeze(1).to(deltaB_x_sim.device)
            C = C * self.s_D.unsqueeze(-1).to(C.device)
            # C_sim = C_sim * self.s_D.unsqueeze(-1).to(C_sim.device)
        if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
            deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
            # deltaA_sim = deltaA_sim * self.sA.unsqueeze(-1).to(deltaA_sim.device)
            deltaB = deltaB * self.sB.to(deltaB.device)
            # deltaB_x_sim = deltaB_x_sim * self.sB.to(deltaB_x_sim.device)
            C = C * self.sC.to(C.device)
            # C_sim = C_sim * self.sC.to(C_sim.device
        
        # quantize weight
        D_sim = self.quant_weight(D)
        # quantize input
        x_sim, deltaA_sim, deltaB_sim, C_sim = self.quant_input(x, deltaA, deltaB, C)

        # forward
        deltaB_x_sim = deltaB_sim * x_sim.unsqueeze(-1)

        h_sim = 0
        ys_sim = []

        for i in range(L):
            h_sim, y_sim = self.ssm_step(h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)
            # quantize hidden
            h_sim = self.quant_hidden(h_sim, i)
            ys_sim.append(y_sim)
        y_sim = torch.stack(ys_sim, dim=-1)

        # --
        # if self.smooth_quant == 4:
        #     y_sim = y_sim * self.s_C.unsqueeze(-1).to(y_sim.device)
        # if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
        #     y_sim = y_sim * self.sY.unsqueeze(-1).to(y_sim.device)
        
        # out = y_sim + x_sim * D_sim.unsqueeze(-1) # B, d_inner, L # --
        out = y_sim # ++

        if self.o_config.k_scaled_config.k_scaled:
            for cluster_index in range(self.o_config.k_scaled_config.k):
                similarities = []
                for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
                    p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
                    cur_o_interval = torch.where(self.o_config.k_scaled_config.k_scaled_clusters == cluster_index, output_interval_candidates[cluster_index][p_st:p_ed].unsqueeze(-1), self.o_config.interval.cuda()[self.o_config.k_scaled_config.k_scaled_clusters])
                    # quantize output
                    if self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                        out = out.transpose(-2, -1)
                    out_sim = (out.transpose(1, 2).unsqueeze(-2) / cur_o_interval).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1) * cur_o_interval
                    # calculate similarity and store them
                    if self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                        out = out.transpose(-2, -1)
                        out_sim = out_sim.transpose(-3, -1)
                    # ++
                    if self.smooth_quant == 4:
                        out_sim = out_sim * self.s_C.to(out_sim.device)
                    if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
                        out_sim = out_sim * self.sY.to(out_sim.device)
                    similarity = self._get_similarity(self.raw_out.transpose(1, 2).unsqueeze(-2), out_sim, self.o_config.similarity_config.metric)
                    similarity = torch.mean(similarity, dim=(0, 1))
                    similarities.append(similarity)
                # store best output interval and store in o_interval
                similarities = torch.cat(similarities, dim=0)
                o_best_index = similarities.argmax(dim=0)
                self.o_config.interval[cluster_index] = output_interval_candidates[cluster_index][o_best_index]

            if self.o_config.k_scaled_config.k_scaled_power_of_two_scaling:
                for cluster_index in range(1, self.o_config.k_scaled_config.k):
                    self.o_config.interval[cluster_index] = self.o_config.interval[0] * torch.pow(2, torch.log2(self.o_config.interval[cluster_index] / self.o_config.interval[0]).round())
        else:
            similarities = []

            for p_st in range(0, self.o_config.similarity_config.eq_n ,self.o_config.similarity_config.parallel_eq_n):
                p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
                cur_o_interval = output_interval_candidates[p_st:p_ed].view(-1, 1, 1, 1) # parallel_eq_n, 1, 1, 1
                # quantize output
                out_sim = (out / cur_o_interval).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1) * cur_o_interval # parallel_eq_n, B, d_inner, L
                # ++
                if self.smooth_quant == 4:
                    out_sim = out_sim * self.s_C.unsqueeze(-1).to(out_sim.device)
                if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
                    out_sim = out_sim * self.sY.unsqueeze(-1).to(out_sim.device)
                # calculate similarity and store them
                similarity = self._get_similarity(self.raw_out.unsqueeze(0).transpose(2, 3), out_sim.transpose(2, 3), self.o_config.similarity_config.metric) # parallel_eq_n, B, L
                similarity = torch.mean(similarity, dim=(1, 2)) # parallel_eq_n
                similarities.append(similarity)
            # store best output interval and store in o_interval
            similarities = torch.cat(similarities, dim=0) # eq_n
            o_best_index = similarities.argmax(dim=0)
            self.o_config.interval = output_interval_candidates[o_best_index]


    def forward(self, x, deltaA, deltaB, C, D):
        if self.mode == "raw":
            out = self.ssm(x, deltaA, deltaB, C, D)
            out = out + x * D.unsqueeze(-1)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x, deltaA, deltaB, C, D)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x, deltaA, deltaB, C, D)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x, deltaA, deltaB, C, D)
        else:
            raise NotImplementedError
        return out
    
    def quant_input(self, x, deltaA, deltaB, C):
        if not self.i_config.quant:
            return x, deltaA, deltaB, C
        
        x_sim = (x / (self.i_config.interval[0] + 1e-15)).round_().clamp_(-self.i_config.qmax, self.i_config.qmax - 1).mul_(self.i_config.interval[0])
        deltaA_sim = (deltaA / (self.i_config.interval[1] + 1e-15)).round_().clamp_(-self.i_config.qmax, self.i_config.qmax - 1).mul_(self.i_config.interval[1])
        deltaB_sim = (deltaB / (self.i_config.interval[2] + 1e-15)).round_().clamp_(-self.i_config.qmax, self.i_config.qmax - 1).mul_(self.i_config.interval[2])
        C_sim = (C / (self.i_config.interval[3] + 1e-15)).round_().clamp_(-self.i_config.qmax, self.i_config.qmax - 1).mul_(self.i_config.interval[3])
        return x_sim, deltaA_sim, deltaB_sim, C_sim

    def quant_weight(self, D):
        if not self.w_config.quant:
            return D

        D_sim = (D / (self.w_config.interval + 1e-15)).round_().clamp_(-self.w_config.qmax, self.w_config.qmax - 1).mul_(self.w_config.interval)
        return D_sim

    def quant_hidden(self, h, i):
        if not self.h_config.quant:
            return h

        # if i % 32 != 0:
        #     return h

        h_sim = h
        cur_h_interval = self.h_config.interval[i]
        if self.h_config.k_scaled_config.k_scaled:
            if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                h_sim = h.transpose(-1, -2) # hidden-state-wise
            # cur_h_interval = torch.tensor([self.h_config.interval[i][j] for j in self.h_config.k_scaled_config.k_scaled_clusters]).cuda()
            cur_h_interval = torch.tensor(self.h_config.interval).cuda()[i, torch.tensor(self.h_config.k_scaled_config.k_scaled_clusters).cuda()]

        h_sim = (h_sim / (cur_h_interval + 1e-15)).round_().clamp_(-self.h_config.qmax, self.h_config.qmax - 1).mul_(cur_h_interval)

        if self.h_config.k_scaled_config.k_scaled:
            if self.h_config.k_scaled_config.k_scaled_mode == "hidden_dimension_wise":
                h_sim = h_sim.transpose(-1, -2) # hidden-state-wise

        return h_sim

    def quant_output(self, out):
        if not self.o_config.quant:
            return out
        
        cur_o_interval = self.o_config.interval
        if self.o_config.k_scaled_config.k_scaled:
            if self.o_config.k_scaled_config.k_scaled_mode == "channel_wise":
                    pass
            elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                out = out.transpose(-2, -1)
            cur_o_interval = torch.tensor([self.o_config.interval[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).unsqueeze(-1).to(out.device)

        out_sim = (out / (cur_o_interval + 1e-15)).round_().clamp_(-self.o_config.qmax, self.o_config.qmax - 1).mul_(cur_o_interval)

        if self.o_config.k_scaled_config.k_scaled & (self.o_config.k_scaled_config.k_scaled_mode == "token_wise"):
                out_sim = out_sim.transpose(-2, -1)
            
        return out_sim

    def quant_forward(self, x, deltaA, deltaB, C, D):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        D_sim = self.quant_weight(D)

        if self.smooth_quant == 1:
            deltaA = deltaA / self.s.transpose(0, 1).unsqueeze(-1).to(deltaA.device)
            deltaB = deltaB / self.s.transpose(0, 1).unsqueeze(-1).to(deltaB.device)
        if self.smooth_quant == 2:
            deltaB = deltaB / self.s.unsqueeze(-1).unsqueeze(-1).to(deltaB.device)
        if self.smooth_quant == 4:
            deltaB = deltaB / self.s.unsqueeze(1).to(deltaB.device)
            C = C * self.s_D.unsqueeze(-1).to(C.device)
        if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
            deltaA = deltaA * self.sA.unsqueeze(-1).to(deltaA.device)
            deltaB = deltaB * self.sB.to(deltaB.device)
            C = C * self.sC.to(C.device)

        x_sim, deltaA_sim, deltaB_sim, C_sim = self.quant_input(x, deltaA, deltaB, C)
        L = x.shape[2]
        deltaB_x_sim = deltaB_sim * x_sim.unsqueeze(-1)
        h_sim = 0
        ys = []
        for i in range(L):
            h, y = self.ssm_step(h_sim, deltaA_sim, deltaB_x_sim, C_sim, i)

            if self.smooth_quant == 3:
                h = h / self.s.to(h.device)

            h_sim = self.quant_hidden(h, i)

            if self.smooth_quant == 1:
                h_sim = h_sim * self.s[i].unsqueeze(-1).to(h_sim.device)
                y = y * self.s[i].to(y.device)
            if self.smooth_quant == 3:
                h_sim = h_sim * self.s.to(h_sim.device)

            ys.append(y)
        y = torch.stack(ys, dim=-1)

        y = self.quant_output(y) # ++

        if self.smooth_quant == 2:
            y = y * self.s.unsqueeze(-1).to(y.device)
        if self.smooth_quant == 4:
            y = y * self.s_C.unsqueeze(-1).to(y.device)
        if self.smooth_quant == 5 or self.smooth_quant == 6 or self.smooth_quant == 8 or self.smooth_quant == 9:
            y = y * self.sY.unsqueeze(-1).to(y.device)
        
        # --
        # out = y + x_sim * D_sim.unsqueeze(-1)
        # out_sim = self.quant_output(out)

        out_sim = y + x_sim * D_sim.unsqueeze(-1) # ++
        
        return out_sim
    
    def calibration_step1(self, x, deltaA, deltaB, C, D):
        out = self.ssm(x, deltaA, deltaB, C, D)
        self.raw_input = x.cpu().detach(), deltaA.cpu().detach(), deltaB.cpu().detach(), C.cpu().detach(), D.cpu().detach()
        self.raw_out = out.cpu().detach()
        out = out + x * D.unsqueeze(-1) # ++
        return out
    

    def calibration_step2(self, x, deltaA, deltaB, C, D):
        D = D[0]
        self._initialize_intervals(x, deltaA, deltaB, C, D)

        self.raw_out = self.raw_out.to(x.device)
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None
        
        L = x.shape[2]

        if self.h_config.k_scaled_config.k_scaled:
            hidden_interval_candidates = [torch.tensor([self.h_config.similarity_config.eq_alpha + i * (self.h_config.similarity_config.eq_beta - self.h_config.similarity_config.eq_alpha) / self.h_config.similarity_config.eq_n for i in range(self.h_config.similarity_config.eq_n + 1)]).cuda() * self.h_config.interval[j].unsqueeze(-1).cuda() for j in range(L)]
        else:
            hidden_interval_candidates = [torch.tensor([self.h_config.similarity_config.eq_alpha + i * (self.h_config.similarity_config.eq_beta - self.h_config.similarity_config.eq_alpha) / self.h_config.similarity_config.eq_n for i in range(self.h_config.similarity_config.eq_n + 1)]).cuda() * self.h_config.interval[j] for j in range(L)]

        if self.o_config.k_scaled_config.k_scaled:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval.unsqueeze(-1).cuda()
        else:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval

        for _ in range(self.search_round):
            if self.h_config.qmode == "similarity":
                self._search_best_h_interval(x, deltaA, deltaB, C, hidden_interval_candidates)
            if self.o_config.qmode == "similarity":
                self._search_best_o_interval(x, deltaA, deltaB, C, D, output_interval_candidates)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out = self.quant_forward(x, deltaA, deltaB, C, D)
        del self.raw_input, self.raw_out, self.raw_grad
        return out
