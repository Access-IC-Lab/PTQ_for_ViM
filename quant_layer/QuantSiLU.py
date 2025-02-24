import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_config.QuantConfig import *
from kmeans import *

class QuantSiLU(nn.SiLU):
    def __init__(
        self,
        mode = "raw",
        search_round = 1,
        i_config = QuantConfig(),
        o_config = QuantConfig(),
    ):
        super().__init__()

        self.n_calibration_step = 2
        self.mode = mode

        self.search_round = search_round

        self.i_config = i_config
        self.o_config = o_config

        self.metric = None
        self.next_nodes = []

        self.raw_input = None
        self.raw_out = None
        self.raw_grad = None

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=dim)
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
            similarity = torch.mean(similarity, dim=dim)
        return similarity
    
    def forward(self, x):
        if self.mode == "raw":
            out = F.silu(x)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self, x):
        assert self.calibrated is not None, f"You should run calibrate_forward before run quant_forward for {self}"
        x_sim = self.quant_input(x)
        out = F.silu(x_sim)
        out_sim = self.quant_output(out)
        return out_sim
    
    def quant_input(self, x):
        if not self.i_config.quant: 
            return x
        if self.i_config.bin == "uniform":
            cur_i_interval = self.i_config.interval
            cur_i_zeropoint = self.i_config.zeropoint
            if self.i_config.k_scaled_config.k_scaled:
                cur_i_interval = torch.tensor([self.i_config.interval[i] for i in self.i_config.k_scaled_config.k_scaled_clusters]).to(x.device)
                cur_i_zeropoint = torch.tensor([self.i_config.zeropoint[i] for i in self.i_config.k_scaled_config.k_scaled_clusters]).to(x.device)
            x_sim = ((x - cur_i_zeropoint) / (cur_i_interval + 1e-15)).round_().clamp_(-self.i_config.qmax, self.i_config.qmax - 1)
            x_sim.mul_(cur_i_interval)
            x_sim.add_(cur_i_zeropoint)
            return x_sim
        elif self.i_config.bin == "power_of_two":
            pass  # TODO

    def quant_output(self, out):
        if not self.o_config.quant:
            return out
        if self.o_config.bin == "uniform":
            cur_o_interval = self.o_config.interval
            cur_o_zeropoint = self.o_config.zeropoint
            if self.o_config.k_scaled_config.k_scaled:
                if self.o_config.k_scaled_config.k_scaled_mode == "channel_wise":
                    out = out.transpose(-2, -1)
                elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                    pass
                cur_o_interval = torch.tensor([self.o_config.interval[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                cur_o_zeropoint = torch.tensor([self.o_config.zeropoint[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
            out_sim = ((out - cur_o_zeropoint) / (cur_o_interval + 1e-15)).round_().clamp_(-self.o_config.qmax, self.o_config.qmax - 1)
            out_sim.mul_(cur_o_interval)
            out_sim.add_(cur_o_zeropoint)
            if self.o_config.k_scaled_config.k_scaled & (self.o_config.k_scaled_config.k_scaled_mode == "channel_wise"):
                out_sim = out_sim.transpose(-2, -1)
            return out_sim
        elif self.o_config.bin == "power_of_two":
            pass  # TODO

    def _initialize_intervals(self, x, out):
        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            self.i_config.interval = (x.abs().max() / (self.i_config.qmax - 0.5)).detach()

        if self.o_config.k_scaled_config.k_scaled:
            self.o_config.k_scaled_config.k_scaled_clusters = kmeans(out, self.o_config.k_scaled_config.k, num_iters=100, dim=1) # shape: D or L
            self.o_config.interval = torch.zeros(self.o_config.k_scaled_config.k) # shape: K
            self.o_config.zeropoint = torch.zeros(self.o_config.k_scaled_config.k) # shape: K
            for cluster_index in range(self.o_config.k_scaled_config.k):
                self.o_config.interval[cluster_index] = ((out * torch.eq(self.o_config.k_scaled_config.k_scaled_clusters, cluster_index).unsqueeze(-1)).abs().max() / (self.o_config.qmax - 0.5)).detach() # shape: K
        else:
            self.o_config.interval = (out.abs().max() / (self.o_config.qmax - 0.5)).detach()

    def _search_best_i_interval(self, x, input_interval_candidates):
        similarities = []
        for p_st in range(0, self.i_config.similarity_config.eq_n, self.i_config.similarity_config.parallel_eq_n):
            p_ed = min(self.i_config.similarity_config.eq_n, p_st + self.i_config.similarity_config.parallel_eq_n)
            cur_i_interval = input_interval_candidates[p_st:p_ed].view(-1, 1, 1, 1) # shape: parallel_eq_n, 1, 1, 1
            # quantize input
            x_sim = x.unsqueeze(0)
            x_sim = (x_sim / (cur_i_interval + 1e-15)).round_().clamp_(-self.i_config.qmax, self.i_config.qmax - 1) * cur_i_interval
            # quantize output
            out = F.silu(x_sim)
            out_sim = self.quant_output(out)
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(0), out_sim, self.i_config.similarity_config.metric, dim = 2)
            similarity = torch.mean(similarity, dim = [1, 2])
            similarities.append(similarity)
        # store best input interval and store in i_interval
        similarities = torch.cat(similarities, dim = 0)
        i_best_index = similarities.argmax(dim = 0)
        self.i_config.interval = input_interval_candidates[i_best_index]

    def _search_best_o_interval(self, x, output_interval_candidates):
        if self.o_config.k_scaled_config.k_scaled:
            for cluster_index in range(self.o_config.k_scaled_config.k):
                similarities = []
                for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
                    p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
                    cur_o_interval = torch.where(self.o_config.k_scaled_config.k_scaled_clusters == cluster_index, output_interval_candidates[cluster_index][p_st:p_ed].unsqueeze(-1), self.o_config.interval.cuda()[self.o_config.k_scaled_config.k_scaled_clusters]) # shape: parallel_eq_n, C
                    cur_o_interval = cur_o_interval.unsqueeze(1).unsqueeze(-1) # shape: parallel_eq_n, 1, C, 1
                    # quantize input
                    x_sim = self.quant_input(x)
                    # quantize output
                    out = F.silu(x_sim)
                    out = out.unsqueeze(0) # shape: 1, B, C, L
                    out_sim = (out / (cur_o_interval + 1e-15)).round_().clamp_(-self.o_config.qmax, self.o_config.qmax - 1) * cur_o_interval
                    # calculate similarity and store them
                    similarity = self._get_similarity(self.raw_out.unsqueeze(0), out_sim, self.o_config.similarity_config.metric, dim = 2)
                    similarity = torch.mean(similarity, dim = [1, 2])
                    similarities.append(similarity)
                # store best input interval and store in o_interval
                similarities = torch.cat(similarities, dim = 0)
                o_best_index = similarities.argmax(dim = 0)
                self.o_config.interval[cluster_index] = output_interval_candidates[cluster_index][o_best_index]
        else:
            similarities = []
            for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
                p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
                cur_o_interval = output_interval_candidates[p_st:p_ed].view(-1, 1, 1, 1) # shape: parallel_eq_n, 1, 1, 1
                # quantize input
                x_sim = self.quant_input(x)
                # quantize output
                out = F.silu(x_sim)
                out = out.unsqueeze(0)
                out_sim = (out / (cur_o_interval + 1e-15)).round_().clamp_(-self.o_config.qmax, self.o_config.qmax - 1) * cur_o_interval
                # calculate similarity and store them
                similarity = self._get_similarity(self.raw_out.unsqueeze(0), out_sim, self.o_config.similarity_config.metric, dim = 2)
                similarity = torch.mean(similarity, dim = [1, 2])
                similarities.append(similarity)
            # store best input interval and store in o_interval
            similarities = torch.cat(similarities, dim = 0)
            o_best_index = similarities.argmax(dim = 0)
            self.o_config.interval = output_interval_candidates[o_best_index]

    def calibration_step1(self, x):
        out = F.silu(x)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out
    
    def calibration_step2(self, x):
        out = F.silu(x)
        
        self._initialize_intervals(x, out)

        self.raw_out = self.raw_out.to(x.device)
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad is not None else None

        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            input_interval_candidates = torch.tensor([self.i_config.similarity_config.eq_alpha + i * (self.i_config.similarity_config.eq_beta - self.i_config.similarity_config.eq_alpha) / self.i_config.similarity_config.eq_n for i in range(self.i_config.similarity_config.eq_n + 1)]).cuda() * self.i_config.interval

        if self.o_config.k_scaled_config.k_scaled:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval.unsqueeze(-1).cuda()
        else:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval

        for _ in range(self.search_round):
            if self.i_config.qmode == "similarity":
                self._search_best_i_interval(x, input_interval_candidates)
            if self.o_config.qmode == "similarity":
                self._search_best_o_interval(x, output_interval_candidates)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad is not None else None

        self.calibrated = True
        out = self.quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out