import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_layers.QuantLinear import *

class QuantXProj(QuantLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,

        mode = "raw",
        bias_bit = None,
        bias_correction = False,

        search_round = 1,

        w_config = QuantConfig(),
        i_config = QuantConfig(),
        o_config = QuantConfig(),

        L = 197,
        dt_rank = 12,
        D = 16,
    ):
        super().__init__(in_features, out_features, bias)

        self.n_calibration_step = 2
        self.mode = mode
        self.bias_bit = bias_bit
        self.bias_correction = bias_correction

        self.search_round = search_round

        self.w_config = w_config
        self.i_config = i_config
        self.o_config = o_config

        self.L = L

        self.metric = None
        self.next_nodes = []

        self.raw_input = None
        self.raw_out = None
        self.raw_grad = None

        self.dt_rank = dt_rank
        self.D = D

    def quant_weight_bias(self):
        if not self.w_config.quant:
            return self.weight, self.bias
        if self.w_config.bin == "uniform":
            cur_w_interval_dt = self.w_config.interval[0]
            cur_w_interval_B = self.w_config.interval[1]
            cur_w_interval_C = self.w_config.interval[2]
            # cur_w_zeropoint_dt = self.w_config.zeropoint[0]
            # cur_w_zeropoint_B = self.w_config.zeropoint[1]
            # cur_w_zeropoint_C = self.w_config.zeropoint[2]
            cur_w_zeropoint_dt = 0.
            cur_w_zeropoint_B = 0.
            cur_w_zeropoint_C = 0.
            if self.w_config.k_scaled_config.k_scaled:
                # cur_w_interval_dt = torch.tensor([self.w_config.interval[0][i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                # cur_w_interval_B = torch.tensor([self.w_config.interval[1][i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                # cur_w_interval_C = torch.tensor([self.w_config.interval[2][i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                # cur_w_zeropoint_dt = torch.tensor([self.w_config.zeropoint[0][i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                # cur_w_zeropoint_B = torch.tensor([self.w_config.zeropoint[1][i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                # cur_w_zeropoint_C = torch.tensor([self.w_config.zeropoint[2][i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                pass
            w_sim_dt = ((self.weight[:self.dt_rank,:] - cur_w_zeropoint_dt) / (cur_w_interval_dt + 1e-15)).round_().clamp_(-self.w_config.qmax,self.w_config.qmax-1).mul_(cur_w_interval_dt).add_(cur_w_zeropoint_dt)
            w_sim_B = ((self.weight[self.dt_rank:(self.dt_rank+self.D),:] - cur_w_zeropoint_B) / (cur_w_interval_B + 1e-15)).round_().clamp_(-self.w_config.qmax,self.w_config.qmax-1).mul_(cur_w_interval_B).add_(cur_w_zeropoint_B)
            w_sim_C = ((self.weight[(self.dt_rank+self.D):,:] - cur_w_zeropoint_C) / (cur_w_interval_C + 1e-15)).round_().clamp_(-self.w_config.qmax,self.w_config.qmax-1).mul_(cur_w_interval_C).add_(cur_w_zeropoint_C)
            w_sim = torch.cat((w_sim_dt,w_sim_B,w_sim_C), dim=0)
            return w_sim, self.bias
        elif self.w_config.bin == "power_of_two":
            pass # TODO

    def quant_output(self, out):
        if not self.o_config.quant:
            return out
        if self.o_config.bin == "uniform":
            cur_o_interval_dt = self.o_config.interval[0]
            cur_o_interval_B = self.o_config.interval[1]
            cur_o_interval_C = self.o_config.interval[2]
            # cur_o_zeropoint_dt = self.o_config.zeropoint[0]
            # cur_o_zeropoint_B = self.o_config.zeropoint[1]
            # cur_o_zeropoint_C = self.o_config.zeropoint[2]
            cur_o_zeropoint_dt = 0.
            cur_o_zeropoint_B = 0.
            cur_o_zeropoint_C = 0.
            if self.o_config.k_scaled_config.k_scaled:
                # if self.o_config.k_scaled_config.k_scaled_mode == "channel_wise":
                #     pass
                # elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                #     out = out.transpose(-2, -1)
                # cur_o_interval_dt = torch.tensor([self.o_config.interval[0][i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                # cur_o_interval_B = torch.tensor([self.o_config.interval[1][i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                # cur_o_interval_C = torch.tensor([self.o_config.interval[2][i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                # cur_o_zeropoint_dt = torch.tensor([self.o_config.zeropoint[0][i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                # cur_o_zeropoint_B = torch.tensor([self.o_config.zeropoint[1][i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                # cur_o_zeropoint_C = torch.tensor([self.o_config.zeropoint[2][i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                pass
            out_sim_dt = ((out[:,:self.dt_rank] - cur_o_zeropoint_dt) / (cur_o_interval_dt + 1e-15)).round_().clamp_(-self.o_config.qmax,self.o_config.qmax-1).mul_(cur_o_interval_dt).add_(cur_o_zeropoint_dt)
            out_sim_B = ((out[:,self.dt_rank:(self.dt_rank+self.D)] - cur_o_zeropoint_B) / (cur_o_interval_B + 1e-15)).round_().clamp_(-self.o_config.qmax,self.o_config.qmax-1).mul_(cur_o_interval_B).add_(cur_o_zeropoint_B)
            out_sim_C = ((out[:,(self.dt_rank+self.D):] - cur_o_zeropoint_C) / (cur_o_interval_C + 1e-15)).round_().clamp_(-self.o_config.qmax,self.o_config.qmax-1).mul_(cur_o_interval_C).add_(cur_o_zeropoint_C)
            out_sim = torch.cat((out_sim_dt,out_sim_B,out_sim_C),dim=1)
            if self.o_config.k_scaled_config.k_scaled & (self.o_config.k_scaled_config.k_scaled_mode == "token_wise"):
                # out_sim = out_sim.transpose(-2, -1)
                pass
            return out_sim
        elif self.o_config.bin == "power_of_two":
            pass # TODO

    def _initialize_intervals(self, x, out):
        if self.w_config.k_scaled_config.k_scaled:
            pass
        else:
            # self.w_config.interval=((self.weight.data.abs().max())/(self.w_config.qmax-0.5)).detach()
            self.w_config.interval = torch.zeros(3) # For dt, B, C
            self.w_config.interval[0] = (self.weight[:self.dt_rank,:].data.abs().max() / (self.w_config.qmax-0.5)).detach()
            self.w_config.interval[1] = (self.weight[self.dt_rank:(self.dt_rank+self.D),:].data.abs().max() / (self.w_config.qmax-0.5)).detach()
            self.w_config.interval[2] = (self.weight[(self.dt_rank+self.D):,:].data.abs().max() / (self.w_config.qmax-0.5)).detach()

        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            self.i_config.interval=(x.abs().max()/(self.i_config.qmax-0.5)).detach()

        if self.o_config.k_scaled_config.k_scaled:
            # if self.o_config.k_scaled_config.k_scaled_mode == "channel_wise":
            #     pass
            # elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
            #     out = out.transpose(-2, -1)
            
            # self.o_config.k_scaled_config.k_scaled_clusters = self._kmeans(out, self.o_config.k_scaled_config.k, num_iters=100, dim=-1) # shape: D or L
            # self.o_config.interval = torch.zeros(self.o_config.k_scaled_config.k) # shape: K
            # self.o_config.zeropoint = torch.zeros(self.o_config.k_scaled_config.k) # shape: K
            # for cluster_index in range(self.o_config.k_scaled_config.k):
            #     self.o_config.interval[cluster_index] = ((out * torch.eq(self.o_config.k_scaled_config.k_scaled_clusters, cluster_index)).abs().max() / (self.o_config.qmax - 0.5)).detach() # shape: K

            # # if self.o_config.k_scaled_config.k_scaled_power_of_two_scaling:
            # #     for cluster_index in range(1, self.o_config.k_scaled_config.k):
            # #         self.o_config.interval[cluster_index] = self.o_config.interval[0] * torch.pow(2, torch.log2(self.o_config.interval[cluster_index] / self.o_config.interval[0]).round())
            pass
        else:
            self.o_config.interval = torch.zeros(3)
            self.o_config.interval[0] = (out[:,:self.dt_rank].abs().max()/(self.o_config.qmax-0.5)).detach()
            self.o_config.interval[1] = (out[:,self.dt_rank:(self.dt_rank+self.D)].abs().max()/(self.o_config.qmax-0.5)).detach()
            self.o_config.interval[2] = (out[:,(self.dt_rank+self.D):].abs().max()/(self.o_config.qmax-0.5)).detach()

    def _search_best_w_interval_dt(self, x, weight_interval_candidates):
        similarities = []
        for p_st in range(0, self.w_config.similarity_config.eq_n, self.w_config.similarity_config.parallel_eq_n):
            p_ed = min(self.w_config.similarity_config.eq_n, p_st + self.w_config.similarity_config.parallel_eq_n)
            cur_w_interval_dt = weight_interval_candidates[0][p_st:p_ed].view(-1, 1, 1).cuda() # shape: parallel_eq_n, 1, 1
            cur_w_interval_B = self.w_config.interval[1]
            cur_w_interval_C = self.w_config.interval[2]
            # quantize weight and bias
            oc, ic = self.weight.shape
            w_sim_dt = (self.weight[:self.dt_rank,:] / cur_w_interval_dt).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).cuda().mul_(cur_w_interval_dt) # shape: parallel_eq_n, oc, ic
            w_sim_B = (self.weight[self.dt_rank:(self.dt_rank+self.D),:] / cur_w_interval_B).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).cuda().mul_(cur_w_interval_B).unsqueeze(0).expand(p_ed - p_st, -1, -1) # shape: 1, oc, ic
            w_sim_C = (self.weight[(self.dt_rank+self.D):,:] / cur_w_interval_C).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).cuda().mul_(cur_w_interval_C).unsqueeze(0).expand(p_ed - p_st, -1, -1) # shape: 1, oc, ic
            w_sim = torch.cat((w_sim_dt,w_sim_B,w_sim_C), dim=1)
            w_sim = w_sim.view(-1, ic) # shape: parallel_eq_n * oc, ic
            bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
            # quantize input
            x_sim = self.quant_input(x) # shape: B, *, ic
            # quantize output
            out = F.linear(x_sim, w_sim, bias_sim) # shape: B, *, parallel_eq_n * oc
            out = torch.cat(torch.chunk(out.unsqueeze(-2), p_ed-p_st, dim=-1), dim=-2) # shape: B, *, parallel_eq_n, oc
            out = out.transpose(-3, -2) # shape: B, parallel_eq_n, *, oc
            out_sim = self.quant_output(out) # shape: B, parallel_eq_n, *, oc
            out_sim = out_sim.transpose(-3, -2) # shape: B, *, parallel_eq_n, oc
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(-2), out_sim, self.w_config.similarity_config.metric) # shape: B, *, parallel_eq_n
            similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best weight interval and store in w_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        w_best_index = similarities.argmax(dim=0)
        self.w_config.interval[0] = weight_interval_candidates[0][w_best_index]

    def _search_best_w_interval_B(self, x, weight_interval_candidates):
        similarities = []
        for p_st in range(0, self.w_config.similarity_config.eq_n, self.w_config.similarity_config.parallel_eq_n):
            p_ed = min(self.w_config.similarity_config.eq_n, p_st + self.w_config.similarity_config.parallel_eq_n)
            cur_w_interval_dt = self.w_config.interval[0]
            cur_w_interval_B = weight_interval_candidates[1][p_st:p_ed].view(-1, 1, 1).cuda() # shape: parallel_eq_n, 1, 1
            cur_w_interval_C = self.w_config.interval[2]
            # quantize weight and bias
            oc, ic = self.weight.shape
            w_sim_dt = (self.weight[:self.dt_rank,:] / cur_w_interval_dt).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).mul_(cur_w_interval_dt).unsqueeze(0).expand(p_ed - p_st, -1, -1)
            w_sim_B = (self.weight[self.dt_rank:(self.dt_rank+self.D),:] / cur_w_interval_B).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).mul_(cur_w_interval_B) # shape: parallel_eq_n, oc, ic
            w_sim_C = (self.weight[(self.dt_rank+self.D):,:] / cur_w_interval_C).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).mul_(cur_w_interval_C).unsqueeze(0).expand(p_ed - p_st, -1, -1)
            w_sim = torch.cat((w_sim_dt,w_sim_B,w_sim_C), dim=1)
            w_sim = w_sim.view(-1, ic) # shape: parallel_eq_n * oc, ic
            bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
            # quantize input
            x_sim = self.quant_input(x) # shape: B, *, ic
            # quantize output
            out = F.linear(x_sim, w_sim, bias_sim) # shape: B, *, parallel_eq_n * oc
            out = torch.cat(torch.chunk(out.unsqueeze(-2), p_ed-p_st, dim=-1), dim=-2) # shape: B, *, parallel_eq_n, oc
            out = out.transpose(-3, -2) # shape: B, parallel_eq_n, *, oc
            out_sim = self.quant_output(out) # shape: B, parallel_eq_n, *, oc
            out_sim = out_sim.transpose(-3, -2) # shape: B, *, parallel_eq_n, oc
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(-2), out_sim, self.w_config.similarity_config.metric) # shape: B, *, parallel_eq_n
            similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best weight interval and store in w_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        w_best_index = similarities.argmax(dim=0)
        self.w_config.interval[1] = weight_interval_candidates[1][w_best_index]

    def _search_best_w_interval_C(self, x, weight_interval_candidates):
        similarities = []
        for p_st in range(0, self.w_config.similarity_config.eq_n, self.w_config.similarity_config.parallel_eq_n):
            p_ed = min(self.w_config.similarity_config.eq_n, p_st + self.w_config.similarity_config.parallel_eq_n)
            cur_w_interval_dt = self.w_config.interval[0]
            cur_w_interval_B = self.w_config.interval[1]
            cur_w_interval_C = weight_interval_candidates[2][p_st:p_ed].view(-1, 1, 1).cuda() # shape: parallel_eq_n, 1, 1
            # quantize weight and bias
            oc, ic = self.weight.shape
            w_sim_dt = (self.weight[:self.dt_rank,:] / cur_w_interval_dt).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).mul_(cur_w_interval_dt).unsqueeze(0).expand(p_ed - p_st, -1, -1)
            w_sim_B = (self.weight[self.dt_rank:(self.dt_rank+self.D),:] / cur_w_interval_B).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).mul_(cur_w_interval_B).unsqueeze(0).expand(p_ed - p_st, -1, -1)
            w_sim_C = (self.weight[(self.dt_rank+self.D):,:] / cur_w_interval_C).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1).mul_(cur_w_interval_C) # shape: parallel_eq_n, oc, ic
            w_sim = torch.cat((w_sim_dt,w_sim_B,w_sim_C), dim=1)
            w_sim = w_sim.view(-1, ic) # shape: parallel_eq_n * oc, ic
            bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
            # quantize input
            x_sim = self.quant_input(x) # shape: B, *, ic
            # quantize output
            out = F.linear(x_sim, w_sim, bias_sim) # shape: B, *, parallel_eq_n * oc
            out = torch.cat(torch.chunk(out.unsqueeze(-2), p_ed-p_st, dim=-1), dim=-2) # shape: B, *, parallel_eq_n, oc
            out = out.transpose(-3, -2) # shape: B, parallel_eq_n, *, oc
            out_sim = self.quant_output(out) # shape: B, parallel_eq_n, *, oc
            out_sim = out_sim.transpose(-3, -2) # shape: B, *, parallel_eq_n, oc
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(-2), out_sim, self.w_config.similarity_config.metric) # shape: B, *, parallel_eq_n
            similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best weight interval and store in w_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        w_best_index = similarities.argmax(dim=0)
        self.w_config.interval[2] = weight_interval_candidates[2][w_best_index]

    def _search_best_o_interval_dt(self, x, output_interval_candidates):
        similarities = []
        for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
            p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
            cur_o_interval_dt = output_interval_candidates[0][p_st:p_ed].unsqueeze(-1).cuda() # shape: parallel_eq_n, 1
            cur_o_interval_B = self.o_config.interval[1]
            cur_o_interval_C = self.o_config.interval[2]
            # quantize weight and bias 
            w_sim, bias_sim = self.quant_weight_bias() # shape: oc, ic
            # quantize input
            x_sim = self.quant_input(x) # shape: B, *, ic
            # quantize output
            out = F.linear(x_sim, w_sim, bias_sim) # shape: B, *, oc
            out = out.unsqueeze(-2) # shape: B, *, 1, oc
            out_sim_dt = (out[:, :, :self.dt_rank] / cur_o_interval_dt).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_dt) # shape: B, *, parallel_eq_n, oc
            out_sim_B = (out[:, :, self.dt_rank:(self.dt_rank+self.D)] / cur_o_interval_B).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_B).expand(-1, p_ed - p_st, -1)
            out_sim_C = (out[:, :, (self.dt_rank+self.D):] / cur_o_interval_C).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_C).expand(-1, p_ed - p_st, -1)
            out_sim = torch.cat((out_sim_dt, out_sim_B, out_sim_C), dim=-1)
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(-2), out_sim, self.o_config.similarity_config.metric) # shape: B, *, parallel_eq_n
            similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best output interval and store in o_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        o_best_index = similarities.argmax(dim=0)
        self.o_config.interval[0] = output_interval_candidates[0][o_best_index]

    def _search_best_o_interval_B(self, x, output_interval_candidates):
        similarities = []
        for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
            p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
            cur_o_interval_dt = self.o_config.interval[0]
            cur_o_interval_B = output_interval_candidates[1][p_st:p_ed].unsqueeze(-1).cuda() # shape: parallel_eq_n, 1
            cur_o_interval_C = self.o_config.interval[2]
            # quantize weight and bias 
            w_sim, bias_sim = self.quant_weight_bias() # shape: oc, ic
            # quantize input
            x_sim = self.quant_input(x) # shape: B, *, ic
            # quantize output
            out = F.linear(x_sim, w_sim, bias_sim) # shape: B, *, oc
            out = out.unsqueeze(-2) # shape: B, *, 1, oc
            out_sim_dt = (out[:, :, :self.dt_rank] / cur_o_interval_dt).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_dt).expand(-1, p_ed - p_st, -1)
            out_sim_B = (out[:, :, self.dt_rank:(self.dt_rank+self.D)] / cur_o_interval_B).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_B) # shape: B, *, parallel_eq_n, oc
            out_sim_C = (out[:, :, (self.dt_rank+self.D):] / cur_o_interval_C).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_C).expand(-1, p_ed - p_st, -1)
            out_sim = torch.cat((out_sim_dt, out_sim_B, out_sim_C), dim=-1)
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(-2), out_sim, self.o_config.similarity_config.metric) # shape: B, *, parallel_eq_n
            similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best output interval and store in o_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        o_best_index = similarities.argmax(dim=0)
        self.o_config.interval[1] = output_interval_candidates[1][o_best_index]

    def _search_best_o_interval_C(self, x, output_interval_candidates):
        similarities = []
        for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
            p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
            cur_o_interval_dt = self.o_config.interval[0]
            cur_o_interval_B = self.o_config.interval[1]
            cur_o_interval_C = output_interval_candidates[2][p_st:p_ed].unsqueeze(-1).cuda() # shape: parallel_eq_n, 1
            # quantize weight and bias 
            w_sim, bias_sim = self.quant_weight_bias() # shape: oc, ic
            # quantize input
            x_sim = self.quant_input(x) # shape: B, *, ic
            # quantize output
            out = F.linear(x_sim, w_sim, bias_sim) # shape: B, *, oc
            out = out.unsqueeze(-2) # shape: B, *, 1, oc
            out_sim_dt = (out[:, :, :self.dt_rank] / cur_o_interval_dt).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_dt).expand(-1, p_ed - p_st, -1)
            out_sim_B = (out[:, :, self.dt_rank:(self.dt_rank+self.D)] / cur_o_interval_B).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_B).expand(-1, p_ed - p_st, -1)
            out_sim_C = (out[:, :, (self.dt_rank+self.D):] / cur_o_interval_C).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1).mul_(cur_o_interval_C) # shape: B, *, parallel_eq_n, oc
            out_sim = torch.cat((out_sim_dt, out_sim_B, out_sim_C), dim=-1)
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(-2), out_sim, self.o_config.similarity_config.metric) # shape: B, *, parallel_eq_n
            similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best output interval and store in o_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        o_best_index = similarities.argmax(dim=0)
        self.o_config.interval[2] = output_interval_candidates[2][o_best_index]

    def calibration_step2(self, x):
        out = F.linear(x, self.weight, self.bias) # shape: B, *, oc

        self._initialize_intervals(x, out)

        self.raw_out = self.raw_out.to(x.device)
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None
        
        if self.w_config.k_scaled_config.k_scaled:
            pass
        else:
            weight_interval_candidates = torch.zeros(3, self.w_config.similarity_config.eq_n + 1) # For dt, B, C
            # print(self.w_config.interval[0].shape)
            weight_interval_candidates[0] = torch.tensor([self.w_config.similarity_config.eq_alpha + i * (self.w_config.similarity_config.eq_beta - self.w_config.similarity_config.eq_alpha) / self.w_config.similarity_config.eq_n for i in range(self.w_config.similarity_config.eq_n + 1)]).cuda() * self.w_config.interval[0] # shape: eq_n + 1
            weight_interval_candidates[1] = torch.tensor([self.w_config.similarity_config.eq_alpha + i * (self.w_config.similarity_config.eq_beta - self.w_config.similarity_config.eq_alpha) / self.w_config.similarity_config.eq_n for i in range(self.w_config.similarity_config.eq_n + 1)]).cuda() * self.w_config.interval[1] # shape: eq_n + 1
            weight_interval_candidates[2] = torch.tensor([self.w_config.similarity_config.eq_alpha + i * (self.w_config.similarity_config.eq_beta - self.w_config.similarity_config.eq_alpha) / self.w_config.similarity_config.eq_n for i in range(self.w_config.similarity_config.eq_n + 1)]).cuda() * self.w_config.interval[2] # shape: eq_n + 1

        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            input_interval_candidates = torch.tensor([self.i_config.similarity_config.eq_alpha + i * (self.i_config.similarity_config.eq_beta - self.i_config.similarity_config.eq_alpha) / self.i_config.similarity_config.eq_n for i in range(self.i_config.similarity_config.eq_n + 1)]).cuda() * self.i_config.interval

        if self.o_config.k_scaled_config.k_scaled:
            # output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval.unsqueeze(-1).cuda()
            pass
        else:
            output_interval_candidates = torch.zeros(3, self.w_config.similarity_config.eq_n + 1) # For dt, B, C
            output_interval_candidates[0] = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval[0]
            output_interval_candidates[1] = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval[1]
            output_interval_candidates[2] = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval[2]

        for _ in range(self.search_round):
            if self.w_config.qmode == "similarity":
                self._search_best_w_interval_dt(x, weight_interval_candidates)
                self._search_best_w_interval_B(x, weight_interval_candidates)
                self._search_best_w_interval_C(x, weight_interval_candidates)
            if self.i_config.qmode == "similarity":
                self._search_best_i_interval(x, input_interval_candidates)
            if self.o_config.qmode == "similarity":
                self._search_best_o_interval_dt(x, output_interval_candidates)
                self._search_best_o_interval_B(x, output_interval_candidates)
                self._search_best_o_interval_C(x, output_interval_candidates)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out = self._bias_correction_quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out