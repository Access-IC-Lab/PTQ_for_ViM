import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_layers.QuantLinear import *
from kmeans import *


class QuantConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',

        mode = "raw",
        bias_bit = None,
        bias_correction = False,

        search_round = 1,

        w_config = QuantConfig(),
        i_config = QuantConfig(),
        o_config = QuantConfig(),

        L = 197,
    ):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)

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
    

    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.numel() // 16777216
            return torch.quantile(tensor.view(-1)[:16777216*n].view(n,16777216),quantile,1).mean()
        else:
            return torch.quantile(tensor,quantile)


    # def _kmeans(self, tensor, k, num_iters=100):
    #     """
    #     Example: 

    #     If a = torch.tensor([100, 1, 2, 3, 200, 110, 210]) and self.k = 3

    #     Then _kmeans(a) = [1, 0, 0, 0, 2, 1, 2]
    #     """

    #     indices = torch.randperm(tensor.size(1))[:k]
    #     tensor = tensor.abs().mean([0, 2])
    #     centroids = tensor[indices]

    #     for _ in range(num_iters):
    #         distances = torch.abs(tensor.unsqueeze(-1) - centroids.unsqueeze(0))  # D,K
    #         labels = torch.argmin(distances, dim=-1) # D
    #         new_centroids = torch.stack([tensor[labels == i].mean() for i in range(k)])

    #         if torch.allclose(centroids, new_centroids, rtol=1e-4):
    #             break

    #         centroids = new_centroids

    #     sorted_centroids, sorted_indices = torch.sort(centroids)

    #     new_labels = torch.zeros_like(labels)
    #     for new_idx, old_idx in enumerate(sorted_indices):
    #         new_labels[labels == old_idx] = new_idx

    #     return new_labels
        

    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        out_sim = self.quant_output(out)
        return out_sim

    
    # TODO
    def quant_weight_bias(self):
        if not self.w_config.quant:
            return self.weight, self.bias
        if self.w_config.bin == "uniform":
            cur_w_interval = self.w_config.interval
            cur_w_zeropoint = self.w_config.zeropoint
            if self.w_config.k_scaled_config.k_scaled:
                cur_w_interval = torch.tensor([self.w_config.interval[i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                cur_w_zeropoint = torch.tensor([self.w_config.zeropoint[i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
            w_sim = ((self.weight - cur_w_zeropoint) / (cur_w_interval + 1e-7)).round_().clamp_(-self.w_config.qmax,self.w_config.qmax-1)
            w_sim.mul_(cur_w_interval)
            w_sim.add_(cur_w_zeropoint)
            return w_sim, self.bias
        elif self.w_config.bin == "power_of_two":
            pass # TODO


    def quant_input(self, x):
        if not self.i_config.quant:
            return x
        if self.i_config.bin == "uniform":
            cur_i_interval = self.i_config.interval
            cur_i_zeropoint = self.i_config.zeropoint
            if self.i_config.k_scaled_config.k_scaled:
                cur_i_interval = torch.tensor([self.i_config.interval[i] for i in self.i_config.k_scaled_config.k_scaled_clusters]).to(x.device)
                cur_i_zeropoint = torch.tensor([self.i_config.zeropoint[i] for i in self.i_config.k_scaled_config.k_scaled_clusters]).to(x.device)
            x_sim=((x - cur_i_zeropoint) / (cur_i_interval + 1e-7)).round_().clamp_(-self.i_config.qmax,self.i_config.qmax-1)
            x_sim.mul_(cur_i_interval)
            x_sim.add_(cur_i_zeropoint)
            return x_sim
        elif self.i_config.bin == "power_of_two":
            pass # TODO
    

    def quant_output(self,out):
        if not self.o_config.quant:
            return out
        if self.o_config.bin == "uniform":
            cur_o_interval = self.o_config.interval
            cur_o_zeropoint = self.o_config.zeropoint
            if self.o_config.k_scaled_config.k_scaled:
                cur_o_interval = torch.tensor([self.o_config.interval[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).unsqueeze(-1).to(out.device)
                cur_o_zeropoint = torch.tensor([self.o_config.zeropoint[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).unsqueeze(-1).to(out.device)
            out_sim=((out - cur_o_zeropoint) / (cur_o_interval + 1e-7)).round_().clamp_(-self.o_config.qmax,self.o_config.qmax-1)
            out_sim.mul_(cur_o_interval)
            out_sim.add_(cur_o_zeropoint)
            return out_sim
        elif self.o_config.bin == "power_of_two":
            pass # TODO


    def _initialize_intervals(self, x, out):
        if self.w_config.k_scaled_config.k_scaled:
            pass
        else:
            self.w_config.interval = ((self.weight.data.abs().max()) / (self.w_config.qmax - 0.5)).detach()

        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            self.i_config.interval = (x.abs().max() / (self.i_config.qmax - 0.5)).detach()

        if self.o_config.k_scaled_config.k_scaled:
            # self.o_config.k_scaled_config.k_scaled_clusters = self._kmeans(out, self.o_config.k_scaled_config.k, num_iters=100) # shape: D or L
            self.o_config.k_scaled_config.k_scaled_clusters = kmeans(out, self.o_config.k_scaled_config.k, num_iters=100, dim=1) # shape: D or L
            self.o_config.interval = torch.zeros(self.o_config.k_scaled_config.k) # shape: K
            self.o_config.zeropoint = torch.zeros(self.o_config.k_scaled_config.k) # shape: K
            for cluster_index in range(self.o_config.k_scaled_config.k):
                self.o_config.interval[cluster_index] = ((out * torch.eq(self.o_config.k_scaled_config.k_scaled_clusters, cluster_index).unsqueeze(-1)).abs().max() / (self.o_config.qmax - 0.5)).detach() # shape: K

            # if self.o_config.k_scaled_config.k_scaled_power_of_two_scaling:
            #     for cluster_index in range(1, self.o_config.k_scaled_config.k):
            #         self.o_config.interval[cluster_index] = self.o_config.interval[0] * torch.pow(2, torch.log2(self.o_config.interval[cluster_index] / self.o_config.interval[0]).round())
        else:
            self.o_config.interval = (out.abs().max() / (self.o_config.qmax - 0.5)).detach() # shape: 1

    
    def _search_best_w_interval(self, x, weight_interval_candidates):
        similarities = []
        for p_st in range(0, self.w_config.similarity_config.eq_n, self.w_config.similarity_config.parallel_eq_n):
            p_ed = min(self.w_config.similarity_config.eq_n, p_st + self.w_config.similarity_config.parallel_eq_n)
            cur_w_interval = weight_interval_candidates[p_st:p_ed].view(-1, 1, 1, 1) # shape: parallel_eq_n, 1, 1, 1
            # quantize weight and bias
            oc, ic, kl = self.weight.shape
            w_sim = self.weight.unsqueeze(0) # shape: 1, oc, ic, kl
            w_sim = (self.weight/cur_w_interval).round_().clamp_(-self.w_config.qmax, self.w_config.qmax-1) * cur_w_interval # shape: parallel_eq_n, oc, ic, kl
            w_sim = w_sim.view(-1, ic, kl) # shape: parallel_eq_n * oc, ic, kl
            bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
            # quantize input
            x_sim = self.quant_input(x) # shape: B, ic, il
            # quantize output
            out = F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: B, parallel_eq_n * oc, fl
            out = torch.cat(torch.chunk(out.unsqueeze(2), chunks=oc, dim=1), dim=2) # shape: B, parallel_eq_n, oc, fl
            out_sim = self.quant_output(out) # shape: B, parallel_eq_n, oc, fl
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(1), out_sim, self.w_config.similarity_config.metric, dim=2) # shape: B, parallel_eq_n, fl
            similarity = torch.mean(similarity, dim=[0,2]) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best weight interval and store in w_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        w_best_index = similarities.argmax(dim=0)
        self.w_config.interval = weight_interval_candidates[w_best_index]


    def _search_best_i_interval(self, x, input_interval_candidates):
        similarities = []
        for p_st in range(0, self.i_config.similarity_config.eq_n, self.i_config.similarity_config.parallel_eq_n):
            p_ed = min(self.i_config.similarity_config.eq_n, p_st + self.i_config.similarity_config.parallel_eq_n)
            cur_i_interval = input_interval_candidates[p_st:p_ed].view(-1, 1, 1, 1) # shape: parallel_eq_n, 1, 1, 1
            # quantize weight and bias
            w_sim, bias_sim = self.quant_weight_bias()
            # quantize input
            B, ic, il = x.shape
            x_sim = x.unsqueeze(0) # shape: 1, B, ic, il
            x_sim = (x_sim / cur_i_interval).round_().clamp_(-self.i_config.qmax, self.i_config.qmax-1) * cur_i_interval # shape: parallel_eq_n, B, ic, il
            x_sim = x_sim.view(-1, ic, il) # shape: parallel_eq_n * B, ic, il
            # quantize output
            out = F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: parallel_eq_n * B, oc, fl
            out_sim = self.quant_output(out) # shape: parallel_eq_n, B, oc, fl
            # calculate similarity and store them
            similarity = self._get_similarity(self.raw_out.unsqueeze(0), out_sim, self.i_config.similarity_config.metric, dim=2) # shape: parallel_eq_n, B, fl
            similarity = torch.mean(similarity, dim=[1, 2]) # shape: parallel_eq_n
            similarities.append(similarity)
        # store best input interval and store in i_interval
        similarities = torch.cat(similarities, dim=0) # shape: eq_n
        i_best_index = similarities.argmax(dim=0)
        self.i_config.interval = input_interval_candidates[i_best_index]


    def _search_best_o_interval(self, x, output_interval_candidates):
        if self.o_config.k_scaled_config.k_scaled:
            for cluster_index in range(self.o_config.k_scaled_config.k):
                similarities = []
                for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
                    p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
                    cur_o_interval = torch.where(self.o_config.k_scaled_config.k_scaled_clusters == cluster_index, output_interval_candidates[cluster_index][p_st:p_ed].unsqueeze(-1), self.o_config.interval.cuda()[self.o_config.k_scaled_config.k_scaled_clusters]) # shape: parallel_eq_n, oc
                    cur_o_interval = cur_o_interval.unsqueeze(1).unsqueeze(-1) # shape: parallel_eq_n, 1, oc, 1
                    # quantize weight and bias
                    w_sim, bias_sim = self.quant_weight_bias()
                    # quantize input
                    x_sim = self.quant_input(x) # shape: B, ic, il
                    # quantize output
                    out = F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: B, oc, fl
                    out = out.unsqueeze(0) # shape: 1, B, oc, fl
                    out_sim = (out / cur_o_interval).round_().clamp_(-self.o_config.qmax, self.o_config.qmax-1) * cur_o_interval # shape: parallel_eq_n, B, oc, fl
                    # calculate similarity and store them
                    similarity = self._get_similarity(self.raw_out.unsqueeze(0), out_sim, self.o_config.similarity_config.metric, dim=2) # shape: parallel_eq_n, B, fl
                    similarity = torch.mean(similarity, dim=[1, 2]) # shape: parallel_eq_n
                    similarities.append(similarity)
                # store best output interval and store in o_interval
                similarities = torch.cat(similarities, dim=0)
                o_best_index = similarities.argmax(dim=0)
                self.o_config.interval[cluster_index] = output_interval_candidates[cluster_index][o_best_index]
        else:
            similarities = []
            for p_st in range(0, self.o_config.similarity_config.eq_n, self.o_config.similarity_config.parallel_eq_n):
                p_ed = min(self.o_config.similarity_config.eq_n, p_st + self.o_config.similarity_config.parallel_eq_n)
                cur_o_interval = output_interval_candidates[p_st:p_ed].view(-1, 1, 1, 1) # shape: parallel_eq_n, 1, 1, 1
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim = self.quant_input(x) # shape: B, ic, il
                # quantize output
                out = F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: B, oc, fl
                out = out.unsqueeze(0) # shape: 1, B, oc, fl
                out_sim = (out / cur_o_interval).round_().clamp_(-self.o_config.qmax, self.o_config.qmax - 1) * cur_o_interval # shape: parallel_eq_n, B, oc, fl
                # calculate similarity and store them
                similarity = self._get_similarity(self.raw_out.unsqueeze(0), out_sim, self.o_config.similarity_config.metric, dim=2) # shape: parallel_eq_n, B, fl
                similarity = torch.mean(similarity, dim=[1, 2]) # shape: parallel_eq_n
                similarities.append(similarity)
            # store best output interval and store in o_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n
            o_best_index = similarities.argmax(dim=0)
            self.o_config.interval = output_interval_candidates[o_best_index]
    
    
    def calibration_step1(self,x):
        # step1: collection the FP32 values
        out = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out


    # TODO
    def calibration_step2(self, x): # TODO
        # step2: search for the best S^w and S^o of each layer
        out = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self._initialize_intervals(x, out)

        self.raw_out = self.raw_out.to(x.device) # shape: B, oc, fl
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None
        
        if self.w_config.k_scaled_config.k_scaled:
            pass
        else:
            weight_interval_candidates = torch.tensor([self.w_config.similarity_config.eq_alpha + i * (self.w_config.similarity_config.eq_beta - self.w_config.similarity_config.eq_alpha) / self.w_config.similarity_config.eq_n for i in range(self.w_config.similarity_config.eq_n + 1)]).cuda() * self.w_config.interval # shape: eq_n + 1

        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            input_interval_candidates = torch.tensor([self.i_config.similarity_config.eq_alpha + i * (self.i_config.similarity_config.eq_beta - self.i_config.similarity_config.eq_alpha) / self.i_config.similarity_config.eq_n for i in range(self.i_config.similarity_config.eq_n + 1)]).cuda() * self.i_config.interval

        if self.o_config.k_scaled_config.k_scaled:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval.unsqueeze(-1).cuda()
        else:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval

        for _ in range(self.search_round):
            if self.w_config.qmode == "similarity":
                self._search_best_w_interval(x, weight_interval_candidates)
            if self.i_config.qmode == "similarity":
                self._search_best_i_interval(x, input_interval_candidates)
            if self.o_config.qmode == "similarity":
                self._search_best_o_interval(x, output_interval_candidates)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out = self.quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out
    