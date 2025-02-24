import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_config.QuantConfig import *
from kmeans import *
from rms_norm import *

class QuantNorm(RMSNorm):
    def __init__(
        self,
        hidden_size,
        eps = 1e-5,
        mode = "raw",
        bias_correction=False,
        search_round = 1,
        i_config = QuantConfig(),
        w_config = QuantConfig(),
        o_config = QuantConfig(),
    ):
        super().__init__(hidden_size, eps)

        self.n_calibration_step = 2
        self.mode = mode
        self.bias_correction = bias_correction

        self.search_round = search_round

        self.w_config = w_config
        self.i_config = i_config
        self.o_config = o_config

        self.metric = None
        self.next_nodes = []

        self.raw_input = None
        self.raw_out = None
        self.raw_grad = None

    def forward(self, x, residual=None):
        if self.mode == "raw":
            out = rms_norm(x, self.weight, self.bias, residual, self.eps)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x, residual)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x, residual)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x, residual)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self, x, residual=None):
        assert self.calibrated is not None, f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = rms_norm(x_sim, w_sim, bias_sim, residual, self.eps)
        out_sim = self.quant_output(out)
        return out_sim
    
    def _bias_correction_quant_forward(self, x, residual=None):
        if self.bias_correction and self.bias != None:
            w_sim = self.quant_weight_bias()[0]
            x_sim = self.quant_input(x)
            eps = rms_norm(x_sim, w_sim - self.weight.data, None, residual, self.eps)
            eps = torch.mean(eps, dim=(list(range(len(eps.shape) - 1))), keepdim=False)
            self.bias -= eps
            self.bias_correction = False
        return self.quant_forward(x, residual)
    
    def quant_weight_bias(self):
        if not self.w_config.quant:
            return self.weight, self.bias
        if self.w_config.bin == "uniform":
            cur_w_interval = self.w_config.interval
            cur_w_zeropoint = self.w_config.zeropoint
            if self.w_config.k_scaled_config.k_scaled:
                cur_w_interval = torch.tensor([self.w_config.interval[i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
                cur_w_zeropoint = torch.tensor([self.w_config.zeropoint[i] for i in self.w_config.k_scaled_config.k_scaled_clusters]).to(self.weight.device)
            w_sim = ((self.weight - cur_w_zeropoint) / (cur_w_interval + 1e-15)).round_().clamp_(-self.w_config.qmax, self.w_config.qmax - 1)
            w_sim.mul_(cur_w_interval)
            w_sim.add_(cur_w_zeropoint)
            return w_sim, self.bias
        elif self.w_config.bin == "power_of_two":
            pass  # TODO

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
                    pass
                elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
                    out = out.transpose(-2, -1)
                cur_o_interval = torch.tensor([self.o_config.interval[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
                cur_o_zeropoint = torch.tensor([self.o_config.zeropoint[i] for i in self.o_config.k_scaled_config.k_scaled_clusters]).to(out.device)
            out_sim = ((out - cur_o_zeropoint) / (cur_o_interval + 1e-15)).round_().clamp_(-self.o_config.qmax, self.o_config.qmax - 1)
            out_sim.mul_(cur_o_interval)
            out_sim.add_(cur_o_zeropoint)
            if self.o_config.k_scaled_config.k_scaled & (self.o_config.k_scaled_config.k_scaled_mode == "token_wise"):
                out_sim = out_sim.transpose(-2, -1)
            return out_sim
        elif self.o_config.bin == "power_of_two":
            pass  # TODO

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
            # if self.o_config.k_scaled_config.k_scaled_mode == "channel_wise":
            #     pass
            # elif self.o_config.k_scaled_config.k_scaled_mode == "token_wise":
            #     out = out.transpose(-2, -1)
            # self.o_config.k_scaled_config.k_scaled_clusters = kmeans(out, self.o_config.k_scaled_config.k, num_iters=100, dim=-1)  # shape: D or L
            # self.o_config.interval = torch.zeros(self.o_config.k_scaled_config.k)  # shape: K
            # self.o_config.zeropoint = torch.zeros(self.o_config.k_scaled_config.k)  # shape: K
            # for cluster_index in range(self.o_config.k_scaled_config.k):
            #     self.o_config.interval[cluster_index] = ((out * torch.eq(self.o_config.k_scaled_config.k_scaled_clusters, cluster_index)).abs().max() / (self.o_config.qmax - 0.5)).detach()  # shape: K
            pass
        else:
            self.o_config.interval = (out.abs().max() / (self.o_config.qmax - 0.5)).detach()

    def _search_best_w_interval(self, x, weight_interval_candidates):
        pass

    def _search_best_i_interval(self, x, input_interval_candidates):
        pass

    def _search_best_o_interval(self, x, output_interval_candidates):
        pass

    def calibration_step1(self, x, residual=None):
        out = rms_norm(x, self.weight, self.bias, residual, self.eps)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out
    
    def calibration_step2(self, x, residual=None):
        out = rms_norm(x, self.weight, self.bias, residual, self.eps)
        
        self._initialize_intervals(x, out)

        self.raw_out = self.raw_out.to(x.device)
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad is not None else None

        if self.w_config.k_scaled_config.k_scaled:
            pass
        else:
            weight_interval_candidates = torch.tensor([self.w_config.similarity_config.eq_alpha + i * (self.w_config.similarity_config.eq_beta - self.w_config.similarity_config.eq_alpha) / self.w_config.similarity_config.eq_n for i in range(self.w_config.similarity_config.eq_n + 1)]).cuda() * self.w_config.interval

        if self.i_config.k_scaled_config.k_scaled:
            pass
        else:
            input_interval_candidates = torch.tensor([self.i_config.similarity_config.eq_alpha + i * (self.i_config.similarity_config.eq_beta - self.i_config.similarity_config.eq_alpha) / self.i_config.similarity_config.eq_n for i in range(self.i_config.similarity_config.eq_n + 1)]).cuda() * self.i_config.interval

        if self.o_config.k_scaled_config.k_scaled:
            pass
        else:
            output_interval_candidates = torch.tensor([self.o_config.similarity_config.eq_alpha + i * (self.o_config.similarity_config.eq_beta - self.o_config.similarity_config.eq_alpha) / self.o_config.similarity_config.eq_n for i in range(self.o_config.similarity_config.eq_n + 1)]).cuda() * self.o_config.interval

        for _ in range(self.search_round):
            if self.w_config.qmode == "similarity":
                self._search_best_w_interval(x, weight_interval_candidates)
            if self.i_config.qmode == "similarity":
                self._search_best_i_interval(x, input_interval_candidates)
            if self.o_config.qmode == "similarity":
                self._search_best_o_interval(x, output_interval_candidates)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad is not None else None

        self.calibrated = True
        out = self._bias_correction_quant_forward(x, residual)
        del self.raw_input, self.raw_out, self.raw_grad
        return out          
