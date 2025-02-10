import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_layers.QuantLinear import *

class QuantOutProj(QuantLinear):
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
        # dt_rank = 12,
        # D = 16,
    ):
        super().__init__(
            in_features = in_features, 
            out_features = out_features, 
            bias = bias,

            mode = mode,
            bias_bit = bias_bit,
            bias_correction = bias_correction,

            search_round = search_round,

            w_config = w_config,
            i_config = i_config,
            o_config = o_config,

            L = 197
        ) 
            
        self.n_calibration_step = 2

        self.metric = None
        self.next_nodes = []

        self.raw_input = None
        self.raw_out = None
        self.raw_grad = None

        # self.dt_rank = dt_rank
        # self.D = D

    def quant_forward(self, x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, w_sim, bias_sim)
        out_sim = self.quant_output(out)
        # print(out_sim.shape)
        out_sim = out_sim * self.r_C.to(out_sim.device)
        return out_sim
    
    def calibration_step2(self, x):
        # step2: search for the best S^w and S^o of each layer
        out = F.linear(x, self.weight, self.bias) # shape: B, *, oc

        # print(f"weight:{self.weight.shape}")
        # print(f"output:{out.shape}")
        self.r_C = out.abs().amax((0, 1))
        self.weight.data = self.weight.data / self.r_C.unsqueeze(-1).to(self.weight.data.device)
        out = F.linear(x, self.weight, self.bias)

        self._initialize_intervals(x, out)

        self.raw_out = self.raw_out.to(x.device)
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None
        
        if self.w_config.k_scaled_config.k_scaled:
            pass
        else:
            # print(self.w_config.interval.shape)
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
        out = self._bias_correction_quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out
        