import torch
import torch.nn as nn
import torch.nn.functional as F

class MinMaxQuantLinear(nn.Linear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False):
        super().__init__(in_features,out_features,bias)
        self.n_calibration_step=2
        self.mode = mode
        self.w_bit = w_bit
        self.i_bit = i_bit
        self.o_bit = o_bit
        self.bias_bit=bias_bit
        assert bias_bit is None,"No support bias bit now"
        self.w_interval=None
        self.i_interval=None
        self.o_interval=None
        self.raw_input=None
        self.raw_out=None
        self.metric=None
        self.next_nodes=[]
        self.w_qmax=2**(self.w_bit-1)
        self.i_qmax=2**(self.i_bit-1)
        self.o_qmax=2**(self.o_bit-1)
        self.bias_correction = bias_correction

    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(x)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_weight_bias(self):
        w=(self.weight/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim=w.mul_(self.w_interval)
        if self.bias is not None:
            return w_sim,self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim,None
    
    def quant_input(self, x):
        x_sim=(x/self.i_interval).round_().clamp_(-self.i_qmax,self.i_qmax-1)
        x_sim.mul_(self.i_interval)
        return x_sim
    
    def quant_output(self,out):
        out_sim=(out/self.o_interval).round_().clamp_(-self.o_qmax,self.o_qmax-1)
        out_sim.mul_(self.o_interval)
        return out_sim
    
    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        out_sim=self.quant_output(out)
        return out_sim
    
    def _bias_correction_quant_forward(self, x):
        if self.bias_correction and self.bias != None:
            w_sim = self.quant_weight_bias()[0]
            x_sim = self.quant_input(x)
            eps = F.linear(x_sim, w_sim-self.weight.data, None)
            eps = torch.mean(eps, dim=(list(range(len(eps.shape)-1))), keepdim=False)
            self.bias -= eps
            self.bias_correction = False
        return self.quant_forward(x)

    def calibration_step1(self,x):
        # step1: collection the FP32 values
        out=F.linear(x, self.weight, self.bias)
        self.raw_input=x.cpu().detach()
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval=(out.abs().max()/(self.o_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out
    
class QuantileQuantLinear(MinMaxQuantLinear):
    """
    Quantile quantize weight and output
    """
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False,
        w_quantile=0.9999,i_quantile=0.9999,o_quantile=0.9999):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)
        self.w_quantile = w_quantile
        self.i_quantile = i_quantile
        self.o_quantile = o_quantile

    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.numel()//16777216
            return torch.quantile(tensor.view(-1)[:16777216*n].view(n,16777216),quantile,1).mean()
        else:
            return torch.quantile(tensor,quantile)

    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        self.w_interval=(self._quantile(self.weight.data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.i_interval=(self._quantile(x.abs(),self.i_quantile)/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval=(self._quantile(out.abs(),self.o_quantile)/(self.o_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)      
        return out
    
class AsymmetricMinMaxLinear(MinMaxQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)

    def quant_input(self, x):
        x_sim=((x-self.i_zeropoint)/self.i_interval).round_().clamp_(-self.i_qmax,self.i_qmax-1)
        x_sim.mul_(self.i_interval)
        x_sim.add_(self.i_zeropoint)
        return x_sim
    
    def quant_output(self,out):
        out_sim=((out-self.o_zeropoint)/self.o_interval).round_().clamp_(-self.o_qmax,self.o_qmax-1)
        out_sim.mul_(self.o_interval)
        out_sim.add_(self.o_zeropoint)
        return out_sim

    def calibration_step2(self,x):
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=((x.max()-x.min())/(2*self.i_qmax-1)).detach()
        self.i_zeropoint=((x.max()+x.min())/2).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval=((out.max()-out.min())/(2*self.o_qmax-1)).detach()
        self.o_zeropoint=(out.max()+out.min())/2
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out
    
class AsymmetricQuantileQuantLinear(QuantileQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False,
        w_quantile=0.9999,i_quantile=0.9999,o_quantile=0.9999):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction,w_quantile,i_quantile,o_quantile)

    def quant_input(self,x):
        x_sim=((x-self.i_zeropoint)/self.i_interval).round_().clamp_(-self.i_qmax,self.i_qmax-1)
        x_sim.mul_(self.i_interval)
        x_sim.add_(self.i_zeropoint)
        return x_sim
    
    def quant_output(self,out):
        out_sim=((out-self.o_zeropoint)/self.o_interval).round_().clamp_(-self.o_qmax,self.o_qmax-1)
        out_sim.mul_(self.o_interval)
        out_sim.add_(self.o_zeropoint)
        return out_sim
    
    def calibration_step2(self,x):
        self.w_interval=(self._quantile(self.weight.data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.i_interval=((self._quantile(x,self.i_quantile)-self._quantile(x,(1-self.i_quantile)))/(2*self.i_qmax-1)).detach()
        self.i_zeropoint=((self._quantile(x,self.i_quantile)+self._quantile(x,(1-self.i_quantile)))/2).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval=((self._quantile(out,self.o_quantile)-self._quantile(out,(1-self.o_quantile)))/(2*self.o_qmax-1)).detach()
        self.o_zeropoint=((self._quantile(out,self.o_quantile)+self._quantile(out,(1-self.o_quantile)))/2).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out
    
class MinMaxQuantInProj(MinMaxQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False,
        d_inner = 384):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)
        self.d_inner = d_inner
        
    def quant_weight_bias(self):
        # w=(self.weight/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        # w_sim=w.mul_(self.w_interval)
        w_sim_x=(self.weight[:self.d_inner,:]/self.w_interval_x).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval_x)
        w_sim_z=(self.weight[self.d_inner:,:]/self.w_interval_z).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval_z)
        w_sim=torch.cat((w_sim_x,w_sim_z),dim=0)
        if self.bias is not None:
            return w_sim,self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim,None
    
    def quant_output(self,out):
        out_sim_x=(out[:,:self.d_inner]/self.o_interval_x).round_().clamp_(-self.o_qmax,self.o_qmax-1).mul_(self.o_interval_x)
        out_sim_z=(out[:,self.d_inner:]/self.o_interval_z).round_().clamp_(-self.o_qmax,self.o_qmax-1).mul_(self.o_interval_z)
        out_sim=torch.cat((out_sim_x,out_sim_z),dim=1)
        return out_sim
    
    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        # self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.w_interval_x=(self.weight[:self.d_inner,:].data.abs().max()/(self.w_qmax-0.5)).detach()
        self.w_interval_z=(self.weight[self.d_inner:,:].data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval_x=(out[:,:self.d_inner].abs().max()/(self.o_qmax-0.5)).detach()
        self.o_interval_z=(out[:,self.d_inner:].abs().max()/(self.o_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out
    
class MinMaxQuantXProj(MinMaxQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False,
        dt_rank = 12,
        d_state = 16):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)
        self.dt_rank = dt_rank
        self.d_state = d_state

    def quant_weight_bias(self):
        # w=(self.weight/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        # w_sim=w.mul_(self.w_interval)
        w_sim_dt=(self.weight[:self.dt_rank,:]/self.w_interval_dt).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval_dt)
        w_sim_b=(self.weight[self.dt_rank:(self.dt_rank+self.d_state),:]/self.w_interval_b).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval_b)
        w_sim_c=(self.weight[(self.dt_rank+self.d_state):,:]/self.w_interval_c).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(self.w_interval_c)
        w_sim=torch.cat((w_sim_dt,w_sim_b,w_sim_c),dim=0)
        if self.bias is not None:
            return w_sim,self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim,None
        
    def quant_output(self, out):
        out_sim_dt=(out[:,:self.dt_rank]/self.o_interval_dt).round_().clamp_(-self.o_qmax,self.o_qmax-1).mul_(self.o_interval_dt)
        out_sim_b=(out[:,self.dt_rank:(self.dt_rank+self.d_state)]/self.o_interval_b).round_().clamp_(-self.o_qmax,self.o_qmax-1).mul_(self.o_interval_b)
        out_sim_c=(out[:,(self.dt_rank+self.d_state):]/self.o_interval_c).round_().clamp_(-self.o_qmax,self.o_qmax-1).mul_(self.o_interval_c)
        out_sim=torch.cat((out_sim_dt,out_sim_b,out_sim_c),dim=1)
        return out_sim
    
    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        # self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.w_interval_dt=(self.weight[:self.dt_rank,:].data.abs().max()/(self.w_qmax-0.5)).detach()
        self.w_interval_b=(self.weight[self.dt_rank:(self.dt_rank+self.d_state),:].data.abs().max()/(self.w_qmax-0.5)).detach()
        self.w_interval_c=(self.weight[(self.dt_rank+self.d_state):,:].data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval_dt=(out[:,:self.dt_rank].abs().max()/(self.o_qmax-0.5)).detach()
        self.o_interval_b=(out[:,self.dt_rank:(self.dt_rank+self.d_state)].abs().max()/(self.o_qmax-0.5)).detach()
        self.o_interval_c=(x[:,(self.dt_rank+self.d_state):].abs().max()/(self.o_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out
    
class QuantileQuantInProj(MinMaxQuantInProj):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False,
        d_inner = 384,
        w_quantile=0.9999,i_quantile=0.9999,o_quantile=0.9999):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction,d_inner)
        self.w_quantile = w_quantile
        self.i_quantile = i_quantile
        self.o_quantile = o_quantile

    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.numel()//16777216
            return torch.quantile(tensor.view(-1)[:16777216*n].view(n,16777216),quantile,1).mean()
        else:
            return torch.quantile(tensor,quantile)
    
    def calibration_step2(self,x):
        self.w_interval_x=(self._quantile(self.weight[:self.d_inner,:].data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.w_interval_z=(self._quantile(self.weight[self.d_inner:,:].data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.i_interval=(self._quantile(x.abs(),self.i_quantile)/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval_x=(self._quantile(out[:,:self.d_inner].abs(),self.o_quantile)/(self.o_qmax-0.5)).detach()
        self.o_interval_z=(self._quantile(out[:,self.d_inner:].abs(),self.o_quantile)/(self.o_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out

class QuantileQuantXProj(MinMaxQuantXProj):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False,
        dt_rank = 12,
        d_state = 16,
        w_quantile=0.9999,i_quantile=0.9999,o_quantile=0.9999):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction,dt_rank,d_state)
        self.w_quantile = w_quantile
        self.i_quantile = i_quantile
        self.o_quantile = o_quantile

    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.numel()//16777216
            return torch.quantile(tensor.view(-1)[:16777216*n].view(n,16777216),quantile,1).mean()
        else:
            return torch.quantile(tensor,quantile)

    def calibration_step2(self,x):
        self.w_interval_dt=(self._quantile(self.weight[:self.dt_rank,:].data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.w_interval_b=(self._quantile(self.weight[self.dt_rank:(self.dt_rank+self.d_state),:].data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.w_interval_c=(self._quantile(self.weight[(self.dt_rank+self.d_state):,:].data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.i_interval=(self._quantile(x.abs(),self.i_quantile)/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_interval_dt=(self._quantile(out[:,:self.dt_rank].abs(),self.o_quantile)/(self.o_qmax-0.5)).detach()
        self.o_interval_b=(self._quantile(out[:,self.dt_rank:(self.dt_rank+self.d_state)].abs(),self.o_quantile)/(self.o_qmax-0.5)).detach()
        self.o_interval_c=(self._quantile(out[:,(self.dt_rank+self.d_state):].abs(),self.o_quantile)/(self.o_qmax-0.5)).detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out

class PTQSLQuantLinear(MinMaxQuantLinear):
    """
    PTQSL on linear modules.
    """
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_i=1, n_o=1, init_layerwise=False):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, i_bit=i_bit, o_bit=o_bit, bias_bit=bias_bit, bias_correction=bias_correction)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.n_H = n_H
        self.n_V = n_V
        self.n_i = n_i
        self.n_o = n_o
        self.crb_rows = out_features // n_V
        self.crb_cols = in_features // n_H # ignore remnent != 0 situations
        self.crb_ins = in_features // n_i
        self.crb_outs = out_features // n_o
        self.parallel_eq_n = parallel_eq_n
        self.init_layerwise = init_layerwise
        self.raw_grad = None

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
    
    def quant_weight_bias(self):
        # self.w_interval shape: n_V, 1, n_H, 1
        w=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)/self.w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1)
        w_sim=w.mul_(self.w_interval).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim,self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim,None
    
    def quant_input(self, x):
        # x shape: B,*,ic
        # self.i_interval shape: n_i,1
        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_i, dim=-1), dim=-2)
        x_sim=(x_sim.div_(self.i_interval)).round_().clamp_(-self.i_qmax,self.i_qmax-1)
        x_sim = x_sim.mul_(self.i_interval).reshape_as(x)
        return x_sim
    
    def quant_output(self, out):
        # out shape: B,*,oc
        # self.o_interval shape: n_o,1
        out_sim=torch.cat(torch.chunk(out.unsqueeze(-2), chunks=self.n_o, dim=-1), dim=-2) # shape: B,*,n_o,crb_outs
        out_sim=(out_sim.div_(self.o_interval)).round_().clamp_(-self.o_qmax,self.o_qmax-1) # shape: B,*,n_o,crb_outs
        out_sim = out_sim.mul_(self.o_interval).reshape_as(out) # shape: B,*,oc
        return out_sim

    def _search_best_w_interval(self, x, weight_interval_candidates, raw_out_expanded_chunked):
        """
        Modularization of searching best weight intervals
        """
        tmp_w_interval = self.w_interval.unsqueeze(0) # shape: 1,n_V,1,n_H,1
        for h in range(self.n_H):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                cur_w_interval[:,:,:,h:h+1,:] = weight_interval_candidates[p_st:p_ed,:,:,h:h+1,:]
                # quantize weight and bias 
                w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                w_sim = (w_sim/cur_w_interval).round_().clamp_(-self.w_qmax,self.w_qmax-1).mul_(cur_w_interval) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                w_sim = w_sim.view(-1,self.in_features) # shape: parallel_eq_n*oc,ic
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                # quantize input
                x_sim = self.quant_input(x)
                # quantize output
                out = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n*oc
                out = torch.cat(torch.chunk(out.unsqueeze(-2), chunks=p_ed-p_st, dim=-1), dim=-2) # shape: B,*,parallel_eq_n,oc
                out_sim = self.quant_output(out)
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,parallel_eq_n,n_V,crb_rows
                # calculate similarity and store them
                similarity = self._get_similarity(raw_out_expanded_chunked, out_sim, self.metric) # shape: B,*,parallel_eq_n,n_V
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-2))) # shape: parallel_eq_n, n_V
                similarities.append(similarity)
            # store best weight interval of h into tmp_w_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n, n_V
            h_best_index = similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
            tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(weight_interval_candidates[:,:,:,h:h+1,:],dim=0,index=h_best_index)
        self.w_interval = tmp_w_interval.squeeze(dim=0)
    
    def _search_best_i_interval(self, x, input_interval_candidates, raw_out_expanded):
        tmp_i_interval = self.i_interval.unsqueeze(-1) # shape: n_i,1,1
        for i in range(self.n_i):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_i_interval = tmp_i_interval.repeat(1,1,p_ed-p_st) # shape: n_i,1,parallel_eq_n
                cur_i_interval[i:i+1,:,:] = input_interval_candidates[i:i+1,:,p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_i, dim=-1), dim=-2).unsqueeze(-1) # shape: B,*,ic -> B,*,1,ic -> B,*,1,crb_ins -> B,*,n_i,crb_ins -> B,*,n_i,crb_ins,1
                x_sim=(x_sim/(cur_i_interval)).round_().clamp_(-self.i_qmax,self.i_qmax-1)*(cur_i_interval) # shape: B,*,n_i,crb_ins,parallel_eq_n
                x_sim = x_sim.permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: B,*,n_i,crb_ins,parallel_eq_n -> B,*,paraller_eq_n,n_i,crb_ins -> B,*,parallel_eq_n,ic
                # quantize output
                out = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n,oc
                out_sim = self.quant_output(out)
                # calculate similarity and store them
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: B,*,parallel_eq_n
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
                similarities.append(similarity)
            # store best input interval and store in tmp_i_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n
            i_best_index = similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_i_interval[i:i+1,:,:] = torch.gather(input_interval_candidates[i:i+1,:,:],dim=2,index=i_best_index)
        self.i_interval = tmp_i_interval.squeeze(-1)

    def _search_best_o_interval(self, x, output_interval_candidates, raw_out_expanded):
        tmp_o_interval = self.o_interval.unsqueeze(-1) # shape: n_o,1,1
        for o in range(self.n_o):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_o_interval = tmp_o_interval.repeat(1,1,p_ed-p_st) # shape: n_o,1,parallel_eq_n
                cur_o_interval[o:o+1,:,:] = output_interval_candidates[o:o+1,:,p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim = self.quant_input(x)
                # quantize output
                out = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,oc
                out_sim = torch.cat(torch.chunk(out.unsqueeze(-2), chunks=self.n_o, dim=-1), dim=-2).unsqueeze(-1) # shape: B,*,oc -> B,*,1,oc -> B,*,1,crb_outs -> B,*,n_o,crb_outs -> B,*,n_o,crb_outs,1
                out_sim = (out_sim/(cur_o_interval)).round_().clamp_(-self.o_qmax,self.o_qmax-1)*(cur_o_interval) # shape: B,*,n_o,crb_outs,parallel_eq_n
                out_sim = out_sim.permute(*list(range(len(out_sim.shape)-3)),-1,-3,-2).reshape(*out.shape[:-1],p_ed-p_st,out.shape[-1]) # shape: B,*,n_o,crb_outs,parallel_eq_n -> B,*,paraller_eq_n,n_o,crb_outs -> B,*,parallel_eq_n,oc
                # calculate similarity and store them
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: B,*,parallel_eq_n
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
                similarities.append(similarity)
            # store best output interval and store in tmp_o_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n
            o_best_index = similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_o_interval[o:o+1,:,:] = torch.gather(output_interval_candidates[o:o+1,:,:],dim=2,index=o_best_index)
        self.o_interval = tmp_o_interval.squeeze(-1)

    def _initialize_intervals(self, x, out):
        if self.init_layerwise:
            self.w_interval=((self.weight.abs().max())/(self.w_qmax-0.5)).view(1,1,1,1).repeat(self.n_V,1,self.n_H,1)
            self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach().view(1,1).repeat(self.n_i,1) # shape: n_i,1
            self.o_interval=(out.abs().max()/(self.o_qmax-0.5)).detach().view(1,1).repeat(self.n_o,1) # shape: n_o,1
        else:
            self.w_interval=(self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)/(self.w_qmax-0.5))
            self.i_interval=((x.view(*x.shape[:-1],self.n_i,self.crb_ins).abs().amax(list(range(len(x.shape)-1))+[-1],keepdim=False))/(self.i_qmax-0.5)).unsqueeze(-1)
            self.o_interval=((out.view(*out.shape[:-1],self.n_o,self.crb_outs).abs().amax(list(range(len(out.shape)-1))+[-1],keepdim=False))/(self.o_qmax-0.5)).unsqueeze(-1)

    def calibration_step2(self,x):
        # initialize intervals with minmax intervals
        out=F.linear(x, self.weight, self.bias)
        self._initialize_intervals(x, out)

        # put raw outs on GPU
        raw_out_expanded = self.raw_out.to(x.device).unsqueeze(-2)  # shape: B,*,1,oc
        raw_out_expanded_chunked = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,1,n_V,crb_rows

        # put raw grad on GPU
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None

        # prepare weight intervals and similarities
        weight_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1) * self.w_interval.unsqueeze(0) # shape: eq_n,n_V,1,n_H,1
        input_interval_candidates =  torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(1,1,-1) * self.i_interval.unsqueeze(-1) # shape: n_i,1,eq_n
        output_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(1,1,-1) * self.o_interval.unsqueeze(-1) # shape: n_o,1,eq_n
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(x, weight_interval_candidates, raw_out_expanded_chunked)
            # search for best input interval
            self._search_best_i_interval(x, input_interval_candidates, raw_out_expanded)
            # search for best output interval
            self._search_best_o_interval(x, output_interval_candidates, raw_out_expanded)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out=self._bias_correction_quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out
    
class PowerOfTwoQuantLinear(MinMaxQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction=False):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)

    def quant_output(self,out):
        out_sim=torch.sign(out)*(torch.log2(out.abs()/(self.o_max+1e-8)).abs().round_().clamp_(0, self.o_qmax-1))
        out_sim=torch.sign(out_sim)*torch.pow(2,-out_sim.abs())
        out_sim.mul_(self.o_max)
        return out_sim
    
    def calibration_step2(self, x):
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach()
        out=F.linear(x, self.weight, self.bias)
        self.o_max=out.abs().max().detach()
        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out

# class PowerOfTwoFactorQuantLinear(MinMaxQuantLinear):
    
class KScaleChannelWiseQuantLinear(MinMaxQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction = False,
        k = 2,
        power_of_two_scaling = False
    ):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)
        self.k = k
        self.power_of_two_scaling = power_of_two_scaling
    
    def kmeans_channel_wise(self, tensor, num_iters=100):
        indices = torch.randperm(tensor.size(-1))[:self.k]
        tensor = tensor.abs().mean(torch.arange(tensor.dim()-1).tolist())
        centroids = tensor[indices]

        for _ in range(num_iters):
            distances = torch.abs(tensor.unsqueeze(-1) - centroids.unsqueeze(0))  # D,K
            labels = torch.argmin(distances, dim=-1) # D
            new_centroids = torch.stack([tensor[labels == i].mean() for i in range(self.k)])

            if torch.allclose(centroids, new_centroids, rtol=1e-4):
                break

            centroids = new_centroids

        sorted_centroids, sorted_indices = torch.sort(centroids)

        new_labels = torch.zeros_like(labels)
        for new_idx, old_idx in enumerate(sorted_indices):
            new_labels[labels == old_idx] = new_idx

        return new_labels

    def quant_output(self, out):
        out_sim=(out/torch.tensor([self.o_interval[i] for i in self.clusters]).to(out.device)).round_().clamp_(-self.o_qmax,self.o_qmax-1)
        out_sim.mul_(torch.tensor([self.o_interval[i] for i in self.clusters]).to(out.device))
        return out_sim

    def calibration_step2(self, x):
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach()
        out = F.linear(x, self.weight, self.bias)
        self.clusters = self.kmeans_channel_wise(out)
        self.o_interval = torch.zeros(self.k)
        for cluster_index in range(self.k):
            self.o_interval[cluster_index] = ((out*torch.eq(self.clusters, cluster_index)).abs().max()/(self.o_qmax-0.5)).detach()

        if self.power_of_two_scaling:
            for cluster_index in range(1, self.k):
                self.o_interval[cluster_index] = self.o_interval[0] * torch.pow(2, torch.log2(self.o_interval[cluster_index] / self.o_interval[0]).round())

        self.calibrated=True
        return out

class KScaleTokenWiseQuantLinear(MinMaxQuantLinear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        i_bit = 8,
        o_bit = 8,
        bias_bit = None,
        bias_correction = False,
        k = 2,
        L = 197,
        power_of_two_scaling = False
    ):
        super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)
        self.k = k
        self.L = L
        self.power_of_two_scaling = power_of_two_scaling
    
    def kmeans_token_wise(self, tensor, num_iters=100):
        indices = torch.randperm(tensor.size(-1))[:self.k]
        tensor = tensor.abs().mean(torch.arange(tensor.dim()-1).tolist())
        centroids = tensor[indices]

        for _ in range(num_iters):
            distances = torch.abs(tensor.unsqueeze(-1) - centroids.unsqueeze(0))  # L,K
            labels = torch.argmin(distances, dim=-1) # L
            new_centroids = torch.stack([tensor[labels == i].mean() for i in range(self.k)])

            if torch.allclose(centroids, new_centroids, rtol=1e-4):
                break

            centroids = new_centroids

        sorted_centroids, sorted_indices = torch.sort(centroids)

        new_labels = torch.zeros_like(labels)
        for new_idx, old_idx in enumerate(sorted_indices):
            new_labels[labels == old_idx] = new_idx

        return new_labels

    def quant_output(self, out):
        out=out.permute((0, 2, 1))
        out_sim=(out/torch.tensor([self.o_interval[i] for i in self.clusters]).to(out.device)).round_().clamp_(-self.o_qmax,self.o_qmax-1)
        out_sim.mul_(torch.tensor([self.o_interval[i] for i in self.clusters]).to(out.device))
        out_sim=out_sim.permute((0, 2, 1))
        return out_sim

    def calibration_step2(self, x):
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.i_interval=(x.abs().max()/(self.i_qmax-0.5)).detach()
        out = F.linear(x, self.weight, self.bias)
        out = out.permute((0, 2, 1))
        self.clusters = self.kmeans_token_wise(out)
        self.o_interval = torch.zeros(self.k)
        for cluster_index in range(self.k):
            self.o_interval[cluster_index] = ((out*torch.eq(self.clusters, cluster_index)).abs().max()/(self.o_qmax-0.5)).detach()

        if self.power_of_two_scaling:
            for cluster_index in range(1, self.k):
                self.o_interval[cluster_index] = self.o_interval[0] * torch.pow(2, torch.log2(self.o_interval[cluster_index] / self.o_interval[0]).round())

        self.calibrated=True
        out = out.permute((0, 2, 1))
        return out

# class SmoothQuantLinear(MinMaxQuantLinear):
#     def __init__(self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         mode = "raw",
#         w_bit = 8,
#         i_bit = 8,
#         o_bit = 8,
#         bias_bit = None,
#         bias_correction=False):
#         super().__init__(in_features,out_features,bias,mode,w_bit,i_bit,o_bit,bias_bit,bias_correction)

