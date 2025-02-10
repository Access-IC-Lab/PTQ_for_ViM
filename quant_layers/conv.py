import torch
import torch.nn as nn
import torch.nn.functional as F

class MinMaxQuantConv2d(nn.Conv2d):
    """
    MinMax quantize weight and output
    """
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
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8,bias_bit=None
    ):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.n_calibration_steps=2
        self.mode=mode
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.bias_bit=bias_bit
        assert bias_bit is None,"No support bias bit now"
        self.w_interval=None
        self.a_interval=None
        self.bias_interval=None
        self.raw_input=None
        self.raw_out=None
        self.metric=None
        self.next_nodes=[]
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        # self.bias_qmax=2**(self.bias_bit-1)
        
    def forward(self, x):
        if self.mode=='raw':
            out=F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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
    
    def quant_input(self,x):
        x_sim=(x/self.a_interval).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim.mul_(self.a_interval)
        return x_sim
    
    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    def calibration_step1(self,x):
        # step1: collection the FP32 values
        out=F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.raw_input=x.cpu().detach()
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,x):
        # step2: search for the best S^w and S^a of each layer
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.a_interval=(x.abs().max()/(self.a_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(x)        
        return out


class MinMaxQuantConv1d(nn.Conv1d):
    """
    MinMax quantize weight and output
    """
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
        padding_mode: str = 'zeros',mode='raw',w_bit=8,a_bit=8,bias_bit=None
    ):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.n_calibration_steps=2
        self.mode=mode
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.bias_bit=bias_bit
        assert bias_bit is None,"No support bias bit now"
        self.w_interval=None
        self.a_interval=None
        self.bias_interval=None
        self.raw_input=None
        self.raw_out=None
        self.metric=None
        self.next_nodes=[]
        self.w_qmax=2**(self.w_bit-1)
        self.a_qmax=2**(self.a_bit-1)
        # self.bias_qmax=2**(self.bias_bit-1)
        
    def forward(self, x):
        if self.mode=='raw':
            out=F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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
    
    def quant_input(self,x):
        x_sim=(x/self.a_interval).round_().clamp_(-self.a_qmax,self.a_qmax-1)
        x_sim.mul_(self.a_interval)
        return x_sim
    
    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    def calibration_step1(self,x):
        # step1: collection the FP32 values
        out=F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.raw_input=x.cpu().detach()
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,x):
        # step2: search for the best S^w and S^a of each layer
        self.w_interval=(self.weight.data.abs().max()/(self.w_qmax-0.5)).detach()
        self.a_interval=(x.abs().max()/(self.a_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(x)        
        return out
    

class QuantileQuantConv2d(MinMaxQuantConv2d):
    """
    Quantile quantize weight and output
    """
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        mode='raw',w_bit=8,a_bit=8,bias_bit=None,
        w_quantile=0.9999,a_quantile=0.9999):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,mode,w_bit,a_bit,bias_bit)
        self.w_quantile = w_quantile
        self.a_quantile = a_quantile

    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.numel()//16777216
            return torch.quantile(tensor.view(-1)[:16777216*n].view(n,16777216),quantile,1).mean()
        else:
            return torch.quantile(tensor,quantile)

    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        self.w_interval=(self._quantile(self.weight.data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.a_interval=(self._quantile(x.abs(),self.a_quantile)/(self.a_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(x)        
        return out
    
class QuantileQuantConv1d(MinMaxQuantConv1d):
    """
    Quantile quantize weight and output
    """
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        mode='raw',w_bit=8,a_bit=8,bias_bit=None,
        w_quantile=0.9999,a_quantile=0.9999):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,mode,w_bit,a_bit,bias_bit)
        self.w_quantile = w_quantile
        self.a_quantile = a_quantile

    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            n = tensor.numel()//16777216
            return torch.quantile(tensor.view(-1)[:16777216*n].view(n,16777216),quantile,1).mean()
        else:
            return torch.quantile(tensor,quantile)

    def calibration_step2(self,x):
        # step2: search for the best S^w and S^o of each layer
        self.w_interval=(self._quantile(self.weight.data.abs(),self.w_quantile)/(self.w_qmax-0.5)).detach()
        self.a_interval=(self._quantile(x.abs(),self.a_quantile)/(self.a_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(x)        
        return out