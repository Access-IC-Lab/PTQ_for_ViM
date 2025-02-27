import torch
from quant_layers.QuantConv1d import QuantConv1d
from quant_layers.QuantConv2d import QuantConv2d
from quant_layers.QuantLinear import QuantLinear
from quant_layers.QuantSSM import QuantSSM
from quant_layers.QuantSiLU import QuantSiLU
from quant_layers.QuantSoftplus import QuantSoftplus
from quant_layers.QuantNorm import QuantNorm
from tqdm import tqdm
import torch.nn.functional as F

class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume
    hundreds of GB of memory.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=True, batch_size=1):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
        self.batch_size = batch_size
    
    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        # run calibration
        n_calibration_steps=2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'
            with torch.no_grad():
                for inp,target in self.calib_loader:
                    inp=inp.cuda()
                    self.net(inp)
        
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("sequential calibration finished")
    
    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")
        for name,module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'
        with torch.no_grad():
            for inp,target in self.calib_loader:
                inp=inp.cuda()
                self.net(inp)
        # calibration step2: each module run calibration with collected raw data
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, QuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, QuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, QuantConv1d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, QuantSSM):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda(), module.raw_input[2].cuda(), module.raw_input[3].cuda(), module.raw_input[4].cuda())
                    elif isinstance(module, QuantSiLU):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, QuantSoftplus):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, QuantNorm):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda() if module.raw_input[1] is not None else None)
                    # elif isinstance(module, MinMaxQuantMatMul):
                    #     module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()
                
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("calibration finished")
    
    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Calibration")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, QuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, QuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, QuantConv1d):
                hooks.append(module.register_forward_hook(conv1d_forward_hook))
            if isinstance(module, QuantSSM):
                hooks.append(module.register_forward_hook(ssm_forward_hook))
            if isinstance(module, QuantSiLU):
                hooks.append(module.register_forward_hook(silu_forward_hook))
            if isinstance(module, QuantSoftplus):
                hooks.append(module.register_forward_hook(softplus_forward_hook))
            if isinstance(module, QuantNorm):
                hooks.append(module.register_forward_hook(norm_forward_hook))
            # if isinstance(module, MinMaxQuantMatMul):
            #     hooks.append(module.register_forward_hook(matmul_forward_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    self.net(inp_)
                del inp, target
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, QuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantConv1d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSSM):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSiLU):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSoftplus):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantNorm):
                # module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_input = [torch.cat(module.raw_input[0], dim=0), torch.cat(module.raw_input[1], dim=0) if module.raw_input[1][0] is not None else None]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            # if isinstance(module, MinMaxQuantMatMul):
            #     module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
            #     module.raw_out = torch.cat(module.raw_out, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, QuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantConv1d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantSSM):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda(), module.raw_input[2].cuda(), module.raw_input[3].cuda(), module.raw_input[4].cuda())
                if isinstance(module, QuantSiLU):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantSoftplus):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantNorm):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda() if module.raw_input[1] is not None else module.raw_input[1])
                # if isinstance(module, MinMaxQuantMatMul):
                #     module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("calibration finished")

def grad_hook(module, grad_input, grad_output):
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output[0].cpu().detach())   # that's a tuple!

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def conv1d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def ssm_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[], [], [], [], []]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_input[2].append(input[2].cpu().detach())
    module.raw_input[3].append(input[3].cpu().detach())
    module.raw_input[4].append(input[4].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def silu_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def softplus_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def norm_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[], []]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach() if input[1] is not None else None)
    module.raw_out.append(output.cpu().detach())

# def matmul_forward_hook(module, input, output):
#     if module.raw_input is None:
#         module.raw_input = [[],[]]
#     if module.raw_out is None:
#         module.raw_out = []
#     module.raw_input[0].append(input[0].cpu().detach())
#     module.raw_input[1].append(input[1].cpu().detach())
#     module.raw_out.append(output.cpu().detach())

class HessianQuantCalibrator(QuantCalibrator):
    """
    Modularization of hessian_quant_calib

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.batch_size = batch_size

    def quant_calib(self):
        """
        An implementation of original hessian calibration.
        """

        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp.cuda())
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, QuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, QuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, QuantConv1d):
                hooks.append(module.register_forward_hook(conv1d_forward_hook))
            if isinstance(module, QuantSSM):
                hooks.append(module.register_forward_hook(ssm_forward_hook))
            if isinstance(module, QuantSiLU):
                hooks.append(module.register_forward_hook(silu_forward_hook))
            if isinstance(module, QuantSoftplus):
                hooks.append(module.register_forward_hook(softplus_forward_hook))
            if isinstance(module, QuantNorm):
                hooks.append(module.register_forward_hook(norm_forward_hook))
            # if isinstance(module, MinMaxQuantMatMul):
            #     hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric") and module.metric == "hessian":
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, QuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantConv1d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSSM):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSiLU):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSoftplus):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantNorm):
                module.raw_input = [torch.cat(module.raw_input[0], dim=0), torch.cat(module.raw_input[1], dim=0) if module.raw_input[1][0] is not None else None]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            # if isinstance(module, MinMaxQuantMatMul):
            #     module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
            #     module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric") and module.metric == "hessian":
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, QuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantConv1d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantSSM):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda(), module.raw_input[2].cuda(), module.raw_input[3].cuda(), module.raw_input[4].cuda())
                if isinstance(module, QuantSiLU):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantSoftplus):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantNorm):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda() if module.raw_input[1] is not None else None)
                # if isinstance(module, MinMaxQuantMatMul):
                #     module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp.cuda())
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, QuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, QuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, QuantConv1d):
                hooks.append(module.register_forward_hook(conv1d_forward_hook))
            if isinstance(module, QuantSSM):
                hooks.append(module.register_forward_hook(ssm_forward_hook))
            if isinstance(module, QuantSiLU):
                hooks.append(module.register_forward_hook(silu_forward_hook))
            if isinstance(module, QuantSoftplus):
                hooks.append(module.register_forward_hook(softplus_forward_hook))
            if isinstance(module, QuantNorm):
                hooks.append(module.register_forward_hook(norm_forward_hook))
            # if isinstance(module, MinMaxQuantMatMul):
            #     hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, QuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantConv1d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSSM):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSiLU):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantSoftplus):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, QuantNorm):
                module.raw_input = [torch.cat(module.raw_input[0], dim=0), torch.cat(module.raw_input[1], dim=0) if module.raw_input[1][0] is not None else None]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            # if isinstance(module, MinMaxQuantMatMul):
            #     module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
            #     module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric"):
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, QuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantConv1d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantSSM):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda(), module.raw_input[2].cuda(), module.raw_input[3].cuda(), module.raw_input[4].cuda())
                if isinstance(module, QuantSiLU):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantSoftplus):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, QuantNorm):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda() if module.raw_input[1] is not None else None)
                # if isinstance(module, MinMaxQuantMatMul):
                #     module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")