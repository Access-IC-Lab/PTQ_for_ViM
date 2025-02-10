import torch
import torch.nn as nn
import numpy as np
from ssm import SequentialSSM

def wrap_modules_in_net(net,cfg):
    wrapped_modules={}
    module_dict={}
    linear_module_types = {"head": "qhead", "in_proj": "qin_proj", "x_proj": "qx_proj", "dt_proj": "qdt_proj", "x_proj_b": "qx_proj", "dt_proj_b": "qdt_proj", "out_proj": "qout_proj"}
    conv1d_module_types = {"conv1d": "qconv1d", "conv1d_b": "qconv1d"}
    conv2d_module_types = {"proj": "qconv2d"}
    ssm_module_types = {"ssm": "qssm", "ssm_b": "qssm"}
    # ql = [".in_proj", ".x_proj", ".x_proj_b", ".dt_proj", ".dt_proj_b","head"]
    # module_types = {"qkv":"qlinear_qkv", "proj":'qlinear_proj', 'fc1':'qlinear_MLP_1', 'fc2':"qlinear_MLP_2", 'head':'qlinear_classifier','matmul1':"qmatmul_qk", 'matmul2':"qmatmul_scorev", "reduction": "qlinear_reduction"}
    
    it=[(name,m) for name,m in net.named_modules()]
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        # if isinstance(m,nn.Conv2d):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(conv2d_module_types[name[idx:]],m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
        #     new_m.weight.data=m.weight.data
        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        # if isinstance(m,nn.Conv1d):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(conv1d_module_types[name[idx:]],m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
        #     new_m.weight.data=m.weight.data
        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        # if isinstance(m,nn.Linear) & ("in_proj" in name):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(linear_module_types[name[idx:]],m.in_features,m.out_features)
        #     new_m.weight.data=m.weight.data
        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        # # if isinstance(m,nn.Linear) & ("x_proj" in name):
        # #     idx = idx+1 if idx != 0 else idx
        # #     new_m=cfg.get_module(linear_module_types[name[idx:]],m.in_features,m.out_features)
        # #     new_m.weight.data=m.weight.data
        # #     new_m.bias=m.bias
        # #     replace_m=new_m
        # #     wrapped_modules[name] = new_m
        # #     setattr(father_module,name[idx:],replace_m)
        # if isinstance(m,nn.Linear) & ("x_proj" in name):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(linear_module_types[name[idx:]],m.in_features,m.out_features)
        #     layer_idx = name.split(".")[1]
        #     new_m.weight.data=m.weight.data

        #     model = "tiny"
        #     if "x_proj_b" in name:
        #         r_D = np.loadtxt(f'./{model}_factor/r_D/{layer_idx}_b.txt', dtype=np.float32)
        #     else:
        #         r_D = np.loadtxt(f'./{model}_factor/r_D/{layer_idx}_f.txt', dtype=np.float32)
        #     r_D_tensor = torch.from_numpy(r_D)
        #     result = torch.cat([
        #         # torch.ones(24),
        #         torch.ones(12),             # 前 12 位是 1
        #         1 / r_D_tensor,             # 13~28 位是 1/s_D
        #         r_D_tensor                  # 29~44 位是 s_D
        #     ])
        #     new_m.weight.data = new_m.weight.data * result.unsqueeze(-1).to(new_m.weight.data.device)
        #     # exit()

        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        # if isinstance(m,nn.Linear) & ("dt_proj" in name):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(linear_module_types[name[idx:]],m.in_features,m.out_features)
        #     new_m.weight.data=m.weight.data
        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        # if isinstance(m,nn.Linear) & ("out_proj" in name):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(linear_module_types[name[idx:]],m.in_features,m.out_features)
        #     new_m.weight.data=m.weight.data
        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        # if isinstance(m,nn.Linear) & ("head" in name):
        #     idx = idx+1 if idx != 0 else idx
        #     new_m=cfg.get_module(linear_module_types[name[idx:]],m.in_features,m.out_features)
        #     new_m.weight.data=m.weight.data
        #     new_m.bias=m.bias
        #     replace_m=new_m
        #     wrapped_modules[name] = new_m
        #     setattr(father_module,name[idx:],replace_m)
        if isinstance(m, SequentialSSM):
            idx = idx+1 if idx != 0 else idx
            new_m = cfg.get_module(ssm_module_types[name[idx:]], m.layer_idx, m.direction)
            replace_m = new_m
            wrapped_modules[name] = new_m
            setattr(father_module, name[idx:], replace_m)

    print("Completed net wrap.")
    return wrapped_modules
