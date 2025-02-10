import torch.nn as nn
from quant_layers.conv1d import *
from quant_layers.conv2d import *
from quant_layers.QuantLinear import *
from quant_layers.QuantConv1d import *
from quant_layers.QuantSSM import *

def get_module(module_type, *args, **kwargs):
    if module_type == "qconv2d":
        module=MinMaxQuantConv2d(*args,**kwargs,w_bit=8,i_bit=32,o_bit=8)
        pass
    elif module_type == "qconv1d":
        # module=KScaleChannelWiseQuantConv1d(*args,**kwargs,w_bit=8,i_bit=32,o_bit=8,k=4)
        qconv1d_proj_kwargs = {
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "minmax",
                bin = "uniform",
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "minmax",
                bin = "uniform",
                bit = 8,
            ),
        }
        kwargs.update(qconv1d_proj_kwargs)
        module = QuantConv1d(*args,**kwargs)
        pass
    elif module_type == "qin_proj":
        qin_proj_kwargs = {
            "w_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(qin_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qx_proj":
        qx_proj_kwargs = {
            "w_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(qx_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qdt_proj":
        qdt_proj_kwargs = {
            "w_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(qdt_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qout_proj":
        qout_proj_kwargs = {
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "minmax",
                bin = "uniform",
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "minmax",
                bin = "uniform",
                bit = 8,
            ),
        }
        kwargs.update(qout_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qhead":
        qhead_proj_kwargs = {
            "w_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(qhead_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    if module_type == "qssm":
        ssm_kwargs = {
            "search_round": 3,
            "smooth_quant": 0,
            # 0: no smooth quant
            # 1: method 1
            # 2: method 2
            # 3: method 3
            # 4: method 4
            # 5: method 5
            # 6: method 5 with h_record initialization
            # 7: method 4 with trainable scale
            # 8: method 6 with trainable scale
            "w_config": QuantConfig(
                quant = False,
                qmode = "minmax",
                bin = "uniform",
                bit = 8,
            ),
            "i_config": QuantConfig(
                quant = False,
                qmode = "minmax",
                bin = "uniform",
                bit = 32,
            ),
            "h_config": QuantConfig(
                quant = True,
                qmode = "minmax", # "minmax", "similarity"
                bin = "uniform",
                bit = 8,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "minmax", # "minmax", "similarity"
                bin = "uniform",
                bit = 8,
            ),
        }
        kwargs.update(ssm_kwargs)
        module = QuantSSM(*args,**kwargs)
        pass
    return module
