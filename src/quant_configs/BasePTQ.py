import torch.nn as nn
from quant_layers.QuantConv1d import *
from quant_layers.QuantConv2d import *
from quant_layers.QuantLinear import *
from quant_layers.QuantSSM import *
from quant_layers.QuantSiLU import *
from quant_layers.QuantSoftplus import *
from quant_layers.QuantNorm import *

def get_module(module_type, *args, **kwargs):
    if module_type == "qconv2d":
        qconv2d_kwargs = {
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
        kwargs.update(qconv2d_kwargs)
        module = QuantConv2d(*args,**kwargs)
        pass
    elif module_type == "qconv1d":
        qconv1d_kwargs = {
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
        kwargs.update(qconv1d_kwargs)
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
    elif module_type == "qssm":
        ssm_kwargs = {
            "search_round": 3,
            "reparameterization": False,
            "w_config": QuantConfig(
                quant = False,
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "h_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(ssm_kwargs)
        module = QuantSSM(*args,**kwargs)
        pass
    elif module_type == "qact":
        qact_kwargs = {
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(qact_kwargs)
        module = QuantSiLU(*args,**kwargs)
        pass
    elif module_type == "qsoftplus":
        qsoftplus_kwargs = {
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                bit = 8,
            ),
        }
        kwargs.update(qsoftplus_kwargs)
        module = QuantSoftplus(*args,**kwargs)
        pass
    elif module_type == "qnorm":
        qnorm_kwargs = {
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
        kwargs.update(qnorm_kwargs)
        module = QuantNorm(*args,**kwargs)
        pass
    return module
