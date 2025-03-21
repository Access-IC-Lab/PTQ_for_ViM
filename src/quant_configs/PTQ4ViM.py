import torch.nn as nn
from quant_layers.QuantConv1d import *
from quant_layers.QuantConv2d import *
from quant_layers.QuantLinear import *
from quant_layers.QuantSSM import *
from quant_layers.QuantSiLU import *
from quant_layers.QuantSoftplus import *
from quant_layers.QuantNorm import *

eq_alpha = 0.01
eq_beta = 1.2
eq_n = 100

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
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "minmax",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = True,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = True
                )
            ),
        }
        kwargs.update(qconv1d_kwargs)
        module = QuantConv1d(*args,**kwargs)
        pass
    elif module_type == "qin_proj":
        qin_proj_kwargs = {
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = False,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
            ),
        }
        kwargs.update(qin_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qx_proj":
        qx_proj_kwargs = {
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = False,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
            ),
        }
        kwargs.update(qx_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qdt_proj":
        qdt_proj_kwargs = {
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = False,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
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
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = True,
                    k_scaled_mode = "token_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = True
                )
            ),
        }
        kwargs.update(qout_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qhead":
        qhead_proj_kwargs = {
            "search_round": 3,
            "w_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
            ),
            "i_config": QuantConfig(
                quant = False,
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity",
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine",
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = False,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
            ),
        }
        kwargs.update(qhead_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        pass
    elif module_type == "qssm":
        ssm_kwargs = {
            "search_round": 3,
            "reparameterization": True,
            # "representative": "mean",
            # "dispersion": "std",
            # "power": "normalized",
            # "factor": "original",
            "w_config": QuantConfig(
                quant = False,
                qmode = "minmax",
                bin = "uniform",
                bit = 32,
            ),
            "i_config": QuantConfig(
                quant = False,
                qmode = "minmax",
                bin = "uniform",
                bit = 32,
            ),
            "h_config": QuantConfig(
                quant = True,
                qmode = "similarity", # "minmax", "similarity"
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = False,
                    k_scaled_mode = "hidden_dimension_wise", # "channel_wise", "hidden_dimension_wise"
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
            ),
            "o_config": QuantConfig(
                quant = True,
                qmode = "similarity", # "minmax", "similarity"
                bin = "uniform",
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = True,
                    k_scaled_mode = "token_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = True
                )
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
                qmode = "similarity", # "minmax", "similarity"
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = True,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
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
                qmode = "similarity", # "minmax", "similarity"
                bit = 8,
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
                    eq_alpha = eq_alpha,
                    eq_beta = eq_beta,
                    eq_n = eq_n,
                ),
                k_scaled_config = KScaledQuantConfig(
                    k_scaled = True,
                    k_scaled_mode = "channel_wise",
                    k = 4,
                    k_scaled_power_of_two_scaling = False
                )
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
