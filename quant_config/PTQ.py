import torch.nn as nn
from quant_layer.QuantConv1d import *
from quant_layer.QuantConv2d import *
from quant_layer.QuantLinear import *
from quant_layer.QuantSSM import *

eq_alpha = 0.01
eq_beta = 1.2
eq_n = 100

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
                similarity_config = SimilarityQuantConfig(
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
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
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
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
        kwargs.update(qconv1d_proj_kwargs)
        module = QuantConv1d(*args,**kwargs)
        pass
    elif module_type == "qin_proj":
        # qin_proj_kwargs = {
        #     "w_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        #     "i_config": QuantConfig(
        #         quant = False,
        #     ),
        #     "o_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        # }
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
        # qx_proj_kwargs = {
        #     "w_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        #     "i_config": QuantConfig(
        #         quant = False,
        #     ),
        #     "o_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        # }
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
        # module = QuantXProj(*args,**kwargs)
        pass
    elif module_type == "qdt_proj":
        # qdt_proj_kwargs = {
        #     "w_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        #     "i_config": QuantConfig(
        #         quant = False,
        #     ),
        #     "o_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        # }
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
                    metric = "cosine", # "cosine", "L2_norm", "L1_norm"
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
        kwargs.update(qout_proj_kwargs)
        module = QuantLinear(*args,**kwargs)
        # module = QuantOutProj(*args, **kwargs)
        pass
    elif module_type == "qhead":
        # qhead_proj_kwargs = {
        #     "w_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        #     "i_config": QuantConfig(
        #         quant = False,
        #     ),
        #     "o_config": QuantConfig(
        #         quant = True,
        #         bit = 8,
        #     ),
        # }
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
    if module_type == "qssm":
        ssm_kwargs = {
            "search_round": 3,
            "smooth_quant": 6,
            # "represent": "mean",
            # "trend": "std",
            # "power": "normalized",
            # "factor": "original",
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
    return module
