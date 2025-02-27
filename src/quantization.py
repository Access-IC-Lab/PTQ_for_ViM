import sys

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import os
import quant_utils.net_wrap as net_wrap
from importlib import reload, import_module
from quant_utils.quant_calib import QuantCalibrator, HessianQuantCalibrator


def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _, _, files = next(os.walk("./quant_configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"quant_configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg


def quantization(model, data_loader_calib, config="BasePTQ"):

    model.cuda()
    model.eval()
    quant_cfg = init_config(config)
    wrapped_modules = net_wrap.wrap_modules_in_net(model, quant_cfg)

    quant_calibrator = QuantCalibrator(
        model, wrapped_modules, data_loader_calib, sequential=False, batch_size=8
    )
    # quant_calibrator = HessianQuantCalibrator(model, wrapped_modules, data_loader_calib, sequential=False, batch_size=8)
    quant_calibrator.batching_quant_calib()

    return model
