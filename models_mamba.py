# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

from timm.models.layers import to_2tuple

from mamba_simple import Mamba
from rms_norm import *


__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        return x
    

class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)


    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        # fused_add_norm_fn = rms_norm
        # if residual is None:
        #     hidden_states, residual = fused_add_norm_fn(
        #         hidden_states,
        #         self.norm.weight,
        #         self.norm.bias,
        #         residual=residual,
        #         prenorm=True,
        #         eps=self.norm.eps,
        #     )
        # else:
        #     hidden_states, residual = fused_add_norm_fn(
        #         hidden_states,
        #         self.norm.weight,
        #         self.norm.bias,
        #         residual=residual,
        #         prenorm=True,
        #         eps=self.norm.eps,
        #     )

        # hidden_states, residual = self.norm(hidden_states, residual=residual, prenorm=True)
        new_residual = hidden_states.float() + residual.float() if residual is not None else hidden_states.float()
        hidden_states = self.norm(hidden_states, residual)
        residual = new_residual

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        return hidden_states, residual


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, layer_idx=None, device=None, dtype=None):
    ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, norm_cls)
    block.layer_idx = layer_idx

    return block


class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 norm_epsilon: float = 1e-5, 
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.num_tokens = 1

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = RMSNorm(embed_dim, eps=norm_epsilon, **factory_kwargs)

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        B, M, _ = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        token_position = M // 2
        # add cls token in the middle
        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)

        x = x + self.pos_embed

        # mamba impl
        residual = None
        hidden_states = x

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        # fused_add_norm_fn = rms_norm
        # hidden_states = fused_add_norm_fn(
        #     hidden_states,
        #     self.norm_f.weight,
        #     self.norm_f.bias,
        #     eps=self.norm_f.eps,
        #     residual=residual,
        #     prenorm=False,
        # )
        # hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
        hidden_states = self.norm_f(hidden_states, residual)

        return hidden_states[:, token_position, :]

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        x = self.head(x)
        return x



@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(**kwargs):
    model = VisionMamba(patch_size=16, embed_dim=192, depth=24, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(**kwargs):
    model = VisionMamba(patch_size=16, stride=8, embed_dim=192, depth=24, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(**kwargs):
    model = VisionMamba(patch_size=16, embed_dim=384, depth=24, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(**kwargs):
    model = VisionMamba(patch_size=16, stride=8, embed_dim=384, depth=24, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(**kwargs):
    model = VisionMamba(patch_size=16, embed_dim=768, d_state=16, depth=24, **kwargs)
    model.default_cfg = _cfg()
    return model