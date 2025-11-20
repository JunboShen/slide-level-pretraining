# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
#
# Portions Copyright Prov-GigaPath
# Original File: https://github.com/facebookresearch/mae

from functools import partial

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from .pos_embed import get_2d_sincos_pos_embed
from .torchscale.model.LongNet import make_longnet_from_name


class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=1536,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        

    def forward(self, x):
        B, L, D = x.shape
        
        x = self.proj(x)
        
        x = self.norm(x)
        return x


class LongNetViT(nn.Module):
    """
    Backbone of Vision Transformer for downstream tasks

    Arguments:
    ----------
    in_chans: int
        The number of input channels, should be the tile encoding dimension 1536.
    embed_dim: int
        The embedding dimension of the LongNet model.
    depth: int
        The number of LongNet layers in the LongNet model.
    slide_ngrids: int
        The number of grids in the slide.
    tile_size: int
        The tile size. Default is 256px.
    max_wsi_size: int
        The maximum size of the WSI.
    norm_layer: nn.LayerNorm
        The normalization layer used in the model.
    global_pool: bool
        Whether to use global pooling or not.
    dropout: float
        The dropout rate used in the model.
    drop_path_rate: float
        The drop path rate used in the model.
    """

    def __init__(self, 
                in_chans=1536, 
                embed_dim=256, 
                depth=12, 
                slide_ngrids=1000, 
                tile_size=256,
                max_wsi_size=262144,
                norm_layer=nn.LayerNorm, 
                global_pool=False, 
                dropout=0.25, 
                drop_path_rate=0.1, 
                num_register_tokens=0,
                **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(in_chans, embed_dim)
        self.embed_dim = embed_dim
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed', torch.zeros(1, num_patches + 1, embed_dim), persistent=False)  # fixed sin-cos embedding
        self.num_register_tokens = num_register_tokens
        self.encoder_name = "LongNet_{}_layers_{}_dim".format(depth, embed_dim)
        if kwargs.get("mlp_ratio", 4.0) != 4.0:
            self.encoder_name += "_mlp{}".format(kwargs.get("mlp_ratio"))
        
        # get optimal segment length
        segment_length = self.get_optimal_segment_length(max_wsi_size, tile_size)
        self.encoder = make_longnet_from_name(self.encoder_name, drop_path_rate=drop_path_rate, dropout=dropout, segment_length=segment_length)
        
        self.norm = norm_layer(embed_dim, eps=1e-6)
        # --------------------------------------------------------------------------

        self.global_pool = global_pool
        print("Global Pooling:", self.global_pool)
        self.head = nn.Identity()
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.initialize_vit_weights()


    def initialize_vit_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, cls_token=True)
        #TODO: replace with fp16 for original: self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def get_optimal_segment_length(self, max_wsi_size: int=262144, tile_size: int=256) -> str:
        '''
        Get the optimal segment length based on the maximum image size and tile size.
        
        Arguments:
        ----------
        max_wsi_size: int
            The maximum size of the WSI.
        tile_size: int
            The tile size.
        '''
        max_seq_len = (max_wsi_size // tile_size) ** 2
        # calculate the segment length
        segment_length = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        # convert to str format
        segment_length = str(list(segment_length))
        return segment_length

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def coords_to_pos(self, coords):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [N, L]
        """
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1  # add 1 for the cls token
    
    def prepare_tokens_with_masks(self, x, coords, masks):
        x = self.patch_embed(x)
        # Apply masks
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # get pos indices
        pos = self.coords_to_pos(coords)  # [N, L]
 
        x = x + self.pos_embed[:, pos, :].squeeze(0)#TODO: make sure x match correct data type: half()
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :] #TODO: make sure cls_token match correct data type: half()
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        return x
    
    def forward_list(self, x_list, coords, masks=None, all_layer_embed=False):
        x_list = [self.prepare_tokens_with_masks(x, coord, mask) for x, coord, mask in zip(x_list, coords, masks)]
        #TODO:make sure x_list match correct data type
        #x_list = [x.half() for x in x_list]
        # apply Transformer blocks
        if all_layer_embed:
            x_list = [self.encoder(src_tokens=None, token_embeddings=x, return_all_hiddens=all_layer_embed)["encoder_states"] for x in x_list]
        else:
            x_list = [self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"] for x in x_list]

        outcomes = []
        for x, mask in zip(x_list, masks):
            x_norm = self.norm(x)
            outcomes.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": mask,
                }
            )

        return outcomes
    
    def forward_features(self, x, coords, masks=None, all_layer_embed=False):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        all_layer_embed: bool
            Whether to return embeddings from all layers or not
        """
        if isinstance(x, list):
            return self.forward_list(x, coords, masks, all_layer_embed)

        x = self.prepare_tokens_with_masks(x, coords, masks)
        #TODO: make sure x match correct data type
        #x = x.half()

        # apply Transformer blocks
        if all_layer_embed:
            x_list = self.encoder(src_tokens=None, token_embeddings=x, return_all_hiddens=all_layer_embed)["encoder_states"]
        else:
            x_list = [self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]]

        outcomes = []
        for x in x_list:
            x_norm = self.norm(x)
            
            outcomes.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        outcomes = torch.stack(outcomes, dim=1) if all_layer_embed else outcomes[0]
        
        return outcomes
    
    def forward(self, x, coords, masks=None, all_layer_embed=False, is_training=False):
        ret = self.forward_features(x, coords, masks, all_layer_embed)
        #TODO: make sure all values are not nan, if nan, replace with 0
        # for key in ret.keys():
        #     print("key", ret[key])
        #check if ret is a list
        # if isinstance(ret, list):
        #     for i in range(len(ret)):
        #         for key in ret[i].keys():
        #             #check ret[key] is a tensor and has nan
        #             if isinstance(ret[i][key], torch.Tensor) and torch.isnan(ret[i][key]).any():
        #                 print("nan in forward pass", key)
        #                 ret[i][key] = torch.where(torch.isnan(ret[i][key]), torch.zeros_like(ret[i][key]), ret[i][key])
        # else:
        #     for key in ret.keys():
        #         #check ret[key] is a tensor and has nan
        #         if isinstance(ret[key], torch.Tensor) and torch.isnan(ret[key]).any():
        #             print("nan in forward pass", key)
        #             ret[key] = torch.where(torch.isnan(ret[key]), torch.zeros_like(ret[key]), ret[key])

        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
    
def slide_enc6l384d(**kwargs):
    model = LongNetViT(embed_dim=384, depth=6, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def slide_enc3l384d(**kwargs):
    model = LongNetViT(embed_dim=384, depth=3, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def slide_enc12l768d(**kwargs):
    model = LongNetViT(embed_dim=768, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def slide_enc24l1024d(**kwargs):
    model = LongNetViT(embed_dim=1024, depth=24, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def slide_enc12l1536d(**kwargs):
    model = LongNetViT(embed_dim=1536, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
