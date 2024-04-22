# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_abs_pos(abs_pos, tgt_size):
    """ 根据绝对位置坐标生成归一化的位置嵌入
    """
    # abs_pos: L, C （二维绝对位置编码）
    # tgt_size: (H, W) （目标高度和宽度）
    # return: M, C （目标大小插值后的二维位置编码）

    # 计算原始图像的边长
    src_size = int(math.sqrt(abs_pos.size(0)))
    # tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    # 使用双三次插值对绝对位置编码进行插值，将其调整为目标大小
    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    根据网格小大生成2D正弦余弦位置嵌入
    参数：
    - embed_dim ： 位置嵌入维度
    - grid_size ： 网格的高度和宽度（整数或元组）
    - cls_token ： 是否包含分类标记的额外位置嵌入，默认为 False
    返回：
    - pos_embed：形状为[grid_size*grid_size, embed_dim]
            或[1+grid_size*grid_size, embed_dim]的张量（取决于是否包含cls_token）
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    # np.meshgrid 用于创建多个一维数组（通常是代表不同坐标轴的值）的多维坐标网格的功能。
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    # 从网格生成2D的位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # 若需要包含分类标记的位置嵌入，则在前面添加全零向量
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """ 从网格坐标生成 2D 正弦余弦位置嵌入
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    # 使用一半维度编码行和列
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)，行
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)，列

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    根据给定的一维位置列表生成正弦余弦位置嵌入
    参数：
    - embed_dim: 每个位置的嵌入维度
    - pos: 位置列表，形状为(M,)
    返回
    - emb: (M, D)
    """
    assert embed_dim % 2 == 0
    # 创建频率范围
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    # 计算正弦和余弦编码
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    通过一个带有2D正弦余弦位置嵌入和一组(grid_size**2)可学习query的交叉注意力层，实现对输入的重采样
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,      # 采样的栅格尺寸，即输入空间划分的单元格数量。
            embed_dim,      # 位置嵌入和query向量的维度
            num_heads,
            kv_dim=None,    # key和value向量的原始维度，若与 `embed_dim` 不同，则需通过 `kv_proj` 进行投影。
            norm_layer=partial(nn.LayerNorm, eps=1e-6), # 默认为带eps参数的LayerNorm层，用于规范化输入
            adaptive=False  # 标记是否开启自适应目标大小功能
    ):
        super().__init__()
        # 初始化参数
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive    # 是否启用自适应目标大小

        # 创建预计算的2D正弦余弦位置嵌入参数
        # (grid_size**2, embed_dim)，不参与反向传播更新
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        # query
        # (grid_size**2, embed_dim)
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        # 根据kv_dim的值决定是否使用线性投影层
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        # 创建多头注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        # 后处理归一化层和投影层
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ 初始化权重方法
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_size=None, attn_mask=None):
        # 根据adaptive标志决定如何获取位置嵌入
        if self.adaptive:
            # 动态计算自适应的目标大小下的位置嵌入
            pos_embed = torch.Tensor(get_2d_sincos_pos_embed(self.embed_dim, tgt_size)).float().to(device=x.device, dtype=x.dtype)
        else:
            # 对预计算的位置嵌入进行插值以适应目标大小
            pos_embed = get_abs_pos(self.pos_embed, tgt_size)

        x = self.kv_proj(x)
        # 对键值向量做层归一化并改变顺序以符合注意力层的输入格式
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]  # 获取输入特征的第二维（序列长度）大小
        q = self.ln_q(self.query)
        # 执行注意力机制，将query向量复制并拼接位置嵌入
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        # 使查询向量能够与输入特征匹配，通过在第一维度重复查询向量
        return query.unsqueeze(1).repeat(1, N, 1)

if __name__ == "__main__":
    # 设定参数
    grid_size = 4
    embed_dim = 64
    num_heads = 8
    kv_dim = None
    norm_layer = nn.LayerNorm
    adaptive = True

    # 实例化Resampler类
    resampler = Resampler(grid_size, embed_dim, num_heads, kv_dim, norm_layer, adaptive)

    # 创建随机输入数据
    batch_size = 1
    seq_length = 256  # 假设输入长度（这里简化为一维情况）
    x = torch.randn(batch_size, seq_length, embed_dim)

    tgt_size = (16, 16)

    # 执行前向传播
    output = resampler(x, tgt_size)

    # 输出结果信息
    print("Input shape:", x.shape)
    print("Target size (if adaptive):", tgt_size)
    print("Output shape after Resampler:", output.shape)
