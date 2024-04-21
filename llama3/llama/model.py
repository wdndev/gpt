# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    """ RMS 归一化
        m = 1/n * sum x^2 + eps
        y = x / sqrt(m)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 注意：可学习的权重参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算输入向量x在最后一个维度上的平方均值，加上数值稳定小量eps以防止除以零
        mean_square = x.pow(2).mean(-1, keepdim=True) + self.eps
        # 计算根号下平方均值的倒数，然后与原向量x逐元素相乘得到归一化后的向量
        return x * torch.rsqrt(mean_square)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """ 预计算CIS(正弦和余弦)频率矩阵
        - dim : CIS频率的维数
        - end : CIS频率的最大索引
        - theta : CIS频率的比例因子。默认为10000.0。
    """
    # f = 1 / (theta ^{i/dim})，其中i是偶数索引，且范围在[0, dim//2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 创建一个从0到end-1的浮点型张量t
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # 计算频率与时间步长的外积，生成二维频率矩阵
    # outer(a, b) = a^T * b
    # >>> v1 = torch.arange(1., 5.)
    # >>> v2 = torch.arange(1., 4.)
    # >>> torch.outer(v1, v2)
    # tensor([[  1.,   2.,   3.],
    #         [  2.,   4.,   6.],
    #         [  3.,   6.,   9.],
    #         [  4.,   8.,  12.]])
    freqs = torch.outer(t, freqs)
    # 使用polar函数将幅度设为1，角度为freqs，得到复数形式的频率
    # 这里生成的是形如(cos(f*t), sin(f*t))的复数序列
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 输出数据类型为complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """ reshape 频率张量形状，用于广播张量x, 确保freqs_cis可以正确地与输入x进行广播操作
        - freqs_cis
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 旋转位置编码
        - xq : (b, l, qhn, hd) -> (batch_size, length, q_head_num, hidden_dim)
        - xk : (b, l, kvhn, hd)
        - freqs_cis : (l, hd)
    """
    # 将 xq 和 xk 转换为复数表示，形状变为(..., 频率维度, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # xq_out (b, l, qhn * hd)
    # xk_out (b, l, kvhn * hd)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """ 对输入张量x按照指定的重复次数n_rep进行扩展并在指定维度上堆叠
        torch.repeat_interleave(x, dim=2, repeats=n_rep)
        - x : 形状为 (batch_size, sequence_length, n_kv_heads, head_dim)
        - n_rep : 在第四个维度上重复的次数
        返回：形状为 (batch_size, sequence_length, n_kv_heads * n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # 在张量x的第四个维度上插入一个新维度，使其形状变为 (bs, slen, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # 按照需要在第四个维度上进行扩展，使其形状变为 (bs, slen, n_kv_heads, n_rep, head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # 将扩展后的张量重塑为所需的新形状，即将第四个维度与第三个维度合并
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 获取模型并行的word size
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # 根据模型并行划分头数
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 计算头数的重复次数，用来解决局部kv头数少于局部q头数的情况
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # fairscale.ColumnParallelLinear 和 PyTorch 中的标准 nn.Linear（全连接层）
        # 主要区别在于它们如何处理大型模型中的内存和计算效率问题，特别是对于分布式训练场景。
        # nn.Linear : 随着模型规模的增长，尤其是当模型参数量过大以至于无法适应单个设备的显存时，nn.Linear 会出现内存瓶颈。
        #   适合小型到中型模型，在单个GPU上运行，所有权重和计算都在单个设备上进行。
        # fairscale.ColumnParallelLinear ： 为了解决大规模模型训练中内存分配问题而设计的。
        #   适用于大型模型，支持跨多个GPU并行处理权重，通过分解权重矩阵的列来分摊存储和计算负载，特别适用于模型并行训练策略。
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # 使用fairscale库中的RowParallelLinear创建输出的线性组合层
        # 这个层在行方向上进行模型并行
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # 初始化缓存用于存储键和值向量，用于自回归模型的解码阶段
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 将输入通过线性层进行投影，得到query、key和value
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转嵌入变换到query和key上
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 更新缓存中的键和值向量
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # 将当前批次和序列长度内的key和value存入缓存
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 从缓存中获取已有的键和值向量
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # 如果n_kv_heads < n_heads，重复键和值的头以匹配query的数量
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # 重新排列query、keys和values的维度顺序以方便计算注意力分数
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        # # 计算注意力分数，采用点积注意力的形式
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 应用掩码（例如：填充mask或因果关系mask）
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # 应用softmax函数得到注意力权重
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 计算上下文向量，通过注意力权重对value向量进行加权求和
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # 将上下文向量恢复成原始维度
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """ FFN
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # 确保隐藏层维度是multiple_of参数的整数倍，以满足硬件对齐要求
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim,            # 输入维度
            hidden_dim,     # 调整后的隐藏层维度
            bias=False,     # 不使用偏置项
            gather_output=False,        # 不输出并行计算的结果
            init_method=lambda x: x     # 初始化方法（此处保留输入值作为权重）
        )
        self.w2 = RowParallelLinear(
            hidden_dim,     # 隐藏层维度
            dim,            # 输出维度
            bias=False,     # 不使用偏置项
            input_is_parallel=True,     # 输入是并行化的
            init_method=lambda x: x     # 初始化方法（此处保留输入值作为权重）
        )
        self.w3 = ColumnParallelLinear(
            dim, 
            hidden_dim, 
            bias=False, 
            gather_output=False, 
            init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        # 预计算旋转位置编码，复数形式
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """ 
            - start_pos (int): 当前批次序列在缓存中的起始位置。
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        # 提取当前序列所需的频率信息
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 根据序列长度生成注意力 mask （对于自回归任务）
        mask = None
        if seqlen > 1:
            # 创建mask，全填充张量。
            # 这个负无穷大的值在softmax操作之后会导致对应的注意力权重为0
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            # 通过对角线上方的元素保持不变，对角线下方的元素设置为0，得到上三角矩阵。
            # 这样可以确保同一序列内，每个位置只能看到它之前的位置，而不能看到之后的位置。
            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            # 当有缓存时，只对新输入的部分（从start_pos开始的seqlen个位置）和
            # 其他已缓存部分之间的自注意力施加 mask，
            # 即新增了一个宽度为start_pos的全0列到 mask 矩阵左侧。
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
