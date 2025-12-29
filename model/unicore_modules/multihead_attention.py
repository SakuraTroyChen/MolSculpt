# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from unicore.modules import softmax_dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch, to_dense_adj
from torch_geometric.typing import PairTensor, Adj, OptTensor
import torch.nn.functional as F

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, bias=attn_bias,
            )
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, inplace=False,
            )

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


class CrossMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn = softmax_dropout(attn_weights, self.dropout, self.training, bias=attn_bias)

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        return o


class SelfMultiheadAttentionV2(MessagePassing):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.in_proj_bond = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.bond_embed0 = nn.Embedding(5, embed_dim) # I am only considering bond types here, because the edge_type of unimol can be inferred from the pair atom types
        self.bond_embed1 = nn.Embedding(5, embed_dim)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        bond_type: Optional[Tensor] = None,
        orc: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # print(query.size(), edge_index.size(), bond_type.size(), batch.size())
        ## the new forward. This part follows GraphGPS
        q, k, v = self.in_proj_bond(query.flatten(0,1)).chunk(3, dim=-1) # shape [bsz * tgt_len, embed_dim]
        offset, col, row = orc
        alpha_bias = attn_bias.view(bsz, self.num_heads, tgt_len, tgt_len).transpose(0,1).contiguous[:, offset, col, row] # shape [num_heads, E]
        alpha_bias = alpha_bias.transpose(0, 1).contiguous() # shape [E, num_heads]

        out_x, alpha_weight = self.propagate(edge_index, q=q, k=k, v=v, bond_type=bond_type, size=None) # shape = [bsz * tgt_len, num_heads, head_dim], [E, num_heads]
        out_x = out_x.view(bsz, tgt_len, self.num_heads * self.head_dim) # shape [bsz, tgt_len, embed_dim]
        
        alpha_weight = to_dense_adj(edge_index, batch, alpha_weight, max_num_nodes=tgt_len) # shape [bsz, tgt_len, tgt_len, num_heads]
        alpha_weight = alpha_weight.transpose(2, 3).transpose(1, 2).contiguous().view(bsz * self.num_heads, tgt_len, tgt_len) # shape [bsz * num_heads, tgt_len, tgt_len]

        
        ## the original forward
        q, k, v = self.in_proj(query).chunk(3, dim=-1) # shape [bsz, tgt_len, embed_dim]


        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2)) # shape [bsz * num_heads, tgt_len, src_len]

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            assert False
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, bias=attn_bias,
            )
        else:
            attn_weights += attn_bias + alpha_weight
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, inplace=False,
            )

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )

        o = o + out_x
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn



    def message(self, q_i, k_j, v_j, bond_type, alpha_bias, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        ## q1: should I use this attention weight to update the euclidean distance? let the two be paralel. 
        q_i, k_j, v_j = q_i.view(-1, self.num_heads, self.head_dim), k_j.view(-1, self.num_heads, self.head_dim), v_j.view(-1, self.num_heads, self.head_dim)
        bond_attn = self.bond_embed0(bond_type).view(-1, self.num_heads, self.head_dim)
        bond_attn = torch.tanh(bond_attn)
        alpha_weight = (q_i * k_j * bond_attn).sum(dim=-1) * self.scaling # shape [E, num_heads]
        alpha_weight = alpha_weight + alpha_bias
        alpha = softmax(alpha_weight, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        msg = v_j
        msg = msg * torch.tanh(self.bond_embed1(bond_type).view(-1, self.num_heads, self.head_dim))
        msg = msg * alpha.view(-1, self.num_heads, 1)
        return msg, alpha_weight

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        msg, alpha_weight = inputs
        return self.aggr_module(msg, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim), alpha_weight
    
