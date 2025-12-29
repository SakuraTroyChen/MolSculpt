import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter

disable_compile = torch.cuda.get_device_name(0).find('AMD') >= 0

@torch.compiler.disable
def uncompilable_dropout(x, p, training):
    return F.dropout(x, p=p, training=training, )


def coord2dist(x, edge_index):
    # coordinates to distance
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial

def remove_mean_with_mask(x, node_mask, return_mean=False):
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    if return_mean:
        return x, mean
    return x


def remove_mean(pos, batch):
    mean_pos = scatter(pos, batch, dim=0, reduce='mean') # shape = [B, 3]
    pos = pos - mean_pos[batch]
    return pos


def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale

class LearnedSinusodialposEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class TransLayer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = nn.Linear(edge_dim, heads * out_channels, bias=False)

        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    @torch.compile(dynamic=True, disable=disable_compile)
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = uncompilable_dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class TransLayerOptim(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransLayerOptim, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_qkv = nn.Linear(in_channels, heads * out_channels * 3, bias=bias)

        self.lin_edge = nn.Linear(edge_dim, heads * out_channels * 2, bias=False)

        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.lin_edge.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels
        x_feat = x
        qkv = self.lin_qkv(x_feat).view(-1, H, 3, C)
        query, key, value = qkv.unbind(dim=2)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    @torch.compile(dynamic=True, disable=disable_compile)
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_key, edge_value = torch.tanh(self.lin_edge(edge_attr)).view(-1, self.heads, 2, self.out_channels).unbind(dim=2)

        alpha = (query_i * key_j * edge_key).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = uncompilable_dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j * edge_value * alpha.view(-1, self.heads, 1)
        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features"""
    def __init__(self, K, *args, **kwargs):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, *args, **kwargs):
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class CondEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, residual=True):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.residual = residual

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
        inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        inv = torch.tanh(self.coord_mlp(inv))
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0, reduce='add', dim_size=pos.size(0))
        if self.residual:
            return pos + agg
        else:
            return agg


class EquivariantBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, time_dim, num_heads,
                 cond_time=True, mlp_ratio=4, act=nn.GELU, dropout=0.1, dist_emb=False, equi_pos=False, pair_update=True, fuse_qkv=False):
        super().__init__()

        self.dropout = dropout
        self.act1 = act()
        self.act2 = act()
        self.cond_time = cond_time
        dist_dim = edge_dim
        self.dist_emb = dist_emb
        self.pair_update = pair_update
        if dist_emb:
            self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
            self.dist_layer = GaussianLayer(dist_dim)
        else:
            if self.pair_update:
                self.edge_emb = nn.Linear(edge_dim, edge_dim)
            else:
                self.edge_emb = nn.Sequential(
                    nn.Linear(edge_dim, edge_dim * 2),
                    nn.GELU(),
                    nn.Linear(edge_dim * 2, edge_dim),
                    nn.LayerNorm(edge_dim),
                )

        # message passing layer
        if fuse_qkv:
            self.attn_mpnn = TransLayerOptim(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)
        else:
            self.attn_mpnn = TransLayer(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)

        if pair_update:
            self.node2edge_lin = nn.Linear(node_dim, edge_dim)
        # Feed forward block -> edge.
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)

        # equivariant edge update layer
        self.equi_pos = equi_pos
        if self.equi_pos:
            self.equi_update = CondEquiUpdate(node_dim, edge_dim, dist_dim, time_dim)

        if self.cond_time:
            self.node_time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, node_dim * 6)
            )
            # Normalization for MPNN
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

            if self.pair_update:
                self.edge_time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, edge_dim * 6)
                )
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            if self.pair_update:
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)


    def _ff_block_node(self, x):
        x = uncompilable_dropout(self.act1(self.ff_linear1(x)), p=self.dropout, training=self.training)
        return uncompilable_dropout(self.ff_linear2(x), p=self.dropout, training=self.training)

    def _ff_block_edge(self, x):
        x = uncompilable_dropout(self.act2(self.ff_linear3(x)), p=self.dropout, training=self.training)
        return uncompilable_dropout(self.ff_linear4(x), p=self.dropout, training=self.training)

    def forward_old(self, pos, h, edge_attr, edge_index, node_mask, node_time_emb=None, edge_time_emb=None):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        if self.dist_emb:
            distance = coord2dist(pos, edge_index)
            distance = self.dist_layer(distance, edge_time_emb)
            edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))
        else:
            edge_attr = self.edge_emb(edge_attr)

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)

            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        _h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
                self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(_h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(_h_node)) * node_mask

        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        _h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge) if self.cond_time else \
                    h_edge + self._ff_block_edge(_h_edge)

        # apply equivariant coordinate update
        if self.equi_pos:
            pos = self.equi_update(h_out, pos, edge_index, h_edge_out, distance, edge_time_emb)

        return h_out, h_edge_out, pos


    def forward(self, pos, h, edge_attr, edge_index, node_mask, node_time_emb=None, edge_time_emb=None):
        """
        A more optimized version of forward_old using torch.compile
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        if self.dist_emb:
            distance = coord2dist(pos, edge_index)
            distance = self.dist_layer(distance, edge_time_emb)
            edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))
        else:
            edge_attr = self.edge_emb(edge_attr)

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            if self.pair_update:
                edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                    self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)
                edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            if self.pair_update:
                edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_out = self.node_update(h_in_node, h_node, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp, node_mask)

        if self.pair_update:
            h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
            h_edge_out = self.edge_update(h_in_edge, h_edge, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp)
        else:
            h_edge_out = h_in_edge

        # apply equivariant coordinate update
        if self.equi_pos:
            pos = self.equi_update(h_out, pos, edge_index, h_edge_out, distance, edge_time_emb)

        return h_out, h_edge_out, pos

    @torch.compile(dynamic=True, disable=disable_compile)
    def node_update(self, h_in_node, h_node, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp, node_mask):
        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        _h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
                self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(_h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(_h_node)) * node_mask
        return h_out

    @torch.compile(dynamic=True, disable=disable_compile)
    def edge_update(self, h_in_edge, h_edge, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp):
        h_edge = self.node2edge_lin(h_edge)
        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        _h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge) if self.cond_time else \
                    h_edge + self._ff_block_edge(_h_edge)
        return h_edge_out
    
class Meta2NodesProjector(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, proj_type='mlp'):
        super().__init__()
        if proj_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, out_features),
            )
        elif proj_type == 'linear':
            self.mlp = nn.Linear(in_features, out_features)
        elif proj_type == 'none':
            self.mlp = nn.Identity()
        else:
            raise ValueError(f'Unknown proj_type {proj_type}')

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # transpose to (batch, features, seq_len) to apply MLP on seq_len
        x = x.transpose(1, 2)
        x = self.mlp(x)
        # transpose back to (batch, new_seq_len, features)
        x = x.transpose(1, 2)
        return x


class ExtendedProjector(nn.Module):
    """Extend an existing projector with a new input."""

    def __init__(self, projector, extend_dim, hidden_dim, disable_extra_gelu):
        super().__init__()
        # the following is weight tying
        self.linear1 = projector[0]
        self.act = projector[1]
        self.linear2 = projector[2]
        self.disable_extra_gelu = disable_extra_gelu
        if self.disable_extra_gelu:
            self.projector = nn.Sequential(
                nn.Linear(extend_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, self.linear1.out_features),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(extend_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, self.linear1.out_features),
                nn.GELU(),
            )

    def forward(self, x, new_x):
        x = self.linear1(x) + self.projector(new_x)
        return self.linear2(self.act(x))




class DGTDiffusion(nn.Module):
    """V2: Predict pos noise in the last block."""

    @staticmethod
    def add_args(parser):
        parser.add_argument('--in_node_features', type=int, default=44)
        parser.add_argument('--in_edge_features', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--n_blocks', type=int, default=10)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--enable_equiv', action='store_true', default=False)
        parser.add_argument('--use_original_dgt', action='store_true', default=False)
        parser.add_argument('--pred_noise', action='store_true', default=True)
        parser.add_argument('--mlp_ratio', type=int, default=4)
        parser.add_argument('--disable_com', action='store_true', default=True)
        parser.add_argument('--trans_linear', action='store_true', default=True)
        parser.add_argument('--disable_extra_gelu', action='store_true', default=False)
        parser.add_argument('--not_pair_update', action='store_true', default=False)
        parser.add_argument('--fuse_qkv', action='store_true', default=False)

    def __init__(self, args, in_dim=None):
        super().__init__()
        self.args = args
        self.pred_noise = args.pred_noise
        self.use_original_dgt = args.use_original_dgt
        self.disable_com = args.disable_com
        self.disable_extra_gelu = args.disable_extra_gelu
        self.pair_update = not args.not_pair_update

        time_dim = args.hidden_size
        hidden_dim = args.hidden_size
        edge_dim = hidden_dim // 4
        self.n_blocks = args.n_blocks

        # noise level conditioning embedding
        learned_dim = 16
        sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(learned_dim + 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Conditional MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_lin = nn.Linear(hidden_dim, time_dim) # multiple condition: hidden_dim -> cond_ch * hidden_dim

        # distance GBF embedding
        self.dist_gbf = GaussianLayer(edge_dim)

        self.enable_equiv = args.enable_equiv
        self.delta_train = self.args.delta_train
        self.use_llm = self.args.use_llm
        self.use_meta_projector = self.args.use_meta_projector
        self.num_metaqueries = self.args.num_metaqueries
        self.use_adameta = self.args.use_adameta

        # initial mapping
        if self.enable_equiv:
            self.node_emb = nn.Linear(args.in_node_features, hidden_dim)
        else:
            if self.use_llm and not self.delta_train:
                if self.use_meta_projector:
                    self.node_emb = nn.Sequential(
                        nn.Linear(args.in_node_features + 3, 2 * hidden_dim),
                        nn.GELU(),
                        nn.Linear(2 * hidden_dim, hidden_dim),
                    )
                else:
                    self.node_emb = nn.Sequential(
                        # in_dim+args.in_node_features是因为NExT-Mol的原实现中，将lm_x和node_h拼接起来了
                        nn.Linear(in_dim + args.in_node_features + 3, 2 * hidden_dim),
                        nn.GELU(),
                        nn.Linear(2 * hidden_dim, hidden_dim),
                    )
            else:
                self.node_emb = nn.Sequential(
                        nn.Linear(args.in_node_features + 3, hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, hidden_dim)
                    )

        self.edge_emb = nn.Linear(args.in_edge_features + edge_dim, edge_dim)
        for i in range(self.n_blocks):
            self.add_module(f'block_{i}', EquivariantBlock(hidden_dim, edge_dim, time_dim,
                            args.n_heads, dropout=args.dropout, dist_emb=self.use_original_dgt, equi_pos=self.use_original_dgt, mlp_ratio=args.mlp_ratio, act=nn.GELU, pair_update=self.pair_update, fuse_qkv=args.fuse_qkv))

        if self.use_original_dgt:
            assert self.enable_equiv
            assert not self.pred_noise
        else:
            if self.enable_equiv:
                # last block for predicting pos noise
                self.dist_gbf2 = GaussianLayer(edge_dim)
                self.pred_pos_noise = CondEquiUpdate(hidden_dim, edge_dim, edge_dim, time_dim, residual=False)
            else:
                self.final_linear = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 3, bias=False)
                )


        self.llm_cond = self.args.llm_cond
        if self.use_llm:
            if self.llm_cond:
                if self.use_meta_projector:
                    # qm9_max_n_nodes=29, geom_drug_max_n_nodes=181，对应设置max_n_nodes=32,184
                    self.projector = Meta2NodesProjector(self.num_metaqueries, hidden_dim, self.args.max_n_nodes, proj_type=self.args.meta2nodes_proj_type)
                else:
                    self.projector = nn.Sequential(
                        nn.Linear(in_dim, 4 * hidden_dim),
                        nn.GELU(),
                        nn.Linear(4 * hidden_dim, hidden_dim),
                    )
            else:
                if self.delta_train:
                    self.extended_node_emb = ExtendedProjector(self.node_emb, in_dim, hidden_dim, disable_extra_gelu=self.disable_extra_gelu)

    def forward(self, data, lm_x=None, context=None):
        # sparse to dense format: node_h, node_mask, pos, t_cond, edge_h, edge_mask
        if self.enable_equiv:
            node_h, node_mask = pyg_utils.to_dense_batch(data.x, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, node_nf], [B, N]
            pos, _ = pyg_utils.to_dense_batch(data.pos, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, 3]
            t_cond, _ = pyg_utils.to_dense_batch(data.t_cond, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N]
            edge_h = pyg_utils.to_dense_adj(data.edge_index, data.batch, data.edge_attr, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, N, edge_nf]
        else:
            x = torch.cat((data.x, data.pos, data.t_cond.reshape(-1, 1)), dim=-1)
            dense_x, node_mask = pyg_utils.to_dense_batch(x, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, node_nf], [B, N]
            node_h, pos, t_cond = dense_x[:, :, :-1], dense_x[:, :, -4:-1], dense_x[:, :, -1]
            edge_h = pyg_utils.to_dense_adj(data.edge_index, data.batch, data.edge_attr, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, N, edge_nf]

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # [B, N, N]
        bs, n_nodes = node_mask.size()
        dense_index = edge_mask.nonzero(as_tuple=True)
        edge_h = edge_h[dense_index]
        edge_index, _ = pyg_utils.dense_to_sparse(edge_mask)

        # obtain conditional feature (noise level)
        time_emb = self.time_mlp(t_cond[:,0])  # [B, time_dim]

        if context is not None:
            condition = context.unsqueeze(-1)
            condition = self.cond_lin(self.cond_mlp(condition).reshape(bs, -1))
            time_emb = time_emb + condition

        node_time_emb = time_emb.unsqueeze(1).expand(bs, n_nodes, -1).reshape(bs*n_nodes, -1)
        edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
        edge_time_emb = time_emb[edge_batch_id]  # only keep valid edge

        # add distance to edge feature
        pos = pos.reshape(bs * n_nodes, -1)
        distance = coord2dist(pos, edge_index)
        dist_emb = self.dist_gbf(distance)
        edge_h = torch.cat([edge_h, dist_emb], dim=-1)

        # lm_x: [batch_size, rdmol_len, hidden_size]
        if self.use_llm:
            # llm_cond是控制lm_x怎么注入的：如果llm_cond=True，那么lm_x就直接注入到node_cond中；如果llm_cond=False，那么lm_x就在输入层和node_h concat起来
            # NExT-Mol原始实现是直接和node_h concat起来(不管是de novo还是conditional generation)
            # MetaQuery实现是注入到cond中
            if self.llm_cond:
                node_h = self.node_emb(node_h).reshape(bs * n_nodes, -1)
                if self.use_meta_projector:

                    adapted_lm_x = lm_x if self.use_adameta else self.projector(lm_x)[:, :n_nodes, :]
                    node_cond = adapted_lm_x.reshape(bs * n_nodes, -1)
                else:
                    node_cond = self.projector(lm_x).reshape(bs * n_nodes, -1)
                edge_cond = edge_time_emb
                node_cond = node_cond + node_time_emb
            else:
                if self.delta_train:
                    node_h = self.extended_node_emb(node_h, lm_x).reshape(bs * n_nodes, -1)
                else:
                    node_h = self.node_emb(torch.cat([node_h, lm_x], dim=-1)).reshape(bs * n_nodes, -1)
                node_cond = node_time_emb
                edge_cond = edge_time_emb
        else:
            node_h = self.node_emb(node_h).reshape(bs * n_nodes, -1)
            node_cond = node_time_emb
            edge_cond = edge_time_emb

        edge_h = self.edge_emb(edge_h)

        # run the equivariant block
        for i in range(self.n_blocks):
            node_h, edge_h, pos = self._modules[f'block_{i}'](pos, node_h, edge_h, edge_index, node_mask.reshape(-1, 1),
                                                              node_cond, edge_cond)

        if self.use_original_dgt:
            pos = remove_mean_with_mask(pos.reshape(bs, n_nodes, -1), node_mask.unsqueeze(-1))
            pos = pos.reshape(bs * n_nodes, -1)[node_mask.reshape(-1)]
            pred_noise = (data.pos - pos.detach() * data.alpha_t) / data.sigma_t
            return pos, pred_noise

        # last block for predicting pos noise
        if self.enable_equiv:
            dist_last = self.dist_gbf2(distance)
            pred_noise = self.pred_pos_noise(node_h, pos, edge_index, edge_h, dist_last, edge_time_emb)
        else:
            pred_noise = self.final_linear(node_h)

        # pyg dense to sparse
        pred_noise = pred_noise.reshape(bs * n_nodes, -1)[node_mask.reshape(-1)]
        if not self.disable_com:
            pred_noise = remove_mean(pred_noise, data.batch)
        pred_pos = (data.pos - pred_noise.detach() * data.sigma_t) / data.alpha_t

        if context is not None:
            atom_type, _ = pyg_utils.to_dense_batch(data.atom_type, data.batch) # [B, N]
            h0 = F.one_hot(atom_type, num_classes=5).float()
            h0 = h0.reshape(-1, 5).to(pos.device, torch.float32)

            charges, _ = pyg_utils.to_dense_batch(data.charge, data.batch) # [B, N]

            atom_mask = charges > 0 # [B, N]
            node_mask = atom_mask
            edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2) # [B, N, N]
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(pos.device)
            edge_mask *= diag_mask

            # included_species = torch.unique(charges, sorted=True)[-5:]
            # # if included_species[0] == 0:
            # #     included_species = included_species[1:]
            # one_hot = charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
            # h0 = one_hot.to(pos.device, torch.float32).reshape(bs * n_nodes, -1)

            edges_dic = {}
            def get_adj_matrix(n_nodes, batch_size, device):
                if n_nodes in edges_dic:
                    edges_dic_b = edges_dic[n_nodes]
                    if batch_size in edges_dic_b:
                        return edges_dic_b[batch_size]
                    else:
                        # get edges for a single sample
                        rows, cols = [], []
                        for batch_idx in range(batch_size):
                            for i in range(n_nodes):
                                for j in range(n_nodes):
                                    rows.append(i + batch_idx * n_nodes)
                                    cols.append(j + batch_idx * n_nodes)
                else:
                    edges_dic[n_nodes] = {}
                    return get_adj_matrix(n_nodes, batch_size, device)

                edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
                return edges
            full_edges = get_adj_matrix(n_nodes, bs, pos.device)
            return pred_pos, pred_noise, [h0, pos, full_edges, None, node_mask.reshape(bs * n_nodes, -1), edge_mask.view(bs * n_nodes * n_nodes, 1), n_nodes]

        return pred_pos, pred_noise
