import torch
import torch.nn.functional as F
from model.unimol import UniMolModel, UniMolModelV2, UniMolModelV4
from typing import Optional


class UnimolConfGModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--unimol-distance-loss",
            type=float,
            default=1.0,
            help="weight for the distance loss",
        )
        parser.add_argument(
            "--unimol_distance_from_coord",
            action='store_true',
            default=True,
        )
        parser.add_argument(
            "--unimol-coord-loss",
            type=float,
            default=1.0,
            help="weight for the coordinate loss",
        )

        parser.add_argument(
            "--unimol-num-recycles",
            type=int,
            default=4,
            help="number of cycles to use for coordinate prediction",
        )

        parser.add_argument(
            "--unimol-encoder-layers",
            type=int,
            default=15,
            help="num encoder layers",
        )

        parser.add_argument(
            "--unimol-encoder-embed-dim",
            type=int,
            default=512,
            help="encoder embedding dimension",
        )

        parser.add_argument(
            "--unimol-encoder-ffn-embed-dim",
            type=int,
            default=2048,
            help="encoder embedding dimension for FFN",
        )

        parser.add_argument(
            "--unimol-encoder-attention-heads",
            type=int,
            default=64,
            help="num encoder attention heads",
        )

        parser.add_argument(
            "--unimol-dropout",
            type=float,
            default=0.1,
            help="dropout probability",
        )

        parser.add_argument(
            "--unimol-emb-dropout",
            type=float,
            default=0.1,
            help="dropout probability for embeddings",
        )

        parser.add_argument(
            "--unimol-attention-dropout",
            type=float,
            default=0.1,
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--unimol-activation-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-pooled-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-max-seq-len",
            type=int,
            default=512,
            help="number of positional embeddings to learn",
        )

        parser.add_argument(
            "--unimol-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="gelu",
        )

        parser.add_argument(
            "--unimol-pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
            default="tanh",
        )

        parser.add_argument(
            "--unimol-post-ln",
            type=bool,
            help="use post layernorm or pre layernorm",
            default=False,
        )

        parser.add_argument(
            "--unimol-masked-token-loss",
            type=float,
            help="mask loss ratio",
            default=-1.0,
        )

        parser.add_argument(
            "--unimol-masked-dist-loss",
            type=float,
            default=1.0,
            help="masked distance loss ratio",
        )

        ## additional arguments required by unimol model
        parser.add_argument(
            "--unimol-delta-pair-repr-norm-loss",
            type=float,
            default=-1.0,
            help="delta encoder pair repr norm loss ratio",
        )

        parser.add_argument(
            "--unimol-masked-coord-loss",
            type=float,
            default=1.0,
            help="masked coord loss ratio",
        )

    def __init__(self, args, mol_dictionary):
        super().__init__()
        self.args = args
        self.unimol = UniMolModel(self.args, mol_dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        encoded_atom_x=None,
        **kwargs
    ):
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimol.gbf(dist, et)
            gbf_result = self.unimol.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        def single_encoder(
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
        ):
            x = self.unimol.encoder.emb_layer_norm(emb)
            x = F.dropout(x, p=self.unimol.encoder.emb_dropout, training=self.training)

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask, padding_mask = fill_attn_mask(
                attn_mask, padding_mask, fill_val=float("-inf")
            )

            for i in range(len(self.unimol.encoder.layers)):
                x, attn_mask, _ = self.unimol.encoder.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

            return x, attn_mask

        padding_mask = src_tokens.eq(self.unimol.padding_idx)
        input_padding_mask = padding_mask

        if encoded_atom_x is None:
            x = self.unimol.embed_tokens(src_tokens)
        else:
            x = encoded_atom_x

        attn_mask = get_dist_features(src_distance, src_edge_type)
        input_attn_mask = attn_mask
        bsz = x.size(0)
        seq_len = x.size(1)

        for _ in range(self.args.unimol_num_recycles):
            x, attn_mask = single_encoder(
                x, padding_mask=padding_mask, attn_mask=attn_mask
            )

        if self.unimol.encoder.final_layer_norm is not None:
            x = self.unimol.encoder.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask, _ = fill_attn_mask(attn_mask, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        distance_predict, coords_predict = None, None

        if self.args.unimol_coord_loss > 0 or True:
            if padding_mask is not None:
                atom_num = (torch.sum(~padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            else:
                atom_num = src_coord.shape[1] - 1
            delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
            attn_probs = self.unimol.pair2coord_proj(delta_pair_repr)
            coords_update = delta_pos / atom_num * attn_probs
            coords_update = torch.sum(coords_update, dim=2)
            coords_predict = src_coord + coords_update # [bsz, n_node, 3]

        if self.args.unimol_distance_loss > 0 or True:
            if self.args.unimol_distance_from_coord:
                distance_predict = torch.cdist(coords_predict, coords_predict, p=2)
            else:
                distance_predict = self.unimol.dist_head(attn_mask)

        return [distance_predict, coords_predict]


class UnimolConfGModelV2(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--unimol-distance-loss",
            type=float,
            default=1.0,
            help="weight for the distance loss",
        )
        parser.add_argument(
            "--unimol_distance_from_coord",
            action='store_true',
            default=True,
        )
        parser.add_argument(
            "--unimol-coord-loss",
            type=float,
            default=1.0,
            help="weight for the coordinate loss",
        )

        parser.add_argument(
            "--unimol-num-recycles",
            type=int,
            default=4,
            help="number of cycles to use for coordinate prediction",
        )

        parser.add_argument(
            "--unimol-encoder-layers",
            type=int,
            default=15,
            help="num encoder layers",
        )

        parser.add_argument(
            "--unimol-encoder-embed-dim",
            type=int,
            default=512,
            help="encoder embedding dimension",
        )

        parser.add_argument(
            "--unimol-encoder-ffn-embed-dim",
            type=int,
            default=2048,
            help="encoder embedding dimension for FFN",
        )

        parser.add_argument(
            "--unimol-encoder-attention-heads",
            type=int,
            default=64,
            help="num encoder attention heads",
        )

        parser.add_argument(
            "--unimol-dropout",
            type=float,
            default=0.1,
            help="dropout probability",
        )

        parser.add_argument(
            "--unimol-emb-dropout",
            type=float,
            default=0.1,
            help="dropout probability for embeddings",
        )

        parser.add_argument(
            "--unimol-attention-dropout",
            type=float,
            default=0.1,
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--unimol-activation-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-pooled-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-max-seq-len",
            type=int,
            default=512,
            help="number of positional embeddings to learn",
        )

        parser.add_argument(
            "--unimol-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="gelu",
        )

        parser.add_argument(
            "--unimol-pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
            default="tanh",
        )

        parser.add_argument(
            "--unimol-post-ln",
            type=bool,
            help="use post layernorm or pre layernorm",
            default=False,
        )

        parser.add_argument(
            "--unimol-masked-token-loss",
            type=float,
            help="mask loss ratio",
            default=-1.0,
        )

        parser.add_argument(
            "--unimol-masked-dist-loss",
            type=float,
            default=1.0,
            help="masked distance loss ratio",
        )

        ## additional arguments required by unimol model
        parser.add_argument(
            "--unimol-delta-pair-repr-norm-loss",
            type=float,
            default=-1.0,
            help="delta encoder pair repr norm loss ratio",
        )

        parser.add_argument(
            "--unimol-masked-coord-loss",
            type=float,
            default=1.0,
            help="masked coord loss ratio",
        )

    def __init__(self, args, mol_dictionary):
        super().__init__()
        self.args = args
        self.unimol = UniMolModelV2(self.args, mol_dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        encoded_atom_x=None,
        **kwargs
    ):
        def get_dist_features(dist, et, bt):
            n_node = dist.size(-1)
            gbf_feature = self.unimol.gbf(dist, et)
            gbf_feature_bt = self.unimol.gbf_bt(dist, bt)
            gbf_result = self.unimol.gbf_proj(gbf_feature)
            gbf_bt_result = self.unimol.gbf_proj_bt(gbf_feature_bt)
            graph_attn_bias = gbf_result + gbf_bt_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        def single_encoder(
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
        ):
            x = self.unimol.encoder.emb_layer_norm(emb)
            x = F.dropout(x, p=self.unimol.encoder.emb_dropout, training=self.training)

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask, padding_mask = fill_attn_mask(
                attn_mask, padding_mask, fill_val=float("-inf")
            )

            for i in range(len(self.unimol.encoder.layers)):
                x, attn_mask, _ = self.unimol.encoder.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

            return x, attn_mask

        padding_mask = src_tokens.eq(self.unimol.padding_idx)
        input_padding_mask = padding_mask

        if encoded_atom_x is None:
            x = self.unimol.embed_tokens(src_tokens)
        else:
            x = encoded_atom_x

        attn_mask = get_dist_features(src_distance, src_edge_type, src_bond_type)
        input_attn_mask = attn_mask
        bsz = x.size(0)
        seq_len = x.size(1)

        for _ in range(self.args.unimol_num_recycles):
            x, attn_mask = single_encoder(
                x, padding_mask=padding_mask, attn_mask=attn_mask
            )

        if self.unimol.encoder.final_layer_norm is not None:
            x = self.unimol.encoder.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask, _ = fill_attn_mask(attn_mask, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        distance_predict, coords_predict = None, None

        if self.args.unimol_coord_loss > 0 or True:
            if padding_mask is not None:
                atom_num = (torch.sum(~padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            else:
                atom_num = src_coord.shape[1] - 1
            delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
            attn_probs = self.unimol.pair2coord_proj(delta_pair_repr)
            coords_update = delta_pos / atom_num * attn_probs
            coords_update = torch.sum(coords_update, dim=2)
            coords_predict = src_coord + coords_update # [bsz, n_node, 3]

        if self.args.unimol_distance_loss > 0 or True:
            if self.args.unimol_distance_from_coord:
                distance_predict = torch.cdist(coords_predict, coords_predict, p=2)
            else:
                distance_predict = self.unimol.dist_head(attn_mask)

        return [distance_predict, coords_predict]


class UnimolConfGModelV3(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--unimol-distance-loss",
            type=float,
            default=1.0,
            help="weight for the distance loss",
        )
        parser.add_argument(
            "--unimol_distance_from_coord",
            action='store_true',
            default=True,
        )
        parser.add_argument(
            "--unimol-coord-loss",
            type=float,
            default=1.0,
            help="weight for the coordinate loss",
        )

        parser.add_argument(
            "--unimol-num-recycles",
            type=int,
            default=4,
            help="number of cycles to use for coordinate prediction",
        )

        parser.add_argument(
            "--unimol-encoder-layers",
            type=int,
            default=15,
            help="num encoder layers",
        )

        parser.add_argument(
            "--unimol-encoder-embed-dim",
            type=int,
            default=512,
            help="encoder embedding dimension",
        )

        parser.add_argument(
            "--unimol-encoder-ffn-embed-dim",
            type=int,
            default=2048,
            help="encoder embedding dimension for FFN",
        )

        parser.add_argument(
            "--unimol-encoder-attention-heads",
            type=int,
            default=64,
            help="num encoder attention heads",
        )

        parser.add_argument(
            "--unimol-dropout",
            type=float,
            default=0.1,
            help="dropout probability",
        )

        parser.add_argument(
            "--unimol-emb-dropout",
            type=float,
            default=0.1,
            help="dropout probability for embeddings",
        )

        parser.add_argument(
            "--unimol-attention-dropout",
            type=float,
            default=0.1,
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--unimol-activation-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-pooled-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-max-seq-len",
            type=int,
            default=512,
            help="number of positional embeddings to learn",
        )

        parser.add_argument(
            "--unimol-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="gelu",
        )

        parser.add_argument(
            "--unimol-pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
            default="tanh",
        )

        parser.add_argument(
            "--unimol-post-ln",
            type=bool,
            help="use post layernorm or pre layernorm",
            default=False,
        )

        parser.add_argument(
            "--unimol-masked-token-loss",
            type=float,
            help="mask loss ratio",
            default=-1.0,
        )

        parser.add_argument(
            "--unimol-masked-dist-loss",
            type=float,
            default=1.0,
            help="masked distance loss ratio",
        )

        ## additional arguments required by unimol model
        parser.add_argument(
            "--unimol-delta-pair-repr-norm-loss",
            type=float,
            default=-1.0,
            help="delta encoder pair repr norm loss ratio",
        )

        parser.add_argument(
            "--unimol-masked-coord-loss",
            type=float,
            default=1.0,
            help="masked coord loss ratio",
        )

    def __init__(self, args, mol_dictionary):
        super().__init__()
        self.args = args
        self.unimol = UniMolModel(self.args, mol_dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        encoded_atom_x=None,
        **kwargs
    ):
        def get_dist_features(dist, et, bt):
            n_node = dist.size(-1)
            et = et * 5 + bt # 5 stands for the number of bond types
            gbf_feature = self.unimol.gbf(dist, et)
            gbf_result = self.unimol.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        def single_encoder(
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
        ):
            x = self.unimol.encoder.emb_layer_norm(emb)
            x = F.dropout(x, p=self.unimol.encoder.emb_dropout, training=self.training)

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask, padding_mask = fill_attn_mask(
                attn_mask, padding_mask, fill_val=float("-inf")
            )

            for i in range(len(self.unimol.encoder.layers)):
                x, attn_mask, _ = self.unimol.encoder.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

            return x, attn_mask

        padding_mask = src_tokens.eq(self.unimol.padding_idx)
        input_padding_mask = padding_mask

        if encoded_atom_x is None:
            x = self.unimol.embed_tokens(src_tokens)
        else:
            x = encoded_atom_x

        attn_mask = get_dist_features(src_distance, src_edge_type, src_bond_type)
        input_attn_mask = attn_mask
        bsz = x.size(0)
        seq_len = x.size(1)

        for _ in range(self.args.unimol_num_recycles):
            x, attn_mask = single_encoder(
                x, padding_mask=padding_mask, attn_mask=attn_mask
            )

        if self.unimol.encoder.final_layer_norm is not None:
            x = self.unimol.encoder.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask, _ = fill_attn_mask(attn_mask, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        distance_predict, coords_predict = None, None

        if self.args.unimol_coord_loss > 0 or True:
            if padding_mask is not None:
                atom_num = (torch.sum(~padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            else:
                atom_num = src_coord.shape[1] - 1
            delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
            attn_probs = self.unimol.pair2coord_proj(delta_pair_repr)
            coords_update = delta_pos / atom_num * attn_probs
            coords_update = torch.sum(coords_update, dim=2)
            coords_predict = src_coord + coords_update # [bsz, n_node, 3]

        if self.args.unimol_distance_loss > 0 or True:
            if self.args.unimol_distance_from_coord:
                distance_predict = torch.cdist(coords_predict, coords_predict, p=2)
            else:
                distance_predict = self.unimol.dist_head(attn_mask)

        return [distance_predict, coords_predict]



class UnimolConfGModelV4(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--unimol-distance-loss",
            type=float,
            default=1.0,
            help="weight for the distance loss",
        )
        parser.add_argument(
            "--unimol_distance_from_coord",
            action='store_true',
            default=True,
        )
        parser.add_argument(
            "--unimol-coord-loss",
            type=float,
            default=1.0,
            help="weight for the coordinate loss",
        )

        parser.add_argument(
            "--unimol-num-recycles",
            type=int,
            default=4,
            help="number of cycles to use for coordinate prediction",
        )

        parser.add_argument(
            "--unimol-encoder-layers",
            type=int,
            default=15,
            help="num encoder layers",
        )

        parser.add_argument(
            "--unimol-encoder-embed-dim",
            type=int,
            default=512,
            help="encoder embedding dimension",
        )

        parser.add_argument(
            "--unimol-encoder-ffn-embed-dim",
            type=int,
            default=2048,
            help="encoder embedding dimension for FFN",
        )

        parser.add_argument(
            "--unimol-encoder-attention-heads",
            type=int,
            default=64,
            help="num encoder attention heads",
        )

        parser.add_argument(
            "--unimol-dropout",
            type=float,
            default=0.1,
            help="dropout probability",
        )

        parser.add_argument(
            "--unimol-emb-dropout",
            type=float,
            default=0.1,
            help="dropout probability for embeddings",
        )

        parser.add_argument(
            "--unimol-attention-dropout",
            type=float,
            default=0.1,
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--unimol-activation-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-pooled-dropout",
            type=float,
            default=0.0,
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--unimol-max-seq-len",
            type=int,
            default=512,
            help="number of positional embeddings to learn",
        )

        parser.add_argument(
            "--unimol-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="gelu",
        )

        parser.add_argument(
            "--unimol-pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
            default="tanh",
        )

        parser.add_argument(
            "--unimol-post-ln",
            type=bool,
            help="use post layernorm or pre layernorm",
            default=False,
        )

        parser.add_argument(
            "--unimol-masked-token-loss",
            type=float,
            help="mask loss ratio",
            default=-1.0,
        )

        parser.add_argument(
            "--unimol-masked-dist-loss",
            type=float,
            default=1.0,
            help="masked distance loss ratio",
        )

        ## additional arguments required by unimol model
        parser.add_argument(
            "--unimol-delta-pair-repr-norm-loss",
            type=float,
            default=-1.0,
            help="delta encoder pair repr norm loss ratio",
        )

        parser.add_argument(
            "--unimol-masked-coord-loss",
            type=float,
            default=1.0,
            help="masked coord loss ratio",
        )

    def __init__(self, args, mol_dictionary):
        super().__init__()
        self.args = args
        self.unimol = UniMolModelV4(self.args, mol_dictionary)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        encoded_atom_x=None,
        **kwargs
    ):
        def get_dist_features(dist, et, bt):
            n_node = dist.size(-1)
            et = et * 5 + bt # 5 stands for the number of bond types
            gbf_feature = self.unimol.gbf(dist, et)
            gbf_result = self.unimol.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        def single_encoder(
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None,
            bond_type: Optional[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None,
        ):
            x = self.unimol.encoder.emb_layer_norm(emb)
            x = F.dropout(x, p=self.unimol.encoder.emb_dropout, training=self.training)

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask, padding_mask = fill_attn_mask(
                attn_mask, padding_mask, fill_val=float("-inf")
            )

            for i in range(len(self.unimol.encoder.layers)):
                x, attn_mask, _ = self.unimol.encoder.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, edge_index=edge_index, bond_type=bond_type, batch=batch, return_attn=True,
                )

            return x, attn_mask

        def get_edge_index_and_bond_type(bond_type):
            bsz, max_node, max_node = bond_type.shape
            offset_, row_, col_ = (bond_type > 0).nonzero().t() # (E)
            bond_type = bond_type[offset_, row_, col_] # (E)
            row = row_ + offset_ * max_node
            col = col_ + offset_ * max_node
            edge_index = torch.stack([row, col], dim=0) # (2, E)

            batch = torch.arange(bsz, device=bond_type.device).repeat_interleave(max_node)
            return edge_index, bond_type, batch, (offset_, row_, col_)

        edge_index, bond_type, batch, orc = get_edge_index_and_bond_type(src_bond_type)

        edge_index, bond_type, batch = get_edge_index_and_bond_type(src_bond_type)

        padding_mask = src_tokens.eq(self.unimol.padding_idx)
        input_padding_mask = padding_mask

        if encoded_atom_x is None:
            x = self.unimol.embed_tokens(src_tokens)
        else:
            x = encoded_atom_x

        attn_mask = get_dist_features(src_distance, src_edge_type, src_bond_type)
        input_attn_mask = attn_mask
        bsz = x.size(0)
        seq_len = x.size(1)

        for _ in range(self.args.unimol_num_recycles):
            x, attn_mask = single_encoder(
                x, padding_mask=padding_mask, attn_mask=attn_mask, edge_index=edge_index, bond_type=bond_type, batch=batch, orc=orc
            )

        if self.unimol.encoder.final_layer_norm is not None:
            x = self.unimol.encoder.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask, _ = fill_attn_mask(attn_mask, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        distance_predict, coords_predict = None, None

        if self.args.unimol_coord_loss > 0 or True:
            if padding_mask is not None:
                atom_num = (torch.sum(~padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
            else:
                atom_num = src_coord.shape[1] - 1
            delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
            attn_probs = self.unimol.pair2coord_proj(delta_pair_repr)
            coords_update = delta_pos / atom_num * attn_probs
            coords_update = torch.sum(coords_update, dim=2)
            coords_predict = src_coord + coords_update # [bsz, n_node, 3]

        if self.args.unimol_distance_loss > 0 or True:
            if self.args.unimol_distance_from_coord:
                distance_predict = torch.cdist(coords_predict, coords_predict, p=2)
            else:
                distance_predict = self.unimol.dist_head(attn_mask)

        return [distance_predict, coords_predict]

if __name__ == '__main__':
    import os
    from pathlib import Path
    import argparse
    dict_path = Path(os.path.realpath(__file__)).parent.parent / 'data_provider' / 'unimol_dict.txt'
    dictionary = Dictionary.load(str(dict_path))
    dictionary.add_symbol("[MASK]", is_special=True)
    parser = argparse.ArgumentParser()
    UnimolConfGModel.add_args(parser)
    args = parser.parse_args()
    model = UnimolConfGModel(args, dictionary)
    state_dict = torch.load('unimol_ckpt/mol_pre_no_h_220816.pt')['model']
    state_dict.pop('encoder.final_head_layer_norm.weight')
    state_dict.pop('encoder.final_head_layer_norm.bias')
    model.unimol.load_state_dict(state_dict)
