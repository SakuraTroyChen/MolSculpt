import math
import random
import os
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
from torch import optim
import lightning as L
from transformers import AutoTokenizer,Qwen2Config
from model.help_funcs import AttrDict
from model.diffusion_model_dgt import DGTDiffusion, remove_mean_with_mask
from model.property_prediction.egnn import EGNN
from data_provider.conf_gen_cal_metrics import set_rdmol_positions, get_best_rmsd
import copy
from evaluation.eval_functions import get_2D_edm_metric, get_3D_edm_metric, get_moses_metrics
import numpy as np
from torch_geometric.utils import to_dense_batch
from data_provider.diffusion_data_module import sample_com_rand_pos
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import torch.distributed as dist
# from torch_scatter import scatter
from model.diffusion_model_dgt import remove_mean
from peft import get_peft_model, LoraConfig
from model.modeling_llama import LlamaForCausalLM
import pickle
from pathlib import Path
from diffusers.models.normalization import RMSNorm
from model.transformer_encoder import Qwen2Encoder


def to_dense_batch_list_tensor(list_of_tensor, batch, bs, max_num_nodes):
    shape = []
    for t in list_of_tensor:
        shape.append(t.shape[1])
    list_of_tensor = torch.cat(list_of_tensor, dim=1)
    to_dense_batch_list, batch_mask = to_dense_batch(list_of_tensor, batch, batch_size=bs, max_num_nodes=max_num_nodes) # shape = [batch_size, max_num_nodes, 3]
    return torch.split(to_dense_batch_list, shape, dim=2), batch_mask

disable_compile = torch.cuda.get_device_name(0).find('AMD') >= 0

def get_precision(precision):
    if precision in {'16', '16-mixed'}:
        return torch.float16
    elif precision in {'bf16', 'bf16-mixed'}:
        return torch.bfloat16
    elif precision in {'32', 'fp32', '32-true'}:
        return torch.float32
    else:
        print(precision)
        raise NotImplementedError

class LinearWarmupCosineLRSchedulerV2:
    def __init__(
        self,
        optimizer,
        max_iters,
        min_lr,
        init_lr,
        warmup_iters=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.lr_decay_iters = max_iters

    def get_lr(self, it):
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_iters:
            return self.init_lr * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.init_lr - self.min_lr)

    def step(self, cur_step):
        lr = self.get_lr(cur_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr



def get_half_precision_dtype():
    if not torch.cuda.is_available():
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float16


def set_embed_tokens_trainable(model):
    for name, param in model.named_parameters():
        if name.find('embed_tokens') >= 0:
            param.requires_grad = True
            print(name, 'requires_grad = True')


@torch.no_grad()
def kabsch_batch(coords_pred, coords_tar):
    '''
    coords_pred: [batch_size, num_nodes, 3]
    coords_tar: [batch_size, num_nodes, 3]
    '''
    """Batch version of Kabsch algorithm."""
    A = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar)
    A = A.to(torch.float32)
    U, S, Vt = torch.linalg.svd(A)
    sign_detA = torch.sign(torch.det(A))  # [batch_size]
    corr_mat_diag = torch.ones((A.size(0), U.size(-1)), device=A.device)  # [batch_size, 3]
    corr_mat_diag[:, -1] = sign_detA  # [batch_size, 3]
    corr_mat = torch.diag_embed(corr_mat_diag)  # [batch_size, 3, 3]
    rotation = torch.einsum("...ij, ...jk, ...kl -> ...il", U, corr_mat, Vt)  # [batch_size, 3, 3]

    return rotation

@torch.no_grad()
def get_align_pos(pos_t, pos_0):
    '''
    pos_t: [batch_size, num_nodes, 3]
    '''
    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
    return align_pos_0


@torch.no_grad()
def get_align_noise(pos_t, pos_0, pos_pred, alpha_t, sigma_t, batch_mask=None, translation_correction=False, align_prediction=False):
    if translation_correction:
        # center the coordinates with mask
        batch_mask = batch_mask.unsqueeze(-1)
        pos_0_centered, _ = remove_mean_with_mask(pos_0, batch_mask, return_mean=True)
        if align_prediction:
            pos_pred_centered, pos_pred_mean = remove_mean_with_mask(pos_pred, batch_mask, return_mean=True) # shape = [batch_size, num_nodes, 3], [batch_size, 1, 3]
            rotations = kabsch_batch(pos_pred_centered, pos_0_centered)  # [batch_size, 3, 3]
            align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_centered) + pos_pred_mean
        else:
            pos_t_centered, pos_t_mean = remove_mean_with_mask(pos_t, batch_mask, return_mean=True) # shape = [batch_size, num_nodes, 3], [batch_size, 1, 3]
            rotations = kabsch_batch(pos_t_centered, pos_0_centered)  # [batch_size, 3, 3]
            align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_centered) + pos_t_mean
        aligned_noise = (pos_t - alpha_t * align_pos_0) / sigma_t
        return aligned_noise
    else:
        if align_prediction:
            rotations = kabsch_batch(pos_pred, pos_0)  # [batch_size, 3, 3]
        else:
            rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
        align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
        aligned_noise = (pos_t - alpha_t * align_pos_0) / sigma_t
        return aligned_noise


class LLMProjector(nn.Module):
    def __init__(self, in_dim, hidden_size, llm_jk, use_self_att_proj, llm_num_layers):
        super().__init__()
        self.llm_jk = llm_jk
        self.use_self_att_proj = use_self_att_proj
        if self.llm_jk == 'mean':
            self.mean_weight = nn.Parameter(torch.zeros(1, llm_num_layers))
            self.mean_ln = nn.LayerNorm(in_dim)

        if self.use_self_att_proj:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True, norm_first=True, dropout=0.)
            self.self_att_proj = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.linear_proj = nn.Linear(in_dim, hidden_size)

    def forward(self, hidden_states, rdmol2selfies, selfies_batch):
        if self.llm_jk == 'last':
            lm_embeds = hidden_states[-1] # shape = [batch_size, seq_len, hidden_size]
        elif self.llm_jk == 'mean':
            lm_embeds = torch.stack(hidden_states[1:], dim=2) # shape = [batch_size, seq_len, num_layers, hidden_size]
            lm_embeds = (self.mean_weight.softmax(dim=-1) @ lm_embeds).squeeze(2) # shape = [batch_size, seq_len, hidden_size]
            lm_embeds = self.mean_ln(lm_embeds)
        else:
            raise NotImplementedError

        # 不用self_att_proj就是直接bmm到rdmols上
        # 用self_att_proj就是先linear proj到hidden_size，然后self att，最后bmm到rdmols上
        if self.use_self_att_proj:
            lm_embeds = self.self_att_proj(self.linear_proj(lm_embeds), src_key_padding_mask=~selfies_batch.attention_mask.bool())

        lm_x = torch.bmm(rdmol2selfies.to(lm_embeds.dtype), lm_embeds) # shape = [batch_size, rdmol_len, selfies_len], [batch_size, selfies_len, hidden_size] -> [batch_size, rdmol_len, hidden_size]
        norm = torch.clamp(torch.sum(rdmol2selfies, dim=-1, keepdim=True), min=1) # shape = [batch_size, 1, 1]
        lm_x = lm_x / norm # shape = [batch_size, rdmol_len, hidden_size]
        return lm_x
    
class MetaProjector(nn.Module):
    def __init__(self, connector_in_dim, connector_out_dim,num_meta_hidden_layers=24):
        super().__init__()
        rmsnorm = RMSNorm(connector_out_dim, eps=1e-5, elementwise_affine=True)
        # TODO: input_scale怎么取？
        input_scale = 1
        with torch.no_grad():
            rmsnorm.weight.fill_(input_scale)

        if num_meta_hidden_layers == 0:
            encoder = nn.Identity()
        else:
            encoder = Qwen2Encoder(
                Qwen2Config(
                    hidden_size=connector_in_dim,
                    intermediate_size=connector_in_dim * 4,
                    num_hidden_layers=num_meta_hidden_layers,
                    num_attention_heads=connector_in_dim // 64,
                    num_key_value_heads=connector_in_dim // 64,
                    initializer_range=0.014,
                    use_cache=False,
                    rope=True,
                    qk_norm=True,
                ),
            )
            
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(connector_in_dim, connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(connector_out_dim, connector_out_dim),
            rmsnorm,
        )
        if not isinstance(self.connector, nn.Identity):
            for module in self.connector:
                if isinstance(module, Qwen2Encoder):
                    module.gradient_checkpointing_enable({"use_reentrant": False})


    def forward(self, prompt_embeds):
        # [batch_size, num_metaqueries, hidden_size]
        lm_embeds = self.connector(prompt_embeds)
        return lm_embeds



class DiffussionPL(L.LightningModule):

    def __init__(self, args, tokenizer=None, max_sf_tokens=30, noise_scheduler=None, pos_std=None, property_normalizations=None, property_distribution=None):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.max_sf_tokens = max_sf_tokens
        self.noise_scheduler = noise_scheduler
        self.reduce_node_mean = not args.not_reduce_node_mean
        self.infer_time = args.infer_time
        self.pos_std = pos_std
        self.pred_noise = args.pred_noise
        self.align_loss = not args.not_align_loss
        self.t_cond = args.t_cond
        self.disable_com = args.disable_com
        self.llm_tune = args.llm_tune
        self.save_eval_only = args.save_eval_only
        self.translation_correction = args.translation_correction
        self.align_prediction = args.align_prediction
        self.use_self_att_proj = args.use_self_att_proj

        self.tokenizer = tokenizer if tokenizer is not None else self.init_tokenizer(args)
        self.use_llm = args.use_llm
        self.use_adameta = args.use_adameta
        in_dim = None
        
        if self.use_llm:
            self.llm_model = self.init_llm(args)
            self.resize_token_embeddings(self.tokenizer)

            self.llm_jk = args.llm_jk
            self.use_llm_projector = args.use_llm_projector
            self.use_meta_projector = args.use_meta_projector

            if self.use_meta_projector:
                # 获取原始词嵌入的数量
                num_embeddings = self.llm_model.get_input_embeddings().num_embeddings
                self.num_embeddings = num_embeddings
                
                # 如果配置中指定了 `num_metaqueries`（元查询数量）大于0，
                # 则扩展模型的词汇表，增加 `num_metaqueries + 2` 个新的 token。有两个token是bos和eos吧
                # 这通常用于添加任务相关的特殊 token。
                if args.num_metaqueries > 0:
                    try:
                        self.llm_model.resize_token_embeddings(
                            num_embeddings + args.num_metaqueries + 2
                        )
                    except:
                        # 兼容不同版本的 `resize_token_embeddings` 方法
                        self.llm_model.resize_token_embeddings(
                            num_embeddings + args.num_metaqueries + 2, mean_resizing=False
                        )

                # 冻结原始词嵌入
                # 定义一个 PyTorch hook 函数，它会在计算梯度时被调用。
                # `grad[: self.num_embeddings].zero_()` 会将原始词嵌入的梯度清零。
                def freeze_hook(grad):
                    grad[: self.num_embeddings].zero_()
                    return grad

                # 将 hook 注册到词嵌入层的权重上。
                # 这意味着在模型训练时，只有新添加的 token（metaqueries）的嵌入会被更新，
                # 而原始的预训练词嵌入保持不变（被冻结）。
                self.llm_model.model.embed_tokens.requires_grad_(True)
                self.llm_model.model.embed_tokens.weight.register_hook(
                    freeze_hook
                )

                # 获取模型隐藏层大小
                self.llm_hidden_size = self.llm_model.config.hidden_size

                # 替换语言模型头部
                # 将语言模型的头部（通常是用于预测下一个词的线性层）替换为一个恒等映射 (Identity)。
                # 这意味着模型的最终输出将是最后一层 Transformer 的隐藏状态（hidden states），
                # 而不是词汇表的 logits。这强烈表明该模型被用作特征提取器。
                self.llm_model.lm_head = nn.Identity()
                
                if args.num_metaqueries > 0:
                    tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
                    tokenizer.add_special_tokens(
                        {
                            "additional_special_tokens": [
                                f"<pad_token_{i}>"
                                for i in range(num_embeddings - len(tokenizer))
                            ]
                        }
                    )
                    tokenizer.add_special_tokens(
                        {
                            "additional_special_tokens": ["<begin_of_mol>", "<end_of_mol>"]
                            + [f"<mol{i}>" for i in range(args.num_metaqueries)]
                        }
                    )
                    self.bom_token_id = tokenizer.convert_tokens_to_ids("<begin_of_mol>")
                    self.eom_token_id = tokenizer.convert_tokens_to_ids("<end_of_mol>")

                    self.connector_in_dim = self.llm_model.config.hidden_size
                    self.connector_out_dim = args.hidden_size
                    in_dim = args.hidden_size
                    self.meta_projector = MetaProjector(self.connector_in_dim, self.connector_out_dim,self.args.num_meta_hidden_layers)

            elif self.use_llm_projector:
                self.llm_projector = LLMProjector(self.llm_model.config.hidden_size, args.hidden_size, self.llm_jk, self.use_self_att_proj, self.llm_model.config.num_hidden_layers)
                if self.use_self_att_proj:
                    in_dim = args.hidden_size
                else:
                    in_dim = self.llm_model.config.hidden_size
            else:
                if self.llm_jk == 'mean':
                    self.mean_weight = nn.Parameter(torch.zeros(1, self.llm_model.config.num_hidden_layers))
                    self.mean_ln = nn.LayerNorm(self.llm_model.config.hidden_size)

                in_dim = self.llm_model.config.hidden_size
                if self.use_self_att_proj:
                    encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, batch_first=True, norm_first=True, dropout=0.1)
                    self.self_att_proj = nn.TransformerEncoder(encoder_layer, num_layers=1)
                    self.linear_proj = nn.Linear(in_dim, args.hidden_size)
                    in_dim = args.hidden_size

            self.hidden_size = self.llm_model.config.hidden_size
            self.condition_mlp = nn.Sequential(
                nn.Linear(1, self.hidden_size * 4),
                nn.GELU(),
                nn.Linear(self.hidden_size * 4, 4 * self.hidden_size)
            )

            if args.llm_ckpt is not None: # have trained llm
                print(f"Loading llmpl model from: {args.llm_ckpt} ... ", end="")
                checkpoint_path = os.path.join(args.llm_ckpt, 'checkpoint/mp_rank_00_model_states.pt')
                llm_ckpt = torch.load(checkpoint_path, map_location='cpu')

                with torch.no_grad():
                    self.condition_mlp[0].weight.copy_(llm_ckpt['module']['condition_mlp.0.weight'])
                    self.condition_mlp[0].bias.copy_(llm_ckpt['module']['condition_mlp.0.bias'])
                    self.condition_mlp[2].weight.copy_(llm_ckpt['module']['condition_mlp.2.weight'])
                    self.condition_mlp[2].bias.copy_(llm_ckpt['module']['condition_mlp.2.bias'])

                print("Done.")

                for param in self.condition_mlp.parameters():
                    param.requires_grad = False

        self.diffusion_model = self.init_diffusion(args, in_dim)

        if args.condition_property == None:
            pass
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            self.condition_property = args.condition_property
            assert property_normalizations is not None
            assert property_distribution is not None
            self.property_normalizations = property_normalizations
            self.property_distribution = property_distribution
            self.condition_property_mean = self.property_normalizations[self.condition_property]['mean']
            self.condition_property_mad = self.property_normalizations[self.condition_property]['mad']
            self.property_prediction_model = self.init_property_prediction(args).to(self.device)
            property_output_norms = {'mu': 1., 'alpha': 1, 'homo': 1000., 'lumo': 1000., 'gap': 1000, 'Cv': 1.}
            self.property_output_norm = property_output_norms[self.condition_property]
        else:
            raise NotImplementedError(f"{args.conditon} is not supported")

        self.delta_train = False
        self.save_hyperparameters(args)

        try:
            self.llm_model.gradient_checkpointing_enable(
                {"use_reentrant": False}
            )
        except:
            pass

        # self.diffusion_model.enable_gradient_checkpointing()

    # Helper methods for consistent metric logging
    def log_train(self, key, value, **kwargs):
        """Log training metrics with train/ prefix"""
        self.log(f"train/{key}", value, **kwargs)

    def log_val(self, key, value, **kwargs):
        """Log validation metrics with val/ prefix"""
        self.log(f"val/{key}", value, **kwargs)

    def log_test(self, key, value, **kwargs):
        """Log test metrics with test/ prefix"""
        # Force test metrics to be epoch-level (not dataloader-specific)
        kwargs['on_epoch'] = True
        kwargs['on_step'] = False
        kwargs['add_dataloader_idx'] = False
        self.log(f"test/{key}", value, **kwargs)

    def set_trainble_params(self, param_list, delta_train):
        self.delta_train = delta_train
        self.orignal_requires_grad = {}
        print('setting trainble params:', param_list)
        for name, param in self.named_parameters():
            self.orignal_requires_grad[name] = param.requires_grad
            match = False
            for p in param_list:
                if name.find(p) != -1:
                    match = True
                    continue
            if match:
                print('set trainble params:', name, param.requires_grad)
            param.requires_grad = match

    def restore_trainble_params(self, delta_train):
        self.delta_train = delta_train
        for name, param in self.named_parameters():
            param.requires_grad = self.orignal_requires_grad[name]
        print('restore trainble params')

    def configure_optimizers(self):
        if self.delta_train:
            self.scheduler = None
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4, weight_decay=self.args.weight_decay)
            return optimizer
        self.trainer.fit_loop.setup_data()
        # warmup_steps = min(len(self.trainer.train_dataloader) // self.args.accumulate_grad_batches, self.args.warmup_steps)
        warmup_steps = self.args.warmup_steps
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        max_iters = self.args.max_epochs * len(self.trainer.train_dataloader) // self.args.accumulate_grad_batches
        assert max_iters > warmup_steps
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRSchedulerV2(optimizer, max_iters, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler in {'None', 'none'}:
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @classmethod
    def init_tokenizer(cls, args):
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        return tokenizer

    @classmethod
    def init_llm(cls, args):
        if args.use_flash_attention:
            llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, torch_dtype=get_half_precision_dtype(), attn_implementation='flash_attention_2')
        else:
            llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, torch_dtype=get_half_precision_dtype())

        if args.load_llm_ckpt:
            model_state_dict = {}
            checkpoint_path = os.path.join(args.load_llm_ckpt, 'checkpoint/mp_rank_00_model_states.pt')
            main_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model_state_dict.update(main_model['module'])

            missing_keys, unexpected_keys = llm_model.load_state_dict(model_state_dict, strict=False)

        if args.llm_tune == 'freeze':
            assert not args.tune_embedding
            for param in llm_model.parameters():
                param.requires_grad = False
        elif args.llm_tune == 'full':
            for param in llm_model.parameters():
                param.requires_grad = True
        elif args.llm_tune == 'lora':
            lora_config = LoraConfig(r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     target_modules=["q_proj", "v_proj"])
            llm_model = get_peft_model(llm_model, lora_config)
            if args.tune_embedding:
                set_embed_tokens_trainable(llm_model)
            llm_model.print_trainable_parameters()
        elif args.llm_tune == 'mid_lora':
            lora_config = LoraConfig(r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     target_modules=["q_proj", "v_proj", 'k_proj', 'o_proj', "gate_proj", "up_proj", "down_proj"])
            llm_model = get_peft_model(llm_model, lora_config)
            if args.tune_embedding:
                set_embed_tokens_trainable(llm_model)
            llm_model.print_trainable_parameters()
        else:
            raise NotImplementedError()
        return llm_model

    def create_condition_prompt(self, context):
        batch_size = context.shape[0]
        context = context.unsqueeze(1)  # [batch_size, 1]
        out = self.condition_mlp(context)  # [batch_size, 4 * hidden_size]
        return out.view(batch_size, 4, self.hidden_size)

    @classmethod
    def init_diffusion(cls, args, llm_dim=None):
        diffusion_model = DGTDiffusion(args, in_dim=llm_dim)
        if args.diff_ckpt is not None:
            diff_ckpt = torch.load(args.diff_ckpt, map_location='cpu')
            if 'state_dict' in diff_ckpt:
                diff_ckpt = {'.'.join(k.split('.')[1:]): v for k, v in diff_ckpt['state_dict'].items()}
            else:
                diff_ckpt = {'.'.join(k.split('.')[1:]): v for k, v in diff_ckpt.items()}
            loading_info = diffusion_model.load_state_dict(diff_ckpt, strict=False)
            print(loading_info)
        if args.diff_tune == 'full':
            for param in diffusion_model.parameters():
                param.requires_grad = True
        elif args.diff_tune == 'freeze':
            for param in diffusion_model.parameters():
                param.requires_grad = False
        elif args.diff_tune == 'lora':
            lora_config = LoraConfig(r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout,
                                     target_modules=["proj", "ff_linear1", "ff_linear2", "ff_linear3", "ff_linear4", "node2edge_lin"],
                                     modules_to_save=["projector"])
            diffusion_model = get_peft_model(diffusion_model, lora_config)
            diffusion_model.print_trainable_parameters()
        else:
            raise NotImplementedError()
        return diffusion_model

    @classmethod
    def init_property_prediction(cls, args):
        property_path = Path(f"data/exp_class_{args.condition_property}")
        classifier_path = property_path / "best_checkpoint.npy"
        args_classifier_path = property_path / "args.pickle"
        with open(args_classifier_path, 'rb') as f:
            args_classifier = pickle.load(f)
        classifier = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args_classifier.nf, device='cpu', n_layers=args_classifier.n_layers, coords_weight=1.0, attention=args_classifier.attention, node_attr=args_classifier.node_attr)
        classifier_state_dict = torch.load(classifier_path, map_location=torch.device('cpu'))
        classifier.load_state_dict(classifier_state_dict)
        for param in classifier.parameters():
            param.requires_grad = False
        return classifier

    def resize_token_embeddings(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        self.llm_model.resize_token_embeddings(len(tokenizer))

    @torch.compile(dynamic=True, disable=disable_compile)
    def get_noise_loss(self, noise_pred, noise_gt, pos_0, pos_t, pos_pred, alpha_t, sigma_t, n_nodes, batch_mask, reduce_node_mean=False, align=True, translation_correction=False, align_prediction=False):
        '''
        coordinate_predict: [batch_size, num_nodes, 3]
        coordinate_target: [batch_size, num_nodes, 3]
        '''
        if align:
            align_noise = get_align_noise(pos_t, pos_0, pos_pred, alpha_t.view(-1, 1, 1), sigma_t.view(-1, 1, 1), batch_mask, translation_correction, align_prediction)
            noise_loss = torch.square(noise_pred - align_noise.detach()) # shape = [batch_size, max_num_nodes, 3]
        else:
            noise_loss = torch.square(noise_pred - noise_gt) # shape = [batch_size, max_num_nodes, 3]
        noise_loss = torch.mean(noise_loss, dim=-1) # shape = [batch_size, max_num_nodes]
        noise_loss = torch.sum(noise_loss, dim=-1) # shape = [batch_size]
        if reduce_node_mean:
            ## my prior implementation
            noise_loss = (noise_loss / n_nodes).sum()
        else:
            noise_loss = noise_loss / batch_mask.sum(dim=1) # shape = [batch_size]
            noise_loss = noise_loss.mean()
        return noise_loss

    def get_pos_loss(self, pos_pred, pos_0, pos_t, n_nodes, loss_norm, batch_mask, reduce_node_mean=False, align_loss=True):
        '''
        coordinate_predict: [batch_size, num_nodes, 3]
        coordinate_target: [batch_size, num_nodes, 3]
        '''
        if align_loss:
            align_pos0 = get_align_pos(pos_t, pos_0) # shape = [batch_size, max_num_nodes, 3]
            pos_loss = torch.square(pos_pred - align_pos0) # shape = [batch_size, max_num_nodes, 3]
        else:
            pos_loss = torch.square(pos_pred - pos_0) # shape = [batch_size, max_num_nodes, 3]
        pos_loss = torch.mean(pos_loss, dim=-1) # shape = [batch_size, max_num_nodes]
        pos_loss = torch.sum(pos_loss, dim=-1) # shape = [batch_size]

        pos_loss = pos_loss * loss_norm

        if reduce_node_mean:
            ## my prior implementation
            pos_loss = (pos_loss / n_nodes).sum()
        else:
            pos_loss = pos_loss / batch_mask.sum(dim=1) # shape = [batch_size]
            pos_loss = pos_loss.mean()
        return pos_loss


    def training_step(self, batch):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)
        data_batch, selfies_batch = batch
        batch_size = len(data_batch.smiles)
        with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
            loss, lm_loss, diff_loss, MAE_loss = self.forward(data_batch, selfies_batch, data_batch.context if hasattr(data_batch, 'context') else None)

        # Keep LR as-is for continuity
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)

        # Use helper methods for consistent train/ prefix
        self.log_train('lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
        self.log_train('diff_loss', diff_loss, sync_dist=True, batch_size=batch_size)
        self.log_train('loss', loss, sync_dist=True, batch_size=batch_size)

        if MAE_loss is not None:
            self.log_train(f"mae_{self.condition_property}", MAE_loss, sync_dist=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.test_rdmol_list = []
        self.test_metrics = {}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.diffusion_model.train()
        data_batch, selfies_batch = batch
        batch_size = len(data_batch.smiles)

        eval_test_only = self.args.mode in {'eval', 'eval_test_conform'}

        if dataloader_idx == 0 and not eval_test_only:
            # Validation dataloader - compute and log validation losses
            # This only happens during training mode when we have both val and test dataloaders
            train_epoch_condition = (self.current_epoch + 1) % self.args.conform_eval_epoch == 0 and self.args.mode == 'train'
            with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
                loss, lm_loss, diff_loss, MAE_loss = self.forward(data_batch, selfies_batch, data_batch.context if hasattr(data_batch, 'context') else None)
            # Use helper methods for consistent val/ prefix
            self.log_val('lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
            self.log_val('diff_loss', diff_loss, sync_dist=True, batch_size=batch_size)
            self.log_val('loss', loss, sync_dist=True, batch_size=batch_size)

        elif dataloader_idx == 1 or eval_test_only:
            # Test dataloader - perform conformer generation
            # In eval_test_conform mode, we only have one dataloader (test) with idx=0
            train_epoch_condition = (self.current_epoch + 1) % self.args.test_conform_epoch == 0 and self.args.mode == 'train'
            ## inference on the test set, using rdkit predicted conf as input
            eval_condition = self.args.mode in {'eval', 'eval_test_conform'}
            if not train_epoch_condition and not eval_condition:
                return
            
            with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
                data_batch, sampled_positions, MAE_loss = self.sample(data_batch, selfies_batch)


            # Log property MAE as a test metric if available
            if MAE_loss is not None and hasattr(self, 'condition_property'):
                self.log_test(f"mae_{self.condition_property}", MAE_loss, sync_dist=True, batch_size=batch_size)

            else:
                # Accumulate rdmols for end-of-epoch conformer evaluation
                sampled_positions = sampled_positions.float().cpu().numpy()
                node_index = 0
                for i in range(len(data_batch.rdmol)):
                    mol_idx = int(data_batch.mol_idx[i])
                    seed_pos_idx = int(data_batch.seed_pos_idx[i])
                    corrected_smiles = data_batch.corrected_smiles[i]
                    molecule = copy.deepcopy(data_batch.rdmol[i])
                    molecule.RemoveAllConformers()
                    num_nodes = molecule.GetNumAtoms()
                    positions = sampled_positions[node_index:node_index+num_nodes]
                    molecule = set_rdmol_positions(molecule, positions, removeHs=False, add_conformer=True)
                    self.test_rdmol_list.append((seed_pos_idx, mol_idx, corrected_smiles, molecule))
                    node_index += num_nodes


    @torch.no_grad()
    @torch.amp.autocast("cuda",dtype=torch.bfloat16)
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        ## evaluate the conformation generation performance
        if self.args.dataset.lower().find('qm9') >= 0:
            threshold = 0.5
        elif self.args.dataset.lower().find('drugs') >= 0:
            threshold = 0.75
        else:
            raise NotImplementedError

        if len(self.test_rdmol_list) > 0:
            if dist.is_initialized():
                gather_box = [None for _ in range(self.trainer.world_size)]
                dist.all_gather_object(gather_box, self.test_rdmol_list)
            else:
                gather_box = [self.test_rdmol_list]
            self.test_rdmol_list = []
            test_rdmol_list = {seed_pos_idx: (mol_idx, corrected_smiles, rdmol) for data in gather_box for seed_pos_idx, mol_idx, corrected_smiles, rdmol in data}
            test_rdmol_list = list(test_rdmol_list.values())
            if self.trainer.is_global_zero:
                num_failures = self.trainer.datamodule.test_dataset.num_failures # the ones that fail in data pre-processing
                log_dir = Path(self.logger.log_dir)
                if self.save_eval_only:
                    with open(log_dir / 'predict.pkl', 'wb') as f:
                        pickle.dump((test_rdmol_list, threshold, num_failures), f)
                # Don't log metrics during evaluation, just return them
                metrics = conformer_evaluation_V2(test_rdmol_list, self.trainer.datamodule.test_dataset.gt_conf_list, threshold, num_failures, logger=None, dataset_name=self.args.dataset, num_process=10)
                # Store metrics to log later
                self.test_metrics = metrics

        # Log test metrics after all validation steps are complete
        if hasattr(self, 'test_metrics') and self.test_metrics:
            for metric in ['recall_coverage_mean', 'recall_coverage_median', 'recall_amr_mean', 'recall_amr_median', 
                           'precision_coverage_mean', 'precision_coverage_median', 'precision_amr_mean', 'precision_amr_median']:
                if metric in self.test_metrics:
                    self.log(f"test/{metric}", self.test_metrics[metric], sync_dist=False)
            for metric in ['MolStable_3D', 'AtomStable_3D', 'Validity_3D', 'Unique_3D', 'Novelty_3D', 'Complete_3D']:
                if metric in self.test_metrics:
                    self.log(f"test/{metric}", self.test_metrics[metric], sync_dist=False)
            self.test_metrics = {}  # Clear for next epoch

    def simple_validation_eval(self, rdmol_list, threshold):
        '''
        this is a simple but non-standard implementation for the conformation generation evaluation, for validation use only
        '''
        pos1_list = []
        pos2_list = []
        for smiles, rdmol in rdmol_list:
            pos1 = rdmol.GetConformer(0).GetPositions()
            pos2 = rdmol.GetConformer(1).GetPositions()
            pos1_list.append(pos1)
            pos2_list.append(pos2)
        pos1_list = np.concatenate(pos1_list, axis=0)
        pos2_list = np.concatenate(pos2_list, axis=0)
        # print(f"{pos1_list.shape=}, {pos2_list.shape=}")
        # print(f"{pos1_list.std()=}, {pos2_list.std()=}")

        cov_list = []
        mat_list = []
        predict_mol_list = []

        for smiles, rdmol in rdmol_list:
            assert rdmol.GetNumConformers() == 2
            ## creat the rd_mol with the ground truth conformer
            gt_rdmol = copy.deepcopy(rdmol)
            gt_rdmol.RemoveAllConformers()
            gt_rdmol.AddConformer(rdmol.GetConformer(0), assignId=True)

            ## create the rdmol with the predicted conformer
            conf = rdmol.GetConformer(1)
            rdmol = copy.deepcopy(rdmol)
            rdmol.RemoveAllConformers()
            rdmol.AddConformer(conf, assignId=True)
            rmsd = get_best_rmsd(rdmol, gt_rdmol)
            predict_mol_list.append(rdmol)
            if np.isinf(rmsd) or np.isnan(rmsd) or rmsd > 1000.0:
                print(f"Warning: RMSD is either inf or nan: {rmsd}")
                print('\n\n\n\n\n\n\n')
                continue
            ## this is a simplified evaluation for fast evaluation. This cannot be used in experimental results
            cov = rmsd <= threshold
            mat = rmsd
            cov_list.append(cov)
            mat_list.append(mat)

        cov_list = np.asarray(cov_list)
        mat_list = np.asarray(mat_list)
        cov_mean = np.mean(cov_list)
        mat_mean = np.mean(mat_list)
        cov_median = np.median(cov_list)
        mat_median = np.median(mat_list)

        # Use val/ prefix with standardized snake_case names
        self.log_val('cov_mean', cov_mean, sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('mat_mean', mat_mean, sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('cov_median', cov_median, sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('mat_median', mat_median, sync_dist=True, batch_size=len(rdmol_list))

        eval_results_3d_unimol, _ = get_3D_edm_metric(predict_mol_list)
        self.log_val('mol_stable_3d', eval_results_3d_unimol['mol_stable'], sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('atom_stable_3d', eval_results_3d_unimol['atom_stable'], sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('validity_3d', eval_results_3d_unimol['Validity'], sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('unique_3d', eval_results_3d_unimol['Unique'], sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('novelty_3d', eval_results_3d_unimol['Novelty'], sync_dist=True, batch_size=len(rdmol_list))
        self.log_val('complete_3d', eval_results_3d_unimol['Complete'], sync_dist=True, batch_size=len(rdmol_list))


    # @torch.compile(dynamic=True, disable=disable_compile)
    def forward_llm(self, data_batch, selfies_batch, context=None):
        if context is not None:
            token_embeds = self.llm_model.get_input_embeddings()(selfies_batch.input_ids)
            condition_embeds = self.create_condition_prompt(context)
            inputs_embeds = torch.cat([condition_embeds, token_embeds], dim=1)

            soft_prompt_attention = torch.ones((selfies_batch.attention_mask.shape[0], 4),
                                            device=selfies_batch.attention_mask.device)
            attention_mask = torch.cat([soft_prompt_attention, selfies_batch.attention_mask], dim=1)

            ignore_prefix = torch.full((selfies_batch.input_ids.shape[0], 4), -100,
                                    device=selfies_batch.input_ids.device)


            if self.llm_tune == 'freeze':
                outputs = self.llm_model(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True)
            else:
                target = torch.cat([ignore_prefix, selfies_batch.input_ids], dim=1)
                target = target.masked_fill(~attention_mask.bool(), -100)
                outputs = self.llm_model(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=target,
                    output_hidden_states=True)

            outputs_hidden_states = [h[:, 4:] for h in outputs.hidden_states]
            outputs_logits = outputs.logits[:, 4:,:]
        else:
            if self.llm_tune == 'freeze':
                outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                        attention_mask=selfies_batch.attention_mask,
                                        return_dict=True,
                                        output_hidden_states=True)
            else:
                targets = selfies_batch.input_ids.masked_fill(~selfies_batch.attention_mask.bool(), -100)
                outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                        attention_mask=selfies_batch.attention_mask,
                                        return_dict=True,
                                        labels=targets,
                                        output_hidden_states=True)
                
            outputs_hidden_states = outputs.hidden_states
            outputs_logits = outputs.logits

        if self.llm_tune == 'freeze':
            hidden_states = tuple(h.detach() for h in outputs_hidden_states)
            lm_loss = 0
        else:
            hidden_states = outputs_hidden_states
            lm_loss = outputs.loss

        if self.use_meta_projector:
            if self.tokenizer.num_metaqueries > 0:
                # import ipdb
                # ipdb.set_trace()
                # Get positions for all sequences in batch at once
                bom_pos = torch.where(selfies_batch.input_ids == self.bom_token_id)[1]
                eom_pos = torch.where(selfies_batch.input_ids == self.eom_token_id)[1]

                # Create mask for selecting tokens between BOI and EOI
                batch_size, seq_len = selfies_batch.input_ids.shape
                indices = torch.arange(seq_len, device=selfies_batch.input_ids.device)[None, :].expand(
                    batch_size, -1
                )

                mask = (indices > bom_pos[:, None]) & (indices < eom_pos[:, None])

                if self.use_adameta:
                    sequences = [outputs_logits[i, mask[i]] for i in range(batch_size)]

                    outputs_logits = torch.nn.utils.rnn.pad_sequence(
                        sequences, batch_first=True, padding_value=0
                    )
                else:
                    outputs_logits = outputs_logits[mask].view(
                        batch_size, -1, outputs_logits.size(-1)
                    )

                # attention_mask = attention_mask[mask].view(batch_size, -1)
            else:
                raise ValueError("Meta projector is used but tokenizer.num_metaqueries is 0.")

            lm_x = self.meta_projector(outputs_logits)
        elif self.use_llm_projector:
            lm_x = self.llm_projector(hidden_states, data_batch.rdmol2selfies, selfies_batch)
        else:
            # 只要use_llm=True，就默认会走llm_projector逻辑，下面和use_llm_projector=True的逻辑是一致的
            if self.llm_jk == 'last':
                lm_embeds = hidden_states[-1] # shape = [batch_size, seq_len, hidden_size]
            elif self.llm_jk == 'mean':
                lm_embeds = torch.stack(hidden_states[1:], dim=2) # shape = [batch_size, seq_len, num_layers, hidden_size]
                lm_embeds = (self.mean_weight.softmax(dim=-1) @ lm_embeds).squeeze(2) # shape = [batch_size, seq_len, hidden_size]
                lm_embeds = self.mean_ln(lm_embeds)
            else:
                raise NotImplementedError

            if self.use_self_att_proj:
                lm_embeds = self.self_att_proj(self.linear_proj(lm_embeds), src_key_padding_mask=~selfies_batch.attention_mask.bool())

            lm_x = torch.bmm(data_batch.rdmol2selfies.to(lm_embeds.dtype), lm_embeds) # shape = [batch_size, rdmol_len, selfies_len], [batch_size, selfies_len, hidden_size] -> [batch_size, rdmol_len, hidden_size]
            norm = torch.clamp(torch.sum(data_batch.rdmol2selfies, dim=-1, keepdim=True), min=1) # shape = [batch_size, 1, 1]
            lm_x = lm_x / norm # shape = [batch_size, rdmol_len, hidden_size]
        
        return lm_x, lm_loss

    def forward(self, data_batch, selfies_batch, context=None):
        lm_loss = 0
        lm_x = None
        if self.use_llm:
            lm_x, lm_loss = self.forward_llm(data_batch, selfies_batch, context)

        bs = len(data_batch['smiles'])
        max_num_nodes = data_batch.max_seqlen
        total_num_nodes = data_batch.x.shape[0]
        if context is not None:
            pred_pos, pred_noise, classifier_args = self.diffusion_model(data_batch, lm_x, context)
            MAE_loss = None # NOT CHECK THE MAE in training
        else:
            pred_pos, pred_noise = self.diffusion_model(data_batch, lm_x)
            MAE_loss = None
        if self.pred_noise:
            pred_batch = to_dense_batch(pred_pos, data_batch.batch, batch_size=bs, max_num_nodes=max_num_nodes)[0] if self.align_prediction else None
            (pos_t_batch, pos_0_batch, pred_noise_batch, gt_noise_batch), batch_mask = to_dense_batch_list_tensor([data_batch.pos, data_batch.gt_pos, pred_noise, data_batch.noise], data_batch.batch, bs, max_num_nodes)
            diff_loss = self.get_noise_loss(pred_noise_batch, gt_noise_batch, pos_0_batch, pos_t_batch, pred_batch, data_batch.alpha_t_batch, data_batch.sigma_t_batch, total_num_nodes, batch_mask, self.reduce_node_mean, self.align_loss, self.translation_correction, self.align_prediction)
        else:
            pos_t_batch, batch_mask = to_dense_batch(data_batch.pos, data_batch.batch, batch_size=bs, max_num_nodes=max_num_nodes)
            pos_0_batch, _ = to_dense_batch(data_batch.gt_pos, data_batch.batch, batch_size=bs, max_num_nodes=max_num_nodes)
            pred_pos_batch, _ = to_dense_batch(pred_pos, data_batch.batch, batch_size=bs)
            diff_loss = self.get_pos_loss(pred_pos_batch, pos_0_batch, pos_t_batch, total_num_nodes, data_batch.loss_norm, batch_mask, self.reduce_node_mean, self.align_loss)

        loss = 0
        if self.args.lm_loss > 0:
            loss += lm_loss * self.args.lm_loss
        if self.args.diff_loss > 0:
            loss += diff_loss * self.args.diff_loss
        return loss, lm_loss, diff_loss,MAE_loss

    @torch.no_grad()
    def sample(self, data_batch, selfies_batch, T=None):
        num_nodes = data_batch.x.shape[0]
        device = data_batch.x.device
        bs = len(data_batch['smiles'])
        if T is None:
            T = self.noise_scheduler.T
        time_steps = torch.linspace(T, 0.001, self.args.sampling_steps, device=device)
        t_array = time_steps
        s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])

        lm_x = None
        if self.use_llm:
            lm_x, _ = self.forward_llm(data_batch, selfies_batch, data_batch.context if hasattr(data_batch, 'context') else None)

        classifier_args = None

        for i in range(len(t_array)):
            t = t_array[i]
            s = s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t
            noise_level = torch.log(alpha_t**2 / sigma_t**2)

            if self.t_cond == 't':
                data_batch['t_cond'] = torch.ones(num_nodes, device=device) * t
            elif self.t_cond == 'noise_level':
                data_batch['t_cond'] = torch.ones(num_nodes, device=device) * noise_level
            else:
                raise NotImplementedError(f"t_cond {self.t_cond} is not implemented")

            data_batch['alpha_t'] = torch.ones((num_nodes, 1), device=device) * alpha_t
            data_batch['sigma_t'] = torch.ones((num_nodes, 1), device=device) * sigma_t

            if hasattr(data_batch, "context"):
                pred_pos, _, classifier_args = self.diffusion_model(data_batch, lm_x, data_batch.context)
            else:
                pred_pos, _ = self.diffusion_model(data_batch, lm_x)

            pos_mean = (alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2) * data_batch.pos + (alpha_s * sigma2_t_given_s / sigma_t ** 2) * pred_pos
            pos_mean = remove_mean(pos_mean, data_batch.batch)

            if self.disable_com:
                epsilon_pos = torch.randn(data_batch.pos.shape, device=data_batch.pos.device)
            else:
                epsilon_pos = sample_com_rand_pos(data_batch.pos.shape, data_batch.batch)
            pos = pos_mean + sigma * epsilon_pos
            assert not torch.isnan(pred_pos).any(), print('here 22', epsilon_pos.shape)
            data_batch['pos'] = pos

        pos = pos_mean * self.pos_std # std of qm9's dataset
        data_batch['pos'] = pos

        if hasattr(data_batch, "context"):
            assert classifier_args is not None
            h0, _, full_edges, _, node_mask, edge_mask, n_nodes = classifier_args
            full_pos, _ = to_dense_batch(data_batch.pos, data_batch.batch)  # [B, N, 3]
            full_pos = full_pos.reshape(-1, 3) # [B * N, 3]
            context_prediction = self.property_prediction_model(h0, full_pos, full_edges, None, node_mask, edge_mask, n_nodes) # Predict context using the classifier
            context_prediction = context_prediction * self.condition_property_mad + self.condition_property_mean  # Rescale the predictions
            context_target = data_batch.context.clone()
            context_target = context_target * self.condition_property_mad + self.condition_property_mean
            MAE_loss = torch.functional.F.l1_loss(context_prediction, context_target, reduction='mean') * self.property_output_norm
        else:
            MAE_loss = None

        return data_batch, pos, MAE_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group()
        parser.add_argument('--llm_model', type=str, default="all_checkpoints/mollama")
        parser.add_argument('--load_llm_ckpt', type=str, default=None)
        parser.add_argument('--num_beams', type=int, default=1)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--use_llm', action='store_true', default=False)
        parser.add_argument('--llm_cond', action='store_true', default=False)
        parser.add_argument('--tune_embedding', action='store_true', default=False)
        parser.add_argument('--sample_num', type=int, default=10000)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--generate_eval_epoch', type=int, default=10)
        parser.add_argument('--conform_eval_epoch', type=int, default=5)
        parser.add_argument('--test_conform_epoch', type=int, default=20)
        parser.add_argument('--use_llm_projector', action='store_true', default=False)
        parser.add_argument('--use_meta_projector', action='store_true', default=False)
        # 使用自适应metaquery的时候，args.num_metaqueries要设置成：qm9_max_n_nodes=29, geom_drug_max_n_nodes=181
        parser.add_argument('--use_adameta', action='store_true', default=False)
        # 不能太大，MoLlama的max position embeddings只有512
        parser.add_argument('--num_metaqueries', type=int, default=0)
        parser.add_argument('--num_meta_hidden_layers', type=int, default=24)
        # 由于qm9_max_n_nodes=29, geom_drug_max_n_nodes=181，则使用qm9的时候max_n_nodes设置成32，使用geom_drugs的时候max_n_nodes设置成184
        # qm9时num_metaqueries设置成64以上；geom_drugs的时候num_metaqueries设置成256以上
        parser.add_argument('--max_n_nodes', type=int, default=32, help='Maximum number of nodes for the meta2nodes projector output, should be larger than the max nodes in the dataset.')
        parser.add_argument('--meta2nodes_proj_type', type=str, default='mlp', help='Type of projection for meta2nodes, mlp or linear')

        
        ## llm lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)
        parser.add_argument('--llm_jk', type=str, default='last')

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or
        parser.add_argument('--not_reduce_node_mean', action='store_true', default=False) # or

        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default=None)
        parser.add_argument('--ckpt_path', type=str, default=None)

        parser.add_argument('--eval_smiles_path', type=str, default=None)

        ## loss
        parser.add_argument('--lm_loss', type=float, default=0)
        parser.add_argument('--diff_loss', type=float, default=1.0)
        parser.add_argument('--save_eval_only', action='store_true', default=False)

        ## diffusion scheduler
        parser.add_argument('--noise_scheduler', type=str, default='cosine')
        parser.add_argument('--continuous_beta_0', type=float, default=0.1)
        parser.add_argument('--continuous_beta_1', type=float, default=20)
        parser.add_argument('--sampling_steps', type=int, default=100)

        ## diffusion parameters
        parser.add_argument('--not_align_loss', action='store_true', default=False)
        parser.add_argument('--translation_correction', action='store_true', default=False)
        parser.add_argument('--align_prediction', action='store_true', default=False)
        parser.add_argument('--use_self_att_proj', action='store_true', default=False)
        parser.add_argument('--diff_tune', type=str, default="full")
        parser.add_argument('--diff_ckpt', type=str, default=None)

        parser.add_argument('--llm_ckpt', type=str, default=None)
        DGTDiffusion.add_args(parser)
        return parent_parser

def calculate_rmsd(gt_conf, predict_mols):
    rmsd_list = []
    for smiles, rdmol in predict_mols:
        try:
            rmsd = AllChem.GetBestRMS(Chem.RemoveHs(gt_conf), Chem.RemoveHs(rdmol))
            rmsd_list.append(rmsd)
        except Exception as e:
            print(e)
            print('Additional failure', smiles)
            return [np.nan] * len(predict_mols), 1
    return rmsd_list, 0

def process_molecule(args):
    mol_idx, gt_conf_list, predict_mols = args
    rmsd_results = []
    num_additional_failure = 0
    for gt_conf in gt_conf_list:
        gt_conf = copy.deepcopy(gt_conf)
        rmsd_list, failures = calculate_rmsd(gt_conf, predict_mols)
        num_additional_failure += failures
        rmsd_array = np.asarray(rmsd_list)
        rmsd_results.append(rmsd_array)
    return mol_idx, rmsd_results, num_additional_failure

def calc_performance_stats(rmsd_array, threshold):
    coverage_recall = float(np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0))
    amr_recall = float(rmsd_array.min(axis=1).mean())
    coverage_precision = float(np.mean(rmsd_array.min(axis=0, keepdims=True) < threshold, axis=1))
    amr_precision = float(rmsd_array.min(axis=0).mean())
    return coverage_recall, amr_recall, coverage_precision, amr_precision

def conformer_evaluation(predict_rdmol_list, gt_conf_list_list, threshold, num_failures, logger=None, num_process=1):
    id2predict_mols = {}
    for data in predict_rdmol_list:
        mol_idx, smiles, rdmol = data
        mol_idx = int(mol_idx)
        if mol_idx not in id2predict_mols:
            id2predict_mols[mol_idx] = [(smiles, rdmol),]
        else:
            id2predict_mols[mol_idx].append((smiles, rdmol))

    print('num existing failures:', num_failures)
    rmsd_results = {}
    num_additional_failure = 0
    args_list = [(mol_idx, gt_conf_list, id2predict_mols[mol_idx]) for mol_idx, gt_conf_list in enumerate(gt_conf_list_list) if len(gt_conf_list) > 0 and len(id2predict_mols[mol_idx]) > 0]
    results = []
    if num_process > 1:
        with multiprocessing.Pool(processes=num_process) as pool:
            for result in tqdm(pool.imap_unordered(process_molecule, args_list), total=len(args_list), desc='Evaluating conformer'):
                results.append(result)
    else:
        for args in tqdm(args_list, total=len(args_list), desc='Evaluating conformer'):
            result = process_molecule(args)
            results.append(result)

    for mol_idx, rmsd_result, failures in results:
        rmsd_results[mol_idx] = rmsd_result
        num_additional_failure += failures
    print('\n\n --------------------')
    print(f"Failure count: {num_additional_failure}")
    stats = []
    for mol_idx in rmsd_results:
        rmsd_matrix = np.stack(rmsd_results[mol_idx], axis=0) # shape = [num_gt_pos, num_predict_pos]
        cr, mr, cp, mp = calc_performance_stats(rmsd_matrix, threshold)
        stats.append((cr, mr, cp, mp))
    coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

    ## consider the failures in the data pre-processing
    coverage_recall, amr_recall, coverage_precision, amr_precision = list(coverage_recall), list(amr_recall), list(coverage_precision), list(amr_precision)
    coverage_recall.extend([0.0] * num_failures)
    coverage_precision.extend([0.0] * num_failures)
    recall_coverage_mean = np.mean(coverage_recall)
    recall_coverage_median = np.median(coverage_recall)
    recall_amr_mean = np.nanmean(amr_recall)
    recall_amr_median = np.nanmedian(amr_recall)
    precision_coverage_mean = np.mean(coverage_precision)
    precision_coverage_median = np.median(coverage_precision)
    precision_amr_mean = np.nanmean(amr_precision)
    precision_amr_median = np.nanmedian(amr_precision)

    if logger is not None:
        logger.log('test/recall_coverage_mean', recall_coverage_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/recall_coverage_median', recall_coverage_median, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/recall_amr_mean', recall_amr_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/recall_amr_median', recall_amr_median, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/precision_coverage_mean', precision_coverage_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/precision_coverage_median', precision_coverage_median, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/precision_amr_mean', precision_amr_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/precision_amr_median', precision_amr_median, sync_dist=False, batch_size=len(predict_rdmol_list))
    else:
        print('recall_coverage_mean', recall_coverage_mean)
        print('recall_coverage_median', recall_coverage_median)
        print('recall_amr_mean', recall_amr_mean)
        print('recall_amr_median', recall_amr_median)
        print('precision_coverage_mean', precision_coverage_mean)
        print('precision_coverage_median', precision_coverage_median)
        print('precision_amr_mean', precision_amr_mean)
        print('precision_amr_median', precision_amr_median)
        print('\n\n\n')

    predict_rdmol_list = [data[2] for data in predict_rdmol_list]
    eval_results_3d_unimol,_ = get_3D_edm_metric(predict_rdmol_list)

    if logger is not None:
        logger.log('test/MolStable_3D', eval_results_3d_unimol['mol_stable'], sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/AtomStable_3D', eval_results_3d_unimol['atom_stable'], sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/Validity_3D', eval_results_3d_unimol['Validity'], sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/Unique_3D', eval_results_3d_unimol['Unique'], sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/Novelty_3D', eval_results_3d_unimol['Novelty'], sync_dist=False, batch_size=len(predict_rdmol_list))
        logger.log('test/Complete_3D', eval_results_3d_unimol['Complete'], sync_dist=False, batch_size=len(predict_rdmol_list))
    else:
        print('MolStable_3D', eval_results_3d_unimol['mol_stable'])
        print('AtomStable_3D', eval_results_3d_unimol['atom_stable'])
        print('Validity_3D', eval_results_3d_unimol['Validity'])
        print('Unique_3D', eval_results_3d_unimol['Unique'])
        print('Novelty_3D', eval_results_3d_unimol['Novelty'])
        print('Complete_3D', eval_results_3d_unimol['Complete'])


def process_rmsd(inputs):
    idx, args_list = inputs
    result_list = []
    for args in tqdm(args_list, total=len(args_list), desc=f'Conformer {idx}', position=idx):
        mol_idx, gt_conf, predict_mols = args
        rmsd_list, failures = calculate_rmsd(gt_conf, predict_mols)
        rmsd_list = np.asarray(rmsd_list)
        result_list.append((mol_idx, rmsd_list, failures))
    return result_list

def conformer_evaluation_V2(predict_rdmol_list, gt_conf_list_list, threshold, num_failures, logger=None, num_process=1, dataset_name='QM9'):
    id2predict_mols = {}
    for data in predict_rdmol_list:
        mol_idx, smiles, rdmol = data
        mol_idx = int(mol_idx)
        if mol_idx not in id2predict_mols:
            id2predict_mols[mol_idx] = [(smiles, rdmol),]
        else:
            id2predict_mols[mol_idx].append((smiles, rdmol))

    print(f"{num_failures} existing failures")

    args_list = []
    # exceptional_mol_idx = []
    for mol_idx, gt_conf_list in enumerate(gt_conf_list_list):
        # if mol_idx not in id2predict_mols:
        #     exceptional_mol_idx.append(mol_idx)
        #     continue
        if len(gt_conf_list) > 0 and len(id2predict_mols[mol_idx]) > 0:
            for gt_conf in gt_conf_list:
                args_list.append((mol_idx, copy.deepcopy(gt_conf), id2predict_mols[mol_idx]))
    random.shuffle(args_list)
    # print(f"{len(exceptional_mol_idx)} mol_idx not found in the predict_rdmol_list")

    rmsd_results = {}
    num_additional_failure = 0
    if num_process > 1:
        args_chunks = [(idx, args_list[idx::num_process]) for idx in range(num_process)]
        pool = multiprocessing.get_context('spawn').Pool(processes=num_process)
        try:
            results = pool.map(process_rmsd, args_chunks)
            for result in results:
                for mol_idx, rmsd_list, failures in result:
                    if mol_idx not in rmsd_results:
                        rmsd_results[mol_idx] = []
                    rmsd_results[mol_idx].append(rmsd_list)
                    num_additional_failure += failures
        finally:
            pool.close()
            pool.join()
    else:
        results = process_rmsd((1, args_list))
        for mol_idx, rmsd_list, failures in results:
            if mol_idx not in rmsd_results:
                rmsd_results[mol_idx] = []
            rmsd_results[mol_idx].append(rmsd_list)
            num_additional_failure += failures

    print('\n\n --------------------')
    print(f"Failure count: {num_additional_failure}")

    metrics = {}
    stats = []
    for mol_idx in rmsd_results:
        rmsd_matrix = np.stack(rmsd_results[mol_idx], axis=0) # shape = [num_gt_pos, num_predict_pos]
        cr, mr, cp, mp = calc_performance_stats(rmsd_matrix, threshold)
        stats.append((cr, mr, cp, mp))
    coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

    ## consider the failures in the data pre-processing
    coverage_recall, amr_recall, coverage_precision, amr_precision = list(coverage_recall), list(amr_recall), list(coverage_precision), list(amr_precision)
    coverage_recall.extend([0.0] * num_failures)
    coverage_precision.extend([0.0] * num_failures)
    recall_coverage_mean = np.mean(coverage_recall)
    recall_coverage_median = np.median(coverage_recall)
    recall_amr_mean = np.nanmean(amr_recall)
    recall_amr_median = np.nanmedian(amr_recall)
    precision_coverage_mean = np.mean(coverage_precision)
    precision_coverage_median = np.median(coverage_precision)
    precision_amr_mean = np.nanmean(amr_precision)
    precision_amr_median = np.nanmedian(amr_precision)

    metrics['recall_coverage_mean'] = recall_coverage_mean
    metrics['recall_coverage_median'] = recall_coverage_median
    metrics['recall_amr_mean'] = recall_amr_mean
    metrics['recall_amr_median'] = recall_amr_median
    metrics['precision_coverage_mean'] = precision_coverage_mean
    metrics['precision_coverage_median'] = precision_coverage_median
    metrics['precision_amr_mean'] = precision_amr_mean
    metrics['precision_amr_median'] = precision_amr_median
    print(metrics)

    predict_rdmol_list = [data[2] for data in predict_rdmol_list]
    eval_results_3d_unimol,_ = get_3D_edm_metric(predict_rdmol_list, dataset_name=dataset_name)

    metrics['MolStable_3D'] = eval_results_3d_unimol['mol_stable']
    metrics['AtomStable_3D'] = eval_results_3d_unimol['atom_stable']
    metrics['Validity_3D'] = eval_results_3d_unimol['Validity']
    metrics['Unique_3D'] = eval_results_3d_unimol['Unique']
    metrics['Novelty_3D'] = eval_results_3d_unimol['Novelty']
    metrics['Complete_3D'] = eval_results_3d_unimol['Complete']
    print(metrics)

    if logger is not None:
        # Log metrics using the standard Lightning approach
        for metric in ['recall_coverage_mean', 'recall_coverage_median', 'recall_amr_mean', 'recall_amr_median', 'precision_coverage_mean', 'precision_coverage_median', 'precision_amr_mean', 'precision_amr_median']:
            logger.log(f"test/{metric}", metrics[metric], sync_dist=False, batch_size=len(predict_rdmol_list), on_epoch=True, on_step=False, add_dataloader_idx=False)
        for metric in ['MolStable', 'AtomStable', 'Validity', 'Unique', 'Novelty', 'Complete']:
            logger.log(f"test/{metric}_3D", metrics[f"{metric}_3D"], sync_dist=False, batch_size=len(predict_rdmol_list), on_epoch=True, on_step=False, add_dataloader_idx=False)

    return metrics