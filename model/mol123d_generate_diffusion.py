"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import math
import torch
import torch.nn as nn
from torch import optim
import lightning as L
from transformers import AutoTokenizer
from model.modeling_llama import LlamaForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from model.help_funcs import AttrDict
from pathlib import Path
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
from data_provider.conf_gen_cal_metrics import set_rdmol_positions
from evaluation.eval_functions import get_2D_edm_metric, get_3D_edm_metric, get_moses_metrics
import numpy as np
import copy
from data_provider.conf_gen_cal_metrics import get_best_rmsd
from torch_geometric.utils import to_dense_batch
from data_provider.diffusion_data_module import sample_com_rand_pos
import torch.distributed as dist
from rdkit import Chem
from rdkit.Chem import AllChem

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

def get_precision(precision):
    if precision in {'16', '16-mixed'}:
        return torch.float16
    elif precision in {'bf16', 'bf16-mixed'}:
        return torch.bfloat16
    elif precision in {'32',}:
        return torch.float32
    else:
        raise NotImplementedError

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
def get_align_noise(pos_t, pos_0, alpha_t, sigma_t):
    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
    aligned_noise = (pos_t - alpha_t * align_pos_0) / sigma_t
    return aligned_noise

class Mol123DGenerateDiffusion(L.LightningModule):
    # def set_trainble_params(self, param_list):
    #     self.delta_train = True
    #     self.orignal_requires_grad = {}
    #     print('setting trainble params:', param_list)
    #     for name, param in self.named_parameters():
    #         self.orignal_requires_grad[name] = param.requires_grad
    #         match = False
    #         for p in param_list:
    #             if name.find(p) != -1:
    #                 match = True
    #                 continue
    #         if match:
    #             print('set trainble params:', name, param.requires_grad)
    #         param.requires_grad = match

    # def restore_trainble_params(self):
    #     self.delta_train = False
    #     for name, param in self.named_parameters():
    #         param.requires_grad = self.orignal_requires_grad[name]
    #     print('restore trainble params')

    def configure_optimizers(self):
        if self.delta_train:
            self.scheduler = None
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4, weight_decay=self.args.weight_decay)
            return optimizer
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        max_iters = self.args.max_epochs * len(self.trainer.train_dataloader)
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
        llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, torch_dtype=get_half_precision_dtype())
        if args.llm_tune == 'freeze':
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

    @classmethod
    def init_conf_generator(cls, args):
        diffusion_model = JODODiffusion(args)

        dictionary = Dictionary.load('./data_provider/unimol_dict.txt')
        dictionary.add_symbol("[MASK]", is_special=True)

        if args.conf_gen_tune == 'freeze':
            for param in diffusion_model.parameters():
                param.requires_grad = False
        elif args.conf_gen_tune == 'full':
            for param in diffusion_model.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError()
        return diffusion_model, dictionary


    def resize_token_embeddings(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        self.llm_model.resize_token_embeddings(len(tokenizer))

    def get_noise_loss(self, noise_pred, noise_gt, pos_0, pos_t, alpha_t, sigma_t, n_nodes, reduce_mean=False, align=True):
        '''
        coordinate_predict: [batch_size, num_nodes, 3]
        coordinate_target: [batch_size, num_nodes, 3]
        '''
        if align:
            align_noise = get_align_noise(pos_t, pos_0, alpha_t.view(-1, 1, 1), sigma_t.view(-1, 1, 1))
            noise_loss = torch.square(noise_pred - align_noise) # shape = [batch_size, max_num_nodes, 3]
        else:
            noise_loss = torch.square(noise_pred - noise_gt) # shape = [batch_size, max_num_nodes, 3]
        noise_loss = torch.mean(noise_loss, dim=-1) # shape = [batch_size, max_num_nodes]
        noise_loss = torch.sum(noise_loss, dim=-1) # shape = [batch_size]
        if reduce_mean:
            ## my prior implementation
            noise_loss = (noise_loss / n_nodes).sum()
        else:
            ## following jodo's implementation
            noise_loss = noise_loss.mean()
        return noise_loss

    def get_pos_loss(self, pos_pred, pos_0, pos_t, n_nodes, loss_norm, reduce_mean=False, align_loss=True):
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

        if reduce_mean:
            ## my prior implementation
            pos_loss = (pos_loss / n_nodes).sum()
        else:
            ## following jodo's implementation
            pos_loss = pos_loss.mean()
        return pos_loss


    def __init__(self, args, tokenizer=None, max_sf_tokens=30, noise_scheduler=None, pos_std=None):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.detach_lm = args.detach_lm
        self.max_sf_tokens = max_sf_tokens
        self.noise_scheduler = noise_scheduler

        ## init llm
        self.llm_model = self.init_llm(args)
        if tokenizer is None:
            self.tokenizer = self.init_tokenizer(args)
        else:
            self.tokenizer = tokenizer

        self.reduce_mean = not args.not_reduce_mean
        self.infer_time = args.infer_time
        self.pos_std = pos_std
        self.pred_noise = args.pred_noise
        self.align_loss = not args.not_align_loss

        ## init diffusion
        self.diffusion_model, self.dictionary = self.init_conf_generator(args)

        ## init projector
        in_dim = self.llm_model.config.hidden_size
        out_dim = args.hidden_size
        self.projector = nn.Sequential(nn.Linear(in_dim, out_dim),
                                       nn.GELU(),
                                       nn.Linear(out_dim, out_dim))

        self.delta_train = False
        self.resize_token_embeddings(self.tokenizer)
        self.save_hyperparameters(args)

    def training_step(self, batch):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)

        data_batch, selfies_batch = batch
        with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
            loss, lm_loss, coordinate_loss = self.forward(data_batch, selfies_batch)

        batch_size = selfies_batch.input_ids.shape[0]
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
        self.log('train/loss', loss, sync_dist=True, batch_size=batch_size)
        self.log('train/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
        self.log('train/coordinate_loss', coordinate_loss, sync_dist=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.val_rdmol_list = []
        self.test_rdmol_list = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_batch, selfies_batch = batch
        batch_size = selfies_batch.input_ids.shape[0]

        if dataloader_idx == 0: # validation set
            with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
                loss, lm_loss, coordinate_loss = self.forward(data_batch, selfies_batch)

            self.log('val/loss', loss, sync_dist=True, batch_size=batch_size)
            self.log('val/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
            self.log('val/coordinate_loss', coordinate_loss, sync_dist=True, batch_size=batch_size)

            train_epoch_condition = (self.current_epoch + 1) % self.args.conform_eval_epoch == 0 and self.args.mode == 'train'
            eval_condition = self.args.mode in {'eval', 'eval_conf'}
            if not train_epoch_condition and not eval_condition:
                return

            ## inference on the validation set, using randomly perturbed ground truth conf as input
            with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
                data_batch, sampled_positions = self.sample(data_batch)
            sampled_positions = sampled_positions.float().cpu().numpy()

            node_index = 0
            for i in range(len(data_batch.rdmol)):
                molecule = copy.deepcopy(data_batch.rdmol[i])
                smiles = data_batch.smiles[i]
                num_nodes = molecule.GetNumAtoms()
                positions = sampled_positions[node_index:node_index+num_nodes]
                molecule = set_rdmol_positions(molecule, positions, removeHs=False, add_conformer=True)
                self.val_rdmol_list.append((smiles, molecule))
                node_index += num_nodes

        elif dataloader_idx == 1: # test set
            ## inference on the test set, using rdkit predicted conf as input
            train_epoch_condition = (self.current_epoch + 1) % self.args.test_conform_epoch == 0 and self.args.mode == 'train'
            eval_condition = self.args.mode in {'eval', 'eval_test_conform'}
            if not train_epoch_condition and not eval_condition:
                return

            with torch.amp.autocast("cuda",dtype=get_precision(self.trainer.precision)):
                data_batch, sampled_positions = self.sample(data_batch, self.infer_time)
            sampled_positions = sampled_positions.float().cpu().numpy()

            node_index = 0
            for i in range(len(data_batch.rdmol)):
                mol_idx = data_batch.mol_idx[i]
                seed_pos_idx = data_batch.seed_pos_idx[i]
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
        train_epoch_condition = (self.current_epoch + 1) % self.args.conform_eval_epoch == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval', 'eval_conf'}
        if not train_epoch_condition and not eval_condition:
            return

        if self.args.dataset.lower().find('qm9') >= 0:
            threshold = 0.5
        elif self.args.dataset.lower().find('drugs') >= 0:
            threshold = 0.75
        else:
            raise NotImplementedError

        if len(self.val_rdmol_list) > 0:
            self.simple_validation_eval(self.val_rdmol_list, threshold)
            self.val_rdmol_list = [] # important, clean the rdmol_list

        if len(self.test_rdmol_list) > 0:
            gather_box = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gather_box, self.test_rdmol_list)
            self.test_rdmol_list = []
            test_rdmol_list = {seed_pos_idx: (mol_idx, corrected_smiles, rdmol) for data in gather_box for seed_pos_idx, mol_idx, corrected_smiles, rdmol in data}
            test_rdmol_list = list(test_rdmol_list.values())
            if self.trainer.is_global_zero:
                num_failures = self.trainer.datamodule.test_dataset.num_failures # the ones that fail in data pre-processing
                self.inference_eval(test_rdmol_list, self.trainer.datamodule.test_dataset.gt_conf_list, threshold, num_failures)


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
        print(pos1_list.shape, pos2_list.shape)
        print(pos1_list.std(), pos2_list.std())

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
        self.log('val/cov_mean', cov_mean, sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/mat_mean', mat_mean, sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/cov_median', cov_median, sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/mat_median', mat_median, sync_dist=True, batch_size=len(rdmol_list))

        eval_results_3d_unimol = get_3D_edm_metric(predict_mol_list)
        self.log('val/mol_stable_3d', eval_results_3d_unimol['mol_stable'], sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/atom_stable_3d', eval_results_3d_unimol['atom_stable'], sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/validity_3d', eval_results_3d_unimol['Validity'], sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/unique_3d', eval_results_3d_unimol['Unique'], sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/novelty_3d', eval_results_3d_unimol['Novelty'], sync_dist=True, batch_size=len(rdmol_list))
        self.log('val/complete_3d', eval_results_3d_unimol['Complete'], sync_dist=True, batch_size=len(rdmol_list))

    def inference_eval(self, predict_rdmol_list, gt_conf_list_list, threshold, num_failures):
        def calc_performance_stats(rmsd_array, threshold):
            coverage_recall = float(np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0))
            amr_recall = float(rmsd_array.min(axis=1).mean())
            coverage_precision = float(np.mean(rmsd_array.min(axis=0, keepdims=True) < threshold, axis=1))
            amr_precision = float(rmsd_array.min(axis=0).mean())
            return coverage_recall, amr_recall, coverage_precision, amr_precision

        id2predict_mols = {}
        for data in predict_rdmol_list:
            mol_idx, smiles, rdmol = data
            mol_idx = int(mol_idx)
            if mol_idx not in id2predict_mols:
                id2predict_mols[mol_idx] = [(smiles, rdmol),]
            else:
                id2predict_mols[mol_idx].append((smiles, rdmol))
        # print(sorted(id2predict_mols.keys()))
        rmsd_results = {}
        for mol_idx, gt_conf_list in enumerate(tqdm(gt_conf_list_list, desc='Evaluating Conformer')):
            rmsd_results[mol_idx] = []
            predict_mols = id2predict_mols[mol_idx]
            for gt_conf in gt_conf_list:
                gt_conf = copy.deepcopy(gt_conf)
                rmsd_list = []
                for smiles, rdmol in predict_mols:
                    try:
                        rmsd = AllChem.GetBestRMS(Chem.RemoveHs(gt_conf), Chem.RemoveHs(rdmol))
                        rmsd_list.append(rmsd)
                    except:
                        print('Additional failure', smiles)
                        rmsd_list = [np.nan] * len(predict_mols)
                        break
                rmsd_array = np.asarray(rmsd_list)
                rmsd_results[mol_idx].append(rmsd_array)

        stats = []
        for mol_idx in tqdm(rmsd_results):
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

        self.log('test/recall_coverage_mean', recall_coverage_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/recall_coverage_median', recall_coverage_median, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/recall_amr_mean', recall_amr_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/recall_amr_median', recall_amr_median, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/precision_coverage_mean', precision_coverage_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/precision_coverage_median', precision_coverage_median, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/precision_amr_mean', precision_amr_mean, sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/precision_amr_median', precision_amr_median, sync_dist=False, batch_size=len(predict_rdmol_list))


        predict_rdmol_list = [data[2] for data in predict_rdmol_list]
        eval_results_3d_unimol = get_3D_edm_metric(predict_rdmol_list)
        self.log('test/mol_stable_3d', eval_results_3d_unimol['mol_stable'], sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/atom_stable_3d', eval_results_3d_unimol['atom_stable'], sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/validity_3d', eval_results_3d_unimol['Validity'], sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/unique_3d', eval_results_3d_unimol['Unique'], sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/novelty_3d', eval_results_3d_unimol['Novelty'], sync_dist=False, batch_size=len(predict_rdmol_list))
        self.log('test/complete_3d', eval_results_3d_unimol['Complete'], sync_dist=False, batch_size=len(predict_rdmol_list))


    @torch.no_grad()
    def sample(self, data_batch, T=None):
        num_nodes = data_batch.x.shape[0]
        device = data_batch.x.device
        if T is None:
            T = self.noise_scheduler.T
        time_steps = torch.linspace(T, 1e-3, self.args.sampling_steps, device=device)
        t_array = time_steps
        s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])

        pos = data_batch.pos
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

            data_batch['t_cond'] = torch.ones(num_nodes, device=device) * t
            data_batch['alpha_t'] = torch.ones((num_nodes, 1), device=device) * alpha_t
            data_batch['sigma_t'] = torch.ones((num_nodes, 1), device=device) * sigma_t
            pred_pos, _ = self.diffusion_model(data_batch)

            pos_mean = (alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2) * pos + (alpha_s * sigma2_t_given_s / sigma_t ** 2) * pred_pos
            epsilon_pos = sample_com_rand_pos(pos.shape, data_batch.batch)
            pos = pos_mean + sigma * epsilon_pos
            assert not torch.isnan(pred_pos).any(), print('here 22')
            data_batch['pos'] = pos

        pos = pos_mean * self.pos_std # std of qm9's dataset
        data_batch['pos'] = pos
        return data_batch, pos


    def forward(self, data_batch, selfies_batch):
        targets = selfies_batch.input_ids.masked_fill(~selfies_batch.attention_mask.bool(), -100)
        if not self.args.bi_attend:
            lm_loss = 0
            outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                    attention_mask=selfies_batch.attention_mask,
                                    return_dict=True,
                                    labels=targets,
                                    output_hidden_states=True)
            lm_embeds = outputs.hidden_states[-1] # shape = [batch_size, seq_len, hidden_size]
            if self.args.lm_loss > 0:
                lm_loss = outputs.loss
        else:
            lm_loss = 0
            if self.args.lm_loss > 0:
                self.llm_model.set_mode('causal')
                outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                        attention_mask=selfies_batch.attention_mask,
                                        return_dict=True,
                                        labels=targets,
                                        output_hidden_states=False)
                lm_loss = outputs.loss

            self.llm_model.set_mode('noncausal')
            outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                    attention_mask=selfies_batch.attention_mask,
                                    return_dict=True,
                                    output_hidden_states=True)
            lm_embeds = outputs.hidden_states[-1] # shape = [batch_size, seq_len, hidden_size]

        ## use the last hidden state as the representation of the molecule
        if self.detach_lm or self.args.llm_tune == 'freeze':
            lm_embeds = lm_embeds.detach()
        data_batch.rdmol2selfies = data_batch.rdmol2selfies.to(lm_embeds.dtype)
        lm_x = torch.bmm(data_batch.rdmol2selfies, lm_embeds) # shape = [batch_size, rdmol_len, selfies_len], [batch_size, selfies_len, hidden_size] -> [batch_size, rdmol_len, hidden_size]
        norm = torch.clamp(torch.sum(data_batch.rdmol2selfies, dim=-1, keepdim=True), min=1) # shape = [batch_size, 1, 1]
        lm_x = self.projector(lm_x / norm) # shape = [batch_size, rdmol_len, hidden_size]
        # x = self.embed_tokens(data_batch.x) # shape = [batch_size, seq_len, embed_dim]
        x_batch, batch_mask = to_dense_batch(data_batch.x, data_batch.batch, max_num_nodes=lm_x.shape[1]) # shape = [batch_size, max_n_nodes_batch]
        # full_mask = torch.full((lm_x.shape[0], lm_x.shape[1]), False) # shape = [batch_size, max_n_nodes_all]
        # full_mask[:, :batch_mask.shape[1]] = batch_mask
        # lm_x_cat = torch.cat([lm_x[i][full_mask[i]] for i in range(lm_x.shape[0])], dim=0) # shape = [sum(n_nodes), embed_dim]
        lm_x_cat = lm_x[batch_mask] # shape = [sum(n_nodes), embed_dim]
        data_batch.lm_x = lm_x_cat # shape = [sum(n_nodes), embed_dim]
        if False:
            ## sanity check here
            assert (lm_x[~data_batch.rdmol2selfies_mask] == 0).all()
            assert (lm_x[data_batch.rdmol2selfies_mask].sum(-1) != 0).any()
            print(data_batch.atom_vec[~data_batch.rdmol2selfies_mask].unique())
            print('pass')

        ## compute the conformation generation loss
        batch_size_smiles = len(data_batch['smiles'])
        total_num_nodes = data_batch.x.shape[0]
        position_prediction, noise_prediction = self.diffusion_model(data_batch)
        pos_t_batch, _ = to_dense_batch(data_batch.pos, data_batch.batch, batch_size=batch_size_smiles)
        pos_0_batch, _ = to_dense_batch(data_batch.gt_pos, data_batch.batch, batch_size=batch_size_smiles)

        if self.pred_noise:
            pred_noise_batch, _ = to_dense_batch(noise_prediction, data_batch.batch, batch_size=batch_size_smiles)
            gt_noise_batch, _ = to_dense_batch(data_batch.noise, data_batch.batch, batch_size=batch_size_smiles)
            noise_loss = self.get_noise_loss(pred_noise_batch, gt_noise_batch, pos_0_batch, pos_t_batch, data_batch.alpha_t_batch, data_batch.sigma_t_batch, total_num_nodes, self.reduce_mean, self.align_loss)
            coordinate_loss = noise_loss
        else:
            pred_pos_batch, _ = to_dense_batch(position_prediction, data_batch.batch, batch_size=batch_size_smiles)
            pos_loss = self.get_pos_loss(pred_pos_batch, pos_0_batch, pos_t_batch, total_num_nodes, data_batch.loss_norm, self.reduce_mean, self.align_loss)
            coordinate_loss = pos_loss

        loss = self.args.lm_loss * lm_loss + self.args.cg_loss * coordinate_loss

        return loss, lm_loss, coordinate_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group()
        parser.add_argument('--llm_model', type=str, default="all_checkpoints/mollama")
        parser.add_argument('--num_beams', type=int, default=1)
        # parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--tune_embedding', action='store_true', default=True)
        parser.add_argument('--sample_num', type=int, default=10000)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--generate_eval_epoch', type=int, default=10)
        parser.add_argument('--conform_eval_epoch', type=int, default=2)
        parser.add_argument('--test_conform_epoch', type=int, default=20)
        parser.add_argument('--not_align_loss', action='store_true', default=False)
        parser.add_argument('--bi_attend', action='store_true', default=False)
        parser.add_argument('--lm_loss', type=float, default=1.0)
        parser.add_argument('--cg_loss', type=float, default=1.0)

        parser.add_argument('--noise_scheduler', type=str, default='cosine')
        parser.add_argument('--continuous_beta_0', type=float, default=0.1)
        parser.add_argument('--continuous_beta_1', type=float, default=20)
        parser.add_argument('--sampling_steps', type=int, default=100)


        parser.add_argument('--conf_gen_tune', type=str, default="full")

        parser.add_argument('--detach_lm', action='store_true', default=False)
        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or
        parser.add_argument('--not_reduce_mean', action='store_true', default=False) # or

        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default=None)


        JODODiffusion.add_args(parser)
        return parent_parser

    def sample_selfies(
        self,
        batch_size,
        num_beams=5,
        max_length=30,
        temperature=1,
        num_output=1,
    ):
        # assert batch == None
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids = torch.LongTensor([[bos_token_id] for _ in range(batch_size)]).to(self.device)
        self.llm_model.set_mode('causal')
        outputs = self.llm_model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=1,
            eos_token_id=eos_token_id,
            num_return_sequences=num_output)
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text


def canonicalize_selfies(selfies):
    smiles = sf.decoder(selfies)
    try:
        canon_smiles = Chem.CanonSmiles(smiles)
    except Exception:
        return '', ''
    try:
        canon_selfies = sf.encoder(canon_smiles)
    except:
        return '', ''
    return canon_selfies, canon_smiles, smiles
