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
from model.conf_gen import UnimolConfGModel, UnimolConfGModelV2, UnimolConfGModelV3, UnimolConfGModelV4
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


class Mol123DGenerate(L.LightningModule):
    def set_trainble_params(self, param_list):
        self.delta_train = True
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

    def restore_trainble_params(self):
        self.delta_train = False
        for name, param in self.named_parameters():
            param.requires_grad = self.orignal_requires_grad[name]
        print('restore trainble params')

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
        # elif self.args.scheduler == 'linear_warmup_step_lr':
        #     self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
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
        dictionary = Dictionary.load('./data_provider/unimol_dict.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        if args.unimol_version == 'v1':
            unimol_model = UnimolConfGModel(args, dictionary)
        elif args.unimol_version == 'v2':
            unimol_model = UnimolConfGModelV2(args, dictionary)
        elif args.unimol_version == 'v3':
            unimol_model = UnimolConfGModelV3(args, dictionary)
        elif args.unimol_version == 'v4':
            unimol_model = UnimolConfGModelV4(args, dictionary)
        else:
            raise NotImplementedError()
        ckpt = torch.load(args.unimol_path, map_location=torch.device('cpu'))['model']

        strict = True if args.unimol_version not in {'v2', 'v4'} else False
        if str(args.unimol_path).find('qm9_220908') >= 0:
            unimol_model.load_state_dict(ckpt, strict=strict)
        elif str(args.unimol_path).find('drugs_220908') >= 0:
            unimol_model.load_state_dict(ckpt, strict=strict)
        elif str(args.unimol_path).find('mol_pre_no_h_220816') >= 0:
            ckpt.pop('encoder.final_head_layer_norm.weight')
            ckpt.pop('encoder.final_head_layer_norm.bias')
            unimol_model.unimol.load_state_dict(ckpt, strict=strict)
        else:
            raise NotImplementedError()


        if args.unimol_version in {'v3', 'v4'}:
            print('resize gbf embeddding')
            unimol_model.unimol.gbf.resize_embedding(5)

        if args.conf_gen_tune == 'freeze':
            for param in unimol_model.parameters():
                param.requires_grad = False
        elif args.conf_gen_tune == 'full':
            for param in unimol_model.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError()
        return unimol_model, dictionary


    def resize_token_embeddings(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        self.llm_model.resize_token_embeddings(len(tokenizer))

    def __init__(self, args, tokenizer=None, max_sf_tokens=30):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.detach_lm = args.detach_lm
        self.max_sf_tokens = max_sf_tokens

        ## init llm
        self.llm_model = self.init_llm(args)
        if tokenizer is None:
            self.tokenizer = self.init_tokenizer(args)
        else:
            self.tokenizer = tokenizer

        ## init unimol
        self.unimol_conf, self.dictionary = self.init_conf_generator(args)

        self.conf_loss = MyMolConfGLoss(self.dictionary)

        ## init projector
        in_dim = self.llm_model.config.hidden_size
        out_dim = self.unimol_conf.unimol.unimol_encoder_embed_dim
        self.projector = nn.Sequential(nn.Linear(in_dim, out_dim),
                                       nn.GELU(),
                                       nn.Linear(out_dim, out_dim))

        self.delta_train = False
        self.resize_token_embeddings(self.tokenizer)
        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)

        rdmol_batch, selfies_batch = batch
        loss, lm_loss, distance_loss, coord_loss = self.forward(rdmol_batch, selfies_batch)

        batch_size = selfies_batch.input_ids.shape[0]
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
        self.log('train/loss', loss, sync_dist=True, batch_size=batch_size)
        self.log('train/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
        self.log('train/distance_loss', distance_loss, sync_dist=True, batch_size=batch_size)
        self.log('train/coord_loss', coord_loss, sync_dist=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.rdmol_list = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        During validation, each batch contains 10 different init conformations for each molecule. I will evaluate:
        1) the lm_loss, distance_loss, coord_loss for each molecule
        2) the performance of conformation prediction.
        '''
        train_epoch_condition = (self.current_epoch + 1) % self.args.conform_eval_epoch == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval', 'eval_conf'}
        if train_epoch_condition or eval_condition:
            rdmol_batch, selfies_batch = batch

            loss, lm_loss, distance_loss, coord_loss, coords_predict_list = self.forward(rdmol_batch, selfies_batch, return_conformers=True)
            batch_size = selfies_batch.input_ids.shape[0]
            self.log('val/loss', loss, sync_dist=True, batch_size=batch_size)
            self.log('val/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
            self.log('val/distance_loss', distance_loss, sync_dist=True, batch_size=batch_size)
            self.log('val/coord_loss', coord_loss, sync_dist=True, batch_size=batch_size)

            ## prepare data for evaluation of the conformation generation performance
            rdmols = copy.deepcopy(rdmol_batch.rdmols)
            for i in range(len(rdmols)):
                mol = rdmols[i]
                coords_predict = coords_predict_list[i]
                rdmols[i] = set_rdmol_positions(mol, coords_predict, removeHs=False, add_conformer=True)

            ## rdmols; ## the first conformer of these rdmols are the ground truth; the second conformers of these rdmols are the rdkit predicted conformers; the third conformers of these rdmols are the unimol predicted conformers
            self.rdmol_list.extend(rdmols)

    @torch.no_grad()
    @torch.amp.autocast("cuda",dtype=torch.bfloat16)
    def on_validation_epoch_end(self):
        train_epoch_condition = (self.current_epoch + 1) % self.args.conform_eval_epoch == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval', 'eval_conf'}
        if train_epoch_condition or eval_condition:
            ## evaluate the conformation generation performance
            rdmol_list = self.rdmol_list
            assert len(rdmol_list) % 10 == 0

            threshold = 0.5
            rdmol_list_list = [rdmol_list[i:i+10] for i in range(0, len(rdmol_list), 10)]
            cov_list = []
            mat_list = []
            for rdmol_list in rdmol_list_list:
                ## sanity check
                smiles_list = [Chem.MolToSmiles(mol) for mol in rdmol_list]
                assert len(set(smiles_list)) == 1

                ## creat the gt_rdmol
                gt_rdmol = copy.deepcopy(rdmol_list[0])
                gt_rdmol.RemoveAllConformers()
                gt_rdmol.AddConformer(rdmol_list[0].GetConformer(0), assignId=True)

                rmsd_mat = np.zeros((1, len(rdmol_list)))
                for i in range(len(rdmol_list)):
                    rdmol = rdmol_list[i]
                    conf = rdmol.GetConformer(2)
                    rdmol = copy.deepcopy(rdmol)
                    rdmol.RemoveAllConformers()
                    rdmol.AddConformer(conf, assignId=True)
                    rmsd = get_best_rmsd(rdmol, gt_rdmol)
                    rmsd_mat[0, i] = rmsd

                rmsd_mat_min = rmsd_mat.min(axis=-1)
                cov = (rmsd_mat_min <= threshold).mean()
                mat = rmsd_mat_min.mean()
                cov_list.append(cov)
                mat_list.append(mat)
            cov_mean = np.mean(cov_list)
            mat_mean = np.mean(mat_list)
            cov_median = np.median(cov_list)
            mat_median = np.median(mat_list)
            self.log('val/cov_mean', cov_mean, sync_dist=True)
            self.log('val/mat_mean', mat_mean, sync_dist=True)
            self.log('val/cov_median', cov_median, sync_dist=True)
            self.log('val/mat_median', mat_median, sync_dist=True)

        if self.args.mode == 'delta_train':
            return

        train_epoch_condition = (self.current_epoch + 1) % self.args.generate_eval_epoch == 0 and self.current_epoch > 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval', 'eval_gen', 'eval_1d_gen'}
        if train_epoch_condition or eval_condition:
            log_dir = Path(self.logger.log_dir)
            if self.args.eval_smiles_path is not None:
                with open(self.args.eval_smiles_path) as f:
                    lines = f.readlines()
                    sampled_smiles = [line.strip() for line in lines]
                    sampled_canon_smiles, sampled_orig_smiles = zip(*sampled_smiles)
            else:
                sampled_smiles = self.sample_molecules()
                sampled_canon_smiles, sampled_orig_smiles = zip(*sampled_smiles)

            ## compute the moses metrics
            sampled_rdmols = [Chem.MolFromSmiles(smiles) for smiles in sampled_canon_smiles]
            sampled_rdmols = [Chem.AddHs(mol) for mol in sampled_rdmols]
            eval_results_2d = get_2D_edm_metric(sampled_rdmols, self.trainer.datamodule.train_rdmols)
            self.log('test/mol_stable', eval_results_2d['mol_stable'], sync_dist=True)
            self.log('test/atom_stable', eval_results_2d['atom_stable'], sync_dist=True)
            self.log('test/validity', eval_results_2d['Validity'], sync_dist=True)
            self.log('test/novelty', eval_results_2d['Novelty'], sync_dist=True)
            self.log('test/complete', eval_results_2d['Complete'], sync_dist=True)

            moses_metrics = self.trainer.datamodule.get_moses_metrics(sampled_rdmols)
            self.log('test/fcd', moses_metrics['FCD'], sync_dist=True)
            self.log('test/snn', moses_metrics['SNN'], sync_dist=True)
            self.log('test/frag', moses_metrics['Frag'], sync_dist=True)
            self.log('test/scaf', moses_metrics['Scaf'], sync_dist=True)
            self.log('test/intdiv', moses_metrics['IntDiv'], sync_dist=True)
            self.log('test/filters', moses_metrics['Filters'], sync_dist=True)
            self.log('test/qed', moses_metrics['QED'], sync_dist=True)
            self.log('test/sa', moses_metrics['SA'], sync_dist=True)
            self.log('test/logp', moses_metrics['logP'], sync_dist=True)
            self.log('test/weight', moses_metrics['weight'], sync_dist=True)

            if self.args.mode == 'eval_1d_gen':
                return
            self.trainer.datamodule.setup_predict_dataset(sampled_orig_smiles)
            save_path = log_dir / f'coords_epoch{self.current_epoch}.pt'
            dataloader = self.trainer.datamodule.predict_dataloader()

            predictions = []
            for batch in tqdm(dataloader, desc='Predict molecule coords'):
                rdmol_batch, selfies_batch = batch
                rdmol_batch = rdmol_batch.to(self.device)
                selfies_batch = selfies_batch.to(self.device)
                coords_predict_list = self.generate_conformer(rdmol_batch, selfies_batch)
                for i in range(len(coords_predict_list)):
                    rdmol = set_rdmol_positions(rdmol_batch.rdmols[i], coords_predict_list[i], removeHs=False)
                    smiles = rdmol_batch.smiles[i]
                    selfies = rdmol_batch.selfies[i]

                    ## save the predictions. The first rdmol's conformer is predicted by unimol, the second rdmol's conformer is predicted by rdkit
                    predictions.append((smiles, selfies, rdmol, rdmol_batch.rdmols[i]))
            ## save_predictions
            torch.save(predictions, save_path)

            ## conduct evaluation
            predict_mols = [item[2] for item in predictions]
            eval_results_3d_unimol = get_3D_edm_metric(predict_mols, self.trainer.datamodule.train_rdmols)
            sub_geometry_metric = self.trainer.datamodule.get_sub_geometry_metric(predict_mols)

            ## evaluate rdkit performance
            rdkit_predict_mols = [item[3] for item in predictions]
            eval_results_3d_rdkit = get_3D_edm_metric(rdkit_predict_mols, self.trainer.datamodule.train_rdmols)
            sub_geometry_metric_rdkit = self.trainer.datamodule.get_sub_geometry_metric(rdkit_predict_mols)


            self.log('test/mol_stable_3d_unimol', eval_results_3d_unimol['mol_stable'], sync_dist=True)
            self.log('test/atom_stable_3d_unimol', eval_results_3d_unimol['atom_stable'], sync_dist=True)
            self.log('test/validity_3d_unimol', eval_results_3d_unimol['Validity'], sync_dist=True)
            self.log('test/novelty_3d_unimol', eval_results_3d_unimol['Novelty'], sync_dist=True)
            self.log('test/complete_3d_unimol', eval_results_3d_unimol['Complete'], sync_dist=True)

            self.log('test/mol_stable_3d_rdkit', eval_results_3d_rdkit['mol_stable'], sync_dist=True)
            self.log('test/atom_stable_3d_rdkit', eval_results_3d_rdkit['atom_stable'], sync_dist=True)
            self.log('test/validity_3d_rdkit', eval_results_3d_rdkit['Validity'], sync_dist=True)
            self.log('test/novelty_3d_rdkit', eval_results_3d_rdkit['Novelty'], sync_dist=True)
            self.log('test/complete_3d_rdkit', eval_results_3d_rdkit['Complete'], sync_dist=True)

            self.log('test/bond_length_mean_unimol', sub_geometry_metric['bond_length_mean'], sync_dist=True)
            self.log('test/bond_angle_mean_unimol', sub_geometry_metric['bond_angle_mean'], sync_dist=True)
            self.log('test/dihedral_angle_mean_unimol', sub_geometry_metric['dihedral_angle_mean'], sync_dist=True)

            self.log('test/bond_length_mean_rdkit', sub_geometry_metric_rdkit['bond_length_mean'], sync_dist=True)
            self.log('test/bond_angle_mean_rdkit', sub_geometry_metric_rdkit['bond_angle_mean'], sync_dist=True)
            self.log('test/dihedral_angle_mean_rdkit', sub_geometry_metric_rdkit['dihedral_angle_mean'], sync_dist=True)


    @torch.no_grad()
    def sample_molecules(self):
        ## sample selfies from the molecule language model
        sample_num = self.args.sample_num
        print('sample_num:', sample_num)
        loop_count = 0
        sampled_smiles = [] # we use smiles as the intermediate data structure for its easy conversion to rdkit mol
        pbar = tqdm(total=sample_num, desc='sample molecules sequences')
        while True:
            sf_list = self.sample_selfies(
                batch_size=200,
                num_beams=self.args.num_beams,
                temperature=self.args.temperature,
                num_output=1,
                max_length=self.max_sf_tokens - 1) # -1 for the bos token, which is already included

            canon_list = [canonicalize_selfies(item) for item in sf_list]
            smiles_list = []
            for canon_selfies, canon_smiles, orig_smiles in canon_list:
                if not canon_selfies:
                    continue
                selfies_tokens = sf.split_selfies(canon_selfies)
                skip = False
                for token in selfies_tokens:
                    if token not in self.tokenizer.vocab:
                        skip = True
                        break
                if skip:
                    continue
                smiles_list.append((canon_smiles, orig_smiles))

            sampled_smiles.extend(smiles_list)
            loop_count += 1
            pbar.update(len(sampled_smiles)-pbar.n)
            if len(sampled_smiles) >= sample_num:
                pbar.close()
                break
        print(f'loop count: {loop_count}')
        sampled_smiles = list(sampled_smiles)[:sample_num]
        sampled_smiles.sort()

        log_dir = Path(self.logger.log_dir)
        ## save the sampled smiles
        save_path = log_dir / f'smiles_epoch{self.current_epoch}.txt'
        with save_path.open('w', encoding='utf8') as f:
            for canon_smiles, orig_smiles in sampled_smiles:
                f.write(f'{canon_smiles}\t{orig_smiles}' + '\n')
        return sampled_smiles


    def forward(self, rdmol_batch, selfies_batch, return_conformers=False):
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
        rdmol_batch.rdmol2selfies = rdmol_batch.rdmol2selfies.to(lm_embeds.dtype)
        lm_x = torch.bmm(rdmol_batch.rdmol2selfies, lm_embeds) # shape = [batch_size, rdmol_len, selfies_len], [batch_size, selfies_len, hidden_size] -> [batch_size, rdmol_len, hidden_size]
        norm = torch.clamp(torch.sum(rdmol_batch.rdmol2selfies, dim=-1, keepdim=True), min=1) # shape = [batch_size, 1, 1]
        lm_x = self.projector(lm_x / norm) # shape = [batch_size, rdmol_len, hidden_size]
        x = self.unimol_conf.unimol.embed_tokens(rdmol_batch.atom_vec) # shape = [batch_size, seq_len, embed_dim]
        x = x + lm_x # shape = [batch_size, rdmol_len, embed_dim]

        if False:
            ## sanity check here
            assert (lm_x[~rdmol_batch.rdmol2selfies_mask] == 0).all()
            assert (lm_x[rdmol_batch.rdmol2selfies_mask].sum(-1) != 0).any()
            print(rdmol_batch.atom_vec[~rdmol_batch.rdmol2selfies_mask].unique())
            print('pass')

        ## compute the conformation generation loss
        if self.args.unimol_version == 'v1':
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type, encoded_atom_x=x)
        else:
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type, rdmol_batch.bond_type, encoded_atom_x=x)
        distance_loss, coord_loss = self.conf_loss(rdmol_batch.atom_vec, distance_predict, coords_predict, rdmol_batch.tgt_dist, rdmol_batch.tgt_coordinates)
        loss = self.args.lm_loss * lm_loss + self.args.unimol_distance_loss * distance_loss + self.args.unimol_coord_loss * coord_loss

        if not return_conformers:
            return loss, lm_loss, distance_loss, coord_loss

        token_mask = self.conf_loss.get_token_mask(rdmol_batch.atom_vec)
        coords_predict_list = []
        for i in range(coords_predict.shape[0]):
            coords = coords_predict[i]
            coords = coords[token_mask[i]]
            coords_predict_list.append(coords.cpu().numpy())
        return loss, lm_loss, distance_loss, coord_loss, coords_predict_list

    @torch.no_grad()
    def generate_conformer(self, rdmol_batch, selfies_batch):
        if not self.args.bi_attend:
            self.llm_model.set_mode('causal')
            outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                    attention_mask=selfies_batch.attention_mask,
                                    return_dict=True,
                                    output_hidden_states=True)
            lm_embeds = outputs.hidden_states[-1] # shape = [batch_size, seq_len, hidden_size]
        else:
            self.llm_model.set_mode('noncausal')
            outputs = self.llm_model(input_ids=selfies_batch.input_ids,
                                     attention_mask=selfies_batch.attention_mask,
                                     return_dict=True,
                                     output_hidden_states=True)
            lm_embeds = outputs.hidden_states[-1]

        ## use the last hidden state as the representation of the molecule
        rdmol_batch.rdmol2selfies = rdmol_batch.rdmol2selfies.to(lm_embeds.dtype)
        lm_x = torch.bmm(rdmol_batch.rdmol2selfies, lm_embeds) # shape = [batch_size, rdmol_len, selfies_len], [batch_size, selfies_len, hidden_size] -> [batch_size, rdmol_len, hidden_size]
        norm = torch.clamp(torch.sum(rdmol_batch.rdmol2selfies, dim=-1, keepdim=True), min=1) # shape = [batch_size, 1, 1]
        lm_x = self.projector(lm_x / norm) # shape = [batch_size, rdmol_len, hidden_size]
        x = self.unimol_conf.unimol.embed_tokens(rdmol_batch.atom_vec) # shape = [batch_size, seq_len, embed_dim]
        x = x + lm_x # shape = [batch_size, rdmol_len, embed_dim]

        if False:
            ## sanity check here
            assert (lm_x[~rdmol_batch.rdmol2selfies_mask] == 0).all()
            assert (lm_x[rdmol_batch.rdmol2selfies_mask].sum(-1) != 0).any()
            print(rdmol_batch.atom_vec[~rdmol_batch.rdmol2selfies_mask].unique())
            print('pass')

        ## compute the conformation generation loss
        if self.args.unimol_version == 'v1':
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type, encoded_atom_x=x)
        else:
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type, rdmol_batch.bond_type, encoded_atom_x=x)

        token_mask = self.conf_loss.get_token_mask(rdmol_batch.atom_vec)
        coords_predict_list = []
        for i in range(coords_predict.shape[0]):
            coords = coords_predict[i]
            coords = coords[token_mask[i]]
            coords_predict_list.append(coords.cpu().numpy())
        return coords_predict_list


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
        parser.add_argument('--eval_smiles_path', type=str, default=None)
        parser.add_argument('--bi_attend', action='store_true', default=False)
        parser.add_argument('--lm_loss', type=float, default=1.0)

        parser.add_argument('--unimol_version', type=str, default='v1')
        parser.add_argument('--unimol_path', type=str, default="unimol_ckpt/mol_pre_no_h_220816.pt")
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

        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default=None)


        ## add unimol-conf config
        UnimolConfGModel.add_args(parser)
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
