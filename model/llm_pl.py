"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import math
import torch
from torch import optim
import lightning as L
from transformers import AutoTokenizer
# from model.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers import LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, LoraConfig
from model.help_funcs import AttrDict
from pathlib import Path
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
from evaluation.eval_functions import get_2D_edm_metric
from torch.nn import CrossEntropyLoss

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


def obtain_loss_and_ppl(logits, labels, attn_mask, return_nll=False, context_length=0):
    if context_length > 0:
        logits = logits[:, context_length:, :]
        labels = labels[:, context_length:]
        attn_mask = attn_mask[:, context_length:]

    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    nll = loss_fct(shift_logits.transpose(1,2), shift_labels) * shift_attention_mask_batch
    loss = nll.sum() / shift_attention_mask_batch.sum()
    if return_nll:
        avg_nll = nll.sum(dim=1) / shift_attention_mask_batch.sum(dim=1)
        return loss, avg_nll
    else:
        ppl = torch.exp(nll.sum(dim=1) / shift_attention_mask_batch.sum(dim=1))
        return loss, ppl


class LLMPL(L.LightningModule):
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
        config = LlamaConfig.from_pretrained(args.llm_model)
        config.attention_dropout = args.attention_dropout
        if args.load_random_llm:
            if args.use_flash_attention:
                config._attn_implementation = 'flash_attention_2'
            llm_model = LlamaForCausalLM(config).to(get_half_precision_dtype())
        else:
            if args.use_flash_attention:
                llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, config=config, torch_dtype=get_half_precision_dtype(), attn_implementation='flash_attention_2')
            else:
                llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, config=config, torch_dtype=get_half_precision_dtype())

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

    def create_condition_prompt(self, context):
        batch_size = context.shape[0]
        context = context.unsqueeze(1)  # [batch_size, 1]
        out = self.condition_mlp(context)  # [batch_size, 4 * hidden_size]
        return out.view(batch_size, 4, self.hidden_size)

    def resize_token_embeddings(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        self.llm_model.resize_token_embeddings(len(tokenizer))

    def __init__(self, args, tokenizer=None, max_sf_tokens=30, property_distribution=None):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.aug_inv = args.aug_inv
        self.max_sf_tokens = max_sf_tokens

        ## init llm
        self.llm_model = self.init_llm(args)
        if tokenizer is None:
            self.tokenizer = self.init_tokenizer(args)
        else:
            self.tokenizer = tokenizer

        self.delta_train = False
        self.resize_token_embeddings(self.tokenizer)
        self.tokenizer.num_metaqueries = args.num_metaqueries

        self.hidden_size = self.llm_model.config.hidden_size
        self.condition_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, self.hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 4, 4 * self.hidden_size)
        )

        self.property_distribution = property_distribution

        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)

        if self.aug_inv > 0:
            selfies_batch0, selfies_batch1 = batch
            lm_loss0, ppl0 = self.forward(selfies_batch0)
            lm_loss1, ppl1 = self.forward(selfies_batch1)
            lm_loss = (lm_loss0 + lm_loss1) / 2
            inv_loss = ((ppl0 - ppl1) ** 2).mean()
            loss = lm_loss + self.aug_inv * inv_loss
            batch_size = selfies_batch0.input_ids.shape[0]
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
            self.log('train/loss', loss, sync_dist=True, batch_size=batch_size)
            self.log('train/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
            self.log('train/inv_loss', inv_loss, sync_dist=True, batch_size=batch_size)
            return loss
        else:
            selfies_batch = batch
            lm_loss, _ = self.forward(selfies_batch)
            batch_size = selfies_batch.input_ids.shape[0]
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
            self.log('train/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
            return lm_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        During validation, each batch contains 10 different init conformations for each molecule. I will evaluate:
        1) the lm_loss, distance_loss, coord_loss for each molecule
        2) the performance of conformation prediction.
        '''
        train_epoch_condition = (self.current_epoch + 1) % self.args.conform_eval_epoch == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval', 'eval_conf'}
        if not train_epoch_condition and not eval_condition:
            return

        if self.aug_inv > 0:
            selfies_batch0, selfies_batch1 = batch
            lm_loss0, ppl0 = self.forward(selfies_batch0)
            lm_loss1, ppl1 = self.forward(selfies_batch1)
            lm_loss = (lm_loss0 + lm_loss1) / 2
            inv_loss = ((ppl0 - ppl1) ** 2).mean()
            loss = lm_loss + self.aug_inv * inv_loss
            batch_size = selfies_batch0.input_ids.shape[0]
            self.log('val/loss', loss, sync_dist=True, batch_size=batch_size)
            self.log('val/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
            self.log('val/inv_loss', inv_loss, sync_dist=True, batch_size=batch_size)
        else:
            selfies_batch = batch
            lm_loss, _ = self.forward(selfies_batch)
            batch_size = selfies_batch.input_ids.shape[0]
            self.log('val/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)


    @torch.no_grad()
    @torch.amp.autocast("cuda",dtype=torch.bfloat16)
    def on_validation_epoch_end(self):
        train_epoch_condition = (self.current_epoch + 1) % self.args.generate_eval_epoch == 0 and self.args.mode == 'train'
        eval_condition = self.args.mode in {'eval'}
        if not train_epoch_condition and not eval_condition:
            return

        if not self.trainer.is_global_zero:
            return

        sampled_sequences = self.sample_molecules()

        if self.args.skip_eval:
            return


        smiles_without_chirality_list = [results_tuple[2] for results_tuple in sampled_sequences]
        ## compute the moses metrics
        sampled_rdmols = [Chem.MolFromSmiles(smiles_without_chirality) for smiles_without_chirality in smiles_without_chirality_list]
        sampled_rdmols = [Chem.AddHs(mol) for mol in sampled_rdmols if mol is not None]
        eval_results_2d = get_2D_edm_metric(sampled_rdmols, self.trainer.datamodule.train_rdmols)
        self.log('test/mol_stable', eval_results_2d['mol_stable'], rank_zero_only=True)
        self.log('test/atom_stable', eval_results_2d['atom_stable'], rank_zero_only=True)
        self.log('test/validity', eval_results_2d['Validity'], rank_zero_only=True)
        self.log('test/novelty', eval_results_2d['Novelty'], rank_zero_only=True)
        self.log('test/complete', eval_results_2d['Complete'], rank_zero_only=True)
        self.log('test/unique', eval_results_2d['Unique'], rank_zero_only=True)

        moses_metrics = self.trainer.datamodule.get_moses_metrics(sampled_rdmols)
        self.log('test/fcd', moses_metrics['FCD'], rank_zero_only=True)
        self.log('test/snn', moses_metrics['SNN'], rank_zero_only=True)
        self.log('test/frag', moses_metrics['Frag'], rank_zero_only=True)
        self.log('test/scaf', moses_metrics['Scaf'], rank_zero_only=True)
        self.log('test/intdiv', moses_metrics['IntDiv'], rank_zero_only=True)
        self.log('test/filters', moses_metrics['Filters'], rank_zero_only=True)
        self.log('test/qed', moses_metrics['QED'], rank_zero_only=True)
        self.log('test/sa', moses_metrics['SA'], rank_zero_only=True)
        self.log('test/logp', moses_metrics['logP'], rank_zero_only=True)
        self.log('test/weight', moses_metrics['weight'], rank_zero_only=True)


    @torch.no_grad()
    def sample_molecules(self):
        ## sample selfies from the molecule language model
        sample_num = self.args.sample_num
        print('sample_num:', sample_num)
        loop_count = 0
        sampled_sequences = [] # we use smiles as the intermediate data structure for its easy conversion to rdkit mol
        pbar = tqdm(total=sample_num, desc='sample molecules sequences')

        while True:
            sf_list, context = self.sample_selfies(
                batch_size=200,
                num_beams=self.args.num_beams,
                temperature=self.args.temperature,
                num_output=1,
                max_length=self.max_sf_tokens - 1) # -1 for the bos token, which is already included

            tuple_list = [reencode_selfies(item) for item in sf_list]
            tuple_list_valid = []
            for index, (selfies, smiles_with_chirality, smiles_without_chirality) in enumerate(tuple_list):
                if not selfies:
                    continue
                selfies_tokens = sf.split_selfies(selfies)
                skip = False
                for token in selfies_tokens:
                    if token not in self.tokenizer.vocab:
                        skip = True
                        break
                if skip:
                    continue
                if context is not None:
                    tuple_list_valid.append((selfies, smiles_with_chirality, smiles_without_chirality, context[index]))
                else:
                    tuple_list_valid.append((selfies, smiles_with_chirality, smiles_without_chirality))

            sampled_sequences.extend(tuple_list_valid)
            loop_count += 1
            pbar.update(len(tuple_list_valid))
            pbar.set_postfix(loop_count=loop_count)
            if len(sampled_sequences) >= sample_num:
                pbar.close()
                break

        sampled_sequences = list(sampled_sequences)[:sample_num]
        sampled_sequences.sort()

        log_dir = Path(self.logger.log_dir)
        ## save the sampled sequences
        if self.args.condition_property is None:
            save_path = log_dir / f'sequences_epoch{self.current_epoch}.txt'
            with save_path.open('w', encoding='utf8') as f:
                for selfies, smiles_with_chirality, smiles_without_chirality in sampled_sequences:
                    f.write(f'{selfies}\t{smiles_with_chirality}\t{smiles_without_chirality}' + '\n')
        else:
            save_path = log_dir / f'sequences_epoch{self.current_epoch}_{self.args.condition_property}.txt'
            with save_path.open('w', encoding='utf8') as f:
                for selfies, smiles_with_chirality, smiles_without_chirality, context in sampled_sequences:
                    f.write(f'{selfies}\t{smiles_with_chirality}\t{smiles_without_chirality}\t{context}' + '\n')

        return sampled_sequences


    def forward(self, selfies_batch):
        if hasattr(selfies_batch, "context"):
            context_length = 4
            token_embeds = self.llm_model.get_input_embeddings()(selfies_batch.input_ids)
            condition_embeds = self.create_condition_prompt(selfies_batch.context)
            inputs_embeds = torch.cat([condition_embeds, token_embeds], dim=1)
            soft_prompt_attention = torch.ones((selfies_batch.attention_mask.shape[0], context_length),
                                            device=selfies_batch.attention_mask.device)
            attention_mask = torch.cat([soft_prompt_attention, selfies_batch.attention_mask], dim=1)

            ignore_prefix = torch.full((selfies_batch.input_ids.shape[0], 4), -100,
                                    device=selfies_batch.input_ids.device)
            targets = torch.cat([ignore_prefix, selfies_batch.input_ids], dim=1)
            targets = targets.masked_fill(~attention_mask.bool(), -100)

            outputs = self.llm_model(inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                # labels=target,
                output_hidden_states=True)
        else:
            context_length = 0
            input_ids = selfies_batch.input_ids
            attention_mask = selfies_batch.attention_mask
            targets = selfies_batch.input_ids.masked_fill(~selfies_batch.attention_mask.bool(), -100)
            outputs = self.llm_model(input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                # labels=targets,
                output_hidden_states=True)

        lm_loss, avg_nll = obtain_loss_and_ppl(outputs.logits, targets, attention_mask, True, context_length)
        return lm_loss, avg_nll


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group()
        parser.add_argument('--llm_model', type=str, default="all_checkpoints/mollama")
        parser.add_argument('--load_random_llm', action='store_true', default=False)
        parser.add_argument('--num_beams', type=int, default=1)
        # parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--tune_embedding', action='store_true', default=True)
        parser.add_argument('--sample_num', type=int, default=10000)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--generate_eval_epoch', type=int, default=10)
        parser.add_argument('--conform_eval_epoch', type=int, default=2)
        # parser.add_argument('--eval_smiles_path', type=str, default=None)
        parser.add_argument('--bi_attend', action='store_true', default=False)
        parser.add_argument('--lm_loss', type=float, default=1.0)
        parser.add_argument('--num_metaqueries', type=int, default=0)
        parser.add_argument('--use_adameta', action='store_true', default=False)

        parser.add_argument('--aug_inv', type=float, default=0)

        ## llm config
        parser.add_argument('--attention_dropout', type=float, default=0)

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
        parser.add_argument('--skip_eval', action='store_true', default=False)

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

        if self.property_distribution is not None:
            context = self.property_distribution.sample_batch(batch_size).to(self.device)
        else:
            context = None

        if context is not None:
            condition_embeds = self.create_condition_prompt(context)
            input_embeds = self.llm_model.get_input_embeddings()(torch.LongTensor([[bos_token_id]]).to(self.device))
            inputs_embeds = torch.cat([condition_embeds, input_embeds.repeat(batch_size, 1, 1)], dim=1)
            attention_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=self.device)
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=1,
                eos_token_id=eos_token_id,
                num_return_sequences=num_output,
                use_cache=True
            )
        else:
            input_ids = torch.LongTensor([[bos_token_id] for _ in range(batch_size)]).to(self.device)
            outputs = self.llm_model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=1,
                eos_token_id=eos_token_id,
                num_return_sequences=num_output
            )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if context is not None:
            return output_text, context.squeeze(1).tolist()
        return output_text, None


def canonicalize_selfies(selfies):
    smiles = sf.decoder(selfies)
    try:
        canon_smiles = Chem.CanonSmiles(smiles)
        canon_selfies = sf.encoder(canon_smiles)
    except Exception:
        return '', '', ''
    return canon_selfies, canon_smiles, smiles

def reencode_selfies(selfies):
    decoded_smiles = sf.decoder(selfies)
    try:
        molecule = Chem.MolFromSmiles(decoded_smiles)
        smiles_with_chirality = Chem.MolToSmiles(molecule, kekuleSmiles=True)
        reencoded_selfies = sf.encoder(smiles_with_chirality)
        smiles_without_chirality = Chem.MolToSmiles(molecule, isomericSmiles=False)
    except Exception:
        return '', '', ''
    return reencoded_selfies, smiles_with_chirality, smiles_without_chirality