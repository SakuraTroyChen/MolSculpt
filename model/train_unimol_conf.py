import math
import torch
from torch import optim
import lightning as L
from model.conf_gen import UnimolConfGModel, UnimolConfGModelV2, UnimolConfGModelV3, UnimolConfGModelV4
from transformers import AutoTokenizer
from model.help_funcs import AttrDict
from rdkit import Chem
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


class UniMolConfTrain(L.LightningModule):
    def configure_optimizers(self):
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

        strict = True if args.unimol_version != 'v2' else False
        if str(args.unimol_path).find('qm9_220908') >= 0:
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


        self.tokenizer = tokenizer if tokenizer is not None else self.init_tokenizer(args)

        ## init unimol
        self.unimol_conf, self.dictionary = self.init_conf_generator(args)
        self.conf_loss = MyMolConfGLoss(self.dictionary)
        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step)

        rdmol_batch, selfies_batch = batch
        loss, distance_loss, coord_loss = self.forward(rdmol_batch)

        batch_size = selfies_batch.input_ids.shape[0]
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, batch_size=batch_size)
        self.log('train/loss', loss, sync_dist=True, batch_size=batch_size)
        # self.log('train/lm_loss', lm_loss, sync_dist=True, batch_size=batch_size)
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

        rdmol_batch, selfies_batch = batch

        loss, distance_loss, coord_loss, coords_predict_list = self.forward(rdmol_batch, return_conformers=True)
        batch_size = selfies_batch.input_ids.shape[0]
        self.log('val/loss', loss, sync_dist=True, batch_size=batch_size)
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
        ## evaluate the conformation generation performance
        rdmol_list = self.rdmol_list
        assert len(rdmol_list) % 10 == 0

        threshold = 0.5
        rdmol_list_list = [rdmol_list[i:i+10] for i in range(0, len(rdmol_list), 10)]
        cov_list = []
        mat_list = []
        predict_mol_list = []
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
                conformer_ids = [conf.GetId() for conf in rdmol.GetConformers()]
                predict_mol_list.append(rdmol)

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

        eval_results_3d_unimol, _ = get_3D_edm_metric(predict_mol_list)
        self.log('val/mol_stable_3d_unimol', eval_results_3d_unimol['mol_stable'], sync_dist=True)
        self.log('val/atom_stable_3d_unimol', eval_results_3d_unimol['atom_stable'], sync_dist=True)
        self.log('val/validity_3d_unimol', eval_results_3d_unimol['Validity'], sync_dist=True)
        self.log('val/novelty_3d_unimol', eval_results_3d_unimol['Novelty'], sync_dist=True)
        self.log('val/complete_3d_unimol', eval_results_3d_unimol['Complete'], sync_dist=True)


    def forward(self, rdmol_batch, return_conformers=False):
        ## use the last hidden state as the representation of the molecule
        ## compute the conformation generation loss
        if self.args.unimol_version == 'v1':
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type)
        else:
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type, rdmol_batch.bond_type)
        distance_loss, coord_loss = self.conf_loss(rdmol_batch.atom_vec, distance_predict, coords_predict, rdmol_batch.tgt_dist, rdmol_batch.tgt_coordinates)
        loss = self.args.unimol_distance_loss * distance_loss + self.args.unimol_coord_loss * coord_loss

        if not return_conformers:
            return loss, distance_loss, coord_loss

        token_mask = self.conf_loss.get_token_mask(rdmol_batch.atom_vec)
        coords_predict_list = []
        for i in range(coords_predict.shape[0]):
            coords = coords_predict[i]
            coords = coords[token_mask[i]]
            coords_predict_list.append(coords.cpu().numpy())
        return loss, distance_loss, coord_loss, coords_predict_list

    @torch.no_grad()
    def generate_conformer(self, rdmol_batch):
        ## compute the conformation generation loss
        if self.args.unimol_version == 'v1':
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type)
        else:
            distance_predict, coords_predict = self.unimol_conf(rdmol_batch.atom_vec, rdmol_batch.dist, rdmol_batch.coordinates, rdmol_batch.edge_type, rdmol_batch.bond_type)

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


        parser.add_argument('--unimol_version', type=str, default="v1")
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
        parser.add_argument('--init_checkpoint', type=str, default='')


        ## add unimol-conf config
        UnimolConfGModel.add_args(parser)
        return parent_parser
