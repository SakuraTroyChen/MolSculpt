import torch
import torch.utils
from torch.utils.data import DataLoader, Sampler, DistributedSampler
import random
import lightning as L
import torch.utils.data
from torch_geometric.data import Batch
from rdkit.Chem.rdchem import BondType as BT
from data_provider.qm9_dataset_tordf import QM9TorDF, QM9TorDFInfer
from data_provider.geom_dataset_tordf import GeomDrugsTorDF, GeomDrugsTorDFInfer
from pathlib import Path
from data_provider.diffusion_scheduler import NoiseScheduleVPV2
import math
from torch_scatter import scatter
from scipy.spatial.transform import Rotation
import torch.distributed as dist
from lightning.fabric.utilities.rank_zero import _get_rank

bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge
atom_name = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
atomic_number_list = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]


def sample_com_rand_pos(pos_shape, batch, bs=None):
    noise = torch.randn(pos_shape, device=batch.device) # shape = [\sum N_i, 3]
    if True:
        mean_noise = scatter(noise, batch, dim=0, reduce='mean') # shape = [B, 3]
        noise = noise - mean_noise[batch]
        assert len(noise.shape) == 2 and noise.shape[1] == 3
        return noise
    else:
        mean_noise = torch.empty((bs, noise.shape[1]), device=noise.device)
        mean_noise.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, noise.shape[1]), noise, reduce='mean', include_self=False)
        noise = noise - mean_noise[batch]
        return noise

class QM9Collater(object):
    def __init__(self, max_atoms: int, max_sf_tokens: int, selfies_tokenizer, noise_scheduler, aug_rotation=False, t_cond='t', use_eigvec=False, disable_com=False, aug_translation=False, load_mapping=True):
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer
        self.noise_scheduler = noise_scheduler
        self.aug_rotation = aug_rotation
        self.t_cond = t_cond
        self.use_eigvec = use_eigvec
        self.disable_com = disable_com
        self.aug_translation = aug_translation
        self.load_mapping = load_mapping

    def add_noise(self, data):
        t_eps = 1e-5

        bs = len(data['ptr']) - 1
        t = (torch.rand(1) + torch.linspace(0, 1, bs)) % 1
        data['t'] = t * (1. - t_eps) + t_eps ## time

        alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
        data['alpha_t_batch'] = alpha_t
        data['sigma_t_batch'] = sigma_t
        data['loss_norm'] = torch.sqrt(alpha_t / sigma_t)
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        noise_level, alpha_t, sigma_t = noise_level[data.batch], alpha_t[data.batch], sigma_t[data.batch]

        dtype = torch.float
        bs = len(data['smiles'])
        if self.aug_rotation:
            rot_aug = Rotation.random(bs)
            rot_aug = rot_aug[data.batch.numpy()]
            data['pos'] = torch.from_numpy(rot_aug.apply(data['pos'].numpy())).to(dtype)

        if self.aug_translation:
            trans_aug = 0.01 * torch.randn(bs, 3, dtype=dtype)
            data['pos'] = data['pos'] + trans_aug[data.batch]

        data['gt_pos'] = data['pos'].clone()

        ## sample noise the same size as the pos and remove the mean
        if self.disable_com:
            noise = torch.randn(data.pos.shape)
        else:
            noise = sample_com_rand_pos(data.pos.shape, data.batch, bs=bs)
        ## perturb the positions
        data['pos'] = alpha_t.view(-1, 1) * data['pos'] + sigma_t.view(-1, 1) * noise
        data['alpha_t'] = alpha_t.view(-1, 1)
        data['sigma_t'] = sigma_t.view(-1, 1)
        data['noise'] = noise
        if self.t_cond == 't':
            data['t_cond'] = t[data.batch]
        elif self.t_cond == 'noise_level':
            data['t_cond'] = noise_level
        else:
            raise ValueError(f'Unknown t_cond {self.t_cond}')
        return data

    def __call__(self, data_list):
        ## selfies
        selfies = [data['selfies'] for data in data_list]

        self.selfies_tokenizer.padding_side = 'right'

        if self.selfies_tokenizer.use_adameta:
            prompts = []
            for i, p in enumerate(selfies):
                num_metaqueries = data_list[i].rdmol2selfies.shape[0]
                suffix = (
                    "\n<begin_of_mol>"
                    + "".join([f"<mol{j}>" for j in range(num_metaqueries)])
                    + "<end_of_mol>"
                )
                prompts.append(p + suffix)
        else:
            if self.selfies_tokenizer.num_metaqueries <= 0:
                suffix = ""
            else:
                suffix = (
                    "\n<begin_of_mol>"
                    + "".join([f"<mol{i}>" for i in range(self.selfies_tokenizer.num_metaqueries)])
                    + "<end_of_mol>"
                )

            prompts = [p + suffix for p in selfies]

        selfie_batch = self.selfies_tokenizer(prompts, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)

        ## construct mapping from rdmol to selfies
        batch_size = len(data_list)
        rdmol2selfies = [data.pop('rdmol2selfies') for data in data_list]
        rdmol2selfies_mask = [data.pop('rdmol2selfies_mask') for data in data_list]
        [data.pop('passed_conf_matching', None) for data in data_list]

        ## graph batch
        data_batch = Batch.from_data_list(data_list)
        data_batch['max_seqlen'] = int((data_batch['ptr'][1:] - data_batch['ptr'][:-1]).max())
        data_batch = self.add_noise(data_batch)


        ## construct mapping from rdmol to selfies
        sf_max_len = selfie_batch.input_ids.shape[1]
        atom_max_len = int((data_batch['ptr'][1:] - data_batch['ptr'][:-1]).max())

        if self.load_mapping:
            padded_rdmol2selfies_mask = rdmol2selfies_mask[0].new_zeros((batch_size, atom_max_len))
            padded_rdmol2selfies = rdmol2selfies[0].new_zeros((batch_size, atom_max_len, sf_max_len))
            for i in range(batch_size):
                mask = rdmol2selfies_mask[i]
                padded_rdmol2selfies_mask[i, :mask.shape[0]].copy_(mask)
                mapping = rdmol2selfies[i]
                padded_rdmol2selfies[i, :mapping.shape[0], 1:1+mapping.shape[1]].copy_(mapping)

            data_batch['rdmol2selfies'] = padded_rdmol2selfies
            data_batch['rdmol2selfies_mask'] = padded_rdmol2selfies_mask

        if self.use_eigvec:
            data_batch['x'] = torch.cat([data_batch['x'], data_batch['EigVecs']], dim=1)
        data_batch.x = data_batch.x.to(torch.float)
        return data_batch, selfie_batch


class QM9InferCollater(object):
    def __init__(self, max_atoms: int, max_sf_tokens: int, selfies_tokenizer, noise_scheduler=None, use_eigvec=False, disable_com=False, load_mapping=True):
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer
        self.noise_scheduler = noise_scheduler
        self.use_eigvec = use_eigvec
        self.disable_com = disable_com
        self.load_mapping = load_mapping

    def __call__(self, data_list):
        ## selfies
        selfies = [data['selfies'] for data in data_list]

        self.selfies_tokenizer.padding_side = 'right'

        if self.selfies_tokenizer.use_adameta:
            prompts = []
            for i, p in enumerate(selfies):
                num_metaqueries = data_list[i].rdmol2selfies.shape[0]
                suffix = (
                    "\n<begin_of_mol>"
                    + "".join([f"<mol{j}>" for j in range(num_metaqueries)])
                    + "<end_of_mol>"
                )
                prompts.append(p + suffix)
        else:
            if self.selfies_tokenizer.num_metaqueries <= 0:
                suffix = ""
            else:
                suffix = (
                    "\n<begin_of_mol>"
                    + "".join([f"<mol{i}>" for i in range(self.selfies_tokenizer.num_metaqueries)])
                    + "<end_of_mol>"
                )

            prompts = [p + suffix for p in selfies]

        selfie_batch = self.selfies_tokenizer(prompts, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)

        ## construct mapping from rdmol to selfies
        batch_size = len(data_list)
        rdmol2selfies = [data.pop('rdmol2selfies') for data in data_list]
        rdmol2selfies_mask = [data.pop('rdmol2selfies_mask') for data in data_list]
        [data.pop('passed_conf_matching', None) for data in data_list]

        ## graph batch
        data_batch = Batch.from_data_list(data_list)
        data_batch['max_seqlen'] = int((data_batch['ptr'][1:] - data_batch['ptr'][:-1]).max())

        shape = (data_batch.x.shape[0], 3)
        if self.disable_com:
            data_batch['pos'] = torch.randn(shape)
        else:
            data_batch['pos'] = sample_com_rand_pos(shape, data_batch.batch)

        ## construct mapping from rdmol to selfies
        sf_max_len = selfie_batch.input_ids.shape[1]
        atom_max_len = int((data_batch['ptr'][1:] - data_batch['ptr'][:-1]).max())

        if self.load_mapping:
            padded_rdmol2selfies_mask = rdmol2selfies_mask[0].new_zeros((batch_size, atom_max_len))
            padded_rdmol2selfies = rdmol2selfies[0].new_zeros((batch_size, atom_max_len, sf_max_len))
            for i in range(batch_size):
                mask = rdmol2selfies_mask[i]
                padded_rdmol2selfies_mask[i, :mask.shape[0]].copy_(mask)
                mapping = rdmol2selfies[i]
                padded_rdmol2selfies[i, :mapping.shape[0], 1:1+mapping.shape[1]].copy_(mapping)

            data_batch['rdmol2selfies'] = padded_rdmol2selfies
            data_batch['rdmol2selfies_mask'] = padded_rdmol2selfies_mask

        if self.use_eigvec:
            data_batch['x'] = torch.cat([data_batch['x'], data_batch['EigVecs']], dim=1)
        return data_batch, selfie_batch



class QM9TorDFDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/tordf_qm9',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        load_test_only = False,
        args=None,
    ):
        super().__init__()
        root = Path(root)
        self.args = args
        self.root = root
        self.discrete_schedule = args.discrete_schedule
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer
        self.use_eigvec = args.use_eigvec
        self.infer_batch_size = args.infer_batch_size
        self.flatten_dataset = args.flatten_dataset
        self.disable_com = args.disable_com
        self.add_unseen_selfies_tokens(self.selfies_tokenizer, root)

        self.transform = None

        rand_smiles = args.rand_smiles
        addHs = args.addHs
        infer_noise = args.infer_noise

        self.prop_norms = None
        self.prop_dist = None
        self.nodes_dist = None


        if self.use_eigvec:
            if not load_test_only:
                self.train_dataset = QM9TorDF(f'{root}/processed_train_eig.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/tordf.train', self.transform, 'train', flatten_dataset=self.flatten_dataset)
                self.valid_dataset = QM9TorDF(f'{root}/processed_val_eig.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/tordf.val', self.transform, 'valid', flatten_dataset=self.flatten_dataset)
            self.test_dataset  = QM9TorDFInfer(f'{root}/processed_inference_test_eig.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/test_smiles.csv', f'{root}/test_mols.pkl', infer_noise)
        else:
            if not load_test_only:
                self.train_dataset = QM9TorDF(f'{root}/processed_train.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/tordf.train', self.transform, 'train', flatten_dataset=self.flatten_dataset)
                self.valid_dataset = QM9TorDF(f'{root}/processed_val.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/tordf.val', self.transform, 'valid', flatten_dataset=self.flatten_dataset)
            self.test_dataset  = QM9TorDFInfer(f'{root}/processed_inference_test.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/test_smiles.csv', f'{root}/test_mols.pkl', infer_noise)

        self.aug_rotation = not args.not_aug_rotation
        self.aug_translation = args.aug_translation
        self.t_cond = args.t_cond
        self.pos_std = self.test_dataset.pos_std

        self.max_atoms = 31
        if addHs:
            max_sf_tokens = 76
        else:
            max_sf_tokens = 28

        self.max_sf_tokens = max(max_sf_tokens + 2 + 10 + self.args.num_metaqueries + 2, 128)

        print('QM9 max num atoms', self.max_atoms)
        print('max selfies tokens', self.max_sf_tokens)

        noise_scheduler = args.noise_scheduler
        continuous_beta_0 = args.continuous_beta_0
        continuous_beta_1 = args.continuous_beta_1
        self.noise_scheduler = NoiseScheduleVPV2(noise_scheduler, continuous_beta_0=continuous_beta_0, continuous_beta_1=continuous_beta_1, discrete_mode=self.discrete_schedule)


    def add_unseen_selfies_tokens(self, tokenizer, root_path):
        with open(root_path / 'unseen_selfies_tokens.txt', 'r') as f:
            unseen_tokens = f.read().splitlines()
        vocab = tokenizer.get_vocab()
        for token in unseen_tokens:
            if token not in vocab:
                tokenizer.add_tokens(token)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=QM9Collater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.aug_rotation, self.t_cond, self.use_eigvec, self.disable_com, self.aug_translation),
        )
        return loader

    def val_dataloader(self):
        # Check if we're in evaluation-only mode
        if hasattr(self.args, 'mode') and self.args.mode in {'eval', 'eval_gen', 'eval_conf', 'eval_test_conform'}:
            # For evaluation modes, only return test dataloader to avoid unnecessary validation dataset loading
            print(f'Evaluation mode: {self.args.mode} - loading test dataset only')
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, use_eigvec=self.use_eigvec, disable_com=self.disable_com),
            )
            return test_loader
        else:
            # For training mode, return both validation and test dataloaders
            print('Training mode - loading validation and test datasets')
            val_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=QM9Collater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.aug_rotation, self.t_cond, self.use_eigvec, self.disable_com, self.aug_translation),
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, use_eigvec=self.use_eigvec, disable_com=self.disable_com),
            )
            return [val_loader, test_loader]


    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--infer_batch_size', type=int, default=64)
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--root', type=str, default='data/tordf_qm9')
        parser.add_argument('--addHs', action='store_true', default=False)
        parser.add_argument('--infer_time', type=float, default=0.9946)
        parser.add_argument('--infer_noise', type=float, default=0.9999)
        parser.add_argument('--use_eigvec', action='store_true', default=False)
        parser.add_argument('--t_cond', type=str, default='t')
        parser.add_argument('--discrete_schedule', action='store_true', default=False)
        parser.add_argument('--not_aug_rotation', action='store_true', default=False)
        parser.add_argument('--aug_translation', action='store_true', default=False)
        parser.add_argument('--flatten_dataset', action='store_true', default=False)
        return parent_parser


class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        assert self.drop_last
        assert shuffle
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = len(self.dataset) // self.num_replicas
        print('num samples', self.num_samples, self.num_replicas * self.num_samples)
        self.num_samples -= 96
        self.total_size = self.num_samples * self.num_replicas
        print('total size', self.total_size)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        assert self.shuffle
        assert self.drop_last

        # deterministically shuffle based on epoch and seed
        valid_ids = [i for i, data in enumerate(self.dataset._data_list) if data is not None]
        random.Random(self.seed + self.epoch).shuffle(valid_ids)
        assert len(valid_ids) >= self.num_samples
        valid_ids = valid_ids[:self.num_samples]
        return iter(valid_ids)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class GeomDrugsTorDFDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/tordf_drugs',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        load_test_only = False,
        args=None,
    ):
        super().__init__()
        root = Path(root)
        self.args = args
        self.root = root
        self.discrete_schedule = args.discrete_schedule

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer
        self.infer_batch_size = args.infer_batch_size
        self.add_unseen_selfies_tokens(self.selfies_tokenizer, root)

        self.transform = None
        self.prop_norms = None
        self.prop_dist = None

        rand_smiles = args.rand_smiles
        addHs = args.addHs

        # if dist.is_initialized():
        #     self.world_size = dist.get_world_size()
        #     self.global_rank = dist.get_global_rank()
        if _get_rank() is not None and args.world_size is not None:
            self.world_size = args.world_size
            self.global_rank = _get_rank()
        elif args.world_size is not None:
            self.world_size = args.world_size
            self.global_rank = 0
        else:
            raise ValueError('Please specify world size')

        print('world size', self.world_size, 'global rank', self.global_rank)
        # if not args.mode in {'eval', 'eval_gen', 'eval_conf', 'eval_test_conform'}:
        if not load_test_only:
            self.train_dataset = GeomDrugsTorDF(f'{root}/processed_train.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/tordf.train', self.transform, 'train', distributed_path=f'{root}/distributed_train', args=args)
        else:
            self.train_dataset = torch.utils.data.TensorDataset(torch.zeros(1))
        self.valid_dataset = GeomDrugsTorDF(f'{root}/processed_val.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/tordf.val', self.transform, 'valid', args=args)
        self.test_dataset = GeomDrugsTorDFInfer(f'{root}/processed_inference_test.pt', selfies_tokenizer, rand_smiles, addHs, f'{root}/test_smiles.csv', f'{root}/test_mols.pkl')

        self.aug_rotation = not args.not_aug_rotation
        self.t_cond = args.t_cond
        self.use_eigvec = args.use_eigvec
        self.disable_com = args.disable_com
        self.aug_translation = args.aug_translation
        self.pos_std = self.test_dataset.pos_std

        self.max_atoms = 178
        max_sf_tokens = 190
        self.max_sf_tokens = max_sf_tokens + 2 + 20 + self.args.num_metaqueries + 2


        print('Drugs max num atoms', self.max_atoms)
        print('max selfies tokens', self.max_sf_tokens)

        noise_scheduler = args.noise_scheduler
        continuous_beta_0 = args.continuous_beta_0
        continuous_beta_1 = args.continuous_beta_1
        self.noise_scheduler = NoiseScheduleVPV2(noise_scheduler, continuous_beta_0=continuous_beta_0, continuous_beta_1=continuous_beta_1, discrete_mode=self.discrete_schedule)

    def add_unseen_selfies_tokens(self, tokenizer, root_path):
        with open(root_path / 'unseen_selfies_tokens.txt', 'r') as f:
            unseen_tokens = f.read().splitlines()
        vocab = tokenizer.get_vocab()
        for token in unseen_tokens:
            if token not in vocab:
                tokenizer.add_tokens(token)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            sampler=CustomDistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, seed=0, drop_last=True),
            collate_fn=QM9Collater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.aug_rotation, self.t_cond, self.use_eigvec, self.disable_com, self.aug_translation),
        )
        return loader

    def val_dataloader(self):
        # Check if we're in evaluation-only mode
        if hasattr(self.args, 'mode') and self.args.mode in {'eval', 'eval_gen', 'eval_conf', 'eval_test_conform'}:
            # For evaluation modes, only return test dataloader to avoid unnecessary validation dataset loading
            print(f'Evaluation mode: {self.args.mode} - loading test dataset only')
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                sampler=DistributedSampler(self.test_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False, drop_last=False, seed=0),
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, use_eigvec=self.use_eigvec, disable_com=self.disable_com),
            )
            return test_loader
        else:
            # For training mode, return both validation and test dataloaders
            print('Training mode - loading validation and test datasets')
            val_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                sampler=DistributedSampler(self.valid_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False, drop_last=False, seed=0),
                collate_fn=QM9Collater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.aug_rotation, self.t_cond, self.use_eigvec, self.disable_com, self.aug_translation),
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                sampler=DistributedSampler(self.test_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False, drop_last=False, seed=0),
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, use_eigvec=self.use_eigvec, disable_com=self.disable_com),
            )
            return [val_loader, test_loader]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--infer_batch_size', type=int, default=64)
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--root', type=str, default='data/tordf_drugs')
        parser.add_argument('--addHs', action='store_true', default=False)
        parser.add_argument('--infer_time', type=float, default=0.9946)
        parser.add_argument('--infer_noise', type=float, default=0.9999)
        parser.add_argument('--t_cond', type=str, default='t')
        parser.add_argument('--discrete_schedule', action='store_true', default=False)
        parser.add_argument('--not_aug_rotation', action='store_true', default=False)
        parser.add_argument('--aug_translation', action='store_true', default=False)
        parser.add_argument('--flatten_dataset', action='store_true', default=False)
        return parent_parser

