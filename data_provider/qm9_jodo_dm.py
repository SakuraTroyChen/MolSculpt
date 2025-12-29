import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import lightning as L
from torch_geometric.data import Batch
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
# from data_provider.qm9_dataset_v6 import QM9Dataset
from data_provider.qm9_dataset_v5 import QM9Dataset
from data_provider.property_distribution import DistributionProperty
from data_provider.conf_gen_cal_metrics import set_rdmol_positions
from pathlib import Path
from data_provider.diffusion_scheduler import NoiseScheduleVPV2
import selfies as sf
from torch_scatter import scatter
from scipy.spatial.transform import Rotation
from data_provider.dataset_config import get_dataset_info
from evaluation.eval_functions import get_moses_metrics
from evaluation.eval_functions import get_sub_geometry_metric
from rdkit import Chem
import copy
from data_provider.mol_mapping_utils import get_smiles2selfies_mapping, build_rdkit2rand_smiles_withoutH_mapping
from mol_utils.featurization import featurize_mol, featurize_mol_from_smiles_v2, qm9_types, drugs_types
from data_provider.geom_drugs_jodo_dm import PredictDataset


bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge
atom_name = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
atomic_number_list = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]


def sample_com_rand_pos(pos_shape, batch):
    noise = torch.randn(pos_shape, device=batch.device) # shape = [\sum N_i, 3]
    mean_noise = scatter(noise, batch, dim=0, reduce='mean') # shape = [B, 3]
    noise = noise - mean_noise[batch]
    assert len(noise.shape) == 2 and noise.shape[1] == 3
    return noise

class QM9Collater(object):
    def __init__(self, max_atoms: int, max_sf_tokens: int, selfies_tokenizer, noise_scheduler, aug_rotation=False, t_cond='t', use_eigvec=False, disable_com=False, aug_translation=False, condition=False, prop_norm=None, mode='train'):
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer
        self.noise_scheduler = noise_scheduler
        self.aug_rotation = aug_rotation
        self.t_cond = t_cond
        self.use_eigvec = use_eigvec
        self.disable_com = disable_com
        self.aug_translation = aug_translation
        self.condition = condition
        self.prop_norm = prop_norm
        self.mode = mode

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


        dtype = data.pos.dtype
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
            noise = sample_com_rand_pos(data.pos.shape, data.batch)
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
        if self.mode == 'infer':
            data['pos'] = torch.randn_like(data['pos'])
        return data

    def __call__(self, data_list):
        ## selfies
        selfies = [data['selfies'] for data in data_list]
        self.selfies_tokenizer.padding_side = 'right'

        # num_atom: data_list[0].num_atom.item()

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

        if self.condition:
            context = data_batch.property
            assert len(self.prop_norm) == 1 # TODO only single property here
            # for i, key in enumerate(self.prop_norm.keys()):
            #     context[:, i] = (context[:, i] - self.prop_norm[key]['mean']) / self.prop_norm[key]['mad']
            condition_property = list(self.prop_norm.keys())[0]
            context = (context - self.prop_norm[condition_property]['mean']) / self.prop_norm[condition_property]['mad']
            data_batch['context'] = context
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



class tordf_version(Dataset):
    def __init__(self, dataset, max_atoms, rand_smiles=False, addHs=False, transform=None, mode='train', infer_time=0.001):
        self.dataset = dataset

        self.max_atoms = max_atoms
        self.remove_hydrogen = False
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.beta = 4.0
        self.smooth = 0.1
        self.topN = 10
        # self.retain_sf_hs = retain_sf_hs
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        self.transform = transform
        self.mode = mode
        self.pos_std = 1.7226

        if self.mode == 'infer':
            assert 1 >= infer_time >= 0
            self.infer_time = infer_time

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if self.mode != 'infer':
            data = self.dataset[idx].clone()
            molecule = data['rdmol']
            Chem.SanitizeMol(molecule)
            _data = featurize_mol(molecule, types='qm9')
            data['x'] = _data['x']
            data['z'] = _data['x']
            data['edge_index'] = _data['edge_index']
            data['edge_attr'] = _data['edge_attr']
            canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            data['canonical_selfies'] = sf.encoder(canonical_smiles)

            data['pos'] -= data['pos'].mean(dim=0, keepdim=True)
            rdmol = copy.deepcopy(data['rdmol'])
            rdmol.RemoveAllConformers()
            data['rdmol'] = set_rdmol_positions(rdmol, data['pos'], removeHs=False, )
            assert data['rdmol'].GetNumConformers() == 1

            data['pos'] /= self.pos_std
            if self.transform:
                data = self.transform(data)
        else:
            # mol_idx, seed_pos = self.seed_pos_list[idx]
            # data = self.dataset[idx]
            # data['seed_pos_idx'] = idx
            # data['mol_idx'] = mol_idx
            # data['pos'] = seed_pos / self.pos_std

            # ## setup perturb seed
            # ## fake randomness; this is to generate reproducible test results
            # data['t'] = self.infer_time

            # data['rdmol'].RemoveAllConformers()
            # # data['rdmol'] = set_rdmol_positions(rdmol, data['pos'], removeHs=False, )
            # # assert data['rdmol'].GetNumConformers() == 1

            data = self.dataset[idx].clone()
            data['mol_idx'] = idx
            molecule = data['rdmol']
            Chem.SanitizeMol(molecule)
            _data = featurize_mol(molecule, types='qm9')
            data['x'] = _data['x']
            data['z'] = _data['x']
            data['edge_index'] = _data['edge_index']
            data['edge_attr'] = _data['edge_attr']
            canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            data['canonical_selfies'] = sf.encoder(canonical_smiles)

            data['pos'] -= data['pos'].mean(dim=0, keepdim=True)
            rdmol = copy.deepcopy(data['rdmol'])
            rdmol.RemoveAllConformers()
            data['rdmol'] = set_rdmol_positions(rdmol, data['pos'], removeHs=False, )
            assert data['rdmol'].GetNumConformers() == 1

            data['pos'] /= self.pos_std
            if self.transform:
                data = self.transform(data)

        rdmol2smiles, output_smiles = build_rdkit2rand_smiles_withoutH_mapping(data.rdmol, self.rand_smiles)
        rdmol2smiles = rdmol2smiles.tolist()
        smiles2selfies, selfies_tokens, selfies = get_smiles2selfies_mapping(output_smiles) # smiles2selfies is a dict
        ## update the data object with new information
        data['smiles'] = output_smiles
        data['selfies'] = selfies
        data['rdmol2smiles'] = rdmol2smiles

        ## add the mapping from rdmol to selfies
        rdmol2selfies = torch.zeros((data.rdmol.GetNumAtoms(), len(selfies_tokens)), dtype=torch.float) # shape = [num_atoms, num_selfies_tokens]
        rdmol2selfies_mask = torch.zeros((data.rdmol.GetNumAtoms(),), dtype=torch.bool)
        for i, v in enumerate(rdmol2smiles):
            if v in smiles2selfies:
                for j in smiles2selfies[v]:
                    rdmol2selfies[i, j] = 1
                rdmol2selfies_mask[i] = True
        data['rdmol2selfies'] = rdmol2selfies
        data['rdmol2selfies_mask'] = rdmol2selfies_mask
        return data




class QM9DM(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/qm9v6_old',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
    ):
        super().__init__()
        root = Path(root)
        self.args = args
        self.root = root
        self.discrete_schedule = args.discrete_schedule if hasattr(args, 'discrete_schedule') else False
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer
        self.use_eigvec = args.use_eigvec if hasattr(args, 'use_eigvec') else False
        self.infer_batch_size = args.infer_batch_size if hasattr(args, 'infer_batch_size') else 64
        self.flatten_dataset = args.flatten_dataset if hasattr(args, 'flatten_dataset') else False
        self.disable_com = args.disable_com if hasattr(args, 'disable_com') else False
        self.add_unseen_selfies_tokens(self.selfies_tokenizer, root / 'processed')

        if not hasattr(args, 'condition_property') or args.condition_property == None:
            self.transform = None
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            dataset_info = get_dataset_info('qm9_second_half')
            prop2idx = dataset_info['prop2idx']
            include_aromatic = False
            self.transform = EdgeComCondTransform(dataset_info['atom_encoder'].values(), include_aromatic, prop2idx[args.condition_property])
        else:
            raise NotImplementedError(f"{args.conditon} is not supported")

        self.rand_smiles = args.rand_smiles if hasattr(args, 'rand_smiles') else 'restricted'
        self.addHs = args.addHs if hasattr(args, 'addHs') else False
        self.infer_noise = args.infer_noise if hasattr(args, 'infer_noise') else 0.9999

        assert not self.use_eigvec, 'old version of QM9 dataset does not have eigenvectors'

        dataset = QM9Dataset(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=self.rand_smiles)
        self.dataset = dataset
        max_atoms = int(dataset._data.num_atom.max()) + 2 # +2 because of the bos and eos token;
        self.max_atoms = max_atoms
        print('QM9 max num atoms', max_atoms)

        ## obtain max selfies token length
        selfies_list = dataset._data['selfies']
        selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]
        # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling; + num_metaqueries + 2 because of the metaqueries, <begin_of_mol> and <end_of_mol>
        self.max_sf_tokens = max(max(selfies_lens) + 2 + 10 + self.args.num_metaqueries + 2, 128)
        print('max selfies tokens', self.max_sf_tokens)
        print('num metaqueries', self.args.num_metaqueries)
        print('max selfies_lens', max(selfies_lens))

        # self.max_atoms = 31
        # self.max_sf_tokens = 28


        # print('QM9 max num atoms', self.max_atoms)
        # print('max selfies tokens', self.max_sf_tokens)

        if args.condition_property == None:
            splits = dataset.get_idx_split()
            train_idx = splits['train']
            valid_idx = splits['valid']
            test_idx = splits['test']
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            splits = dataset.get_cond_idx_split()
            first_train_idx = splits['first_train']
            second_train_idx = splits['second_train']
            valid_idx = splits['valid']
            test_idx = splits['test']

            train_idx = second_train_idx

        ## filter the ones without selfies
        selfies = np.array(dataset._data.selfies)

        print('before filtering', len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[train_idx < len(dataset)]
        valid_idx = valid_idx[valid_idx < len(dataset)]
        test_idx = test_idx[test_idx < len(dataset)]
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(valid_idx), len(test_idx))
        self.train_dataset = tordf_version(dataset.index_select(train_idx), max_atoms, self.rand_smiles, self.addHs, self.transform, 'train')
        self.valid_dataset = tordf_version(dataset.index_select(valid_idx), max_atoms, self.rand_smiles, self.addHs, self.transform, 'valid')
        self.test_dataset = tordf_version(dataset.index_select(test_idx), max_atoms, self.rand_smiles, self.addHs, self.transform, 'infer', self.infer_noise)
        self.predict_dataset = None

        if args.condition_property == None:
            pass
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            prop2idx_sub = {
                args.condition_property: prop2idx[args.condition_property]
            }
            self.prop_norms = dataset.index_select(valid_idx).compute_property_mean_mad(prop2idx_sub)
            self.prop_dist = DistributionProperty(dataset.index_select(train_idx), prop2idx_sub, normalizer=self.prop_norms)
        else:
            raise NotImplementedError(f"{args.conditon} is not supported")

        ## load rdmols of subsets
        rdmols = dataset._data.rdmol
        train_idx = train_idx.tolist()
        valid_idx = valid_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]

        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 32)
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('qm9_with_h'), os.path.join(root, 'processed'))

        self.aug_rotation = (not args.not_aug_rotation) if hasattr(args, 'not_aug_rotation') else True
        self.aug_translation = args.aug_translation if hasattr(args, 'aug_translation') else False
        self.t_cond = args.t_cond if hasattr(args, 't_cond') else 't'
        # self.pos_std = self.test_dataset.pos_std
        self.pos_std = 1.7226

        noise_scheduler = args.noise_scheduler if hasattr(args, 'noise_scheduler') else 'cosine'
        continuous_beta_0 = args.continuous_beta_0 if hasattr(args, 'continuous_beta_0') else 0.1
        continuous_beta_1 = args.continuous_beta_1 if hasattr(args, 'continuous_beta_1') else 20
        self.noise_scheduler = NoiseScheduleVPV2(noise_scheduler, continuous_beta_0=continuous_beta_0, continuous_beta_1=continuous_beta_1, discrete_mode=self.discrete_schedule)

        if args.eval_smiles_path is not None:
            with open(args.eval_smiles_path, 'r') as f:
                lines = f.readlines()
                sampled_sequences = [line.strip().split() for line in lines]
                _, _, smiles_smiles = zip(*sampled_sequences)
                self.predict_dataset = PredictDataset(smiles_smiles, self.max_atoms, self.rand_smiles, args.dataset)


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
            collate_fn=QM9Collater(
                max_atoms=self.max_atoms,
                max_sf_tokens=self.max_sf_tokens,
                selfies_tokenizer=self.selfies_tokenizer,
                noise_scheduler=self.noise_scheduler,
                aug_rotation=self.aug_rotation,
                t_cond=self.t_cond,
                use_eigvec=self.use_eigvec,
                disable_com=self.disable_com,
                aug_translation=self.aug_translation,
                condition=(self.transform is not None),
                prop_norm=self.prop_norms if self.transform is not None else None,
                mode='train',
            )
        )
        return loader

    def val_dataloader(self):
        print('validating')
        val_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=QM9Collater(
                max_atoms=self.max_atoms,
                max_sf_tokens=self.max_sf_tokens,
                selfies_tokenizer=self.selfies_tokenizer,
                noise_scheduler=self.noise_scheduler,
                aug_rotation=self.aug_rotation,
                t_cond=self.t_cond,
                use_eigvec=self.use_eigvec,
                disable_com=self.disable_com,
                aug_translation=self.aug_translation,
                condition=(self.transform is not None),
                prop_norm=self.prop_norms if self.transform is not None else None,
                mode='infer',
            )
        )
        if hasattr(self, 'predict_dataset'):
            predict_loader = DataLoader(
                self.predict_dataset,
                batch_size=self.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, load_mapping=True)
                ),
            return [val_loader, predict_loader]
        else:
            return val_loader

    # def setup_predict_dataset(self, smiles_list):
    #     if self.predict_dataset is not None:
    #         return

    #     print('Setup predict dataset...', end='')
    #     self.predict_dataset = predict_dataset(smiles_list, self.max_atoms, self.rand_smiles, self.args.dataset)
    #     print('done')

    # def predict_dataloader(self):
    #     assert self.predict_dataset is not None
    #     loader = DataLoader(
    #             self.predict_dataset,
    #             batch_size=self.args.infer_batch_size,
    #             shuffle=False,
    #             num_workers=self.num_workers * 2,
    #             pin_memory=False,
    #             drop_last=False,
    #             persistent_workers=False,
    #             collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.use_eigvec, self.disable_com),
    #         )
    #     return loader

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
        parser.add_argument('--condition_property', type=str, default=None)
        parser.add_argument('--eval_smiles_path', type=str, default=None)
        return parent_parser

class EdgeComCondTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx = property_idx

    def __call__(self, data: Data):
        # # add atom type one_hot
        # atom_type = data.atom_type
        # edge_type = data.edge_type

        # atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        # data.atom_one_hot = atom_one_hot.float()

        # # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        # edge_bond = edge_type.clone()
        # edge_bond[edge_bond == 4] = 0
        # edge_bond = edge_bond / 3.
        # edge_feat = [edge_bond]
        # if self.include_aromatic:
        #     edge_aromatic = (edge_type == 4).float()
        #     edge_feat.append(edge_aromatic)
        # edge_feat = torch.stack(edge_feat, dim=-1)

        # edge_index = data.edge_index
        # dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        # dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        # idx1, idx2 = edge_index[0], edge_index[1]
        # idx = idx1 * data.num_nodes + idx2
        # idx = idx.unsqueeze(-1).expand(edge_feat.size())
        # dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        # dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        # edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        # dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        # data.edge_one_hot = dense_edge_one_hot

        properties = data.y
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

            data.property = properties[0, self.property_idx:self.property_idx+1] - \
                            torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
            property = properties[0, self.property_idx:self.property_idx+1]
            data.property = property

        return data
