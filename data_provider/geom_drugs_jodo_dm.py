import os
import lightning as L
import numpy as np
import torch
from rdkit import Chem
from data_provider.geom_drugs_jodo_dataset import MyGeomDrugDatasetJODO
# from data_provider.geom_dataset_tordf import Geom
from evaluation.eval_functions import get_moses_metrics, get_sub_geometry_metric
from data_provider.dataset_config import get_dataset_info
from torch.utils.data import DataLoader
from data_provider.diffusion_scheduler import NoiseScheduleVPV2
from data_provider.diffusion_data_module import QM9Collater, QM9InferCollater
from torch.utils.data import Dataset
from mol_utils.featurization import featurize_mol, qm9_types, drugs_types
from data_provider.mol_mapping_utils import build_rdkit2rand_smiles_withoutH_mapping, get_smiles2selfies_mapping


class PredictDataset(Dataset):
    def __init__(self, smiles_list, max_atoms, rand_smiles=False, dataset=None):
        self.smiles_list = smiles_list
        ## this is a pre-processing to remove the smiles that contains unknown atom types
        self.dataset = dataset
        if 'qm9' in self.dataset.lower():
            atom_types = qm9_types
            self.types = 'qm9'
        elif 'drugs' in self.dataset.lower():
            atom_types = drugs_types
            self.types = 'drugs'
        else:
            raise NotImplementedError(f'dataset {self.dataset} not implemented')

        processed_smiles_list = []
        dropped_smiles = []
        for smiles in self.smiles_list:
            rdmol = Chem.MolFromSmiles(smiles)
            exceptional_atom = None
            for atom in rdmol.GetAtoms():
                if atom.GetSymbol() not in atom_types:
                    exceptional_atom = atom.GetSymbol()
                    break
            if exceptional_atom is not None:
                dropped_smiles.append((smiles, exceptional_atom))
                continue
            processed_smiles_list.append(smiles)
        self.smiles_list = processed_smiles_list
        print(f"after filtering, {len(self.smiles_list)} smiles left. {len(dropped_smiles)} smiles dropped due to unknown atom type: ")
        for smiles, atom in dropped_smiles:
            print(smiles, atom)

        self.max_atoms = max_atoms
        self.rand_smiles = rand_smiles

        if 'qm9' in self.dataset.lower():
            self.pos_std = 1.7226
        elif 'drugs' in self.dataset.lower():
            self.pos_std = 2.3860
        else:
            raise NotImplementedError(f'dataset {self.dataset} not implemented')

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        ## transform the data into uni-mol version
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        data = featurize_mol(mol, types=self.types)
        data['rdmol'] = mol

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

        data['idx'] = idx
        return data


class LMCollater(object):
    def __init__(self, tokenizer, max_sf_tokens):
        self.tokenizer = tokenizer
        self.max_sf_tokens = max_sf_tokens

    def __call__(self, data_list):
        '''
        data_list: a list of data
        '''
        # if isinstance(data_list[0], list):
        #     data_list = [item for sublist in data_list for item in sublist]


        selfies = [data['selfies'] for data in data_list]
        self.tokenizer.padding_side = 'right'

        if self.tokenizer.use_adameta:
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
            if self.tokenizer.num_metaqueries <= 0:
                suffix = ""
            else:
                suffix = (
                    "\n<begin_of_mol>"
                    + "".join([f"<mol{i}>" for i in range(self.tokenizer.num_metaqueries)])
                    + "<end_of_mol>"
                )

            prompts = [p + suffix for p in selfies]


        selfie_batch = self.tokenizer(prompts, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)
        return selfie_batch

class GeomDrugsJODODFDM(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
    ):
        super().__init__()
        self.root = root
        self.args = args
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer

        rand_smiles = args.rand_smiles
        dataset = MyGeomDrugDatasetJODO(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=rand_smiles, addHs=args.addHs)
        self.max_atoms = dataset.max_num_atoms + 2
        print('GEOM Drugs max num atoms', self.max_atoms)

        ## obtain max selfies token length
        # +2 because of the bos and eos token; +20 to enlarge the space of molecule sampling
        self.max_sf_tokens = dataset.max_num_sf_tokens + 2 + 20 + self.args.num_metaqueries + 2
        print('max selfies tokens', self.max_sf_tokens)

        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones without selfies
        # selfies = np.array(dataset.data['canonical_selfies'])
        canonical_selfies_list = []
        for i in range(dataset.get_real_len()):
            data = dataset.get_idx_data(i)
            canonical_selfies_list.append(data['canonical_selfies'])
        
        # 转换为 numpy 数组
        selfies = np.array(canonical_selfies_list)

        # selfies_list = dataset._data['selfies']
        # selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]

        print('before filtering', len(train_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(test_idx))
        self.train_dataset = dataset.index_select(train_idx)
        self.test_dataset = dataset.index_select(test_idx)
        self.predict_dataset = None
        self.prop_norms = None
        self.prop_dist = None

        ## load rdmols of subsets
        rdmols = dataset.data.rdmol
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]
        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 5, cache_path=os.path.join(root, 'moses_stat.pkl'))
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('geom_with_h_1'), root)

        self.pos_std = dataset.pos_std
        noise_scheduler = args.noise_scheduler
        continuous_beta_0 = args.continuous_beta_0
        continuous_beta_1 = args.continuous_beta_1
        self.disable_com = args.disable_com
        self.aug_rotation = not args.not_aug_rotation
        self.t_cond = args.t_cond
        self.discrete_schedule = args.discrete_schedule
        self.infer_batch_size = args.infer_batch_size
        self.aug_translation = args.aug_translation

        self.noise_scheduler = NoiseScheduleVPV2(noise_scheduler, continuous_beta_0=continuous_beta_0, continuous_beta_1=continuous_beta_1, discrete_mode=self.discrete_schedule)


        ## load smiles for evaluation
        if args.eval_smiles_path is not None:
            with open(args.eval_smiles_path, 'r') as f:
                lines = f.readlines()
                sampled_sequences = [line.strip().split() for line in lines]
                _, _, smiles_smiles = zip(*sampled_sequences)
                self.predict_dataset = PredictDataset(smiles_smiles, self.max_atoms, rand_smiles, args.dataset)
        
    def setup_predict_dataset(self, smiles_list):
        if self.predict_dataset is not None:
            return

        print('Setup predict dataset...', end='')
        self.predict_dataset = PredictDataset(smiles_list, self.max_atoms, self.rand_smiles, self.args.dataset)
        print('done')

    def predict_dataloader(self):
        assert self.predict_dataset is not None
        loader = DataLoader(
                self.predict_dataset,
                batch_size=self.args.infer_batch_size,
                shuffle=False,
                num_workers=self.num_workers * 2,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=QM9InferCollater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, use_eigvec=False, disable_com=self.disable_com),
            )
        return loader
    
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=QM9Collater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.aug_rotation, self.t_cond, False, self.disable_com, self.aug_translation, load_mapping=False),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.infer_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=QM9Collater(self.max_atoms, self.max_sf_tokens, self.selfies_tokenizer, self.noise_scheduler, self.aug_rotation, self.t_cond, False, self.disable_com, self.aug_translation, load_mapping=False),
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
            return [loader, predict_loader]
        else:
            return loader

    # def predict_dataloader(self):
    #     assert self.predict_dataset is not None
    #     loader = DataLoader(
    #             self.predict_dataset,
    #             batch_size=16,
    #             shuffle=False,
    #             num_workers=self.num_workers * 2,
    #             pin_memory=False,
    #             drop_last=False,
    #             persistent_workers=False,
    #             collate_fn=QM9InferCollater(self.selfies_tokenizer, self.max_sf_tokens, load_mapping=False),
    #         )
    #     return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--root', type=str, default='data/drugs')
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--addHs', action='store_true', default=False)
        parser.add_argument('--not_aug_rotation', action='store_true', default=False)
        # parser.add_argument('--noise_scheduler', type=str, default='cosine')
        parser.add_argument('--t_cond', type=str, default='t')
        parser.add_argument('--discrete_schedule', action='store_true', default=False)
        parser.add_argument('--infer_batch_size', type=int, default=128)
        parser.add_argument('--infer_time', type=float, default=0.9946)
        parser.add_argument('--infer_noise', type=float, default=0.9999)
        parser.add_argument('--aug_translation', action='store_true', default=False)
        parser.add_argument('--eval_smiles_path', type=str, default=None)
        return parent_parser


class GeomDrugsJODODM(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.root = root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer

        rand_smiles = args.rand_smiles
        dataset = MyGeomDrugDatasetJODO(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=rand_smiles, addHs=args.addHs)
        dataset.add_unseen_selfies_tokens(selfies_tokenizer)
        self.max_atoms = dataset.max_num_atoms + 2
        print('GEOM Drugs max num atoms', self.max_atoms)

        ## obtain max selfies token length
        # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling
        self.max_sf_tokens = dataset.max_num_sf_tokens + 2 + 10 + self.args.num_metaqueries + 2 
        print('max selfies tokens', self.max_sf_tokens)

        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones without selfies
        # selfies = np.array(dataset.data['canonical_selfies'])
        canonical_selfies_list = []
        for i in range(dataset.get_real_len()):
            data = dataset.get_idx_data(i)
            canonical_selfies_list.append(data['canonical_selfies'])
        
        # 转换为 numpy 数组
        selfies = np.array(canonical_selfies_list)

        print('before filtering', len(train_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(test_idx))
        self.train_dataset = dataset.index_select(train_idx)
        self.test_dataset = dataset.index_select(test_idx)
        self.predict_dataset = None
        self.prop_norms = None
        self.prop_dist = None

        ## load rdmols of subsets
        rdmols = dataset.data.rdmol
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]
        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 5, cache_path=os.path.join(root, 'moses_stat.pkl'))
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('geom_with_h_1'), root)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=LMCollater(self.selfies_tokenizer, self.max_sf_tokens),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size // 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=LMCollater(self.selfies_tokenizer, self.max_sf_tokens),
        )
        return loader

    def predict_dataloader(self):
        assert self.predict_dataset is not None
        loader = DataLoader(
                self.predict_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=self.num_workers * 2,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=LMCollater(self.selfies_tokenizer, self.max_sf_tokens),
            )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--root', type=str, default='data/drugs')
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--addHs', action='store_true', default=False)
        parser.add_argument('--not_aug_rotation', action='store_true', default=False)
        return parent_parser

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = GeomDrugsJODODM.add_model_specific_args(parser)
    args = parser.parse_args()
    dm = GeomDrugsJODODM(root = 'data/archive/jodo_data/geom', num_workers=2, batch_size=2, args=args)
    print(dm)