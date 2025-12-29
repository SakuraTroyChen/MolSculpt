import os.path as osp
import copy
import numpy as np
from typing import Callable, Optional
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from mol_utils.featurization import featurize_mol
from torch_geometric.data.separate import separate
from rdkit import Chem
from torch_geometric.data.in_memory_dataset import nested_iter
import selfies as sf
from tqdm import tqdm
from data_provider.mol_mapping_utils import get_smiles2selfies_mapping, build_rdkit2rand_smiles_withoutH_mapping
from data_provider.conf_gen_cal_metrics import set_rdmol_positions





class MyGeomDrugDatasetJODO(Dataset):
    def __init__(self, root, selfies_tokenizer=None, rand_smiles=None, addHs=False):
        super().__init__()
        self.root = root
        self.selfies_tokenizer = selfies_tokenizer
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        if osp.exists(osp.join(root, self.processed_file_names[0])):
            self.data, self.slices = torch.load(osp.join(root, self.processed_file_names[0]))
        else:
            self.data, self.slices = self.my_process()
        # self.pos_std = 2.3860
        # self.max_num_atoms = 181
        # self.max_num_sf_tokens = 167
        self.pos_std = 4.4785
        self.max_num_atoms = max([mol.GetNumAtoms() for mol in self.data['rdmol'] if mol is not None])
        self.max_num_sf_tokens = max([len(list(sf.split_selfies(selfies))) for selfies in self.data['canonical_selfies'] if selfies is not None])

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return self.get_real_len()

    def len(self):
        return self.__len__()

    def get_real_len(self):
        if hasattr(self, 'length'):
            return self.length
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get_idx_data(self, idx: int):
        # TODO (matthias) Avoid unnecessary copy here.
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.get_real_len() * [None]
        elif self._data_list[idx] is not None:
            return self._data_list[idx].clone()

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = data.clone()
        return data.clone()

    def __getitem__(self, idx):
        ## obtain the data first
        idx = self.indices()[idx]
        data = self.get_idx_data(idx)
        data['rdmol'] = copy.deepcopy(data['rdmol'])
        assert data['rdmol'].GetNumConformers() == 1
        data['pos'] /= self.pos_std

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

    def add_unseen_selfies_tokens(self, tokenizer):
        with open(osp.join(self.root, self.processed_file_names[1]), 'r') as f:
            unseen_tokens = f.read().splitlines()
        vocab = tokenizer.get_vocab()
        for token in unseen_tokens:
            if token not in vocab:
                tokenizer.add_tokens(token)

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = osp.join(self.root, self.raw_file_names[1])
        print('Loading existing split data.')
        return torch.load(split_path)

    @property
    def raw_file_names(self):
        return ['data_geom_drug_1.pt', 'split_dict_geom_drug_1.pt']

    @property
    def processed_file_names(self):
        return ['processed_data.pt', 'unseen_sf_tokens.txt']

    def my_process(self):
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            raise ImportError("Please install 'rdkit' to alternatively process the raw data.")

        geom_drugs_dataset = _GeomDrugDatasetJODO(self.root, self.raw_file_names[0])
        data_list = []
        for data in tqdm(geom_drugs_dataset.data):
            rdmol = data['rdmol']
            data = featurize_mol(rdmol, types='drugs')
            pos = rdmol.GetConformers()[0].GetPositions()
            pos = pos - pos.mean(axis=0, keepdims=True)
            rdmol.RemoveAllConformers()
            rdmol = set_rdmol_positions(rdmol, pos, removeHs=False)
            assert rdmol.GetNumConformers() == 1
            data['rdmol'] = rdmol
            data['pos'] = torch.from_numpy(pos)
            canonical_smiles = Chem.MolToSmiles(rdmol)
            canonical_smiles = Chem.CanonSmiles(canonical_smiles)
            canonical_selfies = sf.encoder(canonical_smiles)
            data['canonical_selfies'] = canonical_selfies
            data['canonical_smiles'] = canonical_smiles
            if canonical_smiles.find('.') >= 0:
                ## if the smiles contains multiple molecules, then set the canonical smiles and selfies to empty and filter out the data
                data['canonical_smiles'] = ''
                data['canonical_selfies'] = ''
            data_list.append(data)

        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), osp.join(self.root, self.processed_file_names[0]))

        unseen_selfies_tokens = set()
        vocab = self.selfies_tokenizer.get_vocab()
        for data in data_list:
            selfies_tokens = sf.split_selfies(data.canonical_selfies)
            for t in selfies_tokens:
                if t not in vocab:
                    unseen_selfies_tokens.add(t)
        with open(osp.join(self.root, self.processed_file_names[1]), 'w') as f:
            f.write('\n'.join(unseen_selfies_tokens))
        return data, slices



class _GeomDrugDatasetJODO(Dataset):
    def __init__(self, root: str, data_file: str, transform: Optional[Callable] = None):
        super(_GeomDrugDatasetJODO, self).__init__()
        self.root = root
        self.data_file = data_file
        self.data = torch.load(osp.join(root, data_file))
        self.transform = transform

    def __getitem__(self, idx):
        data = copy.copy(self.data[self.indices()[idx]])
        data = data if self.transform is None else self.transform(data)
        data.idx = idx
        return data

    def len(self):
        return len(self.data)

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = osp.join(self.root, self.data_file.replace('data', 'split_dict'))
        if osp.exists(split_path):
            print('Loading existing split data.')
            return torch.load(split_path)

        data_num = len(self.indices())

        val_proportion = 0.1
        test_proportion = 0.1

        valid_index = int(val_proportion * data_num)
        test_index = valid_index + int(test_proportion * data_num)

        # Generate random permutation
        data_perm = np.random.permutation(data_num)

        valid, test, train = np.split(data_perm, [valid_index, test_index])

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits


if __name__ == '__main__':
    from transformers import AutoTokenizer
    root_path = 'data/archive/jodo_data/geom'
    dataset = MyGeomDrugDatasetJODO(root=root_path)
    tokenizer = AutoTokenizer.from_pretrained('all_checkpoints/mollama')
    canonical_selfies = dataset.data.canonical_selfies
    vocab = tokenizer.get_vocab()
    unseen_selfies_tokens = set()
    for ss in canonical_selfies:
        if ss:
            ss_list = list(sf.split_selfies(ss))
            for s in ss_list:
                if s not in vocab:
                    unseen_selfies_tokens.add(s)
    with open('unseen_sf_tokens.txt', 'w') as f:
        f.write('\n'.join(unseen_selfies_tokens))