import os.path as osp
import copy
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import selfies as sf
from data_provider.mol_mapping_utils import build_rdkit2cano_smiles_withoutH_mapping, get_smiles2selfies_mapping
from data_provider.conf_gen_cal_metrics import generate_conformers
from data_provider.conf_gen_cal_metrics import get_best_rmsd, set_rdmol_positions

class GeomDrugs(InMemoryDataset):
    def __init__(self, root='./data/drugs', selfies_tokenizer: Optional[List[str]] = None,transform=None, pre_transform=None, pre_filter=None):
        """
        ## Args
            - `root` (`str`): root directory where the dataset should be saved
            - `selfies_tokenizer` (`optional`): Optional tokenizer to be applied on a sample.
            - `transform` (`callable`, `optional`): Optional transform to be applied on a sample.
            - `pre_transform` (`callable`, `optional`): Optional transform to be applied on the whole dataset.
            - `pre_filter` (`callable`, `optional`): Optional filter to be applied on the whole dataset.
        """
        self.root = root
        self.selfies_tokenizer = selfies_tokenizer
        self.transform = transform

        super(GeomDrugs, self).__init__(self.root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.add_unseen_selfies_tokens(self.selfies_tokenizer)

    def add_unseen_selfies_tokens(self, tokenizer):
        with open(self.processed_paths[1], 'r') as f:
            unseen_tokens = f.read().splitlines()
        vocab = tokenizer.get_vocab()
        for token in unseen_tokens:
            if token not in vocab:
                tokenizer.add_tokens(token)

    @property
    def raw_file_names(self):
        return ['geom_drugs_1.npy', 'geom_drugs_n_1.npy', 'geom_drugs_smiles.txt']

    @property
    def processed_file_names(self):
        return ['Geom_pyg_1.pt', 'unseen_selfies_tokens.txt']

    def process(self):
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            raise ImportError('`GeomDrugs Dataset` requires `rdkit`.')


        raw_dir = self.root
        data = np.load(osp.join(raw_dir, self.raw_file_names[0]))
        num_index = np.load(osp.join(raw_dir, self.raw_file_names[1]))
        smiles = np.loadtxt(osp.join(raw_dir, self.raw_file_names[2]), dtype=str, comments=None, delimiter=None)

        data_list = []
        sum_nodes = 0

        R_list = data[:, 2:]
        z_list = data[:, 1]

        bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge
        atom_name = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
        atomic_number_list = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
        charge_dict = dict(zip(atom_name, atomic_number_list))

        for i in tqdm(range(len(num_index))):
            R_i = torch.tensor(R_list[sum_nodes:sum_nodes + num_index[i], :], dtype=torch.float32)
            z_i = torch.tensor(z_list[sum_nodes:sum_nodes + num_index[i]], dtype=int)

            sum_nodes += num_index[i]

            molecule = Chem.MolFromSmiles(smiles[i])
            molecule = Chem.AddHs(molecule)

            charges = [charge_dict[atom.GetSymbol()] for atom in molecule.GetAtoms()]
            formal_charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]

            row, col, edge_type = [], [], []
            for bond in molecule.GetBonds():
                # if bond.GetBondType() == BT.AROMATIC:
                #     print('meet aromatic bond!')

                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)

            try:
                rdmol2smiles, cano_smiles_woh = build_rdkit2cano_smiles_withoutH_mapping(molecule)
                selfies = sf.encoder(cano_smiles_woh)
                rdkit_cluster_coords_list = generate_conformers(molecule, M=10, removeHs=False, addHs=False)
            except Chem.rdchem.AtomValenceException:
                rdmol2smiles, cano_smiles_woh, selfies = np.asarray([]), '', ''
                rdkit_cluster_coords_list = []

            ## compute the rmsd for rdkit_cluster_coords_list
            if len(rdkit_cluster_coords_list) > 0:
                new_rdkit_cluster_coords_list = []
                for coord in rdkit_cluster_coords_list:
                    new_rdmol = set_rdmol_positions(molecule, coord, False)
                    rmsd = get_best_rmsd(new_rdmol, molecule)
                    new_rdkit_cluster_coords_list.append((coord, rmsd))
                new_rdkit_cluster_coords_list.sort(key=lambda x: x[1])
                rdkit_cluster_coords_list = new_rdkit_cluster_coords_list

            data = Data(atom_type=z_i, pos=R_i, charge=torch.tensor(charges), fc=torch.tensor(formal_charges), edge_index=edge_index, edge_type=edge_type, y=torch.tensor(i), num_atom=molecule.GetNumAtoms(), idx=i, rdmol=copy.deepcopy(molecule), selfies=selfies, rdmol2smiles=rdmol2smiles, cano_smiles_woh=cano_smiles_woh, rdkit_cluster_coords_list=rdkit_cluster_coords_list)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])

        ## check and save the unseen selfies tokens
        unseen_selfies_tokens = set()
        vocab = self.selfies_tokenizer.get_vocab()
        for data in data_list:
            selfies_tokens = sf.split_selfies(data.selfies)
            for t in selfies_tokens:
                if t not in vocab:
                    unseen_selfies_tokens.add(t)
        unseen_selfies_tokens = list(unseen_selfies_tokens)
        unseen_selfies_tokens.sort()
        with open(self.processed_paths[1], 'w') as f:
            for t in unseen_selfies_tokens:
                f.write(t + '\n')

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = osp.join(self.processed_dir, 'split_dict_drugs.pt')
        if osp.exists(split_path):
            print('Loading existing split data.')
            return torch.load(split_path)

        data_num = len(self.indices())
        assert data_num == 292035, print(f"Data num is {data_num}")

        val_proportion = 0.1
        test_proportion = 0.1

        valid_index = int(val_proportion * data_num)
        test_index = valid_index + int(test_proportion * data_num)

        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(data_num)

        valid, test, train = np.split(data_perm, [valid_index, test_index])

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits

class GeomDrugsDataset(GeomDrugs):
    def process(self):
        super(GeomDrugsDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)
        smiles2selfies, selfies_tokens = get_smiles2selfies_mapping(data.cano_smiles_woh) # smiles2selfies is a dict

        rdmol2smiles = data.rdmol2smiles.tolist()

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

if __name__ == '__main__':
    from transformers import AutoTokenizer
    root_path = "./data/drugs"
    tokenizer = AutoTokenizer.from_pretrained('all_checkpoints/mollama')
    dataset = GeomDrugsDataset(root=root_path, selfies_tokenizer=tokenizer)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    ## filter the ones with no selfies
    print("Filtering the ones with no selfies")
    selfies = np.array(dataset.data.selfies)
    print(f"Befor filtering: {len(train_idx)}, {len(valid_idx)}, {len(test_idx)}")
    train_idx = train_idx[selfies[train_idx] != np.array('')]
    valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
    test_idx = test_idx[selfies[test_idx] != np.array('')]
    print(f"After filtering: {len(train_idx)}, {len(valid_idx)}, {len(test_idx)}")

    train_dataset = dataset.index_select(train_idx)
    val_dataset = dataset.index_select(valid_idx)
    test_dataset = dataset.index_select(test_idx)

    for i in range(len(train_dataset)):
        assert train_dataset[i].selfies != '', print([train_dataset[i].selfies])
    for i in range(len(val_dataset)):
        assert val_dataset[i].selfies != '', print([val_dataset[i].selfies])
    for i in range(len(test_dataset)):
        assert test_dataset[i].selfies != '', print([test_dataset[i].selfies])

    # save split idx
    np.save('train_idx.npy', train_idx)
    np.save('valid_idx.npy', valid_idx)
    np.save('test_idx.npy', test_idx)
