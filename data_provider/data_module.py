import torch
import os
import lightning as L
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.data import Batch
from torch_geometric.data import Data
# from data_provider.qm9_dataset_v5 import QM9Dataset, QM9LMDataset
from data_provider.qm9_dataset_v6 import QM9Dataset, QM9LMDataset
from data_provider.geom_dataset_v2 import GeomDrugsDataset, GeomDrugsLMDataset
from data_provider.conf_gen_cal_metrics import set_rdmol_positions, get_best_rmsd, generate_conformers
from rdkit import Chem
import selfies as sf
from data_provider.mol_mapping_utils import build_rdkit2cano_smiles_withoutH_mapping, get_smiles2selfies_mapping, build_rdkit2rand_smiles_withoutH_mapping
from data_provider.dataset_config import get_dataset_info
from evaluation.eval_functions import get_moses_metrics
from evaluation.eval_functions import get_sub_geometry_metric
from rdkit.Chem.rdchem import BondType as BT
from data_provider.property_distribution import DistributionPropertyV2 as DistributionProperty
from data_provider.node_distribution import get_node_dist
from data_provider.diffusion_data_module_v2 import EdgeComCondTransform

def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge

class D3Collater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, atom_vec, edge_type, bond_type, coordinates, dist, tgt_coordinates=None, tgt_dist=None):
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_bond_type = data_utils.collate_tokens_2d(bond_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]

        padded_coordinates = collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]

        if tgt_coordinates is None or tgt_dist is None:
            return padded_atom_vec, padded_edge_type, padded_bond_type, padded_coordinates, padded_dist
        else:
            padded_tgt_coordinates = collate_tokens_coords(tgt_coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
            padded_tgt_dist = data_utils.collate_tokens_2d(tgt_dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
            return padded_atom_vec, padded_edge_type, padded_bond_type, padded_coordinates, padded_dist, padded_tgt_coordinates, padded_tgt_dist


class InferenceCollater(object):
    def __init__(self, tokenizer, max_atoms, max_sf_tokens, d3_pad_id):
        self.d3_collater = D3Collater(d3_pad_id)
        self.tokenizer = tokenizer
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens

    def __call__(self, data_list):
        '''
        data_list: a list of data
        '''
        rdmols = [data['rdmol'] for data in data_list]
        smiles = [data['smiles'] for data in data_list]
        selfies = [data['selfies'] for data in data_list]
        atom_vec = [data['atom_vec'] for data in data_list]
        edge_type = [data['edge_type'] for data in data_list]
        bond_type = [data['bond_type'] for data in data_list]
        coordinates = [data['coordinates'] for data in data_list]
        dist = [data['dist'] for data in data_list]
        rdmol2selfies = [data['rdmol2selfies'] for data in data_list]
        rdmol2selfies_mask = [data['rdmol2selfies_mask'] for data in data_list]

        padded_atom_vec, padded_edge_type, padded_bond_type, padded_coordinates, padded_dist = self.d3_collater(atom_vec, edge_type, bond_type, coordinates, dist)

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

        self.tokenizer.padding_side = 'right'

        selfie_batch = self.tokenizer(prompts, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)

        if True:
            ## sanity check
            selfie_length = selfie_batch.attention_mask.sum(dim=1)
            for i in range(len(rdmol2selfies)):
                mol_num = len(atom_vec[i])
                selfie_num = int(selfie_length[i])
                if mol_num != rdmol2selfies[i].shape[0] + 2:
                    print('------------------')
                    print(mol_num, rdmol2selfies[i].shape)
                    print(selfies[i], self.max_sf_tokens)
                    exit()
                if selfie_num != rdmol2selfies[i].shape[1] + 2:
                    print('------------------')
                    print(selfie_num, rdmol2selfies[i].shape)
                    print(selfies[i], self.max_sf_tokens)
                    exit()
            ## end sanity check

        batch_size, d_atom = padded_atom_vec.shape[:2]
        d_selfies = selfie_batch.input_ids.shape[1]

        padded_rdmol2selfies_mask = rdmol2selfies_mask[0].new_zeros((batch_size, d_atom))
        padded_rdmol2selfies = rdmol2selfies[0].new_zeros((batch_size, d_atom, d_selfies))
        for i in range(batch_size):
            mask = rdmol2selfies_mask[i]
            padded_rdmol2selfies_mask[i, 1:1+mask.shape[0]].copy_(mask) # +1 because of the bos token in uni-mol atoms
            mapping = rdmol2selfies[i]
            padded_rdmol2selfies[i, 1:1+mapping.shape[0], 1:1+mapping.shape[1]].copy_(mapping) # +1 because of the bos token in uni-mol atoms

        data = Data(atom_vec=padded_atom_vec, edge_type=padded_edge_type, bond_type=padded_bond_type, coordinates=padded_coordinates, dist=padded_dist, rdmol2selfies=padded_rdmol2selfies, rdmol2selfies_mask=padded_rdmol2selfies_mask, selfies=selfies, smiles=smiles, rdmols=rdmols)
        return data, selfie_batch


class TrainCollater(object):
    def __init__(self, tokenizer, max_atoms, max_sf_tokens, d3_pad_id):
        self.d3_collater = D3Collater(d3_pad_id)
        self.tokenizer = tokenizer
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens

    def __call__(self, data_list):
        '''
        data_list: a list of data
        '''
        if isinstance(data_list[0], list):
            data_list = [item for sublist in data_list for item in sublist]

        rdmols = [data['rdmol'] for data in data_list]
        smiles = [data['smiles'] for data in data_list]
        selfies = [data['selfies'] for data in data_list]
        atom_vec = [data['atom_vec'] for data in data_list]
        edge_type = [data['edge_type'] for data in data_list]
        bond_type = [data['bond_type'] for data in data_list]
        coordinates = [data['coordinates'] for data in data_list]
        dist = [data['dist'] for data in data_list]
        tgt_coordinates = [data['tgt_coordinates'] for data in data_list]
        tgt_dist = [data['tgt_dist'] for data in data_list]
        rdmol2selfies = [data['rdmol2selfies'] for data in data_list]
        rdmol2selfies_mask = [data['rdmol2selfies_mask'] for data in data_list]

        padded_atom_vec, padded_edge_type, padded_bond_type, padded_coordinates, padded_dist, padded_tgt_coordinates, padded_tgt_dist = self.d3_collater(atom_vec, edge_type, bond_type, coordinates, dist, tgt_coordinates, tgt_dist)

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

        self.tokenizer.padding_side = 'right'

        selfie_batch = self.tokenizer(prompts, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)

        if False:
            ## sanity check
            selfie_length = selfie_batch.attention_mask.sum(dim=1)
            for i in range(len(rdmol2selfies)):
                mol_num = len(atom_vec[i])
                selfie_num = int(selfie_length[i])
                assert mol_num == rdmol2selfies[i].shape[0] + 2
                assert selfie_num == rdmol2selfies[i].shape[1] + 2
            ## end sanity check

        batch_size, d_atom = padded_atom_vec.shape[:2]
        d_selfies = selfie_batch.input_ids.shape[1]

        padded_rdmol2selfies_mask = rdmol2selfies_mask[0].new_zeros((batch_size, d_atom))
        padded_rdmol2selfies = rdmol2selfies[0].new_zeros((batch_size, d_atom, d_selfies))
        for i in range(batch_size):
            mask = rdmol2selfies_mask[i]
            padded_rdmol2selfies_mask[i, 1:1+mask.shape[0]].copy_(mask) # +1 because of the bos token in uni-mol atoms
            mapping = rdmol2selfies[i]
            padded_rdmol2selfies[i, 1:1+mapping.shape[0], 1:1+mapping.shape[1]].copy_(mapping) # +1 because of the bos token in uni-mol atoms

        data = Data(atom_vec=padded_atom_vec, edge_type=padded_edge_type, bond_type=padded_bond_type, coordinates=padded_coordinates, dist=padded_dist, tgt_coordinates=padded_tgt_coordinates, tgt_dist=padded_tgt_dist, rdmol2selfies=padded_rdmol2selfies, rdmol2selfies_mask=padded_rdmol2selfies_mask, selfies=selfies, smiles=smiles, rdmols=rdmols)
        return data, selfie_batch


class LMCollater(object):
    def __init__(self, tokenizer, max_sf_tokens, transform=None, condition=False, prop_norm=None):
        self.tokenizer = tokenizer
        self.max_sf_tokens = max_sf_tokens
        self.transform = transform
        self.condition = condition
        self.prop_norm = prop_norm

    def __call__(self, data_list):
        '''
        data_list: a list of data
        '''
        if isinstance(data_list[0], list):
            data_list = [item for sublist in data_list for item in sublist]

        # rdmols = [data['rdmol'] for data in data_list]
        # smiles = [data['smiles'] for data in data_list]
        selfies = [data['selfies'] for data in data_list]

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

        self.tokenizer.padding_side = 'right'

        selfie_batch0 = self.tokenizer(prompts, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)

        if self.condition:
            data_batch = Batch.from_data_list([self.transform(data) for data in data_list])
            context = data_batch.property
            assert len(self.prop_norm) == 1 # TODO only single property here
            # for i, key in enumerate(self.prop_norm.keys()):
            #     context[:, i] = (context[:, i] - self.prop_norm[key]['mean']) / self.prop_norm[key]['mad']
            condition_property = list(self.prop_norm.keys())[0]
            context = (context - self.prop_norm[condition_property]['mean']) / self.prop_norm[condition_property]['mad']
            selfie_batch0['context'] = context

        if 'selfies2' in data_list[0]:
            selfies2 = [data['selfies2'] for data in data_list]
            selfie_batch1 = self.tokenizer(selfies2, padding='max_length', return_tensors='pt', max_length=self.max_sf_tokens, truncation=True, add_special_tokens=True)
            return selfie_batch0, selfie_batch1
        else:
            return selfie_batch0


def normalize_rmsd_score(x, beta=1.0, smooth=0.1):
    x = 1.0 / (x**beta + smooth)
    return x / x.sum()


class UniMolVersion(Dataset):
    '''
    transform the data into uni-mol version
    '''
    def __init__(self, dataset, max_atoms, dataset_type, is_train=True):
        self.dataset = dataset
        if dataset_type == 'QM9':
            self.types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        elif dataset_type == 'GeomDrugs':
            self.types = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7, 'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
            self.charge_dict = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'As': 33, 'Br': 35, 'I': 53, 'Hg': 80, 'Bi': 83}
        self.id2type = {v: k for k, v in self.types.items()}
        self.id2charge = {v: k for k, v in self.charge_dict.items()}

        dict_path = Path(os.path.realpath(__file__)).parent / 'unimol_dict.txt'
        dictionary = Dictionary.load(str(dict_path))
        dictionary.add_symbol("[MASK]", is_special=True)
        self.num_types = len(dictionary)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.max_atoms = max_atoms
        self.remove_hydrogen = False
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.beta = 4.0
        self.smooth = 0.1
        self.topN = 10
        # self.retain_sf_hs = retain_sf_hs
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ## transform the data into uni-mol version
        # data = Data(atom_type=x, pos=pos, charge=torch.tensor(charges), fc=torch.tensor(formal_charges),
                    # edge_index=edge_index, edge_type=edge_type, y=y, num_atom=N, idx=i, rdmol=copy.deepcopy(mol), selfies=selfies)
        if self.is_train:
            return self.get_item_train(idx)
        else:
            return self.get_item_valid(idx)

    def get_item_train(self, idx):
        data = self.dataset[idx]
        atom_type = data.atom_type.tolist()
        atoms = [self.id2type[xx] for xx in atom_type]
        atoms = np.array(atoms)
        tgt_coordinates = data['pos'].numpy() # shape = [N, 3]

        ## sample a rdkit conformer
        rdkit_coords_cluster_list = data['rdkit_cluster_coords_list'][:self.topN] # only use top N conformers for sampling
        rmsd_score = np.asarray([item[1] for item in rdkit_coords_cluster_list])
        rmsd_score = normalize_rmsd_score(rmsd_score, self.beta, self.smooth)
        idx = np.random.choice(rmsd_score.shape[0], 1, replace=False, p=rmsd_score)[0]
        coordinates = rdkit_coords_cluster_list[idx][0]

        # data.rdmol = set_rdmol_positions(data.rdmol, tgt_coordinates, removeHs=False, add_conformer=True)
        data.rdmol = set_rdmol_positions(data.rdmol, coordinates, removeHs=False, add_conformer=True)

        ## process the coordinates
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            tgt_coordinates = tgt_coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                tgt_coordinates = tgt_coordinates[:-end_idx]

        ## deal with cropping
        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]
            tgt_coordinates = tgt_coordinates[index]

        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)
            tgt_coordinates = tgt_coordinates - tgt_coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)
            tgt_coordinates = np.concatenate([np.zeros((1, 3)), tgt_coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        tgt_dist = distance_matrix(tgt_coordinates, tgt_coordinates).astype(np.float32)
        coordinates, dist, tgt_coordinates, tgt_dist = torch.from_numpy(coordinates), torch.from_numpy(dist), torch.from_numpy(tgt_coordinates), torch.from_numpy(tgt_dist)

        ## prepare the bond type matrix
        # data.edge_index # shape = [2, num_edges]; data.edge_type # shape = [num_edges]
        num_atoms = dist.shape[0]
        sp_bond_type = torch.sparse_coo_tensor(data.edge_index + 1, data.edge_type, size=(num_atoms, num_atoms), dtype=torch.long) # +1 because of the bos token in uni-mol atoms
        bond_type = sp_bond_type.to_dense()

        if False:
            return {'atom_vec': atom_vec, 'coordinates': coordinates, 'dist': dist, 'tgt_coordinates': tgt_coordinates, 'tgt_dist': tgt_dist, 'edge_type': edge_type, 'idx': data.idx, 'selfies': data.selfies, 'rdmol2selfies': data.rdmol2selfies, 'rdmol2selfies_mask': data.rdmol2selfies_mask, 'smiles': data.cano_smiles_woh, 'rdmol': data.rdmol, 'bond_type': bond_type}
        else:
            return {'atom_vec': atom_vec, 'coordinates': coordinates, 'dist': dist, 'tgt_coordinates': tgt_coordinates, 'tgt_dist': tgt_dist, 'edge_type': edge_type, 'idx': data.idx, 'selfies': data.selfies, 'rdmol2selfies': data.rdmol2selfies, 'rdmol2selfies_mask': data.rdmol2selfies_mask, 'smiles': data.smiles, 'rdmol': data.rdmol, 'bond_type': bond_type}

    def get_item_valid(self, idx):
        '''
        the difference is that we use all the conformers for validation without sampling
        '''
        data = self.dataset[idx]
        atom_type = data.atom_type.tolist()
        atoms = [self.id2type[xx] for xx in atom_type]
        atoms = np.array(atoms)
        tgt_coordinates = data['pos'].numpy() # shape = [N, 3]

        ## pad the rdkit coords to the same length
        rdkit_coords_cluster_list = data['rdkit_cluster_coords_list'][:self.topN] # only use top N conformers for sampling
        coordinates_list = [item[0] for item in rdkit_coords_cluster_list]
        if len(coordinates_list) < self.topN:
            diff = self.topN - len(coordinates_list)
            coordinates_list.extend([coordinates_list[-1]] * diff)

        data_list = []
        for coordinates in coordinates_list:
            _tgt_coordinates = tgt_coordinates.copy()
            rdmol = set_rdmol_positions(data.rdmol, coordinates, removeHs=False, add_conformer=True)

            ## process the coordinates
            if self.remove_hydrogen:
                mask_hydrogen = atoms != "H"
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]
                _tgt_coordinates = _tgt_coordinates[mask_hydrogen]

            if not self.remove_hydrogen and self.remove_polar_hydrogen:
                end_idx = 0
                for i, atom in enumerate(atoms[::-1]):
                    if atom != "H":
                        break
                    else:
                        end_idx = i + 1
                if end_idx != 0:
                    atoms = atoms[:-end_idx]
                    coordinates = coordinates[:-end_idx]
                    _tgt_coordinates = _tgt_coordinates[:-end_idx]

            ## deal with cropping
            if len(atoms) > self.max_atoms:
                index = np.random.permutation(len(atoms))[:self.max_atoms]
                atoms = atoms[index]
                coordinates = coordinates[index]
                _tgt_coordinates = _tgt_coordinates[index]

            atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

            if self.normalize_coords:
                coordinates = coordinates - coordinates.mean(axis=0)
                _tgt_coordinates = _tgt_coordinates - _tgt_coordinates.mean(axis=0)

            if self.add_special_token:
                atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
                coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)
                _tgt_coordinates = np.concatenate([np.zeros((1, 3)), _tgt_coordinates, np.zeros((1, 3))], axis=0)

            ## obtain edge types; which is defined as the combination of two atom types
            edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
            dist = distance_matrix(coordinates, coordinates).astype(np.float32)
            tgt_dist = distance_matrix(_tgt_coordinates, _tgt_coordinates).astype(np.float32)
            coordinates, dist, _tgt_coordinates, tgt_dist = torch.from_numpy(coordinates), torch.from_numpy(dist), torch.from_numpy(_tgt_coordinates), torch.from_numpy(tgt_dist)

            ## prepare the bond type matrix
            # data.edge_index # shape = [2, num_edges]; data.edge_type # shape = [num_edges]
            num_atoms = dist.shape[0]
            sp_bond_type = torch.sparse_coo_tensor(data.edge_index + 1, data.edge_type, size=(num_atoms, num_atoms), dtype=torch.long) # +1 because of the bos token in uni-mol atoms
            bond_type = sp_bond_type.to_dense()

            if False:
                data_dict =  {'atom_vec': atom_vec, 'coordinates': coordinates, 'dist': dist, 'tgt_coordinates': _tgt_coordinates, 'tgt_dist': tgt_dist, 'edge_type': edge_type, 'idx': data.idx, 'selfies': data.selfies, 'rdmol2selfies': data.rdmol2selfies, 'rdmol2selfies_mask': data.rdmol2selfies_mask, 'smiles': data.cano_smiles_woh, 'rdmol': rdmol, 'bond_type': bond_type}
            else:
                data_dict =  {'atom_vec': atom_vec, 'coordinates': coordinates, 'dist': dist, 'tgt_coordinates': _tgt_coordinates, 'tgt_dist': tgt_dist, 'edge_type': edge_type, 'idx': data.idx, 'selfies': data.selfies, 'rdmol2selfies': data.rdmol2selfies, 'rdmol2selfies_mask': data.rdmol2selfies_mask, 'smiles': data.smiles, 'rdmol': rdmol, 'bond_type': bond_type}

            data_list.append(data_dict)
        return data_list


def process_smiles2rdmol(smiles):
    ## generate conformations for the sampled smiles
    ## need 1: the mapping between rdmol to selfies tokens
    ## need 2: initial conformations generated by RDKit
    rdmol = Chem.MolFromSmiles(smiles)
    ## add hs to the rdmol, this is necessary for rdkit to generate the initial conformations
    rdmol = Chem.AddHs(rdmol)
    rdmol2smiles, cano_smiles_woh = build_rdkit2cano_smiles_withoutH_mapping(rdmol)
    selfies = sf.encoder(cano_smiles_woh)
    rdkit_cluster_coords = generate_conformers(rdmol, M=10, removeHs=False, addHs=False, use_clustering=True, N=1)[0] # use only the cluster center as the initial conformation

    N = rdmol.GetNumAtoms()
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        if bond.GetBondType() == BT.AROMATIC:
            print('meet aromatic bond!')

        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]


    data = Data(selfies=selfies, edge_index=edge_index, edge_type=edge_type, rdmol2smiles=rdmol2smiles, rdkit_cluster_coords=rdkit_cluster_coords, cano_smiles_woh=cano_smiles_woh, rdmol=rdmol)

    ## obtain the mapping between rdmol to selfies tokens
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



def process_smiles2rdmol_v2(smiles, canonicalize=True):
    ## generate conformations for the sampled smiles
    ## need 1: the mapping between rdmol to selfies tokens
    ## need 2: initial conformations generated by RDKit
    rdmol = Chem.MolFromSmiles(smiles)
    ## add hs to the rdmol, this is necessary for rdkit to generate the initial conformations
    rdmol = Chem.AddHs(rdmol)
    if canonicalize:
        rdmol2smiles, output_smiles = build_rdkit2cano_smiles_withoutH_mapping(rdmol)
    else:
        rdmol2smiles, output_smiles = build_rdkit2rand_smiles_withoutH_mapping(rdmol, 'none')

    selfies = sf.encoder(output_smiles)
    rdkit_cluster_coords = generate_conformers(rdmol, M=10, removeHs=False, addHs=False, use_clustering=True, N=1)[0] # use only the cluster center as the initial conformation

    N = rdmol.GetNumAtoms()
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        if bond.GetBondType() == BT.AROMATIC:
            print('meet aromatic bond!')

        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]


    data = Data(selfies=selfies, edge_index=edge_index, edge_type=edge_type, rdmol2smiles=rdmol2smiles, rdkit_cluster_coords=rdkit_cluster_coords, smiles=output_smiles, rdmol=rdmol)

    ## obtain the mapping between rdmol to selfies tokens
    smiles2selfies, selfies_tokens = get_smiles2selfies_mapping(data.smiles) # smiles2selfies is a dict
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


class UnCondPredictDataset(Dataset):
    '''
    transform the data into uni-mol version
    '''
    def __init__(self, smiles_list, max_atoms):
        self.smiles_list = smiles_list

        dict_path = Path(os.path.realpath(__file__)).parent / 'unimol_dict.txt'
        dictionary = Dictionary.load(str(dict_path))
        dictionary.add_symbol("[MASK]", is_special=True)
        self.num_types = len(dictionary)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.max_atoms = max_atoms
        self.remove_hydrogen = False
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.beta = 4.0
        self.smooth = 0.1
        self.topN = 20

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        ## transform the data into uni-mol version
        smiles = self.smiles_list[idx]
        data = process_smiles2rdmol(smiles)

        ## use the unimol model to refine the conformations
        atoms = [atom.GetSymbol() for atom in data.rdmol.GetAtoms()]
        atoms = np.asarray(atoms)

        ## sample a rdkit conformer
        coordinates = data['rdkit_cluster_coords'] # only use top N conformers for sampling
        data.rdmol = set_rdmol_positions(data.rdmol, coordinates, removeHs=False)

        ## process the coordinates
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

        ## prepare the bond type matrix
        # data.edge_index # shape = [2, num_edges]; data.edge_type # shape = [num_edges]
        num_atoms = dist.shape[0]
        sp_bond_type = torch.sparse_coo_tensor(data.edge_index + 1, data.edge_type, size=(num_atoms, num_atoms), dtype=torch.long) # +1 because of the bos token in uni-mol atoms
        bond_type = sp_bond_type.to_dense()

        return {'atom_vec': atom_vec, 'coordinates': coordinates, 'dist': dist, 'edge_type': edge_type, 'selfies': data.selfies, 'rdmol2selfies': data.rdmol2selfies, 'rdmol2selfies_mask': data.rdmol2selfies_mask, 'smiles': smiles, 'rdmol': data.rdmol, 'bond_type': bond_type}


class UnCondPredictDatasetV2(Dataset):
    '''
    transform the data into uni-mol version
    in this version, we do not canonicalize the smiles and keep them in the original order
    '''
    def __init__(self, smiles_list, max_atoms):
        self.smiles_list = smiles_list


        ## this is a pre-processing to prevent that the smiles will change after being processed by RDKit
        processed_smiles_list = []
        for s in self.smiles_list:
            rdmol = Chem.MolFromSmiles(s)
            s = Chem.MolToSmiles(rdmol, canonical=False, isomericSmiles=False)
            processed_smiles_list.append(s)
        self.smiles_list = processed_smiles_list

        dict_path = Path(os.path.realpath(__file__)).parent / 'unimol_dict.txt'
        dictionary = Dictionary.load(str(dict_path))
        dictionary.add_symbol("[MASK]", is_special=True)
        self.num_types = len(dictionary)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.max_atoms = max_atoms
        self.remove_hydrogen = False
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.beta = 4.0
        self.smooth = 0.1
        self.topN = 20

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        ## transform the data into uni-mol version
        smiles = self.smiles_list[idx]
        data = process_smiles2rdmol_v2(smiles, False)

        ## use the unimol model to refine the conformations
        atoms = [atom.GetSymbol() for atom in data.rdmol.GetAtoms()]
        atoms = np.asarray(atoms)

        ## sample a rdkit conformer
        coordinates = data['rdkit_cluster_coords'] # only use top N conformers for sampling
        data.rdmol = set_rdmol_positions(data.rdmol, coordinates, removeHs=False)

        ## process the coordinates
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

        ## prepare the bond type matrix
        # data.edge_index # shape = [2, num_edges]; data.edge_type # shape = [num_edges]
        num_atoms = dist.shape[0]
        sp_bond_type = torch.sparse_coo_tensor(data.edge_index + 1, data.edge_type, size=(num_atoms, num_atoms), dtype=torch.long) # +1 because of the bos token in uni-mol atoms
        bond_type = sp_bond_type.to_dense()

        return {'atom_vec': atom_vec, 'coordinates': coordinates, 'dist': dist, 'edge_type': edge_type, 'selfies': data.selfies, 'rdmol2selfies': data.rdmol2selfies, 'rdmol2selfies_mask': data.rdmol2selfies_mask, 'smiles': smiles, 'rdmol': data.rdmol, 'bond_type': bond_type}



class QM9DataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer

        rand_smiles = args.rand_smiles if args is not None else 'None'
        dataset = QM9Dataset(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=rand_smiles)
        self.dataset = dataset
        max_atoms = int(dataset._data.num_atom.max()) + 2 # +2 because of the bos and eos token;
        self.max_atoms = max_atoms
        print('QM9 max num atoms', max_atoms)

        ## obtain max selfies token length
        selfies_list = dataset._data['selfies']
        selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]
        self.max_sf_tokens = max(max(selfies_lens) + 2 + 5, 96) # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling
        print('max selfies tokens', self.max_sf_tokens)

        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones without selfies
        selfies = np.array(dataset._data.selfies)

        print('before filtering', len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(valid_idx), len(test_idx))
        self.train_dataset = UniMolVersion(dataset.index_select(train_idx), max_atoms, dataset_type='QM9')
        # self.valid_dataset = UniMolVersion(dataset.index_select(valid_idx), max_atoms, dataset_type='QM9')
        # self.test_dataset = UniMolVersion(dataset.index_select(test_idx), max_atoms, dataset_type='QM9')
        self.test_dataset = UniMolVersion(dataset.index_select(test_idx), max_atoms, dataset_type='QM9', is_train=False)
        self.predict_dataset = None

        ## load rdmols of subsets
        rdmols = dataset._data.rdmol
        train_idx = train_idx.tolist()
        valid_idx = valid_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]

        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('qm9_with_h'), os.path.join(root, 'processed'))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.selfies_tokenizer, self.max_atoms, self.max_sf_tokens, self.train_dataset.dictionary.pad()),
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
            collate_fn=TrainCollater(self.selfies_tokenizer, self.max_atoms, self.max_sf_tokens, self.train_dataset.dictionary.pad()),
        )
        return loader

    def setup_predict_dataset(self, smiles_list):
        self.predict_dataset = UnCondPredictDatasetV2(smiles_list, self.max_atoms + 10) # +10 to allow sampling larger molecules during inference

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
                collate_fn=InferenceCollater(self.selfies_tokenizer, self.max_atoms, self.max_sf_tokens, self.train_dataset.dictionary.pad()),
            )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--addHs', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/qm9')
        parser.add_argument('--infer_time', type=float, default=0.9946)
        parser.add_argument('--infer_noise', type=float, default=0.9999)
        parser.add_argument('--not_aug_rotation', action='store_true', default=False)
        parser.add_argument('--t_cond', type=str, default='t')
        return parent_parser


class GeomDrugsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/drugs',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer

        dataset = GeomDrugsDataset(root=root, selfies_tokenizer=selfies_tokenizer)
        self.dataset = dataset
        max_atoms = int(dataset._data.num_atom.max()) + 2 # +2 because of the bos and eos token;
        self.max_atoms = max_atoms
        print('GEOM Drugs max num atoms', max_atoms)

        ## obtain max selfies token length
        selfies_list = dataset._data['selfies']
        selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]
        self.max_sf_tokens = max(selfies_lens) + 2 + 5 # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling
        print('max selfies tokens', self.max_sf_tokens)

        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones without selfies
        selfies = np.array(dataset._data.selfies)

        print('before filtering', len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(valid_idx), len(test_idx))
        self.train_dataset = UniMolVersion(dataset.index_select(train_idx), max_atoms, dataset_type='GeomDrugs')
        self.test_dataset = UniMolVersion(dataset.index_select(test_idx), max_atoms, dataset_type='GeomDrugs', is_train=False)
        self.predict_dataset = None

        ## load rdmols of subsets
        rdmols = dataset._data.rdmol
        train_idx = train_idx.tolist()
        valid_idx = valid_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]

        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('geom_with_h_1'), os.path.join(root, 'processed'))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.selfies_tokenizer, self.max_atoms, self.max_sf_tokens, self.train_dataset.dictionary.pad()),
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
            collate_fn=TrainCollater(self.selfies_tokenizer, self.max_atoms, self.max_sf_tokens, self.train_dataset.dictionary.pad()),
        )
        return loader

    def setup_predict_dataset(self, smiles_list):
        self.predict_dataset = UnCondPredictDatasetV2(smiles_list, self.max_atoms + 10) # +10 to allow sampling larger molecules during inference

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
                collate_fn=InferenceCollater(self.selfies_tokenizer, self.max_atoms, self.max_sf_tokens, self.train_dataset.dictionary.pad()),
            )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--root', type=str, default='data/drugs')
        parser.add_argument('--rand_smiles', type=str, default='restricted')
        parser.add_argument('--addHs', action='store_true', default=False)
        return parent_parser



class QM9LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer

        if not hasattr(args, 'condition_property') or args.condition_property == None:
            self.transform = None
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            dataset_info = get_dataset_info('qm9_second_half')
            prop2idx = dataset_info['prop2idx']
            include_aromatic = False
            self.transform = EdgeComCondTransform(dataset_info['atom_encoder'].values(), include_aromatic, prop2idx[args.condition_property])
        else:
            raise NotImplementedError(f"{args.conditon} is not supported")

        rand_smiles = args.rand_smiles if args is not None else 'None'
        dataset = QM9LMDataset(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=rand_smiles, aug_inv=args.aug_inv > 0)
        self.dataset = dataset
        max_atoms = int(dataset._data.num_atom.max()) + 2 # +2 because of the bos and eos token;
        self.max_atoms = max_atoms
        print('QM9 max num atoms', max_atoms)

        ## obtain max selfies token length
        selfies_list = dataset._data['selfies']
        selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]
        if args.addHs:
            self.max_sf_tokens = max(max(selfies_lens) + 2 + 5, 96) # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling
        else:
            self.max_sf_tokens = max(selfies_lens) + 2 + 5 # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling

        print('max selfies tokens', self.max_sf_tokens)

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
        self.train_dataset = dataset.index_select(train_idx)
        self.test_dataset = dataset.index_select(test_idx)
        self.predict_dataset = None

        if args.condition_property == None:
            self.prop_norms = None
            self.prop_dist = None
            pass
        elif args.condition_property in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv']:
            prop2idx_sub = {
                args.condition_property: prop2idx[args.condition_property]
            }
            self.prop_norms = dataset.index_select(valid_idx).compute_property_mean_mad(prop2idx_sub)
            self.prop_dist = DistributionProperty(dataset.index_select(train_idx), prop2idx_sub, normalizer=self.prop_norms)
            self.nodes_dist = get_node_dist(dataset_info)
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

        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('qm9_with_h'), os.path.join(root, 'processed'))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=LMCollater(
                tokenizer=self.selfies_tokenizer,
                max_sf_tokens=self.max_sf_tokens,
                transform=self.transform,
                condition=(self.transform is not None),
                prop_norm=self.prop_norms if self.transform is not None else None
            ),
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
            collate_fn=LMCollater(
                tokenizer=self.selfies_tokenizer,
                max_sf_tokens=self.max_sf_tokens,
                transform=self.transform,
                condition=(self.transform is not None),
                prop_norm=self.prop_norms if self.transform is not None else None
            ),
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
                collate_fn=LMCollater(
                    tokenizer=self.selfies_tokenizer,
                    max_sf_tokens=self.max_sf_tokens,
                    transform=self.transform,
                    condition=(self.transform is not None),
                    prop_norm=self.prop_norms if self.transform is not None else None
            ),
            )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--rand_smiles', type=str, default='')
        parser.add_argument('--root', type=str, default='data/qm9')
        return parent_parser


class GeomDrugsLMDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = 'data/drugs',
        num_workers: int = 0,
        batch_size: int = 256,
        selfies_tokenizer = None,
        args=None,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.selfies_tokenizer = selfies_tokenizer

        rand_smiles = args.rand_smiles if args is not None else 'None'
        dataset = GeomDrugsLMDataset(root=root, selfies_tokenizer=selfies_tokenizer, rand_smiles=rand_smiles, aug_inv=args.aug_inv > 0)
        self.dataset = dataset
        max_atoms = int(dataset._data.num_atom.max()) + 2 # +2 because of the bos and eos token;
        self.max_atoms = max_atoms
        print('GEOM Drugs max num atoms', max_atoms)

        ## obtain max selfies token length
        selfies_list = dataset._data['selfies']
        selfies_lens = [len(list(sf.split_selfies(selfies))) for selfies in selfies_list]
        self.max_sf_tokens = max(selfies_lens) + 2 + 5 # +2 because of the bos and eos token; +5 to enlarge the space of molecule sampling
        print('max selfies tokens', self.max_sf_tokens)

        splits = dataset.get_idx_split()
        train_idx = splits['train']
        valid_idx = splits['valid']
        test_idx = splits['test']

        ## filter the ones without selfies
        selfies = np.array(dataset._data.selfies)

        print('before filtering', len(train_idx), len(valid_idx), len(test_idx))
        train_idx = train_idx[selfies[train_idx] != np.array('')]
        valid_idx = valid_idx[selfies[valid_idx] != np.array('')]
        test_idx = test_idx[selfies[test_idx] != np.array('')]
        print('after filtering', len(train_idx), len(valid_idx), len(test_idx))
        self.train_dataset = dataset.index_select(train_idx)
        self.test_dataset = dataset.index_select(test_idx)
        self.predict_dataset = None

        ## load rdmols of subsets
        rdmols = dataset._data.rdmol
        train_idx = train_idx.tolist()
        valid_idx = valid_idx.tolist()
        test_idx = test_idx.tolist()
        self.train_rdmols = [rdmols[i] for i in train_idx]
        self.valid_rdmols = [rdmols[i] for i in valid_idx]
        self.test_rdmols = [rdmols[i] for i in test_idx]

        self.get_moses_metrics = get_moses_metrics(self.test_rdmols, 1)
        self.get_sub_geometry_metric = get_sub_geometry_metric(self.test_rdmols, get_dataset_info('geom_with_h_1'), os.path.join(root, 'processed'))

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
    from transformers import AutoTokenizer
    from utils_v2 import set_seed
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained('all_checkpoints/mollama')
    # print(tokenizer.unk_token_id, tokenizer.unk_token)
    dataset = QM9Dataset(root='./data/qm9v6', selfies_tokenizer=tokenizer)
    for i in range(len(dataset)):
        for coord in dataset[i].rdkit_cluster_coords_list:
            new_rdmol = set_rdmol_positions(dataset[i].rdmol, coord, False)
            rmsd = get_best_rmsd(dataset[i].rdmol, new_rdmol)
