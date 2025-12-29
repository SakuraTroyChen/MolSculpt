import os
import random
import torch
from tqdm import tqdm
import copy
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.in_memory_dataset import nested_iter
import pickle
from torch_geometric.data.separate import separate
from data_provider.conf_gen_cal_metrics import set_rdmol_positions
import selfies as sf
from data_provider.mol_mapping_utils import get_smiles2selfies_mapping, build_rdkit2rand_smiles_withoutH_mapping
from mol_utils.featurization import featurize_mol, featurize_mol_from_smiles_v2
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
from model.torsional_diffusion.sampling import embed_seeds_v3
from data_provider.lap_utils import compute_posenc_stats


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]



def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]

class QM9TorDFInfer(Dataset):
    def __init__(self, processed_path, selfies_tokenizer, rand_smiles=False, addHs=False, raw_smiles_path=None, raw_conf_file=None, infer_time=0.001):
        # self.pos_std = torch.FloatTensor([1.9181, 1.2353, 0.9104]).view(1, -1)
        self.pos_std = 1.4182
        self.threshold = 0.5
        self.selfies_tokenizer = selfies_tokenizer
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        if os.path.exists(processed_path):
            self.data, self.slices, self.seed_pos_list, self.gt_conf_list, self.num_failures = torch.load(processed_path)
        else:
            raw_smiles = pd.read_csv(raw_smiles_path)
            with open(raw_conf_file, 'rb') as f:
                raw_conf = pickle.load(f)
            self.data, self.slices, self.seed_pos_list, self.gt_conf_list, self.num_failures = self.process_data(raw_smiles, raw_conf, processed_path)

        assert len(self.seed_pos_list) == len(self.gt_conf_list)
        self.mol_len = len(self.seed_pos_list)
        for seed_pos in self.seed_pos_list:
            assert len(seed_pos) > 0

        ## pad the seed_pos_list to have twice the number of conformers as the ground truth
        count = 0
        for i, seed_pos in enumerate(self.seed_pos_list):
            gt_conf = self.gt_conf_list[i]
            if len(gt_conf) == 0:
                ## skip molecules that have no high quality ground truth conformers
                self.seed_pos_list[i] = []
                continue
            num_missing = 2 * len(gt_conf) - len(seed_pos)
            if num_missing < 0:
                # print('-------------------')
                # print('-------------------')
                # print('warning: wrong number of ground truth conformers')
                # print('-------------------')
                # print('-------------------')
                continue
            for _ in range(num_missing):
                rand_pos = torch.randn_like(seed_pos[0]) * self.pos_std
                # rand_pos -= rand_pos.mean(dim=0, keepdim=True)
                seed_pos.append(rand_pos)
                count += 1
        print(f'Added {count} random conformers')

        if True:
            ## sort the seed_pos_list by molecule size; this is to make the inference faster
            self.seed_pos_list = [(mol_idx, pos_list) for mol_idx, pos_list in enumerate(self.seed_pos_list) if len(pos_list) > 0]
            self.seed_pos_list = sorted(self.seed_pos_list, key=lambda x: len(x[1][0]))
            self.seed_pos_list = [(mol_idx, pos) for mol_idx, pos_list in self.seed_pos_list for pos in pos_list]
        else:
            self.seed_pos_list = [(mol_idx, pos) for mol_idx, pos_list in enumerate(self.seed_pos_list) for pos in pos_list]

        assert 1 >= infer_time >= 0
        self.infer_time = infer_time


    def process_data(self, raw_smiles, raw_conf, processed_path):
        smiles_data = raw_smiles.values.tolist()
        data_list = []
        seed_pos_list = []
        gt_conf_list = []

        print('Processing inference data')
        num_failures = 0
        for smi, num_conf, corrected_smi in tqdm(smiles_data):
            mol, data = featurize_mol_from_smiles_v2(corrected_smi, 'qm9')
            if not mol:
                num_failures += 1
                continue
            conformers, _ = embed_seeds_v3(mol, data, num_conf * 2, False, corrected_smi, None, None, False)
            if not conformers:
                num_failures += 1
                continue

            data['corrected_smiles'] = corrected_smi
            data['corrected_selfies'] = sf.encoder(corrected_smi)
            data['rdmol'] = mol
            data = compute_posenc_stats(data, is_undirected=True, max_freqs=28, eigvec_norm=None)
            data_list.append(data)

            seed_pos = [conf.pos for conf in conformers]
            seed_pos = [pos - pos.mean(dim=0, keepdim=True) for pos in seed_pos]
            seed_pos_list.append(seed_pos)

            gt_confs = clean_confs(corrected_smi, raw_conf[smi])
            gt_conf_list.append(gt_confs)

        print(f'Passed {num_failures} out of {len(smiles_data)}')
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices, seed_pos_list, gt_conf_list, num_failures), processed_path)
        return data, slices, seed_pos_list, gt_conf_list, num_failures

    def __len__(self):
        return len(self.seed_pos_list)

    def len(self):
        return self.__len__()

    def get_idx_data(self, idx: int):
        # TODO (matthias) Avoid unnecessary copy here.
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.mol_len * [None]
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
        mol_idx, seed_pos = self.seed_pos_list[idx]
        data = self.get_idx_data(mol_idx)
        data['seed_pos_idx'] = idx
        data['mol_idx'] = mol_idx
        data['pos'] = seed_pos / self.pos_std

        ## setup perturb seed
        ## fake randomness; this is to generate reproducible test results
        data['t'] = self.infer_time

        data['rdmol'].RemoveAllConformers()
        # data['rdmol'] = set_rdmol_positions(rdmol, data['pos'], removeHs=False, )
        # assert data['rdmol'].GetNumConformers() == 1

        rdmol2smiles, output_smiles = build_rdkit2rand_smiles_withoutH_mapping(data.rdmol, self.rand_smiles, self.addHs)
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


class QM9TorDF(Dataset):
    def __init__(self, processed_path, selfies_tokenizer, rand_smiles=False, addHs=False, raw_path=None, transform=None, mode='train', flatten_dataset=False):
        self.selfies_tokenizer = selfies_tokenizer
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        self.transform = transform
        self.mode = mode
        self.conf_limit = 30
        if os.path.exists(processed_path):
            self.data, self.slices, self.pos_list = torch.load(processed_path)
        else:
            with open(raw_path, 'rb') as f:
                raw_data = pickle.load(f)
            self.data, self.slices, self.pos_list = self.process_data(raw_data, processed_path)

        self.pos_std = 1.4182
        self.flatten_dataset = flatten_dataset
        if self.flatten_dataset:
            idx2mol = []
            idx2conf = []
            for mol_idx, pos_list in enumerate(self.pos_list):
                idx2mol.extend([mol_idx] * len(pos_list))
                idx2conf.extend(list(range(len(pos_list))))
            self.idx2mol = idx2mol
            self.idx2conf = idx2conf

    def process_data(self, raw_data, processed_path):
        ## the original data uses torsional diffusion features, I will now use unimol features
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            raise ImportError("Please install 'rdkit' to alternatively process the raw data.")

        data_list = []
        pos_list = []
        for i, data in enumerate(tqdm(raw_data)):
            # remove unnecessary data
            data.pop('x')
            data.pop('edge_index')
            data.pop('edge_attr')
            data.pop('edge_mask', None)
            data.pop('mask_rotate', None)
            assert isinstance(data['pos'], list) and len(data['pos']) >= 1
            pos = data.pop('pos')
            pos_list.append(pos)
            data['rdmol'] = data.pop('mol')

            _data = featurize_mol(data['rdmol'], types='qm9')
            data['x'] = _data['x']
            data['z'] = _data['x']
            data['edge_index'] = _data['edge_index']
            data['edge_attr'] = _data['edge_attr']
            data['canonical_selfies'] = sf.encoder(data['canonical_smi'])
            data = compute_posenc_stats(data, is_undirected=True, max_freqs=28, eigvec_norm=None)
            data_list.append(data)

        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices, pos_list), processed_path)
        return data, slices, pos_list

    def __len__(self):
        if self.flatten_dataset:
            return len(self.idx2conf)
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def len(self):
        return self.__len__()

    def get_idx_data(self, idx: int):
        # TODO (matthias) Avoid unnecessary copy here.
        if self.len() == 1:
            return self.data.clone()

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return self._data_list[idx].clone()

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        data['pos'] = self.pos_list[idx]

        self._data_list[idx] = data.clone()
        return data.clone()

    def __getitem__(self, idx):
        ## obtain the data first
        if self.flatten_dataset:
            mol_idx = self.idx2mol[idx]
            conf_idx = self.idx2conf[idx]
            data = self.get_idx_data(mol_idx)
            data['pos'] = data['pos'][conf_idx]
        else:
            data = self.get_idx_data(idx)
            assert isinstance(data['pos'], list)
            if self.mode == 'train':
                ## true random
                data['pos'] = random.choice(data['pos'])
            elif self.mode == 'valid':
                ## fake random that does not change with time; this is to generate reproducible validation results
                rng = random.Random(idx)
                data['pos'] = rng.choice(data['pos'])
            else:
                raise NotImplementedError

        data['pos'] -= data['pos'].mean(dim=0, keepdim=True)
        rdmol = copy.deepcopy(data['rdmol'])
        rdmol.RemoveAllConformers()
        data['rdmol'] = set_rdmol_positions(rdmol, data['pos'], removeHs=False, )
        assert data['rdmol'].GetNumConformers() == 1

        data['pos'] /= self.pos_std
        if self.transform:
            data = self.transform(data)

        rdmol2smiles, output_smiles = build_rdkit2rand_smiles_withoutH_mapping(data.rdmol, self.rand_smiles, self.addHs)
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


if __name__ == '__main__':
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('all_checkpoints/mollama')
    # tokenizer.add_bos_token = True
    # tokenizer.add_eos_token = True
    # transform = TorsionNoiseTransform()
    dataset = QM9TorDF('data/tordf_qm9/processed_train.pt', None, 'restricted', False, 'data/torsion_data/tordf_qm9/tordf.train', transform=None, mode='valid')
    pos_list_list = dataset.pos_list
    pos_list = [pos for pos_list in pos_list_list for pos in pos_list]
    pos_array = torch.cat(pos_list, dim=0)
    pos_array = pos_array - pos_array.mean(dim=0, keepdim=True)
    print(pos_array.std(), pos_array.std(dim=0))

    # print(dataset[0].pos)
    # print(dataset[0].pos)
    # print(dataset[0].pos)
    # print(dataset[0].pos)
    # print(dataset[0].pos)
    # print(dataset[0].pos)


    # dataset = QM9TorDF('data/torsion_data/tordf_qm9/processed_val.pt', None, 'restricted', False, 'data/torsion_data/tordf_qm9/tordf.val', transform=None)
    # dataset = QM9TorDF('data/torsion_data/tordf_qm9/processed_test.pt', None, 'restricted', False, 'data/torsion_data/tordf_qm9/tordf.test', transform=None)

    # conf_num = []
    # # for i in tqdm(range(len(dataset))):
    # for i in tqdm(range(1000)):
    #     data = dataset[i]
    #     print(data)
    #     print(data['pos'].shape, data.x.shape)
    #     conf_num.append(len(data['pos']))
    #     input()
    # print(max(conf_num), min(conf_num), sum(conf_num)/len(conf_num))
    # dataset = QM9TorDFInfer('data/tordf_qm9/processed_inference_test.pt', None, 'restricted', False, 'data/tordf_qm9/test_smiles.csv', 'data/tordf_qm9/test_mols.pkl')
    # dataset = QM9TorDFInfer('data/tordf_qm9/processed_inference_test_v2.pt', None, 'restricted', False, 'data/tordf_qm9/test_smiles.csv', 'data/tordf_qm9/test_mols.pkl')