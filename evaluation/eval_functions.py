# Description: Evaluation functions for 2D and 3D molecules
import copy
import numpy as np
import pickle
import os
import torch
from rdkit import Chem
from evaluation.jodo.rdkit_metric import eval_rdmol
from evaluation.jodo.mose_metric import compute_intermediate_statistics, mapper, get_smiles, reconstruct_mol, MeanProperty
from fcd_torch import FCD as FCDMetric
from multiprocessing import Pool
from moses.metrics.metrics import SNNMetric, FragMetric, ScafMetric, internal_diversity, \
    fraction_passes_filters, weight, logP, SA, QED
from evaluation.jodo.stability import bond_list, allowed_fc_bonds, stability_bonds
from rdkit.Geometry import Point3D
from evaluation.jodo.bond_analyze import get_bond_order, geom_predictor, allowed_bonds, allowed_fc_bonds
from evaluation.jodo.cal_geometry import load_target_geometry, compute_geo_mmd, cal_bond_distance, cal_bond_angle, cal_dihedral_angle
from tqdm import tqdm
from rdkit.Chem import AllChem


def check_2D_stability(rdmol):
    """Convert the generated tensors to rdkit mols and check stability"""
    rdmol = Chem.AddHs(rdmol)
    atom_num = rdmol.GetNumAtoms()
    # kekulize mol and iterate bonds for stability
    new_mol = copy.deepcopy(rdmol)
    try:
        Chem.Kekulize(new_mol)
    except:
        print('Can\'t Kekulize mol.')
        pass

    nr_bonds = np.zeros(atom_num, dtype='int')
    for bond in new_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # if bond_type in stability_bonds:
        #     order = stability_bonds[bond_type]
        # else:
        #     order = 0
        order = stability_bonds[bond_type]
        nr_bonds[start] += order
        nr_bonds[end] += order

    # stability
    nr_stable_bonds = 0
    atom_types_str = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    formal_charges = [atom.GetFormalCharge() for atom in rdmol.GetAtoms()]
    for atom_type_i, nr_bonds_i, fc_i in zip(atom_types_str, nr_bonds, formal_charges):
        # fc_i = fc_i.item()
        possible_bonds = allowed_fc_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[fc_i] if fc_i in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == nr_bonds_i if type(expected_bonds) == int else nr_bonds_i in expected_bonds
        else:
            is_stable = nr_bonds_i in possible_bonds
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == atom_num
    return molecule_stable, nr_stable_bonds, atom_num
    

def get_2D_edm_metric(predict_mols, train_mols=None):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]
        train_smiles = [Chem.CanonSmiles(s) for s in train_smiles]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    for mol in tqdm(predict_mols):
        try:
            validity_res = check_2D_stability(mol)
        except:
            print('Check stability failed.')
            validity_res = [0, 0, mol.GetNumAtoms()]
        molecule_stable += int(validity_res[0])
        nr_stable_bonds += int(validity_res[1])
        n_atoms += int(validity_res[2])

    # Stability
    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    
    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    # Basic rdkit metric result (Validity, Fragment, Unique)
    rdkit_dict = eval_rdmol(predict_mols, train_smiles)
    
    ## union the two dicts
    output_dict.update(rdkit_dict)
    return output_dict


def check_3D_stability(positions, atoms, dataset_name, debug=False, rdmol=None, use_mmff=False):
    """Look up for bond types and construct a Rdkit Mol"""
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    if use_mmff:
        try:
            AllChem.MMFFOptimizeMolecule(rdmol, confId=0, maxIters=200)
            positions = rdmol.GetConformer(0).GetPositions()
        except:
            print('MMFF failed, use original coordinates.')

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    # atoms
    # convert to RDKit Mol, add atom first
    mol = Chem.RWMol()
    for atom in atoms:
        a = Chem.Atom(atom)
        mol.AddAtom(a)

    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atoms[i], atoms[j]
            pair = sorted([atoms[i], atoms[j]])
            if 'QM9' in dataset_name:
                order = get_bond_order(atom1, atom2, dist)
                
            elif 'Geom' in dataset_name:
                order = geom_predictor((pair[0], pair[1]), dist)
            else:
                raise ValueError('Fail to get dataset bond info.')
            nr_bonds[i] += order
            nr_bonds[j] += order
            # add bond to RDKIT Mol
            if order > 0:
                mol.AddBond(i, j, bond_list[order])
    
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atoms, nr_bonds):
        possible_bonds = allowed_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_type_i, nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x), mol


def get_3D_edm_metric(predict_mols, train_mols=None, dataset_name='QM9', use_mmff=False):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    rd_mols = []
    for mol in tqdm(predict_mols):
        pos = mol.GetConformer(0).GetPositions()
        pos = pos - pos.mean(axis=0)
        atom_type = [atom.GetSymbol() for atom in mol.GetAtoms()]
        try:
            validity_res = check_3D_stability(pos, atom_type, dataset_name, rdmol=mol, use_mmff=use_mmff, debug=False)
        except:
            print('Check stability failed.')
            validity_res = [0, 0, mol.GetNumAtoms(), mol]

        molecule_stable += int(validity_res[0])
        nr_stable_bonds += int(validity_res[1])
        n_atoms += int(validity_res[2])
        rd_mols.append(validity_res[3])

    # Stability
    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    # Basic rdkit metric result (Validity, Fragment, Unique)
    rdkit_dict = eval_rdmol(rd_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict, rd_mols


def get_3D_edm_metric_batch(predict_mols, train_mols=None, dataset_name='QM9'):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    rd_mols = []
    predict_mols = [predict_mols[i:i+10] for i in range(0, len(predict_mols), 10)]
    for mol_list in tqdm(predict_mols):
        validity_res_list = []
        # sanity check
        smiles = [Chem.MolToSmiles(mol) for mol in mol_list]
        assert len(set(smiles)) == 1

        for mol in mol_list:
            pos = mol.GetConformer(0).GetPositions()
            pos = pos - pos.mean(axis=0)
            atom_type = [atom.GetSymbol() for atom in mol.GetAtoms()]
            validity_res = check_3D_stability(pos, atom_type, dataset_name, rdmol=mol)
            validity_res_list.append(validity_res)
        max_validity_res = max(validity_res_list, key=lambda x: x[0])
        molecule_stable += int(max_validity_res[0])
        nr_stable_bonds += int(max_validity_res[1])
        n_atoms += int(max_validity_res[2])
        rd_mols.append(max_validity_res[3])

    # Stability
    fraction_mol_stable = molecule_stable / float(len(predict_mols))
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    output_dict = {
        'mol_stable': fraction_mol_stable,
        'atom_stable': fraction_atm_stable,
    }

    # Basic rdkit metric result (Validity, Fragment, Unique)
    rdkit_dict = eval_rdmol(rd_mols, train_smiles)
    output_dict.update(rdkit_dict)
    return output_dict

def get_moses_metrics(test_mols, n_jobs=1, device='cpu', batch_size=2000, ptest_pool=None, cache_path=None):
    # compute intermediate statistics for test rdmols
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            ptest = pickle.load(f)
    else:
        ptest = compute_intermediate_statistics(test_mols, n_jobs=n_jobs, device=device,
                                                batch_size=batch_size, pool=ptest_pool)
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump(ptest, f)

    def moses_metrics(gen_mols, pool=None):
        metrics = {}
        if pool is None:
            if n_jobs != 1:
                pool = Pool(n_jobs)
                close_pool = True
            else:
                pool = 1
                close_pool = False
        kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        gen_smiles = mapper(pool)(get_smiles, gen_mols)
        gen_smiles = list(set(gen_smiles) - {None})
        re_mols = mapper(pool)(reconstruct_mol, gen_smiles)
        metrics['FCD'] = FCDMetric(**kwargs_fcd)(gen=gen_smiles, pref=ptest['FCD'])
        metrics['SNN'] = SNNMetric(**kwargs)(gen=re_mols, pref=ptest['SNN'])
        metrics['Frag'] = FragMetric(**kwargs)(gen=re_mols, pref=ptest['Frag'])
        metrics['Scaf'] = ScafMetric(**kwargs)(gen=re_mols, pref=ptest['Scaf'])
        metrics['IntDiv'] = internal_diversity(re_mols, pool, device=device)
        metrics['Filters'] = fraction_passes_filters(re_mols, pool)

        # drug properties
        metrics['QED'] = MeanProperty(re_mols, QED, n_jobs)
        metrics['SA'] = MeanProperty(re_mols, SA, n_jobs)
        metrics['logP'] = MeanProperty(re_mols, logP, n_jobs)
        metrics['weight'] = MeanProperty(re_mols, weight, n_jobs)

        if close_pool:
            pool.close()
            pool.join()
        return metrics

    return moses_metrics


def get_sub_geometry_metric(test_mols, dataset_info, root_path):
    tar_geo_stat = load_target_geometry(test_mols, dataset_info, root_path)

    def sub_geometry_metric(gen_mols):
        bond_length_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_bond_distance,
                                           dataset_info['top_bond_sym'], mean_name='bond_length_mean')
        bond_angle_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_bond_angle,
                                          dataset_info['top_angle_sym'], mean_name='bond_angle_mean')
        dihedral_angle_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_dihedral_angle,
                                              dataset_info['top_dihedral_sym'], mean_name='dihedral_angle_mean')
        metric = {**bond_length_dict, **bond_angle_dict, **dihedral_angle_dict}

        return metric

    return sub_geometry_metric




