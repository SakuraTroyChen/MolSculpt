import argparse
import torch
from rdkit import Chem
from evaluation.eval_functions import get_2D_edm_metric, get_3D_edm_metric
from tqdm import tqdm
from transformers import AutoTokenizer
from data_provider.geom_drugs_jodo_dm import GeomDrugsJODODM
from data_provider.qm9_jodo_dm import QM9DM
import random

def test_valency(molecule):
    has_open_valencies = False
    for atom in molecule.GetAtoms():
        if atom.GetImplicitValence() > 0:
            has_open_valencies = True
            print(True)
            break
    return has_open_valencies



def eval_(args, path, mode):
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    if mode == 'smiles':
        if args.dataset == 'QM9-jodo':
            dm = QM9DM(args.root, 2, 64, tokenizer, args)
        elif args.dataset == 'GeomDrugs-JODO':
            dm = GeomDrugsJODODM(args.root, 2, 64, tokenizer, args)
        else:
            raise NotImplementedError(f"dataset {args.dataset} not implemented")
        smiles = load_smiles(path)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        edm2d_dout = get_2D_edm_metric(mols, dm.train_rdmols)

        properties = ['atom_stable', 'mol_stable', 'Complete', 'Unique', 'Novelty']
        print(properties)
        print('\t'.join([str(edm2d_dout[prop]) for prop in properties]))

        print(edm2d_dout)

        moses_out = dm.get_moses_metrics(mols)
        print(moses_out)

        properties = ['SNN', 'Frag', 'Scaf', 'FCD']
        print(properties)
        print('\t'.join([str(moses_out[prop]) for prop in properties]))
    elif mode == '3dmol':
        if path == "train":
            mols = random.sample(dm.train_rdmols, 10000)
        else:
            mols = torch.load(path)

        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        if args.dataset == 'QM9-jodo':
            dm = QM9DM(args.root, 2, 64, tokenizer, args)
            dataset_name = 'QM9'
        elif args.dataset == 'GeomDrugs-JODO':
            dm = GeomDrugsJODODM(args.root, 2, 64, tokenizer, args)
            dataset_name = 'GeomDrugs'
        else:
            raise NotImplementedError(f"dataset {args.dataset} not implemented")

        # if False:
            ## temporary fix on 3d coordinates
            # for mol in tqdm(mols):
            #     conf = mol.GetConformer().GetPositions()
            #     conf = conf / 2.3860 * 2.4777
            #     mol.RemoveAllConformers()
            #     num_atoms = mol.GetNumAtoms()
            #     new_conformer = Chem.Conformer(num_atoms)
            #     for i in range(num_atoms):
            #         new_conformer.SetAtomPosition(i, conf[i].tolist())
            #     mol.AddConformer(new_conformer, assignId=True)


        # print("######## 2D EDM metric ########")
        # edm2d_dout = get_2D_edm_metric(mols, dm.train_rdmols)
        # print(edm2d_dout)
        # moses_out = dm.get_moses_metrics(mols)
        # print(moses_out)
        
        print("######## 3D EDM metric ########")
        eval_results_3d, reconstructed_3d_mols = get_3D_edm_metric(mols, train_mols=dm.train_rdmols, dataset_name=dataset_name)
        print(eval_results_3d)
        eval_results_moses = dm.get_moses_metrics(reconstructed_3d_mols)
        print(eval_results_moses)
        sub_geometry_metric = dm.get_sub_geometry_metric(mols)
        print(sub_geometry_metric)
        print(f"bond_length_mean: {sub_geometry_metric.get('bond_length_mean', None)}")
        print(f"bond_angle_mean: {sub_geometry_metric.get('bond_angle_mean', None)}")
        print(f"dihedral_angle_mean: {sub_geometry_metric.get('dihedral_angle_mean', None)}")

def load_smiles(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        sampled_sequences = [line.strip().split() for line in lines]
        _, _, smiles = zip(*sampled_sequences)
    return smiles


if __name__ == '__main__':
    # from data_provider.conf_gen_cal_metrics import get_best_rmsd
    # from data_provider.dataset_config import get_dataset_info
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the file', default=None)
    parser.add_argument('--mode', type=str, help='Mode for evaluation', default='smiles')
    parser.add_argument('--llm_model', type=str, help='Path to the llm model', default='all_checkpoints/mollama')
    # parser.add_argument('--root', type=str, help='Path to the llm model', default=None)
    parser.add_argument('--dataset', type=str, help='Path to the llm model', default='GeomDrugs-JODO')
    parser.add_argument('--num_metaqueries', type=int, default=0)
    QM9DM.add_model_specific_args(parser)
    args = parser.parse_args()
    eval_(args, args.path, args.mode)

