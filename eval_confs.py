import argparse
import torch
import pickle
import warnings
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from model.diffusion_pl import DiffussionPL, conformer_evaluation_V2
from data_provider.diffusion_data_module import QM9TorDFDataModule, GeomDrugsTorDFDataModule
from rdkit import RDLogger

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
RDLogger.DisableLog('rdApp.*')
# warnings.filterwarnings(action='ignore')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



def main(args):
    L.seed_everything(args.seed)
    # tokenizer
    tokenizer = DiffussionPL.init_tokenizer(args)
    print('before inserting total tokens:', len(tokenizer))

    if args.dataset == 'QM9-df':
        dm = QM9TorDFDataModule(args.root, args.num_workers, args.batch_size, tokenizer, True, args)
    elif args.dataset == 'Geom-drugs-df':
        dm = GeomDrugsTorDFDataModule(args.root, args.num_workers, args.batch_size, tokenizer, True, args)
    else:
        raise NotImplementedError(f"dataset {args.dataset} not implemented")
    with open(args.input, 'rb') as f:
        predict_mols = pickle.load(f)[0]
    metrics = conformer_evaluation_V2(predict_mols, dm.test_dataset.gt_conf_list, threshold=dm.test_dataset.threshold, num_failures=dm.test_dataset.num_failures, logger=None, num_process=20, dataset_name=args.dataset)

    print('\n--------------------------')
    for metric in ['recall_coverage_mean', 'recall_coverage_median', 'recall_amr_mean', 'recall_amr_median', 'precision_coverage_mean', 'precision_coverage_median', 'precision_amr_mean', 'precision_amr_median']:
        print(metric, metrics[metric])
    print('--------------------------')
    for metric in ['MolStable', 'AtomStable', 'Validity', 'Unique', 'Novelty', 'Complete']:
        print(f"{metric}_3D", metrics[f"{metric}_3D"])
    print('--------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser = DiffussionPL.add_model_specific_args(parser)
    parser.add_argument('--dataset', type=str, default='QM9-df')
    parser.add_argument('--world_size', type=int, default=1)
    parser = QM9TorDFDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

