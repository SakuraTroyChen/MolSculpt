import argparse
import pandas as pd
from pathlib import Path

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
pd.options.display.width = 1000
# pd.set_option('display.width', 1000) 

def print_std(accs, stds, categories, append_mean=False):
    category_line = ' '.join(categories)
    if append_mean:
        category_line += ' Mean'
    
    line = ''
    if stds is None:
        for acc in accs:
            line += '{:0.1f} '.format(acc)
    else:
        for acc, std in zip(accs, stds):
            line += '{:0.1f}±{:0.1f} '.format(acc, std)
    
    if append_mean:
        line += '{:0.1f}'.format(sum(accs) / len(accs))
    print(category_line)
    print(line)



def read_log(df, args):
    # df = df.round(2)
    if 'train_coord_loss' in df.columns:
        train_cols = ['epoch', 'step', 'train_loss', 'train_lm_loss', 'train_distance_loss', 'train_coord_loss']
        train_log = df[~df['train_coord_loss'].isnull()][train_cols]
        train_log.set_index('step', inplace=True)
        print(train_log.head(200))
    
    if 'val_coord_loss' in df.columns:
        val_cols = ['epoch', 'step', 'val_loss', 'val_lm_loss', 'val_distance_loss', 'val_coord_loss']
        val_log = df[~df['val_coord_loss'].isnull()][val_cols]
        val_log.set_index('step', inplace=True)
        print(val_log.to_csv(sep='\t', index=False))

    if 'cov_mean' in df.columns:
        conf_cols = ['epoch', 'step', 'cov_mean', 'cov_median', 'mat_mean', 'mat_median']
        
        df[['cov_mean', 'cov_median']] = (df[['cov_mean', 'cov_median']] * 100).round(2)
        df[['mat_mean', 'mat_median']] = df[['mat_mean', 'mat_median']].round(4)
        cov_log = df[~df['cov_mean'].isnull()][conf_cols]
        cov_log.set_index('step', inplace=True)
        print(cov_log.to_csv(sep='\t', index=False))
    
    if 'valid/MolStable_3D_unimol' in df.columns:
        sta_cols = ['epoch', 'val_loss', 'val_coordinate_loss']
        sta_log1 = df[~df['valid/MolStable_3D_unimol'].isnull()][sta_cols].round(3)
        sta_log1.set_index('epoch', inplace=True)
        print(sta_log1.to_csv(sep='\t', index=False))

        
        sta_cols = ['epoch', 'valid/AtomStable_3D_unimol', 'valid/MolStable_3D_unimol', 'valid/Complete_3D_unimol', 'valid/Unique_3D_unimol',  'valid/Novelty_3D_unimol', 'valid/cov_mean', 'valid/cov_median', 'valid/mat_mean', 'valid/mat_median']
        
        sta_log2 = df[~df['valid/MolStable_3D_unimol'].isnull()][sta_cols].round(3)
        sta_log2.set_index('epoch', inplace=True)
        print(sta_log2.to_csv(sep='\t', index=False))

    df = df.rename(columns={'val_coordinate_loss/dataloader_idx_0': 'val_coordinate_loss'})
    df = df.rename(columns={'val_loss/dataloader_idx_0': 'val_loss'})
    df = df.rename(columns={'val_diff_loss/dataloader_idx_0': 'val_coordinate_loss'})
    
    if 'val_loss' in df.columns:
        sta_cols = ['epoch', 'val_loss', 'val_coordinate_loss']
        sta_log1 = df[~df['val_loss'].isnull()][sta_cols].round(3)
        # sta_log1.set_index('epoch', inplace=True)
        print(sta_log1.to_csv(sep='\t', index=False))

    if 'valid/MolStable_3D' in df.columns:
        sta_cols = ['epoch', 'valid/AtomStable_3D', 'valid/MolStable_3D', 'valid/Complete_3D', 'valid/Unique_3D',  'valid/Novelty_3D', 'valid/cov_mean', 'valid/cov_median', 'valid/mat_mean', 'valid/mat_median']
        
        sta_log2 = df[~df['valid/MolStable_3D'].isnull()][sta_cols].round(3)
        # sta_log2.set_index('epoch', inplace=True)
        print(sta_log2.to_csv(sep='\t', index=False))

    if 'test/AtomStable_3D' in df.columns:
        sta_cols = ['epoch', 'test/AtomStable_3D', 'test/MolStable_3D', 'test/Complete_3D', 'test/Unique_3D',  'test/Novelty_3D']
        
        sta_log = df[~df['test/AtomStable_3D'].isnull()][sta_cols].round(3)
        # sta_log.set_index('epoch', inplace=True)
        print(sta_log.to_csv(sep='\t', index=False))

        sta_cols = ['epoch', 'test/recall_coverage_mean','test/recall_coverage_median', 'test/recall_amr_mean','test/recall_amr_median', 'test/precision_coverage_mean','test/precision_coverage_median', 'test/precision_amr_mean','test/precision_amr_median',]
        sta_log = df[~df['test/AtomStable_3D'].isnull()][sta_cols].round(3)
        # sta_log.set_index('epoch', inplace=True)
        print(sta_log.to_csv(sep='\t', index=False))

    if 'AtomStable' in df.columns:
        stab_cols = ['epoch', 'step', 'AtomStable', 'MolStable', 'Complete', 'Unique', 'Novelty', 'SNN', 'Frag', 'Scaf', 'FCD']
    
        stab_log = df[~df['AtomStable'].isnull()][stab_cols]
        stab_log.set_index('step', inplace=True)
        print(stab_log.to_csv(sep='\t', index=False))

        sta_3d_unimol_cols = ['epoch', 'step', 'AtomStable_3D_unimol', 'MolStable_3D_unimol', 'Complete_3D_unimol', 'Validity_3D_unimol', 'Novelty_3D_unimol', ]
        sta_3d_unimol_log = df[~df['AtomStable_3D_unimol'].isnull()][sta_3d_unimol_cols]
        sta_3d_unimol_log.set_index('step', inplace=True)
        print(sta_3d_unimol_log.to_csv(sep='\t', index=False))

        subgeometry_unimol_cols = ['epoch', 'step', 'bond_length_mean_unimol', 'bond_angle_mean_unimol', 'dihedral_angle_mean_unimol']
        subgeometry_unimol_log = df[~df['bond_length_mean_unimol'].isnull()][subgeometry_unimol_cols]
        subgeometry_unimol_log.set_index('step', inplace=True)
        print(subgeometry_unimol_log.to_csv(sep='\t', index=False))

        sta_3d_rdkit_cols = ['epoch', 'step', 'AtomStable_3D_rdkit', 'MolStable_3D_rdkit', 'Complete_3D_rdkit', 'Validity_3D_rdkit', 'Novelty_3D_rdkit', ]
        sta_3d_rdkit_log = df[~df['AtomStable_3D_rdkit'].isnull()][sta_3d_rdkit_cols]
        sta_3d_rdkit_log.set_index('step', inplace=True)
        print(sta_3d_rdkit_log.to_csv(sep='\t', index=False))

        
        
        subgeometry_rdkit_cols = ['epoch', 'step', 'bond_length_mean_rdkit', 'bond_angle_mean_rdkit', 'dihedral_angle_mean_rdkit']
        subgeometry_rdkit_log = df[~df['bond_length_mean_rdkit'].isnull()][subgeometry_rdkit_cols]
        subgeometry_rdkit_log.set_index('step', inplace=True)
        print(subgeometry_rdkit_log.to_csv(sep='\t', index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--tag', type=str, default='train_loss_gtm')
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--disable_rerank', action='store_true', default=False)
    parser.add_argument('--qa_question', action='store_true', default=False)
    args = parser.parse_args()
    args.path = Path(args.path)
    
    
    log_hparas = args.path / 'hparams.yaml'
    with open(log_hparas, 'r') as f:
        line = f.readline()
        file_name = line.strip().split(' ')[1]
    
    log_path = args.path / 'metrics.csv'
    log = pd.read_csv(log_path)
    
    print(f'File name: {file_name}')
    
    read_log(log, args)
    