dataset_name="qm9"
type="3D"
dir_path=".../path-to-JODO/JODO"
root_path=${dir_path}/datasets/data/
file_path=".../path-to-your-predictions.pt"

python ${dir_path}/eval_rdkit_pkl.py \
    --file_path $file_path \
    --dataset_name $dataset_name \
    --type $type \
    --sub_geometry True \
    --root_path $root_path \
