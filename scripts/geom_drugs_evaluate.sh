file_path=".../path-to-your-predictions.pt"
eval_mode="3dmol"
llm_model="acharkq/MoLlama"
dataset="GeomDrugs-JODO"
num_metaqueries=0
ROOT_PATH=.../path-to-MolSculpt
DATA_ROOT=.../path-to-GeomDrugs-JODO

python ${ROOT_PATH}/evaluate.py \
    --path $file_path \
    --mode $eval_mode \
    --llm_model $llm_model \
    --dataset $dataset \
    --num_metaqueries $num_metaqueries \
    --root ${DATA_ROOT}
