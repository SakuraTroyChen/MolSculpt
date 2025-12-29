ROOT_PATH=.../path-to-MolSculpt
DATA_ROOT=.../path-to-QM92014
file_name="conditional_eval_${condition}_your_setting"

condition="mu"; # 'mu', 'alpha', 'homo', 'lumo', 'gap', 'Cv'
python ${ROOT_PATH}/llm_train.py \
    --dataset "QM9" \
    --root ${DATA_ROOT} \
    --llm_tune full  \
    --filename "qm9_llm_conditional_${condition}" \
    --generate_eval_epoch 5 \
    --llm_model "acharkq/MoLlama" \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 5 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --max_epochs 100 \
    --condition_property "${condition}"



python ${ROOT_PATH}/train_lm_conf.py \
    --dataset "QM9-df" \
    --root ${DATA_ROOT} \
    --batch_size 512 \
    --infer_batch_size 512 \
    --num_workers 4 \
    --save_every_n_epochs 50  \
    --max_epochs 1000 \
    --filename ${file_name} \
    --llm_model acharkq/MoLlama \
    --check_val_every_n_epoch 200 \
    --conform_eval_epoch 100 \
    --generate_eval_epoch 50  \
    --dropout 0 \
    --num_beams 5 \
    --sampling_steps 100  \
    --accumulate_grad_batches 2 \
    --use_llm \
    --llm_ckpt .../path-to-llm-ckpt/qm9_llm_conditional_${condition}/epoch=99.ckpt \
    --eval_smiles_path .../path-to-llm-ckpt/qm9_llm_conditional_${condition}/lightning_logs/version_0/sequences_epoch99_${condition}.txt \
    --mode train \
    --condition_property "${condition}" \
    --llm_cond \
    --use_meta_projector \
    --num_metaqueries 64 \
    --num_meta_hidden_layers 12 \
    --max_n_nodes 32 \
    --meta2nodes_proj_type linear \
    --use_flash_attention 