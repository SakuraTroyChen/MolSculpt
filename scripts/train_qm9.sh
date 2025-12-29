ROOT_PATH=.../path-to-MolSculpt
DATA_ROOT=.../path-to-QM92014

CUDA_VISIBLE_DEVICES="4,5,6,7" python ${ROOT_PATH}/llm_train.py  \
    --dataset "QM9" \
    --root ${DATA_ROOT} \
    --llm_tune full  \
    --filename '512bs_qm9_llm' \
    --generate_eval_epoch 5 \
    --llm_model "acharkq/MoLlama" \
    --rand_smiles restricted \
    --temperature 1.0 \
    --num_beams 1 \
    --sample_num 10000 \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --max_epochs 100

python ${ROOT_PATH}/train_uncond_gene.py \
    --dataset "QM9-jodo" \
    --root ${DATA_ROOT} \
    --filename "meta64_layer24_qm9_denovo_gen" \
    --llm_model "acharkq/MoLlama" \
    --num_workers 4 \
    --batch_size 128 \
    --max_epochs 1000 \
    --save_every_n_epochs 100 \
    --check_val_every_n_epoch 20 \
    --accumulate_grad_batches 2 \
    --dropout 0.05 \
    --test_conform_epoch 100000 \
    --conform_eval_epoch 100000 \
    --cache_epoch 2 \
    --generate_eval_epoch 100 \
    --eval_smiles_path .../path-to-your-generated-sequences.txt \
    --mode train \
    --use_llm \
    --llm_cond \
    --use_meta_projector \
    --num_metaqueries 64 \
    --num_meta_hidden_layers 24 \
    --max_n_nodes 32 \
    --meta2nodes_proj_type linear \
    --use_flash_attention