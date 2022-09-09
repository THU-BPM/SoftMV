#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

datasetname="xnli"
outputpath="."


declare -a list_of_num_shots=(4)
# declare -a list_of_seeds=(1 2 3 4 5)
declare -a list_of_seeds=(1)

for ((which_seed=0;which_seed<${#list_of_seeds[@]};++which_seed)); do
    for ((which_num_shots=0;which_num_shots<${#list_of_num_shots[@]};++which_num_shots)); do
        python cli.py \
        --pattern_ids 2 \
        --pet_repetitions 1 \
        --overwrite_output_dir \
        --prompt_encoder_type lstm \
        --prompt_length 4 \
        --embed_size 768 \
        --hidden_size 768 \
        --pet_gradient_accumulation_steps 1 \
        --data_dir none \
        --model_type xlm-roberta \
        --model_name_or_path xlm-roberta-base \
        --task_name $datasetname \
        --output_dir $outputpath/$datasetname/seed_${list_of_seeds[which_seed]}-SHOTS_${list_of_num_shots[which_num_shots]} \
        --do_train \
        --pet_per_gpu_train_batch_size 24 \
        --pet_num_train_epochs 50 \
        --pet_per_gpu_eval_batch_size 24 \
        --learning_rate 1e-5 \
        --pattern_lang en \
        --do_eval \
        --pet_max_seq_length 256 \
        --seed ${list_of_seeds[which_seed]} \
        --num_shots ${list_of_num_shots[which_num_shots]} \
        # --do_ddp \
        # --num_ranks 3
    done
done