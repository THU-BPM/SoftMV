#!/bin/bash
# A single GPU is used for 1-shot experiments. Two and three GPUs are used for 2- and # 4shot experiments. Other experiments use 6 GPUs.
# --do_ddp \
# --num_ranks 2

export CUDA_VISIBLE_DEVICES=5

datasetname="xnli"
outputpath="."

declare -a list_of_num_shots=(4)
declare -a list_of_seeds=(1)
declare -a list_of_rates=(0.1)

for ((which_seed=0;which_seed<${#list_of_seeds[@]};++which_seed)); do
    for ((which_num_shots=0;which_num_shots<${#list_of_num_shots[@]};++which_num_shots)); do
        for ((which_rate=0;which_rate<${#list_of_rates[@]};++which_rate)); do
        python pct.py \
        --pattern_ids 0 \
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
        --output_dir $outputpath/$datasetname/pct/epoch70_seed_${list_of_seeds[which_seed]}-SHOTS_${list_of_num_shots[which_num_shots]}-rate_${list_of_rates[which_rate]} \
        --do_train \
        --do_eval \
        --pet_per_gpu_train_batch_size 24\
        --pet_num_train_epochs 70 \
        --pet_per_gpu_eval_batch_size 24\
        --learning_rate 1e-5 \
        --pattern_lang en \
        --pet_max_seq_length 256 \
        --seed ${list_of_seeds[which_seed]} \
        --num_shots ${list_of_num_shots[which_num_shots]} \
        --cosda_rate ${list_of_rates[which_rate]}
        done
    done
done