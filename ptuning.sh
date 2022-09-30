#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

datasetname="xnli"
outputpath="."

declare -a list_of_num_shots=(8)
declare -a list_of_seeds=(1)
declare -a list_of_rates=(0)
declare -a list_of_lr=(1e-05)

for ((i=0;i<${#list_of_rates[@]};++i)); do
    python cli.py \
    --pattern_ids 0 \
    --overwrite_output_dir \
    --prompt_encoder_type lstm \
    --prompt_length 4 \
    --embed_size 768 \
    --hidden_size 768 \
    --model_type xlm-roberta \
    --model_name_or_path xlm-roberta-base \
    --task_name $datasetname \
    --output_dir $outputpath/$datasetname/Analysis/seed_${list_of_seeds[0]}-SHOTS_${list_of_num_shots[0]}-rate_${list_of_rates[i]}-wocode \
    --do_train \
    --do_eval \
    --prompt_per_gpu_train_batch_size 12 \
    --prompt_num_train_epochs 70 \
    --prompt_per_gpu_eval_batch_size 12 \
    --learning_rate ${list_of_lr[i]} \
    --pattern_lang en \
    --prompt_max_seq_length 256 \
    --seed ${list_of_seeds[0]} \
    --num_shots ${list_of_num_shots[0]} \
    --cosda_rate ${list_of_rates[i]} \
    --init_from_vocab
done

# for ((which_seed=0;which_seed<${#list_of_seeds[@]};++which_seed)); do
#     for ((which_num_shots=0;which_num_shots<${#list_of_num_shots[@]};++which_num_shots)); do
#         for ((which_rate=0;which_rate<${#list_of_rates[@]};++which_rate)); do
#             python cli.py \
#             --pattern_ids 0 \
#             --overwrite_output_dir \
#             --prompt_encoder_type lstm \
#             --prompt_length 4 \
#             --embed_size 768 \
#             --hidden_size 768 \
#             --model_type xlm-roberta \
#             --model_name_or_path xlm-roberta-base \
#             --task_name $datasetname \
#             --output_dir $outputpath/$datasetname/best/seed_${list_of_seeds[which_seed]}-SHOTS_${list_of_num_shots[which_num_shots]}-rate_${list_of_rates[which_rate]} \
#             --do_train \
#             --do_eval \
#             --prompt_per_gpu_train_batch_size 12\
#             --prompt_num_train_epochs 70 \
#             --prompt_per_gpu_eval_batch_size 12\
#             --learning_rate 7.20718366480073e-06 \
#             --pattern_lang en \
#             --prompt_max_seq_length 256 \
#             --seed ${list_of_seeds[which_seed]} \
#             --num_shots ${list_of_num_shots[which_num_shots]} \
#             --cosda_rate ${list_of_rates[which_rate]} \
#             --init_from_vocab 
#         done
#     done
# done

