#!/bin/sh

# Models:  
# BERT (bert-base-uncased)
# BERT-small (prajjwal1/bert-small)
# BERT-tiny (prajjwal1/bert-tiny)
# DistilBERT (distilbert-base-uncased)
#
# Tasks:
# sst2
#
# Example usage: ./finetune.sh bert-base-uncased sst2 `lr`
# Example output dir: models/bert-base-uncased_sst2

python3 run_glue.py \
    --model_name_or_path $1 \
    --task_name $2 \
    --output_dir models/$1_$2/ \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate $3 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --seed 100 \
    --max_seq_length 128