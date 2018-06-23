#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main_2.py \
    --output_dir /data/model_weights/luozx/experiment_nodule_seg/step8 \
    --summary_dir /data/model_weights/luozx/experiment_nodule_seg/step8/log \
    --mode train \
    --pre_trained_model False \
    --checkpoint /data/model_weights/luozx/experiment_nodule_seg/step7/model-1 \
    --size 512 \
    --batch_size 2 \
    --train_dir /data/model_weights/luozx/ \
    --is_training True \
    --epsilon 0.001 \
    --b_momentum 0.99 \
    --learning_rate 1e-6 \
    --decay_steps 4000 \
    --decay_rate 0.9 \
    --stair False \
    --beta 0.9 \
    --display_freq 10 \
    --max_iter 1000000 \
    --save_freq 100

