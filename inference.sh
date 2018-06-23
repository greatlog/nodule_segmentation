#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python main.py \
    --output_dir /data/model_weights/luozx/nodule_seg/experiment_nodule_seg/test1 \
    --summary_dir /data/model_weights/luozx/nodule_seg/experiment_nodule_seg/test1/log \
    --mode inference \
    --pre_trained_model True \
    --checkpoint /data/model_weights/luozx/nodule_seg/experiment_nodule_seg/step1/model-15000 \
    --size 512 \
    --batch_size 4 \
    --train_dir /data/model_weights/luozx/ \
    --is_training False \
    --epsilon 0.001 \
    --b_momentum 0.99 \
    --learning_rate 1e-4 \
    --decay_steps 40000 \
    --decay_rate 0.9 \
    --stair False \
    --beta 0.9 \
    --display_freq 100 \
    --max_iter 10000000 \
    --save_freq 1000