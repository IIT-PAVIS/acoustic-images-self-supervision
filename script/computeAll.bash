#!/usr/bin/env bash

HOME_DIR=/home/vsanguineti/

EXP=(distillation_ )
MODEL=(HearNet )
ST=(1 2 3 4 5)

for ((i = 0 ; i < ${#EXP[@]} ; i++))
do
    for t in "${ST[@]}"
    do

            CUDA_VISIBLE_DEVICES=1 python3 extract_features_one.py --model ${MODEL[$i]} --temporal_pooling 1 \
    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt --num_classes 128  --nr_frames 2 \
    --sample_length 2 --train_file training  --embedding 0

            CUDA_VISIBLE_DEVICES=1 python3 extract_features_one.py  --model ${MODEL[$i]} --temporal_pooling 1 \
    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt --num_classes 128 --nr_frames 2 \
    --sample_length 2 --train_file testing  --embedding 0

            CUDA_VISIBLE_DEVICES=1 python3 extract_features_one.py  --model ${MODEL[$i]} --temporal_pooling 1 \
    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt --num_classes 128 --nr_frames 2 \
    --sample_length 2 --train_file validation --embedding 0


    done
done


EXP=(distillation_ )
MODEL=(HearNet )
ST=(1 2 3 4 5)

for ((i = 0 ; i < ${#EXP[@]} ; i++))
do
    for t in "${ST[@]}"
    do

            python3 knn.py  ${MODEL[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt 0 testing

            python3 knn.py  ${MODEL[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt 0 validation


    done
done



