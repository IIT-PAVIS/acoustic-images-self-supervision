#!/usr/bin/env bash

HOME_DIR="/home/vsanguineti/Documents/Code/audio-video/"
ROOT_DIR="/data/vsanguineti/tfrecords/lists/"

ST=(5)
RESTORE=(14)
ALPHA=( 0.1 0.3 0.5 0.7 0.9)

P[0]=0
P[1]=0
i=0

for ((a = 0 ; a < ${#ST[@]} ; a++))
    do
    for t in "${ALPHA[@]}"
    do
        CUDA_VISIBLE_DEVICES=$i python main.py --mode train --model_1 ResNet18_v1 --model_2 HearNet --train_file $ROOT_DIR"/training.txt" --valid_file $ROOT_DIR"/validation.txt" --test_file $ROOT_DIR"/testing.txt" --exp_name "transfer"$t"_"${ST[$a]}  --batch_size 64 --total_length 2 --number_of_crops 1 --sample_length 2 --buffer_size 10 --log_dir $HOME_DIR"/tensorboard/" --checkpoint_dir $HOME_DIR"/checkpoints/" --num_epochs 20 --learning_rate 0.00001 --embedding 1 --temporal_pooling 1 --margin 0.2 --transfer 1 --alpha $t --restore_checkpoint $HOME_DIR"/checkpoints/Acoustic12820epochs_"${ST[$a]}"/model_"${RESTORE[$a]}".ckpt" &
        #echo "gpu: $i" &
        P[$i]=$!
        #echo "pid: ${P[$i]}"
        if [ $i -eq 1 ]
        then
            wait ${P[0]}
            wait ${P[1]}
        fi
        (( i = (i+1) % 2 ))
    done
done