#!/usr/bin/env bash
HOME_DIR="/data/vsanguineti/"
ROOT_DIR="/data/vsanguineti/tfrecords/lists/"

EXP=(Acoustic12820epochs_ Audio12820epochs_ )
MODEL=(DualCamHybridNet HearNet)
MODEL1=( ResNet18_v1  ResNet18_v1)
ST=(1 2 3 4 5)
P[0]=0
P[1]=0
i=0

for t in "${ST[@]}"
do
    for ((a = 0 ; a < ${#EXP[@]} ; a++))
    do
        CUDA_VISIBLE_DEVICES=$i python main.py --mode train --model_1 ${MODEL1[$a]} --model_2 ${MODEL[$a]} --train_file  $ROOT_DIR"/training.txt" --valid_file $ROOT_DIR"/validation.txt" --test_file $ROOT_DIR"/testing.txt"  --exp_name ${EXP[$a]}$t --batch_size 64 --total_length 2 --number_of_crops 1 --sample_length 2 --buffer_size 10 --log_dir $HOME_DIR"/tensorboard/" --checkpoint_dir $HOME_DIR"/checkpoints/" --num_epochs 20 --learning_rate 0.00001 --embedding 1 --temporal_pooling 1 --margin 0.2 &
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