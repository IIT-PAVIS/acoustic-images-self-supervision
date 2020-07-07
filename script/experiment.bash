#!/bin/bash

HOME_DIR="/home/vsanguineti/Documents/Code/audio-video/"
ROOT_DIR="/data/vsanguineti/dualcam_actions_dataset/30_seconds/lists"

EXP_NAME[0]="ResNet_oldHear"
MODEL_NAME[0]="ResNet18_v1"

EXP_NAME[1]="DualCamDataset"
MODEL_NAME[1]="DualCamHybridNet"

EXP_NAME[2]="HearNetDataset"
MODEL_NAME[2]="HearNet"

EXP_NAME[3]="ResNet_old3"
MODEL_NAME[3]="ResNet18_v1"

EXP_NAME[4]="DualCamtDatasetRestored"
MODEL_NAME[4]="DualCamHybridNet"

EXP_NAME[5]="HearNetDatasetRestored"
MODEL_NAME[5]="HearNet"

EXP_NAME[6]="ResNet_oldDual"
MODEL_NAME[6]="ResNet18_v1"

LEN=${#EXP_NAME[@]}
P[0]=0
P[1]=0
i=0
for (( a=0; a<$LEN; a++ ))
do
	MODEL_PATH=$HOME_DIR"/checkpoints/"${EXP_NAME[$a]}
	BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
	EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"

	CUDA_VISIBLE_DEVICES=$i python main.py --mode train --model ${MODEL_NAME[$a]} --train_file $ROOT_DIR"/training.txt" --valid_file $ROOT_DIR"/validation.txt" --test_file $ROOT_DIR"/testing.txt" --exp_name ${EXP_NAME[$a]}"_2" --batch_size 32 --total_length 30 --number_of_crops 15 --sample_length 2 --buffer_size 10 --log_dir $HOME_DIR"/tensorboard/" --checkpoint_dir $HOME_DIR"/checkpoints/" --num_epochs 100 --learning_rate 0.001 --embedding 0 --temporal_pooling 1 --num_class 14 --restore_checkpoint $EPOCH_FILE &
	P[$i]=$!
	if [ $i -eq 1 ]
	then
		wait ${P[0]}
		wait ${P[1]}
	fi
	(( i = (i+1) % 2 ))
done
