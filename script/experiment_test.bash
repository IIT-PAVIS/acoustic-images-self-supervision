#!/usr/bin/env bash
HOME_DIR="/home/vsanguineti/Documents/Code/audio-video/"
ROOT_DIR="/data/vsanguineti/tfrecords/lists/"

EXP_NAME[0]="ResNetframetfrecords_restore2"
MODEL_NAME[0]="ResNet50"

EXP_NAME[1]="DualCamtfrecords"
MODEL_NAME[1]="DualCamHybridNet"

EXP_NAME[2]="HearNettfrecords"
MODEL_NAME[2]="HearNet"

LEN=${#EXP_NAME[@]}

for (( i=0; i<$LEN; i++ ))
do
	MODEL_PATH=$HOME_DIR"/checkpoints/"${EXP_NAME[$i]}
	BEST_EPOCH=$(grep "Epoch" $MODEL_PATH"/model.txt" | cut -d ':' -f 2 | tr -d ' ')
	EPOCH_FILE=$MODEL_PATH"/epoch_"$BEST_EPOCH".ckpt"

	CUDA_VISIBLE_DEVICES=$((i % 2)) python main.py --mode test --model ${MODEL_NAME[$i]} --test_file $ROOT_DIR"/testing.txt" --exp_name ${EXP_NAME[$i]} --batch_size 32 --total_length 2 --number_of_crops 1 --sample_length 2 --buffer_size 10 --log_dir $HOME_DIR"/tensorboard/" --checkpoint_dir $HOME_DIR"/checkpoints/" --embedding 0 --temporal_pooling 1 --num_class 10 --restore_checkpoint $EPOCH_FILE &
done

#CUDA_VISIBLE_DEVICES=0 python main.py --mode test --model DualCamHybridNet --test_file $ROOT_DIR"/testing.txt" --exp_name $EXP_NAME --batch_size 32 --total_length 2 --number_of_crops 1 --sample_length 2 --buffer_size 10 --log_dir $HOME_DIR"/tensorboard/" --checkpoint_dir $HOME_DIR"/checkpoints/" --embedding 0 --temporal_pooling 1 --num_class 10 --restore_checkpoint $HOME_DIR"/checkpoints/"$EXP_NAME"/epoch_87.ckpt" &
#EXP_NAME="HearNettfrecords"
#CUDA_VISIBLE_DEVICES=1 python main.py --mode test --model HearNet --test_file $ROOT_DIR"/testing.txt" --exp_name $EXP_NAME --batch_size 32 --total_length 2 --number_of_crops 1 --sample_length 2 --buffer_size 10 --log_dir $HOME_DIR"/tensorboard/" --checkpoint_dir $HOME_DIR"/checkpoints/" --embedding 0 --temporal_pooling 0 --num_class 10 --restore_checkpoint $HOME_DIR"/checkpoints/"$EXP_NAME"/epoch_85.ckpt" &
