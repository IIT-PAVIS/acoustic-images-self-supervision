#!/usr/bin/env bash

cd ..
#HOME_DIR="/media/vsanguineti/data/"

#audio

#EXP=Audio12820epochs_
#MODEL=HearNet
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#    for CHECKPOINT in random 9 19
#    do
#    CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'training' \
#    --init_checkpoint $HOME_DIR/checkpoints/$EXP$t/model_$CHECKPOINT.ckpt --num_classes 128 --embedding 1
#    CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'validation' \
#    --init_checkpoint $HOME_DIR/checkpoints/$EXP$t/model_$CHECKPOINT.ckpt --num_classes 128 --embedding 1
#    CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'testing' \
#    --init_checkpoint $HOME_DIR/checkpoints/$EXP$t/model_$CHECKPOINT.ckpt --num_classes 128 --embedding 1
#    done
#done
#
##transferring
#
#ST=5
#NAME=transfer
#EXP=(0.1_ 0.3_ 0.5_ 0.7_ 0.9_)
#MODEL=HearNet
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#    for CHECKPOINT in random 9 19
#    do
#        CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'training' \
#        --init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --embedding 1
#        CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'validation' \
#        --init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --embedding 1
#        CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'testing' \
#        --init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --embedding 1
#    done
#done

#one model to compute TUT features

#EXP=HearNet_layer
#MODEL=HearNet
#CHECKPOINT=72
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file  'training' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 10 --embedding 0
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'validation'  \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 10 --embedding 0
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'testing' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 10 --embedding 0


HOME_DIR=/home/vsanguineti/

EXP=(distilation_ )
MODEL=HearNet
ST=(2)

for ((i = 0 ; i < ${#EXP[@]} ; i++))
do
    for t in "${ST[@]}"
    do
        CUDA_VISIBLE_DEVICES=0 python3 computeTUTfeatures.py --model $MODEL --train_file  'training' \
        --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt --num_classes 10 --embedding 0
        CUDA_VISIBLE_DEVICES=0 python3 computeTUTfeatures.py --model $MODEL --train_file 'validation'  \
        --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt --num_classes 10 --embedding 0
        CUDA_VISIBLE_DEVICES=0 python3 computeTUTfeatures.py --model $MODEL --train_file 'testing' \
        --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_best.ckpt --num_classes 10 --embedding 0
    done
done

EXP=(distilation_ )
MODEL=(HearNet )
ST=(2)

for ((i = 0 ; i < ${#EXP[@]} ; i++))
do
    for t in "${ST[@]}"
    do

            python3 knn.py  $MODEL $HOME_DIR/checkpoints/$EXP$t/model_best.ckpt 1 testing


    done
done

#EXP=HearNet_selfsupervised2
#MODEL=HearNet
#CHECKPOINT=68
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file  'training' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 14 --embedding 0
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'validation' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 14 --embedding 0
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'testing' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 14 --embedding 0
#
#
#EXP=HearNet_actions
#MODEL=HearNet
#CHECKPOINT=84
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'training'  \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 14 --embedding 0
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file 'validation' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 14 --embedding 0
#CUDA_VISIBLE_DEVICES=0 python computeTUTfeatures.py --model $MODEL --train_file  'testing' \
#--init_checkpoint $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt --num_classes 14 --embedding 0




