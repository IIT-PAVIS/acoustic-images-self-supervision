#!/usr/bin/env bash

HOME_DIR=/data/
#audio

EXP=(Audio12820epochs_ Acoustic12820epochs_ )
MODEL=(HearNet DualCamHybridNet)
MODEL1=(ResNet18_v1 ResNet18_v1)
ST=(1 2 3 4 5)
for t in ${ST[@]}
do
    for ((i = 0 ; i < ${#EXP[@]} ; i++))
    do
        for CHECKPOINT in 19
        do

            CUDA_VISIBLE_DEVICES=1 python3 testing2.py --model1 ${MODEL[$i]} --model2  ${MODEL1[$i]} --number_of_crops 1 \
    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128  --nr_frames 2 \
    --sample_length 2 --mode training

            CUDA_VISIBLE_DEVICES=1 python3 testing2.py --model1 ${MODEL[$i]} --model2 ${MODEL1[$i]} --number_of_crops 1 \
    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
    --sample_length 2 --mode testing

            CUDA_VISIBLE_DEVICES=1 python3 testing2.py --model1 ${MODEL[$i]} --model2 ${MODEL1[$i]}  --number_of_crops 1 \
    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
    --sample_length 2 --mode validation

        done

    done
done

#EXP=(Audio12820epochs_ Acoustic12820epochs_ )
#MODEL=(HearNet DualCamHybridNet)
#MODEL1=(ResNet18_v1 ResNet18_v1)
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#    for ((i = 0 ; i < ${#EXP[@]} ; i++))
#    do
#        for CHECKPOINT in random 0 2 4 6 9 14 19
#        do
#
#            CUDA_VISIBLE_DEVICES=1 python extract_features_product.py --model1 ${MODEL[$i]} --model2  ${MODEL1[$i]} --number_of_crops 1 \
#    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128  --nr_frames 2 \
#    --sample_length 2 --mode training --epoch "$CHECKPOINT"
#
#            CUDA_VISIBLE_DEVICES=1 python extract_features_product.py --model1 ${MODEL[$i]} --model2 ${MODEL1[$i]} --number_of_crops 1 \
#    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#    --sample_length 2 --mode testing --epoch "$CHECKPOINT"
#
#            CUDA_VISIBLE_DEVICES=1 python extract_features_product.py --model1 ${MODEL[$i]} --model2 ${MODEL1[$i]}  --number_of_crops 1 \
#    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#    --sample_length 2 --mode validation --epoch "$CHECKPOINT"
#
#        done
#
#    done
#done

#EXP=(Audio12820epochs_2_ Acoustic12820epochs_2_ )
#MODEL=(HearNet DualCamHybridNet )
#MODEL1=(ResNet18_v1 ResNet18_v1 1)
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#    for ((i = 0 ; i < ${#EXP[@]} ; i++))
#    do
#        for CHECKPOINT in random 0 2 4 6 9 14 19
#        do
#
#            CUDA_VISIBLE_DEVICES=1 python extract_features_product_old.py --model1 ${MODEL[$i]} --model2  ${MODEL1[$i]} --number_of_crops 1 \
#    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128  --nr_frames 2 \
#    --sample_length 2 --mode training --epoch "$CHECKPOINT"
#
#            CUDA_VISIBLE_DEVICES=1 python extract_features_product_old.py --model1 ${MODEL[$i]} --model2 ${MODEL1[$i]} --number_of_crops 1 \
#    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#    --sample_length 2 --mode testing --epoch "$CHECKPOINT"
#
#            CUDA_VISIBLE_DEVICES=1 python extract_features_product_old.py --model1 ${MODEL[$i]} --model2 ${MODEL1[$i]}  --number_of_crops 1 \
#    --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#    --sample_length 2 --mode validation --epoch "$CHECKPOINT"
#
#        done
#
#    done
#done


#transferring

#NAME=transfer
#ST=5
#EXP=(0.1_ 0.3_ 0.5_ 0.7_ 0.9_)
#MODEL=HearNet
#MODEL1=ResNet18_v1
#
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#	for CHECKPOINT in random 0 2 4 6 9 14 19
#	do
#
#	    CUDA_VISIBLE_DEVICES=1 python extract_features_product.py --model1 $MODEL --model2  $MODEL1 --number_of_crops 1 \
#--init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128  --nr_frames 2 \
#--sample_length 2 --mode training --epoch "$CHECKPOINT"
#
#        CUDA_VISIBLE_DEVICES=1 python extract_features_product.py --model1 $MODEL --model2 $MODEL1 --number_of_crops 1 \
#--init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#--sample_length 2 --mode testing --epoch "$CHECKPOINT"
#
#        CUDA_VISIBLE_DEVICES=1 python extract_features_product.py --model1 $MODEL --model2  $MODEL1 --number_of_crops 1 \
#--init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#--sample_length 2 --mode validation --epoch "$CHECKPOINT"
#
#	done
#
#done
#
#NAME=transfer
#EXP=(0.1_2_ 0.3_2_ 0.5_2_ 0.7_2_ 0.9_2_)
#MODEL=HearNet
#MODEL1=ResNet18_v1
#ST=5
#
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#	for CHECKPOINT in random 0 2 4 6 9 14 19
#	do
#
#	    CUDA_VISIBLE_DEVICES=1 python extract_features_product_old.py --model1 $MODEL --model2  $MODEL1 --number_of_crops 1 \
#--init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128  --nr_frames 2 \
#--sample_length 2 --mode training --epoch "$CHECKPOINT"
#
#        CUDA_VISIBLE_DEVICES=1 python extract_features_product_old.py --model1 $MODEL --model2 $MODEL1 --number_of_crops 1 \
#--init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#--sample_length 2 --mode testing --epoch "$CHECKPOINT"
#
#        CUDA_VISIBLE_DEVICES=1 python extract_features_product_old.py --model1 $MODEL --model2  $MODEL1 --number_of_crops 1 \
#--init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --num_classes 128 --nr_frames 2 \
#--sample_length 2 --mode validation --epoch "$CHECKPOINT"
#
#	done
#
#done