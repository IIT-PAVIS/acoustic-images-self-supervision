#!/usr/bin/env bash
HOME_DIR="/data/"

#one model for TUT

#EXP=HearNet_actions
#MODEL=HearNet
#CHECKPOINT=84
#python knn.py  $MODEL $HOME_DIR/checkpoints/$EXP/model_$CHECKPOINT.ckpt 1 testing
#
#
#EXP=HearNet_layer
#MODEL=HearNet
#CHECKPOINT=72
#python knn.py  $MODEL $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt 1 testing
#
#EXP=HearNet_selfsupervised2
#MODEL=HearNet
#CHECKPOINT=68
#python knn.py  $MODEL $HOME_DIR/checkpoints/$EXP/epoch_$CHECKPOINT.ckpt 1 testing

#audio
#EXP=Audio12820epochs_
#MODEL=HearNet
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#
#    for CHECKPOINT in random 9 19 # 29 39   49 59 69 79 89 99
#    do
#
#        python knn.py  $MODEL $HOME_DIR/checkpoints/$EXP$t/model_$CHECKPOINT.ckpt 1 testing
#
#    done
#
#done

EXP=(Audio12820epochs_ Acoustic12820epochs_ )
MODEL=(HearNetResNet18_v1 DualCamHybridNetResNet18_v1)
ST=(1 2 3 4 5)
for t in ${ST[@]}
do
    for ((i = 0 ; i < ${#EXP[@]} ; i++))
    do
        for CHECKPOINT in 19 # 29 39   49 59 69 79 89 99
        do
            python3 knn.py  ${MODEL[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt 0 testing

        done

    done
done

#EXP=(Audio12820epochs_ Acoustic12820epochs_ Audio12820epochs_2_ Acoustic12820epochs_2_ )
#MODEL1=(HearNet DualCamHybridNet HearNet DualCamHybridNet)
#MODEL=(ResNet18_v1 ResNet18_v1 ResNet18_v1 ResNet18_v1)
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#    for ((i = 0 ; i < ${#EXP[@]} ; i++))
#    do
#        for CHECKPOINT in random 0 2 4 6 9 14 19 # 29 39   49 59 69 79 89 99
#        do
#            python knn.py  ${MODEL[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt 0 testing
#
#            python knn.py  ${MODEL[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt 0 validation
#
#            python knn.py  ${MODEL1[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt 0 testing
#
#            python knn.py  ${MODEL1[$i]} $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt 0 validation
#
#        done
#
#    done
#done

#EXP=(Audio12820epochs_ Acoustic12820epochs_)
#MODEL1=(HearNet DualCamHybridNet)
#MODEL=(ResNet18_v1 ResNet18_v1)
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#
#    for ((i = 0 ; i < ${#EXP[@]} ; i++))
#    do
#        for CHECKPOINT in random 0 2 4 6 9 14 19 #29 49 59 69 79 99
#        do
#            python crossmodalretrieval_featuremaps.py --model1 ${MODEL1[$i]} --model2 ${MODEL[$i]} --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --mode testing
#        done
#    done
#done
#
#EXP=(Audio12820epochs_2_ Acoustic12820epochs_2_ )
#MODEL1=(HearNet DualCamHybridNet)
#MODEL=(ResNet18_v1 ResNet18_v1)
#ST=(1 2 3 4 5)
#for t in ${ST[@]}
#do
#
#    for ((i = 0 ; i < ${#EXP[@]} ; i++))
#    do
#        for CHECKPOINT in random 0 2 4 6 9 14 19 #29 49 59 69 79 99
#        do
#            python crossmodalretrieval_featuremaps_old.py --model1 ${MODEL1[$i]} --model2 ${MODEL[$i]} --init_checkpoint $HOME_DIR/checkpoints/${EXP[$i]}$t/model_$CHECKPOINT.ckpt --mode testing
#        done
#    done
#done

#transferring

#ST=2
#NAME=transfer
#EXP=(0.1_ 0.3_ 0.5_ 0.7_ 0.9_)
#MODEL=HearNet
#
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#	for CHECKPOINT in random 9 19 # 29 39   49 59 69 79 89 99
#	do
#
#		python knn.py  $MODEL $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt 1 testing
#
#	done
#
#done
#
#
#NAME=transfer
#ST=2
#EXP=(0.1_ 0.3_ 0.5_ 0.7_ 0.9_ 0.1_2_ 0.3_2_ 0.5_2_ 0.7_2_ 0.9_2_)
#MODEL=ResNet18_v1
#MODEL1=HearNet
#
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#    for CHECKPOINT in random 0 2 4 6 9 14 19
#    do
#        python knn.py  $MODEL $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt 0 testing
#
#        python knn.py  $MODEL $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt 0 validation
#
#        python knn.py  $MODEL1 $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt 0 testing
#
#        python knn.py  $MODEL1 $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt 0 validation
#
#    done
#done
#
#
#ST=2
#NAME=transfer
#EXP=(0.1_ 0.3_ 0.5_ 0.7_ 0.9_)
#MODEL1=HearNet
#MODEL=ResNet18_v1
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#for CHECKPOINT in random 0 2 4 6 9 14 19
#	do
#        python crossmodalretrieval_featuremaps.py --model1 $MODEL1 --model2 $MODEL --init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --mode testing
#
#    done
#done
#
#ST=2
#NAME=transfer
#EXP=(0.1_2_ 0.3_2_ 0.5_2_ 0.7_2_ 0.9_2_)
#MODEL1=HearNet
#MODEL=ResNet18_v1
#for ((i = 0 ; i < ${#EXP[@]} ; i++))
#do
#for CHECKPOINT in random 0 2 4 6 9 14 19
#	do
#        python crossmodalretrieval_featuremaps_old.py --model1 $MODEL1 --model2 $MODEL --init_checkpoint $HOME_DIR/checkpoints/$NAME${EXP[$i]}$ST/model_$CHECKPOINT.ckpt --mode testing
#
#    done
#done
