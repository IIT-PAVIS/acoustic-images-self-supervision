#!/usr/bin/env bash
HOME_DIR="/data/vsanguineti/"

EXP=Acoustic6402_3
MODEL2=ResNet18_v1
MODEL1=DualCamHybridNet
for i in random 0 2 4 6 9 14 19
do
python computedistance2.py --model1 $MODEL1 --model2 $MODEL2 --init_checkpoint $HOME_DIR"/checkpoints/"$EXP"/model_"$i".ckpt"
done

EXP=Audio6402_3
MODEL2=ResNet18_v1
MODEL1=HearNet
for i in random 0 2 4 6 9 14 19
do
python computedistance2.py --model1 $MODEL1 --model2 $MODEL2 --init_checkpoint $HOME_DIR"/checkpoints/"$EXP"/model_"$i".ckpt"
done