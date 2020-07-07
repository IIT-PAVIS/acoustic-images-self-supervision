import argparse
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_audio', type=str)
    parser.add_argument('TUT', help='init_checkpoint', type=int)
    parser.add_argument('--init_checkpoint', help='init_checkpoint',
                        nargs='*', type=str)
    parsed_args = parser.parse_args()

    model_audio = parsed_args.model_audio
    init_checkpoint = parsed_args.init_checkpoint
    TUT = parsed_args.TUT
    num = [1, 3, 4, 5]
    num = np.array(num)
    thisdict = {
        'random': 0,
        '0': 1,
        '2': 2,
        '4': 3,
        '6': 4,
        '9': 5,
        '14': 6,
        '19': 7
    }
    if TUT:
        thisdict = {
            'random': 0,
            '9': 1,
            '19': 2
        }
    dict2 = {
        'Average': 0,
        'Rank1': 1,
        'Rank2': 2,
        'Rank5': 3,
        'Rank10': 4,
        'Rank30': 5
    }
    datasetTesting = 'testing'
    names = ['random', '1', '3', '5', '7', '10', '15', '20']
    if TUT:
        names = ['random', '10', '20']
    name2 = ['Classification acc', 'Rank1', 'Rank2', 'Rank5', 'Rank10', 'Rank30']
    floatnum = ['', '0.1', '0.3', '0.5', '0.7', '0.9']
    accuracies = np.zeros((len(init_checkpoint), len(name2), len(names)))
    dev = np.zeros((len(init_checkpoint), len(name2), len(names)))
    for a in range(len(init_checkpoint)):
        nameTest_audio = '{}_{}'.format(model_audio, datasetTesting)
        if TUT:
            nameTest_audio = 'TUT/' + nameTest_audio
        string = str.join('/', init_checkpoint[a].split('/')[:-1] + [nameTest_audio])
        if os.path.isfile('{}/acc{}_{}.txt'.format(str.join('/', string.split('/')[:-1]), model_audio, datasetTesting)):
            file = open('{}/acc{}_{}.txt'.format(str.join('/', string.split('/')[:-1]), model_audio, datasetTesting), 'r')
            lines = file.readline()
            while lines:
                lines = lines.split(' ')
                epochs = lines[0]
                vec = lines[1]
                mean = lines[3]
                std = lines[6].split('\n')[0]
                accuracies[a][dict2[vec]][thisdict[epochs]] = np.float(mean)
                dev[a][dict2[vec]][thisdict[epochs]] = np.float(std)
                print('{} {} {} {}'.format(vec, epochs, mean, std))
                lines = file.readline()
            file.close()
    #fig1 = plt.figure()
    #fig1.suptitle('{}'.format(name2[0]), fontsize=24)
    for a in range(len(init_checkpoint)):
        plt.errorbar(names, accuracies[a, 0, :], dev[a, 0, :], marker='o', label='{} {} {}'.format(name2[0], model_audio, floatnum[a]))
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.legend()
    plt.show()

    for i in num:
        #fig = plt.figure()
        #fig.suptitle('{}'.format(name2[i]), fontsize=24)
        for a in range(len(init_checkpoint)):
            plt.errorbar(names, accuracies[a, i, :], dev[a, i, :], marker='o', label='{} {} {}'.format(name2[i], model_audio, floatnum[a]))
            # plt.errorbar(names, accuracies2[i, :], dev2[i, :], marker='o', label='{} {}'.format(name2[i], 'ResNet18 DualCamHybridNet 128 vector'))
            legend = plt.legend(loc=4, prop={'size': 18})
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('accuracy', fontsize=16)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
    
# HearNet
# --init_checkpoint
# /data/vsanguineti/checkpoints2/embeddingTransferTriplet0.1_5/model_100.ckpt
# /data/vsanguineti/checkpoints2/embeddingTransferTriplet0.3_5/model_100.ckpt
# /data/vsanguineti/checkpoints2/embeddingTransferTriplet0.5_5/model_100.ckpt
# /data/vsanguineti/checkpoints2/embeddingTransferTriplet0.7_5/model_100.ckpt
# /data/vsanguineti/checkpoints2/embeddingTransferTriplet0.9_5/model_100.ckpt
# /data/vsanguineti/checkpoints2/embeddingAudioScalar2MapFarDifferentDot0.00001vers2_5/model_100.ckpt
#
# DualCamHybridNet
# --init_checkpoint
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic6402triplet2_2_1/model_100.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic6402triplet_2_1/model_100.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic64023_2_1/model_100.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic64022_2_1/model_100.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic6402_2_1/model_100.ckpt

# For resnet
# ResNet18_v1
# --init_checkpoint
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Audio64023_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic64023_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.1_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.3_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.5_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.7_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.9_1/model_1.ckpt
# --TUT
# 0
# 0
# 0
# 0
# 0
# 0
# 0

# HearNet
# --init_checkpoint
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Audio64023_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.1_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.3_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.5_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.7_1/model_1.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.9_1/model_1.ckpt
# --TUT
# 1
# 1
# 1
# 1
# 1
#1