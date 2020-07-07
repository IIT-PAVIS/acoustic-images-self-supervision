import argparse
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_audio', type=str)
    parser.add_argument('model_acoustic', type=str)
    parser.add_argument('set', type=str)
    parser.add_argument('--init_checkpoint_audio', help='init_checkpoint',
                        nargs='*', type=str)
    parser.add_argument('--init_checkpoint_acoustic', help='init_checkpoint',
                        nargs='*', type=str)
    parsed_args = parser.parse_args()
    
    model_audio = parsed_args.model_audio
    model_acoustic = parsed_args.model_acoustic
    init_checkpoint_audio = parsed_args.init_checkpoint_audio
    init_checkpoint_acoustic = parsed_args.init_checkpoint_acoustic
    #assert len(init_checkpoint_acoustic) == len(init_checkpoint_audio)
    num = [1, 3, 4, 5]
    thisdict = {
        'random': 0,
        '0': 1,
        '2': 2,
        '4': 3,
        '6': 4,
        '9': 5,
        '14': 6,
        '19': 7}#, '29': 8, '39':9,'49':10,'59':11, '69':12,'79':13, '89':14,'99':15
    #}
    dict2 = {
        'Average': 0,
        'Rank1': 1,
        'Rank2': 2,
        'Rank5': 3,
        'Rank10': 4,
        'Rank30': 5
    }
    datasetTesting = parsed_args.set
    names = ['random', '1', '3', '5', '7', '10', '15', '20']#, '29', '39', '49', '59', '69','79', '89','99']
    name2 = ['Classification acc', 'Rank1', 'Rank2', 'Rank5', 'Rank10', 'Rank30']
    nameTest_audio = '{}_{}'.format(model_audio, datasetTesting)
    nameTest_acoustic = '{}_{}'.format(model_acoustic, datasetTesting)
    floatnum = ['', '0.1', '0.3', '0.5', '0.7', '0.9', '']
    # '', '0.1', '0.3', '0.5', '0.7', '0.9', 'without privileged information'
    accuracies = np.zeros((len(init_checkpoint_audio), len(name2), len(names)))
    dev = np.zeros((len(init_checkpoint_audio), len(name2), len(names)))
    accuracies1 = np.zeros((len(init_checkpoint_acoustic), len(name2), len(names)))
    dev1 = np.zeros((len(init_checkpoint_acoustic), len(name2), len(names)))
    
    for a in range(len(init_checkpoint_audio)):
        string = str.join('/', init_checkpoint_audio[a].split('/')[:-1] + [nameTest_audio])
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
    
    for a in range(len(init_checkpoint_acoustic)):
        string = str.join('/', init_checkpoint_acoustic[a].split('/')[:-1] + [nameTest_acoustic])
        if os.path.isfile('{}/acc{}_{}.txt'.format(str.join('/', string.split('/')[:-1]), model_acoustic, datasetTesting)):
            file = open('{}/acc{}_{}.txt'.format(str.join('/', string.split('/')[:-1]), model_acoustic, datasetTesting), 'r')
            lines = file.readline()
            while lines:
                lines = lines.split(' ')
                epochs = lines[0]
                vec = lines[1]
                mean = lines[3]
                std = lines[6].split('\n')[0]
                accuracies1[a][dict2[vec]][thisdict[epochs]] = np.float(mean)
                dev1[a][dict2[vec]][thisdict[epochs]] = np.float(std)
                print('{} {} {} {}'.format(vec, epochs, mean, std))
                lines = file.readline()
            file.close()
    # fig = plt.figure()
    # fig.suptitle('{}'.format(name2[0]), fontsize=24)
    for a in range(len(init_checkpoint_audio)):
        plt.errorbar(names, accuracies[a, 0, :], dev[a, 0, :], marker='o',
                     label='{} {} {}'.format(name2[0], model_audio, floatnum[a]))# + H
    meanmax = np.max(accuracies[0, 0, :])
    epochmax = np.argmax(accuracies[0, 0, :])
    meanmaxr1 = np.max(accuracies[0, 1, :])
    epochmaxr1 = np.argmax(accuracies[0, 1, :])
    print ('maximum {} {} {}'.format(model_audio, epochmax, meanmax))
    print ('maximum rank 1{} {} {}'.format(model_audio, epochmaxr1, meanmaxr1))
    for a in range(len(init_checkpoint_acoustic)):
        plt.errorbar(names, accuracies1[a, 0, :], dev1[a, 0, :], marker='o',
                     label='{} {} {}'.format(name2[0], model_acoustic, floatnum[a]))# + D
    meanmax1 = np.max(accuracies1[0, 0, :])
    epochmax1 = np.argmax(accuracies1[0, 0, :])
    meanmaxr11 = np.max(accuracies1[0, 1, :])
    epochmaxr11 = np.argmax(accuracies1[0, 1, :])
    print ('maximum {} {} {}'.format(model_acoustic, epochmax1, meanmax1))
    print ('maximum rank 1{} {} {}'.format(model_acoustic, epochmaxr11, meanmaxr11))
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.legend(loc=0, fontsize=7)
    plt.show()
    for i in num:
        # fig = plt.figure()
        # fig.suptitle('{}'.format(name2[i]), fontsize=24)
        for a in range(len(init_checkpoint_audio)):
            plt.errorbar(names, accuracies[a, i, :], dev[a, i, :], marker='o',
                         label='{} {} {}'.format(name2[i], model_audio, floatnum[a]))
        for a in range(len(init_checkpoint_acoustic)):
            plt.errorbar(names, accuracies1[a, i, :], dev[a, i, :], marker='o',
                         label='{} {} {}'.format(name2[i], model_acoustic, floatnum[a]))
        #plt.legend(loc=4, prop={'size': 18})
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('accuracy', fontsize=16)
        plt.legend(loc=0, fontsize=7)
        plt.show()


if __name__ == '__main__':
    main()

# HearNet
# DualCamHybridNet
# --init_checkpoint_audio
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Audio64023_2_1/model_10.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.1_2_1/model_0.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.3_2_1/model_0.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.5_2_1/model_0.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.7_2_1/model_0.ckpt
# /media/vsanguineti/TOSHIBAEXT/checkpoints/embeddingTransferTriplet20_0.9_2_1/model_0.ckpt
# --init_checkpoint_acoustic
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Acoustic64023_2_1/model_10.ckpt