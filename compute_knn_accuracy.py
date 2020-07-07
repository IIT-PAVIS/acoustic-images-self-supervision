import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('TUT', type=int)
    parser.add_argument('set', type=str)
    parser.add_argument('--init_checkpoints', help='init_checkpoints',
                        nargs='*', type=str)

    parsed_args = parser.parse_args()
    TUT = parsed_args.TUT
    model = parsed_args.model
    init_checkpoints = parsed_args.init_checkpoints
    datasetTesting = parsed_args.set
    for init_checkpoint in init_checkpoints:
        # s = init_checkpoint.split('/')[-2]
        # namecheckpoint = s.split('vers2_')[-1]
        
        nameTest = '{}_{}'.format(model, datasetTesting)
        names = ['random', '0', '2', '4', '6', '9', '14', '19']#, '29', '39','49','59', '69','79', '89','99']
        if TUT:
            nameTest = 'TUT/' + nameTest
            names = ['random', '9', '19']
        numbers = [1, 2, 3, 4, 5]
        #, '30', '40', '50', '60', '70', '80', '90', '100']
        accuracies = np.zeros((len(numbers), len(names)))
        neighbour = np.zeros((len(numbers), len(names)))
        neighbour2 = np.zeros((len(numbers), len(names)))
        neighbour5 = np.zeros((len(numbers), len(names)))
        neighbour10 = np.zeros((len(numbers), len(names)))
        neighbour30 = np.zeros((len(numbers), len(names)))
        for a in range(len(numbers)):
            if numbers[a] > numbers[0]:
                init_checkpoint = init_checkpoint.replace('_{}/'.format(numbers[a]-1), '_{}/'.format(numbers[a]))
                print(init_checkpoint + "/n")
        #save in file computed accuracy
            for i in range(len(names)):
                data_dirTest = str.join('/', init_checkpoint.split('/')[:-1] + [nameTest]) + '_' + names[i]
                if os.path.isfile('{}_{}_{}_knn_value.txt'.format(data_dirTest, model, datasetTesting)):
                    filefinal = open('{}_{}_{}_knn_value.txt'.format(data_dirTest, model, datasetTesting), 'r')
                    print('{}_{}_{}_knn_value.txt'.format(data_dirTest, model, datasetTesting) + "/n")
                    s = filefinal.read()
                    acc = s.split(' ')[0]
                    num = acc.split('=')[1]
                    print(num)
                    accuracies[a][i] = float(num)
                    filefinal.close()
                if os.path.isfile('{}_{}_{}_retrieval.txt'.format(data_dirTest, model, datasetTesting)):
                    filefinal = open('{}_{}_{}_retrieval.txt'.format(data_dirTest, model, datasetTesting), 'r')
                    print('{}_{}_{}_retrieval.txt'.format(data_dirTest, model, datasetTesting) + "/n")
                    stringa = filefinal.read()
                    acc = stringa.split(' ')[1]
                    acc2 = stringa.split(' ')[3]
                    acc5 = stringa.split(' ')[5]
                    acc10 = stringa.split(' ')[7]
                    acc30 =  stringa.split(' ')[9]
    
                    print('Rank1 {} rank2 {} rank5 {} rank10 {}'.format(acc, acc2, acc5, acc10))
                    neighbour[a][i] = float(acc)
                    neighbour2[a][i] = float(acc2)
                    neighbour5[a][i] = float(acc5)
                    neighbour10[a][i] = float(acc10)
                    neighbour30[a][i] = float(acc30)
                    filefinal.close()
                    
        average = np.mean(accuracies, axis=0)
        standard_deviation = np.std(accuracies, axis=0)
        
        averageacc = np.mean(neighbour, axis=0)
        standard_deviationacc = np.std(neighbour, axis=0)
        
        averageacc2 = np.mean(neighbour2, axis=0)
        standard_deviationacc2 = np.std(neighbour2, axis=0)
        
        averageacc5 = np.mean(neighbour5, axis=0)
        standard_deviationacc5 = np.std(neighbour5, axis=0)
        
        averageacc10 = np.mean(neighbour10, axis=0)
        standard_deviationacc10 = np.std(neighbour10, axis=0)
        
        averageacc30 = np.mean(neighbour30, axis=0)
        standard_deviationacc30 = np.std(neighbour30, axis=0)
        
        file = open('{}/acc{}_{}.txt'.format(str.join('/', data_dirTest.split('/')[:-1]), model, datasetTesting), 'w')
        print('{}/acc{}_{}.txt'.format(str.join('/', data_dirTest.split('/')[:-1]), model, datasetTesting) + "/n")
        for k in range(len(names)):
            print('{} Average = {:6f} std = {:6f}'.format(names[k], average[k], standard_deviation[k]))
            print('{} Rank1 = {:6f} std = {:6f}'.format(names[k], averageacc[k], standard_deviationacc[k]))
            print('{} Rank2 = {} std = {:6f}'.format(names[k], averageacc2[k], standard_deviationacc2[k]))
            print('{} Rank5 = {:6f} std = {:6f}'.format(names[k], averageacc5[k], standard_deviationacc5[k]))
            print('{} Rank10 = {:6f} std = {:6f}'.format(names[k], averageacc10[k], standard_deviationacc10[k]))
            print('{} Rank30 = {:6f} std = {:6f}'.format(names[k], averageacc30[k], standard_deviationacc30[k]))
            file.write('{} Average = {:6f} std = {:6f}\n'.format(names[k], average[k], standard_deviation[k]))
            file.write('{} Rank1 = {:6f} std = {:6f}\n'.format(names[k], averageacc[k], standard_deviationacc[k]))
            file.write('{} Rank2 = {} std = {:6f}\n'.format(names[k], averageacc2[k], standard_deviationacc2[k]))
            file.write('{} Rank5 = {:6f} std = {:6f}\n'.format(names[k], averageacc5[k], standard_deviationacc5[k]))
            file.write('{} Rank10 = {:6f} std = {:6f}\n'.format(names[k], averageacc10[k], standard_deviationacc10[k]))
            file.write('{} Rank30 = {:6f} std = {:6f}\n'.format(names[k], averageacc30[k], standard_deviationacc30[k]))
        file.close()
        
        

if __name__ =='__main__':
    main()
