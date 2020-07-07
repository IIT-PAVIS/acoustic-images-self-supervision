import argparse
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
# RS = 123
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('init_checkpoint', type=str)
    parser.add_argument('TUT', type=int)
    parser.add_argument('set', type=str)
    parsed_args = parser.parse_args()
    
    model = parsed_args.model
    init_checkpoint = parsed_args.init_checkpoint
    TUT = parsed_args.TUT
    datasetTesting = parsed_args.set
    s = init_checkpoint.split('/')[-1]
    namecheckpoint = (s.split('_')[1]).split('.ckpt')[0]
    dataset = 'training'
    datasetValidation = 'validation'
    name = '{}_{}'.format(model, dataset)
    nameTest = '{}_{}'.format(model, datasetTesting)
    nameValidation = '{}_{}'.format(model, datasetValidation)
    if TUT:
        name = 'TUT/' + name
        nameTest = 'TUT/' + nameTest
        nameValidation = 'TUT/' + nameValidation
    data_dir = str.join('/', init_checkpoint.split('/')[:-1] + [name]) + '_' + namecheckpoint
    data_dirTest = str.join('/', init_checkpoint.split('/')[:-1] + [nameTest]) + '_' + namecheckpoint
    data_dirValidation = str.join('/', init_checkpoint.split('/')[:-1] + [nameValidation]) + '_' + namecheckpoint
    
    featurestraining = []
    labelstraining = []
    featuresvalidation = []
    labelsvalidation = []
    featurestest = []
    labelstest = []
    n_neighbors = [7, 9, 11, 13, 15]
    num_frames = 12
    sample_length = 5
    
    if os.path.isfile('{}/{}_{}_labels.npy'.format(data_dir, model, dataset)) \
            and os.path.isfile('{}/{}_{}_data.npy'.format(data_dir, model, dataset)):
        featurestraining = np.load('{}/{}_{}_data.npy'.format(data_dir, model, dataset))
        labelstraining = np.load('{}/{}_{}_labels.npy'.format(data_dir, model, dataset))
        print(labelstraining.shape[0])
    if os.path.isfile('{}/{}_{}_labels.npy'.format(data_dirTest, model, datasetTesting)) \
            and os.path.isfile('{}/{}_{}_data.npy'.format(data_dirTest, model, datasetTesting)):
        featurestest = np.load('{}/{}_{}_data.npy'.format(data_dirTest, model, datasetTesting))
        labelstest = np.load('{}/{}_{}_labels.npy'.format(data_dirTest, model, datasetTesting))
        print(labelstest.shape[0])
    if os.path.isfile('{}/{}_{}_labels.npy'.format(data_dirValidation, model, datasetValidation)) \
            and os.path.isfile('{}/{}_{}_data.npy'.format(data_dirValidation, model, datasetValidation)):
        featuresvalidation = np.load('{}/{}_{}_data.npy'.format(data_dirValidation, model, datasetValidation))
        labelsvalidation = np.load('{}/{}_{}_labels.npy'.format(data_dirValidation, model, datasetValidation))
        print(labelsvalidation.shape[0])
    labelstest = np.argmax(labelstest, axis=1)
    labelstraining = np.argmax(labelstraining, axis=1)
    labelsvalidation = np.argmax(labelsvalidation, axis=1)
    labelstest2 = labelstest
    # plot TSNE
    # x = TSNE(n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300).fit_transform(featurestest)
    # num_classes = len(np.unique(labelstest2))
    # palette = np.array(sns.color_palette("hls", num_classes))
    # f = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labelstest2.astype(np.int)])
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    # ax.axis('tight')
    #
    # # add the labels for each digit corresponding to the label
    # txts = []
    #
    # for i in range(num_classes):
    #
    #     # Position of each label at median of data points.
    #     xtext, ytext = np.median(x[labelstest2 == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txts.append(txt)
    #
    # # plt.figure(2)
    # # plt.scatter(x[:, 0], x[:, 1], c=palette[labelstest2.astype(np.int)], s=3)
    # plt.figure()

    bestk = 0
    bestacc = 0.0
    filev = open('{}_{}_{}_knn.txt'.format(data_dirValidation, model, datasetValidation), 'w')
    for k in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=k)
        # Train the classifier
        clf.fit(featurestraining, labelstraining)
        y_pred = clf.predict(featuresvalidation)
        counter = 0
        percentage2 = labelsvalidation.shape[0]
        for i in range(percentage2):
            if y_pred[i] == labelsvalidation[i]:
                counter += 1
        perc = counter / float(percentage2)
        print('Accuracy={} k={}\n'.format(perc, k))
        filev.write('Accuracy={} k={}\n'.format(perc, k))
        if perc > bestacc:
            bestk = k
            bestacc = perc
    print('Best k={}\n'.format(bestk))
    filev.write('Best k={}\n'.format(bestk))
    filev.close()
    
    file = open('{}_{}_{}_knn.txt'.format(data_dirTest, model, datasetTesting), 'w')
    for k in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=k)
        # Train the classifier
        clf.fit(featurestraining, labelstraining)
        y_pred = clf.predict(featurestest)
        counter = 0
        percentage2 = labelstest.shape[0]
        for i in range(percentage2):
            if y_pred[i] == labelstest[i]:
                counter += 1
        perc = counter / float(percentage2)
        print('Accuracy={} k={}\n'.format(perc, k))
        file.write('Accuracy={} k={}\n'.format(perc, k))
    file.close()
    
    filefinal = open('{}_{}_{}_knn_value.txt'.format(data_dirTest, model, datasetTesting), 'w')
    clf = KNeighborsClassifier(n_neighbors=bestk)
    # Train the classifier
    clf.fit(featurestraining, labelstraining)
    y_pred = clf.predict(featurestest)
    counter = 0
    percentage2 = labelstest.shape[0]
    for i in range(percentage2):
        if y_pred[i] == labelstest[i]:
            counter += 1
    perc = counter / float(percentage2)
    print('Accuracy={} k={}\n'.format(perc, bestk))
    filefinal.write('Accuracy={} k={}\n'.format(perc, bestk))
    filefinal.close()
    #plot confusion matrix
    # cm = confusion_matrix(labelstest, y_pred)
    # classes = ['Clapping', 'Snapping fingers', 'Speaking', 'Whistling', 'Playing kendama', 'Clicking', 'Typing', \
    #            'Knocking', 'Hammering', 'Peanut breaking', 'Paper ripping', 'Plastic crumpling', 'Paper shaking',
    #            'Stick dropping']
    # plot_confusion_matrix(cm, classes)
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    
    
# /data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/training.txt
# DualCamHybridNet
# /data/vsanguineti/checkpoints2/embeddingAcousticScalar2MapFarDifferentDot0.00001savemodel/model_400.ckpt