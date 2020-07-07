from datetime import datetime
import numpy as np
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt
import argparse
#from PIL import Image
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--model1')#it can be one of \'DualCamHybridNet\', or \'HearNet\'
parser.add_argument('--model2')#it can be one of \'ResNet18_v1\', or  \'ResNet50\'
parser.add_argument('--root_dir')# root directory for data
parser.add_argument('--cp_id')#id for checkpoint
parser.add_argument('--mode')#testing
parser.add_argument('--datatype')#music
args = parser.parse_args()
#Come lanciare il codice python3 crossmodalretrieval_array.py --model1 HearNet --model2 ResNet50 --root_dir
# C:\Users\nega9\Desktop\progetto\src\drive-download-20200220T184036Z-001 --cp_id 2 --mode testing --datatype "music"

#le dimensioni dei dati sono queste
# data ([data_size, num_embedding], dtype=float)
# labels ([data_size, numcl], dtype=int)
# scenario ([data_size, num_scenario], dtype=int)
# map [data_size, size1, size2], dtype=float)
# images ([data_size, size11, size22, size33], dtype=float)
# numcl = 9
# num_embedding = 128
# num_scenario = 11
# size1 = 12
# size2 = 16
# size11 = 224
# size22 = 298
# size33 = 3

#Questo codice legge i file numpy e calcola le distanze

def add_border(rgb_image, color='green'):
    rows = rgb_image.shape[0]
    cols = rgb_image.shape[1]
    background = np.zeros((rows + 40, cols + 40, 3), dtype=int)
    if color == 'red':
        background[:, :, 0] = 255
    elif color == 'green':
        background[:, :, 1] = 255
    else:
        background[:, :, 2] = 255

    background[20:-20, 20:-20, :] = rgb_image
    return background

def normalize(rgb_image):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    means = [_R_MEAN, _G_MEAN,_B_MEAN ]
    im2 = np.zeros_like(rgb_image)
    for i in range(3):
        im2[:, :, i] = rgb_image[:, :, i] + np.ones_like(rgb_image[:, :, i])*means[i]
    # rgb_image[:, :, 2] += _R_MEAN
    # rgb_image[:, :, 1] += _G_MEAN
    # rgb_image[:, :, 0] += _B_MEAN
    #im2 = np.concatenate(channels, axis=2)
    # (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    image = np.round(im2).astype(int)
    im2 = np.stack((image[:, :, 2], image[:, :, 1], image[:, :, 0]), axis=2)
    return im2

def add_alpha(rgb_image, transparent = True):
    if transparent:
        alpha = np.ones((image.shape[0], image.shape[1], 0))
    else:
        alpha = np.zeros((image.shape[0], image.shape[1], 0))
    return np.concatenate((rgb_image, alpha), axis=2)

def load_image(infilename):
    img = cv2.imread(infilename)
    return img

if __name__ == '__main__':
    dataset = args.mode
    _FRAMES_PER_SECOND = 12
    if args.datatype == 'outdoor':
        numcl = 10
    elif args.datatype == 'music':
        numcl = 9
    else:
        numcl = 14
    print('Computing cross-modal {}'.format(dataset))
    name1 = '{}_{}'.format(args.model1, dataset)
    root_dir = args.root_dir
    cp_id = args.cp_id

    model1 = args.model1
    model2 = args.model2

    name1 = '{}_{}'.format(model1, dataset)
    name2 = '{}_{}'.format(model2, dataset)

    data_dir_format = (root_dir + '/' + '{}' + '_' + dataset + '_' + cp_id)

    data_dir1 = data_dir_format.format(model1)
    data_dir2 = data_dir_format.format(model2)

    #Legge i file numpy mancano da leggere per model2 images e maps, fai load

    if os.path.isfile('{}/{}_{}_labels.npy'.format(data_dir1, model1, dataset)) \
            and os.path.isfile('{}/{}_{}_data.npy'.format(data_dir1, model1, dataset)):
        featurestraining1 = np.load('{}/{}_{}_data.npy'.format(data_dir1, model1, dataset))
        labelstraining1 = np.load('{}/{}_{}_labels.npy'.format(data_dir1, model1, dataset))
        labelstraining1 = np.argmax(labelstraining1, axis=1)
        scenariotraining1 = np.load('{}/{}_{}_scenario.npy'.format(data_dir1, model1, dataset))
        scenariotraining1 = np.argmax(scenariotraining1, axis=1)
        dataset_scenario = scenariotraining1
    if os.path.isfile('{}/{}_{}_labels.npy'.format(data_dir2, model2, dataset)) \
            and os.path.isfile('{}/{}_{}_data.npy'.format(data_dir2, model2, dataset))\
            and os.path.isfile('{}/{}_{}_images.npy'.format(data_dir2, model2, dataset)):
        featurestraining2 = np.load('{}/{}_{}_data.npy'.format(data_dir2, model2, dataset))
        imagestraining2 = np.load('{}/{}_{}_images.npy'.format(data_dir2, model2, dataset))
        print(featurestraining2.shape[0])



    #Ci sono tanti vettori quante immagini e hanno gli stessi indici quindi puoi usarli
    dataset_labels = labelstraining1

    dataset_audio_list_features = featurestraining1
    dataset_video_list_features = featurestraining2
    imagestraining2 = [normalize(img) for img in imagestraining2]
    note = load_image('/home/vsanguineti/Downloads/nota_musicale.png')
    count = 0

    #per ognuno degli indici leggo un vettore audio e guardo quale e' il vettore video piu' vicino
    for a in range(9, dataset_audio_list_features.shape[0], 5):
        #leggo un vettore audio
        audio_features = dataset_audio_list_features[a]
        #leggo tutti i vettori video
        productvectnorm_video = dataset_video_list_features
        # aggiungo dimensione in modo che si possano confrontare
        productvectnorm_audio = np.expand_dims(audio_features, axis=0)
        #calcolo la distanza tra il vettore audio e tutti i vettori video
        distancearray = distance.cdist(productvectnorm_audio, productvectnorm_video, 'euclidean')
        print('{} distance matrix {} {}'.format(datetime.now(), a, np.shape(distancearray)[1]))
        # for every acoustic feature vector find close one ord
        #ordina gli indici a seconda della distanza dei vettori
        # il primo indice indica la distanza minore, cioè l' indice del vettore video piu vicino
        index = np.argsort(distancearray)
        index = np.squeeze(index)

        # considera i primi 10 indici piu' vicini, leggi le immagini corrispondenti agli indici e
        # l'immagine dell'audio corrispondente al suo indice "a" e salva le figure
        # genera un'unica immagine con sulla sinistra la figura dell'audio con il simbolino dell'audio sopra
        # e sulla destra le 10 del video in orizzontale  con cornici di diversi colori per le due modalita'

        fig = plt.figure(figsize=[19.20, 10.80])
        pos = 1

        elem = fig.add_subplot(1, 11, pos)
        plt.axis('off')
        image = imagestraining2[a]
        #image = normalize(image)
        #image = add_alpha(image)
        #image = Image.fromarray(image, 'RGBA')
        #image.alpha_composite(note_image)
        #image = np.asarray(image)[:,:,:3]
        g = note[:, :, 1]/255
        mask = np.expand_dims(g, axis=2)
        masknot = 1 - mask
        #remove note
        image = image * masknot
        image = image + note
        image = add_border(image, 'green')
        plt.imshow(image)
        elem.set_title('audio')

        for i in range(5):
            pos = pos + 1
            elem = fig.add_subplot(1, 11, pos)
            plt.axis('off')
            image = imagestraining2[index[i]]
            #image = normalize(image)
            if dataset_labels[a] == dataset_labels[index[i]] and dataset_scenario[a] == dataset_scenario[index[i]]:
                image = add_border(image, 'blue')
            elif dataset_labels[a] == dataset_labels[index[i]]:
                image = add_border(image, 'green')
            else:
                image = add_border(image, 'red')
            plt.imshow(image)
            elem.set_title('k = {}'.format(i+1))
        plt.savefig('{}/{} example.png'.format(root_dir, count))
        #plt.show()

        count = count + 1
        # if count == 10:
        #     break

        # order distances and take position
        # if they belong to same class