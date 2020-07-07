from datetime import datetime
from dataloader.actions_data import ActionsDataLoader
from models.vision import ResNet50Model
from models.vision import ResNet18Model
from models.vision import ResNet18_v1
from models.vision import ResNet50TemporalModel
from models.audition import HearModel
from models.audition import SoundNet5Model
from tensorflow.python.ops import nn_ops
from models.audition import DualCamHybridModel
import numpy as np
import tensorflow as tf
import os
import sys
from scipy.spatial import distance
import sklearn
import matplotlib.pyplot as plt
import itertools

flags = tf.app.flags
slim = tf.contrib.slim
flags.DEFINE_string('model1', 'DualCamHybridNet', 'Model type, it can be one of \'DualCamHybridNet\', or \'HearNet\'')
flags.DEFINE_string('model2', 'ResNet18_v1', 'Model type, it can be one of \'ResNet18_v1\', or  \'DualCamHybridNet\'')
flags.DEFINE_integer('temporal_pooling', 1, 'Temporal pooling')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('nr_frames', 2, 'Number of frames')  # 12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 2, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('total_length', 2, 'Length in seconds of total sequence sample')
flags.DEFINE_integer('number_of_crops', 1, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('num_class', 128, 'Classes')
flags.DEFINE_string('mode', 'testing', 'training or testing or validation')
FLAGS = flags.FLAGS


def main(_):
    dataset = FLAGS.mode
    _FRAMES_PER_SECOND = 12
    numcl = 10
    num_embedding = 128
    num_scenario = 61
    size1 = 12
    size2 = 16
    size11 = 224
    size22 = 298
    size33 = 3
    if dataset == 'training':
        data_size = 6628
    elif dataset == 'validation':
        data_size = 1212
    else:
        data_size = 1399
    batch_size = 4
    print('Computing cross-modal {}'.format(dataset))
    name1 = '{}_{}'.format(FLAGS.model1, dataset)
    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]
    data_dir1 = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name1]) + '_' + name
    
    num_classes = FLAGS.num_class
    
    # Create data loaders according to the received program arguments
    print('{} - Creating data loaders'.format(datetime.now()))
    modalities = []
    if FLAGS.model1 == 'DualCamNet' or FLAGS.model1 == 'DualCamHybridNet':
        modalities.append(0)
    if FLAGS.model1 == 'SoundNet5' or FLAGS.model1 == 'HearNet':
        modalities.append(1)
    if FLAGS.model2 == 'ResNet50' or FLAGS.model2 == 'ResNet18' or FLAGS.model2 == 'ResNet18_v1' or FLAGS.model2 == 'TemporalResNet50':
        modalities.append(2)
    if FLAGS.model2 == 'DualCamNet' or FLAGS.model2 == 'DualCamHybridNet':
        modalities.append(0)
    if FLAGS.model2 == 'ResNet18_v1' and FLAGS.nr_frames < 12 * FLAGS.sample_length:
        random_pick = True
    else:
        random_pick = False
    
    normalize = False
    build_spectrogram = False
    if FLAGS.model1 == 'HearNet':
        normalize = True
        build_spectrogram = True
    # if compute errors on classes
    errors = True
    name_file = '/data/vsanguineti/tfrecords/lists/{}.txt'.format(dataset)
    train_data = ActionsDataLoader(name_file, 'inference', batch_size,
                                   num_epochs=1, normalize=normalize, build_spectrogram=build_spectrogram,
                                   number_of_crops=FLAGS.number_of_crops, random_pick=random_pick,
                                   sample_rate=22050, total_length=FLAGS.total_length,
                                   sample_length=FLAGS.sample_length,
                                   buffer_size=10, shuffle=False, modalities=modalities, nr_frames=FLAGS.nr_frames)
    
    # iterator = train_data.data.make_one_shot_iterator()
    # next_batch = iterator.get_next()
    
    # Build model
    print('{} - Building model'.format(datetime.now()))
    
    if FLAGS.model2 == 'ResNet50':
        model2 = ResNet50Model(input_shape=[224, 298, 3], num_classes=num_classes)
    elif FLAGS.model2 == 'ResNet18_v1':
        model2 = ResNet18_v1(input_shape=[224, 298, 3], num_classes=num_classes, map=True)
    elif FLAGS.model2 == 'ResNet18':
        model2 = ResNet18Model(input_shape=[224, 298, 3], num_classes=num_classes, nr_frames=12)
    elif FLAGS.model2 == 'TemporalResNet50':
        model2 = ResNet50TemporalModel(input_shape=[224, 298, 3], num_classes=num_classes, nr_frames=12)
    elif FLAGS.model2 == 'DualCamHybridNet':
        model2 = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=num_classes)
    else:
        # Not necessary but set model to None to avoid warning about using unassigned local variable
        model = None
        raise ValueError('Unknown model type')
    if FLAGS.model1 == 'DualCamHybridNet':
        model1 = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=num_classes)
    elif FLAGS.model1 == 'SoundNet5':
        model1 = SoundNet5Model(input_shape=[22050 * 2, 1, 1], num_classes=num_classes)
    elif FLAGS.model1 == 'HearNet':
        model1 = HearModel(input_shape=[200, 1, 257], num_classes=num_classes)
    else:
        # Not necessary but set model to None to avoid warning about using unassigned local variable
        model = None
        raise ValueError('Unknown model type')
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()
    
    datashape1 = [model1.height, model1.width, model1.channels]
    datashape2 = [model2.height, model2.width, model2.channels]
    
    data1 = tf.reshape(next_batch[modalities[0]],
                       shape=[-1, datashape1[0], datashape1[1], datashape1[2]])
    data2 = tf.reshape(next_batch[modalities[1]],
                       shape=[-1, datashape2[0], datashape2[1], datashape2[2]])
    label = tf.reshape(next_batch[3],
                       shape=[-1, 10])
    scenario = tf.reshape(next_batch[4],
                          shape=[-1, 61])
    
    model1._build_model(data1)
    model2._build_model(data2)

    
    if FLAGS.model2 == 'ResNet18_v1':
        logits = model2.output
        expanded_shape = [-1, FLAGS.nr_frames, 12, 16, num_classes]
        visual_logits = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    elif FLAGS.model2 == 'ResNet50':
        logits = model2.output
        expanded_shape = [-1, FLAGS.nr_frames, num_classes]
        visual_logits = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    else:
        acousticlogits_multiple = model2.output
        expanded_shape = [-1, FLAGS.sample_length * _FRAMES_PER_SECOND, 12, 16, num_classes]
        visual_logits = tf.reduce_mean(tf.reshape(acousticlogits_multiple, shape=expanded_shape), axis=1)
    # normalize vector of audio with positive and then negative
    if FLAGS.model1 == 'DualCamHybridNet' and FLAGS.temporal_pooling:
        # logits = model1.output
        # expanded_shape = [-1, FLAGS.sample_length * 12, num_classes]
        # logits1 = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
        acousticlogits_multiple = model1.output
        expanded_shape = [-1, FLAGS.sample_length * _FRAMES_PER_SECOND, 12, 16, num_classes]
        acousticlogits_reshape = tf.reduce_mean(tf.reshape(acousticlogits_multiple, shape=expanded_shape), axis=1)
        # acousticlogits_reshape = tf.nn.l2_normalize(acousticlogits_reshape, dim=[0, 1])
    else:
        logits1 = model1.output
        acousticlogits = logits1  # tf.nn.l2_normalize(logits1, dim=1)
        # Define contrastive loss after having logits
        # compute video anchor, positive and negative audio
        # multiply acoustic vector by 12, 16 times
        acousticlogits_multiple = tf.tile(acousticlogits, [1, 12 * 16])
        # reshape in order to have same dimension of video feature map
        acousticlogits_reshape = tf.reshape(acousticlogits_multiple, [-1, 12, 16, num_classes])
        acousticlogits_reshape = nn_ops.relu(acousticlogits_reshape)
    print('{} - Starting'.format(datetime.now()))
    
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        # Initialize student model
        if FLAGS.model1 == 'DualCamHybridNet' or FLAGS.model1 == 'SoundNet5':
            var_list = slim.get_variables(model1.scope)
        else:
            var_list = slim.get_model_variables(model1.scope)
        if FLAGS.model2 == 'DualCamHybridNet':
            var_list2 = slim.get_variables(model2.scope)
        else:
            var_list2 = slim.get_model_variables(model2.scope)
        #var_list_attention = slim.get_model_variables('attention_logits')
        #weights, bias = session.run(var_list_attention)
        if FLAGS.init_checkpoint is None:
            print('{} - Initializing student model'.format(datetime.now()))
            # model1.init_model(session, FLAGS.init_checkpoint)
            # model2.init_model(session, FLAGS.init_checkpoint)
            logits_init_op = tf.variables_initializer(var_list + var_list2) #+ var_list_attention)
            # Initialize the new logits layer
            session.run(logits_init_op)
            print('{} - Done'.format(datetime.now()))
        else:
            print('{} - Restoring student model'.format(datetime.now()))
            saver = tf.train.Saver(var_list=var_list + var_list2) #+ var_list_attention)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
        
        dataset_audio_list_features = np.zeros([data_size, size1, size2, num_embedding], dtype=float)
        dataset_video_list_features = np.zeros([data_size, size1, size2, num_embedding], dtype=float)
        dataset_labels = np.zeros([data_size], dtype=int)
        dataset_scenario = np.zeros([data_size], dtype=int)
        if errors:
            # initialize cm for rank1, rank5 and rank10
            confusion_matrix1 = np.zeros([numcl, numcl], dtype=float)
            confusion_matrix5 = np.zeros([numcl, numcl], dtype=float)
            confusion_matrix10 = np.zeros([numcl, numcl], dtype=float)
            # compute number of samples for each class
            num_samples_class = np.zeros([numcl], dtype=int)
        session.run(train_iterat.initializer)
        total_size = 0
        batch_count = 0
        accuracy_sum = 0.0
        while True:
            try:
                start_time = datetime.now()
                print('{} - Processing batch {}'.format(start_time, batch_count + 1))
                features_audio, features_video, labels, scenarios = session.run(
                    [acousticlogits_reshape, visual_logits, label, scenario],
                    feed_dict={handle: train_handle,
                               model1.network['keep_prob']: 1.0,
                               model2.network['keep_prob']: 1.0,
                               model1.network['is_training']: 0,
                               model2.network['is_training']: 0})
                
                batchnum = labels.shape[0]
                labels = np.argmax(labels, 1)
                scenarios = np.argmax(scenarios, 1)
                # copy block of data
                dataset_audio_list_features[total_size:total_size + batchnum, :] = features_audio
                dataset_video_list_features[total_size:total_size + batchnum, :] = features_video
                dataset_labels[total_size:total_size + batchnum] = labels
                dataset_scenario[total_size:total_size + batchnum] = scenarios
                # increase number of data
                total_size += batchnum
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
        total_size = dataset_video_list_features.shape[0]
        print('{} - Completed, got {} samples'.format(datetime.now(), total_size))
        # for all acoustic samples compute the audio-visual vectors and check if you find the correct one
        rank1 = 0
        rank2 = 0
        rank5 = 0
        rank10 = 0
        rank30 = 0
        for a in range(dataset_audio_list_features.shape[0]):
            audio_features = dataset_audio_list_features[a]
            audio_features = np.expand_dims(audio_features, axis=0)
            # generate all maps depending on acoustic image
            #audion = sklearn.preprocessing.normalize(np.reshape(audio_features,(-1,128)), axis=1)
            #videon = sklearn.preprocessing.normalize(np.reshape(dataset_video_list_features,(-1,128)), axis=1)
            #innerdot = np.reshape(videon, (-1, 12, 16, 128)) * np.reshape(audion, (-1, 12, 16, 128))
            innerdot = audio_features * dataset_video_list_features
            # innerdot = np.sum(innerdot, axis=3, keepdims=True)
            # audio-visual map
            product = innerdot * dataset_video_list_features
            # sum video map feature along two spatial dimensions
            videoweighted = np.sum(product, axis=2)
            videoweighted = np.sum(videoweighted, axis=1)
            # sun audio along two spatial dimensions
            audioweighted = np.sum(audio_features, axis=2)
            audioweighted = np.sum(audioweighted, axis=1)
            # normalize audio-visual vector
            productvectnorm = sklearn.preprocessing.normalize(videoweighted, axis=1)
            # normalize audio vector
            productvectnorm2 = sklearn.preprocessing.normalize(audioweighted, axis=1)
            # for every audio vector compute distance to all video vectors
            distancearray = distance.cdist(productvectnorm2, productvectnorm, 'euclidean')
            print('{} distance matrix {} {}'.format(datetime.now(), a, np.shape(distancearray)[1]))
            # for every acoustic feature vector find close one
            index = np.argsort(distancearray)
            index = np.squeeze(index)
            # order distances and take position
            # if they belong to same class
            if dataset_labels[a] == dataset_labels[index[0]]:
                rank1 += 1
                rank2 += 1
                rank5 += 1
                rank10 += 1
                rank30 += 1
            elif dataset_labels[a] in dataset_labels[index[[0, 1]]]:
                rank2 += 1
                rank5 += 1
                rank10 += 1
                rank30 += 1
            elif dataset_labels[a] in dataset_labels[index[:5]]:
                rank5 += 1
                rank10 += 1
                rank30 += 1
            elif dataset_labels[a] in dataset_labels[index[:10]]:
                rank10 += 1
                rank30 += 1
            elif dataset_labels[a] in dataset_labels[index[:30]]:
                rank30 += 1
            
            if errors:
                # add sample for this class
                num_samples_class[dataset_labels[a]] += 1
                # add in position of predicted class
                confusion_matrix1[dataset_labels[a], dataset_labels[index[0]]] += 1
                for b in range(5):
                    confusion_matrix5[dataset_labels[a], dataset_labels[index[b]]] += 1
                    confusion_matrix10[dataset_labels[a], dataset_labels[index[b]]] += 1
                for b in range(5, 10):
                    confusion_matrix10[dataset_labels[a], dataset_labels[index[b]]] += 1

        # divide each row for number of samples of that row
        confusion_matrix1 = confusion_matrix1 / num_samples_class.reshape(-1, 1)
        confusion_matrix5 = confusion_matrix5 / num_samples_class.reshape(-1, 1)
        confusion_matrix10 = confusion_matrix10 / num_samples_class.reshape(-1, 1)
        # divide for rank > 1
        confusion_matrix5 = confusion_matrix5 / 5.0
        confusion_matrix10 = confusion_matrix10 / 10.0
        print(confusion_matrix1)
        print(confusion_matrix5)
        print(confusion_matrix10)
        classes = ['Train', 'Boat', 'Drone', 'Fountain', 'Drill',
                   'Razor', 'Hair dryer', 'Vacuumcleaner', 'Cart', 'Traffic']
        cmap = plt.cm.Blues
        plt.imshow(confusion_matrix10, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f'
        thresh = confusion_matrix10.max() / 2.
        for i, j in itertools.product(range(confusion_matrix10.shape[0]), range(confusion_matrix10.shape[1])):
            plt.text(j, i, format(confusion_matrix10[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix10[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(data_dir1 + '/confusion_matrix.png')
        
        accuracy = 1.0 * rank1 / np.shape(distancearray)[1]
        rank2 = 1.0 * rank2 / np.shape(distancearray)[1]
        rank5 = 1.0 * rank5 / np.shape(distancearray)[1]
        rank10 = 1.0 * rank10 / np.shape(distancearray)[1]
        rank30 = 1.0 * rank30 / np.shape(distancearray)[1]
        print ('Accuracy {:6f} rank2 {:6f} rank5 {:6f} rank10 {:6f} rank30 {:6f}'.format(accuracy, rank2, rank5, rank10,
                                                                                         rank30))
        file = open('{}_{}_{}_retrieval.txt'.format(data_dir1, FLAGS.model1, dataset), 'w')
        file.write(
            'Accuracy {:6f} rank2 {:6f} rank5 {:6f} rank10 {:6f} rank30 {:6f}'.format(accuracy, rank2, rank5, rank10,
                                                                                      rank30))
        file.close()
        
        end_time = datetime.now()
        print('{} - Completed in {} seconds'.format(end_time, (end_time - start_time).total_seconds()))


if __name__ == '__main__':
    tf.app.run()

# --model1
# DualCamHybridNet
# --model2
# ResNet18_v1
# --train_file
# /data/vsanguineti/dualcam_actions_dataset/30_seconds/lists/testing.txt
# --init_checkpoint
# /data/vsanguineti/checkpoints2/embeddingAcousticNetMap/model.ckpt
# --num_classes
# 128
# --nr_frames
# 2
# --sample_length
# 2