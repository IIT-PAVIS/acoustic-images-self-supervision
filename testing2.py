from datetime import datetime
from dataloader.actions_data import ActionsDataLoader
from models.vision import ResNet50Model
from models.vision import ResNet18Model
from models.vision import ResNet18_v1
from models.vision import ResNet50TemporalModel
from models.audition import HearModel
from models.audition import SoundNet5Model

from models.audition import DualCamHybridModel
import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.contrib import layers
from tensorflow.python.ops import nn_ops

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

    features(FLAGS.mode, 0)


def features(dataset, savemap):
    _FRAMES_PER_SECOND = 12
    numcl = 10
    num_embedding = 128
    num_scenario = 61
    size1 = 12
    size2 = 16
    size11 = 224
    size22 = 298
    size33 = 3
    batch_size = 4
    if dataset == 'training':
        data_size = 6628
    elif dataset == 'validation':
        data_size = 1212
    else:
        data_size = 1399
    print('Computing features {}'.format(dataset))
    name1 = '{}{}_{}'.format(FLAGS.model1, FLAGS.model2, dataset)
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

    name_file = '/home/vsanguineti/Datasets/tfrecords/lists/{}.txt'.format(dataset)
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
        model1 = HearModel(input_shape=[200, 1, 257], num_classes=num_classes, embedding=1)
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
        logits2 = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    elif FLAGS.model2 == 'ResNet50':
        logits = model2.output
        expanded_shape = [-1, FLAGS.nr_frames, num_classes]
        logits2 = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    else:
        acousticlogits_multiple = model2.output
        expanded_shape = [-1, FLAGS.sample_length * _FRAMES_PER_SECOND, 12, 16, num_classes]
        logits2 = tf.reduce_mean(tf.reshape(acousticlogits_multiple, shape=expanded_shape), axis=1)
    # normalize matrix
    visuallogits = logits2  # tf.nn.l2_normalize(logits2, dim=[0, 1])
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
    # inner product between frame feature map and positive audio

    innerdot = visuallogits * acousticlogits_reshape
    product = innerdot * visuallogits
    # product2 = innerdot * acousticlogits_reshape
    # max instead of sum
    videoweighted = tf.reduce_sum(product, axis=[1, 2])
    audioweighted = tf.reduce_sum(acousticlogits_reshape, axis=[1, 2])
    # videovector = tf.reduce_sum(visuallogits, axis=[1, 2])
    # videoweighted = tf.squeeze(videoweighted, axis=[1, 2])
    productvectnorm = tf.nn.l2_normalize(videoweighted, dim=1)
    productvectnorm2 = tf.nn.l2_normalize(audioweighted, dim=1)
    # videovector = tf.nn.l2_normalize(videovector, dim=1)
    total_size = 0
    batch_count = 0

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

        if FLAGS.init_checkpoint is None:
            print('{} - Initializing student model'.format(datetime.now()))
            # model1.init_model(session, FLAGS.init_checkpoint)
            # model2.init_model(session, FLAGS.init_checkpoint)
            logits_init_op = tf.variables_initializer(var_list + var_list2)
            # Initialize the new logits layer
            session.run(logits_init_op)
            print('{} - Done'.format(datetime.now()))
        else:
            print('{} - Restoring student model'.format(datetime.now()))
            saver = tf.train.Saver(var_list=var_list + var_list2)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))

        dataset_videoaudio_list_features = np.zeros([data_size, num_embedding], dtype=float)
        dataset_labels = np.zeros([data_size, numcl], dtype=int)
        dataset_scenario = np.zeros([data_size, num_scenario], dtype=int)
        if savemap:
            dataset_map = np.zeros([data_size, size1, size2, num_embedding], dtype=float)

        session.run(train_iterat.initializer)
        while True:
            try:
                start_time = datetime.now()
                print('{} - Processing batch {}'.format(start_time, batch_count + 1))
                labels_data, scenario_data, features_audio, features_video, product = session.run(
                    [label, scenario, productvectnorm2, productvectnorm, innerdot],
                    feed_dict={handle: train_handle,
                               model1.network['keep_prob']: 1.0,
                               model2.network['keep_prob']: 1.0,
                               model1.network['is_training']: 0,
                               model2.network['is_training']: 0})
                batchnum = labels_data.shape[0]
                # copy block of data
                features = features_audio + features_video#np.concatenate((features_video, features_audio), axis=1)
                dataset_videoaudio_list_features[total_size:total_size + batchnum, :] = features
                dataset_labels[total_size:total_size + batchnum, :] = labels_data
                dataset_scenario[total_size:total_size + batchnum, :] = scenario_data
                if savemap:
                    # take one image for each block
                    dataset_map[total_size:total_size + batchnum, :, :, :] = product
                # increase number of data
                total_size += batchnum
                end_time = datetime.now()
                print('{} - Completed in {} seconds'.format(end_time, (end_time - start_time).total_seconds()))
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1
    if os.path.exists(data_dir1):
        print("Features already computed!")
    else:
        os.makedirs(data_dir1)  # mkdir creates one directory, makedirs all intermediate directories
    np.save('{}/{}{}_{}_data.npy'.format(data_dir1, FLAGS.model1, FLAGS.model2, dataset), dataset_videoaudio_list_features)
    np.save('{}/{}{}_{}_labels.npy'.format(data_dir1, FLAGS.model1, FLAGS.model2, dataset), dataset_labels)
    np.save('{}/{}{}_{}_scenario.npy'.format(data_dir1, FLAGS.model1, FLAGS.model2, dataset), dataset_scenario)
    if savemap:
        np.save('{}/{}_{}_maps.npy'.format(data_dir1, FLAGS.model1, FLAGS.model2, dataset), dataset_map)

    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))


# def generate_triplets(data1, data2):
#     # data1 is that with half data, data2 has positive and negative samples
#     half = tf.shape(data1)[0]
#     positive = tf.slice(data2, [0, 0], [half, tf.shape(data2)[1]])
#     negative = tf.slice(data2, [half, 0], [half, tf.shape(data2)[1]])
#     return data1, positive, negative

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