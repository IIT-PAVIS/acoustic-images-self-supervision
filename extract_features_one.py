from datetime import datetime
from dataloader.actions_data_old import ActionsDataLoader
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
flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can be one of \'DualCamNet\', or \'ResNet50\'')
flags.DEFINE_integer('temporal_pooling', 1, 'Temporal pooling')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('num_classes', None, 'Number of classes')
flags.DEFINE_integer('nr_frames', 2*12, 'Number of frames')#12*FLAGS.sample_length max
flags.DEFINE_integer('sample_length', 2, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('embedding', 1, 'hearnet from self supervised or supervised')
FLAGS = flags.FLAGS


def main(_):
    
    dataset = FLAGS.train_file.split('/')[-1]
    dataset = dataset.split('.')[0]
    name1 = '{}_{}'.format(FLAGS.model, dataset)
    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + [name1]) + '_' + name
    dataset = FLAGS.train_file

    numcl = 14
    batch_size = 8
    if dataset == 'training':
        data_size = 4530
    elif dataset == 'validation':
        data_size = 555
    else:
        data_size = 585

    if FLAGS.init_checkpoint is None:
        num_classes = None
    else:
        num_classes = 128

    is_dualcamnet = FLAGS.model == 'DualCamHybridNet'
    
    # Create data loaders according to the received program arguments
    print('{} - Creating data loaders'.format(datetime.now()))
    modalities = []
    if FLAGS.model == 'DualCamNet' or FLAGS.model == 'DualCamHybridNet':
        modalities.append(0)
    elif FLAGS.model == 'SoundNet5' or FLAGS.model == 'HearNet':
        modalities.append(1)
    elif FLAGS.model == 'ResNet50'or FLAGS.model == 'ResNet18' or FLAGS.model == 'ResNet18_v1' or FLAGS.model == 'TemporalResNet50':
        modalities.append(2)
        
    if FLAGS.model == 'ResNet18_v1' and FLAGS.nr_frames < 12*FLAGS.sample_length:
        random_pick = True
    else:
        random_pick = False
        
    normalize = False
    build_spectrogram = False
    if FLAGS.model == 'HearNet':
        normalize = True
        build_spectrogram = True
    train_file = '/home/vsanguineti/Datasets/dualcam_actions_dataset/30_seconds/lists/{}.txt'.format(dataset)
    with tf.device('/cpu:0'):
        train_data = ActionsDataLoader(train_file, 'inference', batch_size,
                                       num_epochs=1, normalize=normalize, build_spectrogram=build_spectrogram,
                                       number_of_crops=1, random_pick=random_pick,
                                       sample_rate=22050, total_length=2,
                                       sample_length=FLAGS.sample_length,
                                       buffer_size=10, shuffle=False, modalities=modalities, nr_frames=FLAGS.nr_frames)
        # iterator = train_data.data.make_one_shot_iterator()
        # next_batch = iterator.get_next()

    # Build model
    print('{} - Building model'.format(datetime.now()))
    numcl2 = 10
    with tf.device('/gpu:0'):
        if FLAGS.model == 'ResNet50':
            model = ResNet50Model(input_shape=[224, 298, 3], num_classes=numcl2)
        elif FLAGS.model == 'ResNet18_v1':
            model = ResNet18_v1(input_shape=[224, 298, 3], num_classes=numcl2, map=FLAGS.embedding)
        elif FLAGS.model == 'ResNet18':
            model = ResNet18Model(input_shape=[224, 298, 3], num_classes=numcl2, nr_frames=12)
        elif FLAGS.model == 'TemporalResNet50':
            model = ResNet50TemporalModel(input_shape=[224, 298, 3], num_classes=numcl2, nr_frames=12)
        elif FLAGS.model == 'DualCamHybridNet':
            model = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=numcl2, embedding=FLAGS.embedding)
        elif FLAGS.model == 'SoundNet':
            model = SoundNet5Model(input_shape=[22050 * 2, 1, 1], num_classes=numcl2)
        elif FLAGS.model == 'HearNet':
            model = HearModel(input_shape=[200, 1, 257], num_classes=numcl2, embedding=FLAGS.embedding)
        else:
            # Not necessary but set model to None to avoid warning about using unassigned local variable
            model = None
            raise ValueError('Unknown model type')
    handle = tf.placeholder(tf.string, shape=())
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.data.output_types,
                                                   train_data.data.output_shapes)
    train_iterat = train_data.data.make_initializable_iterator()
    next_batch = iterator.get_next()

    datashape = [model.height, model.width, model.channels]
    #data, label =_retrieve_batch(next_batch, datashape)
    data = tf.reshape(next_batch[modalities[0]],
                      shape=[-1, datashape[0], datashape[1], datashape[2]])

    label = tf.reshape(next_batch[3],
                       shape=[-1, 14])
    model._build_model(data)
        
    if FLAGS.model == 'ResNet18_v1':
        logits = model.network['embedding']
        logits = tf.squeeze(logits, [1, 2])
        expanded_shape = [-1, FLAGS.nr_frames, num_classes]
        logits = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    elif FLAGS.model == 'ResNet50':
        logits = model.output
        expanded_shape = [-1, FLAGS.nr_frames, num_classes]
        logits = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    elif FLAGS.model == 'DualCamHybridNet' and FLAGS.temporal_pooling:
        logits = model.network[8]
        expanded_shape = [-1, FLAGS.sample_length * 12, num_classes]
        logits = tf.reduce_mean(tf.reshape(logits, shape=expanded_shape), axis=1)
    elif FLAGS.model == 'SoundNet5':
        logits = model.output
    elif FLAGS.model == 'HearNet':
        logits = model.network['hear_net/fc2']
    else:
        logits = model.output


    # if is_dualcamnet:
    #     acoustic_data = tf.reshape(next_batch[modalities[0]], shape=[-1, model.height, model.width, model.channels])
    #     data, labels_data = session.run([
    #         acoustic_data,
    #         next_batch[3]
    #     ])
    #     logits = model.network['DualCamNet/fc2']
    # elif FLAGS.model == 'HearNet':
    #     acoustic_data = tf.reshape(next_batch[modalities[0]], shape=[-1, model.height, model.width, model.channels])
    #     data, labels_data = session.run([
    #         acoustic_data,
    #         next_batch[3]
    #     ])
    #     logits = model.network['DualCamNet/fc2']
    # elif FLAGS.model == 'ResNet18_v1':
    #     video_data = tf.reshape(next_batch[modalities[0]], shape=[-1, model.height, model.width, model.channels])
    #     data, labels_data = session.run([video_data, next_batch[3]])
    #     logits = model.network['final_conv']
    # else:
    #     video_data = tf.reshape(next_batch[modalities[0]], shape=[-1, model.height, model.width, model.channels])
    #     data, labels_data = session.run([video_data, next_batch[3]])
    #     logits = model.network['predictions']
    #     logits = tf.squeeze(logits, [1, 2])

    
    total_size = 0
    batch_count = 0

    print('{} - Starting'.format(datetime.now()))

    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        train_handle = session.run(train_iterat.string_handle())
        # Initialize student model
        if FLAGS.init_checkpoint is None:
            print('{} - Initializing student model'.format(datetime.now()))
            model.init_model(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))
        else:
            print('{} - Restoring student model'.format(datetime.now()))
            if is_dualcamnet:
                var_list = slim.get_variables(model.scope)
            else:
                var_list = slim.get_model_variables(model.scope)
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))

        features_list = np.zeros([data_size, num_classes], dtype=float)
        labels_list = np.zeros([data_size, numcl], dtype=int)
        session.run(train_iterat.initializer)
        while True:
            try:
                start_time = datetime.now()
                print('{} - Processing batch {}'.format(start_time, batch_count + 1))
                labels_data, features = session.run([label, logits],
                                       feed_dict={handle: train_handle,
                                                  model.network['keep_prob']: 1.0,
                                                  model.network['is_training']: 0})

                batchnum = labels_data.shape[0]
                features_list[total_size:total_size + batchnum, :] = features
                labels_list[total_size:total_size + batchnum, :] = labels_data
                total_size += labels_data.shape[0]
                end_time = datetime.now()
                print('{} - Completed in {} seconds'.format(end_time, (end_time - start_time).total_seconds()))
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1

    if os.path.exists(data_dir):
        print("Features already computed!")
    else:
        os.makedirs(data_dir)  # mkdir creates one directory, makedirs all intermediate directories


    np.save('{}/{}_{}_data.npy'.format(data_dir, FLAGS.model, dataset), features_list)
    np.save('{}/{}_{}_labels.npy'.format(data_dir, FLAGS.model, dataset), labels_list)

    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))

if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()

# --model
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