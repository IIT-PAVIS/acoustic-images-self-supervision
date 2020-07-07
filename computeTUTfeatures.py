from datetime import datetime
from dataloader.tut_data import TUTDataLoader
from models.audition import HearModel
from models.audition import SoundNet5Model
import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.python.ops import nn_ops
flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model', None, 'Model type, it can be one of \'SoundNet\', or \'HearNet\'')
flags.DEFINE_string('train_file', None, 'File for training data')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model initialization')
flags.DEFINE_integer('num_classes', 128, 'Number of classes')
flags.DEFINE_integer('sample_length', 2, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('embedding', 1, 'hearnet from self supervised or supervised')
FLAGS = flags.FLAGS


def main(_):
    dataset = FLAGS.train_file
    numcl = 10
    batch_size = 4
    if dataset == 'training':
        data_size = 30450
    elif dataset == 'validation':
        data_size = 12590
    else:
        data_size = 12590
    print('Computing features {}'.format(dataset))
    name1 = '{}_{}'.format(FLAGS.model, dataset)
    s = FLAGS.init_checkpoint.split('/')[-1]
    name = (s.split('_')[1]).split('.ckpt')[0]
    data_dir = str.join('/', FLAGS.init_checkpoint.split('/')[:-1] + ['TUT'] + [name1]) + '_' + name
    embedding = FLAGS.embedding
    if os.path.exists(data_dir):
        print("Features already computed!")
        sys.exit(0)
    else:
        os.makedirs(data_dir)  # mkdir creates one directory, makedirs all intermediate directories
    
    if FLAGS.init_checkpoint is None:
        num_classes = None
    else:
        num_classes = FLAGS.num_classes
    
    # Create data loaders according to the received program arguments
    print('{} - Creating data loaders'.format(datetime.now()))
    
    normalize = False
    build_spectrogram = False
    if FLAGS.model == 'HearNet':
        normalize = True
        build_spectrogram = True
    
    with tf.device('/cpu:0'):
        train_data = TUTDataLoader(FLAGS.train_file, 'inference', batch_size, num_classes=10, num_epochs=1,
                                   shuffle=False, spectrogram=build_spectrogram, normalize=normalize)
        # iterator = train_data.data.make_one_shot_iterator()
        # next_batch = iterator.get_next()
    
    # Build model
    print('{} - Building model'.format(datetime.now()))
    
    with tf.device('/gpu:0'):
        if FLAGS.model == 'SoundNet':
            model = SoundNet5Model(input_shape=[22050 * 2, 1, 1], num_classes=num_classes)
        elif FLAGS.model == 'HearNet':
            model = HearModel(input_shape=[200, 1, 257], num_classes=num_classes, embedding=embedding)
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
    # data, label =_retrieve_batch(next_batch, datashape)
    data = tf.reshape(next_batch[0],
                      shape=[-1, datashape[0], datashape[1], datashape[2]])

    label = tf.reshape(next_batch[1],
                       shape=[-1, 10])
    model._build_model(data)
    logits = model.network['hear_net/fc2']
    if embedding:
        logits = nn_ops.relu(logits)
    
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
            var_list = slim.get_model_variables(model.scope)
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, FLAGS.init_checkpoint)
            print('{} - Done'.format(datetime.now()))

        dataset_list_features = np.zeros([data_size, 128], dtype=float)
        dataset_labels = np.zeros([data_size, numcl], dtype=int)
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
                #copy block of data
                dataset_list_features[total_size:total_size+batchnum, :] = features
                dataset_labels[total_size:total_size+batchnum, :] = labels_data
                #increase number of data
                total_size += batchnum
                end_time = datetime.now()
                print('{} - Completed in {} seconds'.format(end_time, (end_time - start_time).total_seconds()))
            except tf.errors.OutOfRangeError:
                break
            batch_count += 1

    
    np.save('{}/{}_{}_data.npy'.format(data_dir, FLAGS.model, dataset), dataset_list_features)
    np.save('{}/{}_{}_labels.npy'.format(data_dir, FLAGS.model, dataset), dataset_labels)
    
    print('{} - Completed, got {} samples'.format(datetime.now(), total_size))


if __name__ == '__main__':
    flags.mark_flags_as_required(['train_file'])
    tf.app.run()

# --model
# HearNet
# --train_file
# /media/vsanguineti/TOSHIBAEXT/tfrecords/lists/training.txt
# --init_checkpoint
# /media/vsanguineti/TOSHIBAEXT/checkpoints/Audio6410-4_1/model_10.ckpt
# --folder
# /data/TUT/tfrecords/recordstrain10seconds22050/