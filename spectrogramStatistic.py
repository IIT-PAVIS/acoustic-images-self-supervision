from __future__ import division
from datetime import datetime
import glob
import argparse
import tensorflow as tf
import numpy as np
import os
import sys
from dataloader.actions_data import ActionsDataLoader
from dataloader.tut_data import TUTDataLoader
# "/data/dualcam-net/30_seconds"
# --train_file
# "/data/TUT/tfrecords/recordstrain10seconds22050/"
# --batch_size
# 8
# --num_channels
# 1
# --network
# Sound
# --network
# Hear
# /data/vsanguineti/dualcam_actions_dataset/30_seconds_location_3
# --train_file
# /data/vsanguineti/dualcam_actions_dataset/30_seconds_location_3/lists/training.txt
# --batch_size
# 8
# --num_channels
# 257
parser = argparse.ArgumentParser()
parser.add_argument('first_dir', help='Directory holding all lists containing generated lists and computed statistics', type=str)
parser.add_argument('--batch_size', help='Size of the mini-batch', type=int, default=64)
parser.add_argument('--num_channels', help='Number of channels from the spectrogram', type=int, default=257)
parser.add_argument('--folder', help='folder for TUT', type=str, default='/data/TUT/tfrecords/recordstrain10seconds22050/')
parsed_args = parser.parse_args()

first_dir = parsed_args.first_dir
batch_size = parsed_args.batch_size
num_channels = parsed_args.num_channels
folder = parsed_args.folder
# network = parsed_args.network
spectrogram = True
# data_dirs = sorted(glob.glob('{}/30_seconds*'.format(first_dir)))

# for root_dir in data_dirs:
root_dir = first_dir
stats_dir = '{}/statsDCASE'.format(root_dir)
lists_dir = '{}/lists'.format(root_dir)
train_file = '{}/training.txt'.format(lists_dir)
if os.path.exists(stats_dir) and os.listdir(stats_dir):
    print("Statistics already computed!")
    sys.exit(0)

if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)

with tf.device('/cpu:0'):
    # train_data = DataLoader(train_file, 'inference', batch_size, total_length=1, sample_length=1, num_epochs=1,
    #                         num_channels=12)
    train_data = TUTDataLoader(train_file, folder, 'inference', batch_size, num_classes=10, num_epochs=1, shuffle=False, spectrogram=spectrogram, normalize=False)
    # train_data = ActionsDataLoader(train_file, 'inference', batch_size, num_epochs=1, normalize=False, sample_rate=22050, total_length=2, sample_length=2, number_of_crops=1,
    #                                buffer_size=20, shuffle=False, build_spectrogram=spectrogram, modalities=[1])
    iterator = train_data.data.make_one_shot_iterator()
    next_batch = iterator.get_next()

data_size = train_data.data_size

global_min_value = np.full(num_channels, np.inf, np.float64)
global_max_value = np.full(num_channels, 0.0, np.float64)
global_sum_value = np.full(num_channels, 0.0, np.float64)
global_sum_squared_value = np.full(num_channels, 0.0, np.float64)
# http://mathcentral.uregina.ca/qq/database/qq.09.02/carlos1.html
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

global_min_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))
global_max_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))
global_sum_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))
global_sum_squared_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))

batch_min = tf.cast(tf.reduce_min(next_batch[0], axis=(0, 1, 2)), dtype=tf.float64)#[0] tut [1] actions
global_min = tf.reduce_min(tf.stack([global_min_curr, batch_min]), axis=0)

batch_max = tf.cast(tf.reduce_max(next_batch[0], axis=(0, 1, 2)), dtype=tf.float64)
global_max = tf.reduce_max(tf.stack([global_max_curr, batch_max]), axis=0)

batch_sum = tf.cast(tf.reduce_sum(next_batch[0], axis=(0, 1, 2)), dtype=tf.float64)
global_sum = tf.add(global_sum_curr, batch_sum)

batch_sum_squared = tf.cast(tf.reduce_sum(tf.square(next_batch[0]), axis=(0, 1, 2)), dtype=tf.float64)
global_sum_squared = tf.add(global_sum_squared_curr, batch_sum_squared)

batch_count = 0

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:

    print('{} Starting'.format(datetime.now()))

    while True:
        try:
            start_time = datetime.now()
            print('{} Processing batch {}'.format(start_time, batch_count + 1))
            global_min_value, global_max_value, global_sum_value, global_sum_squared_value = session.run(
                [global_min, global_max, global_sum, global_sum_squared],
                feed_dict={global_min_curr: global_min_value,
                           global_max_curr: global_max_value,
                           global_sum_curr: global_sum_value,
                           global_sum_squared_curr: global_sum_squared_value})
            end_time = datetime.now()
            print('{} Completed in {} seconds'.format(end_time, (end_time - start_time).total_seconds()))
        except tf.errors.OutOfRangeError:
            print('{} Cancelled'.format(datetime.now()))
            break
        batch_count += 1

    print('{} Completed'.format(datetime.now()))

if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

np.save('{}/global_min.npy'.format(stats_dir), global_min_value.astype(np.float32))
np.save('{}/global_max.npy'.format(stats_dir), global_max_value.astype(np.float32))
np.save('{}/global_sum.npy'.format(stats_dir), global_sum_value.astype(np.float32))
np.save('{}/global_sum_squared.npy'.format(stats_dir), global_sum_squared_value.astype(np.float32))
n = data_size * 200 * 1 * 257
#500 * 1 * 257
global_mean = global_sum_value / n
global_var = (global_sum_squared_value - (global_sum_value ** 2) / n) / n
global_std_dev = np.sqrt(global_var)

np.save('{}/global_mean.npy'.format(stats_dir), global_mean.astype(np.float32))
np.save('{}/global_std_dev.npy'.format(stats_dir), global_std_dev.astype(np.float32))