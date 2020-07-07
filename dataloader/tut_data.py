import tensorflow as tf
import numpy as np
import os.path
import math


class TUTDataLoader(object):
    def __init__(self, txt_file, mode, batch_size, num_classes=None, num_epochs=None, shuffle=True,
                 buffer_size=10, sample_rate=22050, min_length=10,
                 sample_length=2, number_of_crops=5, normalize=False, spectrogram=False):
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.sample_length = sample_length
        self.min_length = min_length
        self.sample_rate = sample_rate
        self.number_of_crops = number_of_crops
        self.normalize = normalize
        self.batch_size = batch_size
        self.frame_length = 440
        self.frame_step = 219  # 220
        self.fft_length = 512
        # self.frame_length = np.floor(sample_rate * 0.020).astype(int)  # 440
        # self.frame_step = np.floor(sample_rate * 0.010).astype(int)  # 220
        # self.fft_length = (2 ** (np.floor(np.log2(self.frame_length)) + 1)).astype(int)  # 512
        # self.frame_length = tf.cast(self.frame_length, dtype=tf.int32)
        # self.frame_step = tf.cast(self.frame_step, dtype=tf.int32)
        # self.fft_length = tf.cast(self.fft_length, dtype=tf.int32)
        # retrieve the data from the folder of tfrecords

        dataset = self.txt_file
        if dataset == 'training':
            self.tfrecordsfolder = '/home/vsanguineti/Datasets/TUT/tfrecords/recordstrain10seconds22050/'
        if dataset == 'testing':
            self.tfrecordsfolder = '/home/vsanguineti/Datasets/TUT/tfrecords/recordstest10seconds22050/'
        if dataset == 'validation':
            self.tfrecordsfolder = '/home/vsanguineti/Datasets/TUT/tfrecords/recordsevaluate10seconds22050/'
           
        
        self.data_size = self._tfrecord_fn()
        
        # initial shuffling of the file and label lists is done when tfrecords are saved
        
        # create dataset of tfrecordsfile
        
        # load statistics
        if normalize:
            stats_dir = '/home/vsanguineti/Datasets/TUT/tfrecords/statsDCASE/'
            mean = np.load('{}/global_mean.npy'.format(stats_dir))
            std = np.load('{}/global_std_dev.npy'.format(stats_dir))
            if spectrogram:
                self.global_mean = tf.tile(
                    tf.expand_dims(
                        input=tf.expand_dims(input=tf.convert_to_tensor(mean), axis=0),
                        axis=0
                    ),
                    [200, 1, 1]
                )
                
                self.global_standard_deviation = tf.tile(
                    tf.expand_dims(
                        input=tf.expand_dims(input=tf.convert_to_tensor(std), axis=0),
                        axis=0
                    ),
                    [200, 1, 1]
                )
        data = tf.data.TFRecordDataset(self.img_paths)
        
        # parse dataset
        if mode == 'training':
            num_samples = self.data_size * self.number_of_crops
            # shuffle at the beginning of each epoch
            data = data.map(self._map_function, num_parallel_calls=4)
            data = data.map(self._map_function_training, num_parallel_calls=4)
        
        elif mode == 'inference':
            num_samples = self.data_size * int(self.min_length / self.sample_length)
            data = data.map(self._map_function, num_parallel_calls=4)
            data = data.map(self._map_function_inference, num_parallel_calls=4)
        else:
            raise ValueError('Unknown mode')
        self.num_samples = num_samples
        data = data.apply(tf.contrib.data.unbatch())  # groups of segments are flatten
        
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
            # # build spectrogram for each waveform
        if spectrogram == True:
            data = data.map(self._map_function_build_spectrogram, num_parallel_calls=4)
        if normalize:
            data = data.map(self._map_func_audio_samples_mean_norm, num_parallel_calls=4)
        # create batched dataset that repeats indefinitely
        data = data.batch(self.batch_size)
        data = data.repeat(num_epochs)
        data = data.prefetch(buffer_size=buffer_size)
        
        self.data = data
    
    def _tfrecord_fn(self):
        """Read the content of the folder of the tfrecords and store it into lists."""
        self.img_paths = []
        for file in os.listdir(self.tfrecordsfolder):
            if file.endswith(".tfrecords"):
                self.img_paths.append(self.tfrecordsfolder + str(file))
        c = 0
        for fn in self.img_paths:
            for record in tf.python_io.tf_record_iterator(fn):
                c += 1
        print(c)  # pieces of 10 seconds inside all tfrecords
        return c
    
    def _map_function_training(self, audio_tensor, one_hot):
        """Take random crops of minimum length in tfrecords"""
        len = np.floor(self.sample_rate * self.min_length).astype('int')  # pieces of 10 s
        # Do random crop of minimum length
        segment = np.floor(self.sample_length * self.sample_rate).astype('int')
        lengthtodrawsamples = len - segment
        
        # Compute indices of random crops
        index = np.random.randint(lengthtodrawsamples, size=self.number_of_crops)
        # Crop files tensor according to the pre-computed indices
        cropped_audio_tensor = list()
        for i in range(0, index.shape[0]):
            cropped_audio_tensor.append(
                tf.slice(audio_tensor, [tf.cast(index[i], dtype=tf.int64)], [tf.cast(segment, dtype=tf.int64)]))
        cropped_audio = tf.stack(cropped_audio_tensor)
        # Convert label number into one-hot-encoding
        one_hot = tf.tile(one_hot, [self.number_of_crops])
        one_hot = tf.reshape(one_hot, [-1, self.num_classes])
        return cropped_audio, one_hot
    
    def _map_function_inference(self, audio_tensor, one_hot):
        """Take equispaced crops of minimum length in tfrecord"""
        len = np.floor(self.sample_rate * self.min_length).astype('int')  # pieces of 10 s
        # Do random crop of minimum length
        segment = np.floor(self.sample_length * self.sample_rate).astype('int')
        crops = np.floor(self.min_length / self.sample_length).astype('int')
        
        # Compute indices of random crops
        index = np.arange(crops)
        index = index * segment
        # Crop files tensor according to the pre-computed indices
        cropped_audio_tensor = list()
        for i in range(0, index.shape[0]):
            cropped_audio_tensor.append(
                tf.slice(audio_tensor, [tf.cast(index[i], dtype=tf.int64)], [tf.cast(segment, dtype=tf.int64)]))
        cropped_audio = tf.stack(cropped_audio_tensor)
        # Convert label number into one-hot-encoding
        one_hot = tf.tile(one_hot, [crops])
        one_hot = tf.reshape(one_hot, [-1, self.num_classes])
        return cropped_audio, one_hot
    
    def _map_function(self, serialized_example):
        """Input parser for samples of the validation/testing set."""
        feature_set = {'label': tf.FixedLenFeature([], tf.string),
                       'audio_raw': tf.FixedLenFeature([], tf.string)}
        audio_features = tf.parse_single_example(serialized_example, feature_set)
        labels = tf.decode_raw(audio_features['label'], tf.int64)
        audio = tf.decode_raw(audio_features['audio_raw'], tf.float32)
        min_length = np.floor(self.sample_rate * self.min_length).astype('int')
        # Reshape image data into the original shape
        #  Tke random crop of minimum length in tfrecords
        label = tf.reshape(labels, [])
        cropped_audio_tensor = tf.reshape(audio, [min_length])
        # Convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes, dtype=tf.int64)
        return cropped_audio_tensor, one_hot
    
    def _map_function_build_spectrogram(self, audio_wav, label):
        """Input mapper to build spectrogram from waveform audio."""
        
        audio_stfts = tf.contrib.signal.stft(audio_wav, frame_length=self.frame_length,
                                             frame_step=self.frame_step, fft_length=self.fft_length)
        
        magnitude_spectrograms = tf.expand_dims(tf.abs(audio_stfts), 1)
        
        return magnitude_spectrograms, label
    
    def _map_function_min_max_spectrogram_norm(self, audio_samples, label):
        """Input mapper to apply min-max normalization to the audio images."""
        
        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_min),
                                       tf.subtract(self.global_max, self.global_min))
        
        return audio_samples_norm, label
    
    def _map_func_audio_samples_mean_norm(self, audio_samples, label):
        """Input mapper to apply min-max normalization to the audio samples."""
        
        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_mean), self.global_standard_deviation)
        
        return audio_samples_norm, label
    
    @property
    def total_batches(self):
        total_batches = int(math.floor(self.num_samples / self.batch_size))
        return total_batches