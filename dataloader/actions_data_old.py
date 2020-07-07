from __future__ import division
import librosa
import tensorflow as tf
import numpy as np
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

_NUMBER_OF_MICS = 128
_NUMBER_OF_SAMPLES = 1024
_FRAMES_PER_SECOND = 12
_NUMBER_OF_CHANNELS = 12
_NUM_ACTIONS = 14
_NUM_LOCATIONS = 3
_NUM_SUBJECTS = 9
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


class ActionsDataLoader(object):
    
    def __init__(self, txt_file, mode, batch_size, sample_rate=22050, total_length=30, sample_length=5,
                 number_of_crops=20, buffer_size=1, num_epochs=None, shuffle=False, normalize=False,
                 random_pick=False, build_spectrogram=False, modalities=None, nr_frames=12):
        
        # self.seed = tf.placeholder(tf.int64, shape=(), name='data_seed')   =epoch
        self.nr_frames = nr_frames
        self.txt_file = txt_file
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.total_length = total_length
        self.sample_length = sample_length
        self.number_of_crops = number_of_crops
        
        self.frame_length = 440
        self.frame_step = 219  # 220
        self.fft_length = 512
        
        self.include_audio_images = modalities is None or 0 in modalities
        self.include_audio_data = modalities is None or 1 in modalities
        self.include_video_data = modalities is None or 2 in modalities
        
        assert txt_file is not None
        assert total_length % sample_length == 0
        assert (self.include_audio_images or self.include_audio_data or self.include_video_data) is True
        # TODO Fix this assertion to check that there are enough samples to provide the required number of crops
        # assert number_of_crops <= total_length - sample_length
        
        # load statistics
        # if normalize and self.include_audio_images:
        #     self.global_min, self.global_max, self.threshold = self._load_acoustic_images_stats()
        if normalize and self.include_audio_data:
            # self.global_min, self.global_max, self.global_standard_deviation,\
            #     self.global_mean= self._load_spectrogram_stats()
            self.global_standard_deviation, self.global_mean = self._load_spectrogram_stats()
        
        # retrieve the data from the text file
        # how many files are in file, add them to img_paths
        self.data_size = self._read_txt_file()
        
        # convert lists to TF tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.files = tf.data.Dataset.from_tensor_slices(self.img_paths)
        
        assert self.data_size % self.total_length == 0
        
        if mode == 'training':
            num_samples = int(self.data_size / self.total_length) * self.number_of_crops
            self.files = self.files.batch(self.total_length)
            self.files = self.files.flat_map(self._map_files_training)
            self.files = self.files.apply(tf.contrib.data.unbatch())
        elif mode == 'inference':
            num_samples = int(self.data_size / self.total_length) * int(self.total_length / self.sample_length)
        else:
            raise ValueError('Unknown mode')
        
        self.num_samples = num_samples
        
        # shuffle `num_samples` blocks of files and repeat them `num_epochs`
        if shuffle:
            self._shuffle_and_repeat_lists(num_epochs, num_samples)
        
        # create data set
        data = self.files.flat_map(lambda ds: tf.data.TFRecordDataset(ds, compression_type='GZIP'))
        
        # parse data set
        data = data.map(self._parse_sequence, num_parallel_calls=4)
        
        # prefetch `buffer_size` batches of elements of the dataset
        data = data.prefetch(buffer_size=buffer_size * batch_size * sample_length)
        
        # batch elements in groups of `sample_length` seconds
        data = data.batch(self.sample_length)
        data = data.map(self._map_function)
        
        # build waveform for each sequence of `sample_length`
        if self.include_audio_data:
            data = data.map(self._map_func_audio_samples_build_wav, num_parallel_calls=4)
        
        # pick random frames from video sequence
        if random_pick:
            data = data.map(self._map_func_video_images_pick_frames, num_parallel_calls=4)
        
        # build spectrogram for each waveform and normalize it
        if self.include_audio_data and build_spectrogram:
            data = data.map(self._map_func_audio_samples_build_spectrogram, num_parallel_calls=4)
        if self.include_audio_data and normalize:
            data = data.map(self._map_func_audio_samples_mean_norm, num_parallel_calls=4)

        # for multispectral acoustic images
        # if self.include_audio_images and normalize:
        #     data = data.map(self._map_func_audio_images_piece_wise, num_parallel_calls=4)
        #     data = data.map(self._map_func_audio_images_min_max_norm, num_parallel_calls=4)
        
        if self.include_video_data:
            data = data.map(self._map_func_video_images, num_parallel_calls=4)
        
        # create batched dataset
        data = data.batch(batch_size)
        
        self.data = data
    
    def _load_acoustic_images_stats(self):
        """Load acoustic images statistics."""
        
        stats_dir = str.join('/', self.txt_file.split('/')[:-2] + ['stats'])
        threshold_value = 1e12
        min_value = np.load('{}/global_min.npy'.format(stats_dir)).clip(None, threshold_value)
        max_value = np.load('{}/global_max.npy'.format(stats_dir)).clip(None, threshold_value)
        
        global_min = tf.tile(
            tf.expand_dims(
                input=tf.expand_dims(input=tf.expand_dims(input=tf.convert_to_tensor(min_value), axis=0), axis=0),
                axis=0
            ),
            [_FRAMES_PER_SECOND * self.sample_length, 36, 48, 1]
        )
        
        global_max = tf.tile(
            tf.expand_dims(
                input=tf.expand_dims(input=tf.expand_dims(input=tf.convert_to_tensor(max_value), axis=0), axis=0),
                axis=0
            ),
            [_FRAMES_PER_SECOND * self.sample_length, 36, 48, 1]
        )
        
        threshold = tf.constant(
            value=threshold_value,
            dtype=tf.float32,
            shape=[_FRAMES_PER_SECOND * self.sample_length, 36, 48, _NUMBER_OF_CHANNELS]
        )
        
        return threshold, global_min, global_max
    
    def _load_spectrogram_stats(self):
        """Load spectrogram statistics."""
        
        stats_dir = str.join('/', self.txt_file.replace('//', '/').split('/')[:-2] + ['stats2s'])  # statsHearMean
        mean = np.load('{}/global_mean.npy'.format(stats_dir))
        std = np.load('{}/global_std_dev.npy'.format(stats_dir))
        global_mean = tf.tile(
            tf.expand_dims(
                input=tf.expand_dims(input=tf.convert_to_tensor(mean), axis=0),
                axis=0
            ),
            [200, 1, 1]
        )
        
        global_standard_deviation = tf.tile(
            tf.expand_dims(
                input=tf.expand_dims(input=tf.convert_to_tensor(std), axis=0),
                axis=0
            ),
            [200, 1, 1]
        )
        
        return global_standard_deviation, global_mean  # ,global_min, global_max
    
    def _read_txt_file(self):
        """Read the content of the text file and store it into a list."""
        self.img_paths = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.rstrip('\n')
                self.img_paths.append(img_path)
        return len(self.img_paths)
    
    def _shuffle_and_repeat_lists(self, num_epochs, num_samples):
        """Shuffle and repeat the list of paths."""
        self.files = self.files.batch(self.sample_length)
        self.files = self.files.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
        self.files = self.files.repeat(num_epochs)
        self.files = self.files.apply(tf.contrib.data.unbatch())
    
    def _parse_sequence(self, sequence_example_proto):
        """Input parser for samples of the training set."""
        
        context_features = {'action': tf.FixedLenFeature([], tf.int64),
                            'location': tf.FixedLenFeature([], tf.int64),
                            'subject': tf.FixedLenFeature([], tf.int64)}
        sequence_features = {}
        
        if self.include_audio_images:
            context_features.update({
                'audio_image/height': tf.FixedLenFeature([], tf.int64),
                'audio_image/width': tf.FixedLenFeature([], tf.int64),
                'audio_image/depth': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'audio/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })
        
        if self.include_audio_data:
            context_features.update({
                'audio_data/mics': tf.FixedLenFeature([], tf.int64),
                'audio_data/samples': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'audio/data': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })
        
        if self.include_video_data:
            context_features.update({
                'video/height': tf.FixedLenFeature([], tf.int64),
                'video/width': tf.FixedLenFeature([], tf.int64),
                'video/depth': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'video/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })
        
        # Parse single example
        parsed_context_features, parsed_sequence_features = tf.parse_single_sequence_example(sequence_example_proto,
                                                                                             context_features=context_features,
                                                                                             sequence_features=sequence_features)
        
        action = tf.cast(parsed_context_features['action'], tf.int32)
        location = tf.cast(parsed_context_features['location'], tf.int32)
        subject = tf.cast(parsed_context_features['subject'], tf.int32)
        
        if self.include_audio_images:
            # Retrieve parsed context features
            audio_height = tf.cast(parsed_context_features['audio_image/height'], tf.int32)
            audio_width = tf.cast(parsed_context_features['audio_image/width'], tf.int32)
            audio_depth = tf.cast(parsed_context_features['audio_image/depth'], tf.int32)
            # Retrieve parsed audio image features
            audio_image_decoded = tf.decode_raw(parsed_sequence_features['audio/image'], tf.float32)
            # Reshape decoded audio image
            audio_image_shape = tf.stack([-1, audio_height, audio_width, audio_depth])
            audio_images = tf.reshape(audio_image_decoded, audio_image_shape)
            audio_images = tf.image.flip_left_right(audio_images)
            audio_images = tf.image.flip_up_down(audio_images)
        else:
            audio_images = tf.zeros([], tf.int32)
        
        if self.include_audio_data:
            # Retrieve parsed context features
            num_mics = tf.cast(parsed_context_features['audio_data/mics'], tf.int32)
            num_samples = tf.cast(parsed_context_features['audio_data/samples'], tf.int32)
            # Retrieve parsed audio data features
            audio_sample_decoded = tf.decode_raw(parsed_sequence_features['audio/data'], tf.int32)
            # Reshape decoded audio data
            audio_sample_shape = tf.stack([-1, num_samples])  # num_mics, num_samples
            audio_samples = tf.reshape(audio_sample_decoded, audio_sample_shape)
        else:
            audio_samples = tf.zeros([], tf.int32)
        
        if self.include_video_data:
            # Retrieve parsed video image features
            video_image_decoded = tf.decode_raw(parsed_sequence_features['video/image'], tf.uint8)
            # Retrieve parsed context features
            video_height = tf.cast(parsed_context_features['video/height'], tf.int32)
            video_width = tf.cast(parsed_context_features['video/width'], tf.int32)
            video_depth = tf.cast(parsed_context_features['video/depth'], tf.int32)
            # Reshape decoded video image
            video_image_shape = tf.stack([-1, video_height, video_width, video_depth])  # 224, 298, 3
            video_images = tf.reshape(video_image_decoded, video_image_shape)
        else:
            video_images = tf.zeros([], tf.int32)
        
        return audio_images, audio_samples, video_images, action, location, subject
    
    def _map_files_training(self, files):
        """Input mapper for files of the training set."""
        
        # Compute indices of random crops
        shapes = tf.constant(self.total_length, dtype=tf.int32, shape=[self.number_of_crops])
        sizes = tf.constant(self.sample_length, dtype=tf.int32, shape=[self.number_of_crops])
        limit = shapes - sizes + 1
        offset = tf.random_uniform(tf.shape(shapes), dtype=sizes.dtype, maxval=sizes.dtype.max, seed=3) % limit
        
        # Crop files tensor according to the pre-computed indices
        cropped_files = tf.map_fn(lambda o: tf.slice(files, tf.convert_to_tensor([o]), [self.sample_length]), offset,
                                  dtype=files.dtype)
        
        return tf.data.Dataset.from_tensor_slices(cropped_files)
    
    def _map_function(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapping function."""
        
        # Convert labels into one-hot-encoded tensors
        action_encoded = tf.one_hot(
            tf.squeeze(tf.gather(action, tf.range(self.sample_length, delta=self.sample_length))), _NUM_ACTIONS)
        location_encoded = tf.one_hot(
            tf.squeeze(tf.gather(location, tf.range(self.sample_length, delta=self.sample_length))), _NUM_LOCATIONS)
        subject_encoded = tf.one_hot(
            tf.squeeze(tf.gather(subject, tf.range(self.sample_length, delta=self.sample_length))), _NUM_SUBJECTS)
        
        # Reshape audio_images to be the length of a full video of `sample_length` seconds
        if self.include_audio_images:
            reshaped_audio_images = tf.reshape(audio_images, [-1, 36, 48, _NUMBER_OF_CHANNELS])
        else:
            reshaped_audio_images = tf.zeros([], tf.int32)
        
        # Reshape audio_samples to be the length of a full video of `sample_length` seconds
        if self.include_audio_data:
            # reshaped_audio_samples = tf.reshape(audio_samples, [-1, _NUMBER_OF_MICS, _NUMBER_OF_SAMPLES])
            reshaped_audio_samples = tf.reshape(audio_samples, [-1, 1, _NUMBER_OF_SAMPLES])
        else:
            reshaped_audio_samples = tf.zeros([], tf.int32)
        
        # Reshape audio_samples to be the length of a full video of `sample_length` seconds
        if self.include_video_data:
            # reshaped_video_images = tf.reshape(video_images, [-1, 480, 640, 3])
            reshaped_video_images = tf.reshape(video_images, [-1, 224, 298, 3])  # 224, 298, 3
        else:
            reshaped_video_images = tf.zeros([], tf.int32)
        
        return reshaped_audio_images, reshaped_audio_samples, reshaped_video_images, action_encoded, location_encoded, subject_encoded
    
    def _map_func_video_images_pick_frames(self, audio_images, audio_samples, video_images, action, location, subject):
        """Pick nr_frames random frames."""
        
        selected_video_images = self._pick_random_frames(video_images)
        
        return audio_images, audio_samples, selected_video_images, action, location, subject
    
    def _pick_random_frames(self, video_images):
        num_frames = tf.shape(video_images)[0]  # how many images
        n_to_sample = tf.constant([self.nr_frames])  # how many to keep, 5 in temporal resnet 50, 8 in resnet18
        mask = self._sample_mask(num_frames, n_to_sample)  # pick n_to_sample
        frames = tf.boolean_mask(video_images, mask)  # keep element in ones position
        return frames
    
    def _sample_mask(self, num_frames, sample_size):
        # randomly choose between uniform or random sampling
        end = tf.subtract(num_frames, 1)  # last index
        indexes = tf.to_int32(tf.linspace(
            0.0, tf.to_float(end), sample_size[0]))  # uses linspace to draw uniformly samples between 0 and end
        # find indexes
        updates = tf.ones(sample_size, dtype=tf.int32)  # ones
        mask = tf.scatter_nd(tf.expand_dims(indexes, 1),
                             updates, tf.expand_dims(num_frames, 0))  # put ones in indexes positions
        
        compare = tf.ones([num_frames], dtype=tf.int32)  # ones in all num_frames
        mask = tf.equal(mask, compare)  # see where are ones
        return mask
    
    def _map_func_audio_samples_build_wav(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to build waveform audio from raw audio samples."""
        
        audio_wav = tf.py_func(self._build_wav_py_function, [audio_samples], tf.float32)
        
        return audio_images, audio_wav, video_images, action, location, subject
    
    def _build_wav_py_function(self, audio_data):
        """Python function to build a waveform audio from audio samples."""
        audio_data = audio_data.astype(np.float32)
        # concatenate 1/12
        audio_data = audio_data.flatten('C')
        
        # audio_data = audio_data / abs(
        #     max(audio_data.min(), audio_data.max(), key=abs))
        # size is correct because we take at least one second of data
        # Re-sample audio to 22 kHz
        audio_wav = librosa.core.resample(audio_data, audio_data.shape[0] / self.sample_length,
                                          self.sample_rate)
        # range between -1 and 1
        audio_wav = audio_wav / abs(max(audio_wav.min(), audio_wav.max(), key=abs))
        # Make range [-256, 256]
        audio_wav *= 256.0
        
        return audio_wav
    
    def _map_func_audio_images_piece_wise(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to apply piece-wise normalization to the audio images."""
        
        audio_images_norm = tf.where(tf.greater(audio_images, self.threshold), self.threshold, audio_images)
        
        return audio_images_norm, audio_samples, video_images, action, location, subject
    
    def _map_func_audio_images_min_max_norm(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to apply min-max normalization to the audio images."""
        
        audio_images_norm = tf.divide(tf.subtract(audio_images, self.global_min),
                                      tf.subtract(self.global_max, self.global_min))
        
        return audio_images_norm, audio_samples, video_images, action, location, subject
    
    def _map_func_audio_samples_min_max_norm(self, audio_images, audio_samples, video_images, action, location,
                                             subject):
        """Input mapper to apply min-max normalization to the audio samples."""
        
        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_min),
                                       tf.subtract(self.global_max, self.global_min))
        
        return audio_images, audio_samples_norm, video_images, action, location, subject
    
    def _map_func_audio_samples_mean_norm(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to apply min-max normalization to the audio samples."""
        
        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_mean), self.global_standard_deviation)
        
        return audio_images, audio_samples_norm, video_images, action, location, subject
    
    def _map_func_video_images(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to pre-processes the given image for training."""
        
        # function to rescale image, normalizing subtracting the mean and takes random crops in different positions
        # and to flip right left images for is_training=True for augmenting data
        # so we leave is_training=false only to take central crop
        def prepare_image(image):
            # return vgg_preprocessing.preprocess_image(image, _IMAGE_SIZE, _IMAGE_SIZE, is_training=False)
            # return self._aspect_preserving_resize(image, _IMAGE_SIZE)
            return self._normalize_images_rescaled(image)
        
        processed_images = tf.map_fn(prepare_image, video_images, dtype=tf.float32, back_prop=False)
        
        return audio_images, audio_samples, processed_images, action, location, subject
    
    def _normalize_images_rescaled(self, image):
        image.set_shape([224, 298, 3])
        image = tf.to_float(image)
        image = self._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        return image
    
    def _aspect_preserving_resize(self, image, smallest_side):
        """Resize images preserving the original aspect ratio.
        Args:
          image: A 3-D image `Tensor`.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.
        Returns:
          resized_image: A 3-D tensor containing the resized image.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
        
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        
        return resized_image
    
    def _smallest_size_at_least(self, height, width, smallest_side):
        """Computes new shape with the smallest side equal to `smallest_side`.
        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.
        Args:
          height: an int32 scalar tensor indicating the current height.
          width: an int32 scalar tensor indicating the current width.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.
        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: and int32 scalar tensor indicating the new width.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
        
        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)
        
        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / height,
                        lambda: smallest_side / width
                        )
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)
        
        return new_height, new_width
    
    def _mean_image_subtraction(self, image, means):
        """Subtracts the given means from each image channel.
        For example:
          means = [123.68, 116.779, 103.939]
          image = _mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.
        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.
        Returns:
          the centered image.
        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        
        channels = tf.split(image, num_channels, axis=2)
        for i in range(num_channels):
            channels[i] -= means[i]
        
        return tf.concat(channels, axis=2)
    
    def _map_func_audio_samples_build_spectrogram(self, audio_images, audio_wav, processed_images, action, location,
                                                  subject):
        """Input mapper to build spectrogram from waveform audio."""
        
        audio_stfts = tf.contrib.signal.stft(audio_wav, frame_length=self.frame_length,
                                             frame_step=self.frame_step, fft_length=self.fft_length)
        
        magnitude_spectrograms = tf.expand_dims(tf.abs(audio_stfts), 1)
        
        return audio_images, magnitude_spectrograms, processed_images, action, location, subject
    
    @property
    def total_batches(self):
        total_batches = int(math.ceil(self.num_samples / self.batch_size))
        return total_batches
