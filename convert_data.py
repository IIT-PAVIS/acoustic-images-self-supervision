import argparse
import cv2
import glob
import math
import numpy as np
import os
import re
import librosa
import tensorflow as tf
from collections import namedtuple
from datetime import datetime
from scipy import io as spio
from traceback import print_exc

Image = namedtuple('Image', 'rows cols depth data')
Audio = namedtuple('Audio', 'mics samples data')

# _BROKEN_MICS_IDX = [79, 105]
_NUMBER_OF_MICS = 128
_NUMBER_OF_SAMPLES = 1024
_FRAMES_PER_SECOND = 12


# _MIN_LENGTH = 30


def _read_acoustic_image(filename):
    print('{} - Reading {}'.format(datetime.now(), filename))

    # img_raw = spio.loadmat(filename)['Beam'].astype('<f4')
    img_padded = spio.loadmat(filename)['MFCC']
    # pad_width = [(before, 0) for before in np.array([36, 48, 13]) - np.array(img_raw.shape)]
    # img_padded = np.pad(img_raw, pad_width=pad_width, mode='constant', constant_values=0)

    rows = img_padded.shape[0]
    cols = img_padded.shape[1]
    depth = img_padded.shape[2]
    image_serialized = img_padded.tostring()

    return Image(rows=rows, cols=cols, depth=depth, data=image_serialized)


def one_microphone(audio_data):
    """Python function to build a waveform audio from audio samples."""
    # choose index
    mic_id = 0
    # consider audio of one microphone
    audio_data_mic = audio_data[mic_id, :]
    return audio_data_mic


def _read_raw_audio_data(audio_sample_file):
    print('{} - Reading {}'.format(datetime.now(), audio_sample_file))
    with open(audio_sample_file) as fid:
        audio_data_sample = np.fromfile(fid, np.int32).reshape((_NUMBER_OF_MICS, _NUMBER_OF_SAMPLES), order='F')
        # audio_data_sample[_BROKEN_MICS_IDX, :] = 0
        audio_data_sample = one_microphone(audio_data_sample)
    audio_serialized = audio_data_sample.tostring()

    return Audio(mics=1, samples=_NUMBER_OF_SAMPLES, data=audio_serialized)


def str2dir(dir_name):
    if not os.path.isdir(dir_name):
        raise argparse.ArgumentTypeError('{} is not a directory!'.format(dir_name))
    # elif os.access(dirname, os.R_OK):
    #     return argparse.ArgumentTypeError('{} is not a readable directory!'.format(dirname))
    else:
        return os.path.abspath(os.path.expanduser(dir_name))


def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.
    Args:
      image: A 3-D image.
      smallest_side: A python integer or scalar indicating the size of
        the smallest side after resize.
    Returns:
      resized_image: A 3-D resized image.
    """
    shape = np.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar indicating the current height.
      width: an int32 scalar indicating the current width.
      smallest_side: A python integer or scalar indicating the size of
        the smallest side after resize.
    Returns:
      new_height: an int32 scalar indicating the new height.
      new_width: and int32 scalar indicating the new width.
    """
    height = float(height)
    width = float(width)
    smallest_side = float(smallest_side)

    if height > width:
        scale = smallest_side / width
    else:
        scale = smallest_side / height
    new_height = int(height * scale)
    new_width = int(width * scale)
    return new_height, new_width


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar indicating the height offset.
      offset_width: a scalar indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.
    Returns:
      the cropped (and resized) image.
    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = np.shape(image)

    assert np.rank(image) == 3
    # 'Rank of image must be equal to 3.']
    cropped_shape = np.stack([crop_height, crop_width, original_shape[2]])

    assert original_shape[0] >= crop_height and original_shape[1] >= crop_width
    # 'Crop size greater than the image size.']
    offsets = np.stack([offset_height, offset_width, 0])

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = image[offset_height:crop_height + offset_height, offset_width:crop_width + offset_width, :]
    return image


def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image.
    Args:
      image: image.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      cropped image.
    """
    outputs = []
    image_height = np.shape(image)[0]
    image_width = np.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    image = _crop(image, offset_height, offset_width,
                  crop_height, crop_width)
    return image


def _read_video_frame(filename):
    print('{} - Reading {}'.format(datetime.now(), filename))

    image_raw = cv2.imread(filename)

    # rows = image_raw.shape[0]
    # cols = image_raw.shape[1]
    # depth = image_raw.shape[2]
    # image rescaled to give in input image aligned with acoustic image
    image = _aspect_preserving_resize(image_raw, 224)
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    # don't take crop
    # image = _central_crop(image, _IMAGE_SIZE, _IMAGE_SIZE)
    image_serialized = image.tostring()
    return Image(rows=rows, cols=cols, depth=depth, data=image_serialized)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='Matlab files data set root directory', type=str2dir)
    parser.add_argument('root_raw_dir', help='Synchronized raw files data set root directory', type=str2dir)
    parser.add_argument('out_dir', help='Directory where to store the converted data', type=str2dir)
    parser.add_argument('--modalities', help='Modalities to consider. 0: Audio images. 1: Audio data. 2: Video data.',
                        nargs='*', type=int)
    parsed_args = parser.parse_args()

    root_dir = parsed_args.root_dir
    root_raw_dir = parsed_args.root_raw_dir
    out_dir = parsed_args.out_dir
    modalities = parsed_args.modalities
    include_audio_images = modalities is None or 0 in modalities
    include_audio_data = modalities is None or 1 in modalities
    include_video_data = modalities is None or 2 in modalities

    # data_dirs = sorted(glob.glob('{}/*/*/*/Multispectral_Acoustic_Image/'.format(root_dir)))
    data_dirs = sorted(glob.glob('{}/*/*/MFCC_Image/'.format(root_dir)))

    for data_mat_dir in data_dirs:
        # / media / vsanguineti / D90A - E56D / dualcam_actions_dataset / sync / Location_1 / 02
        # _Nuno / data_001
        # / media / vsanguineti / D90A - E56D / dualcam_actions_dataset / sync / classes/ data_001
        splitted_data_dir = data_mat_dir.split('/')
        data_dir_file = str.join('/', splitted_data_dir[:-2])
        video_time_filename = 'video_time.txt'
        try:
            fd = open(data_dir_file + '/' + video_time_filename, 'r')
            info_time = fd.readline().split(':')[1]
            fd.close()
            video_time = int(info_time.strip())

        except:
            print('error during reading video_time.txt for ' + c)
            print_exc()

        classes = int(next(filter(re.compile('class_.*').match, splitted_data_dir)).split('_')[1])
        # subject = filter(re.compile('0[1-9]_.*').match, splitted_data_dir)[0]
        # subject_idx = int(subject.split('_')[0])
        location = int(next(filter(re.compile('data_.*').match, splitted_data_dir)).split('_')[1])

        # data_raw_audio_dir = data_mat_dir.replace(root_dir, root_raw_dir).replace('Multispectral_Acoustic_Image', 'audio')
        # data_raw_video_dir = data_mat_dir.replace(root_dir, root_raw_dir).replace('Multispectral_Acoustic_Image', 'video')
        data_raw_audio_dir = data_mat_dir.replace(root_dir, root_raw_dir).replace('MFCC_Image', 'audio')
        data_raw_video_dir = data_mat_dir.replace(root_dir, root_raw_dir).replace('MFCC_Image', 'video')

        num_mat_files = len([name for name in os.listdir(data_mat_dir) if name.endswith('.mat')])
        num_raw_audio_files = len([name for name in os.listdir(data_raw_audio_dir) if name.endswith('.dc')])
        num_raw_video_files = len([name for name in os.listdir(data_raw_video_dir) if name.endswith('.bmp')])

        # Ensure there are the same number of acoustic images and raw audio and video files
        assert num_mat_files == num_raw_audio_files
        assert num_mat_files == num_raw_video_files
        # compute number of videos (number ot total video files)

        frames_per_video = _FRAMES_PER_SECOND * video_time

        num_samples = _FRAMES_PER_SECOND

        # changed
        for idx in range(video_time):

            start_index = idx * num_samples

            # mat_files is a tuple containing audio_images' name for the num_frames video
            if include_audio_images:
                mat_files = ['{}/Data_{}.mat'.format(data_mat_dir, index + 1) for index in
                             range(start_index, start_index + num_samples)]
                audio_images = [_read_acoustic_image(filename) for filename in mat_files]
            else:
                audio_images = None

            if include_audio_data:

                raw_audio_files = ['{}/A_{:06d}.dc'.format(data_raw_audio_dir, index + 1) for
                                   index in range(start_index, start_index + num_samples)]
                audio_data = [_read_raw_audio_data(filename) for filename in raw_audio_files]
            else:
                audio_data = None

            if include_video_data:
                raw_video_files = ['{}/I_{:06d}.bmp'.format(data_raw_video_dir, index + 1) for
                                   index in range(start_index, start_index + num_samples)]
                video_images = [_read_video_frame(filename) for filename in raw_video_files]
            else:
                video_images = None

            out_data_dir = '{}/class_{}/data_{:0>3d}/'.format(out_dir, classes, location)
            out_filename = '{}/Data_{:0>3d}.tfrecord'.format(out_data_dir, idx + 1)

            if not os.path.exists(out_data_dir):
                os.makedirs(out_data_dir)

            print('{} - Writing {}'.format(datetime.now(), out_filename))

            with tf.python_io.TFRecordWriter(out_filename, options=tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP)) as writer:
                # Store audio and video data properties as context features, assuming all sequences are the same size
                feature = {
                    'classes': _int64_feature(classes),
                    'location': _int64_feature(location)
                }
                if include_audio_images:
                    feature.update({
                        'audio_image/height': _int64_feature(audio_images[0].rows),
                        'audio_image/width': _int64_feature(audio_images[0].cols),
                        'audio_image/depth': _int64_feature(audio_images[0].depth)
                    })
                if include_audio_data:
                    feature.update({
                        'audio_data/mics': _int64_feature(audio_data[0].mics),
                        'audio_data/samples': _int64_feature(audio_data[0].samples)
                    })
                if include_video_data:
                    feature.update({
                        'video/height': _int64_feature(video_images[0].rows),
                        'video/width': _int64_feature(video_images[0].cols),
                        'video/depth': _int64_feature(video_images[0].depth),
                    })
                feature_list = {}
                if include_audio_images:
                    feature_list.update({
                        'audio/image': tf.train.FeatureList(
                            feature=[_bytes_feature(audio_image.data) for audio_image in audio_images])
                    })
                if include_audio_data:
                    feature_list.update({
                        'audio/data': tf.train.FeatureList(
                            feature=[_bytes_feature(audio_sample.data) for audio_sample in audio_data])
                    })
                if include_video_data:
                    feature_list.update({
                        'video/image': tf.train.FeatureList(
                            feature=[_bytes_feature(video_image.data) for video_image in video_images])
                    })
                context = tf.train.Features(feature=feature)
                feature_lists = tf.train.FeatureLists(feature_list=feature_list)
                sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                writer.write(sequence_example.SerializeToString())
