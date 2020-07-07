import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import scipy.io.wavfile
import subprocess

_NUMBER_OF_MICS = 128
_NUMBER_OF_SAMPLES = 1024
_FPS = 13

def generate_audio(folder_name):
    data_dir = folder_name
    out_dir = folder_name
    mic_id = 0

    audio_dir = '{}/audio'.format(data_dir)
    video_dir = '{}/beam_matlab'.format(data_dir)

    num_files = len([name for name in os.listdir(audio_dir) if name.endswith('.dc')])

    audio_data = np.zeros((num_files, _NUMBER_OF_MICS, _NUMBER_OF_SAMPLES), dtype=np.float32)

    print('Reading audio data from directory {} and microphone {}'.format(audio_dir, mic_id))

    for h in range(0, num_files):
        # Compose audio file name
        audio_sample_file = '{}/A_{:06d}.dc'.format(audio_dir, h + 1)

        # Read audo file
        with open(audio_sample_file) as fid:
            audio_data_mic = np.fromfile(fid, np.int32).reshape((_NUMBER_OF_MICS, _NUMBER_OF_SAMPLES), order='F')
            audio_data[h, :, :] = audio_data_mic

    print('Extracting microphone data')

    audio_data_mic = audio_data[:, mic_id, :]
    audio_data_mic_flat = audio_data_mic.flatten('C')
    audio_data_mic_norm = audio_data_mic_flat / abs(max(audio_data_mic_flat.min(), audio_data_mic_flat.max(), key=abs))

    print('Creating audio track')

    audio_file = '{}/audio_track.wav'.format(out_dir)
    scipy.io.wavfile.write('{}'.format(audio_file), _FPS * 1000, audio_data_mic_norm)

    # print('Creating video track')
    #
    # video_file = '{}/video_track.avi'.format(out_dir)
    # command = 'ffmpeg -y -r {} -f image2 -s 640x480 -i {}/I_%06d.bmp -vcodec libx264 -crf 25 -pix_fmt yuv420p {}'.format(
    #     _FPS, video_dir.replace(' ', '\ '), video_file.replace(' ', '\ '))
    # exit_code = subprocess.call(command, shell=True)
    #
    # if exit_code:
    #     print('Failed')
    #     exit(1)
    # else:
    #     print('Done')
    #
    # print('Merging audio and video tracks')
    #
    # command = 'ffmpeg -y -i {} -i {} -codec copy -shortest {}/video.avi'.format(audio_file.replace(' ', '\ '),
    #                                                                             video_file.replace(' ', '\ '),
    #                                                                             out_dir.replace(' ', '\ '))
    # exit_code = subprocess.call(command, shell=True)
    #
    # if exit_code:
    #     print('Failed')
    #     exit(1)
    # else:
    #     print('Done')
    #
    # print('Cleaning temporary files')
    #
    # try:
    #     os.remove(audio_file)
    #     os.remove(video_file)
    # except OSError as e:
    #     print('An unexpected error occurred while remove temporary audio and video track files. {}', e)
    #
    # print('Done')
    return audio_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=str)
    parsed_args = parser.parse_args()
    
    audio = parsed_args.audio
    audio = generate_audio(audio)
    directory = str.join('/', audio.split('/')[:-1])
    """
    Compute spectrogram for audio
    """
    x, fs = librosa.load(audio, sr=None)
    hop_length=(np.round(0.02*fs)).astype('int')
    win_length=(np.round(0.04*fs)).astype('int')
    #(np.floor((sig_len - win_len) / win_shift)).astype(int) + 1
    n_fft = (2 ** (np.floor(np.log2(win_length)) + 1)).astype(int)
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(x[0:2 * fs], n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True))[1:,:],
        ref=np.max)
    res=np.transpose(D)
    plt.imshow(res)
    plt.axis('off')
    plt.set_cmap(cmap='viridis')
    plt.savefig(directory+'/spectrogram.png')
    plt.clf()
    
if __name__ == '__main__':
    main()