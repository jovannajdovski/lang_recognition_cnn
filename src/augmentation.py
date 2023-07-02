import os
import librosa as lr
import numpy as np
from glob import glob
import soundfile as sf

languages = ['english', 'spanish', 'serbian']
categories = ['train', 'test']

data_root_path = '../data/'

# how much of the training data should be augmented
augment_data_factor = 1.0 


def load_audio_file(audio_file_path):  
    audio_segment, sample_rate = lr.load(audio_file_path)
    return audio_segment, sample_rate


def add_noise(audio_segment, gain):
    num_samples = audio_segment.shape[0]
    noise = gain * np.random.normal(size=num_samples)
    return audio_segment + noise


def augment_audio_file_with_noise(audio_file_path):
    audio_segment, sample_rate = load_audio_file(audio_file_path)
    audio_segment_with_noise = add_noise(audio_segment, 0.005)
    audio_file_path_without_extension = os.path.splitext(audio_file_path)[0]
    augmented_audio_file_path = audio_file_path_without_extension + '_augmented_noise.wav'
    sf.write(augmented_audio_file_path, audio_segment_with_noise, sample_rate)


if __name__ == '__main__':
    audio_files = {}

    for lang in languages:
        for category in categories:
            audio_files[lang + '.' + category] = glob(data_root_path + category + '/' + lang + '/*.mp3')

    for lang in languages:
        category = 'train'
            
        all_audio_files = audio_files[lang + '.' + category]
            
        num_files = int(len(all_audio_files) * augment_data_factor)

        for i in range(num_files):
            if i % (num_files / 10) == 0:
                print('Still processing ' + lang + ' ' + category + ' ' + str(i) + '/' + str(num_files))
            augment_audio_file_with_noise(all_audio_files[i])
    