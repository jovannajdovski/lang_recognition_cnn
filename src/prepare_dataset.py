import librosa as lr
import os
from random import shuffle
from shutil import copy2
import numpy as np
import pandas as pd


languages = ['english', 'spanish', 'serbian']
categories = ['train', 'test']

original_dataset_paths = {}

original_dataset_paths['english'] = '../data/en/' 
original_dataset_paths['spanish'] = '../data/es/'
original_dataset_paths['serbian'] = '../data/sr/'

data_root_path = '../data/'

num_files_for_each_language = 20000
train_rate = 0.8

for lang in languages:
    if not os.path.isdir(original_dataset_paths[lang]):
        raise
    for category in categories:
        if not os.path.isdir(data_root_path + category + '/' + lang):
            raise

for lang in languages:
    if not os.path.isfile(original_dataset_paths[lang] + 'validated.tsv'):
        raise
    if not os.path.isdir(original_dataset_paths[lang] + 'clips'):
        raise


def copy_audio_files_for_language(lang):
    
    print('')
    print('Copying files for language ' + lang + '...')
    print('')
    
    # Only take validated speech data
    df = pd.read_csv(original_dataset_paths[lang] + 'validated.tsv', sep='\t')
    all_filenames = df['path'].tolist()
    shuffle(all_filenames)
    
    counter = 0
    
    category = 'train'    
    
    # Process files
    for filename in all_filenames:
        file = original_dataset_paths[lang] + 'clips/' + filename
        try:
            audio_segment, sample_rate = lr.load(file)
            if np.count_nonzero(audio_segment) == 0:
                raise Exception('Audio is silent!')
            if audio_segment.ndim != 1:
                raise Exception('Audio signal has wrong number of dimensions: ' + str(audio_segment.ndim))
            duration_sec = lr.core.get_duration(y=audio_segment, sr=sample_rate)
        except Exception as e:
            print('WARNING! Error while loading file \"' + file + '\": ' + str(e) + ' - Skipping...')
            continue
        
        # Only copy audio files with a certain minimum duration
        if 7.5 < duration_sec < 10.0:
            copy2(file, data_root_path + category + '/' + lang)
            counter += 1
        
        # Stop after collecting enough files
        if counter == int(num_files_for_each_language * train_rate):
            category = 'test'
        if counter == num_files_for_each_language:
            break

if __name__ == '__main__':
    for lang in languages:
        copy_audio_files_for_language(lang)
    
