import os
import librosa as lr
from librosa.display import waveplot
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import imageio
from IPython.display import Audio


languages = ['english', 'spanish', 'serbian']
categories = ['train', 'test']

data_root_path = '../data/'

sample_rate = 8000
image_width = 500
image_height = 128


def load_audio_file(audio_file_path):
    audio_segment, _ = lr.load(audio_file_path, sr=sample_rate)
    return audio_segment


def fix_audio_segment_to_10_seconds(audio_segment):
    target_len = 10 * sample_rate
    audio_segment = np.concatenate([audio_segment]*3, axis=0)
    audio_segment = audio_segment[0:target_len]
    
    return audio_segment


def spectrogram(audio_segment):
    # Compute mel-scaled spectrogram image
    hl = audio_segment.shape[0] // image_width
    spec = lr.feature.melspectrogram(audio_segment, n_mels=image_height, hop_length=int(hl))

    # Logarithmic amplitudes
    # amplitudes => db
    image = lr.core.power_to_db(spec)

    # Convert to numpy matrix
    image_np = np.asmatrix(image)

    # Normalize and scale
    image_np_scaled_temp = (image_np - np.min(image_np))
    
    image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
    return image_np_scaled[:, 0:image_width]


def to_integer(image_float):
    image_float_255 = image_float * 255.
    image_int = image_float_255.astype(np.uint8)
    return image_int


def present_example_on_plt(audio_file):
    audio = load_audio_file(audio_file)
    waveplot(audio, sr=sample_rate)
    plt.show()

    audio_fixed = fix_audio_segment_to_10_seconds(audio)
    waveplot(audio_fixed, sr=sample_rate)
    plt.show()

    spectro = spectrogram(audio_fixed)
    plt.imshow(spectro, origin='lower', aspect='auto')
    plt.show()
    print("Spectogram dim: "+spectro.shape)


def present_example_as_img():
    list_of_image_files = glob(data_root_path + 'test' + '/' + 'english' + '/*.png')
    image_file_path = list_of_image_files[0]

    print("Image file" + image_file_path)

    image = imageio.imread(image_file_path)

    plt.imshow(image, origin='lower', aspect='auto')
    plt.show()

    print("Image dim:" + image.shape)

    # listen as an audio
    audio_file_path = os.path.splitext(image_file_path)[0]
    Audio(audio_file_path)


def audio_to_image_file(audio_file):
    out_image_file = audio_file + '.png'
    audio = load_audio_file(audio_file)
    audio_fixed = fix_audio_segment_to_10_seconds(audio)
    if np.count_nonzero(audio_fixed) != 0:
        spectro = spectrogram(audio_fixed)
        spectro_int = to_integer(spectro)
        imageio.imwrite(out_image_file, spectro_int)
    else:
        print('WARNING! Detected an empty audio signal. Skipping...')


if __name__ == '__main__':
    audio_files = {}

    for lang in languages:
        for category in categories:
            dataset_path = data_root_path + category + '/' + lang
            
            # preprocess all data
            audio_files[lang + '.' + category] = glob(dataset_path + '/*.mp3') + glob(dataset_path + '/*.wav')
            
            # :TODO preprocess augmented data 
            #audio_files[lang + '.' + category] = glob(dataset_path + '/*.wav')

    print(audio_files.keys())

    key = list(audio_files.keys())[0]
    audio_file = audio_files[key][0]
    present_example_on_plt(audio_file)


    for lang in languages:
        for category in categories:
            all_audio_files = audio_files[lang + '.' + category]
            
            num_files = len(all_audio_files)
            
            for i in range(num_files):
                if i % (num_files / 50) == 0:
                    print('Still processing ' + lang + ' ' + category + ' ' + str(i) + '/' + str(num_files))
                audio_to_image_file(all_audio_files[i])
    
    present_example_as_img()


