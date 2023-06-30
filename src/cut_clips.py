from pydub import AudioSegment
import math
import os

num_files_for_each_language = 1000
train_rate = 0.8
data_root_path = '../data/'
lang = 'serbian'

# cut MP3 audio recording into multiple 10-second MP3 clips
def split_audio(input_file, tag):
    audio = AudioSegment.from_file(input_file)

    # convert to wav format in case input MP3 file might be corrupted or have issues with its header
    temp_wav_file = "temp.wav"
    audio.export(temp_wav_file, format="wav")
    audio = AudioSegment.from_wav(temp_wav_file)

    duration_ms = len(audio)
    num_clips = math.ceil(duration_ms / 10000)  # 10 seconds
    # os.makedirs(output_folder, exist_ok=True)

    category = 'train' 
    
    for i in range(num_clips-1):
        start_time = i * 10000
        end_time = (i + 1) * 10000

        output_folder =  data_root_path + category + '/' + lang
        clip = audio[start_time:end_time]
        clip.export(f"{output_folder}/clip_{tag}_{i+1}.mp3", format="mp3")

        # Stop after collecting enough files
        if i == int(num_files_for_each_language * train_rate):
            category = 'test'
        if i == num_files_for_each_language:
            break

    os.remove(temp_wav_file)
    print(f"Audio file '{input_file}' successfully split into {num_clips} clips.")


input_file = "../data/audio_books/gazdamladen.mp3"
tag = "gazdamladen"
split_audio(input_file, tag)