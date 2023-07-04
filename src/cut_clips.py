from pydub import AudioSegment
import math
import os
import soundfile as sf
import librosa

num_files_for_each_language = 10000
train_rate = 0.8
data_root_path = '../data/'
lang = 'serbian'

# cut MP3 audio recording into multiple 10-second MP3 clips
def split_audio(input_file, tag, counter):
    audio = AudioSegment.from_file(input_file)

    # convert to wav format in case input MP3 file might be corrupted or have issues with its header
    temp_wav_file = "temp.wav"
    audio.export(temp_wav_file, format="wav")
    audio = AudioSegment.from_wav(temp_wav_file)

    duration_ms = len(audio)
    num_clips = math.ceil(duration_ms / 10000)  # 10 seconds
    # os.makedirs(output_folder, exist_ok=True)

    category = 'train' 
    
    for i in range(counter,num_clips-1+counter):
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

    counter += num_clips-1

    os.remove(temp_wav_file)
    print(f"Audio file '{input_file}' successfully split into {num_clips} clips.")
    return counter

def split_audio_from_large_file(input_file, tag, counter):
    duration = librosa.get_duration(path=input_file)
    num_clips = math.ceil(duration / 10)

    category = 'train'

    for i in range(counter, num_clips-1+counter):
        start_time = i * 10
        end_time = (i + 1) * 10

        output_folder =  data_root_path + category + '/' + lang
        
        # Load audio in chunks
        audio, sample_rate = librosa.load(input_file, offset=start_time, duration=10)

        output_file = f"{output_folder}/clip_{tag}_{i+1}.wav"
        sf.write(output_file, audio, sample_rate, format='wav')

        # Convert WAV to MP3 using pydub
        audio_segment = AudioSegment.from_wav(output_file)
        mp3_output_file = f"{output_folder}/clip_{tag}_{i+1}.mp3"
        audio_segment.export(mp3_output_file, format="mp3")

        os.remove(output_file)

        # Stop after collecting enough files
        if i == int(num_files_for_each_language * train_rate):
            category = 'test'
        if i == num_files_for_each_language:
            break

    counter += num_clips - 1

    print(f"Audio file '{input_file}' successfully split into {num_clips} clips.")
    return counter

if __name__ == '__main__':
    counter = 0

    input_files = ["../data/audio_books/necistakrv1.mp3","../data/audio_books/necistakrv2.mp3","../data/audio_books/bozjiljudi.mp3","../data/audio_books/kostana.mp3","../data/audio_books/odisejaSvemira2.mp3","../data/audio_books/odisejaSvemira1.mp3","../data/audio_books/cujtesrbi.mp3","../data/audio_books/tikve.mp3","../data/audio_books/gazdamladen.mp3","../data/audio_books/kafanskeprice.mp3"]
    tags = ["necistakrv1","necistakrv2","bozjiljudi","kostana","odisejaSvemira2","odisejaSvemira1","cujtesrbi","tikve","gazdamladen","kafanskeprice"]
    for i in range(len(input_files)):
        counter = split_audio(input_files[i], tags[i], counter)
    
    # input_file = "../data/audio_books/odiseja_svemira.mp3"
    # tag = "odiseja"
    # counter = split_audio_from_large_file(input_file, tag, counter)