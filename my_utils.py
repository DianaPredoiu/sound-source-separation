#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa, librosa.display
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt


# In[ ]:


def mix_audios(filename1, filename2):
    sound1 = AudioSegment.from_file(filename1)
    sound2 = AudioSegment.from_file(filename2)

    # overlay over the longest audio source
    if np.array(sound1.get_array_of_samples()).shape[0] > np.array(sound2.get_array_of_samples()).shape[0]:
        combined = sound1.overlay(sound2)
    else:
        combined = sound2.overlay(sound1)

    sound1_start, sound1_end = filename1.index('arctic'), filename1.index('.wav')
    sound2_start, sound2_end = filename2.index('arctic'), filename2.index('.wav')
    name = "../recordings/mixes/" + filename1[sound1_start:sound1_end] + '_' + filename2[sound2_start:sound2_end] + ".wav"
    combined.export(name, format='wav')
    
    return name


# In[ ]:


def get_specific_frame_in_ms(audio_array, start, stop):
    # in milliseconds
    newAudio = audio_array[start:stop]
    return newAudio
#     print("aici: ", len(audio_array), start, stop)
    
#     if stop <= len(audio_array):
#     print("1: ", len(audio_array), start, stop)
#     newAudio = audio_array[start:stop]
#     return newAudio
#     elif start >= len(audio_array):
#         print("2: ", len(audio_array), start, stop)
#         newAudio = np.zeros(stop-start)
#         return ndarray_to_audiosegment(newAudio, 16000)
#     elif start < len(audio_array) and stop > len(audio_array):
#         print("3: ", len(audio_array), start, stop)
#         oldAudio = audio_array[start:len(audio_array)-1]
#         newAudio = np.zeros(stop-start)
#         oldAudio.resize(newAudio.shape)
#         newAudio = newAudio + oldAudio
#         newAudio = ndarray_to_audiosegment(newAudio,16000)
#         return newAudio


# In[ ]:


def make_wav_files_same_size(arr1, arr2):
    if arr1.shape[0] < arr2.shape[0] :
        arr1 = np.pad(arr1, (0,(arr2.shape[0] - arr1.shape[0])), 'constant', constant_values=(0))
    else :
        arr2 = np.pad(arr2, (0,(arr1.shape[0] - arr2.shape[0])), 'constant', constant_values=(0))
    
    return arr1, arr2


# In[ ]:


def compute_mask(stft_1, stft_2):
#     print("aici: ", stft_1.shape, stft_2.shape)
    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute model as the sum of spectrograms
    mix = np.abs(stft_1) + np.abs(stft_2)
    
    mask = np.divide(np.abs(stft_1), mix)
    
    return mask


# In[ ]:


def get_stft_matrix_from_mixture(mask, mixture):
    return np.multiply(mask, mixture)


# In[ ]:


def write_new_audio_file(sound, filename):
    inverse_sound_stft = librosa.istft(sound)
    librosa.output.write_wav(filename, inverse_sound_stft, 16000)
    #s = sound.export(filename, format="wav")


# In[ ]:


def audiosegment_to_ndarray(audiosegment):
    samples = audiosegment.get_array_of_samples()
    samples_float = librosa.util.buf_to_float(samples,n_bytes=2,
                                      dtype=np.float32)
    if audiosegment.channels==2:
        sample_left= np.copy(samples_float[::2])
        sample_right= np.copy(samples_float[1::2])
        sample_all = np.array([sample_left,sample_right])
    else:
        sample_all = samples_float
        
        
    return [sample_all,audiosegment.frame_rate]


# In[ ]:


def ndarray_to_audiosegment(y,frame_rate):
    
    if(len(y.shape) == 2):
        new_array = np.zeros((y.shape[1]*2),dtype=float)
        new_array[::2] = y[0]
        new_array[1::2] = y[1]
    else:
        new_array = y
        
    audio_segment = AudioSegment(
    new_array.tobytes(), 
    frame_rate=frame_rate,
    sample_width=new_array.dtype.itemsize, 
    channels = len(y.shape)
)
    return audio_segment


# In[ ]:


def load_and_mix_files(female_filename, male_filename):
    # get 2 audio files
    male, sr_male = librosa.load(male_filename, sr=16000) 
    female, sr_female = librosa.load(female_filename, sr=16000) 

    # pad smaller array with zeros, so both audio files have the same length
    female, male = make_wav_files_same_size(female, male)

    # load the mixed audio 
    mix_filename= mix_audios(male_filename, female_filename)
    mix, sr_mix = librosa.load(mix_filename, sr=16000)

    # durata totala a inregistrarii
    male_rec_ms = float(len(male)) / sr_male * 1000
    female_rec_ms = float(len(female)) / sr_female * 1000
    mixed_audio_rec_ms = float(len(mix)) / 16000 * 1000
    print(male_rec_ms, female_rec_ms, mixed_audio_rec_ms)
    
    return female, male, mix


# In[ ]:


def delete_final_zeros_for_silence(sound):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(sound, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if ranges.size == 0:
        return sound
       
    start = ranges[len(ranges)-1][0]
    stop = ranges[len(ranges)-1][1]
    if stop == sound.shape[0]:
        sound = sound[:start]
    return sound


# In[ ]:


def show_plot(y_stft, title, pos):
    plt.figure(figsize=(20, 20))
    D = librosa.amplitude_to_db(np.abs(y_stft), ref=np.max)
    plt.subplot(4, 2, pos)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    
# In[ ]:


def show_waveplot(y_stft, title, pos):
    plt.figure()
    plt.subplot(3, 1, pos)
    librosa.display.waveplot(y_stft, sr=16000)
    plt.title(title)